""" Uses approximations for both signed square rooting and l2 normalization."""
import argparse
from collections import defaultdict
import cPickle
import functools
from multiprocessing import Pool
from itertools import izip
import numpy as np
import pdb
import os
from scipy import sparse
import tempfile

# from ipdb import set_trace
from joblib import Memory
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from yael import threads

from dataset import Dataset
from fisher_vectors.evaluation import Evaluation
from fisher_vectors.evaluation.utils import average_precision
from fisher_vectors.model.utils import L2_normalize as exact_l2_normalize
from fisher_vectors.model.utils import power_normalize

from load_data import approximate_signed_sqrt
from load_data import load_kernels
from load_data import load_sample_data


# TODO Possible improvements:
# [ ] Evaluate with the exact normalizations.
# [ ] Pre-allocate test data (counts, L2 norms and scores).
# [x] Use sparse matrices for masks.
# [x] Use also empirical standardization.
# [x] Load dummy data.
# [x] Parallelize per-class evaluation.

    
hmdb_stab_dict = {
    'hmdb_split%d.stab' % ii :{
        'dataset_name': 'hmdb_split%d' % ii,
        'dataset_params': {
            'ip_type': 'dense5.track15mbh',
            'nr_clusters': 256,
            'suffix': '.per_slice.delta_15.stab.fold_%d' % ii,
        },
        'eval_name': 'hmdb',
        'eval_params': {
        },
        'metric': 'accuracy',
    } for ii in xrange(1, 4)}


cache_dir = os.path.expanduser('~/scratch2/tmp')
CFG = {
    'trecvid11_devt': {
        'dataset_name': 'trecvid12',
        'dataset_params': {
            'ip_type': 'dense5.track15mbh',
            'nr_clusters': 256,
            'suffix': '.per_slice.small.delta_60.skip_1',
        },
        'eval_name': 'trecvid12',
        'eval_params': {
            'split': 'devt',
        },
        'metric': 'average_precision',
    },
    'hollywood2':{
        'dataset_name': 'hollywood2',
        'dataset_params': {
            'ip_type': 'dense5.track15mbh',
            'nr_clusters': 256,
            'suffix': '.per_slice.delta_60',
        },
        'eval_name': 'hollywood2',
        'eval_params': {
        },
        'metric': 'average_precision',
    },
    'hollywood2.delta_30':{
        'dataset_name': 'hollywood2',
        'dataset_params': {
            'ip_type': 'dense5.track15mbh',
            'nr_clusters': 256,
            'suffix': '.per_slice.delta_30',
        },
        'eval_name': 'hollywood2',
        'eval_params': {
        },
        'metric': 'average_precision',
    },
    'dummy': {
        'dataset_name': '',
        'dataset_params': {
        },
        'eval_name': 'hollywood2',
        'eval_params': {
        },
        'metric': 'accuracy',
    },
    'hmdb_split1':{
        'dataset_name': 'hmdb_split1',
        'dataset_params': {
            'ip_type': 'dense5.track15mbh',
            'nr_clusters': 256,
            'suffix': '.per_slice.delta_30',
        },
        'eval_name': 'hmdb',
        'eval_params': {
        },
        'metric': 'accuracy',
    },
}
CFG.update(hmdb_stab_dict)


LOAD_SAMPLE_DATA_PARAMS = {
    'analytical_fim' : True,
    'pi_derivatives' : False,
    'sqrt_nr_descs'  : False,
    'return_info'    : True,
}


def my_cacher(*args):

    def loader(file, format):
        if format in ('cp', 'cPickle'):
            result = cPickle.load(file)
        elif format in ('np', 'numpy'):
            result = np.load(file)
        else:
            assert False
        return result

    def dumper(file, result, format):
        if format in ('cp', 'cPickle'):
            cPickle.dump(result, file)
        elif format in ('np', 'numpy'):
            np.save(file, result)
        else:
            assert False

    store_format = args

    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            outfile = kwargs.get('outfile', tempfile.mkstemp()[1])
            if os.path.exists(outfile):
                with open(outfile, 'r') as ff:
                    return [loader(ff, sf) for sf in store_format]
            else:
                result = func(*args, **kwargs)
                with open(outfile, 'w') as ff:
                    for rr, sf in izip(result, store_format):
                        dumper(ff, rr, sf)
                return result
        return wrapped

    return decorator


@my_cacher('np', 'cp', 'np', 'np', 'cp', 'cp')
def load_dummy_data(seed, store_format=None, outfile=None):
    N_SAMPLES = 100
    N_CENTERS = 5
    N_FEATURES = 20
    K, D = 2, 5

    te_data, te_labels = make_blobs(
        n_samples=N_SAMPLES, centers=N_CENTERS,
        n_features=N_FEATURES, random_state=seed)

    te_video_mask = sparse.csr_matrix(np.eye(N_SAMPLES))
    te_visual_word_mask = build_visual_word_mask(D, K)

    np.random.seed(seed)
    te_counts = np.random.rand(N_SAMPLES, K)
    te_l2_norms = te_data ** 2 * te_visual_word_mask
    te_labels = list(te_labels)

    return (
        te_data, te_labels, te_counts, te_l2_norms, te_video_mask,
        te_visual_word_mask)


def compute_weights(clf, xx, tr_std=None):
    weights = np.dot(clf.dual_coef_, xx[clf.support_])
    bias = clf.intercept_
    if tr_std is not None:
        weights /= tr_std
    return weights, bias


def predict(yy, weights, bias):
    return (- np.dot(yy, weights.T) + bias).squeeze()


def build_aggregation_mask(names):
    """ Mask to aggregate slice data into video data. """
    ii, idxs, seen = -1, [], []
    for name in names:
        if name not in seen:
            ii += 1
            seen.append(name)
        idxs.append(ii)
        
    nn = len(seen)
    N = len(names)
    mask = np.zeros((N, nn))
    mask[range(N), idxs] = 1

    return sparse.csr_matrix(mask.T)


def build_visual_word_mask(D, K):
    """ Mask to aggregate a Fisher vector into per visual word values. """
    I = np.eye(K)
    mask = np.hstack((I.repeat(D, axis=1), I.repeat(D, axis=1))).T
    return sparse.csr_matrix(mask)


def build_slice_agg_mask(N, n_group):
    # Build mask.
    yidxs = range(N)
    xidxs = map(int, np.array(yidxs) / n_group)

    M = np.int(np.ceil(float(N) / n_group)) 
    mask = np.zeros((M, N))
    mask[xidxs, yidxs] = 1.

    return sparse.csr_matrix(mask)


def scale_by(fisher_vectors, nr_descriptors, video_mask=None):
    if video_mask is None:
        return (
            fisher_vectors * nr_descriptors[:, np.newaxis]
            / np.sum(nr_descriptors))
    else:
        return (
            fisher_vectors * nr_descriptors[:, np.newaxis]
            / ((video_mask * nr_descriptors) * video_mask)[:, np.newaxis])


def group_data(data, nr_to_group):
    """Sums together `nr_to_group` consecutive rows of `data`.

    Parameters
    ----------
    data: array_like, shape (N, D)
        Data matrix.

    nr_to_group: int
        Number of consecutive slices that are up.

    Returns
    -------
    grouped_data: array_like, shape (M, D)
        Grouped data.

    """
    N = data.shape[0]
    return build_slice_agg_mask(N, nr_to_group) * data


def visual_word_l2_norm(fisher_vectors, visual_word_mask):
    return fisher_vectors ** 2 * visual_word_mask  # NxK


def visual_word_scores(fisher_vectors, weights, bias, visual_word_mask):
    return (- fisher_vectors * weights) * visual_word_mask  # NxK


def approx_l2_normalize(data, l2_norms, counts):
    zero_idxs = counts == 0
    masked_norms = np.ma.masked_array(l2_norms, zero_idxs)
    masked_counts = np.ma.masked_array(counts, zero_idxs)
    masked_result = masked_norms / masked_counts
    approx_l2_norm = np.sum(masked_result.filled(0), axis=1)
    return data / np.sqrt(approx_l2_norm[:, np.newaxis])


def approximate_video_scores(
    slice_scores, slice_counts, slice_l2_norms, video_mask):

    video_scores = video_mask * slice_scores
    video_counts = video_mask * slice_counts
    video_l2_norms = video_mask * slice_l2_norms

    zero_idxs = video_counts == 0
    masked_scores = np.ma.masked_array(video_scores, zero_idxs)
    masked_counts = np.ma.masked_array(video_counts, zero_idxs)
    masked_l2_norms = np.ma.masked_array(video_l2_norms, zero_idxs)

    sqrt_scores = np.sum((masked_scores / np.sqrt(masked_counts)).filled(0), axis=1)
    approx_l2_norm = np.sum((masked_l2_norms / masked_counts).filled(0), axis=1)

    return sqrt_scores / np.sqrt(approx_l2_norm)


@my_cacher('np', 'cp', 'np', 'np', 'cp', 'cp')
def load_slices(
    dataset, samples, nr_slices_to_aggregate=1, outfile=None, verbose=0):

    counts = []
    fisher_vectors = []
    labels = []
    names = []
    nr_descs = []

    for jj, sample in enumerate(samples):

        fv, ii, cc, info = load_sample_data(
            dataset, sample, analytical_fim=True, pi_derivatives=False,
            sqrt_nr_descs=False, return_info=True)

        if sample.movie in names:
            continue

        nd = info['nr_descs']
        nd = nd[nd != 0]
        label = ii['label']

        slice_agg_mask = build_slice_agg_mask(fv.shape[0], nr_slices_to_aggregate)
        agg_fisher_vectors = slice_agg_mask * scale_by(fv, nd, video_mask=slice_agg_mask)
        agg_counts = slice_agg_mask * scale_by(cc, nd, video_mask=slice_agg_mask)

        fisher_vectors.append(agg_fisher_vectors)
        counts.append(agg_counts)
        nr_descs.append(slice_agg_mask * nd)

        nr_slices = fisher_vectors[-1].shape[0]
        names += [sample.movie] * nr_slices
        labels += [label]

        if verbose:
            print '%5d %5d %s' % (jj, nr_slices, sample.movie)

    D, K = 64, dataset.VOC_SIZE
    fisher_vectors = np.vstack(fisher_vectors)
    counts = np.vstack(counts)
    nr_descs = np.hstack(nr_descs)

    video_mask = build_aggregation_mask(names)
    visual_word_mask = build_visual_word_mask(D, K)

    fisher_vectors = scale_by(fisher_vectors, nr_descs, video_mask)
    slice_vw_counts = scale_by(counts, nr_descs, video_mask)
    slice_vw_l2_norms = visual_word_l2_norm(fisher_vectors, visual_word_mask)

    return (
        fisher_vectors, labels, slice_vw_counts, slice_vw_l2_norms,
        video_mask, visual_word_mask)


@my_cacher('np', 'cp', 'np', 'np')
def load_train_video_data(
    dataset, nr_slices_to_aggregate=1, outfile=None, verbose=0):

    samples, _ = dataset.get_data('train')
    nr_samples = len(samples)

    D, K = 64, dataset.VOC_SIZE
    vw_mask = build_visual_word_mask(D, K)

    # Pre-allocate.
    data = np.zeros((nr_samples, 2 * D * K), dtype=np.float32)
    counts = np.zeros((nr_samples, K), dtype=np.float32)
    l2_norms = np.zeros((nr_samples, K), dtype=np.float32)
    labels = []

    ii = 0
    seen_samples = []

    for sample in samples:

        if sample.movie in seen_samples:
            continue
        seen_samples.append(sample.movie)

        fisher_vectors, _, slice_counts, info = load_sample_data(
            dataset, sample, **LOAD_SAMPLE_DATA_PARAMS)
        nr_descriptors = info['nr_descs']
        nr_descriptors = nr_descriptors[nr_descriptors != 0]

        slice_agg_mask = build_slice_agg_mask(fisher_vectors.shape[0], nr_slices_to_aggregate)
        agg_fisher_vectors = slice_agg_mask * scale_by(fisher_vectors, nr_descriptors)

        data[ii] = aggregate(scale_by(fisher_vectors, nr_descriptors))
        counts[ii] = aggregate(scale_by(slice_counts, nr_descriptors))
        l2_norms[ii] = aggregate(visual_word_l2_norm(agg_fisher_vectors, vw_mask))

        labels.append(info['label'])

        if verbose:
            print '%5d %5d %s' % (ii, agg_fisher_vectors.shape[0], sample.movie)

        ii += 1

    # The counter `ii` can be smaller than `nr_samples` when there are
    # duplicates in the `samples` as is the case of Hollywood2.
    return data[:ii], labels[:ii], counts[:ii], l2_norms[:ii]


@my_cacher('np', 'np', 'np', 'cp', 'cp')
def load_test_data(dataset, weight, outfile=None, verbose=0):
    samples, _ = dataset.get_data('test')

    D, K = 64, dataset.VOC_SIZE
    visual_word_mask = build_visual_word_mask(D, K)

    slice_vw_scores = []
    slice_vw_counts = []
    slice_vw_l2_norms = []
    labels = []
    idxs = []

    ii = 0
    seen_samples = []

    for sample in samples:

        if sample.movie in seen_samples:
            continue
        seen_samples.append(sample.movie)

        fisher_vectors, _, slice_counts, info = load_sample_data(
            dataset, sample, **LOAD_SAMPLE_DATA_PARAMS)
        nr_descriptors = info['nr_descs']
        nr_descriptors = nr_descriptors[nr_descriptors != 0]
        nn = len(nr_descriptors)

        fisher_vectors = scale_by(fisher_vectors, nr_descriptors)
        slice_counts = scale_by(slice_counts, nr_descriptors)

        slice_vw_scores.append(visual_word_scores(fisher_vectors, weight, 0, visual_word_mask))
        slice_vw_counts.append(slice_counts)
        slice_vw_l2_norms.append(visual_word_l2_norm(fisher_vectors, visual_word_mask))

        labels.append(info['label'])
        idxs += [ii] * nn

        if verbose > 2:
            print '%5d %5d %s' % (ii, nn, sample.movie)

        ii += 1

    slice_vw_scores = np.vstack(slice_vw_scores)
    slice_vw_counts = np.vstack(slice_vw_counts)
    slice_vw_l2_norms = np.vstack(slice_vw_l2_norms)

    return slice_vw_scores, slice_vw_counts, slice_vw_l2_norms, idxs, labels


def aggregate(data, mask=None, scale=None):
    """ Aggregates per-slice data into per-video data. """
    if mask is None and scale is None:
        return np.sum(data, axis=0)
    elif mask is None:
        return np.sum(
            scale[:, np.newaxis] * data /
            np.sum(scale), axis=0)
    elif scale is None:
        scale = np.ones(mask.shape[1])
    return ((mask * (scale[:, np.newaxis] * data)) /
            (mask * scale[:, np.newaxis]))


def compute_average_precision(true_labels, predictions, verbose=0):
    average_precisions = []
    for ii in sorted(true_labels.keys()):

        ap = average_precision(true_labels[ii], predictions[ii])
        average_precisions.append(ap)

        if verbose:
            print "%3d %6.2f" % (ii, 100 * ap)

    if verbose:
        print '----------'

    print "mAP %6.2f" % (100 * np.mean(average_precisions))


def compute_accuracy(label_binarizer, true_labels, predictions, verbose=0):
    all_predictions = np.vstack((predictions[ii] for ii in sorted(predictions.keys())))
    all_true_labels = np.vstack((true_labels[ii] for ii in sorted(true_labels.keys())))

    label_binarizer.multilabel = False
    array_true_labels = label_binarizer.inverse_transform(all_true_labels.T)

    predicted_class = np.argmax(all_predictions, axis=0)
    print "Accuracy %6.2f" % (100 * accuracy_score(array_true_labels, predicted_class))


def evaluate_worker((
    cls, eval, tr_data, tr_std, fisher_vectors, slice_vw_counts,
    slice_vw_l2_norms, video_mask, visual_word_mask, prediction_type, outfile,
    verbose)):

    clf = eval.get_classifier(cls)

    if prediction_type == 'approx':
        weight, bias = compute_weights(clf, tr_data, tr_std)
        slice_vw_scores = visual_word_scores(fisher_vectors, weight, bias, visual_word_mask)
        predictions = approximate_video_scores(slice_vw_scores, slice_vw_counts, slice_vw_l2_norms, video_mask)
    elif prediction_type == 'exact':
        # Aggregate slice data into video data.
        video_data = video_mask * fisher_vectors

        # Apply exact normalization on the test video data.
        video_data = power_normalize(video_data, 0.5) 
        if tr_std is not None: video_data = video_data / tr_std
        video_data = exact_l2_normalize(video_data) 

        # Apply linear classifier.
        weight, bias = compute_weights(clf, tr_data, tr_std=None)
        predictions = np.sum(- video_data * weight, axis=1)

    predictions += bias

    if verbose > 1:
        print cls,

    return cls, predictions


def evaluation(
    src_cfg, sqrt_type, empirical_standardization, l2_norm_type,
    prediction_type, nr_slices_to_aggregate=1, nr_threads=4, verbose=0):

    if verbose:
        print "Loading train data."

    agg_suffix = ('_aggregated_%d' % nr_slices_to_aggregate)
    tr_outfile = ('/scratch2/clear/oneata/tmp/joblib/%s_train%s.dat' % (src_cfg, agg_suffix))
    te_outfile = ('/scratch2/clear/oneata/tmp/joblib/' + src_cfg + '_test_%d' + '%s.dat' % agg_suffix)

    if src_cfg != 'dummy':

        dataset = Dataset(
            CFG[src_cfg]['dataset_name'],
            **CFG[src_cfg]['dataset_params'])

        te_samples, _ = dataset.get_data('test')
        CHUNK_SIZE = 1000

        def train_loader(outfile):
            return load_train_video_data(
                dataset, nr_slices_to_aggregate=nr_slices_to_aggregate,
                outfile=outfile, verbose=verbose)

        def test_loader(samples, outfile):
            return load_slices(
                dataset, samples,
                nr_slices_to_aggregate=nr_slices_to_aggregate, outfile=outfile,
                verbose=verbose)

    else: # Hack.

        te_samples = [1, 3]
        CHUNK_SIZE = 1

        def train_loader(outfile):
            data, labels, counts, l2_norms, _, _ = load_dummy_data(0, outfile=outfile)
            return data, labels, counts, l2_norms

        def test_loader(samples, outfile):
            return load_dummy_data(samples[0], outfile=outfile)

    # Load all the train data at once as it's presuambly small (no slices needed).
    (tr_video_data, tr_video_labels, tr_video_counts,
     tr_video_l2_norms) = train_loader(outfile=tr_outfile)
    # vd, vl, vc, vl2 = load_train_video_data(None, outfile="/scratch2/clear/oneata/tmp/joblib/hmdb_split1.stab_train_aggregated_2.dat")

    if verbose:
        print "Normalizing train data."
        print "\tSigned square rooting:", sqrt_type
        print "\tEmpirical standardization:", empirical_standardization
        print "\tL2 norm:", l2_norm_type

    def l2_normalize(data):
        if l2_norm_type == 'exact':
            return exact_l2_normalize(tr_video_data)
        elif l2_norm_type == 'approx':
            return approx_l2_normalize(tr_video_data, tr_video_l2_norms, tr_video_counts)
        else:
            assert False

    def square_root(data):
        if sqrt_type == 'exact':
            return power_normalize(data, 0.5)
        elif sqrt_type == 'approx':
            return approximate_signed_sqrt(
                data, tr_video_counts, pi_derivatives=False, verbose=verbose)
        else:
            assert False

    # Square rooting.
    tr_video_data = square_root(tr_video_data)

    # Empirical standardization.
    if empirical_standardization:
        scaler = StandardScaler(with_mean=False)
        tr_video_data = scaler.fit_transform(tr_video_data)
        tr_std = scaler.std_
    else:
        tr_std = None

    # L2 normalization ("true" or "approx").
    tr_video_data = l2_normalize(tr_video_data)

    # Computing kernel.
    tr_kernel = np.dot(tr_video_data, tr_video_data.T)

    if verbose > 1:
        print '\tTrain data:   %dx%d.' % tr_video_data.shape
        print '\tTrain kernel: %dx%d.' % tr_kernel.shape

    if verbose:
        print "Training classifier."

    eval = Evaluation(CFG[src_cfg]['eval_name'], **CFG[src_cfg]['eval_params'])
    eval.fit(tr_kernel, tr_video_labels)

    if verbose:
        print "Loading test data."

    true_labels = defaultdict(list)
    predictions = defaultdict(list)

    for ii, low in enumerate(xrange(0, len(te_samples), CHUNK_SIZE)):

        if verbose:
            print "\tPart %3d from %5d to %5d." % (ii, low, low + CHUNK_SIZE)
            print "\tEvaluating on %d threads." % nr_threads

        (fisher_vectors, te_labels, slice_vw_counts, slice_vw_l2_norms,
         video_mask, visual_word_mask) = test_loader(te_samples, te_outfile % ii)

        eval_args = [
            (ii, eval, tr_video_data, tr_std, fisher_vectors, slice_vw_counts,
             slice_vw_l2_norms, video_mask, visual_word_mask, prediction_type,
             te_outfile, verbose)
            for ii in xrange(eval.nr_classes)]
        evaluator = threads.ParallelIter(nr_threads, eval_args, evaluate_worker)

        if verbose > 1: print "\t\tClasses:",
        for ii, pd in evaluator:
            tl = eval.lb.transform(te_labels)[:, ii]
            true_labels[ii].append(tl)
            predictions[ii].append(pd)
        if verbose > 1: print 

    # Prepare labels.
    for cls in true_labels.keys(): 
        true_labels[cls] = np.hstack(true_labels[cls]).squeeze()
        predictions[cls] = np.hstack(predictions[cls]).squeeze()

    # Score results.
    metric = CFG[src_cfg]['metric']
    if metric == 'average_precision':
        compute_average_precision(true_labels, predictions, verbose=verbose)
    elif metric == 'accuracy':
        compute_accuracy(eval.lb, true_labels, predictions, verbose=verbose)
    else:
        assert False, "Unknown metric %s." % metric


def main():
    parser = argparse.ArgumentParser(
        description="Evaluating the normalization approximations.")

    parser.add_argument(
        '-d', '--dataset', required=True, choices=CFG.keys(),
        help="which dataset (use `dummy` for debugging purposes).")
    parser.add_argument(
        '--exact', action='store_true', default=False,
        help="uses exact normalizations at both train and test time.")
    parser.add_argument(
        '-e_std', '--empirical_standardization', default=False,
        action='store_true', help="normalizes data to have unit variance.")
    parser.add_argument(
        '--train_l2_norm', choices={'exact', 'approx'}, required=True,
        help="how to apply L2 normalization at train time.")
    parser.add_argument(
        '-nt', '--nr_threads', type=int, default=1, help="number of threads.")
    parser.add_argument(
        '--nr_slices_to_aggregate', type=int, default=1,
        help="aggregates consecutive FVs.")
    parser.add_argument(
        '-v', '--verbose', action='count', help="verbosity level.")
    args = parser.parse_args()

    tr_sqrt = 'approx'
    pred_type = 'approx'
    tr_l2_norm = args.train_l2_norm

    if args.exact:
        tr_sqrt = 'exact'
        tr_l2_norm = 'exact'
        pred_type = 'exact'

    evaluation(
        args.dataset, tr_sqrt, args.empirical_standardization, tr_l2_norm,
        pred_type, nr_slices_to_aggregate=args.nr_slices_to_aggregate,
        nr_threads=args.nr_threads, verbose=args.verbose)


if __name__ == '__main__':
    main()

