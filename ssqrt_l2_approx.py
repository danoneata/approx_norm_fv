""" Uses approximations for both signed square rooting and l2 normalization."""
import argparse
from collections import defaultdict
import cPickle
from multiprocessing import Pool
import numpy as np
import pdb
import os
from scipy import sparse

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

from load_data import CACHE_PATH
from load_data import CFG
from load_data import approximate_signed_sqrt
from load_data import load_kernels
from load_data import load_sample_data
from load_data import load_video_data
from load_data import my_cacher
from load_data import SliceData


# TODO Possible improvements:
# [ ] Share the `SliceData` data structure with the `detection.py` module.
# [ ] Isolate the dataset configuration into another module and the loading functions.
# [ ] Isolate the utils.
# [ ] Pre-allocate test data (counts, L2 norms and scores).
# [x] Evaluate with the exact normalizations.
# [x] Use sparse matrices for masks.
# [x] Use also empirical standardization.
# [x] Load dummy data.
# [x] Parallelize per-class evaluation.


LOAD_SAMPLE_DATA_PARAMS = {
    'pi_derivatives' : False,
    'sqrt_nr_descs'  : False,
    'return_info'    : True,
}


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


def compute_approx_l2_normalization(l2_norms, counts):
    zero_idxs = counts == 0
    masked_norms = np.ma.masked_array(l2_norms, zero_idxs)
    masked_counts = np.ma.masked_array(counts, zero_idxs)
    masked_result = masked_norms / masked_counts
    return np.sum(masked_result.filled(0), axis=1)


def approx_l2_normalize(data, l2_norms, counts):
    zero_idxs = counts == 0
    masked_norms = np.ma.masked_array(l2_norms, zero_idxs)
    masked_counts = np.ma.masked_array(counts, zero_idxs)
    masked_result = masked_norms / masked_counts
    approx_l2_norm = compute_approx_l2_normalization(l2_norms, counts)
    return data / np.sqrt(approx_l2_norm[:, np.newaxis])


def approximate_video_scores(
    slice_scores, slice_counts, slice_l2_norms, nr_descriptors, video_mask):

    video_scores = sum_by(slice_scores, video_mask) / sum_by(nr_descriptors, video_mask)
    video_counts = sum_by(slice_counts, video_mask) / sum_by(nr_descriptors, video_mask)
    video_l2_norms = sum_by(slice_l2_norms, video_mask) / sum_by(nr_descriptors, video_mask) ** 2

    zero_idxs = video_counts == 0
    masked_scores = np.ma.masked_array(video_scores, zero_idxs)
    masked_counts = np.ma.masked_array(video_counts, zero_idxs)
    masked_l2_norms = np.ma.masked_array(video_l2_norms, zero_idxs)

    sqrt_scores = np.sum((masked_scores / np.sqrt(masked_counts)).filled(0), axis=1)
    approx_l2_norm = np.sum((masked_l2_norms / masked_counts).filled(0), axis=1)

    return sqrt_scores / np.sqrt(approx_l2_norm)


def sum_by(data, mask=None):
    """Sums together rows of `data` according to the sparse matrix `mask`. If
    `mask` is None, then sums all the rows.
   
    """
    if mask is None:
        return np.sum(data, axis=0)
    return mask * data


def expand_by(data, mask=None):
    if mask is None:
        return data
    return mask.T * data


def scale_by(data, coef, mask=None):
    """Multiplies each row of `data` by a normalized version of `coef`; the
    normalization is specified by `mask`. If `mask` is None, the normalization
    term is the sum of all elements in `coef`.

    """
    coef_ = coef[:, np.newaxis]
    return data * coef_ / expand_by(sum_by(coef_, mask), mask)


def scale_and_sum_by(data, coef, data_mask=None, coef_mask=None):
    """Combines two of the previous functions: first scales the rows of `data`
    by `coef` given the mask `coef_mask`; then aggreagtes the scaled rows of
    `data` by a possibly different mask `data_mask`.

    """
    return sum_by(scale_by(data, coef, coef_mask), data_mask)


def sum_and_scale_by(data, coef, mask=None):
    coef_ = coef[:, np.newaxis]
    return sum_by(data * coef_, mask=mask) / sum_by(coef_, mask=mask)


def sum_and_scale_by_squared(data, coef, mask=None):
    coef_ = coef[:, np.newaxis]
    return sum_by(data * (coef_ ** 2), mask=mask) / sum_by(coef_, mask=mask) ** 2


@my_cacher('np')
def load_corrected_norms(
    dataset, samples, nr_slices_to_aggregate, scalers, analytical_fim,
    verbose=0, outfile=None):

    jj = 0
    N = len(samples)
    D, K = dataset.D, dataset.VOC_SIZE

    visual_word_mask = build_visual_word_mask(D, K)
    tr_l2_norms = np.zeros((N, K), dtype=np.float32)
    names = []

    for sample in samples:

        fv, ii, cc, _ = load_sample_data(
            dataset, sample, analytical_fim=analytical_fim,
            **LOAD_SAMPLE_DATA_PARAMS)

        if len(fv) == 0 or str(sample) in names:
            continue

        nd = ii['nr_descs']
        nd = nd[nd != 0]
        ll = ii['label']

        slice_agg_mask = build_slice_agg_mask(fv.shape[0], nr_slices_to_aggregate)
        agg_fisher_vectors = sum_by(fv * nd[:, np.newaxis], mask=slice_agg_mask) / nd.sum()
        for scaler in scalers:
            if scaler is None:
                continue
            agg_fisher_vectors = scaler.transform(agg_fisher_vectors)
        tr_l2_norms[jj] = np.sum(
            visual_word_l2_norm(agg_fisher_vectors, visual_word_mask), axis=0)
        names.append(str(sample))

        jj += 1

        if verbose:
            print '%5d %5d %s' % (jj, fv.shape[0], sample.movie)

    return [tr_l2_norms[:jj]]


@my_cacher('np', 'np', 'np', 'np', 'cp', 'cp')
def load_slices(dataset, samples, analytical_fim=True, outfile=None, verbose=0):

    counts = []
    fisher_vectors = []
    labels = []
    names = []
    nr_descs = []
    nr_slices = []

    for jj, sample in enumerate(samples):

        fv, ii, cc, info = load_sample_data(
            dataset, sample, analytical_fim=analytical_fim,
            **LOAD_SAMPLE_DATA_PARAMS)

        if len(fv) == 0:
            continue

        if str(sample) in names:
            continue

        nd = info['nr_descs']
        nd = nd[nd != 0]
        label = ii['label']

        #slice_agg_mask = build_slice_agg_mask(fv.shape[0], nr_slices_to_aggregate)

        #agg_fisher_vectors = scale_and_sum_by(fv, nd, data_mask=slice_agg_mask, coef_mask=slice_agg_mask)
        #agg_counts = scale_and_sum_by(cc, nd, data_mask=slice_agg_mask, coef_mask=slice_agg_mask)

        fisher_vectors.append(fv)
        counts.append(cc)
        nr_descs.append(nd)

        nr_slices.append(fisher_vectors[-1].shape[0])
        names.append(str(sample))
        labels += [label]

        if verbose:
            print '%5d %5d %s' % (jj, nr_slices[-1], sample.movie)

    fisher_vectors = np.vstack(fisher_vectors)
    counts = np.vstack(counts)
    nr_descs = np.hstack(nr_descs)
    nr_slices = np.array(nr_slices)

    return fisher_vectors, counts, nr_descs, nr_slices, names, labels


def slice_aggregator(slice_data, nr_slices, nr_agg):

    # Generate idxs.
    idxs = []
    ii = 0
    for dd in nr_slices:
        idxs += [j / nr_agg for j in range(ii, ii + dd)]
        ii = int(np.ceil((ii + dd) / float(nr_agg))) * nr_agg

    assert len(idxs) == nr_slices.sum()
    mask = build_aggregation_mask(idxs)

    agg_fisher_vectors = scale_and_sum_by(
        slice_data.fisher_vectors, slice_data.nr_descriptors,
        data_mask=mask, coef_mask=mask)
    agg_counts = scale_and_sum_by(
        slice_data.counts, slice_data.nr_descriptors,
        data_mask=mask, coef_mask=mask)
    agg_nr_descs = sum_by(slice_data.nr_descriptors, mask)

    return SliceData(agg_fisher_vectors, agg_counts, agg_nr_descs)


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


def compute_accuracy(true_labels, predictions, verbose=0):
    all_predictions = np.vstack((predictions[ii] for ii in sorted(predictions.keys())))
    all_true_labels = np.vstack((true_labels[ii] for ii in sorted(true_labels.keys())))

    # FIXME Dodgy
    array_true_labels = np.hstack([np.where(row==1)[0] for row in all_true_labels.T])

    predicted_class = np.argmax(all_predictions, axis=0)
    print "Accuracy %6.2f" % (100 * accuracy_score(array_true_labels, predicted_class))


def evaluate_worker((
    cls, eval, tr_data, tr_scalers, slice_data, video_mask, visual_word_mask,
    prediction_type, verbose)):

    clf = eval.get_classifier(cls)
    weight, bias = compute_weights(clf, tr_data, tr_std=None)

    if prediction_type == 'approx':
        slice_vw_counts = slice_data.counts * slice_data.nr_descriptors[:, np.newaxis]
        slice_vw_l2_norms = visual_word_l2_norm(slice_data.fisher_vectors, visual_word_mask)
        slice_vw_scores = visual_word_scores(slice_data.fisher_vectors, weight, bias, visual_word_mask)
        predictions = approximate_video_scores(
            slice_vw_scores, slice_vw_counts, slice_vw_l2_norms,
            slice_data.nr_descriptors[:, np.newaxis], video_mask)
    elif prediction_type == 'exact':
        # Aggregate slice data into video data.
        video_data = (
            sum_by(slice_data.fisher_vectors, video_mask) /
            sum_by(slice_data.nr_descriptors, video_mask)[:, np.newaxis])

        # Apply exact normalization on the test video data.
        if tr_scalers[0] is not None:
            video_data = tr_scalers[0].transform(video_data)
        video_data = power_normalize(video_data, 0.5)
        if tr_scalers[1] is not None:
            video_data = tr_scalers[1].transform(video_data)
        video_data = exact_l2_normalize(video_data)

        # Apply linear classifier.
        predictions = np.sum(- video_data * weight, axis=1)

    predictions += bias

    if verbose > 1:
        print cls,

    return cls, predictions


def load_normalized_tr_data(
    dataset, nr_slices_to_aggregate, l2_norm_type, empirical_standardizations,
    sqrt_type, analytical_fim, tr_outfile, verbose, samples=None):

    D, K = dataset.D, dataset.VOC_SIZE

    if samples is None:
        samples, _ = dataset.get_data('train')

    tr_video_data, tr_video_counts, tr_video_labels = load_video_data(
        dataset, samples, analytical_fim=analytical_fim, verbose=verbose,
        outfile=tr_outfile)

    if verbose:
        print "Normalizing train data."
        print "\tAnalytical Fisher information matrix:", analytical_fim
        print "\tEmpirical standardization:", empirical_standardizations[0]
        print "\tSigned square rooting:", sqrt_type
        print "\tEmpirical standardization:", empirical_standardizations[1]
        print "\tL2 norm:", l2_norm_type

    def l2_normalize(data, **kwargs):
        if l2_norm_type == 'exact':
            return exact_l2_normalize(data)
        elif l2_norm_type == 'approx':
            if sqrt_type == 'none':
                counts = np.ones(tr_video_counts.shape)
            else:
                counts = tr_video_counts
            # Prepare the L2 norms using the possibly modified `tr_slice_data`.
            scalers = kwargs.get('scalers')
            norm_filename = tr_outfile + ".norm_slices%d_scaler%s" % (
                nr_slices_to_aggregate, np.any(scalers))
            tr_video_l2_norms = load_corrected_norms(
                dataset, samples, nr_slices_to_aggregate,
                analytical_fim=analytical_fim, scalers=scalers,
                verbose=verbose, outfile=norm_filename)[0]
            return approx_l2_normalize(data, tr_video_l2_norms, counts)
        elif l2_norm_type == 'none':
            return data
        else:
            assert False

    def square_root(data):
        if sqrt_type == 'exact':
            return power_normalize(data, 0.5)
        elif sqrt_type == 'approx':
            return approximate_signed_sqrt(
                data, tr_video_counts, pi_derivatives=False, verbose=verbose)
        elif sqrt_type == 'none':
            return data
        else:
            assert False

    scalers = []

    if empirical_standardizations[0]:
        scaler = StandardScaler(with_mean=False)
        tr_video_data = scaler.fit_transform(tr_video_data)
        scalers.append(scaler)
    else:
        scalers.append(None)

    # Square rooting.
    tr_video_data = square_root(tr_video_data)

    # Empirical standardization.
    if empirical_standardizations[1]:
        scaler = StandardScaler(with_mean=False)
        tr_video_data = scaler.fit_transform(tr_video_data)
        scalers.append(scaler)
    else:
        scalers.append(None)

    # L2 normalization ("true" or "approx").
    tr_video_data = l2_normalize(tr_video_data, scalers=scalers)

    return tr_video_data, tr_video_labels, scalers


def predict_main(
    src_cfg, sqrt_type, empirical_standardizations, l2_norm_type,
    prediction_type, analytical_fim, part, nr_slices_to_aggregate=1,
    nr_threads=4, verbose=0):

    dataset = Dataset(CFG[src_cfg]['dataset_name'], **CFG[src_cfg]['dataset_params'])
    D, K = dataset.D, dataset.VOC_SIZE

    if verbose:
        print "Loading train data."

    tr_outfile = os.path.join(
        CACHE_PATH, "%s_train_afim_%s_pi_%s_sqrt_nr_descs_%s.dat" % (
            src_cfg, analytical_fim, False, False))
    tr_video_data, tr_video_labels, tr_scalers = load_normalized_tr_data(
        dataset, nr_slices_to_aggregate, l2_norm_type,
        empirical_standardizations, sqrt_type, analytical_fim, tr_outfile,
        verbose)

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

    te_samples, _ = dataset.get_data('test')
    visual_word_mask = build_visual_word_mask(D, K)

    te_outfile = os.path.join(
        CACHE_PATH, "%s_test_afim_%s_pi_%s_sqrt_nr_descs_%s_part_%s.dat" % (
            src_cfg, analytical_fim, False, False, "%d"))

    low = CFG[src_cfg]['samples_chunk'] * part
    high = np.minimum(CFG[src_cfg]['samples_chunk'] * (part + 1), len(te_samples))

    if verbose:
        print "\tPart %3d from %5d to %5d." % (part, low, high)
        print "\tEvaluating on %d threads." % nr_threads

    te_outfile_ii = te_outfile % part
    fisher_vectors, counts, nr_descs, nr_slices, _, te_labels = load_slices(
        dataset, te_samples[low: high], analytical_fim, outfile=te_outfile_ii,
        verbose=verbose)
    slice_data = SliceData(fisher_vectors, counts, nr_descs)

    agg_slice_data = slice_aggregator(slice_data, nr_slices, nr_slices_to_aggregate)
    agg_slice_data = agg_slice_data._replace(
        fisher_vectors=(agg_slice_data.fisher_vectors *
                        agg_slice_data.nr_descriptors[:, np.newaxis]))

    video_mask = build_aggregation_mask(
        sum([[ii] * int(np.ceil(float(nn) / nr_slices_to_aggregate))
             for ii, nn in enumerate(nr_slices)],
            []))

    if verbose:
        print "\tTest data: %dx%d." % agg_slice_data.fisher_vectors.shape

    # Scale the FVs in the main program, to avoid blowing up the memory
    # when doing multi-threading, since each thread will make a copy of the
    # data when transforming the data.
    if prediction_type == 'approx':
        for tr_scaler in tr_scalers:
            if tr_scaler is None:
                continue
            agg_slice_data = agg_slice_data._replace(
                fisher_vectors=tr_scaler.transform(
                    agg_slice_data.fisher_vectors))

    eval_args = [
        (ii, eval, tr_video_data, tr_scalers, agg_slice_data, video_mask,
         visual_word_mask, prediction_type, verbose)
        for ii in xrange(eval.nr_classes)]
    evaluator = threads.ParallelIter(nr_threads, eval_args, evaluate_worker)

    if verbose > 1:
        print "\t\tClasses:",

    true_labels = {}
    predictions = {}

    for ii, pd in evaluator:
        tl = eval.lb.transform(te_labels)[:, ii]
        true_labels[ii] = tl
        predictions[ii] = pd

    if verbose > 1:
        print

    preds_path = os.path.join(
        CACHE_PATH, "%s_predictions_afim_%s_pi_%s_sqrt_nr_descs_%s_nagg_%d_part_%d.dat" % (
            src_cfg, analytical_fim, False, False, nr_slices_to_aggregate, part))

    with open(preds_path, 'w') as ff:
        cPickle.dump(true_labels, ff)
        cPickle.dump(predictions, ff)


def evaluate_main(src_cfg, analytical_fim, nr_slices_to_aggregate, verbose):

    dataset = Dataset(CFG[src_cfg]['dataset_name'], **CFG[src_cfg]['dataset_params'])
    te_samples, _ = dataset.get_data('test')
    nr_parts = int(np.ceil(float(len(te_samples)) / CFG[src_cfg]['samples_chunk']))

    preds_path = os.path.join(
        CACHE_PATH, "%s_predictions_afim_%s_pi_%s_sqrt_nr_descs_%s_nagg_%d_part_%s.dat" % (
            src_cfg, analytical_fim, False, False, nr_slices_to_aggregate, "%d"))

    true_labels = None

    for part in xrange(nr_parts):

        # Loads scores from file.
        with open(preds_path % part, 'r') as ff:
            tl = cPickle.load(ff)
            pd = cPickle.load(ff)

        # Prepares labels.
        if true_labels is None:
            true_labels = tl
            predictions = pd
        else:
            for cls in true_labels.keys():
                true_labels[cls] = np.hstack((true_labels[cls], tl[cls])).squeeze()
                predictions[cls] = np.hstack((predictions[cls], pd[cls])).squeeze()

    # Remove scores of duplicate samples.
    str_te_samples = map(str, te_samples)
    idxs = [str_te_samples.index(elem) for elem in set(str_te_samples)]

    for cls in true_labels.keys():
        true_labels[cls] = true_labels[cls][idxs]
        predictions[cls] = predictions[cls][idxs]

    # Scores results.
    metric = CFG[src_cfg]['metric']
    if metric == 'average_precision':
        compute_average_precision(true_labels, predictions, verbose=verbose)
    elif metric == 'accuracy':
        compute_accuracy(true_labels, predictions, verbose=verbose)
    else:
        assert False, "Unknown metric %s." % metric


def main():
    parser = argparse.ArgumentParser(
        description="Evaluating the normalization approximations.")

    parser.add_argument(
        '-d', '--dataset', required=True, choices=CFG.keys(),
        help="which dataset (use `dummy` for debugging purposes).")
    parser.add_argument(
        '-t', '--task', choices=('predict', 'evaluate'), required=True,
        help="what to do.")
    parser.add_argument(
        '--exact', action='store_true', default=False,
        help="uses exact normalizations at both train and test time.")
    parser.add_argument(
        '--e_std_1', default=False, action='store_true',
        help=("applies empirical standardization at the first stage, before "
              "square rooting."))
    parser.add_argument(
        '--e_std_2', default=False, action='store_true',
        help=("applies empirical standardization at the second stage, after "
              "square rooting."))
    parser.add_argument(
        '--no_afim', default=False, action='store_true',
        help=("uses FVs that are standardized with the analytical Fisher "
              "information matrix."))
    parser.add_argument(
        '--train_l2_norm', choices={'exact', 'approx'}, required=True,
        help="how to apply L2 normalization at train time.")
    parser.add_argument(
        '-nt', '--nr_threads', type=int, default=1, help="number of threads.")
    parser.add_argument(
        '--nr_slices_to_aggregate', type=int, default=1,
        help="aggregates consecutive FVs.")
    parser.add_argument(
        '--part', type=int,
        help=("part of the test data; the batches are of 100 samples."))
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

    analytical_fim = not args.no_afim
    empirical_standardizations = [args.e_std_1, args.e_std_2]

    if args.task == 'predict':
        predict_main(
            args.dataset, tr_sqrt, empirical_standardizations, tr_l2_norm,
            pred_type, analytical_fim, part=args.part,
            nr_slices_to_aggregate=args.nr_slices_to_aggregate,
            nr_threads=args.nr_threads, verbose=args.verbose)
    elif args.task == 'evaluate':
        evaluate_main(args.dataset, analytical_fim, args.nr_slices_to_aggregate, verbose=args.verbose)


if __name__ == '__main__':
    main()

