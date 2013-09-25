""" Uses approximations for both signed square rooting and l2 normalization."""
import argparse
from multiprocessing import Pool
import numpy as np
import pdb
from pdb import set_trace
import os

# from ipdb import set_trace
from joblib import Memory
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from yael import threads

from dataset import Dataset
from fisher_vectors.evaluation import Evaluation
from fisher_vectors.evaluation.utils import average_precision
from fisher_vectors.model.utils import compute_L2_normalization

from load_data import approximate_signed_sqrt
from load_data import load_kernels
from load_data import load_sample_data


# TODO Possible improvements:
# [ ] Use sparse matrices for masks, especially for `video_agg_mask`.
# [x] Use also empirical standardization.
# [x] Load dummy data.
# [x] Parallelize per-class evaluation.


cache_dir = os.path.expanduser('~/scratch2/tmp')
memory = Memory(cachedir=cache_dir)

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
    },
    'hollywood2':{
        'dataset_name': 'hollywood2',
        'dataset_params': {
            'ip_type': 'dense5.track15mbh',
            'nr_clusters': 256,
            'suffix': '.per_slice.delta_60'
        },
        'eval_name': 'hollywood2',
        'eval_params': {
        },
    },
    'dummy': {
        'dataset_name': '',
        'dataset_params': {
        },
        'eval_name': 'hollywood2',
        'eval_params': {
        },
    }
}


def load_dummy_data(seed):
    N_SAMPLES = 100
    N_CENTERS = 5
    N_FEATURES = 20
    K, D = 2, 5

    te_data, te_labels = make_blobs(
        n_samples=N_SAMPLES, centers=N_CENTERS,
        n_features=N_FEATURES, random_state=seed)

    te_video_mask = np.eye(N_SAMPLES)
    te_visual_word_mask = build_visual_word_mask(D, K)

    np.random.seed(seed)
    te_counts = np.random.rand(N_SAMPLES, K)
    te_l2_norms = np.dot(te_data ** 2, te_visual_word_mask)

    return te_data, te_labels, te_counts, te_l2_norms, te_video_mask, te_visual_word_mask


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

    return mask.T


def build_visual_word_mask(D, K):
    """ Mask to aggregate a Fisher vector into per visual word values. """
    I = np.eye(K)
    return np.hstack((I.repeat(D, axis=1), I.repeat(D, axis=1))).T


def scale_by(fisher_vectors, nr_descriptors, video_mask):
    return (
        fisher_vectors * nr_descriptors[:, np.newaxis]
        / np.dot(np.dot(video_mask, nr_descriptors),
                 video_mask)[:, np.newaxis])


def visual_word_l2_norm(fisher_vectors, visual_word_mask):
    return np.dot(fisher_vectors ** 2, visual_word_mask)  # NxK


def visual_word_scores(fisher_vectors, weights, bias, visual_word_mask):
    return np.dot(- fisher_vectors * weights, visual_word_mask)# + bias  # NxK


def approximate_video_scores(
    slice_scores, slice_counts, slice_l2_norms, video_mask):

    video_scores = np.dot(video_mask, slice_scores)
    video_counts = np.dot(video_mask, slice_counts)
    video_l2_norm = np.dot(video_mask, slice_l2_norms)

    sqrt_scores = np.sum(video_scores / np.sqrt(video_counts), axis=1)
    approx_l2_norm = np.sum(video_l2_norm / video_counts, axis=1)

    return sqrt_scores / np.sqrt(approx_l2_norm)


@memory.cache
def load_slices(dataset, samples):
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

        nn = fv.shape[0]
        names += [sample.movie] * nn
        labels += [ii['label']]
        fisher_vectors.append(fv)
        counts.append(cc)

        nd = info['nr_descs']
        nr_descs.append(nd[nd != 0])

    D, K = 64, dataset.VOC_SIZE
    fisher_vectors = np.vstack(fisher_vectors)
    counts = np.vstack(counts)
    nr_descs = np.hstack(nr_descs)

    video_mask = build_aggregation_mask(names)
    visual_word_mask = build_visual_word_mask(D, K)

    fisher_vectors = scale_by(fisher_vectors, nr_descs, video_mask)
    slice_vw_counts = scale_by(counts, nr_descs, video_mask)
    slice_vw_l2_norms = visual_word_l2_norm(fisher_vectors, visual_word_mask)

    return fisher_vectors, labels, slice_vw_counts, slice_vw_l2_norms, video_mask, visual_word_mask


def aggregate(fisher_vectors, mask, nr_descriptors=None):
    """ Aggregates per-slice data into per-video data. """
    if nr_descriptors is None:
        nr_descriptors = np.ones(mask.shape[1])
    return (np.dot(mask, nr_descriptors[:, np.newaxis] * fisher_vectors) /
            np.dot(mask, nr_descriptors)[:, np.newaxis])


def evaluate_worker(
    (ii, eval, tr_data, tr_std, te_data, te_labels, visual_word_mask,
     video_mask, slice_vw_counts, slice_vw_l2_norms)):
    true_labels = te_labels[:, ii]
    weight, bias = compute_weights(eval.clf[ii], tr_data, tr_std)
    slice_vw_scores = visual_word_scores(te_data, weight, bias, visual_word_mask)
    predictions = approximate_video_scores(slice_vw_scores, slice_vw_counts, slice_vw_l2_norms, video_mask)
    return ii, average_precision(true_labels, predictions)


def evaluation(
    src_cfg, tr_l2_norm_type, empirical_standardization, nr_threads=4,
    verbose=0):

    if verbose:
        print "Loading train data."

    if src_cfg != 'dummy':
        dataset = Dataset(
            CFG[src_cfg]['dataset_name'],
            **CFG[src_cfg]['dataset_params'])
        tr_samples, _ = dataset.get_data('train')
        SAMPLES_CHUNK = 1000
        NR_SAMPLES = len(tr_samples)
        def loader(samples): return load_slices(dataset, samples)
        def chunker():
            for low in range(0, NR_SAMPLES, SAMPLES_CHUNK):
                yield tr_samples[low: low + SAMPLES_CHUNK]
    else:
        # Hack.
        def loader(seed): return load_dummy_data(0)
        def chunker(): yield 0

    # Memory friendly loading: load data in small chunks and aggregated them in
    tr_video_data = []
    tr_video_counts = []
    tr_video_l2_norms = []
    tr_video_labels = []

    # Fisher vectors for each video.
    for samples in chunker():
        (tr_data, tr_labels, tr_counts, tr_l2_norms, tr_video_mask,
         tr_visual_word_mask) = loader(samples)

        tr_video_data.append(np.dot(tr_video_mask, tr_data))
        tr_video_counts.append(np.dot(tr_video_mask, tr_counts))
        tr_video_l2_norms.append(np.dot(tr_video_mask, tr_l2_norms))
        tr_video_labels += tr_labels

    tr_video_data = np.hstack(tr_video_data)
    tr_video_counts = np.hstack(tr_video_counts)
    tr_video_l2_norms = np.hstack(tr_video_l2_norms)

    if verbose:
        print "Normalizing train data."

    def l2_norm(type_):
        if type_ == 'true':
            l2 = compute_L2_normalization(tr_video_data)
        elif type_ == 'approx':
            l2 = np.sum(tr_video_l2_norms / tr_video_counts, axis=1)
        else:
            assert False, "Unknown L2 norm type."
        return np.sqrt(l2[:, np.newaxis])

    # Square rooting.
    tr_video_data = approximate_signed_sqrt(
            tr_video_data, tr_video_counts,
            pi_derivatives=False, verbose=verbose)

    # Empirical standardization.
    if empirical_standardization:
        scaler = StandardScaler(with_mean=False)
        tr_video_data = scaler.fit_transform(tr_video_data)
        tr_std = scaler.std_
    else:
        tr_std = None

    # L2 normalization ("true" or "approx").
    tr_video_data = tr_video_data / l2_norm(tr_l2_norm_type)

    tr_kernel = np.dot(tr_video_data, tr_video_data.T)

    if verbose > 1:
        print '\tTrain data:   %dx%d.' % tr_video_data.shape
        print '\tTrain kernel: %dx%d.' % tr_kernel.shape

    if verbose:
        print "Training classifier."

    eval = Evaluation(CFG[src_cfg]['eval_name'], **CFG[src_cfg]['eval_params'])
    eval.fit(tr_kernel, tr_labels)

    if verbose:
        print "Loading test data."

    if src_cfg != 'dummy':
        te_samples, _ = dataset.get_data('test')
        te_data, te_labels, te_counts, te_l2_norms, te_video_mask, te_visual_word_mask = load_slices(dataset, te_samples)
    else:
        te_data, te_labels, te_counts, te_l2_norms, te_video_mask, te_visual_word_mask = load_dummy_data(1)

    if src_cfg in ('hollywood2', 'dummy'):
        te_labels = eval.lb.transform(te_labels)

    if verbose > 1:
        print "\tTest data (slices): %dx%d." % te_data.shape
        print "\tNumber of samples: %d." % len(te_labels)

    if verbose:
        print "Evaluating on %d threads." % nr_threads

    evaluator = threads.ParallelIter(
        nr_threads,
        [(ii, eval, tr_video_data, tr_std, te_data, te_labels, te_visual_word_mask, te_video_mask, te_counts, te_l2_norms)
         for ii in xrange(eval.nr_classes)], evaluate_worker)

    average_precisions = []
    for ii, ap in evaluator:
        average_precisions.append(ap)
        if verbose: print "%3d %5.2f" % (ii, 100 * ap)
    if verbose: print '---------'
    print "mAP %.2f" % (100 * np.mean(average_precisions))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluating the normalization approximations.")

    parser.add_argument(
        '-d', '--dataset', required=True,
        choices={'trecvid11_devt', 'hollywood2', 'dummy'},
        help="which dataset (use `dummy` for debugging purposes).")
    parser.add_argument(
        '-e_std', '--empirical_standardization', default=False,
        action='store_true', help="normalizes data to have unit variance.")
    parser.add_argument(
        '--train_l2_norm', choices={'true', 'approx'}, required=True,
        help="how to apply L2 normalization at train time.")
    parser.add_argument(
        '-nt', '--nr_threads', type=int, default=1, help="number of threads.")
    parser.add_argument(
        '-v', '--verbose', action='count', help="verbosity level.")
    args = parser.parse_args()

    evaluation(
        args.dataset, args.train_l2_norm, args.empirical_standardization,
        nr_threads=args.nr_threads, verbose=args.verbose)


if __name__ == '__main__':
    main()

