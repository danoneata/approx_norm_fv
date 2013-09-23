""" Uses approximations for both signed square rooting and l2 normalization."""
from multiprocessing import Pool
import numpy as np
import pdb
from pdb import set_trace
import os

# from ipdb import set_trace
from joblib import Memory

from dataset import Dataset
from fisher_vectors.evaluation import Evaluation
from fisher_vectors.evaluation.utils import average_precision
from fisher_vectors.model.utils import compute_L2_normalization

from load_data import load_kernels
from load_data import load_sample_data


# TODO Possible improvements:
# [ ] Use sparse matrices for masks, especially for `video_agg_mask`.
# [ ] Parallelize per-class evaluation.


cache_dir = os.path.expanduser('~/scratch2/tmp')
memory = Memory(cachedir=cache_dir)


def compute_weights(clf, xx):
    return (
        np.dot(clf.dual_coef_, xx[clf.support_]),
        clf.intercept_)


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

    return sqrt_scores #  / approx_l2_norm


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

    fisher_vectors = np.vstack(fisher_vectors)
    counts = np.vstack(counts)
    nr_descs = np.hstack(nr_descs)

    return names, labels, fisher_vectors, counts, nr_descs


def aggregate(fisher_vectors, mask, nr_descriptors=None):
    """ Aggregates per-slice data into per-video data. """
    if nr_descriptors is None:
        nr_descriptors = np.ones(mask.shape[1])
    return (np.dot(mask, nr_descriptors[:, np.newaxis] * fisher_vectors) /
            np.dot(mask, nr_descriptors)[:, np.newaxis])


def test_aggregate():
    dataset = Dataset('hollywood2', suffix='.per_slice.delta_60', nr_clusters=256)
    samples, _ = dataset.get_data('test')
    samples = samples[:10]
    names, _, slice_fisher_vectors, _, nr_descriptors = load_slices(dataset, samples)
    mask = build_aggregation_mask(names)
    video_fisher_vectors_1 = aggregate(slice_fisher_vectors, mask, nr_descriptors=None)
    video_fisher_vectors_2, _, _, info = load_sample_data(
        dataset, 'test', analytical_fim=True, pi_derivatives=False,
        sqrt_nr_descs=False, return_info=True)
    pdb.set_trace()


def test_evaluation():
    D, K = 64, 256
    dataset = Dataset('hollywood2', suffix='.per_slice.delta_60', nr_clusters=K)

    # Load data.
    tr_kernel, tr_labels, scalers, tr_data = load_kernels(
        dataset, tr_norms=['sqrt_cnt'], te_norms=[], analytical_fim=True,
        pi_derivatives=False, sqrt_nr_descs=False, only_train=True)

    # Training.
    eval = Evaluation('hollywood2')
    eval.fit(tr_kernel, tr_labels)

    # Evaluation.
    te_samples, _ = dataset.get_data('test')
    te_names, te_labels, te_data, te_counts, nr_descs = load_slices(dataset, te_samples)
    te_labels = eval.lb.transform(te_labels)

    video_mask = build_aggregation_mask(te_names)
    visual_word_mask = build_visual_word_mask(D, K)

    fisher_vectors = scale_by(te_data, nr_descs, video_mask)
    slice_vw_counts = scale_by(te_counts, nr_descs, video_mask)
    slice_vw_l2_norms = visual_word_l2_norm(fisher_vectors, visual_word_mask)

    ap = []
    for ii in xrange(eval.nr_classes):
        true_labels = te_labels[:, ii]
        weight, bias = compute_weights(eval.clf[ii], tr_data)
        slice_vw_scores = visual_word_scores(fisher_vectors, weight, bias, visual_word_mask)
        predictions = approximate_video_scores(slice_vw_scores, slice_vw_counts, slice_vw_l2_norms, video_mask)
        ap.append(average_precision(true_labels, predictions))
        print "%.2f" % (100 * ap[-1])

    print '-----'
    print "%.2f" % (100 * np.mean(ap))


def main():
    test_evaluation()


if __name__ == '__main__':
    main()
