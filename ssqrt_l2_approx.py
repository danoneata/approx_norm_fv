""" Uses approximations for both signed square rooting and l2 normalization."""
import argparse
from multiprocessing import Pool
import numpy as np
import pdb
from pdb import set_trace
import os

# from ipdb import set_trace
from joblib import Memory
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
# [ ] Load dummy data.
# [x] Parallelize per-class evaluation.


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
    (ii, eval, tr_data, te_data, te_labels, visual_word_mask, video_mask,
    slice_vw_counts, slice_vw_l2_norms)):
    true_labels = te_labels[:, ii]
    weight, bias = compute_weights(eval.clf[ii], tr_data)
    slice_vw_scores = visual_word_scores(te_data, weight, bias, visual_word_mask)
    predictions = approximate_video_scores(slice_vw_scores, slice_vw_counts, slice_vw_l2_norms, video_mask)
    return ii, average_precision(true_labels, predictions)


def evaluation(nr_threads=4, verbose=0):
    D, K = 64, 256
    dataset = Dataset('hollywood2', suffix='.per_slice.delta_60', nr_clusters=K)

    if verbose:
        print "Loading train data."

    tr_samples, _ = dataset.get_data('train')
    tr_data, tr_labels, tr_counts, tr_l2_norms, tr_video_mask, tr_visual_word_mask = load_slices(dataset, tr_samples)

    tr_video_data = np.dot(tr_video_mask, tr_data)
    tr_video_counts = np.dot(tr_video_mask, tr_counts)
    sqrt_tr_video_data = approximate_signed_sqrt(tr_video_data, tr_video_counts, pi_derivatives=False, verbose=verbose)

    tr_kernel = np.dot(sqrt_tr_video_data, sqrt_tr_video_data.T)
    
    #tr_kernel_2, tr_labels_2, _, tr_data_2 = load_kernels(dataset, tr_norms=['sqrt_cnt'], analytical_fim=True, only_train=True)

    if verbose > 1:
        print '\tTrain data:   %dx%d.' % sqrt_tr_video_data.shape
        print '\tTrain kernel: %dx%d.' % tr_kernel.shape

    if verbose:
        print "Training classifier."

    eval = Evaluation('hollywood2')
    eval.fit(tr_kernel, tr_labels)

    if verbose:
        print "Loading test data."

    te_samples, _ = dataset.get_data('test')
    te_data, te_labels, te_counts, te_l2_norms, te_video_mask, te_visual_word_mask = load_slices(dataset, te_samples)
    te_labels = eval.lb.transform(te_labels)

    if verbose > 1:
        print "\tTest data (slices): %dx%d." % te_data.shape
        print "\tNumber of samples: %d." % len(te_labels)

    if verbose:
        print "Evaluating on %d threads." % nr_threads

    evaluator = threads.ParallelIter(
        nr_threads,
        [(ii, eval, sqrt_tr_video_data, te_data, te_labels, te_visual_word_mask, te_video_mask, te_counts, te_l2_norms)
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
        '-nt', '--nr_threads', type=int, default=1, help="number of threads.")
    parser.add_argument(
        '-v', '--verbose', action='count', help="verbosity level.")
    args = parser.parse_args()

    evaluation(nr_threads=args.nr_threads, verbose=args.verbose)


if __name__ == '__main__':
    main()

