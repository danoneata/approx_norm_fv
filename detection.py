import argparse
from collections import namedtuple
import cPickle
from fractions import gcd
from itertools import groupby
import numpy as np
import pdb
import os
from scipy import sparse
import sys

from sklearn.preprocessing import Scaler

from dataset import Dataset
from dataset import SampID

from fisher_vectors.evaluation import Evaluation
from fisher_vectors.model.utils import power_normalize
from fisher_vectors.model.utils import compute_L2_normalization

from load_data import approximate_signed_sqrt
from load_data import load_kernels
from load_data import load_sample_data

from result_file_functions import get_det_ap

from ssqrt_l2_approx import approximate_video_scores
from ssqrt_l2_approx import build_slice_agg_mask
from ssqrt_l2_approx import build_visual_word_mask
from ssqrt_l2_approx import compute_weights
from ssqrt_l2_approx import load_normalized_tr_data
from ssqrt_l2_approx import my_cacher
from ssqrt_l2_approx import predict
from ssqrt_l2_approx import scale_and_sum_by
from ssqrt_l2_approx import scale_by
from ssqrt_l2_approx import sum_and_scale_by
from ssqrt_l2_approx import sum_by
from ssqrt_l2_approx import visual_word_l2_norm
from ssqrt_l2_approx import visual_word_scores

from ssqrt_l2_approx import CFG
from ssqrt_l2_approx import LOAD_SAMPLE_DATA_PARAMS


# TODO Things to improve
# [x] Check if frames are contiguous. If not treat them specifically.
# [x] Mask with 1, -1 and integral quantities.
# [ ] Write fast ESS for approximate norms in Cython.
# [ ] Remove duplicate code from `approx_ess` and `cy_approx_ess`.


TRACK_LEN = 15
NULL_CLASS_IDX = 0
SliceData = namedtuple(
    'SliceData', ['fisher_vectors', 'counts', 'nr_descriptors', 'begin_frames',
                  'end_frames'])


def timer(func):
    import time
    def timed_func(*args, **kwargs):
        start = time.time()
        out = func(*args, **kwargs)
        print "Elapsed %.1f s." % (time.time() - start)
        return out
    return timed_func


def mgcd(*args):
    """Greatest common divisor that accepts multiple arguments."""
    return mgcd(args[0], mgcd(*args[1:])) if len(args) > 2 else gcd(*args)


@my_cacher('np', 'np', 'np', 'np', 'np')
def load_data_delta_0(
    dataset, movie, part, class_idx, outfile=None, delta_0=30, verbose=0):
    """ Loads Fisher vectors for the test data for detection datasets. """

    D = 64
    K = dataset.VOC_SIZE
    class_name = dataset.IDX2CLS[class_idx]
    class_limits = dataset.CLASS_LIMITS[movie][class_name][part]

    nr_frames = class_limits[1] - class_limits[0] + 1
    N = nr_frames / delta_0 + 1
    FV_LEN = 2 * D * K

    fisher_vectors = np.zeros((N, FV_LEN), dtype=np.float32)
    counts = np.zeros((N, K), dtype=np.float32)

    nr_descs = np.zeros(N)
    begin_frames = np.zeros(N)
    end_frames = np.zeros(N)

    # Filter samples by movie name.
    samples, _ = dataset.get_data('test')
    samples = [sample for sample in samples if str(sample).startswith(movie)]

    ii = 0  # Slices count.
    for jj, sample in enumerate(samples):

        if verbose: sys.stdout.write("%5d %30s\r" % (jj, str(sample)))

        if sample.bf < class_limits[0] or class_limits[1] < sample.ef:
            continue

        # Read sufficient statistics and associated information.
        sample_fisher_vectors, _, sample_counts, sample_info = load_sample_data(
            dataset, sample, analytical_fim=True, **LOAD_SAMPLE_DATA_PARAMS)

        sample_nr_descs = sample_info['nr_descs']

        # I haven't stored the Fisher vectors for the empty slices. Put zeros
        # when these are missing.
        idxs = np.where(sample_nr_descs != 0)[0]
        fisher_vectors[ii + idxs] = sample_fisher_vectors
        counts[ii + idxs] = sample_counts

        nn = len(sample_nr_descs)
        nr_descs[ii: ii + nn] = sample_nr_descs
        begin_frames[ii: ii + nn] = sample_info['begin_frames']
        end_frames[ii: ii + nn] = sample_info['end_frames']

        # Update the slices count.
        ii += nn

    return fisher_vectors[:ii], counts[:ii], nr_descs[:ii], begin_frames[:ii], end_frames[:ii]


def build_sliding_window_mask(N, nn, dd=1):
    """Builds mask to aggregate a vectors [x_1, x_2, ..., x_N] into
    [x_1 + ... + x_n, x_2 + ... + x_{n+1}, ...].

    """
    M = (N - nn) / dd + 1

    idxs = np.hstack([
        [np.ones(nn, dtype=np.int) * ii,
         np.arange(nn, dtype=np.int) + ii * dd]
        for ii in xrange(M)])

    values = np.ones(idxs.shape[1])

    return sparse.csr_matrix((values, idxs))


def build_integral_sliding_window_mask(N, nn, dd=1):
    """Builds mask for efficient integral sliding window."""
    H = (N - nn) / dd + 1
    W = N + 1

    row_idxs = range(H)
    neg_idxs = [ii * dd for ii in range(H)]
    pos_idxs = [ii * dd + nn for ii in range(H)]

    all_row_idxs = np.hstack((row_idxs, row_idxs))
    all_col_idxs = np.hstack((neg_idxs, pos_idxs))

    idxs = np.vstack((all_row_idxs, all_col_idxs))

    values = np.hstack((
        - np.ones(len(row_idxs)),
        + np.ones(len(row_idxs))))

    return sparse.csr_matrix((values, idxs))


class OverlappingSelector:
    def __init__(self, chunk, stride, containing, integral):
        self.chunk = chunk
        self.stride = stride
        self.integral = integral
        self.containing = containing
        self.mask_builder = (
            build_integral_sliding_window_mask if integral
            else build_sliding_window_mask)

    def get_mask(self, N, window_size):
        track_len = 15 if self.containing else 0
        return self.mask_builder(
            N,
            (window_size - track_len) / self.chunk,
            self.stride / self.chunk)

    def get_frame_idxs(self, N, window_size):
        extra = 15 / self.chunk if self.containing else 0
        nr_agg = window_size / self.chunk
        begin_frame_idxs = np.arange(
            0,
            N - nr_agg + extra + 1,
            self.stride / self.chunk)
        end_frame_idxs = np.arange(
            nr_agg - 1,
            N + extra,
            self.stride / self.chunk)
        end_frame_idxs = np.minimum(N - 1, end_frame_idxs)
        return begin_frame_idxs, end_frame_idxs


class NonOverlappingSelector:
    def __init__(self, nr_agg):
        self.nr_agg = nr_agg

    def get_mask(self, N):
        return build_slice_agg_mask(N, self.nr_agg)

    def get_frame_idxs(self, N):
        begin_frame_idxs = np.arange(0, N, self.nr_agg)
        end_frame_idxs = np.arange(self.nr_agg - 1, N + self.nr_agg - 1, self.nr_agg)
        end_frame_idxs = np.minimum(N - 1, end_frame_idxs)
        return begin_frame_idxs, end_frame_idxs


def aggregate(slice_data, sparse_mask, frame_idxs):
    """Aggregates slices of data into `N` slices."""

    # Scale by number of descriptors and sum.
    agg_fisher_vectors = sum_and_scale_by(
        slice_data.fisher_vectors, slice_data.nr_descriptors, mask=sparse_mask)
    agg_fisher_vectors[np.isnan(agg_fisher_vectors)] = 0

    # Sum counts.
    agg_counts = sum_and_scale_by(
        slice_data.counts, slice_data.nr_descriptors, mask=sparse_mask)
    agg_counts[np.isnan(agg_counts)] = 0

    # Sum number of descriptors.
    agg_nr_descs = sum_by(slice_data.nr_descriptors, mask=sparse_mask)

    # Correct begin frames and end frames.
    begin_frame_idxs, end_frame_idxs = frame_idxs
    agg_begin_frames = slice_data.begin_frames[begin_frame_idxs]
    agg_end_frames = slice_data.end_frames[end_frame_idxs]

    assert len(agg_fisher_vectors) == len(agg_begin_frames) == len(agg_end_frames)

    # Return new slice_data structure.
    return SliceData(
        agg_fisher_vectors, agg_counts, agg_nr_descs, agg_begin_frames,
        agg_end_frames)


def integral(X):
    if X.ndim == 1:
        return np.hstack((0, np.cumsum(X)))
    elif X.ndim == 2:
        return np.vstack((np.zeros((1, X.shape[1])), np.cumsum(X, axis=0)))
    else:
        assert False


def only_positive(X):
    return np.ma.masked_less(X, 0).filled(0)


def only_negative(X):
    return np.ma.masked_greater(X, 0).filled(0)


@timer
def exact_sliding_window(
    slice_data, clf, deltas, selector, scalers, sqrt_type='', l2_norm_type=''):

    results = []
    weights, bias = clf

    nr_descriptors_T = slice_data.nr_descriptors[:, np.newaxis]

    # Multiply by the number of descriptors.
    fisher_vectors = slice_data.fisher_vectors * nr_descriptors_T
    begin_frames, end_frames = slice_data.begin_frames, slice_data.end_frames
    N = fisher_vectors.shape[0]

    if selector.integral:
        fisher_vectors = integral(fisher_vectors)
        nr_descriptors_T = integral(nr_descriptors_T)

    for delta in deltas:

        # Build mask.
        mask = selector.get_mask(N, delta)
        begin_frame_idxs, end_frame_idxs = selector.get_frame_idxs(N, delta)

        # Aggregate data into bigger slices.
        agg_fisher_vectors = (
            sum_by(fisher_vectors, mask) /
            sum_by(nr_descriptors_T, mask))
        agg_fisher_vectors[np.isnan(agg_fisher_vectors)] = 0

        agg_begin_frames = begin_frames[begin_frame_idxs]
        agg_end_frames = end_frames[end_frame_idxs]

        assert len(agg_fisher_vectors) == len(agg_begin_frames) == len(agg_end_frames)

        # Normalize aggregated data.
        if scalers[0] is not None:
            agg_fisher_vectors = scalers[0].transform(agg_fisher_vectors)
        if sqrt_type != 'none':
            agg_fisher_vectors = power_normalize(agg_fisher_vectors, 0.5)
        if scalers[1] is not None:
            agg_fisher_vectors = scalers[1].transform(agg_fisher_vectors)
        # More efficient, to apply L2 on the scores than on the FVs.
        l2_norms = (
            compute_L2_normalization(agg_fisher_vectors)
            if l2_norm_type != 'none'
            else np.ones(len(agg_fisher_vectors)))

        # Predict with the linear classifier.
        scores = (
            - np.dot(agg_fisher_vectors, weights.T).squeeze()
            / np.sqrt(l2_norms)
            + bias)

        nan_idxs = np.isnan(scores)
        results += zip(
            agg_begin_frames[~nan_idxs],
            agg_end_frames[~nan_idxs],
            scores[~nan_idxs])

    return results


@timer
def approx_sliding_window(
    slice_data, clf, deltas, selector, scalers, visual_word_mask):

    results = []
    weights, bias = clf

    # Prepare sliced data.
    fisher_vectors = slice_data.fisher_vectors
    for scaler in scalers:
        if scaler is None:
            continue
        fisher_vectors = scaler.transform(fisher_vectors)
    nr_descriptors_T = slice_data.nr_descriptors[:, np.newaxis]

    # Multiply by the number of descriptors.
    fisher_vectors = fisher_vectors * nr_descriptors_T
    slice_vw_counts = slice_data.counts * nr_descriptors_T

    #
    slice_vw_l2_norms = visual_word_l2_norm(fisher_vectors, visual_word_mask)
    slice_vw_scores = visual_word_scores(fisher_vectors, weights, bias, visual_word_mask)

    if selector.integral:
        slice_vw_counts = integral(slice_vw_counts)
        slice_vw_l2_norms = integral(slice_vw_l2_norms)
        slice_vw_scores = integral(slice_vw_scores)
        nr_descriptors_T = integral(nr_descriptors_T)

    N = fisher_vectors.shape[0]

    for delta in deltas:

        # Build mask.
        mask = selector.get_mask(N, delta)
        begin_frame_idxs, end_frame_idxs = selector.get_frame_idxs(N, delta)

        # Approximated predictions.
        scores = approximate_video_scores(
            slice_vw_scores, slice_vw_counts, slice_vw_l2_norms, nr_descriptors_T, mask)
        agg_begin_frames = slice_data.begin_frames[begin_frame_idxs]
        agg_end_frames = slice_data.end_frames[end_frame_idxs]

        assert len(scores) == len(agg_begin_frames) == len(agg_end_frames)

        nan_idxs = np.isnan(scores)
        results += zip(
            agg_begin_frames[~nan_idxs],
            agg_end_frames[~nan_idxs],
            scores[~nan_idxs])

    return results


@timer
def approx_sliding_window_ess(
    slice_data, clf, deltas, selector, scalers, rescore, visual_word_mask):

    from ess import Bounds
    from ess import efficient_subwindow_search
    from ess import bounds_in_blacklist

    def eval_integral(X, bb):
        low, high = bb
        return X[high] - X[low] if high > low else 0

    weights, bias = clf

    # Prepare sliced data.
    fisher_vectors = slice_data.fisher_vectors
    for scaler in scalers:
        if scaler is None:
            continue
        fisher_vectors = scaler.transform(fisher_vectors)
    nr_descriptors_T = slice_data.nr_descriptors[:, np.newaxis]

    # Multiply by the number of descriptors.
    fisher_vectors = fisher_vectors * nr_descriptors_T / np.sum(nr_descriptors_T)
    slice_vw_counts = slice_data.counts * nr_descriptors_T / np.sum(nr_descriptors_T)

    #
    slice_vw_l2_norms = visual_word_l2_norm(fisher_vectors, visual_word_mask)
    slice_vw_scores = visual_word_scores(fisher_vectors, weights, bias, visual_word_mask)

    assert selector.integral

    slice_vw_l2_norms_no_integral = slice_vw_l2_norms
    slice_vw_scores_no_integral = slice_vw_scores

    slice_vw_counts = integral(slice_vw_counts)
    slice_vw_l2_norms = integral(slice_vw_l2_norms)
    pos_slice_vw_scores = integral(only_positive(slice_vw_scores))
    neg_slice_vw_scores = integral(only_negative(slice_vw_scores))
    nr_descriptors_T = integral(nr_descriptors_T)

    N = fisher_vectors.shape[0]

    def bounding_function(bounds, banned_intervals, weight_by_slice_length):

        union = bounds.get_union()
        inter = bounds.get_intersection()

        if inter[0] == inter[1] == union[0] == union[1] or union[0] >= union[1]:
            return - np.inf

        if inter[1] - inter[0] > max(deltas) / selector.chunk:
            return - np.inf

        if union[1] - union[0] < min(deltas) / selector.chunk:
            return - np.inf

        if inter[1] <= inter[0]:
            counts_union = eval_integral(slice_vw_counts, union)
            idxs_union = counts_union == 0
            if np.all(idxs_union):
                return + np.inf
            l2_norms_union = np.min(slice_vw_l2_norms_no_integral[union[0]:union[1]], axis=0)
            scores_union = np.max(slice_vw_scores_no_integral[union[0]:union[1]], axis=0)
            return ((
                np.ma.array(scores_union, mask=idxs_union) /
                np.ma.array(np.sqrt(counts_union), mask=idxs_union)).filled(0).sum() /
                np.sqrt((
                    np.ma.array(l2_norms_union, mask=idxs_union) /
                    np.ma.array(counts_union, mask=idxs_union)).filled(0).sum()))

        l2_norms_inter = eval_integral(slice_vw_l2_norms, inter)
        if np.all(l2_norms_inter == 0):
            return - np.inf

        if len(banned_intervals) > 0 and bounds_in_blacklist(bounds, banned_intervals):
            return - np.inf

        score_union = eval_integral(pos_slice_vw_scores, union)
        score_inter = eval_integral(neg_slice_vw_scores, inter)

        counts_union = eval_integral(slice_vw_counts, union)
        counts_inter = eval_integral(slice_vw_counts, inter)

        idxs_inter = counts_inter == 0
        idxs_union = counts_union == 0

        bound_sqrt_scores = np.sum(((
            np.ma.array(score_union, mask=idxs_union) +
            np.ma.array(score_inter, mask=idxs_inter)) /
            np.sqrt(np.ma.array(counts_inter, mask=idxs_inter))).filled(0))

        bound_approx_l2_norm = np.sum((
            np.ma.array(l2_norms_inter, mask=idxs_inter) /
            np.ma.array(counts_union, mask=idxs_union)).filled(0))

        max_slice_length = union[1] - union[0] if weight_by_slice_length else 1.
        return bound_sqrt_scores / np.sqrt(bound_approx_l2_norm) * max_slice_length

    banned_intervals = []
    results = []
    T = slice_data.end_frames[-1] - slice_data.begin_frames[0]

    ii = 0
    covered = 0
    heap = [(0, Bounds(np.array((0, 0)), np.array((N, N))))]

    while True:

        ii += 1

        score, idxs, heap = efficient_subwindow_search(
            lambda bounds: bounding_function(bounds, banned_intervals, rescore),
            heap, blacklist=banned_intervals, verbose=2)

        banned_intervals.append(idxs)
        results.append((
            slice_data.begin_frames[np.minimum(N - 1, idxs[0])],
            slice_data.begin_frames[idxs[1]] if idxs[1] < N else slice_data.end_frames[-1],
            score))

        covered += idxs[1] - idxs[0]

        if covered >= N or score == - np.inf:
            break

        if results[-1][0] >= results[-1][1]:
            pdb.set_trace()

    return results


@timer
def cy_approx_sliding_window_ess(
    slice_data, clf, deltas, selector, scalers, rescore, visual_word_mask,
    timings_file=None):

    import operator

    from ess import Bounds
    from ess import efficient_subwindow_search

    from utils_ess import ApproxNormsBoundingFunction
    from utils_ess import b_get_union
    from utils_ess import b_get_intersection
    from utils_ess import b_in_blacklist
    from utils_ess import b_init_bounds
    from utils_ess import b_init_interval
    from utils_ess import efficient_subwindow_search as cy_efficient_subwindow_search

    weights, bias = clf

    # Prepare sliced data.
    fisher_vectors = slice_data.fisher_vectors
    for scaler in scalers:
        if scaler is None:
            continue
        fisher_vectors = scaler.transform(fisher_vectors)
    nr_descriptors_T = slice_data.nr_descriptors[:, np.newaxis]

    # Multiply by the number of descriptors.
    fisher_vectors = fisher_vectors * nr_descriptors_T / np.sum(nr_descriptors_T)
    slice_vw_counts = slice_data.counts * nr_descriptors_T / np.sum(nr_descriptors_T)

    #
    slice_vw_l2_norms = visual_word_l2_norm(fisher_vectors, visual_word_mask)
    slice_vw_scores = visual_word_scores(fisher_vectors, weights, bias, visual_word_mask)

    assert selector.integral

    slice_vw_l2_norms_no_integral = slice_vw_l2_norms
    slice_vw_scores_no_integral = slice_vw_scores

    slice_vw_counts = integral(slice_vw_counts)
    slice_vw_l2_norms = integral(slice_vw_l2_norms)
    pos_slice_vw_scores = integral(only_positive(slice_vw_scores))
    neg_slice_vw_scores = integral(only_negative(slice_vw_scores))
    nr_descriptors_T = integral(nr_descriptors_T)

    N = fisher_vectors.shape[0]

    banned_intervals = []
    results = []
    T = slice_data.end_frames[-1] - slice_data.begin_frames[0]

    ii = 0
    covered = 0
    heap = [(0, b_init_bounds((0, 0), (N, N)))]

    bounding_function = ApproxNormsBoundingFunction(
        slice_vw_scores_no_integral, slice_vw_l2_norms_no_integral,
        pos_slice_vw_scores, neg_slice_vw_scores, slice_vw_counts,
        slice_vw_l2_norms, min_window=min(deltas) / selector.chunk,
        max_window=max(deltas) / selector.chunk)

    while True:

        ii += 1

        bounding_function.set_banned_intervals(banned_intervals)
        score, idxs, heap = cy_efficient_subwindow_search(
            bounding_function, heap, blacklist=banned_intervals, verbose=0)

        if covered >= N or score == - np.inf:
            break

        banned_intervals.append(b_init_interval(idxs))
        results.append((
            slice_data.begin_frames[np.minimum(N - 1, idxs[0])],
            slice_data.begin_frames[idxs[1]] if idxs[1] < N else slice_data.end_frames[-1],
            score))

        if results[-1][0] >= results[-1][1]:
            pdb.set_trace()

        covered += idxs[1] - idxs[0]

    return results


def save_results(dataset, class_idx, adrien_results, deltas=None):
    res_fn = os.path.join(dataset.SSTATS_DIR, "class_%d_%d_results.pickle")
    get_duration = lambda xx: xx[0].ef - xx[0].bf
    for delta, res in groupby(sorted(adrien_results, key=get_duration), key=get_duration):
        if deltas is not None and delta not in deltas:
            continue
        with open(res_fn % (class_idx, delta), 'w') as ff:
            cPickle.dump(list(res), ff)


def evaluation(
    algo_type, src_cfg, class_idx, stride, deltas, no_integral, containing,
    do_save_results=False, verbose=0):

    dataset = Dataset(CFG[src_cfg]['dataset_name'], **CFG[src_cfg]['dataset_params'])
    D, K = 64, dataset.VOC_SIZE
    visual_word_mask = build_visual_word_mask(D, K)

    MOVIE = 'cac.mpg'
    SAMPID = '%s-frames-%d-%d'
    ALGO_PARAMS = {
        'none': {
            'train_params': {
                'l2_norm_type': 'none',
                'empirical_standardizations': [False, False],
                'sqrt_type': 'none'
            },
            'sliding_window': exact_sliding_window,
            'sliding_window_params': {
                'l2_norm_type': 'none',
                'sqrt_type': 'none'
            },
        },
        'exact': {
            'train_params': {
                'l2_norm_type': 'exact',
                'empirical_standardizations': [False, True],
                'sqrt_type': 'exact'
            },
            'sliding_window': exact_sliding_window,
            'sliding_window_params': {
                'l2_norm_type': 'exact',
                'sqrt_type': 'exact'
            },
        },
        'approx': {
            'train_params': {
                'l2_norm_type': 'approx',
                'empirical_standardizations': [False, True],
                'sqrt_type': 'approx'
            },
            'sliding_window': approx_sliding_window,
            'sliding_window_params': {
                'visual_word_mask': visual_word_mask
            },
        },
        'approx_ess': {
            'train_params': {
                'l2_norm_type': 'approx',
                'empirical_standardizations': [False, True],
                'sqrt_type': 'approx'
            },
            'sliding_window': approx_sliding_window_ess,
            'sliding_window_params': {
                'visual_word_mask': visual_word_mask,
                'rescore': rescore,
            },
        },
        'cy_approx_ess': {
            'train_params': {
                'l2_norm_type': 'approx',
                'empirical_standardizations': [False, True],
                'sqrt_type': 'approx'
            },
            'sliding_window': cy_approx_sliding_window_ess,
            'sliding_window_params': {
                'visual_word_mask': visual_word_mask,
                'rescore': rescore,
                'timings_file': timings_file,
            },
        },
    }

    chunk_size = CFG[src_cfg]['chunk_size']
    base_chunk_size = mgcd(stride, *deltas)
    tr_nr_agg = base_chunk_size / chunk_size

    # For the old C&C features I have one FV for the entire sample for the
    # train data, so I cannot aggregate.
    if src_cfg == 'cc':
        tr_nr_agg = 1

    tr_outfile = '/scratch2/clear/oneata/tmp/joblib/%s_cls%d_train.dat' % (src_cfg, class_idx)
    tr_video_data, tr_video_labels, tr_stds = load_normalized_tr_data(
        dataset, tr_nr_agg, tr_outfile=tr_outfile, verbose=verbose,
        analytical_fim=True, **ALGO_PARAMS[algo_type]['train_params'])

    # Sub-sample data.
    no_tuple_labels = np.array([ll[0] for ll in tr_video_labels])
    idxs = (no_tuple_labels == class_idx) | (no_tuple_labels == NULL_CLASS_IDX)
    binary_labels = (no_tuple_labels[idxs] == class_idx) * 2 - 1
    class_tr_video_data = tr_video_data[idxs]
    tr_kernel = np.dot(class_tr_video_data, class_tr_video_data.T)

    eval = Evaluation(CFG[src_cfg]['eval_name'], **CFG[src_cfg]['eval_params'])
    eval.fit(tr_kernel, binary_labels)
    clf = compute_weights(eval.get_classifier(), class_tr_video_data)

    results = []
    class_name = dataset.IDX2CLS[class_idx]
    non_overlapping_selector = NonOverlappingSelector(base_chunk_size / chunk_size)
    overlapping_selector = OverlappingSelector(
        base_chunk_size, stride, containing,
        integral=(not no_integral))
    for part in xrange(len(dataset.CLASS_LIMITS[MOVIE][class_name])):
        te_outfile = (
            '/scratch2/clear/oneata/tmp/joblib/%s_cls%d_part%d_test.dat' %
            (src_cfg, class_idx, part))
        te_slice_data = SliceData(*load_data_delta_0(
            dataset, MOVIE, part, class_idx, delta_0=chunk_size,
            outfile=te_outfile))

        # Aggregate data into non-overlapping chunks of size `base_chunk_size`.
        N = te_slice_data.fisher_vectors.shape[0]
        agg_slice_data = aggregate(
            te_slice_data,
            non_overlapping_selector.get_mask(N),
            non_overlapping_selector.get_frame_idxs(N))

        results += ALGO_PARAMS[algo_type]['sliding_window'](
            agg_slice_data, clf, deltas, overlapping_selector, tr_stds,
            **ALGO_PARAMS[algo_type]['sliding_window_params'])

    gt_path = os.path.join(dataset.FL_DIR, 'keyframes_test_%s.list' % class_name)
    adrien_results = [
        (SampID(SAMPID % (MOVIE, bf, ef)), score) for bf, ef, score in results]
    ap = get_det_ap(adrien_results, gt_path, 'OV20', 'OV20')

    if do_save_results:
        save_results(dataset, class_idx, adrien_results, deltas)

    print "%10s %.2f" % (class_name, 100 * ap)


def main():
    parser = argparse.ArgumentParser(
        description="Approximating the normalizations for the detection task.")

    parser.add_argument(
        '-d', '--dataset', required=True, choices=('cc', 'cc.stab', 'cc.no_stab'),
        help="which dataset.")
    parser.add_argument(
        '-a', '--algorithm', required=True,
        choices=('none', 'approx', 'approx_ess', 'cy_approx_ess', 'exact'),
        help="specifies the type of normalizations.")
    parser.add_argument(
        '--rescore', action='store_true', default=False,
        help="rescores the slices according to their length.")
    parser.add_argument(
        '--containing', action='store_true', default=False,
        help=("considers the FVs corresponding to the dense trajectories that"
              "are completely contained in a given window; otherwise "
              "considers the FVs corresponding to the dense trajectories that "
              "start in the given window."))
    parser.add_argument(
        '--class_idx', default=1, type=int,
        help="index of the class to evaluate.")
    parser.add_argument(
        '--no_integral', action='store_true', default=False,
        help="does not use integral quantities for aggregation.")
    parser.add_argument('-S', '--stride', type=int, help="window displacement step size.")
    parser.add_argument('-D', '--delta', type=int, help="base slice length.")
    parser.add_argument('--begin', type=int, help="smallest slice length.")
    parser.add_argument('--end', type=int, help="largest slice length.")
    parser.add_argument(
        '--save_results', action='store_true', default=False,
        help="dumps results to disk in seperate files for each delta.")
    parser.add_argument(
        '-v', '--verbose', action='count', help="verbosity level.")

    args = parser.parse_args()
    deltas = range(args.begin, args.end + args.delta, args.delta)

    evaluation(
        args.algorithm, args.dataset, args.class_idx, args.stride, deltas,
        no_integral=args.no_integral, containing=args.containing,
        do_save_results=args.save_results, verbose=args.verbose)


if __name__ == '__main__':
    main()

