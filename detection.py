import argparse
from collections import namedtuple
import cPickle
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
from fisher_vectors.model.utils import L2_normalize

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
# [ ] Check if frames are contiguous. If not treat them specifically.
# [ ] Build mask with Cython.
# [ ] Mask with 1, -1 and integral quantities.


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


def build_sliding_window_mask(N, nn):
    """Builds mask to aggregate a vectors [x_1, x_2, ..., x_N] into
    [x_1 + ... + x_n, x_2 + ... + x_{n+1}, ...].

    """
    mask = np.zeros((N - nn + 1, N))
    idxs = [
        map(int, np.hstack(ii))
        for ii in zip(
            *[[np.ones(nn) * ii, np.arange(nn) + ii]
              for ii in xrange(N - nn + 1)])]
    mask[idxs] = 1
    return sparse.csr_matrix(mask)


def build_integral_sliding_window_mask(N, nn):
    """Builds mask for efficient integral sliding window."""
    H, W = N - nn + 1, N + 1
    mask = np.zeros((H, W))

    row_idxs = range(H)
    neg_idxs = range(H)
    pos_idxs = [nn + i for i in range(H)]

    mask[row_idxs, neg_idxs] = -1
    mask[row_idxs, pos_idxs] = +1

    return sparse.csr_matrix(mask)


def builg_aggregation_mask_and_limits(
    agg_type, delta, stride, n_slices, integral=False):
    
    if agg_type == 'overlap':

        n_mask = delta / stride
        n_frames = delta / stride

        mask_builder = build_integral_sliding_window_mask if integral else build_sliding_window_mask

        begin_frame_idxs = slice(0, n_slices - n_frames + 1)
        end_frame_idxs = slice(n_frames - 1, n_slices)

    elif agg_type == 'overlap_containing':
        # Consider only tracks that are fully contained in the window.
        assert (delta - TRACK_LEN) % stride == 0

        n_mask = (delta - TRACK_LEN) / stride
        n_frames = delta / stride

        mask_builder = build_integral_sliding_window_mask if integral else build_sliding_window_mask

        # TODO Fix this. The last frame doesn't have the proper length.
        begin_frame_idxs = slice(0, n_slices - n_mask + 1)
        end_frame_idxs = np.minimum(
            n_slices - 1,
            range(n_frames - 1, n_slices + n_frames - n_mask))

    elif agg_type == 'no_overlap':

        assert integral == False

        n_mask = delta / stride
        n_frames = delta / stride

        mask_builder = build_slice_agg_mask

        begin_frame_idxs = slice(0, n_slices, n_frames)
        end_frame_idxs = np.minimum(
            n_slices - 1,
            range(n_frames - 1, n_slices + n_frames - 1, n_frames))

    else:
        assert False, "Unknown aggregation type %s." % agg_type

    return mask_builder(n_slices, n_mask), begin_frame_idxs, end_frame_idxs


def aggregate(slice_data, delta, stride, agg_type):
    """Aggregates slices of data into `N` slices."""

    N = slice_data.fisher_vectors.shape[0]

    # Build mask for aggregation.
    sparse_mask, begin_frame_idxs, end_frame_idxs = builg_aggregation_mask_and_limits(
        agg_type, delta, stride, N)

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
    agg_begin_frames = slice_data.begin_frames[begin_frame_idxs]
    agg_end_frames = slice_data.end_frames[end_frame_idxs]

    assert len(agg_fisher_vectors) == len(agg_begin_frames) == len(agg_end_frames)

    # Return new slice_data structure.
    return SliceData(
        agg_fisher_vectors, agg_counts, agg_nr_descs, agg_begin_frames,
        agg_end_frames)


@timer
def exact_sliding_window(
    slice_data, clf, scalers, stride, deltas, containing, sqrt_type='',
    l2_norm_type=''):

    results = []
    weights, bias = clf

    agg_type = 'overlap' if not containing else 'overlap_containing'

    for delta in deltas:

        # Aggregate data into bigger slices.
        nr_agg = delta / stride
        agg_data = aggregate(slice_data, delta, stride, agg_type)
        agg_fisher_vectors = agg_data.fisher_vectors

        # Normalize aggregated data.
        if scalers[0] is not None:
            agg_fisher_vectors = scalers[0].transform(agg_fisher_vectors)
        if sqrt_type != 'none':
            agg_fisher_vectors = power_normalize(agg_fisher_vectors, 0.5)
        if scalers[1] is not None:
            agg_fisher_vectors = scalers[1].transform(agg_fisher_vectors)
        if l2_norm_type != 'none':
            agg_fisher_vectors = L2_normalize(agg_fisher_vectors)

        nan_idxs = np.isnan(agg_fisher_vectors)

        # Predict with the linear classifier.
        scores = predict(agg_fisher_vectors, weights, bias)
        nan_idxs = np.isnan(scores)
        results += zip(
            agg_data.begin_frames[~nan_idxs],
            agg_data.end_frames[~nan_idxs],
            scores[~nan_idxs])

    return results


@timer
def approx_sliding_window(
    slice_data, clf, scalers, stride, deltas, visual_word_mask,
    containing, use_integral_values=True):

    def integral(X):
        return np.vstack((
            np.zeros((1, X.shape[1])),
            np.cumsum(X, axis=0)))

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

    if use_integral_values:
        slice_vw_counts = integral(slice_vw_counts)
        slice_vw_l2_norms = integral(slice_vw_l2_norms)
        slice_vw_scores = integral(slice_vw_scores)
        nr_descriptors_T = integral(nr_descriptors_T)
        build_mask = build_integral_sliding_window_mask
    else:
        build_mask = build_sliding_window_mask

    N = fisher_vectors.shape[0]
    agg_type = 'overlap' if not containing else 'overlap_containing'

    for delta in deltas:

        # Build mask.
        mask, begin_frame_idxs, end_frame_idxs = builg_aggregation_mask_and_limits(
            agg_type, delta, stride, N, integral=use_integral_values)

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


def save_results(dataset, class_idx, adrien_results, deltas=None):
    res_fn = os.path.join(dataset.SSTATS_DIR, "class_%d_%d_results.pickle")
    get_duration = lambda xx: xx[0].ef - xx[0].bf
    for delta, res in groupby(sorted(adrien_results, key=get_duration), key=get_duration):
        if deltas is not None and delta not in deltas:
            continue
        with open(res_fn % (class_idx, delta), 'w') as ff:
            cPickle.dump(list(res), ff)


def evaluation(
    algo_type, src_cfg, class_idx, stride, deltas, containing,
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
    }

    chunk_size = CFG[src_cfg]['chunk_size']
    tr_nr_agg = stride / chunk_size

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
    for part in xrange(len(dataset.CLASS_LIMITS[MOVIE][class_name])):
        te_outfile = (
            '/scratch2/clear/oneata/tmp/joblib/%s_cls%d_part%d_test.dat' %
            (src_cfg, class_idx, part))
        te_slice_data = SliceData(*load_data_delta_0(
            dataset, MOVIE, part, class_idx, delta_0=chunk_size,
            outfile=te_outfile))
        agg_slice_data = aggregate(te_slice_data, stride, chunk_size, 'no_overlap')

        results += ALGO_PARAMS[algo_type]['sliding_window'](
            agg_slice_data, clf, tr_stds, stride, deltas, containing=containing,
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
        '-d', '--dataset', required=True, choices=('cc', 'cc.stab'),
        help="which dataset.")
    parser.add_argument(
        '-a', '--algorithm', required=True, choices=('none', 'approx', 'exact'),
        help="specifies the type of normalizations.")
    parser.add_argument(
        '--containing', action='store_true', default=False,
        help=("considers the FVs corresponding to the dense trajectories that"
              "are completely contained in a given window; otherwise "
              "considers the FVs corresponding to the dense trajectories that "
              "start in the given window."))
    parser.add_argument(
        '--class_idx', default=1, type=int,
        help="index of the class to evaluate.")
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

    # Some checking.
    assert args.stride % CFG[args.dataset]['chunk_size'] == 0
    for delta in deltas:
        assert delta % args.stride == 0

    evaluation(
        args.algorithm, args.dataset, args.class_idx, args.stride, deltas,
        containing=args.containing, do_save_results=args.save_results,
        verbose=args.verbose)


if __name__ == '__main__':
    main()

