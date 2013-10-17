import argparse
from collections import namedtuple
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


NULL_CLASS_IDX = 0
SliceData = namedtuple(
    'SliceData', ['fisher_vectors', 'counts', 'nr_descriptors', 'begin_frames',
                  'end_frames'])


def in_intervals(start, end, intervals):
    # Check if the interval given by (start, end) is included in intervals.
    return np.any([low <= start and end <= high for low, high in intervals])


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
    dataset, movie, class_idx, outfile=None, delta_0=30, verbose=0):
    """ Loads Fisher vectors for the test data for detection datasets. """

    D = 64
    K = dataset.VOC_SIZE
    class_limits = dataset.CLASS_LIMITS[movie][dataset.IDX2CLS[class_idx]]

    nr_frames = sum(ef - bf for bf, ef in class_limits) + 1
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

        if not in_intervals(sample.bf, sample.ef, class_limits):
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


def aggregate(slice_data, nn, agg_type):
    """Aggregates slices of data into `N` slices."""

    fisher_vectors = slice_data.fisher_vectors
    N = fisher_vectors.shape[0] 

    P = {
        'overlap': {
            'mask_builder': build_sliding_window_mask,
            'begin_frame_idxs': slice(0, N - nn + 1),
            'end_frame_idxs': slice(nn - 1, N),
        },
        'no_overlap': {
            'mask_builder': build_slice_agg_mask,
            'begin_frame_idxs': slice(0, N, nn),
            'end_frame_idxs': np.minimum(N - 1, range(nn - 1, N + nn - 1, nn)),
        },
    }

    # Build mask for aggregation.
    sparse_mask = P[agg_type]['mask_builder'](N, nn)

    # Scale by number of descriptors and sum.
    agg_fisher_vectors = sum_and_scale_by(
        fisher_vectors, slice_data.nr_descriptors, mask=sparse_mask)
    agg_fisher_vectors[np.isnan(agg_fisher_vectors)] = 0

    # Sum counts.
    agg_counts = sum_by(slice_data.counts, mask=sparse_mask)
    agg_counts[np.isnan(agg_counts)] = 0

    # Sum number of descriptors.
    agg_nr_descs = sum_by(slice_data.nr_descriptors, mask=sparse_mask)

    # Correct begin frames and end frames.
    agg_begin_frames = slice_data.begin_frames[P[agg_type]['begin_frame_idxs']]
    agg_end_frames = slice_data.end_frames[P[agg_type]['end_frame_idxs']] 

    assert len(agg_fisher_vectors) == len(agg_begin_frames) == len(agg_end_frames)

    # Return new slice_data structure.
    return SliceData(
        agg_fisher_vectors, agg_counts, agg_nr_descs, agg_begin_frames,
        agg_end_frames)


@timer
def exact_sliding_window(
    slice_data, clf, scalers, stride, deltas, sqrt_type='', l2_norm_type=''):

    results = [] 
    weights, bias = clf

    for delta in deltas:

        # Aggregate data into bigger slices.
        nr_agg = delta / stride
        agg_data = aggregate(slice_data, nr_agg, 'overlap')

        # Normalize aggregated data.
        if scalers[0] is not None:
            agg_fisher_vectors = scalers[0].transform(agg_fisher_vectors)
        agg_fisher_vectors = agg_data.fisher_vectors
        if sqrt_type != 'none':
            agg_fisher_vectors = power_normalize(agg_fisher_vectors, 0.5)
        if scalers[1] is not None:
            agg_fisher_vectors = scalers[1].transform(agg_fisher_vectors)
        if l2_norm_type != 'none':
            agg_fisher_vectors = L2_normalize(agg_fisher_vectors)

        nan_idxs = np.isnan(agg_fisher_vectors)
        agg_fisher_vectors[nan_idxs] = 0

        # Predict with the linear classifier.
        scores = predict(agg_fisher_vectors, weights, bias)
        results += zip(agg_data.begin_frames, agg_data.end_frames, scores)

    return results


@timer
def approx_sliding_window(
    slice_data, clf, scalers, stride, deltas, visual_word_mask,
    use_integral_values=True):

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

    for delta in deltas:

        # Build mask.
        nn = delta / stride
        mask = build_mask(N, nn)

        # Approximated predictions.
        scores = approximate_video_scores(
            slice_vw_scores, slice_vw_counts, slice_vw_l2_norms, nr_descriptors_T, mask)
        agg_begin_frames = slice_data.begin_frames[:N - nn + 1]
        agg_end_frames = slice_data.end_frames[nn - 1:] 

        assert len(scores) == len(agg_begin_frames) == len(agg_end_frames)

        scores[np.isnan(scores)] = -10000
        results += zip(agg_begin_frames, agg_end_frames, scores)

    return results


def evaluation(algo_type, src_cfg, class_idx, stride, deltas, verbose=0):

    dataset = Dataset(CFG[src_cfg]['dataset_name'], **CFG[src_cfg]['dataset_params'])
    D, K = 64, dataset.VOC_SIZE
    visual_word_mask = build_visual_word_mask(D, K)

    MOVIE = 'cac.mpg'
    SAMPID = '%s-frames-%d-%d'
    ALGO_PARAMS = {
        'none': {
            'train_params': {'l2_norm_type': 'none', 'empirical_standardizations': [False, False], 'sqrt_type': 'none'},
            'sliding_window': exact_sliding_window,
            'sliding_window_params': {'l2_norm_type': 'none', 'sqrt_type': 'none'},
        },
        'exact': {
            'train_params': {'l2_norm_type': 'exact', 'empirical_standardizations': [False, True], 'sqrt_type': 'exact'},
            'sliding_window': exact_sliding_window,
            'sliding_window_params': {'l2_norm_type': 'exact', 'sqrt_type': 'exact'},
        },
        'approx': {
            'train_params': {'l2_norm_type': 'approx', 'empirical_standardizations': [False, True], 'sqrt_type': 'approx'},
            'sliding_window': approx_sliding_window,
            'sliding_window_params': {'visual_word_mask': visual_word_mask},
        },
    }

    nr_agg = stride / CFG[src_cfg]['chunk_size']
    tr_nr_agg = te_nr_agg = nr_agg

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

    te_outfile = '/scratch2/clear/oneata/tmp/joblib/%s_cls%d_test.dat' % (src_cfg, class_idx)
    te_slice_data = SliceData(*load_data_delta_0(
        dataset, MOVIE, class_idx, delta_0=CFG[src_cfg]['chunk_size'],
        outfile=te_outfile))
    agg_slice_data = aggregate(te_slice_data, te_nr_agg, 'no_overlap')

    clf = compute_weights(eval.get_classifier(), class_tr_video_data)
    results = ALGO_PARAMS[algo_type]['sliding_window'](
        agg_slice_data, clf, tr_stds, stride, deltas,
        **ALGO_PARAMS[algo_type]['sliding_window_params'])
    
    class_name = dataset.IDX2CLS[class_idx]
    gt_path = os.path.join(dataset.FL_DIR, 'keyframes_test_%s.list' % class_name)
    adrien_results = [
        (SampID(SAMPID % (MOVIE, bf, ef)), score) for bf, ef, score in results]
    ap = get_det_ap(adrien_results, gt_path, 'OV20', 'OV20')

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
        '--class_idx', default=1, type=int,
        help="index of the class to evaluate.")
    parser.add_argument('-S', '--stride', type=int, help="window displacement step size.")
    parser.add_argument('-D', '--delta', type=int, help="base slice length.")
    parser.add_argument('--begin', type=int, help="smallest slice length.")
    parser.add_argument('--end', type=int, help="largest slice length.")
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
        verbose=args.verbose)


if __name__ == '__main__':
    main()

