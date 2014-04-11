
import argparse
import functools
import ipdb as pdb
import numpy as np
import os

from sklearn.preprocessing import StandardScaler

from dataset import Dataset

from fisher_vectors.evaluation import Evaluation
from fisher_vectors.model.utils import power_normalize
from fisher_vectors.model.utils import compute_L2_normalization as compute_exact_l2_normalization

# Local imports.
from evaluate import print_scores

from load_data import CACHE_PATH
from load_data import CFG
from load_data import approximate_signed_sqrt
from load_data import load_video_data

from ssqrt_l2_approx import compute_approx_l2_normalization as compute_approx_l2_normalization_
from ssqrt_l2_approx import load_corrected_norms


PI_DERIVATIVES = False 
SQRT_NR_DESCS = False

# Caching path.
OUTFILE = os.path.join(
    CACHE_PATH, "%s_%s_afim_%s_pi_%s_sqrt_nr_descs_%s_enc%s_spm%s.dat" % (
        "%s", "%s", "%s", PI_DERIVATIVES, SQRT_NR_DESCS, "%s", "%s"))

def compute_kernels(loader, scaler_1, square_root, scaler_2, compute_l2_norm):

    tr_data, tr_counts, tr_labels = loader('train')

    tr_data = scaler_1.fit_transform(tr_data)
    tr_data = square_root(tr_data, counts=tr_counts)
    tr_data = scaler_2.fit_transform(tr_data)
    tr_Z = compute_l2_norm(
        tr_data, split='train', scalers=[scaler_1, scaler_2], counts=tr_counts)

    te_data, te_counts, te_labels = loader('test')

    te_data = scaler_1.transform(te_data)
    te_data = square_root(te_data, counts=te_counts)
    te_data = scaler_2.transform(te_data)
    te_Z = compute_l2_norm(
        te_data, split='test', scalers=[scaler_1, scaler_2], counts=te_counts)

    tr_kernel = np.dot(tr_data, tr_data.T)
    te_kernel = np.dot(te_data, tr_data.T)

    return tr_kernel, tr_labels, tr_Z, te_kernel, te_labels, te_Z


def load_kernels_l2_norm_enc(counter, loader, normalizations, spms, encodings):

    N_tr = counter('train')
    N_te = counter('test')

    tr_kernel = np.zeros((N_tr, N_tr), dtype=np.float32)
    te_kernel = np.zeros((N_te, N_tr), dtype=np.float32)

    for spm in spms:
        for bin in xrange(np.prod(spm)):

            tr_kernel_enc = np.zeros((N_tr, N_tr), dtype=np.float32)
            te_kernel_enc = np.zeros((N_te, N_tr), dtype=np.float32)

            tr_Z = np.zeros(N_tr, dtype=np.float32)
            te_Z = np.zeros(N_te, dtype=np.float32)

            for encoding in encodings:

                print spm, bin, encoding

                current_loader = functools.partial(
                    loader, spm=spm, encoding=encoding, bin=bin)
                normalizations['compute_l2_norm'] = functools.partial(
                    normalizations['compute_l2_norm'],
                    spm=spm, encoding=encoding, bin=bin)

                (tr_kernel_, tr_labels, tr_Z_,
                 te_kernel_, te_labels, te_Z_) = compute_kernels(
                     current_loader, **normalizations)

                tr_kernel_enc += tr_kernel_
                te_kernel_enc += te_kernel_
                tr_Z += tr_Z_
                te_Z += te_Z_

            # Normalize at encoding level.
            tr_kernel_enc /= np.sqrt(tr_Z[:, np.newaxis] * tr_Z[np.newaxis])
            te_kernel_enc /= np.sqrt(te_Z[:, np.newaxis] * tr_Z[np.newaxis])

            tr_kernel_enc[np.isinf(tr_kernel_enc)] = 0
            te_kernel_enc[np.isinf(te_kernel_enc)] = 0

            tr_kernel += tr_kernel_enc
            te_kernel += te_kernel_enc

    return tr_kernel, tr_labels, te_kernel, te_labels


def load_kernels_all(
    src_cfg, e_std_1, sqrt, e_std_2, l2_norm, afim,
    nr_slices_to_aggregate=None, verbose=0):

    dataset = Dataset(
        CFG[src_cfg]['dataset_name'],
        **CFG[src_cfg]['dataset_params'])

    spms = CFG[src_cfg].get('spms', [(1, -1, -1)])  # FIXME Hack.
    encodings = CFG[src_cfg].get('encodings', ['fv'])

    class DummyScaler(object):
        def fit_transform(self, X):
            return X
        def transform(self, X):
            return X

    def loader(split, spm=None, encoding=None, bin=None):
        """Loads sufficient statistics."""

        samples, _ = dataset.get_data(split)
        N = len(set(map(str, samples)))

        outfile = OUTFILE % (
            src_cfg, split,
            '_' + encoding,
            ''.join(map(str, spm)))

        data, counts, labels = load_video_data(
            dataset, samples, outfile=outfile, analytical_fim=ANALYTICAL_FIM,
            pi_derivatives=PI_DERIVATIVES, sqrt_nr_descs=SQRT_NR_DESCS,
            encoding=encoding, spm=spm, verbose=verbose)
        N_bins = np.prod(spm)

        _, D_data = data.shape
        _, D_counts = counts.shape

        def get_slice(D):
            return slice(D / N_bins * bin, D / N_bins * (bin + 1))

        I_data = get_slice(D_data)
        I_counts = get_slice(D_counts)

        return data[:, I_data], counts[:, I_counts], labels

    def sample_counter(split):
        samples, _ = dataset.get_data(split)
        return len(set(map(str, samples)))

    def compute_approx_l2_normalization(
        data, split, scalers, counts, spm=None, encoding=None, bin=None):

        if sqrt == 'none':
            counts = np.ones(counts.shape)

        # Prepare cached filename.
        outfile = OUTFILE % (
            src_cfg, split,
            '_' + encoding,
            ''.join(map(str, spm)))
        suffix = "norm_slices_%d_scalers_%s_%s" % (
            nr_slices_to_aggregate, e_std_1, e_std_2)
        norm_filename = '.'.join((outfile % "train", suffix))

        samples, _ = dataset.get_data(split)
        tr_video_l2_norms = load_corrected_norms(
            dataset, samples, nr_slices_to_aggregate,
            analytical_fim=ANALYTICAL_FIM, scalers=scalers,
            verbose=verbose, outfile=norm_filename)[0]

        return compute_approx_l2_normalization_(data, tr_video_l2_norms, counts)

    SQUARE_ROOT_TABLE = {
        'exact': lambda data, **kwargs: power_normalize(data, 0.5),
        'approx': lambda data, counts: approximate_signed_sqrt(
            data, counts, pi_derivatives=PI_DERIVATIVES),
        'none': lambda data, **kwargs: data,
    }

    COMPUTE_L2_NORM_TABLE = {
        'exact': lambda data, **kwargs: compute_exact_l2_normalization(data),
        'approx': compute_approx_l2_normalization,
        'none': lambda data, **kwargs: np.ones(data.shape[0], dtype=np.float32),
    }

    def get_scaler(bool):
        return StandardScaler(with_mean=False) if bool else DummyScaler()

    normalizations = {
        'scaler_1'        : get_scaler(e_std_1),
        'square_root'     : SQUARE_ROOT_TABLE[sqrt],
        'scaler_2'        : get_scaler(e_std_2),
        'compute_l2_norm' : COMPUTE_L2_NORM_TABLE[l2_norm],
    }

    return load_kernels_l2_norm_enc(
        sample_counter, loader, normalizations, spms, encodings)


def evaluate(src_cfg, tr_kernel, tr_labels, te_kernel, te_labels):

    eval = Evaluation(CFG[src_cfg]['eval_name'], **CFG[src_cfg]['eval_params'])
    eval.fit(tr_kernel, tr_labels)
    scores = eval.score(te_kernel, te_labels)

    if CFG[src_cfg]['metric'] == 'average_precision':
        print_scores(scores)

    if CFG[src_cfg]['metric'] == 'average_precision':
        print "%.2f" % np.mean(scores)
    elif CFG[src_cfg]['metric'] == 'accuracy':
        print "%.2f" % scores


def main():

    parser = argparse.ArgumentParser(
        description="Experiments for the CVPR'14 camera ready version.")

    parser.add_argument(
        '-d', '--src_cfg', required=True, choices=CFG.keys(),
        help="which dataset (use `dummy` for debugging purposes).")
    parser.add_argument(
        '--e_std_1', default=False, action='store_true',
        help=("applies empirical standardization at the first stage, before "
              "square rooting."))
    parser.add_argument(
        '--sqrt', choices=('exact', 'approx', 'none'),
        help="signed square rooting normalization.")
    parser.add_argument(
        '--e_std_2', default=False, action='store_true',
        help=("applies empirical standardization at the second stage, after "
              "square rooting."))
    parser.add_argument(
        '--l2_norm', choices=('exact', 'approx', 'none'),
        help="L2 normalization.")
    parser.add_argument(
        '-n', '--nr_slices_to_aggregate', type=int, default=1,
        help="aggregates consecutive FVs.")
    parser.add_argument(
        '--afim', default=False, action='store_true',
        help=("uses FVs that are standardized with the analytical Fisher "
              "information matrix."))
    parser.add_argument(
        '-v', '--verbose', action='count', help="verbosity level.")

    args = parser.parse_args()

    if args.l2_norm == 'approx' and args.nr_slices_to_aggregate is None:
        parser.error(
            "Approximate L2 normalization `--l2_norm approx` requires "
            "`--nr_slices_to_aggregate` argument.")

    evaluate(args.src_cfg, *load_kernels_all(**vars(args)))


if __name__ == '__main__':
    main()

