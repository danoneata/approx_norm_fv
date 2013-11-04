import argparse
from ipdb import set_trace
import numpy as np
import os
import pdb

from dataset import Dataset
from fisher_vectors.evaluation import Evaluation

from load_data import CACHE_PATH
from load_data import CFG
from load_data import load_kernels


def print_scores(scores):
    scores = [score for score in scores]
    print "mAP |",
    print " ".join(["%.2f" % score for score in scores]),
    print "| %.3f" % np.mean(scores)


def evaluate(
    src_cfg, tr_norms, te_norms, analytical_fim, pi_derivatives, sqrt_nr_descs,
    do_plot=False, verbose=0):

    outfile = os.path.join(
        CACHE_PATH, "%s_%s_afim_%s_pi_%s_sqrt_nr_descs_%s.dat" % (
            src_cfg, "%s", analytical_fim, pi_derivatives, sqrt_nr_descs))

    dataset = Dataset(CFG[src_cfg]['dataset_name'], **CFG[src_cfg]['dataset_params'])
    (tr_kernel, tr_labels,
     te_kernel, te_labels) = load_kernels(
         dataset, tr_norms=tr_norms, te_norms=te_norms,
         analytical_fim=analytical_fim, pi_derivatives=pi_derivatives,
         sqrt_nr_descs=sqrt_nr_descs, outfile=outfile, do_plot=do_plot,
         verbose=verbose)

    eval = Evaluation(CFG[src_cfg]['eval_name'], **CFG[src_cfg]['eval_params'])
    eval.fit(tr_kernel, tr_labels)
    scores = eval.score(te_kernel, te_labels)

    if verbose > 0:
        print 'Train normalizations:', ', '.join(map(str, tr_norms))
        print 'Test normalizations:', ', '.join(map(str, te_norms))

        if CFG[src_cfg]['metric'] == 'average_precision':
            print_scores(scores)

    if CFG[src_cfg]['metric'] == 'average_precision':
        print "%.2f" % np.mean(scores)
    elif CFG[src_cfg]['metric'] == 'accuracy':
        print "%.2f" % scores


def main():
    parser = argparse.ArgumentParser(
        description="Evaluating the normalization approximations.")

    valid_norms = ['std', 'sqrt', 'L2', 'sqrt_cnt']

    parser.add_argument(
        '-d', '--dataset', choices=CFG.keys(),
        help="name of the dataset.")
    parser.add_argument(
        '--tr_norms', choices=valid_norms, nargs='+',
        default = [], help="normalizations used for training.")
    parser.add_argument(
        '--te_norms', choices=valid_norms, nargs='+',
        help=("normalizations used for testing "
             "(by default, use the same as for training)."))
    parser.add_argument(
        '-afim', '--analytical_fim', action='store_true',
        help=("normalizes by the analytical form of the "
              "Fisher information matrix (FIM)."))
    parser.add_argument(
        '-dpi', '--pi_derivatives', action='store_true',
        help=("uses the derivative wrt mixing weights (default uses only "
              "derivatives wrt means and variances)."))
    parser.add_argument(
        '-sqrtT', '--sqrt_nr_descs', action='store_true',
        help="averages patch descriptors by sqrt(T) (default averages by T).")
    parser.add_argument(
        '--plot', action='store_true',
        help="draws plots with FVs before and after normalization.")
    parser.add_argument(
        '-v', '--verbose', action='count', help="verbosity level.")

    args = parser.parse_args()

    if args.te_norms is None:
        args.te_norms = args.tr_norms

    evaluate(
        args.dataset, args.tr_norms, args.te_norms, args.analytical_fim,
        args.pi_derivatives, args.sqrt_nr_descs, do_plot=args.plot,
        verbose=args.verbose)


if __name__ == '__main__':
    main()

