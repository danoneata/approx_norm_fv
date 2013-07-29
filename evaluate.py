import argparse
from ipdb import set_trace
import numpy as np

from dataset import Dataset
from fisher_vectors.evaluation import Evaluation

from load_data import load_kernels


def print_scores(scores):
    scores = [score * 100 for score in scores]
    print "mAP |",
    print " ".join(["%.2f" % score for score in scores]),
    print "| %.3f" % np.mean(scores)


def evaluate(tr_norms, te_norms, verbose=0):
    dataset = Dataset(
        'hollywood2', suffix='.per_slice.delta_60', nr_clusters=256)
    (tr_kernel, tr_labels,
     te_kernel, te_labels) = load_kernels(
         dataset, tr_norms=tr_norms, te_norms=te_norms)

    eval = Evaluation('hollywood2')
    scores = eval.fit(tr_kernel, tr_labels).score(te_kernel, te_labels)

    if verbose == 1:
        print "mAP: %.3f" % np.mean(scores)
    elif verbose > 1:
        print 'Train normalizations:', ', '.join(map(str, tr_norms))
        print 'Test normalizations:', ', '.join(map(str, te_norms))
        print_scores(scores)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluating the normalization approximations.")

    valid_norms = ['std', 'sqrt', 'L2', 'sqrt_counts']

    parser.add_argument(
        '--tr_norms', choices=valid_norms, nargs='+',
        default = [],
        help="normalizations used for training.")
    parser.add_argument(
        '--te_norms', choices=valid_norms, nargs='+',
        help=("normalizations used for testing "
             "(by default, use the same as for training)."))
    parser.add_argument(
        '-v', '--verbose', action='count', help="verbosity level.")

    args = parser.parse_args()

    if args.te_norms is None:
        args.te_norms = args.tr_norms
 
    evaluate(args.tr_norms, args.te_norms, verbose=args.verbose)


if __name__ == '__main__':
    main()
