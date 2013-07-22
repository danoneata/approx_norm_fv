# A bunch of experiments to check the L2 normalization approximation.
from ipdb import set_trace
import numpy as np
import random

from fisher_vectors.model.utils import compute_L2_normalization


random.seed(0)
np.random.seed(0)


def print_header():
    print "%22s" % 'True value',
    print "%22s" % 'Approximated value',
    print "%22s" % 'Absolute error',
    print "%22s" % 'Relative error'
    print "%s" % ('-' * 22),
    print "%s" % ('-' * 22),
    print "%s" % ('-' * 22),
    print "%s" % ('-' * 22)


def print_errors(true_value, approx_value):
    print "%22.2f" % true_value,
    print "%22.2f" % approx_value,
    print "%22.2f" % np.abs(approx_value - true_value),
    print "%20.2f %%" % (np.abs(approx_value - true_value) / true_value * 100)


def print_footer(true_values, approx_values):
    mean_abs, std_abs, mean_rel, std_rel = mean_std_err_errors(
        true_values, approx_values)
    print (45 * " "),
    print "%22s" % ("%2.2f" % mean_abs + " +/- " + "%2.2f" % std_abs),
    print "%22s" % ("%2.2f" % mean_rel + " +/- " + "%2.2f" % std_rel)


def print_info(true_values, approx_values, verbose):
    print_header()
    if verbose >= 2:
        for true_value, approx_value in zip(true_values, approx_values):
            print_errors(true_value, approx_value)
    print_footer(true_values, approx_values)


def mean_std_err_errors(true_values, approx_values):
    """ Returns mean and standard errors (absolute and relative). """
    true_values = np.array(true_values)
    approx_values = np.array(approx_values)

    absolute_err = np.abs(true_values - approx_values)
    relative_err = absolute_err / true_values * 100

    N = np.size(absolute_err)
    return (
        np.mean(absolute_err), np.std(absolute_err) / N,
        np.mean(relative_err), np.std(relative_err) / N)


def generate_data(N, D, _type):
    """ Generates artificial data to test on the L2 approximation.

    Parameters
    ----------
    N: int
        Number of data points.

    D: int
        Dimension of data points.

    _type: str, {'independent', 'correlated', 'sparse'}
        The type of data to generate.

    """
    if _type == 'independent':
        return np.random.randn(N, D)
    elif _type.startswith('sparse'):
        try:
            kk = int(_type.split('_')[1])
        except ValueError:
            kk = int(float(_type.split('_')[1]) * D)
        xx = np.random.randn(N, D)
        return np.vstack(
            [xx[ii, random.sample(range(D), kk)] for ii in xrange(N)])
    else:
        assert False, "Unknown data type."


def experiment_L2_approx(N, D, _type, nr_repeats, verbose=0):
    true_values, approx_values = [], []
    for ii in xrange(nr_repeats):
        data = generate_data(N, D, _type)

        L2_norm_slice = compute_L2_normalization(data) / N ** 2
        L2_norm_all = compute_L2_normalization(np.atleast_2d(np.mean(data, 0)))
        L2_norm_approx = np.sum(L2_norm_slice)

        true_values.append(L2_norm_all)
        approx_values.append(L2_norm_approx)

    if verbose:
        print "N = %d; D = %d." % (N, D)
        print "Data generated:", _type
        print_info(true_values, approx_values, verbose)
        print

    return mean_std_err_errors(true_values, approx_values)


experiment_L2_approx(10, 1000, 'sparse', 3, 2)
