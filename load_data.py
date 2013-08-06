import cPickle
import matplotlib.pyplot as plt
import numpy as np
import os
from ipdb import set_trace

from sklearn.preprocessing import Scaler
from yael import yael

from fisher_vectors.model.fv_model import FVModel
from fisher_vectors.model.utils import power_normalize
from fisher_vectors.model.utils import L2_normalize
from fisher_vectors.model.utils import sstats_to_sqrt_features


def load_sample_data(
    dataset, sample, analytical_fim=False, pi_derivatives=False,
    sqrt_nr_descs=False):

    if sample in ('train', 'test'):
        stats_file = "%s.dat" % sample
        labels_file = "labels_%s.info" % sample
        info_file = "info_%s.info" % sample
    else:
        stats_file = "stats.tmp/%s.dat" % sample
        labels_file = "stats.tmp/%s.info" % sample
        info_file = labels_file

    stats_path = os.path.join(dataset.SSTATS_DIR, stats_file)
    labels_path = os.path.join(dataset.SSTATS_DIR, labels_file)
    info_path = os.path.join(dataset.SSTATS_DIR, info_file)
        
    with open(dataset.GMM, 'r') as ff:
        gmm = yael.gmm_read(ff)

    K = gmm.k
    D = gmm.k * (2 * gmm.d + 1)

    data = np.fromfile(stats_path, dtype=np.float32)
    data = data.reshape(-1, D)
    counts = data[:, : K]

    if analytical_fim:
        data = FVModel.sstats_to_normalized_features(data, gmm)
    else:
        data = FVModel.sstats_to_features(data, gmm)

    with open(labels_path, 'r') as ff:
        labels = cPickle.load(ff)

    if sqrt_nr_descs:
        with open(info_path, 'r') as ff:
            T = cPickle.load(ff)['nr_descs']
        T = np.sqrt(T)[:, np.newaxis]
    else:
        T = 1.

    if pi_derivatives:
        idxs = slice(D)
    else:
        idxs = slice(K, D)

    return T * data[:, idxs], labels, counts


def load_kernels(
    dataset, tr_norms=['std', 'sqrt', 'L2'], te_norms=['std', 'sqrt', 'L2'],
    analytical_fim=False, pi_derivatives=False, sqrt_nr_descs=False):

    with open(dataset.GMM, 'r') as ff:
        gmm = yael.gmm_read(ff)

    # Load sufficient statistics.
    tr_data, tr_labels, tr_counts = load_sample_data(
        dataset, 'train', analytical_fim=analytical_fim,
        pi_derivatives=pi_derivatives, sqrt_nr_descs=sqrt_nr_descs)
    te_data, te_labels, te_counts = load_sample_data(
        dataset, 'test', analytical_fim=analytical_fim,
        pi_derivatives=pi_derivatives, sqrt_nr_descs=False)

    scalers = []
    plot_fisher_vector(tr_data[0], 'before')
    for norm in tr_norms:
        if norm == 'std':
            scaler = Scaler()
            tr_data = scaler.fit_transform(tr_data)
            scalers.append(scaler)
        elif norm == 'sqrt':
            tr_data = power_normalize(tr_data, 0.5)
        elif norm == 'sqrt_cnt':
            tr_data = approximate_signed_sqrt(tr_data, tr_counts)  # tr_data[:, :gmm.k])
        elif norm == 'L2':
            tr_data = L2_normalize(tr_data)
        plot_fisher_vector(tr_data[0], 'after_%s' % norm)

    ii = 0
    for norm in te_norms:
        if norm == 'std':
            te_data = scalers[ii].transform(te_data)
            ii += 1
        elif norm == 'sqrt':
            te_data = power_normalize(te_data, 0.5)
        elif norm == 'sqrt_cnt':
            te_data = approximate_signed_sqrt(te_data, te_counts)  # te_data[:, :gmm.k])
        elif norm == 'L2':
            te_data = L2_normalize(te_data)

    tr_kernel = np.dot(tr_data, tr_data.T)
    te_kernel = np.dot(te_data, tr_data.T)

    return tr_kernel, tr_labels, te_kernel, te_labels


def approximate_signed_sqrt(data, counts):
    K = counts.shape[1]
    N, dim = data.shape
    D = (dim / K - 1) / 2
    sqrtQ = np.sqrt(np.abs(counts))
    sqrt_counts = np.hstack((
        sqrtQ, np.repeat(sqrtQ, D, axis=1), np.repeat(sqrtQ, D, axis=1)))
    data = data / sqrt_counts
    data[np.isnan(data) | np.isinf(data)] = 0.
    return data


def plot_fisher_vector(xx, name='oo'):
    ii = np.argmax(np.abs(xx))
    print "Maximum at %d is %f." % (ii, xx[ii])
    D = xx.size
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.vlines(np.arange(D), np.zeros(D), xx)
    plt.xlabel('$d$')
    plt.ylabel(r'$\nabla_{\theta}\mathbf{x}$')
    plt.savefig('/tmp/%s.png' % name)
