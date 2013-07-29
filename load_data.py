import cPickle
import numpy as np
import os
from ipdb import set_trace

from sklearn.preprocessing import Scaler
from yael import yael

from fisher_vectors.model.fv_model import FVModel
from fisher_vectors.model.utils import power_normalize
from fisher_vectors.model.utils import L2_normalize
from fisher_vectors.model.utils import sstats_to_sqrt_features


def load_sample_data(dataset, sample):
    if sample in ('train', 'test'):
        stats_file = "%s.dat" % sample
        labels_file = "labels_%s.info" % sample
    else:
        stats_file = "stats.tmp/%s.dat" % sample
        labels_file = "stats.tmp/%s.info" % sample

    stats_path = os.path.join(dataset.SSTATS_DIR, stats_file)
    labels_path = os.path.join(dataset.SSTATS_DIR, labels_file)
        
    with open(dataset.GMM, 'r') as ff:
        gmm = yael.gmm_read(ff)

    data = np.fromfile(stats_path, dtype=np.float32)
    data = data.reshape(-1, gmm.k * (2 * gmm.d + 1))
    data = FVModel.sstats_to_features(data, gmm)
    with open(labels_path, 'r') as ff:
        labels = cPickle.load(ff)

    return data, labels


def load_kernels(
    dataset, tr_norms=['std', 'sqrt', 'L2'], te_norms=['std', 'sqrt', 'L2']):

    # Load sufficient statistics.
    tr_data, tr_labels = load_sample_data(dataset, 'train')
    te_data, te_labels = load_sample_data(dataset, 'test')

    with open(dataset.GMM, 'r') as ff:
        gmm = yael.gmm_read(ff)

    scalers = []
    for norm in tr_norms:
        if norm == 'std':
            scaler = Scaler()
            tr_data = scaler.fit_transform(tr_data)
            scalers.append(scaler)
        elif norm == 'sqrt':
            tr_data = power_normalize(tr_data, 0.5)
        elif norm == 'sqrt_counts':
            counts = np.maximum(1e-10, tr_data[:, :gmm.k] + yael.fvec_to_numpy(gmm.w, gmm.k))
            tr_data = approximate_signed_sqrt(tr_data, counts)  # tr_data[:, :gmm.k])
        elif norm == 'L2':
            tr_data = L2_normalize(tr_data)

    ii = 0
    for norm in te_norms:
        if norm == 'std':
            te_data = scalers[ii].transform(te_data)
            ii += 1
        elif norm == 'sqrt':
            te_data = power_normalize(te_data, 0.5)
        elif norm == 'sqrt_counts':
            counts = np.maximum(1e-10, te_data[:, :gmm.k] + yael.fvec_to_numpy(gmm.w, gmm.k))
            te_data = approximate_signed_sqrt(te_data, counts)  # te_data[:, :gmm.k])
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
    data[:, :K] = counts
    return data / sqrt_counts
