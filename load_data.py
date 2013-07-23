import cPickle
import numpy as np
import os
from ipdb import set_trace

from sklearn.preprocessing import Scaler
from yael import yael

from fisher_vectors.model.fv_model import FVModel
from fisher_vectors.model.utils import power_normalize
from fisher_vectors.model.utils import L2_normalize


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


def load_kernels(dataset):
    tr_data, tr_labels = load_sample_data(dataset, 'train')
    te_data, te_labels = load_sample_data(dataset, 'test')

    scaler = Scaler()

    tr_data = scaler.fit_transform(tr_data)
    tr_data = power_normalize(tr_data, 0.5)
    tr_data = L2_normalize(tr_data)
    tr_kernel = np.dot(tr_data, tr_data.T)

    te_data = scaler.transform(te_data)
    te_data = power_normalize(te_data, 0.5)
    te_data = L2_normalize(te_data)
    te_kernel = np.dot(te_data, tr_data.T)

    return tr_kernel, tr_labels, te_kernel, te_labels
