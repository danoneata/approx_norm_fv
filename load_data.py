import numpy as np
import os
from ipdb import set_trace

from yael import yael

from fisher_vectors.model.fv_model import FVModel


def load_sample_data(dataset, sample):
    if sample == 'train':
        file = "train.dat"
    else:
        file = "stats.tmp/%s.dat" % sample

    stats_path = os.path.join(dataset.SSTATS_DIR, file)
    info_path = stats_path.replace(".dat", ".info")
        
    with open(dataset.GMM, 'r') as ff:
        gmm = yael.gmm_read(ff)

    data = np.fromfile(stats_path, dtype=np.float32)
    data = data.reshape(-1, gmm.k * (2 * gmm.d + 1))
    data = FVModel.sstats_to_features(data, gmm)

    return data
