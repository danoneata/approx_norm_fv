""" Uses approximations for both signed square rooting and l2 normalization."""
from multiprocessing import Pool
from ipdb import set_trace
import numpy as np

from dataset import Dataset
from fisher_vectors.evaluation import Evaluation
from fisher_vectors.evaluation.utils import average_precision
from fisher_vectors.model.utils import compute_L2_normalization

from load_data import load_kernels
from load_data import load_sample_data


# TODO Possible improvements:
# [ ] Use sparse matrices for masks, especially for `video_agg_mask`.


def compute_weights(clf, xx):
    return (
        np.dot(clf.dual_coef_, xx[clf.support_]),
        clf.intercept_)


def predict(yy, weights, bias):
    return (- np.dot(yy, weights.T) + bias).squeeze()


class LocalInfo:
    def __init__(self, names, K, D):
        self.video_agg_mask = self._compute_video_aggregation_mask(names)
        self.scores_mask = self._compute_scores_mask(D, K)

    def _compute_video_aggregation_mask(self, names):
        ii, idxs, seen = -1, [], []
        for name in names:
            if name not in seen:
                ii += 1
                seen.append(name)
            idxs.append(ii)
            
        nn = len(seen)
        N = len(names)
        mask = np.zeros((N, nn))
        mask[range(N), idxs] = 1

        return mask.T  # nxN

    def _compute_scores_mask(self, D, K):
        I = np.eye(K)
        return np.hstack((
            I.repeat(D, axis=1),
            I.repeat(D, axis=1))).T  # 2KDxK

    def compute_counts(self, counts):
        self.counts = counts  # NxK

    def compute_l2_norms(self, fisher_vectors):
        self.l2_norms = np.dot(fisher_vectors ** 2, self.scores_mask)  # NxK

    def compute_scores(self, fisher_vectors, weights, bias=0):
        self.bias = bias
        self.scores = np.dot(- fisher_vectors * weights, self.scores_mask)  # NxK

    def approximate_video_scores(self):
        cn = np.dot(self.video_agg_mask, self.counts)    # nxK
        sn = np.dot(self.video_agg_mask, self.scores)    # nxK
        ln = np.dot(self.video_agg_mask, self.l2_norms)  # nxK

        approx_l2 = np.sum(ln / cn, axis=1)              # n
        approx_sn = np.sum(sn / np.sqrt(cn), axis=1)     # n

        # return approx_sn / approx_l2
        return approx_sn


def load_slices(dataset, samples):
    counts = []
    fisher_vectors = []
    labels = []
    names = []

    for jj, sample in enumerate(samples):
        fv, ii, cc = load_sample_data(
            dataset, sample, analytical_fim=True, pi_derivatives=False,
            sqrt_nr_descs=False)

        if sample.movie in names:
            continue

        nn = fv.shape[0]
        names += [sample.movie] * nn
        labels += [ii['label']]
        fisher_vectors.append(fv)
        counts.append(cc)

    fisher_vectors = np.vstack(fisher_vectors)
    counts = np.vstack(counts)

    return names, labels, fisher_vectors, counts


D, K = 64, 256
dataset = Dataset('hollywood2', suffix='.per_slice.delta_60', nr_clusters=K)

# Load data.
tr_kernel, tr_labels, scalers, tr_data = load_kernels(
    dataset, tr_norms=[], te_norms=[], analytical_fim=True,
    pi_derivatives=False, sqrt_nr_descs=False, only_train=True)


# Training.
eval = Evaluation('hollywood2')
eval.fit(tr_kernel, tr_labels)

# Evaluation.
ap = []
# idxs = slice(0, 10)

te_samples, _ = dataset.get_data('test')
idxs = slice(0, len(te_samples))
te_names, te_labels, te_data, te_counts = load_slices(dataset, te_samples[idxs])
te_labels = eval.lb.transform(te_labels[idxs])

te_local_info = LocalInfo(te_names, K, D)
te_local_info.compute_counts(te_counts)
te_local_info.compute_l2_norms(te_data)

ap = []

set_trace()
fv, ll, nn = load_sample_data(
    dataset, 'test', analytical_fim=True, pi_derivatives=False,
    sqrt_nr_descs=False)

for ii in xrange(eval.nr_classes):
    true_labels = te_labels[:, ii]
    weight, bias = compute_weights(eval.clf[ii], tr_data)
    te_local_info.compute_scores(te_data, weight, bias=bias)
    predictions = te_local_info.approximate_video_scores()
    ap.append(average_precision(true_labels, predictions))
    print "%.2f" % (100 * ap[-1])

print '-----'
print "%.2f" % (100 * np.mean(ap))


#if False:
#    te_data, te_labels, te_counts = load_sample_data(
#        dataset, 'test', analytical_fim=True, pi_derivatives=False,
#        sqrt_nr_descs=False)

