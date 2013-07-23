import numpy as np

from dataset import Dataset
from fisher_vectors.evaluation import Evaluation

from load_data import load_kernels

def print_scores(scores):
    scores = [score * 100 for score in scores]
    print "mAP |",
    print " ".join(["%.2f" % score for score in scores]),
    print "| %.3f" % np.mean(scores)


verbose = 10
dataset = Dataset('hollywood2', suffix='.per_slice.delta_60', nr_clusters=256)
tr_kernel, tr_labels, te_kernel, te_labels = load_kernels(dataset)
eval = Evaluation('hollywood2')
scores = eval.fit(tr_kernel, tr_labels).score(te_kernel, te_labels)
if verbose > 1:
    print_scores(scores)
elif verbose == 1:
    print "mAP: %.3f" % np.mean(scores)
