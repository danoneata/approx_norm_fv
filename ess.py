import argparse
import heapq
import numpy as np
from pdb import set_trace


MAX_NR_ITER = 10000

# TODO Things to improve.
# [x] Fix boundaries for integral images.
# [ ] Write `integral` function in Cython.
# [ ] Generalize algorithm for more dimensions.
# [ ] The `Bounds` class is not well written; replace the indexes with namedtuples.
# [ ] Avoid mutable data.


class Bounds:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __repr__(self):
        return "low=(%s), high=(%s)" % (
            ','.join(map(str, self.low)),
            ','.join(map(str, self.high)))

    def get_maximum_index(self):
        delta = self.high - self.low
        ii = np.argmax(self.high - self.low)
        return -1 if delta[ii] <= 0 else ii

    def is_legal(self):
        return self.low[0] <= self.high[1]

    def get_union(self):
        return self.low[0], self.high[1]

    def get_intersection(self):
        return self.high[0], self.low[1]


def efficient_subwindow_search(init_bounds, bounding_function, verbose=0):

    heap = [(0, init_bounds)]

    for ii in xrange(MAX_NR_ITER):
        
        if verbose > 2:
            print ii, heap

        score, bounds = heapq.heappop(heap)

        # Branch...
        split_index = bounds.get_maximum_index()

        if split_index == -1:
            break

        bounds_i = Bounds(bounds.low.copy(), bounds.high.copy())
        bounds_j = Bounds(bounds.low.copy(), bounds.high.copy())

        middle = (bounds.low[split_index] + bounds.high[split_index]) / 2

        bounds_i.high[split_index] = middle
        bounds_j.low[split_index] = middle + 1

        # ... and bound.
        if bounds_i.is_legal():
            heapq.heappush(heap, (-bounding_function(bounds_i), bounds_i))

        if bounds_j.is_legal():
            heapq.heappush(heap, (-bounding_function(bounds_j), bounds_j))

    return - score, (bounds.low + bounds.high) / 2


def linear_bounding_function(bounds, pos_integral_scores, neg_integral_scores):
    union_low, union_high = bounds.get_union()
    inter_low, inter_high = bounds.get_intersection() 

    pos_union = pos_integral_scores[union_high] - pos_integral_scores[union_low]
    neg_inter = (
        neg_integral_scores[inter_high] - neg_integral_scores[inter_low]
        if inter_high > inter_low
        else 0)

    return pos_union + neg_inter


def integral(scores):
    """Works only for 1D arrays at the moment, but can be easily extended."""
    scores = np.hstack([[0], scores])  # Padding.
    pos_scores, neg_scores = scores.copy(), scores.copy()
    idxs = scores >= 0
    pos_scores[~idxs], neg_scores[idxs] = 0, 0
    return np.cumsum(pos_scores), np.cumsum(neg_scores)


def max_subarray(A):
    """Maximum sub-array search (from Wikipedia)."""
    max_ending_here = max_so_far = 0
    for xx in A:
        max_ending_here = max(0, max_ending_here + xx)
        max_so_far = max(max_so_far, max_ending_here)
    return max_so_far


def test(verbose=0):
    """Some simple tests to check that everything is fine."""
    np.random.seed(0)
    TESTS = [
        [-2, 1, -3, 4, -1, 2, 1, -5, 4],
        [+2, 1, +3, 4, +1, 2, 1, +5, 4],
        np.random.randn(100)]

    for scores in TESTS:
        scores = np.array(scores)
        N = len(scores)

        pos_integral_scores, neg_integral_scores = integral(scores)
        bounding_function = lambda bounds: linear_bounding_function(
            bounds, pos_integral_scores, neg_integral_scores)

        low, high = np.array([0, 0]), np.array([N, N])
        score, idxs = efficient_subwindow_search(
            Bounds(low, high), bounding_function, verbose=verbose)

        if verbose:
            print "Final score:", score
            print "Bounds", idxs

        assert scores[idxs[0]: idxs[1]].sum() == max_subarray(scores)


def main():
    parser = argparse.ArgumentParser(
        description="Efficient sub-window search with branch and bound method.")

    parser.add_argument('--test', action='store_true', help="run tests.")
    parser.add_argument(
        '-v', '--verbose', action='count', help="verbosity level.")

    args = parser.parse_args()

    if args.test:
        test(verbose=args.verbose)


if __name__ == "__main__":
    main()

