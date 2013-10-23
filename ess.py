import argparse
import heapq
import numpy as np
import pdb

from interval import interval


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


def efficient_subwindow_search(
    bounding_function, heap, blacklist=[], verbose=0):

    for ii in xrange(MAX_NR_ITER):

        if verbose > 2:
            print ii, heap

        score, bounds = heapq.heappop(heap)

        if verbose > 2:
            print "Pop", score, bounds

        if bounds_in_blacklist(bounds, blacklist):
            continue

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
            score = bounding_function(bounds_i)
            heapq.heappush(heap, (-score, bounds_i))

            if verbose > 2:
                print "Push", score, bounds_i

        if bounds_j.is_legal():
            score = bounding_function(bounds_j)
            heapq.heappush(heap, (-score, bounds_j))

            if verbose > 2:
                print "Push", score, bounds_j

        if verbose > 2:
            print

    return - score, (bounds.low + bounds.high) / 2, heap


def bounds_in_blacklist(bounds, blacklist):
    union = interval[bounds.get_union()]
    inter = interval[bounds.get_intersection()]
    return (
        any((union in window for window in blacklist)) or
        any((len((inter & window).extrema) > 1 for window in blacklist)))


def integral(X):
    return np.hstack((0, np.cumsum(X)))


def pos_neg_integral(scores):
    """Works only for 1D arrays at the moment, but can be easily extended."""
    scores = np.hstack([[0], scores])  # Padding.
    pos_scores, neg_scores = scores.copy(), scores.copy()
    idxs = scores >= 0
    pos_scores[~idxs], neg_scores[idxs] = 0, 0
    return np.cumsum(pos_scores), np.cumsum(neg_scores)


def eval_integral(X, bb):
    low, high = bb
    return X[high] - X[low] if high > low else 0


def linear_bounding_function_builder(scores):

    pos_integral_scores, neg_integral_scores = pos_neg_integral(scores)

    def linear_bounding_function(bounds):
        union = bounds.get_union()
        inter = bounds.get_intersection()

        pos_union = eval_integral(pos_integral_scores, union)
        neg_inter = eval_integral(neg_integral_scores, inter)

        return pos_union + neg_inter

    return linear_bounding_function


def norm_bounding_function_builder(scores):

    pos_integral_scores, neg_integral_scores = pos_neg_integral(scores)
    norms = integral(scores ** 2)

    def norm_bounding_function(bounds):

        union = bounds.get_union()
        inter = bounds.get_intersection()

        if inter[0] == inter[1] == union[0] == union[1]:
            return - np.inf

        if inter[1] <= inter[0]:
            return np.inf

        pos_union = eval_integral(pos_integral_scores, union)
        neg_inter = eval_integral(neg_integral_scores, inter)

        bound_norm = eval_integral(norms, inter)

        return (pos_union + neg_inter) / np.sqrt(bound_norm)

    return norm_bounding_function


def max_subarray(A):
    """Maximum sub-array search (from Wikipedia)."""
    max_ending_here = max_so_far = 0
    for xx in A:
        max_ending_here = max(0, max_ending_here + xx)
        max_so_far = max(max_so_far, max_ending_here)
    return max_so_far


def test(bounding_function_builder, nr_tests, verbose=0):
    """Some simple tests to check that everything is fine."""
    np.random.seed(0)
    TESTS = [
        [-2, 1, -3, 4, -1, 2, 1, -5, 4],
        [+2, 1, +3, 4, +1, 2, 1, +5, 4],
        np.random.randn(100)]

    for scores in TESTS[:nr_tests]:
        scores = np.array(scores)
        N = len(scores)

        bounding_function = bounding_function_builder(scores)
        low, high = np.array([0, 0]), np.array([N, N])

        heap = [(0, Bounds(low, high))]

        score, idxs, _ = efficient_subwindow_search(
            bounding_function, heap, verbose=verbose)

        if verbose:
            print "Final score:", score
            print "Bounds", idxs

        # assert scores[idxs[0]: idxs[1]].sum() == max_subarray(scores)


def main():
    BUILDERS = {
        'linear': linear_bounding_function_builder,
        'norm': norm_bounding_function_builder,
    }

    parser = argparse.ArgumentParser(
        description="Efficient sub-window search with branch and bound method.")

    parser.add_argument(
        '-f', '--bounding_function', choices=BUILDERS.keys(),
        help="type of the bounding function.")
    parser.add_argument(
        '--nr_tests', type=int, default=1, help="number of tests to run.")
    parser.add_argument(
        '-v', '--verbose', action='count', help="verbosity level.")

    args = parser.parse_args()

    test(
        BUILDERS[args.bounding_function],
        nr_tests=args.nr_tests,
        verbose=args.verbose)


if __name__ == "__main__":
    main()

