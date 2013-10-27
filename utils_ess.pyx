# cython: profile=True
import heapq

import numpy as np
cimport numpy as np


# TODO
# [ ] Use with `for` loops in _eval_integral and maybe `inline`.
# [ ] Change from `tuple` to `Interval` where possible.


cdef extern from "math.h":
    double sqrt(double)


# Data structures.
cdef struct Interval:
   unsigned int elem0
   unsigned int elem1


cdef struct Bounds:
    Interval low
    Interval high


cpdef Interval b_init_interval(tuple tt):
    cdef Interval interval
    interval.elem0 = tt[0]
    interval.elem1 = tt[1]
    return interval


cpdef Bounds b_init_bounds(tuple low, tuple high):
    cdef Bounds bounds
    bounds.low.elem0 = low[0]
    bounds.low.elem1 = low[1]
    bounds.high.elem0 = high[0]
    bounds.high.elem1 = high[1]
    return bounds


cdef class Function:
    cpdef double evaluate(self, Bounds bounds) except *:
        return 0


# General functions.
cdef inline bint intersects(int x0, int x1, int y0, int y1):
    return min(x1, y1) > max(x0, y0)


cdef inline bint contains(int x0, int x1, int y0, int y1):
    return y0 < x0 and x0 < y1


cdef inline tuple get_union(int x0, int x1, int y0, int y1):
    return (x0, y1) if x0 < y1 else (y1, x0)


cdef inline tuple get_inter(int x0, int x1, int y0, int y1):
    return (x1, y0) if x1 < y0 else (y0, x1)


cdef int b_get_maximum_index(Bounds bounds):

    delta0 = bounds.high.elem0 - bounds.low.elem0
    delta1 = bounds.high.elem1 - bounds.low.elem1

    if delta0 <= 0 and delta1 <= 0:
        return -1
    if delta0 >= delta1:
        return 0
    else:
        return 1


# Functions on the `Bounds` data structure.
cdef bint b_is_legal(Bounds bounds):
    return bounds.low.elem0 <= bounds.high.elem1


cpdef tuple b_get_union(Bounds bounds):
    return bounds.low.elem0, bounds.high.elem1


cpdef tuple b_get_intersection(Bounds bounds):
    return bounds.high.elem0, bounds.low.elem1


cdef void b_print(Bounds bounds):
    print "low=(%d,%d), high=(%d,%d)" % (
        bounds.low.elem0, bounds.low.elem1,
        bounds.high.elem0, bounds.high.elem1)


cpdef bint b_in_blacklist(Bounds bounds, list blacklist):

    cdef Interval w
    cdef unsigned int u0, u1, i0, i1

    # Get union interval.
    if bounds.low.elem0 < bounds.high.elem1:
        u0 = bounds.low.elem0
        u1 = bounds.high.elem1
    else:
        u1 = bounds.low.elem0
        u0 = bounds.high.elem1

    # Get intersection interval.
    if bounds.low.elem1 < bounds.high.elem0:
        i0 = bounds.low.elem1
        i1 = bounds.high.elem0
    else:
        i1 = bounds.low.elem1
        i0 = bounds.high.elem0

    for w in blacklist:
        if contains(u0, u1, w.elem0, w.elem1) or intersects(i0, i1, w.elem0, w.elem1):
            return True

    return False


cpdef efficient_subwindow_search(
    Function bounding_function,
    heap,
    list blacklist=[],
    int verbose=0):

    cdef Bounds bounds
    cdef Bounds bounds_i
    cdef Bounds bounds_j
    cdef int ii, split_index, middle
    cdef double score

    for ii in xrange(100000):

        score, bounds = heapq.heappop(heap)

        if len(blacklist) > 0 and b_in_blacklist(bounds, blacklist):
            continue

        # Branch...
        split_index = b_get_maximum_index(bounds)

        if split_index == -1:
            break

        bounds_i.low.elem0 = bounds.low.elem0
        bounds_i.low.elem1 = bounds.low.elem1
        bounds_i.high.elem0 = bounds.high.elem0
        bounds_i.high.elem1 = bounds.high.elem1

        bounds_j.low.elem0 = bounds.low.elem0
        bounds_j.low.elem1 = bounds.low.elem1
        bounds_j.high.elem0 = bounds.high.elem0
        bounds_j.high.elem1 = bounds.high.elem1

        if split_index == 0:
            middle = (bounds.low.elem0 + bounds.high.elem0) / 2
            bounds_i.high.elem0 = middle
            bounds_j.low.elem0 = middle + 1
        elif split_index == 1:
            middle = (bounds.low.elem1 + bounds.high.elem1) / 2
            bounds_i.high.elem1 = middle
            bounds_j.low.elem1 = middle + 1

        # ... and bound.
        if b_is_legal(bounds_i):
            score = bounding_function.evaluate(bounds_i)
            heapq.heappush(heap, (-score, bounds_i))

        if b_is_legal(bounds_j):
            score = bounding_function.evaluate(bounds_j)
            heapq.heappush(heap, (-score, bounds_j))

    elem0 = (bounds.low.elem0 + bounds.high.elem0) / 2
    elem1 = (bounds.low.elem1 + bounds.high.elem1) / 2

    return - score,  (elem0, elem1), heap


cdef class LinearBoundingFunction(Function):
    cdef np.ndarray pos_integral_scores
    cdef np.ndarray neg_integral_scores

    def __init__(self, scores):
        self.pos_integral_scores, self.neg_integral_scores = self._pos_neg_integral(scores)

    cpdef double evaluate(self, Bounds bounds) except *:
        uu = b_get_union(bounds)
        ii = b_get_intersection(bounds)

        pos_union = self._eval_integral(self.pos_integral_scores, uu)
        neg_inter = self._eval_integral(self.neg_integral_scores, ii)

        return pos_union + neg_inter

    def _eval_integral(self, X, bb):
        low, high = bb
        return X[high] - X[low] if high > low else 0

    def _pos_neg_integral(self, scores):
        """Works only for 1D arrays at the moment, but can be easily extended."""
        scores = np.hstack([[0], scores])  # Padding.
        pos_scores, neg_scores = scores.copy(), scores.copy()
        idxs = scores >= 0
        pos_scores[~idxs], neg_scores[idxs] = 0, 0
        return np.cumsum(pos_scores), np.cumsum(neg_scores)


cdef class ApproxNormsBoundingFunction(Function):

    cdef np.ndarray slice_vw_scores_no_integral,
    cdef np.ndarray slice_vw_l2_norms_no_integral,
    cdef np.ndarray pos_slice_vw_scores
    cdef np.ndarray neg_slice_vw_scores
    cdef np.ndarray slice_vw_counts
    cdef np.ndarray slice_vw_l2_norms
    cdef int min_window
    cdef int max_window
    cdef list banned_intervals

    def __init__(
        self,
        np.ndarray[np.float64_t, ndim=2] slice_vw_scores_no_integral,
        np.ndarray[np.float64_t, ndim=2] slice_vw_l2_norms_no_integral,
        np.ndarray[np.float64_t, ndim=2] pos_slice_vw_scores,
        np.ndarray[np.float64_t, ndim=2] neg_slice_vw_scores,
        np.ndarray[np.float64_t, ndim=2] slice_vw_counts,
        np.ndarray[np.float64_t, ndim=2] slice_vw_l2_norms,
        int min_window,
        int max_window):

        self.slice_vw_scores_no_integral = slice_vw_scores_no_integral
        self.slice_vw_l2_norms_no_integral = slice_vw_l2_norms_no_integral
        self.pos_slice_vw_scores = pos_slice_vw_scores 
        self.neg_slice_vw_scores = neg_slice_vw_scores 
        self.slice_vw_counts = slice_vw_counts
        self.slice_vw_l2_norms = slice_vw_l2_norms

        self.min_window = min_window
        self.max_window = max_window

    def set_banned_intervals(self, banned_intervals):
        self.banned_intervals = banned_intervals

    cpdef double evaluate(self, Bounds bounds) except *:

        cdef unsigned int kk
        cdef tuple uu, ii
        cdef np.ndarray[np.float64_t, ndim=1] score_union, score_inter, counts_union, counts_inter, l2_norms_inter
        cdef double bound_sqrt_scores, bound_approx_l2_norm

        bound_sqrt_scores = 0
        bound_approx_l2_norm = 0

        uu = b_get_union(bounds)
        ii = b_get_intersection(bounds)

        if ii[0] == ii[1] == uu[0] == uu[1] or uu[0] >= uu[1]:
            return - np.inf

        if ii[1] - ii[0] > self.max_window:
            return - np.inf

        if uu[1] - uu[0] < self.min_window:
            return - np.inf

        # Empty intersection.
        if ii[1] <= ii[0]:

            counts_union = self._eval_integral(self.slice_vw_counts, uu)
            l2_norms_union = np.min(self.slice_vw_l2_norms_no_integral[uu[0]: uu[1]], axis=0)
            score_union = np.max(self.slice_vw_scores_no_integral[uu[0]: uu[1]], axis=0)

            for kk in xrange(score_union.shape[0]):
                if counts_union[kk] == 0:
                    continue
                bound_sqrt_scores += score_union[kk] / sqrt(counts_union[kk])
                bound_approx_l2_norm += l2_norms_union[kk] / counts_union[kk]

            return bound_sqrt_scores / np.sqrt(bound_approx_l2_norm) if bound_approx_l2_norm != 0 else + np.inf

        l2_norms_inter = self._eval_integral(self.slice_vw_l2_norms, ii)
        if np.all(l2_norms_inter == 0):
            return - np.inf

        if len(self.banned_intervals) > 0 and b_in_blacklist(bounds, self.banned_intervals):
            return - np.inf

        score_union = self._eval_integral(self.pos_slice_vw_scores, uu)
        score_inter = self._eval_integral(self.neg_slice_vw_scores, ii)

        counts_union = self._eval_integral(self.slice_vw_counts, uu)
        counts_inter = self._eval_integral(self.slice_vw_counts, ii)

        for kk in xrange(score_union.shape[0]):
            if counts_inter[kk] == 0 or counts_union[kk] == 0:
                continue
            bound_sqrt_scores += (score_union[kk] + score_inter[kk]) / sqrt(counts_inter[kk])
            bound_approx_l2_norm += l2_norms_inter[kk] / counts_union[kk]

        return bound_sqrt_scores / sqrt(bound_approx_l2_norm)

    cpdef np.ndarray _eval_integral(
        self,
        np.ndarray[np.float64_t, ndim=2] X,
        tuple bb):

        low, high = bb
        return X[high, :] - X[low, :] if high > low else np.zeros(X.shape[1]) 


def test():
    cdef Bounds bounds
    scores = [-2, 1, -3, 4, -1, 2, 1, -5, 4]

    bounds.low.elem0 = 0
    bounds.low.elem1 = 0
    bounds.high.elem0 = len(scores)
    bounds.high.elem1 = len(scores)

    foo = LinearBoundingFunction(scores)
    heap = [(0, bounds)]

    return efficient_subwindow_search(foo, heap, verbose=3)

