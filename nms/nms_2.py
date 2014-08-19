
import argparse
import cPickle
import pdb
import itertools
import numpy as np
import os
import sys
import time


def non_maxima_supression_0(scored_windows, delta, min_slice, max_slice):
    """Non-maxima supression with zero overlap."""

    def generate_overlapping_windows(low, high):
        begin = xrange(low + delta - max_slice, high, delta)
        end = xrange(low + delta, high + max_slice, delta)
        return filter(
            lambda xx: min_slice <= xx[1] - xx[0] <= max_slice,
            itertools.product(begin, end))

    # Convert frame indexes to integers.
    scored_windows = [
        (int(begin_frame), int(end_frame), score)
        for begin_frame, end_frame, score in scored_windows]

    sorted_windows = sorted(scored_windows, key=lambda xx: xx[2], reverse=True)
    window_to_score = {
        (scored_window[0], scored_window[1]): scored_window[2]
        for scored_window in scored_windows}

    selected_windows = set()
    blacklisted_windows = set()

    for low, high, score in sorted_windows:

        if (low, high) in blacklisted_windows:
            continue

        overlapping_windows = generate_overlapping_windows(low, high)

        if any((
            window in selected_windows 
            for window in overlapping_windows)):
            continue

        selected_windows.add((low, high))
        blacklisted_windows.update(overlapping_windows)

    return [
        (window[0], window[1], window_to_score[(window[0], window[1])])
        for window in selected_windows]


def main():

    parser = argparse.ArgumentParser(
        description="Approximating the normalizations for the detection task.")

    parser.add_argument(
        '--infile', help="where the scored slices are stored.")
    parser.add_argument('-D', '--delta', type=int, help="base slice length.")
    parser.add_argument('--begin', type=int, help="smallest slice length.")
    parser.add_argument('--end', type=int, help="largest slice length.")

    args = parser.parse_args()

    with open(args.infile, 'r') as ff:
        results = cPickle.load(ff)

    # For compatibility reasons.
    try:
        # Now I store results for different movies in the same structure.
        movie = results.keys()[0]
        results = results[movie]
    except AttributeError:
        results = [
            (int(rr[0].split('-')[2]), int(rr[0].split('-')[3]), rr[1])
            for rr in results]

    start = time.time()
    out = non_maxima_supression_0(results, args.delta, args.begin, args.end)
    print "Time for NMS: %.2f s" % (time.time() - start)
    pdb.set_trace()


if __name__ == '__main__':
    main()

