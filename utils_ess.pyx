cdef inline bint intersects(int x0, int x1, int y0, int y1):
    return min(x1, y1) > max(x0, y0)


cdef inline bint contains(int x0, int x1, int y0, int y1):
    return y0 < x0 and x0 < y1


cdef inline tuple get_union(int x0, int x1, int y0, int y1):
    return (x0, y1) if x0 < y1 else (y1, x0)


cdef inline tuple get_inter(int x0, int x1, int y0, int y1):
    return (x1, y0) if x1 < y0 else (y0, x1)


def in_blacklist(low, high, list blacklist):

    u0, u1 = get_union(low[0], low[1], high[0], high[1])
    i0, i1 = get_inter(low[0], low[1], high[0], high[1])

    return (
        any(contains(u0, u1, w[0], w[1]) for w in blacklist) or
        any(intersects(i0, i1, w[0], w[1]) for w in blacklist))

