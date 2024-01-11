"""
Nearest Neighbour Interpolation: \n
Simply replicating the nearest pixel to increase the size of the image.
This algorithm is probably the fastest approach but often results in pixelation and jagged edges.
Fastest algorithm
"""

import numpy as np

_valid = lambda r, c, n, m: 0 <= r < n and 0 <= c < m


def perform2x(img):
    res = [[] for _ in range((len(img) << 1))]
    for ridx in range(0, (len(img) << 1), 2):
        rridx = (ridx >> 1)
        for cidx in range(len(img[rridx])):
            res[ridx].append(img[rridx][cidx])
            res[ridx].append(img[rridx][cidx])
            res[ridx + 1].append(img[rridx][cidx])
            res[ridx + 1].append(img[rridx][cidx])
    return np.array(res)


def perform4x(img):
    res = [[] for _ in range((len(img) << 2))]
    for ridx in range(0, (len(img) << 2), 4):
        rridx = (ridx >> 2)
        for cidx in range(len(img[rridx])):
            for repy in range(0, 4, 2):
                for repx in range(2):
                    res[ridx + repy].append(img[rridx][cidx])
                    res[ridx + repy].append(img[rridx][cidx])
                    res[ridx + 1 + repy].append(img[rridx][cidx])
                    res[ridx + 1 + repy].append(img[rridx][cidx])
    return np.array(res)
