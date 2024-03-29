"""
Bilinear Interpolation: \n
Calculates new pixels by averaging the four nearest pixels.
While this algorithm produces smoother results than the nearest neighbour algorithm, it can also produce
blurry and soft images
"""

import numpy as np

_valid = lambda r, c, n, m: 0 <= r < n and 0 <= c < m

adjacencyMatrix = [1, -1, 0, 0]


def perform2x(img):
    res = [[] for _ in range((len(img) << 1))]
    for ridx in range(0, (len(img) << 1), 2):
        rridx = (ridx >> 1)
        for cidx in range(len(img[rridx])):
            res[ridx].append(img[rridx][cidx])
            res[ridx].append((0, 0, 0, 0))
            res[ridx + 1].append((0, 0, 0, 0))
            res[ridx + 1].append(img[rridx][cidx])
    for ridx in range(len(res)):
        for cidx in range(((ridx + 1) & 1), len(res[ridx]), 2):
            avg = [0, 0, 0, 0]
            cntNeighbours = 0
            for k in range(len(adjacencyMatrix)):
                nRow, nCol = ridx + adjacencyMatrix[k], cidx + adjacencyMatrix[-k]
                if not _valid(nRow, nCol, len(res), len(res[ridx])): continue
                cntNeighbours += 1
                for clr in range(4):
                    avg[clr] += img[(nRow >> 1)][(nCol >> 1)][clr]
            for clr in range(4):
                avg[clr] /= cntNeighbours
            res[ridx][cidx] = tuple(avg)
    return np.array(res)


def perform4x(img):
    res = [[] for _ in range((len(img) << 2))]
    for ridx in range(0, (len(img) << 2), 4):
        rridx = (ridx >> 2)
        for cidx in range(len(img[rridx])):
            for repy in range(0, 4, 2):
                for repx in range(2):
                    res[ridx + repy].append(img[rridx][cidx])
                    res[ridx + repy].append((0, 0, 0, 0))
                    res[ridx + 1 + repy].append((0, 0, 0, 0))
                    res[ridx + 1 + repy].append(img[rridx][cidx])
    for ridx in range(len(res)):
        for cidx in range(((ridx + 1) & 1), len(res[ridx]), 2):
            avg = [0, 0, 0, 0]
            cntNeighbours = 0
            for k in range(len(adjacencyMatrix)):
                nRow, nCol = ridx + adjacencyMatrix[k], cidx + adjacencyMatrix[-k]
                if not _valid(nRow, nCol, len(res), len(res[ridx])): continue
                cntNeighbours += 1
                for clr in range(4):
                    avg[clr] += img[(nRow >> 2)][(nCol >> 2)][clr]
            for clr in range(4):
                avg[clr] /= cntNeighbours
            res[ridx][cidx] = tuple(avg)
    return np.array(res)
