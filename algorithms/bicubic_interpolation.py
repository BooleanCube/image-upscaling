"""
Bicubic Interpolation: \n
Interpolates value of unknown / new pixels by assuming a quadratic change in channel value from pixel to pixel.
The intensity values of more neighbouring pixels to create a smoother curve since it uses 4*4 = 16 pixels to
extrapolate pixel value which also makes it computationally more intensive.
While this algorithm produces smoother results than the bilinear interpolation algorithm, it can also produce
blurry and soft images.

Basis functions:
q1 = (-t³ + 2t² — t) / 2
q2 = (3t³ — 5t² + 2 ) / 2
q3 = (-3t³ + 4t + t) / 2
q4 = (t³ — t²) / 2

Interpolation Formula:
p (new) = round(p1q1 + p2q2 + p3q3 + p4q4) / 1.5
where p2 and p3 are the closest to the new point, and p1 and p4 are next to p2 and p3 respectively.
scaled down by a factor of 1.5 to keep brightness low of combined channels.
"""

import numpy as np
import cv2

_valid = lambda r, c, n, m: 0 <= r < n and 0 <= c < m

_adjacencyMatrixR, _adjacencyMatrixC = [], []
for i in range(-1, 2): _adjacencyMatrixR += [i] * 3
for i in range(3): _adjacencyMatrixC += list(range(-1, 2))


def _perform4x(img):
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
            for k in range(len(_adjacencyMatrixR)):
                nRow, nCol = ridx + _adjacencyMatrixR[k], cidx + _adjacencyMatrixC[k]
                if not _valid(nRow, nCol, len(res), len(res[ridx])): continue
                cntNeighbours += 1
                for clr in range(4):
                    avg[clr] += img[(nRow >> 2)][(nCol >> 2)][clr]
            for clr in range(4):
                avg[clr] /= cntNeighbours
            res[ridx][cidx] = tuple(avg)
    return np.array(res)


def perform2x(img):
    b, g, r, a = cv2.split(img)
    height, width = len(img)*2, len(img[0])*2
    b_res = np.zeros((height, width))
    g_res = np.zeros((height, width))
    r_res = np.zeros((height, width))
    a_res = np.zeros((height, width))

    for i in range(0, height, 2):
        for j in range(0, width, 2):
            r_res[i, j] = r[i >> 1, j >> 1]
            g_res[i, j] = g[i >> 1, j >> 1]
            b_res[i, j] = b[i >> 1, j >> 1]
            a_res[i, j] = a[i >> 1, j >> 1]

    for i in range(0, height, 2):
        for j in range(1, width, 2):
            if j < 2 or j >= width - 3:
                r_res[i, j] = r_res[i, j - 1]
                g_res[i, j] = g_res[i, j - 1]
                b_res[i, j] = b_res[i, j - 1]
                a_res[i, j] = a_res[i, j - 1]
            else:
                t = 0.5
                tt = t ** 2
                ttt = t ** 3

                q1 = (-ttt + 2 * tt - t) / 2
                q2 = (3 * ttt - 5 * tt + 2) / 2
                q3 = (-3 * ttt + 4 * t + t) / 2
                q4 = (ttt - tt) / 2

                p1 = r_res[i, j - 3]
                p2 = r_res[i, j - 1]
                p3 = r_res[i, j + 1]
                p4 = r_res[i, j + 3]
                r_res[i, j] = round(p1 * q1 + p2 * q2 + p3 * q3 + p4 * q4) / 1.5

                p1 = g_res[i, j - 3]
                p2 = g_res[i, j - 1]
                p3 = g_res[i, j + 1]
                p4 = g_res[i, j + 3]
                g_res[i, j] = round(p1 * q1 + p2 * q2 + p3 * q3 + p4 * q4) / 1.5

                p1 = b_res[i, j - 3]
                p2 = b_res[i, j - 1]
                p3 = b_res[i, j + 1]
                p4 = b_res[i, j + 3]
                b_res[i, j] = round(p1 * q1 + p2 * q2 + p3 * q3 + p4 * q4) / 1.5

                p1 = a_res[i, j - 3]
                p2 = a_res[i, j - 1]
                p3 = a_res[i, j + 1]
                p4 = a_res[i, j + 3]
                a_res[i, j] = round(p1 * q1 + p2 * q2 + p3 * q3 + p4 * q4)

    for j in range(0, width, 2):
        for i in range(1, height, 2):
            if i < 2 or i >= height - 3:
                r_res[i, j] = r_res[i - 1, j]
                g_res[i, j] = g_res[i - 1, j]
                b_res[i, j] = b_res[i - 1, j]
                a_res[i, j] = a_res[i - 1, j]
            else:
                t = 0.5
                tt = t ** 2
                ttt = t ** 3

                q1 = (-ttt + 2 * tt - t) / 2
                q2 = (3 * ttt - 5 * tt + 2) / 2
                q3 = (-3 * ttt + 4 * t + t) / 2
                q4 = (ttt - tt) / 2

                p1 = r_res[i - 3, j]
                p2 = r_res[i - 1, j]
                p3 = r_res[i + 1, j]
                p4 = r_res[i + 3, j]
                r_res[i, j] = round(p1 * q1 + p2 * q2 + p3 * q3 + p4 * q4) / 1.5

                p1 = g_res[i - 3, j]
                p2 = g_res[i - 1, j]
                p3 = g_res[i + 1, j]
                p4 = g_res[i + 3, j]
                g_res[i, j] = round(p1 * q1 + p2 * q2 + p3 * q3 + p4 * q4) / 1.5

                p1 = b_res[i - 3, j]
                p2 = b_res[i - 1, j]
                p3 = b_res[i + 1, j]
                p4 = b_res[i + 3, j]
                b_res[i, j] = round(p1 * q1 + p2 * q2 + p3 * q3 + p4 * q4) / 1.5

                p1 = a_res[i - 3, j]
                p2 = a_res[i - 1, j]
                p3 = a_res[i + 1, j]
                p4 = a_res[i + 3, j]
                a_res[i, j] = round(p1 * q1 + p2 * q2 + p3 * q3 + p4 * q4)

    for i in range(1, height, 2):
        for j in range(1, width, 2):
            if j < 3 or j > width - 5 or i < 3 or i > height - 5:
                r_res[i, j] = r_res[i - 1, j]
                g_res[i, j] = g_res[i - 1, j]
                b_res[i, j] = b_res[i - 1, j]
                a_res[i, j] = a_res[i - 1, j]
            else:
                t = 0.5
                tt = t ** 2
                ttt = t ** 3

                q1 = (-ttt + 2 * tt - t) / 2
                q2 = (3 * ttt - 5 * tt + 2) / 2
                q3 = (-3 * ttt + 4 * t + t) / 2
                q4 = (ttt - tt) / 2

                p1 = r_res[i - 3, j]
                p2 = r_res[i - 1, j]
                p3 = r_res[i + 1, j]
                p4 = r_res[i + 3, j]
                r_res[i, j] = round(p1 * q1 + p2 * q2 + p3 * q3 + p4 * q4) / 1.5

                p1 = g_res[i - 3, j]
                p2 = g_res[i - 1, j]
                p3 = g_res[i + 1, j]
                p4 = g_res[i + 3, j]
                g_res[i, j] = round(p1 * q1 + p2 * q2 + p3 * q3 + p4 * q4) / 1.5

                p1 = b_res[i - 3, j]
                p2 = b_res[i - 1, j]
                p3 = b_res[i + 1, j]
                p4 = b_res[i + 3, j]
                b_res[i, j] = round(p1 * q1 + p2 * q2 + p3 * q3 + p4 * q4) / 1.5

                p1 = a_res[i - 3, j]
                p2 = a_res[i - 1, j]
                p3 = a_res[i + 1, j]
                p4 = a_res[i + 3, j]
                a_res[i, j] = round(p1 * q1 + p2 * q2 + p3 * q3 + p4 * q4)

    return cv2.merge([b_res, g_res, r_res, a_res])
