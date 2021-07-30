from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Optional, List, Union, Type

import numpy as np

from src.common import Bitmap, Path, Curve, SegmentTag
from src.io.svg import SVGWriter


def cyclic(a: Union[int, float], b: Union[int, float], c: Union[int, float]) -> bool:
    if a <= c:
        return a <= b < c
    return a <= b or b < c


class TurnPolicy(Enum):
    BLACK: int = 0
    WHITE: int = 1
    LEFT: int = 2
    RIGHT: int = 3
    MINORITY: int = 4
    MAJORITY: int = 5


def interval(t: float, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))


def dorth_infty(p0: np.ndarray, p2: np.ndarray, dtype: Type = int) -> np.ndarray:
    return np.array((-np.sign(p2[1] - p0[1]), np.sign(p2[0] - p0[0])), dtype)


def ddenom(p0: np.ndarray, p2: np.ndarray) -> int:
    r: np.ndarray = dorth_infty(p0, p2)
    return r[1] * (p2[0] - p0[0]) - r[0] * (p2[1] - p0[1])


def dpara(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> int:
    x1 = p1[0] - p0[0]
    y1 = p1[1] - p0[1]
    x2 = p2[0] - p0[0]
    y2 = p2[1] - p0[1]
    return x1 * y2 - x2 * y1


def cprod(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> int:
    x1: int = p1[0] - p0[0]
    y1: int = p1[1] - p0[1]
    x2: int = p3[0] - p2[0]
    y2: int = p3[1] - p2[1]

    return x1 * y2 - x2 * y1


def iprod(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> int:
    x1: int = p1[0] - p0[0]
    y1: int = p1[1] - p0[1]
    x2: int = p2[0] - p0[0]
    y2: int = p2[1] - p0[1]
    return x1 * x2 + y1 * y2


def iprod1(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> int:
    x1: int = p1[0] - p0[0]
    y1: int = p1[1] - p0[1]
    x2: int = p3[0] - p2[0]
    y2: int = p3[1] - p2[1]

    return x1 * x2 + y1 * y2


def ddist(p: np.ndarray, q: np.ndarray) -> float:
    return np.sqrt((p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]))


def bezier(t: float, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    s: float = 1 - t
    return np.array((
        s * s * s * p0[0] + 3 * (s * s * t) * p1[0] + 3 * (t * t * s) * p2[0] + t * t * t * p3[0],
        s * s * s * p0[1] + 3 * (s * s * t) * p1[1] + 3 * (t * t * s) * p2[1] + t * t * t * p3[1])
    )


def tangent(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> float:
    A: int = cprod(p0, p1, q0, q1)
    B: int = cprod(p1, p2, q0, q1)
    C: int = cprod(p2, p3, q0, q1)

    a: int = A - 2 * B + C
    b: int = -2 * A + 2 * B
    c: int = A

    d: int = b * b - 4 * a * c
    if a == 0 or d < 0:
        return -1.0

    s: float = np.sqrt(d)

    r1: float = (-b + s) / (2 * a)
    r2: float = (-b - s) / (2 * a)
    if 0 <= r1 <= 1:
        return r1
    elif 0 <= r2 <= 1:
        return r2
    else:
        return -1.0


class Potrace:

    def __init__(self, img: np.ndarray, turdsize: int = 2, turnpolicy: TurnPolicy = TurnPolicy.MINORITY,
                 alphamax: float = 1., optcurve: bool = True, opttolerance: float = .2):

        @dataclass
        class Param:
            turdsize: int
            turnpolicy: TurnPolicy
            alphamax: float
            optcurve: bool
            opttolerance: float

        self.bm = Bitmap(img)
        self.info: Param = Param(turdsize, turnpolicy, alphamax, optcurve, opttolerance)

        def bmToPathlist(bm: Bitmap, info: Param) -> List[Path]:
            def findNext(bm1: Bitmap, point: Tuple[int, int]) -> Optional[Tuple[int, int]]:
                x, y = point
                while y < bm1.h:
                    while x < bm1.w:
                        if bm1.data[y, x]:
                            return x, y
                        x += 1
                    x = 0
                    y += 1
                return None

            def majority(bm1: Bitmap, x: int, y: int) -> int:
                for i in range(2, 5):
                    ct: int = 0
                    for a in range(-i + 1, i):
                        ct += 1 if bm1.at(x + a, y + i - 1) else -1
                        ct += 1 if bm1.at(x + i - 1, y + a - 1) else -1
                        ct += 1 if bm1.at(x + a - 1, y - i) else -1
                        ct += 1 if bm1.at(x - i, y + a) else -1
                    if ct > 0:
                        return 1
                    elif ct < 0:
                        return 0
                return 0

            def xorPath(bm1: Bitmap, path: Path) -> None:
                y1 = path.pt[0][1]
                for p in path.pt[1:]:
                    x: int = p[0]
                    y: int = p[1]
                    if y != y1:
                        minY: int = min(y1, y)
                        for j in range(x, path.maxX):
                            bm1.flip(j, minY)
                        y1 = y

            def findPath(bm: Bitmap, bm1: Bitmap, x0: int, y0: int, turnpolicy: TurnPolicy) -> Path:
                x: int = x0
                y: int = y0
                path: Path = Path()
                dirx: int = 0
                diry: int = 1

                path.sign = +1 if bm.at(x, y) else -1
                while True:
                    path.pt.append(np.array((x, y)))
                    path.minX = min(x, path.minX)
                    path.maxX = max(x, path.maxX)
                    path.minY = min(y, path.minY)
                    path.maxY = max(y, path.maxY)

                    x += dirx
                    y += diry
                    path.area -= x * diry

                    if x == x0 and y == y0:
                        break
                    l: bool = bm1.at(x + (dirx + diry - 1) // 2, y + (diry - dirx - 1) // 2)
                    r: bool = bm1.at(x + (dirx - diry - 1) // 2, y + (diry + dirx - 1) // 2)
                    if r and not l:
                        if (turnpolicy == TurnPolicy.RIGHT or
                                (turnpolicy == TurnPolicy.BLACK and path.sign == +1) or
                                (turnpolicy == TurnPolicy.WHITE and path.sign == -1) or
                                (turnpolicy == TurnPolicy.MAJORITY and majority(bm1, x, y)) or
                                (turnpolicy == TurnPolicy.MINORITY and not majority(bm1, x, y))):
                            tmp = dirx
                            dirx = -diry
                            diry = tmp
                        else:
                            tmp = dirx
                            dirx = diry
                            diry = -tmp
                    elif r:
                        tmp = dirx
                        dirx = -diry
                        diry = tmp
                    elif not l:
                        tmp = dirx
                        dirx = diry
                        diry = -tmp
                return path

            bm1: Bitmap = bm.copy()
            currentPoint: Tuple[int, int] = 0, 0
            pathlist: List[Path] = []

            while currentPoint is not None:
                x, y = currentPoint
                path = findPath(bm, bm1, x, y, info.turnpolicy)
                xorPath(bm1, path)
                """plt.imshow(bm1.data)
                plt.show()"""
                if path.area > info.turdsize:
                    pathlist.append(path)
                currentPoint = findNext(bm1, currentPoint)
            return pathlist

        def calc_sums(path: Path) -> None:
            path.x0 = path.pt[0][0]
            path.y0 = path.pt[0][1]

            s = path.sums = np.zeros((len(path.pt) + 1, 5), dtype=int)
            for i in range(len(path.pt)):
                x = path.pt[i][0] - path.x0
                y = path.pt[i][1] - path.y0
                s[i + 1] = s[i][0] + x, s[i][1] + y, s[i][2] + x * y, s[i][3] + x * x, s[i][4] + y * y

        def calc_lon(path: Path) -> None:
            n = len(path)
            pt = path.pt
            pivk = np.zeros(n, dtype=int)
            nc = np.zeros(n, dtype=int)
            ct = np.zeros(4, dtype=int)
            path.lon = np.zeros(n)
            constraint = [np.zeros((2)), np.zeros((2))]
            cur = 0, 0
            off = 0, 0
            dk = 0, 0
            foundk: bool = False
            i: int = n - 1
            k: int = 0
            while i >= 0:
                if pt[i][0] != pt[k][0] and pt[i][1] != pt[k][1]:
                    k = i + 1
                nc[i] = k
                i -= 1
            i = n - 1
            while i >= 0:
                ct[0] = ct[1] = ct[2] = ct[3] = 0
                dir: int = (3 + 3 * (pt[(i + 1) % n][0] - pt[i][0]) + (pt[(i + 1) % n][1] - pt[i][1])) // 2
                ct[dir] = ct[dir] + 1

                constraint[0][0] = 0
                constraint[0][1] = 0
                constraint[1][0] = 0
                constraint[1][1] = 0

                k = nc[i]
                k1 = i
                while True:
                    foundk = False
                    dir = (3 + 3 * np.sign(pt[k][0] - pt[k1][0]) + np.sign(pt[k][1] - pt[k1][1])) // 2
                    ct[dir] = ct[dir] + 1
                    if ct[0] and ct[1] and ct[2] and ct[3]:
                        pivk[i] = k1
                        foundk = True
                        break
                    cur = pt[k][0] - pt[i][0], pt[k][1] - pt[i][1]
                    if np.cross(constraint[0], np.asarray(cur)) < 0 or np.cross(constraint[1], np.asarray(cur)) > 0:
                        break
                    if abs(cur[0]) <= 1 and abs(cur[1]) <= 1:
                        pass
                    else:
                        off = cur[0] + (1 if (cur[1] >= 0 and (cur[1] > 0 or cur[0] < 0)) else -1), cur[1] + (
                            1 if (cur[0] <= 0 and (cur[0] < 0 or cur[1] < 0)) else -1)
                        if np.cross(constraint[0], np.asarray(off)) >= 0:
                            constraint[0][0] = off[0]
                            constraint[0][1] = off[1]

                        off = cur[0] + (1 if (cur[1] <= 0 and (cur[1] < 0 or cur[0] < 0)) else -1), cur[1] + (
                            1 if (cur[0] >= 0 and (cur[0] > 0 or cur[1] < 0)) else -1)
                        if np.cross(constraint[1], np.asarray(off)) <= 0:
                            constraint[1][0] = off[0]
                            constraint[1][1] = off[1]
                    k1 = k
                    k = nc[k1]
                    if not cyclic(k, i, k1):
                        break
                if not foundk:
                    dk = np.sign(pt[k][0] - pt[k1][0]), np.sign(pt[k][1] - pt[k1][1])
                    cur = pt[k1][0] - pt[i][0], pt[k1][1] - pt[i][1]

                    a = np.cross(constraint[0], np.asarray(cur))
                    b = np.cross(constraint[0], np.asarray(dk))
                    c = np.cross(constraint[1], np.asarray(cur))
                    d = np.cross(constraint[1], np.asarray(dk))

                    j = 10000000
                    if b < 0:
                        j = np.floor(a / -b)

                    if d > 0:
                        j = min(j, np.floor(-c / d))

                    pivk[i] = (k1 + j) % n

                i -= 1
            j = pivk[n - 1]
            path.lon[n - 1] = j
            i: int = n - 2
            while i >= 0:
                if cyclic(i + 1, pivk[i], j):
                    j = pivk[i]
                path.lon[i] = j
                i -= 1
            i = n - 1
            while cyclic((i + 1) % n, j, path.lon[i]):
                path.lon[i] = j
                i -= 1

        def best_polygon(path: Path) -> None:
            def penalty3(path: Path, i: int, j: int) -> float:
                n: int = len(path)
                pt: List[np.ndarray] = path.pt
                sums = path.sums

                r: int = 0
                if j >= n:
                    j -= n
                    r = 1
                x: int
                y: int
                x2: int
                xy: int
                y2: int
                k: int
                if r == 0:
                    x, y, xy, x2, y2 = sums[j + 1] - sums[i]
                    k = j + 1 - i
                else:
                    x, y, xy, x2, y2 = sums[j + 1] - sums[i] + sums[n]
                    k = j + 1 - i + n
                px = (pt[i][0] + pt[j][0]) / 2.0 - pt[0][0]
                py = (pt[i][1] + pt[j][1]) / 2.0 - pt[0][1]
                ey = (pt[j][0] - pt[i][0])
                ex = -(pt[j][1] - pt[i][1])

                a = ((x2 - 2 * x * px) / k + px * px)
                b = ((xy - x * py - y * px) / k + px * py)
                c = ((y2 - 2 * y * py) / k + py * py)

                s = ex * ex * a + 2 * ex * ey * b + ey * ey * c

                return np.sqrt(s)

            n = len(path)
            pen = np.zeros(n + 1, dtype=int)
            prev = np.zeros(n + 1, dtype=int)
            clip0 = np.zeros(n, dtype=int)
            clip1 = np.zeros(n + 1, dtype=int)
            seg0 = np.zeros(n + 1, dtype=int)
            seg1 = np.zeros(n + 1, dtype=int)

            for i in range(n):
                c = (path.lon[((i - 1) % n)] - 1) % n
                if c == i:
                    c = (i + 1) % n

                if c < i:
                    clip0[i] = n
                else:
                    clip0[i] = c

            j = 1
            for i in range(n):
                while j <= clip0[i]:
                    clip1[j] = i
                    j += 1
            i = 0
            j = 0
            while i < n:
                seg0[j] = i
                i = clip0[i]
                j += 1
            seg0[j] = n
            m = j

            i = n
            for j in range(m, 0, -1):
                seg1[j] = i
                i = clip1[i]
            seg1[0] = 0

            pen[0] = 0
            j = 1
            while j <= m:
                i = seg1[j]
                while i <= seg0[j]:
                    best = -1
                    k = seg0[j - 1]
                    while k >= clip1[i]:
                        thispen = penalty3(path, k, i) + pen[k]
                        if best < 0 or thispen < best:
                            prev[i] = k
                            best = thispen

                        k -= 1
                    pen[i] = best
                    i += 1
                j += 1
            path.m = m
            path.po = np.zeros(m, dtype=int)

            i = n
            j = m - 1
            while i > 0:
                i = prev[i]
                path.po[j] = i
                j -= 1

        def adjust_vertices(path: Path) -> None:

            def quadform(Q: np.ndarray, w: np.ndarray) -> float:
                v: np.ndarray = np.asarray([w[0], w[1], 1])
                sum: float = .0
                for i in range(3):
                    for j in range(3):
                        sum += v[i] * Q[i, j] * v[j]
                return sum

            def pointslope(path: Path, i: int, j: int, ctr: np.ndarray, dir: np.ndarray) -> None:
                n = len(path)
                sums = path.sums
                r = 0

                while j >= n:
                    j -= n
                    r += 1

                while i >= n:
                    i -= n
                    r -= 1

                while j < 0:
                    j += n
                    r -= 1

                while i < 0:
                    i += n
                    r += 1

                x, y, xy, x2, y2 = sums[j + 1] - sums[i] + r * sums[n]

                k = j + 1 - i + r * n

                ctr[0] = x / k
                ctr[1] = y / k

                a = (x2 - x * x / k) / k
                b = (xy - x * y / k) / k
                c = (y2 - y * y / k) / k
                lambda2 = (a + c + np.sqrt((a - c) * (a - c) + 4 * b * b)) / 2

                a -= lambda2
                c -= lambda2

                if abs(a) >= abs(c):
                    l = np.sqrt(a * a + b * b)
                    if l != 0:
                        dir[0] = -b / l
                        dir[1] = a / l

                else:
                    l = np.sqrt(c * c + b * b)
                    if l != 0:
                        dir[0] = -c / l
                        dir[1] = b / l

                if l == 0:
                    dir[0] = dir[1] = 0

            m = path.m
            po = path.po
            n = len(path)
            pt = path.pt
            x0 = path.x0
            y0 = path.y0
            ctr = np.zeros((m, 2))
            dir = np.zeros((m, 2))
            q = np.zeros((m, 3, 3))
            v = np.zeros(3)
            s = np.zeros(2)

            path.curve = Curve(m)

            for i in range(m):
                j = po[(i + 1) % m]
                j = ((j - po[i]) % n) + po[i]
                ctr[i] = np.zeros(2)
                dir[i] = np.zeros(2)
                pointslope(path, po[i], j, ctr[i], dir[i])
            for i in range(m):
                d = dir[i][0] * dir[i][0] + dir[i][1] * dir[i][1]
                if d == .0:
                    for j in range(3):
                        for k in range(3):
                            q[i, j, k] = 0
                else:
                    v[0] = dir[i][1]
                    v[1] = -dir[i][0]
                    v[2] = -v[1] * ctr[i][1] - v[0] * ctr[i][0]
                    for l in range(3):
                        for k in range(3):
                            q[i].data[l, k] = v[l] * v[k] / d
            for i in range(m):
                Q = np.zeros((3, 3))
                w = np.zeros(2)

                s[0] = pt[po[i]][0] - x0
                s[1] = pt[po[i]][1] - y0

                j = (i - 1) % m

                for l in range(3):
                    for k in range(3):
                        Q.data[l, k] = q[j, l, k] + q[i, l, k]
                while True:
                    det = Q[0, 0] * Q[1, 1] - Q[0, 1] * Q[1, 0]
                    if det != .0:
                        w[0] = (-Q[0, 2] * Q[1, 1] + Q[1, 2] * Q[0, 1]) / det
                        w[1] = (Q[0, 2] * Q[1, 0] - Q[1, 2] * Q[0, 0]) / det
                        break
                    if Q[0, 0] > Q[1, 1]:
                        v[0] = -Q[0, 1]
                        v[1] = Q[0, 0]
                    elif Q[1, 1]:
                        v[0] = -Q[1, 1]
                        v[1] = Q[1, 0]
                    else:
                        v[0] = 1
                        v[1] = 0
                    d = v[0] * v[0] + v[1] * v[1]
                    v[2] = -v[1] * s[1] - v[0] * s[0]
                    for l in range(3):
                        for k in range(3):
                            Q.data[l, k] += v[l] * v[k] / d
                dx = abs(w[0] - s[0])
                dy = abs(w[1] - s[1])
                if dx <= 0.5 and dy <= 0.5:
                    path.curve.vertex[i] = np.array((w[0] + x0, w[1] + y0))
                    continue
                min = quadform(Q, s)
                xmin = s[0]
                ymin = s[1]

                if Q[0, 0] != 0.0:
                    for z in range(2):
                        w[1] = s[1] - 0.5 + z
                        w[0] = -(Q[0, 1] * w[1] + Q[0, 2]) / Q[0, 0]
                        dx = abs(w[0] - s[0])
                        cand = quadform(Q, w)
                        if dx <= 0.5 and cand < min:
                            min = cand
                            xmin = w[0]
                            ymin = w[1]

                if Q[1, 1] != 0.0:
                    for z in range(2):
                        w[0] = s[0] - 0.5 + z
                        w[1] = -(Q[1, 0] * w[0] + Q[1, 2]) / Q[1, 1]
                        dy = abs(w[1] - s[1])
                        cand = quadform(Q, w)
                        if dy <= 0.5 and cand < min:
                            min = cand
                            xmin = w[0]
                            ymin = w[1]

                for l in range(2):
                    for k in range(2):
                        w[0] = s[0] - 0.5 + l
                        w[1] = s[1] - 0.5 + k
                        cand = quadform(Q, w)
                        if cand < min:
                            min = cand
                            xmin = w[0]
                            ymin = w[1]
                path.curve.vertex[i] = np.array((xmin + x0, ymin + y0))

        def smooth(path: Path, info: Param):
            m = path.curve.n
            curve = path.curve
            for i in range(m):
                j = (i + 1) % m
                k = (i + 2) % m
                p4 = interval(1 / 2.0, curve.vertex[k], curve.vertex[j])
                denom = ddenom(curve.vertex[i], curve.vertex[k])

                if denom != .0:
                    dd = dpara(curve.vertex[i], curve.vertex[j], curve.vertex[k]) / denom
                    dd = abs(dd)
                    alpha = (1 - 1.0 / dd) if dd > 1 else 0
                    alpha = alpha / 0.75
                else:
                    alpha = 4 / 3.0
                curve.alpha0[j] = alpha
                if alpha >= info.alphamax:
                    curve.tag[j] = SegmentTag.CORNER
                    curve.c[j, 1] = curve.vertex[j]
                    curve.c[j, 2] = p4
                else:
                    if alpha < 0.55:
                        alpha = .55
                    elif alpha > 1:
                        alpha = 1.
                    p2 = interval(0.5 + 0.5 * alpha, curve.vertex[i], curve.vertex[j])
                    p3 = interval(0.5 + 0.5 * alpha, curve.vertex[k], curve.vertex[j])
                    curve.tag[j] = SegmentTag.CURVE_TO
                    curve.c[j, 0] = p2
                    curve.c[j, 1] = p3
                    curve.c[j, 2] = p4
                curve.alpha[j] = alpha
                curve.beta[j] = .5
            curve.alphaCurve = 1

        def opticurve(path: Path, info: Param):
            @dataclass
            class Opti:
                pen: float = 0
                c: List = field(default_factory=lambda: [np.zeros(2), np.zeros(2)])
                t: float = 0
                s: float = 0
                alpha: float = 0

            def opti_penalty(path: Path, i: int, j: int, res: Opti, opttolerance, convc, areac) -> int:
                m = path.curve.n
                curve = path.curve
                vertex = curve.vertex

                if i == j:
                    return 1

                k = i
                i1 = (i + 1) % m
                k1 = (k + 1) % m
                conv = convc[k1]

                if conv == 0:
                    return 1
                d = ddist(vertex[i], vertex[i1])
                k = k1
                while k != j:
                    k1 = (k + 1) % m
                    k2 = (k + 2) % m
                    if convc[k1] != conv:
                        return 1
                    if np.sign(cprod(vertex[i], vertex[i1], vertex[k1], vertex[k2])) != conv:
                        return 1
                    if iprod1(vertex[i], vertex[i1], vertex[k1], vertex[k2]) < d * ddist(vertex[k1],
                                                                                         vertex[k2]) * -0.999847695156:
                        return 1
                    k = k1
                p0 = curve.c[i % m, 2].copy()
                p1 = vertex[(i + 1) % m].copy()
                p2 = vertex[j % m].copy()
                p3 = curve.c[j % m, 2].copy()

                area = areac[j] - areac[i]
                area -= dpara(vertex[0], curve.c[i, 2], curve.c[j, 2]) / 2
                if i >= j:
                    area += areac[m]
                A1 = dpara(p0, p1, p2)
                A2 = dpara(p0, p1, p3)
                A3 = dpara(p0, p2, p3)

                A4 = A1 + A3 - A2
                if A2 == A1:
                    return 1
                t = A3 / (A3 - A4)
                s = A2 / (A2 - A1)
                A = A2 * t / 2.0

                if A == 0:
                    return 1
                R = area / A
                alpha = 2 - np.sqrt(4 - R / 0.3)

                res.c[0] = interval(t * alpha, p0, p1)
                res.c[1] = interval(s * alpha, p3, p2)
                res.alpha = alpha
                res.t = t
                res.s = s

                p1 = res.c[0].copy()
                p2 = res.c[1].copy()

                res.pen = 0
                k = (i + 1) % m
                while k != j:
                    k1 = (k + 1) % m
                    t = tangent(p0, p1, p2, p3, vertex[k], vertex[k1])
                    if t < -.5:
                        return 1
                    pt = bezier(t, p0, p1, p2, p3)
                    d = ddist(vertex[k], vertex[k1])
                    if d == 0.0:
                        return 1

                    d1 = dpara(vertex[k], vertex[k1], pt) / d
                    if abs(d1) > opttolerance:
                        return 1

                    if iprod(vertex[k], vertex[k1], pt) < 0 or iprod(vertex[k1], vertex[k], pt) < 0:
                        return 1

                    res.pen += d1 * d1
                    k = k1

                k = i
                while k != j:
                    k1 = (k + 1) % m
                    t = tangent(p0, p1, p2, p3, curve.c[k, 2], curve.c[k1, 2])
                    if t < -0.5:
                        return 1

                    pt = bezier(t, p0, p1, p2, p3)
                    d = ddist(curve.c[k, 2], curve.c[k1, 2])
                    if d == 0.0:
                        return 1

                    d1 = dpara(curve.c[k, 2], curve.c[k1, 2], pt) / d
                    d2 = dpara(curve.c[k, 2], curve.c[k1, 2], vertex[k1]) / d
                    d2 *= 0.75 * curve.alpha[k1]
                    if d2 < 0:
                        d1 = -d1
                        d2 = -d2

                    if d1 < d2 - opttolerance:
                        return 1

                    if d1 < d2:
                        res.pen += (d1 - d2) * (d1 - d2)

                    k = k1
                return 0

            curve = path.curve
            m = curve.n
            vert = curve.vertex
            pt = np.zeros(m + 1, dtype=int)
            pen = np.zeros(m + 1, dtype=float)
            len = np.zeros(m + 1, dtype=int)
            opt = np.full(m + 1, None, dtype=Opti)
            o = Opti()

            convc = np.zeros(m, dtype=int)
            areac = np.zeros(m + 1, dtype=float)
            for i in range(m):
                if curve.tag[i] == SegmentTag.CURVE_TO:
                    convc[i] = np.sign(dpara(vert[(i - 1) % m], vert[i], vert[(i + 1) % m]))
            area = 0.0
            areac[0] = 0.0
            p0 = curve.vertex[0]
            for i in range(m):
                i1 = (i + 1) % m
                if curve.tag[i1] == SegmentTag.CURVE_TO:
                    alpha = curve.alpha[i1]
                    area += 0.3 * alpha * (4 - alpha) * dpara(curve.c[i, 2], vert[i1], curve.c[i1, 2]) / 2
                    area += dpara(p0, curve.c[i, 2], curve.c[i1, 2]) / 2
                areac[i + 1] = area
            pt[0] = -1
            pen[0] = 0
            len[0] = 0

            for j in range(1, m + 1):
                pt[j] = j - 1
                pen[j] = pen[j - 1]
                len[j] = len[j - 1] + 1

                i = j - 2
                while i >= 0:
                    r = opti_penalty(path, i, (j) % m, o, info.opttolerance, convc,
                                     areac)
                    if r:
                        break
                    if len[j] > len[i] + 1 or (len[j] == len[i] + 1 and pen[j] > pen[i] + o.pen):
                        pt[j] = i
                        pen[j] = pen[i] + o.pen
                        len[j] = len[i] + 1
                        opt[j] = o
                        o = Opti()
                    i -= 1
            om = len[m]
            ocurve = Curve(om)
            s = np.zeros(om)
            t = np.zeros(om)

            j = m
            i = om - 1
            while i >= 0:
                if pt[j] == j - 1:
                    ocurve.tag[i] = curve.tag[(j) % m]
                    ocurve.c[i, 0] = curve.c[(j) % m, 0]
                    ocurve.c[i, 1] = curve.c[(j) % m, 1]
                    ocurve.c[i, 2] = curve.c[(j) % m, 2]
                    ocurve.vertex[i] = curve.vertex[(j) % m]
                    ocurve.alpha[i] = curve.alpha[(j) % m]
                    ocurve.alpha0[i] = curve.alpha0[(j) % m]
                    ocurve.beta[i] = curve.beta[(j) % m]
                    s[i] = t[i] = 1.0
                else:
                    ocurve.tag[i] = SegmentTag.CURVE_TO
                    ocurve.c[i, 0] = opt[j].c[0]
                    ocurve.c[i, 1] = opt[j].c[1]
                    ocurve.c[i, 2] = curve.c[(j) % m, 2]
                    ocurve.vertex[i] = interval(opt[j].s, curve.c[(j) % m, 2],
                                                vert[(j) % m])
                    ocurve.alpha[i] = opt[j].alpha
                    ocurve.alpha0[i] = opt[j].alpha
                    s[i] = opt[j].s
                    t[i] = opt[j].t
                j = pt[j]
                i -= 1

            for i in range(om):
                i1 = (i + 1) % om
                ocurve.beta[i] = s[i] / (s[i] + t[i1])
            ocurve.alphacurve = 1
            path.curve = ocurve

        self.pathlist = bmToPathlist(self.bm, self.info)

        for path in self.pathlist:
            calc_sums(path)
            calc_lon(path)
            best_polygon(path)
            adjust_vertices(path)
            if path.sign == -1:
                path.curve.vertex = path.curve.vertex[::-1]
                path.sign = 1
            smooth(path, self.info)
            if self.info.optcurve:
                opticurve(path, self.info)

    def to_svg(self, output: Union[str, Path], size: float = 1) -> None:
        SVGWriter().write(self.bm, self.pathlist, output, size=size)
