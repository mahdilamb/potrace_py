from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Optional, List, Union, Type

import numpy as np


def cyclic(a: Union[int, float], b: Union[int, float], c: Union[int, float]) -> bool:
    if a <= c:
        return a <= b < c
    return a <= b or b < c


class Point:

    def __init__(self, x: Union[int, float] = 0, y: Union[int, float] = 0, dtype: Type = int):
        self._coord: np.ndarray = np.array((x, y), dtype=dtype)

    @property
    def x(self) -> Union[int, float]:
        return self._coord[0]

    @property
    def y(self) -> Union[int, float]:
        return self._coord[1]

    @x.setter
    def x(self, x: Union[int, float]) -> None:
        self._coord[0] = x

    @y.setter
    def y(self, y: Union[int, float]) -> None:
        self._coord[1] = y

    def copy(self) -> 'Point':
        return Point(self.x, self.y, self._coord.dtype)

    def cross(self, p2: 'Point') -> Union[int, float]:
        return np.cross(self._coord, p2._coord)

    def __str__(self) -> str:
        return "Point {{x={x}, y={y}}}".format(x=self.x, y=self.y)

    def __repr__(self) -> str:
        return "({x}, {y})".format(x=self.x, y=self.y)

    def __eq__(self, other: 'Point') -> bool:
        return isinstance(other, Point) and np.array_equal(self._coord, other._coord)


class SegmentTag(Enum):
    CURVE_TO: int = 1
    CORNER: int = 2


class TurnPolicy(Enum):
    BLACK: int = 0
    WHITE: int = 1
    LEFT: int = 2
    RIGHT: int = 3
    MINORITY: int = 4
    MAJORITY: int = 5


_rgb_gray_scale: np.ndarray = np.asarray([0.2126, 0.7153, 0.0721])


class Bitmap:
    def __init__(self, img: np.ndarray, check_input: bool = True):
        if check_input:
            if img.ndim not in [2, 3]:
                raise Exception("Image must be grayscale or RBG(A)")
            if img.ndim == 3:
                if img.shape[-1] == 4:
                    img = img[..., 0:3]
                img = np.dot(img, _rgb_gray_scale)
            if img.dtype != bool:
                img = img < 128
        self.data = img
        self.h, self.w = img.shape
        self.size = img.size

    def range_check(self, x: int, y: int) -> bool:
        return 0 <= x < self.w and 0 <= y < self.h

    def at(self, x: int, y: int) -> bool:
        return self.range_check(x, y) and self.data[y, x]

    def index(self, i: int) -> Point:
        y: int = int(i / self.w)
        return Point(i - y * self.w, y)

    def flip(self, x: int, y: int) -> None:
        if not self.range_check(x, y):
            return
        self.data[y, x] = ~ self.data[y, x]

    def copy(self) -> 'Bitmap':
        return Bitmap(self.data.copy(), check_input=False)


@dataclass
class Quad:
    data: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))

    def at(self, x: int, y: int) -> float:
        return self.data[x, y]


def quadform(Q: Quad, w: Point) -> float:
    v: np.ndarray = np.asarray([w.x, w.y, 1])
    sum: float = .0
    for i in range(3):
        for j in range(3):
            sum += v[i] * Q.at(i, j) * v[j]
    return sum


def interval(t: float, a: Point, b: Point) -> Point:
    return Point(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y), float)


def dorth_infty(p0: Point, p2: Point, dtype: Type = int) -> Point:
    return Point(-np.sign(p2.y - p0.y), np.sign(p2.x - p0.x), dtype)


def ddenom(p0: Point, p2: Point) -> int:
    r: Point = dorth_infty(p0, p2)
    return r.y * (p2.x - p0.x) - r.x * (p2.y - p0.y)


def dpara(p0: Point, p1: Point, p2: Point) -> int:
    x1 = p1.x - p0.x;
    y1 = p1.y - p0.y;
    x2 = p2.x - p0.x;
    y2 = p2.y - p0.y;
    return x1 * y2 - x2 * y1


def cprod(p0: Point, p1: Point, p2: Point, p3: Point) -> int:
    x1: int = p1.x - p0.x
    y1: int = p1.y - p0.y
    x2: int = p3.x - p2.x
    y2: int = p3.y - p2.y

    return x1 * y2 - x2 * y1


def iprod(p0: Point, p1: Point, p2: Point) -> int:
    x1: int = p1.x - p0.x
    y1: int = p1.y - p0.y
    x2: int = p2.x - p0.x
    y2: int = p2.y - p0.y
    return x1 * x2 + y1 * y2


def iprod1(p0: Point, p1: Point, p2: Point, p3: Point) -> int:
    x1: int = p1.x - p0.x
    y1: int = p1.y - p0.y
    x2: int = p3.x - p2.x
    y2: int = p3.y - p2.y

    return x1 * x2 + y1 * y2


def ddist(p: Point, q: Point) -> float:
    return np.sqrt((p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y))


def bezier(t: float, p0: Point, p1: Point, p2: Point, p3: Point) -> Point:
    s: float = 1 - t
    return Point(
        s * s * s * p0.x + 3 * (s * s * t) * p1.x + 3 * (t * t * s) * p2.x + t * t * t * p3.x,
        s * s * s * p0.y + 3 * (s * s * t) * p1.y + 3 * (t * t * s) * p2.y + t * t * t * p3.y,
        float
    )


def tangent(p0: Point, p1: Point, p2: Point, p3: Point, q0: Point, q1: Point) -> float:
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


@dataclass
class Curve:
    n: int
    alphaCurve: float = 0
    tag: Optional[np.ndarray] = None
    c: Optional[np.ndarray] = None
    vertex: Optional[np.ndarray] = None
    alpha: Optional[np.ndarray] = None
    alpha0: Optional[np.ndarray] = None
    beta: Optional[np.ndarray] = None

    def __post_init__(self):
        self.tag = np.zeros(self.n, dtype=SegmentTag)
        self.c = np.zeros((self.n, 3), dtype=Point)
        self.vertex = np.zeros(self.n, dtype=Point)
        self.alpha = np.zeros(self.n)
        self.alpha0 = np.zeros(self.n)
        self.beta = np.zeros(self.n)


@dataclass
class Path:
    area: int = 0
    sign: int = 0
    pt: List[Point] = field(default_factory=list)
    x0: int = -1
    y0: int = -1
    minX: int = 100000
    minY: int = 100000
    maxX: int = -1
    maxY: int = -1
    sums: Optional[np.ndarray] = None
    lon: Optional[np.ndarray] = None
    po: Optional[np.ndarray] = None
    m: int = -1
    curve: Optional[Curve] = None

    def __len__(self):
        return len(self.pt)


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
                y1 = path.pt[0].y
                for p in path.pt[1:]:
                    x: int = p.x
                    y: int = p.y
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
                    path.pt.append(Point(x, y))
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
            path.x0 = path.pt[0].x
            path.y0 = path.pt[0].y

            s = path.sums = np.zeros((len(path.pt) + 1, 5), dtype=int)
            for i in range(len(path.pt)):
                x = path.pt[i].x - path.x0
                y = path.pt[i].y - path.y0
                s[i + 1] = s[i][0] + x, s[i][1] + y, s[i][2] + x * y, s[i][3] + x * x, s[i][4] + y * y

        def calc_lon(path: Path) -> None:
            n = len(path)
            pt = path.pt
            pivk = np.zeros(n, dtype=int)
            nc = np.zeros(n, dtype=int)
            ct = np.zeros(4, dtype=int)
            path.lon = np.zeros(n)
            constraint = [Point(), Point()]
            cur = 0, 0
            off = 0, 0
            dk = 0, 0
            foundk: bool = False
            i: int = n - 1
            k: int = 0
            while i >= 0:
                if pt[i].x != pt[k].x and pt[i].y != pt[k].y:
                    k = i + 1
                nc[i] = k
                i -= 1
            i = n - 1
            while i >= 0:
                ct[0] = ct[1] = ct[2] = ct[3] = 0
                dir: int = (3 + 3 * (pt[(i + 1) % n].x - pt[i].x) + (pt[(i + 1) % n].y - pt[i].y)) // 2
                ct[dir] = ct[dir] + 1

                constraint[0].x = 0
                constraint[0].y = 0
                constraint[1].x = 0
                constraint[1].y = 0

                k = nc[i]
                k1 = i
                while True:
                    foundk = False
                    dir = (3 + 3 * np.sign(pt[k].x - pt[k1].x) + np.sign(pt[k].y - pt[k1].y)) // 2
                    ct[dir] = ct[dir] + 1
                    if ct[0] and ct[1] and ct[2] and ct[3]:
                        pivk[i] = k1
                        foundk = True
                        break
                    cur = pt[k].x - pt[i].x, pt[k].y - pt[i].y
                    if constraint[0].cross(Point(*cur)) < 0 or constraint[1].cross(Point(*cur)) > 0:
                        break
                    if abs(cur[0]) <= 1 and abs(cur[1]) <= 1:
                        pass
                    else:
                        off = cur[0] + (1 if (cur[1] >= 0 and (cur[1] > 0 or cur[0] < 0)) else -1), cur[1] + (
                            1 if (cur[0] <= 0 and (cur[0] < 0 or cur[1] < 0)) else -1)
                        if constraint[0].cross(Point(*off)) >= 0:
                            constraint[0].x = off[0]
                            constraint[0].y = off[1]

                        off = cur[0] + (1 if (cur[1] <= 0 and (cur[1] < 0 or cur[0] < 0)) else -1), cur[1] + (
                            1 if (cur[0] >= 0 and (cur[0] > 0 or cur[1] < 0)) else -1)
                        if constraint[1].cross(Point(*off)) <= 0:
                            constraint[1].x = off[0]
                            constraint[1].y = off[1]
                    k1 = k
                    k = nc[k1]
                    if not cyclic(k, i, k1):
                        break
                if not foundk:
                    dk = np.sign(pt[k].x - pt[k1].x), np.sign(pt[k].y - pt[k1].y)
                    cur = pt[k1].x - pt[i].x, pt[k1].y - pt[i].y

                    a = constraint[0].cross(Point(*cur))
                    b = constraint[0].cross(Point(*dk))
                    c = constraint[1].cross(Point(*cur))
                    d = constraint[1].cross(Point(*dk))

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
                pt: List[Point] = path.pt
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
                px = (pt[i].x + pt[j].x) / 2.0 - pt[0].x
                py = (pt[i].y + pt[j].y) / 2.0 - pt[0].y
                ey = (pt[j].x - pt[i].x)
                ex = -(pt[j].y - pt[i].y)

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
            def pointslope(path: Path, i: int, j: int, ctr: int, dir: Point) -> None:
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

                ctr.x = x / k
                ctr.y = y / k

                a = (x2 - x * x / k) / k
                b = (xy - x * y / k) / k
                c = (y2 - y * y / k) / k
                lambda2 = (a + c + np.sqrt((a - c) * (a - c) + 4 * b * b)) / 2

                a -= lambda2
                c -= lambda2

                if abs(a) >= abs(c):
                    l = np.sqrt(a * a + b * b)
                    if l != 0:
                        dir.x = -b / l
                        dir.y = a / l

                else:
                    l = np.sqrt(c * c + b * b)
                    if l != 0:
                        dir.x = -c / l
                        dir.y = b / l

                if l == 0:
                    dir.x = dir.y = 0

            m = path.m
            po = path.po
            n = len(path)
            pt = path.pt
            x0 = path.x0
            y0 = path.y0
            ctr = np.zeros(m, dtype=Point)
            dir = np.zeros(m, dtype=Point)
            q = np.zeros(m, dtype=Quad)
            v = np.zeros(3)
            s = Point(dtype=float)

            path.curve = Curve(m)

            for i in range(m):
                j = po[(i + 1) % m]
                j = ((j - po[i]) % n) + po[i]
                ctr[i] = Point(dtype=float)
                dir[i] = Point(dtype=float)
                pointslope(path, po[i], j, ctr[i], dir[i])
            for i in range(m):
                q[i] = Quad()
                d = dir[i].x * dir[i].x + dir[i].y * dir[i].y
                if d == .0:
                    for j in range(3):
                        for k in range(3):
                            q[i].data[j, k] = 0
                else:
                    v[0] = dir[i].y
                    v[1] = -dir[i].x
                    v[2] = -v[1] * ctr[i].y - v[0] * ctr[i].x
                    for l in range(3):
                        for k in range(3):
                            q[i].data[l, k] = v[l] * v[k] / d
            for i in range(m):
                Q = Quad()
                w = Point(dtype=float)

                s.x = pt[po[i]].x - x0
                s.y = pt[po[i]].y - y0

                j = (i - 1) % m

                for l in range(3):
                    for k in range(3):
                        Q.data[l, k] = q[j].at(l, k) + q[i].at(l, k)
                while True:
                    det = Q.at(0, 0) * Q.at(1, 1) - Q.at(0, 1) * Q.at(1, 0)
                    if det != .0:
                        w.x = (-Q.at(0, 2) * Q.at(1, 1) + Q.at(1, 2) * Q.at(0, 1)) / det
                        w.y = (Q.at(0, 2) * Q.at(1, 0) - Q.at(1, 2) * Q.at(0, 0)) / det
                        break
                    if Q.at(0, 0) > Q.at(1, 1):
                        v[0] = -Q.at(0, 1)
                        v[1] = Q.at(0, 0)
                    elif Q.at(1, 1):
                        v[0] = -Q.at(1, 1)
                        v[1] = Q.at(1, 0)
                    else:
                        v[0] = 1
                        v[1] = 0
                    d = v[0] * v[0] + v[1] * v[1]
                    v[2] = -v[1] * s.y - v[0] * s.x
                    for l in range(3):
                        for k in range(3):
                            Q.data[l, k] += v[l] * v[k] / d
                dx = abs(w.x - s.x)
                dy = abs(w.y - s.y)
                if dx <= 0.5 and dy <= 0.5:
                    path.curve.vertex[i] = Point(w.x + x0, w.y + y0, float)
                    continue
                min = quadform(Q, s)
                xmin = s.x
                ymin = s.y

                if Q.at(0, 0) != 0.0:
                    for z in range(2):
                        w.y = s.y - 0.5 + z
                        w.x = -(Q.at(0, 1) * w.y + Q.at(0, 2)) / Q.at(0, 0)
                        dx = abs(w.x - s.x)
                        cand = quadform(Q, w)
                        if dx <= 0.5 and cand < min:
                            min = cand
                            xmin = w.x
                            ymin = w.y

                if Q.at(1, 1) != 0.0:
                    for z in range(2):
                        w.x = s.x - 0.5 + z
                        w.y = -(Q.at(1, 0) * w.x + Q.at(1, 2)) / Q.at(1, 1)
                        dy = abs(w.y - s.y)
                        cand = quadform(Q, w)
                        if dy <= 0.5 and cand < min:
                            min = cand
                            xmin = w.x
                            ymin = w.y

                for l in range(2):
                    for k in range(2):
                        w.x = s.x - 0.5 + l
                        w.y = s.y - 0.5 + k
                        cand = quadform(Q, w)
                        if cand < min:
                            min = cand
                            xmin = w.x
                            ymin = w.y
                path.curve.vertex[i] = Point(xmin + x0, ymin + y0, float)

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
                c: List = field(default_factory=lambda: [Point(dtype=float), Point(dtype=float)])
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
        def get_svg(size: float = 1., opt_type: str = None) -> str:
            def to_fixed(x: float, dp: int = 3) -> str:
                return str(round(x, dp))

            def path(curve: Curve) -> str:
                def cubic_bezier(curve: Curve, i: int) -> str:
                    b = 'C ' + to_fixed(curve.c[i, 0].x * size) + ' ' + to_fixed(curve.c[i, 0].y * size) + ','
                    b += to_fixed(curve.c[i, 1].x * size) + ' ' + to_fixed(curve.c[i, 1].y * size) + ','
                    b += to_fixed(curve.c[i, 2].x * size) + ' ' + to_fixed(curve.c[i, 2].y * size) + ' '
                    return b

                def segment(curve: Curve, i: int) -> str:
                    s = 'L ' + to_fixed(curve.c[i, 1].x * size) + ' ' + to_fixed(curve.c[i, 1].y * size) + ' '
                    s += to_fixed(curve.c[i, 2].x * size) + ' ' + to_fixed(curve.c[i, 2].y * size) + ' '
                    return s

                n = curve.n
                p = 'M' + to_fixed(curve.c[(n - 1), 2].x * size) + ' ' + to_fixed(curve.c[(n - 1), 2].y * size) + ' '
                for i in range(n):
                    if curve.tag[i] == SegmentTag.CURVE_TO:
                        p += cubic_bezier(curve, i)
                    elif curve.tag[i] == SegmentTag.CORNER:
                        p += segment(curve, i)
                return p

            w = self.bm.w * size
            h = self.bm.h * size
            svg = '<svg id="svg" version="1.1" width="' + str(w) + '" height="' + str(
                h) + '" xmlns="http://www.w3.org/2000/svg">'
            svg += '<path d="'
            for i in range(len(self.pathlist)):
                c = self.pathlist[i].curve
                svg += path(c)
            strokec: str
            fillc: str
            fillrule: str
            if opt_type == "curve":
                strokec = "black"
                fillc = "none"
                fillrule = ''
            else:
                strokec = "none"
                fillc = "black"
                fillrule = ' fill-rule="evenodd"'
            svg += '" stroke="' + strokec + '" fill="' + fillc + '"' + fillrule + '/></svg>'
            return svg

        with open(output, 'w') as f:
            f.write(get_svg(size=size))


if __name__ == "__main__":
    from skimage.io import imread

    Potrace(imread("yao.jpg")).to_svg("yao.svg")
