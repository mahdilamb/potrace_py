from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Union, Tuple

import numpy as np


class SegmentTag(Enum):
    CURVE_TO: int = 1
    CORNER: int = 2


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

    def index(self, i: int) -> Tuple[int, int]:
        y: int = int(i / self.w)
        return i - y * self.w, y

    def flip(self, x: int, y: int) -> None:
        if not self.range_check(x, y):
            return
        self.data[y, x] = ~ self.data[y, x]

    def copy(self) -> 'Bitmap':
        return Bitmap(self.data.copy(), check_input=False)


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
        self.c = np.zeros((self.n, 3, 2))
        self.vertex = np.zeros((self.n, 2))
        self.alpha = np.zeros(self.n)
        self.alpha0 = np.zeros(self.n)
        self.beta = np.zeros(self.n)


@dataclass
class Path:
    area: int = 0
    sign: int = 0
    pt: List[np.ndarray] = field(default_factory=list)
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


class Writer(ABC):
    file_ext: str = None

    def write(self, bm: Bitmap, pathlist: List[Path], output: Union[str, Path], **kwargs):
        raise NotImplementedError()
