from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto, unique
from typing import Optional, List

import numpy as np


@unique
class TurnPolicy(IntEnum):
    BLACK: int = 0
    WHITE: int = 1
    LEFT: int = 2
    RIGHT: int = 3
    MINORITY: int = 4
    MAJORITY: int = 5

    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def argparse(s):
        try:
            return TurnPolicy[s.upper()]
        except KeyError:
            return s


@unique
class SegmentTag(Enum):
    CURVE_TO: int = auto()
    CORNER: int = auto()


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
    origin: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=int))
    min: int = field(default_factory=lambda: np.array((100000, 100000), dtype=int))
    max: int = field(default_factory=lambda: np.array((-1, -1), dtype=int))
    sums: Optional[np.ndarray] = None
    lon: Optional[np.ndarray] = None
    po: Optional[np.ndarray] = None
    m: int = -1
    curve: Optional[Curve] = None

    def __len__(self):
        return len(self.pt)
