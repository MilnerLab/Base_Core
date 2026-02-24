from dataclasses import dataclass
import math
from typing import Generic, Optional, Protocol, Self, TypeVar
import numpy as np

from base_core.math.enums import AngleUnit


class Angle(float):
    def __new__(cls, value: float, unit: AngleUnit = AngleUnit.RAD, wrap: bool = True):
        radians = float(value) * unit.value
        if wrap:
            radians = cls._wrap_to_minus_pi_pi(radians)
        return super().__new__(cls, radians)

    @staticmethod
    def _wrap_to_minus_pi_pi(rad: float) -> float:
        two_pi = 2 * math.pi
        return (rad + math.pi) % two_pi - math.pi

    @property
    def Rad(self) -> float:
        return float(self)

    @property
    def Deg(self) -> float:
        return float(self) / AngleUnit.DEG.value


@dataclass(slots=True)
class Point:
    x: float
    y: float

    def distance_from_center(self) -> float:
        return math.hypot(self.x, self.y)

    def subtract(self, point: "Point") -> None:
        self.x -= point.x
        self.y -= point.y

    def affine_transform(self, transform_parameter: float) -> None:
        self.x *= transform_parameter

    # Low-level: rotate with precomputed cos/sin (fast in loops)
    def rotate_cs(self, cos_a: float, sin_a: float, center: Optional["Point"] = None) -> None:
        cx = center.x if center is not None else 0.0
        cy = center.y if center is not None else 0.0

        tx = self.x - cx
        ty = self.y - cy

        self.x = tx * cos_a - ty * sin_a + cx
        self.y = tx * sin_a + ty * cos_a + cy

    # Convenience wrapper (but slower if called for every point)
    def rotate(self, angle, center: Optional["Point"] = None) -> None:
        c = math.cos(angle.Rad)
        s = math.sin(angle.Rad)
        self.rotate_cs(c, s, center)

        
        
class SupportsOrdering(Protocol):
    def __lt__(self, other: Self, /) -> bool: ...
    def __le__(self, other: Self, /) -> bool: ...
    def __gt__(self, other: Self, /) -> bool: ...
    def __ge__(self, other: Self, /) -> bool: ...

T = TypeVar("T", bound=SupportsOrdering)

@dataclass(frozen=True)
class Range(Generic[T]):
    min: T
    max: T

    def __post_init__(self):
        if self.min > self.max:
            raise ValueError("min cannot be greater than max")

    def is_in_range(self, value: T, *, inclusive: bool = True) -> bool:
        if inclusive:
            return self.min <= value <= self.max
        else:
            return self.min < value < self.max
        
@dataclass(frozen=True)
class Histogram2D():
    matrix: np.ndarray = None
    x_edges: np.ndarray = None
    y_edges: np.ndarray = None
    def __init__(self,points: list[Point], x_bins: int, y_bins: int, x_range: Optional[Range[float]] = None, y_range: Optional[Range[float]] = None) -> None:
        self.points = points
        self.x_bins = x_bins
        self.y_bins = y_bins
        p_x = np.array([p.x for p in points])
        p_y = np.array([p.y for p in points])
        self.x_range = x_range if x_range is not None else Range(min(p_x), max(p_x))
        self.y_range = y_range if y_range is not None else Range(min(p_y), max(p_y))
        self.compute_histogram(p_x,p_y)
        
    def compute_histogram(self,p_x,p_y) -> None:
        self.matrix, self.x_edges, self.y_edges = np.histogram2d(p_x, p_y, bins=[self.x_bins, self.y_bins], range=[[self.x_range.min, self.x_range.max], [self.y_range.min, self.y_range.max]])