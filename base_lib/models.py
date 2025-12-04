# domain/models.py
from dataclasses import dataclass
from enum import Enum
from math import hypot
from pathlib import Path
from typing import Generic, Optional, Protocol, TypeVar, Self
from base_lib.functions import gaussian
import numpy as np



class Prefix(float, Enum):
    NONE = 1.0   
    PICO   = 1e-12
    NANO   = 1e-9
    MICRO  = 1e-6
    MILLI   = 1e-3
    CENTI   = 1e-2
    KILO   = 1e3
    MEGA   = 1e6
    GIGA   = 1e9
    TERA   = 1e12

class AngleUnit(float, Enum):
    RAD = 1.0          
    DEG = np.pi / 180.0    

class PlotColor(str, Enum):
    BLUE   = "b"
    RED    = "r"
    GREEN  = "g"
    PURPLE = "purple"
    GRAY   = "tab:gray"
    

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


class Length(float):
    def __new__(cls, value: float, prefix: Prefix = Prefix.NONE):
        meters = float(value) * prefix.value
        return super().__new__(cls, meters)

    def value(self, prefix: Prefix = Prefix.NONE) -> float:
        return float(self) / prefix.value

class Time(float):
    def __new__(cls, value: float, prefix: Prefix = Prefix.NONE):
        seconds = float(value) * prefix.value
        return super().__new__(cls, seconds)

    def value(self, prefix: Prefix = Prefix.NONE) -> float:
        return float(self) / prefix.value


class Frequency(float):
    def __new__(cls, value: float, prefix: Prefix = Prefix.NONE):
        hz = float(value) * prefix.value
        return super().__new__(cls, hz)

    def value(self, prefix: Prefix = Prefix.NONE) -> float:
        return float(self) / prefix.value


class Angle(float):
    def __new__(cls, value: float, unit: AngleUnit = AngleUnit.RAD):
        radians = float(value) * unit.value
        return super().__new__(cls, radians)

    @property
    def Rad(self) -> float:
        return float(self)               

    @property
    def Deg(self) -> float:
        return float(self) / AngleUnit.DEG.value



@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def distance_from_center(self) -> float:
        return hypot(self.x, self.y)

    def subtract(self, point: "Point") -> None:
        self._set_x(self.x - point.x)
        self._set_y(self.y - point.y)

    def rotate(self, angle: Angle, center: Optional["Point"] = None) -> None:
       
        if center is None:
            center = Point(0.0, 0.0)

        cx, cy = center.x, center.y

        tx = self.x - cx
        ty = self.y - cy

        cos_a = np.cos(angle.Rad)
        sin_a = np.sin(angle.Rad)

        rx = tx * cos_a - ty * sin_a
        ry = tx * sin_a + ty * cos_a

        self._set_x(rx + cx)
        self._set_y(ry + cy)
    
    def affine_transform(self, transform_parameter: float) -> None:
        self._set_x(transform_parameter * self.x)

    def _set_x(self, value: float) -> None:
        object.__setattr__(self, "x", value)
    
    def _set_y(self, value: float) -> None:
        object.__setattr__(self, "y", value)




@dataclass
class GaussianFitResult:
    amplitude: float
    center: float
    sigma: float
    offset: float

    amplitude_err: float | None = None
    center_err: float | None = None
    sigma_err: float | None = None
    offset_err: float | None = None

    covariance: np.ndarray | None = None

    def get_curve(self, x):
        """
        Evaluate the fitted Gaussian for arbitrary x values.
        """
        x = np.asarray(x)
        return gaussian(x, self.amplitude, self.center, self.sigma, self.offset)