from dataclasses import dataclass
import math
from typing import Generic, Optional, Protocol, Self, TypeVar
import numpy as np

from base_core.math.enums import AngleUnit


class Angle(float):
    def __new__(cls, value: float, unit: AngleUnit = AngleUnit.RAD):
        radians = float(value) * unit.value
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


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def distance_from_center(self) -> float:
        return math.hypot(self.x, self.y)

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