from __future__ import annotations
from dataclasses import dataclass, fields
import math

import numpy as np
import numpy.typing as npt
from typing import Generic, Optional, Protocol, Self, TypeVar
from base_core.framework.serialization.serde import PrimitiveSerde, Primitive

from base_core.math.enums import AngleUnit, CartesianAxis, XZ

FloatArray = npt.NDArray[np.float64]
IntArray = np.ndarray

class SupportsOrdering(Protocol):
    def __lt__(self, other: Self, /) -> bool: ...
    def __le__(self, other: Self, /) -> bool: ...
    def __gt__(self, other: Self, /) -> bool: ...
    def __ge__(self, other: Self, /) -> bool: ...


T = TypeVar("T", bound=SupportsOrdering)

class Angle(float, PrimitiveSerde):
    """
    Float subclass storing the value internally in radians.
    Primitive representation: a single float (radians).
    """

    def __new__(cls, value: float, unit: AngleUnit = AngleUnit.RAD, wrap: bool = True):
        # In your real code, set default unit=AngleUnit.RAD directly.
        if unit is None:
            raise ValueError("AngleUnit must be provided (use AngleUnit.RAD as default in real code)")

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

    # --- serialization ---
    def to_primitive(self) -> float:
        return float(self)

    @classmethod
    def from_primitive(cls, v: Primitive) -> "Angle":
        # Stored value is radians; avoid wrapping to preserve exact stored value.
        return cls(float(v), unit=AngleUnit.RAD, wrap=False)

@dataclass(frozen=True)
class Range(Generic[T], PrimitiveSerde):
    """
    Generic range type.
    Primitive representation uses dataclass field names automatically
    (no hardcoded "min"/"max").
    """

    min: T
    max: T

    def __post_init__(self):
        if self.min > self.max:
            raise ValueError("min cannot be greater than max")

    def is_in_range(self, value: T, *, inclusive: bool = True) -> bool:
        return (self.min <= value <= self.max) if inclusive else (self.min < value < self.max)

    # --- serialization (no hardcoded "min"/"max") ---
    def to_primitive(self) -> dict[str, Primitive]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_primitive(cls, v: Primitive) -> "Range":
        return cls(**{f.name: v[f.name] for f in fields(cls)})

@dataclass(slots=True)
class Point(PrimitiveSerde):
    """
    Minimal 2D point.

    Intended role:
      - small/occasional object (config values, single centers, UI/debug)
      - heavy / bulk point processing should use `Points` (numpy arrays)
    """

    x: float
    y: float

    # Keep only what you actually use on single points.
    def subtract(self, point: "Point") -> None:
        """In-place translation by subtracting another point."""
        self.x -= float(point.x)
        self.y -= float(point.y)

    # --- serialization ---
    def to_primitive(self) -> dict[str, float]:
        return {f.name: float(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_primitive(cls, v: Primitive) -> "Point":
        return cls(**{f.name: float(v[f.name]) for f in fields(cls)})


@dataclass(slots=True)
class Points:
    """
    Fast container for many 2D points (Structure-of-Arrays):
      - x: 1D numpy array of x coordinates
      - y: 1D numpy array of y coordinates
    """

    x: FloatArray
    y: FloatArray

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=np.float64)

        if self.x.ndim != 1 or self.y.ndim != 1:
            raise ValueError("x and y must be 1D arrays")
        if self.x.shape != self.y.shape:
            raise ValueError("x and y must have the same shape")

        if not self.x.flags["C_CONTIGUOUS"]:
            self.x = np.ascontiguousarray(self.x)
        if not self.y.flags["C_CONTIGUOUS"]:
            self.y = np.ascontiguousarray(self.y)

    def __len__(self) -> int:
        return int(self.x.size)

    @classmethod
    def from_xy(cls, x, y) -> "Points":
        """Build Points from any array-like x/y inputs."""
        return cls(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))

    @classmethod
    def from_polar(cls, r, phi) -> "Points":
        """
        Build 2D points from polar coordinates.

        Convention:
            x = r cos(phi)
            y = r sin(phi)

        phi is in radians.
        """

        r_arr = np.asarray(r, dtype=np.float64)
        phi_arr = np.asarray(phi, dtype=np.float64)

        r_b, phi_b = np.broadcast_arrays(r_arr, phi_arr)

        x = r_b * np.cos(phi_b)
        y = r_b * np.sin(phi_b)

        return cls(x.ravel(), y.ravel())

    @classmethod
    def from_pointlist(cls, pts: list["Point"]) -> "Points":
        """Helper for migration: list[Point] -> Points (still O(N) Python iteration)."""
        n = len(pts)
        x = np.fromiter((p.x for p in pts), dtype=np.float64, count=n)
        y = np.fromiter((p.y for p in pts), dtype=np.float64, count=n)
        return cls(x, y)

    def to_pointlist(self) -> list["Point"]:
        """Expensive: creates N Python objects. Use only if really needed."""
        return [Point(float(x), float(y)) for x, y in zip(self.x, self.y, strict=True)]

    def to_polar(self) -> tuple[FloatArray, FloatArray]:
        """
        Convert x/y points to polar coordinates.

        Returns
        -------
        r:
            radius sqrt(x^2 + y^2)

        phi:
            azimuth angle in [0, 2*pi)
        """

        r = np.hypot(self.x, self.y)
        phi = np.mod(np.arctan2(self.y, self.x), 2.0 * np.pi)

        return r, phi

    # -------- vectorized ops (in-place) --------

    def subtract(self, p: "Point") -> None:
        self.x -= float(p.x)
        self.y -= float(p.y)

    def affine_transform(self, transform_parameter: float) -> None:
        self.x *= float(transform_parameter)

    def rotate(self, angle: "Angle", center: Optional["Point"] = None) -> None:
        c = math.cos(angle.Rad)
        s = math.sin(angle.Rad)

        cx = 0.0 if center is None else float(center.x)
        cy = 0.0 if center is None else float(center.y)

        tx = self.x - cx
        ty = self.y - cy

        self.x = tx * c - ty * s + cx
        self.y = tx * s + ty * c + cy

        if not self.x.flags["C_CONTIGUOUS"]:
            self.x = np.ascontiguousarray(self.x)
        if not self.y.flags["C_CONTIGUOUS"]:
            self.y = np.ascontiguousarray(self.y)

    def distance_from_center(self) -> FloatArray:
        return np.hypot(self.x, self.y)

    def filter_by_distance_range(self, r: "Range[float]", *, inclusive: bool = True) -> "Points":
        """Return NEW Points containing only points with radius in r."""
        d = np.hypot(self.x, self.y)
        if inclusive:
            m = (d >= float(r.min)) & (d <= float(r.max))
        else:
            m = (d > float(r.min)) & (d < float(r.max))
        return Points(self.x[m], self.y[m])

    def copy(self) -> "Points":
        return Points(self.x.copy(), self.y.copy())

    def as_array(self) -> FloatArray:
        """
        Return points as array with shape (N, 2).

        This creates a new array.
        """

        return np.column_stack((self.x, self.y))

@dataclass(slots=True)
class MarkedPoints(Points):
    marker: IntArray

    def __post_init__(self) -> None:
        super(MarkedPoints, self).__post_init__()

        self.marker = np.asarray(self.marker, dtype=np.int64)

        if self.marker.ndim != 1:
            raise ValueError("marker must be 1D")
        if self.marker.shape != self.x.shape:
            raise ValueError("marker must have the same shape as x and y")

        if not self.marker.flags["C_CONTIGUOUS"]:
            self.marker = np.ascontiguousarray(self.marker)

    @classmethod
    def from_arrays(cls, marker, x, y) -> "MarkedPoints":
        return cls(
            x=np.asarray(x, dtype=np.float64),
            y=np.asarray(y, dtype=np.float64),
            marker=np.asarray(marker, dtype=np.int64),
        )

    def filter_by_distance_range(self, r: Range[float], *, inclusive: bool = True) -> "MarkedPoints":
        d = np.hypot(self.x, self.y)
        if inclusive:
            m = (d >= float(r.min)) & (d <= float(r.max))
        else:
            m = (d > float(r.min)) & (d < float(r.max))
        return MarkedPoints(self.x[m], self.y[m], self.marker[m])

    def filter_by_mask(self, mask: np.ndarray) -> "MarkedPoints":
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.x.shape:
            raise ValueError("mask must have same shape as data")
        return MarkedPoints(self.x[mask], self.y[mask], self.marker[mask])
    
    def unique_observed_markers(self) -> np.ndarray:
        return np.unique(self.marker)

    @property
    def n_observed_markers(self) -> int:
        if len(self.marker) == 0:
            return 0
        return int(np.unique(self.marker).size)

    @property
    def marker_min(self) -> int | None:
        if len(self.marker) == 0:
            return None
        return int(np.min(self.marker))

    @property
    def marker_max(self) -> int | None:
        if len(self.marker) == 0:
            return None
        return int(np.max(self.marker))

    @property
    def n_marker_span(self) -> int:
        """
        Total number of marker slots assuming consecutive integer markers.
        Includes skipped/empty markers.
        """
        if len(self.marker) == 0:
            return 0
        return int(np.max(self.marker) - np.min(self.marker) + 1)

    def avg_points_per_observed_marker(self) -> float:
        n = self.n_observed_markers
        return 0.0 if n == 0 else len(self) / n

    def avg_points_per_marker(self) -> float:
        """
        Average points per marker INCLUDING skipped empty markers,
        assuming markers are consecutive integer frame IDs.
        """
        n = self.n_marker_span
        return 0.0 if n == 0 else len(self) / n
    
    def append_points(self,pts: "Points") -> None:
        self.x = np.append(self.x,pts.x)
        self.y = np.append(self.y,pts.y)
        
@dataclass(slots=True)
class Points3D:
    x: FloatArray
    y: FloatArray
    z: FloatArray

    def __post_init__(self) -> None:
        self.x = np.ascontiguousarray(self.x, dtype=np.float64)
        self.y = np.ascontiguousarray(self.y, dtype=np.float64)
        self.z = np.ascontiguousarray(self.z, dtype=np.float64)

        if self.x.ndim != 1 or self.y.ndim != 1 or self.z.ndim != 1:
            raise ValueError("x, y and z must be 1D arrays")
        if not (self.x.shape == self.y.shape == self.z.shape):
            raise ValueError("x, y and z must have the same shape")

    def __len__(self) -> int:
        return int(self.x.size)

    @classmethod
    def from_xyz(cls, x, y, z) -> "Points3D":
        return cls(x, y, z)

    @classmethod
    def from_spherical(cls, r, theta, phi) -> "Points3D":
        """
        Physics convention:
            x = r sin(theta) cos(phi)
            y = r sin(theta) sin(phi)
            z = r cos(theta)

        theta: polar angle from +z axis
        phi: azimuth angle in xy plane
        """
        r, theta, phi = np.broadcast_arrays(
            np.asarray(r, dtype=np.float64),
            np.asarray(theta, dtype=np.float64),
            np.asarray(phi, dtype=np.float64),
        )

        sin_theta = np.sin(theta)

        return cls(
            (r * sin_theta * np.cos(phi)).ravel(),
            (r * sin_theta * np.sin(phi)).ravel(),
            (r * np.cos(theta)).ravel(),
        )

    def to_spherical(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        """
        Returns:
            r, theta, phi

        theta in [0, pi]
        phi in [0, 2*pi)
        """
        r = np.sqrt(self.x**2 + self.y**2 + self.z**2)

        theta = np.zeros_like(r)
        mask = r > 0.0
        theta[mask] = np.arccos(np.clip(self.z[mask] / r[mask], -1.0, 1.0))

        phi = np.mod(np.arctan2(self.y, self.x), 2.0 * np.pi)

        return r, theta, phi

    def coordinate(self, axis: CartesianAxis) -> FloatArray:
        axis = axis.require_single()

        if axis is CartesianAxis.X:
            return self.x
        if axis is CartesianAxis.Y:
            return self.y
        if axis is CartesianAxis.Z:
            return self.z

        raise ValueError(f"Unsupported axis: {axis}")

    def project_to_plane(self, plane: CartesianAxis = XZ) -> "Points":
        a, b = plane.require_plane().axes()
        return Points(self.coordinate(a).copy(), self.coordinate(b).copy())

    def radius(self) -> FloatArray:
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> "Points3D":
        r = self.radius()
        if np.any(r == 0.0):
            raise ValueError("Cannot normalize points containing zero vectors")

        return Points3D(self.x / r, self.y / r, self.z / r)

    def normalize_inplace(self) -> None:
        r = self.radius()
        if np.any(r == 0.0):
            raise ValueError("Cannot normalize points containing zero vectors")

        self.x /= r
        self.y /= r
        self.z /= r

    def copy(self) -> "Points3D":
        return Points3D(self.x.copy(), self.y.copy(), self.z.copy())

    def as_array(self) -> FloatArray:
        return np.column_stack((self.x, self.y, self.z))
           
