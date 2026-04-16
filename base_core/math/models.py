from __future__ import annotations
from dataclasses import dataclass, fields
import math

import numpy as np
import numpy.typing as npt
from typing import Generic, Optional, Protocol, Self, TypeVar
from base_core.framework.serialization.serde import PrimitiveSerde, Primitive

from base_core.math.enums import AngleUnit

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
    def from_pointlist(cls, pts: list["Point"]) -> "Points":
        """Helper for migration: list[Point] -> Points (still O(N) Python iteration)."""
        n = len(pts)
        x = np.fromiter((p.x for p in pts), dtype=np.float64, count=n)
        y = np.fromiter((p.y for p in pts), dtype=np.float64, count=n)
        return cls(x, y)

    def to_pointlist(self) -> list["Point"]:
        """Expensive: creates N Python objects. Use only if really needed."""
        return [Point(float(x), float(y)) for x, y in zip(self.x, self.y, strict=True)]

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
    
    def unique_markers(self) -> np.ndarray:
        return np.unique(self.marker)

    @property
    def n_markers(self) -> int:
        if len(self.marker) == 0:
            return 0
        return int(np.unique(self.marker).size)

    def avg_points_per_marker(self) -> float:
        n = self.n_markers
        return 0.0 if n == 0 else len(self) / n
        
@dataclass(frozen=True)
class Histogram2D():
    matrix: np.ndarray = None
    x_edges: np.ndarray = None
    y_edges: np.ndarray = None
    
    @classmethod
    def compute_histogram(cls, points: Points, x_bins: int = 400, y_bins: int = 400, bin_size: float = 0.4, radial_range: Range[float] = Range(0,60)) -> "Histogram2D":
        
        if x_bins is None and y_bins is not None | y_bins is None and x_bins is not None: 
            raise TypeError("x_bins and y_bins must either both be None or both be integers.")
        
        radial_width = radial_range.max - radial_range.min
        if bin_size > 2*radial_width: 
            raise ValueError("Bin size cannot be larger than the region of interest.")
        #x_0 , y_0 = center.x, center.y
        
        p_x = points.x
        p_y = points.y    
        

        x_bins = 2*radial_width/bin_size if x_bins is None else x_bins
        y_bins = 2*radial_width/bin_size if y_bins is None else y_bins
        #matrix, x_edges, y_edges = np.histogram2d(p_x, p_y, bins=[x_bins, y_bins], range=[[x_range.min, x_range.max], [y_range.min, y_range.max]])
        matrix, x_edges, y_edges = np.histogram2d(p_x, p_y, bins=[x_bins, y_bins])
        return cls(matrix,x_edges,y_edges)


@dataclass(frozen=True)
class AngularCovariance:
    matrix: np.ndarray
    theta1_edges: np.ndarray
    theta2_edges: np.ndarray
    n_frames: int

    @classmethod
    def compute_covariance(
        cls,
        hits: MarkedPoints,
        angle_bins: int = 90,
        radial_range: Range[float] | None = None,
        binary_per_frame: bool = False,
    ) -> "AngularCovariance":
        """
        Compute angular covariance in the same spirit as the old ThetaToAngCov code:
        - theta wrapped to [0, 2*pi)
        - per-marker angular histograms
        - covariance formed from centered histograms
        - normalization by total number of hits (not by number of markers)
        - matrix rolled by 90 degrees
        - diagonal set to zero

        Notes
        -----
        This reproduces the *style* of the older code, but uses robust grouping by
        marker instead of assuming the data are already sorted and contiguous by frame.
        """

        if len(hits) == 0:
            raise ValueError("No hits available.")

        x = hits.x
        y = hits.y
        marker = hits.marker

        if radial_range is not None:
            r = np.hypot(x, y)
            mask = (r >= float(radial_range.min)) & (r <= float(radial_range.max))
            x = x[mask]
            y = y[mask]
            marker = marker[mask]

        if x.size == 0:
            raise ValueError("No hits left after radial filter.")

        # same as TidyTheta
        theta = np.mod(np.arctan2(y, x), 2.0 * np.pi)

        # bin edges over [0, 2*pi)
        bin_edges = np.linspace(0.0, 2.0 * np.pi, angle_bins + 1)

        # angle bin index
        bin_idx = np.digitize(theta, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, angle_bins - 1)

        # robust marker grouping
        unique_markers, inv = np.unique(marker, return_inverse=True)
        n_markers = unique_markers.size

        if n_markers == 0:
            raise ValueError("No markers found.")

        # ThDist[marker_index, angle_bin]
        th_dist = np.zeros((n_markers, angle_bins), dtype=np.float32)

        if binary_per_frame:
            # optional occupancy version; old code used counts, so leave False for exact match
            pairs = np.column_stack((inv, bin_idx))
            unique_pairs = np.unique(pairs, axis=0)
            th_dist[unique_pairs[:, 0], unique_pairs[:, 1]] = 1.0
        else:
            np.add.at(th_dist, (inv, bin_idx), 1.0)

        # mean histogram over markers
        th_bar = th_dist.mean(axis=0)

        # centered distributions
        th_diff = th_dist - th_bar

        # same structure as einsum in old code
        ang_cov_unscaled = np.einsum("fi,fj->ij", th_diff, th_diff)

        # IMPORTANT:
        # old code normalized by total number of hits, not by number of markers
        ang_cov = ang_cov_unscaled / len(theta)

        # enforce symmetry (usually already symmetric, but kept to mimic old behavior)
        i_lower = np.tril_indices(angle_bins, -1)
        ang_cov[i_lower] = ang_cov.T[i_lower]

        # shift by 90 degrees in both axes
        shift = angle_bins // 4
        ang_cov = np.roll(ang_cov, shift=shift, axis=0)
        ang_cov = np.roll(ang_cov, shift=shift, axis=1)

        # remove autovariance diagonal
        np.fill_diagonal(ang_cov, 0.0)

        return cls(
            matrix=ang_cov,
            theta1_edges=bin_edges,
            theta2_edges=bin_edges.copy(),
            n_frames=n_markers,
        )