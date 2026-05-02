from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from base_core.math.functions import SphericalHarmonic
from base_core.math.models import MarkedPoints, Points, Points3D, Range


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
        

@dataclass(slots=True)
class SphericalHarmonicSuperposition:
    """
    psi(theta, phi) = sum_{l,m} c_lm Y_l^m(theta, phi)
    """

    coefficients: dict[tuple[int, int], complex] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        coefficients: Mapping[tuple[int, int], complex],
        *,
        normalize: bool = False,
    ) -> "SphericalHarmonicSuperposition":
        obj = cls(dict(coefficients))
        obj.validate()

        return obj.normalized() if normalize else obj

    def validate(self) -> None:
        for l, m in self.coefficients:
            SphericalHarmonic(l, m)

    def add(self, l: int, m: int, coefficient: complex) -> None:
        SphericalHarmonic(l, m)
        self.coefficients[(l, m)] = self.coefficients.get((l, m), 0.0) + coefficient

    def __call__(self, theta, phi) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64)
        phi = np.asarray(phi, dtype=np.float64)

        theta, phi = np.broadcast_arrays(theta, phi)
        psi = np.zeros(theta.shape, dtype=np.complex128)

        for (l, m), c_lm in self.coefficients.items():
            psi += c_lm * SphericalHarmonic(l, m)(theta, phi)

        return psi

    def probability_density(self, theta, phi) -> np.ndarray:
        return np.abs(self(theta, phi)) ** 2

    def probability_density_at_points(self, points: "Points3D") -> np.ndarray:
        """
        Evaluate |psi|^2 at 3D points.

        The points are internally projected onto the unit sphere,
        because spherical harmonics only depend on direction.
        """

        _, theta, phi = points.normalized().to_spherical()
        return self.probability_density(theta, phi)

    def norm(self) -> float:
        """
        Because the spherical harmonics are orthonormal:

            ||psi||^2 = sum |c_lm|^2
        """

        return float(np.sqrt(sum(abs(c) ** 2 for c in self.coefficients.values())))

    def normalized(self) -> "SphericalHarmonicSuperposition":
        norm = self.norm()

        if norm == 0.0:
            raise ValueError("Cannot normalize a zero superposition")

        return SphericalHarmonicSuperposition(
            {key: value / norm for key, value in self.coefficients.items()}
        )

    def normalize_inplace(self) -> None:
        norm = self.norm()

        if norm == 0.0:
            raise ValueError("Cannot normalize a zero superposition")

        for key in self.coefficients:
            self.coefficients[key] /= norm

    def probability(self, l: int, m: int) -> float:
        """
        Probability to measure the basis state |l,m>.
        """

        return float(abs(self.coefficients.get((l, m), 0.0)) ** 2)

    def copy(self) -> "SphericalHarmonicSuperposition":
        return SphericalHarmonicSuperposition(dict(self.coefficients))