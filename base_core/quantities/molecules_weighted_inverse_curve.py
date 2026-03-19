
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Mapping, Optional, TypeAlias

import numpy as np
from numpy import pi

from base_core.math.enums import CartesianAxis
from base_core.quantities.enums import Prefix
from base_core.quantities.constants import (
    BOLTZMANN_J_K,
    EPS0_F_M,
    PLANCK_H_J_S,
    SPEED_OF_LIGHT,
)

from base_core.quantities.models import Frequency, InverseLength, Temperature
from base_core.quantities.specific_models import AtomicMass, Intensity, PolarizabilityVolume

TensorRow: TypeAlias = tuple[
    PolarizabilityVolume,
    PolarizabilityVolume,
    PolarizabilityVolume,
]

PolarizabilityTensor: TypeAlias = tuple[
    TensorRow,
    TensorRow,
    TensorRow,
]


# ----------------- Small domain containers -----------------

@dataclass(frozen=True, slots=True)
class RotationalBD:
    """
    Rotational constant B and centrifugal distortion D.

    Canonical internal representation: Frequency (Hz).
    Convenience constructors allow supplying wavenumbers (InverseLength).
    """
    B: Optional[Frequency] = None
    D: Optional[Frequency] = None
    reference: str = ""
    notes: str = ""


@dataclass(frozen=True, slots=True)
class Polarizability:
    """
    Polarizability stored as 'polarizability volume' (m^3) using your model.
    """
    tensor: Optional[PolarizabilityTensor] = None
    bond_axis: Optional[CartesianAxis] = None
    iso: Optional[PolarizabilityVolume] = None
    aniso: Optional[PolarizabilityVolume] = None
    reference: str = ""
    notes: str = ""


@dataclass(frozen=True, slots=True)
class SpinnabilityCurve:
    time_s: np.ndarray
    centrifuge_frequency_hz: np.ndarray
    raman_frequency_hz: np.ndarray
    intensity_W_per_m2: np.ndarray
    sigma: np.ndarray
    p_J: np.ndarray
    mean_jj1: np.ndarray
    median_J: np.ndarray
    peak_J: np.ndarray
    effective_J_from_mean: np.ndarray
    average_eff_inverse_I_hz: np.ndarray


# ----------------- helper methods ----------------------

def _level_rotation_frequency_hz(J: int, B_hz: float, D_hz: float) -> float:
    """
    Local rotational frequency of a non-rigid linear rotor,
    obtained from the slope d(E/h)/dJ.

        E_J / h = B J(J+1) - D J^2 (J+1)^2

    Returns frequency in Hz.
    """
    j = float(J)
    b_eff = B_hz - 2.0 * D_hz * j * (j + 1.0)
    return (2.0 * j + 1.0) * b_eff


def _find_current_J_from_level_frequency(
    centrifuge_frequency_hz: float,
    B_hz: float,
    D_hz: float,
    j_max: int = 100_000,
) -> int:
    """
    Infer the current J by matching the centrifuge rotation frequency
    to the local rotor frequency f_rot(J) = d(E/h)/dJ.

    This is a semiclassical / local-slope picture, not a discrete
    Raman-step picture.
    """
    if centrifuge_frequency_hz < 0.0:
        raise ValueError("centrifuge_frequency_hz must be >= 0.")
    if B_hz <= 0.0:
        raise ValueError("B_hz must be > 0.")
    if D_hz < 0.0:
        raise ValueError("D_hz must be >= 0.")

    best_J = 0
    best_err = float("inf")
    previous_f = -1.0

    for J in range(j_max + 1):
        f_rot = _level_rotation_frequency_hz(J, B_hz, D_hz)

        if f_rot <= 0.0:
            break

        # stop after passing the maximum accessible frequency
        if previous_f >= 0.0 and f_rot < previous_f:
            break

        err = abs(f_rot - centrifuge_frequency_hz)
        if err < best_err:
            best_err = err
            best_J = J

        previous_f = f_rot

    return best_J


def _cos_theta_up_coeff(J: int, M: int) -> float:
    numerator = (J + 1) ** 2 - M ** 2
    denominator = (2 * J + 1) * (2 * J + 3)
    if numerator <= 0 or denominator <= 0:
        return 0.0
    return float(np.sqrt(numerator / denominator))


def _cos_theta_down_coeff(J: int, M: int) -> float:
    if J <= 0:
        return 0.0

    numerator = J ** 2 - M ** 2
    denominator = (2 * J - 1) * (2 * J + 1)
    if numerator <= 0 or denominator <= 0:
        return 0.0
    return float(np.sqrt(numerator / denominator))


@lru_cache(maxsize=None)
def _rotating_frame_static_operators(
    j_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build basis-independent rotating-frame operators once for a given j_max.

    Returns
    -------
    basis_J : ndarray[int]
        J value for each basis state.
    jj1_diag : ndarray[float]
        J(J+1) for each basis state.
    cos2_theta : ndarray[float]
        Matrix of cos^2(theta) in the field-fixed basis |J,M>_x.
    jz_in_field_basis : ndarray[float]
        Matrix of J_z / hbar in the same basis.
    initial_state : ndarray[float]
        Basis vector corresponding to |0,0>.
    """
    if j_max < 0:
        raise ValueError("j_max must be >= 0.")

    basis: list[tuple[int, int]] = []
    for J in range(j_max + 1):
        for M in range(-J, J + 1):
            basis.append((J, M))

    state_to_index = {state: i for i, state in enumerate(basis)}
    n = len(basis)

    basis_J = np.array([J for J, _ in basis], dtype=int)
    jj1_diag = basis_J.astype(float) * (basis_J.astype(float) + 1.0)

    cos2_theta = np.zeros((n, n), dtype=float)
    jz_in_field_basis = np.zeros((n, n), dtype=float)

    for col, (J, M) in enumerate(basis):
        a_j = _cos_theta_up_coeff(J, M)
        b_j = _cos_theta_down_coeff(J, M)

        # diagonal term
        cos2_theta[col, col] = a_j * a_j + b_j * b_j

        # J -> J+2
        if J + 2 <= j_max and abs(M) <= J + 2:
            row = state_to_index[(J + 2, M)]
            cos2_theta[row, col] = a_j * _cos_theta_up_coeff(J + 1, M)

        # J -> J-2
        if J - 2 >= 0 and abs(M) <= J - 2:
            row = state_to_index[(J - 2, M)]
            cos2_theta[row, col] = b_j * _cos_theta_down_coeff(J - 1, M)

        # J_z in field-fixed basis = J_x / hbar in the usual z-basis
        if M + 1 <= J:
            row = state_to_index[(J, M + 1)]
            jz_in_field_basis[row, col] = 0.5 * np.sqrt(J * (J + 1) - M * (M + 1))

        if M - 1 >= -J:
            row = state_to_index[(J, M - 1)]
            jz_in_field_basis[row, col] = 0.5 * np.sqrt(J * (J + 1) - M * (M - 1))

    # Numerical cleanup
    cos2_theta = 0.5 * (cos2_theta + cos2_theta.T)
    jz_in_field_basis = 0.5 * (jz_in_field_basis + jz_in_field_basis.T)

    initial_state = np.zeros(n, dtype=float)
    initial_state[state_to_index[(0, 0)]] = 1.0

    return basis_J, jj1_diag, cos2_theta, jz_in_field_basis, initial_state


def _gaussian_intensity_profile(
    peak_intensity_W_per_m2: float,
    ramp_duration_s: float,
    time_s: np.ndarray,
    sigma_fraction: float,
) -> np.ndarray:
    if peak_intensity_W_per_m2 < 0.0:
        raise ValueError("peak_intensity must be >= 0.")
    if ramp_duration_s <= 0.0:
        raise ValueError("ramp_duration_s must be > 0.")
    if sigma_fraction <= 0.0:
        raise ValueError("gaussian_sigma_fraction must be > 0.")

    center = 0.5 * ramp_duration_s
    sigma = sigma_fraction * ramp_duration_s
    return peak_intensity_W_per_m2 * np.exp(-0.5 * ((time_s - center) / sigma) ** 2)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Weighted median of a discrete distribution.

    Parameters
    ----------
    values
        Support values, assumed sorted in ascending order.
    weights
        Non-negative weights on the same support.
    """
    if values.ndim != 1 or weights.ndim != 1:
        raise ValueError("values and weights must be 1D arrays.")
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length.")

    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        return float(values[0])

    cumulative = np.cumsum(weights)
    idx = int(np.searchsorted(cumulative, 0.5 * total_weight, side="left"))
    idx = min(idx, len(values) - 1)
    return float(values[idx])


# ----------------- Base molecule class -----------------

@dataclass(frozen=True, slots=True)
class Molecule:
    key: str
    name: str
    formula: str
    cas: Optional[str] = None

    mass: Optional[AtomicMass] = None

    gasphase: RotationalBD = field(default_factory=RotationalBD)
    droplet: RotationalBD = field(default_factory=RotationalBD)

    polarizability: Polarizability = field(default_factory=Polarizability)

    tags: tuple[str, ...] = ()
    notes: str = ""
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("Molecule.key must be non-empty")
        if not self.name.strip():
            raise ValueError("Molecule.name must be non-empty")
        if not self.formula.strip():
            raise ValueError("Molecule.formula must be non-empty")

        if not isinstance(self.meta, MappingProxyType):
            object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))

    def calculate_spinnability(
        self,
        intensity: Intensity,
        temperature: Temperature | None,
        centrifuge_frequency: Frequency,
        angular_acceleration_rad_per_s2: float,
    ) -> float:
        """
        Fast semiclassical estimate using the nearest free-rotor J.

        This is the older, cheaper field-free approximation.
        """
        if angular_acceleration_rad_per_s2 <= 0.0:
            raise ValueError("angular_acceleration_rad_per_s2 must be > 0.")
        if self.polarizability.aniso is None:
            raise ValueError("polarizability.aniso must be set.")
        if self.droplet.B is None or self.droplet.D is None:
            raise ValueError("droplet B and D must be set.")

        intensity_si = float(intensity)
        cfg_freq_hz = float(centrifuge_frequency)
        B_hz = float(self.droplet.B)
        D_hz = float(self.droplet.D)

        if intensity_si < 0.0:
            raise ValueError("intensity must be >= 0.")
        if cfg_freq_hz < 0.0:
            raise ValueError("centrifuge_frequency must be >= 0.")
        if B_hz <= 0.0:
            raise ValueError("B must be > 0.")
        if D_hz < 0.0:
            raise ValueError("D must be >= 0.")

        U0_J = intensity_si * self.polarizability.aniso.to_SI() / (2.0 * SPEED_OF_LIGHT * EPS0_F_M)

        J = _find_current_J_from_level_frequency(
            centrifuge_frequency_hz=cfg_freq_hz,
            B_hz=B_hz,
            D_hz=D_hz,
        )

        eff_inverse_I = B_hz - 2.0 * D_hz * J * (J + 1.0)

        if eff_inverse_I <= 0.0:
            raise ValueError(
                "B_eff_hz became <= 0. The requested frequency is at or beyond the centrifugal wall."
            )

        I_eff_kg_m2 = PLANCK_H_J_S / (8.0 * pi**2 * eff_inverse_I)
        sigma0 = 2.0 * U0_J / (pi * I_eff_kg_m2 * angular_acceleration_rad_per_s2)

        if temperature is None:
            return sigma0

        temperature_K = float(temperature)
        if temperature_K <= 0.0:
            raise ValueError("temperature must be > 0.")

        sigmaT = sigma0 * U0_J / (0.5 * BOLTZMANN_J_K * temperature_K)
        return min(sigma0, sigmaT)

    def calculate_spinnability_curve(
        self,
        peak_intensity: Intensity,
        temperature: Temperature | None,
        angular_acceleration_rad_per_s2: float,
        ramp_duration_s: float,
        *,
        n_points: int = 120,
        j_max: int = 12,
        gaussian_sigma_fraction: float = 1.0 / 6.0,
    ) -> SpinnabilityCurve:
        """
        Compute the whole dressed-state spinnability curve in one pass.

        Model
        -----
        - rotating frame
        - linear centrifuge frequency ramp: Omega(t) = alpha * t
        - Gaussian intensity envelope with peak `peak_intensity`
        - adiabatic branch following starting from |0,0>
        - spinnability based on the probability-weighted average of the local
          effective inverse inertia

              <B_eff> = sum_J p_J [B - 2 D J(J+1)]

          rather than on a single representative J value

        Returns
        -------
        SpinnabilityCurve
            Contains time axis, centrifuge frequency axis, Raman axis (= 2*f_cfg),
            intensity profile, sigma(t), the full field-free J distribution p_J(t),
            <J(J+1)>(t), the weighted median J, the modal J, the old
            J_eff reconstructed from <J(J+1)>, and the averaged effective
            inverse inertia <B_eff>.
        """
        if angular_acceleration_rad_per_s2 <= 0.0:
            raise ValueError("angular_acceleration_rad_per_s2 must be > 0.")
        if ramp_duration_s <= 0.0:
            raise ValueError("ramp_duration_s must be > 0.")
        if n_points < 2:
            raise ValueError("n_points must be >= 2.")
        if self.polarizability.aniso is None:
            raise ValueError("polarizability.aniso must be set.")
        if self.droplet.B is None or self.droplet.D is None:
            raise ValueError("droplet B and D must be set.")

        peak_intensity_si = float(peak_intensity)
        if peak_intensity_si < 0.0:
            raise ValueError("peak_intensity must be >= 0.")

        if temperature is not None:
            temperature_K = float(temperature)
            if temperature_K <= 0.0:
                raise ValueError("temperature must be > 0.")
        else:
            temperature_K = None

        B_hz = float(self.droplet.B)
        D_hz = float(self.droplet.D)
        if B_hz <= 0.0:
            raise ValueError("B must be > 0.")
        if D_hz < 0.0:
            raise ValueError("D must be >= 0.")

        delta_alpha_si = self.polarizability.aniso.to_SI()

        time_s = np.linspace(0.0, ramp_duration_s, n_points)
        centrifuge_frequency_hz = angular_acceleration_rad_per_s2 * time_s / (2.0 * pi)
        intensity_W_per_m2 = _gaussian_intensity_profile(
            peak_intensity_W_per_m2=peak_intensity_si,
            ramp_duration_s=ramp_duration_s,
            time_s=time_s,
            sigma_fraction=gaussian_sigma_fraction,
        )

        basis_J, jj1_diag, cos2_theta, jz_in_field_basis, state_vector = _rotating_frame_static_operators(j_max)

        jj1_diag_sq = jj1_diag ** 2
        h0_diag_hz = B_hz * jj1_diag - D_hz * jj1_diag_sq
        h0_hz = np.diag(h0_diag_hz)

        j_values = np.arange(j_max + 1, dtype=float)
        jj1_by_J = j_values * (j_values + 1.0)

        sigma_values = np.full(n_points, np.nan, dtype=float)
        p_J_values = np.zeros((n_points, j_max + 1), dtype=float)
        mean_jj1_values = np.full(n_points, np.nan, dtype=float)
        median_J_values = np.full(n_points, np.nan, dtype=float)
        peak_J_values = np.full(n_points, np.nan, dtype=float)
        effective_J_values = np.full(n_points, np.nan, dtype=float)
        average_eff_inverse_I_values = np.full(n_points, np.nan, dtype=float)

        previous_vector = state_vector.copy()

        for i in range(n_points):
            current_U0_J = (
                intensity_W_per_m2[i] * delta_alpha_si / (2.0 * SPEED_OF_LIGHT * EPS0_F_M)
            )
            current_u_hz = current_U0_J / PLANCK_H_J_S
            current_f_cfg_hz = centrifuge_frequency_hz[i]

            hamiltonian_hz = h0_hz - current_u_hz * cos2_theta - current_f_cfg_hz * jz_in_field_basis

            _, eigenvectors = np.linalg.eigh(hamiltonian_hz)

            overlaps = np.abs(eigenvectors.T.conj() @ previous_vector)
            best_index = int(np.argmax(overlaps))
            current_vector = eigenvectors[:, best_index]

            overlap = np.vdot(previous_vector, current_vector)
            if np.real(overlap) < 0.0:
                current_vector = -current_vector

            previous_vector = current_vector

            populations = np.abs(current_vector) ** 2
            p_J = np.bincount(basis_J, weights=populations, minlength=j_max + 1)
            p_J_values[i, :] = p_J

            mean_jj1 = float(np.dot(p_J, jj1_by_J))
            mean_jj1_values[i] = mean_jj1
            effective_J_values[i] = 0.5 * (-1.0 + np.sqrt(1.0 + 4.0 * mean_jj1))

            median_J = _weighted_median(j_values, p_J)
            median_J_values[i] = median_J
            peak_J_values[i] = float(np.argmax(p_J))

            eff_inverse_I_by_J = B_hz - 2.0 * D_hz * jj1_by_J
            eff_inverse_I = float(np.dot(p_J, eff_inverse_I_by_J))
            average_eff_inverse_I_values[i] = eff_inverse_I
            if eff_inverse_I <= 0.0:
                continue

            I_eff_kg_m2 = PLANCK_H_J_S / (8.0 * pi**2 * eff_inverse_I)
            sigma0 = 2.0 * current_U0_J / (pi * I_eff_kg_m2 * angular_acceleration_rad_per_s2)

            if temperature_K is None:
                sigma_values[i] = sigma0
            else:
                sigmaT = sigma0 * current_U0_J / (0.5 * BOLTZMANN_J_K * temperature_K)
                sigma_values[i] = min(sigma0, sigmaT)

        return SpinnabilityCurve(
            time_s=time_s,
            centrifuge_frequency_hz=centrifuge_frequency_hz,
            raman_frequency_hz=2.0 * centrifuge_frequency_hz,
            intensity_W_per_m2=intensity_W_per_m2,
            sigma=sigma_values,
            p_J=p_J_values,
            mean_jj1=mean_jj1_values,
            median_J=median_J_values,
            peak_J=peak_J_values,
            effective_J_from_mean=effective_J_values,
            average_eff_inverse_I_hz=average_eff_inverse_I_values,
        )


# ----------------- Fixed-value holders -----------------

class CS2(Molecule):
    def __init__(self) -> None:
        super().__init__(
            key="cs2",
            name="Carbon disulfide",
            formula="CS2",
            mass=AtomicMass.from_u(76.14),
            gasphase=RotationalBD(
                B=InverseLength(0.110, Prefix.CENTI).to_frequency(),
                D=InverseLength(12, Prefix.NANO).to_frequency(),
                reference="MacPhail-Bartley thesis (example)",
            ),
            droplet=RotationalBD(
                B=Frequency(730, Prefix.MEGA),
                D=Frequency(1.2, Prefix.MEGA),
                reference="https://doi.org/10.1103/PhysRevLett.125.013001",
            ),
            polarizability=Polarizability(
                aniso=PolarizabilityVolume.from_angstrom3(8.7),
            ),
            tags=("linear", "droplets"),
        )


class OCS(Molecule):
    def __init__(self) -> None:
        super().__init__(
            key="ocs",
            name="Carbonyl sulfide",
            formula="OCS",
            mass=AtomicMass.from_u(60.07),
            gasphase=RotationalBD(
                B=Frequency(6, Prefix.GIGA),
                D=InverseLength(0.4e-7, Prefix.CENTI).to_frequency(),
            ),
            droplet=RotationalBD(
                B=Frequency(2.18, Prefix.GIGA),
                D=Frequency(9.5, Prefix.MEGA),
                reference="https://doi.org/10.1103/PhysRevLett.125.013001",
            ),
            polarizability=Polarizability(
                tensor=(
                    (
                        PolarizabilityVolume.from_angstrom3(14.0),
                        PolarizabilityVolume.from_angstrom3(0.0),
                        PolarizabilityVolume.from_angstrom3(0.0),
                    ),
                    (
                        PolarizabilityVolume.from_angstrom3(0.0),
                        PolarizabilityVolume.from_angstrom3(19.0),
                        PolarizabilityVolume.from_angstrom3(0.0),
                    ),
                    (
                        PolarizabilityVolume.from_angstrom3(0.0),
                        PolarizabilityVolume.from_angstrom3(0.0),
                        PolarizabilityVolume.from_angstrom3(34.0),
                    ),
                ),
                bond_axis=CartesianAxis.Z,
                aniso=PolarizabilityVolume.from_angstrom3(3.7),
            ),
            tags=("heavy", "droplets"),
        )
