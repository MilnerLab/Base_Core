from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, TypeAlias

import numpy as np
from numpy import pi
from scipy.linalg import eigh

from base_core.math.enums import CartesianAxis
from base_core.quantities.constants import BOLTZMANN_J_K, EPS0_F_M, PLANCK_H_J_S, SPEED_OF_LIGHT
from base_core.quantities.enums import Prefix
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

BasisState: TypeAlias = tuple[int, int]  # (J, M)


# ----------------- Small domain containers -----------------

@dataclass(frozen=True, slots=True)
class RotationalBD:
    """
    Rotational constant B and centrifugal distortion D.

    Canonical internal representation: Frequency (Hz).
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
class TDSECurve:
    time_s: np.ndarray
    centrifuge_frequency_hz: np.ndarray
    raman_frequency_hz: np.ndarray
    intensity_W_per_m2: np.ndarray
    p_J: np.ndarray
    mean_jj1: np.ndarray
    effective_J_from_mean: np.ndarray
    peak_J: np.ndarray
    average_eff_inverse_I_hz: np.ndarray
    sigma_from_avg_inverse_I: np.ndarray


# ----------------- helper methods ----------------------

def _level_rotation_frequency_hz(J: int, B_hz: float, D_hz: float) -> float:
    """
    Local rotational frequency of a non-rigid linear rotor,
    obtained from the slope d(E/h)/dJ.

        E_J / h = B J(J+1) - D J^2(J+1)^2

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

        if previous_f >= 0.0 and f_rot < previous_f:
            break

        err = abs(f_rot - centrifuge_frequency_hz)
        if err < best_err:
            best_err = err
            best_J = J

        previous_f = f_rot

    return best_J


def _basis_states(j_max: int) -> list[BasisState]:
    states: list[BasisState] = []
    for J in range(j_max + 1):
        for M in range(-J, J + 1):
            states.append((J, M))
    return states


def _a_coeff(J: int, M: int) -> float:
    num = (J + 1) ** 2 - M ** 2
    den = (2 * J + 1) * (2 * J + 3)
    if num <= 0:
        return 0.0
    return float(np.sqrt(num / den))


def _b_coeff(J: int, M: int) -> float:
    if J == 0:
        return 0.0
    num = J ** 2 - M ** 2
    den = (2 * J - 1) * (2 * J + 1)
    if num <= 0:
        return 0.0
    return float(np.sqrt(num / den))


def _cos2_theta_matrix_x_basis(j_max: int) -> tuple[np.ndarray, list[BasisState]]:
    """
    Matrix of cos^2(theta) in the basis |J, M> quantized along the fixed
    field axis in the rotating frame.
    """
    states = _basis_states(j_max)
    idx = {state: i for i, state in enumerate(states)}
    n = len(states)

    mat = np.zeros((n, n), dtype=float)

    for J, M in states:
        i = idx[(J, M)]

        a = _a_coeff(J, M)
        b = _b_coeff(J, M)

        mat[i, i] = a * a + b * b

        if J + 2 <= j_max:
            c = _a_coeff(J, M) * _a_coeff(J + 1, M)
            j = idx[(J + 2, M)]
            mat[i, j] = c
            mat[j, i] = c

    return mat, states


def _jz_matrix_in_x_basis(states: list[BasisState]) -> np.ndarray:
    """
    In the x-quantized basis, the operator j_z = J_z / ħ has the same matrix
    structure as j_x in the usual z-quantized basis.
    """
    idx = {state: i for i, state in enumerate(states)}
    n = len(states)
    mat = np.zeros((n, n), dtype=float)

    for J, M in states:
        i = idx[(J, M)]

        if M + 1 <= J:
            c = 0.5 * np.sqrt(J * (J + 1) - M * (M + 1))
            j = idx[(J, M + 1)]
            mat[i, j] = c
            mat[j, i] = c

    return mat


def _gaussian_intensity_W_per_m2(
    peak_intensity_W_per_m2: float,
    t_s: float,
    ramp_duration_s: float,
    gaussian_sigma_fraction: float,
) -> float:
    sigma_s = gaussian_sigma_fraction * ramp_duration_s
    if sigma_s <= 0.0:
        raise ValueError("gaussian_sigma_fraction must be > 0.")
    center_s = 0.5 * ramp_duration_s
    return peak_intensity_W_per_m2 * np.exp(-0.5 * ((t_s - center_s) / sigma_s) ** 2)


def _hamiltonian_rotating_frame_hz(
    *,
    t_s: float,
    B_hz: float,
    D_hz: float,
    delta_alpha_SI: float,
    peak_intensity_W_per_m2: float,
    angular_acceleration_rad_per_s2: float,
    ramp_duration_s: float,
    gaussian_sigma_fraction: float,
    cos2_mat: np.ndarray,
    jz_mat: np.ndarray,
    states: list[BasisState],
) -> tuple[np.ndarray, float, float]:
    """
    Returns H/h in Hz, plus intensity(t), plus centrifuge frequency f_cfg(t) in Hz.
    """
    intensity_t = _gaussian_intensity_W_per_m2(
        peak_intensity_W_per_m2=peak_intensity_W_per_m2,
        t_s=t_s,
        ramp_duration_s=ramp_duration_s,
        gaussian_sigma_fraction=gaussian_sigma_fraction,
    )

    U0_J = intensity_t * delta_alpha_SI / (2.0 * SPEED_OF_LIGHT * EPS0_F_M)
    u_hz = U0_J / PLANCK_H_J_S
    f_cfg_hz = angular_acceleration_rad_per_s2 * t_s / (2.0 * np.pi)

    n = len(states)
    H_hz = np.zeros((n, n), dtype=float)

    for i, (J, _M) in enumerate(states):
        x = J * (J + 1.0)
        H_hz[i, i] = B_hz * x - D_hz * x * x

    H_hz -= u_hz * cos2_mat
    H_hz -= f_cfg_hz * jz_mat

    return H_hz, intensity_t, f_cfg_hz


def _propagate_one_step(psi: np.ndarray, H_mid_hz: np.ndarray, dt_s: float) -> np.ndarray:
    """
    Midpoint unitary propagation using eigen-decomposition.
    Since H_mid_hz = H/h in Hz, the phase is exp(-i 2π λ dt).
    """
    evals, evecs = eigh(H_mid_hz)
    phases = np.exp(-2j * np.pi * evals * dt_s)
    U = (evecs * phases) @ evecs.conj().T
    psi_next = U @ psi
    norm = np.linalg.norm(psi_next)
    if norm > 0.0:
        psi_next /= norm
    return psi_next


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
        Approximate optical-centrifuge spinnability including centrifugal distortion,
        using a local slope-based effective rotational constant.
        """
        if angular_acceleration_rad_per_s2 <= 0.0:
            raise ValueError("angular_acceleration_rad_per_s2 must be > 0.")

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
        if self.polarizability.aniso is None:
            raise ValueError("Polarizability anisotropy must be defined.")

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

    def calculate_tdse_curve(
        self,
        peak_intensity: Intensity,
        temperature: Temperature | None,
        angular_acceleration_rad_per_s2: float,
        ramp_duration_s: float,
        *,
        n_points: int = 300,
        j_max: int = 16,
        gaussian_sigma_fraction: float = 0.42466,
    ) -> TDSECurve:
        """
        Minimal TDSE propagation in the rotating frame.

        Assumptions
        -----------
        - pure initial state |J=0, M=0>
        - droplet B and D only
        - no wall switch
        - no decoherence
        - no thermal ensemble in the propagation
        - temperature only enters the sigma-like diagnostic
        """
        if angular_acceleration_rad_per_s2 <= 0.0:
            raise ValueError("angular_acceleration_rad_per_s2 must be > 0.")
        if ramp_duration_s <= 0.0:
            raise ValueError("ramp_duration_s must be > 0.")
        if n_points < 2:
            raise ValueError("n_points must be >= 2.")
        if j_max < 0:
            raise ValueError("j_max must be >= 0.")

        if self.droplet.B is None or self.droplet.D is None:
            raise ValueError("Droplet B and D must be defined.")
        if self.polarizability.aniso is None:
            raise ValueError("Anisotropic polarizability must be defined.")

        B_hz = float(self.droplet.B)
        D_hz = float(self.droplet.D)
        delta_alpha_SI = self.polarizability.aniso.to_SI()
        peak_intensity_W_per_m2 = float(peak_intensity)

        if B_hz <= 0.0:
            raise ValueError("Droplet B must be > 0.")
        if D_hz < 0.0:
            raise ValueError("Droplet D must be >= 0.")
        if delta_alpha_SI <= 0.0:
            raise ValueError("Polarizability anisotropy must be > 0.")
        if peak_intensity_W_per_m2 < 0.0:
            raise ValueError("Peak intensity must be >= 0.")

        cos2_mat, states = _cos2_theta_matrix_x_basis(j_max)
        jz_mat = _jz_matrix_in_x_basis(states)

        indices_by_J: list[list[int]] = [[] for _ in range(j_max + 1)]
        for i, (J, _M) in enumerate(states):
            indices_by_J[J].append(i)

        n_basis = len(states)
        times_s = np.linspace(0.0, ramp_duration_s, n_points)

        p_J = np.zeros((n_points, j_max + 1), dtype=float)
        mean_jj1 = np.zeros(n_points, dtype=float)
        effective_J_from_mean = np.zeros(n_points, dtype=float)
        peak_J = np.zeros(n_points, dtype=int)
        avg_eff_inverse_I_hz = np.zeros(n_points, dtype=float)
        sigma_from_avg_inverse_I = np.full(n_points, np.nan, dtype=float)
        intensity_W_per_m2 = np.zeros(n_points, dtype=float)
        centrifuge_frequency_hz = np.zeros(n_points, dtype=float)

        psi = np.zeros(n_basis, dtype=complex)
        i00 = states.index((0, 0))
        psi[i00] = 1.0 + 0.0j

        J_values = np.arange(j_max + 1, dtype=float)
        jj1_values = J_values * (J_values + 1.0)

        def record_point(k: int, t_s: float, psi_now: np.ndarray) -> None:
            _H_hz, intensity_t, f_cfg_t = _hamiltonian_rotating_frame_hz(
                t_s=t_s,
                B_hz=B_hz,
                D_hz=D_hz,
                delta_alpha_SI=delta_alpha_SI,
                peak_intensity_W_per_m2=peak_intensity_W_per_m2,
                angular_acceleration_rad_per_s2=angular_acceleration_rad_per_s2,
                ramp_duration_s=ramp_duration_s,
                gaussian_sigma_fraction=gaussian_sigma_fraction,
                cos2_mat=cos2_mat,
                jz_mat=jz_mat,
                states=states,
            )

            probs_basis = np.abs(psi_now) ** 2

            for J in range(j_max + 1):
                p_J[k, J] = probs_basis[indices_by_J[J]].sum()

            intensity_W_per_m2[k] = intensity_t
            centrifuge_frequency_hz[k] = f_cfg_t

            mean_jj1[k] = np.sum(p_J[k] * jj1_values)
            effective_J_from_mean[k] = 0.5 * (-1.0 + np.sqrt(1.0 + 4.0 * mean_jj1[k]))
            peak_J[k] = int(np.argmax(p_J[k]))

            avg_eff_inverse = np.sum(p_J[k] * (B_hz - 2.0 * D_hz * jj1_values))
            avg_eff_inverse_I_hz[k] = avg_eff_inverse

            if avg_eff_inverse > 0.0:
                U0_J = intensity_t * delta_alpha_SI / (2.0 * SPEED_OF_LIGHT * EPS0_F_M)
                I_eff_kg_m2 = PLANCK_H_J_S / (8.0 * np.pi**2 * avg_eff_inverse)
                sigma0 = 2.0 * U0_J / (np.pi * I_eff_kg_m2 * angular_acceleration_rad_per_s2)

                if temperature is None:
                    sigma_from_avg_inverse_I[k] = sigma0
                else:
                    temperature_K = float(temperature)
                    if temperature_K <= 0.0:
                        raise ValueError("temperature must be > 0.")
                    sigmaT = sigma0 * U0_J / (0.5 * BOLTZMANN_J_K * temperature_K)
                    sigma_from_avg_inverse_I[k] = min(sigma0, sigmaT)

        record_point(0, times_s[0], psi)

        for k in range(n_points - 1):
            t_left = times_s[k]
            t_right = times_s[k + 1]
            dt_s = t_right - t_left
            t_mid = 0.5 * (t_left + t_right)

            H_mid_hz, _, _ = _hamiltonian_rotating_frame_hz(
                t_s=t_mid,
                B_hz=B_hz,
                D_hz=D_hz,
                delta_alpha_SI=delta_alpha_SI,
                peak_intensity_W_per_m2=peak_intensity_W_per_m2,
                angular_acceleration_rad_per_s2=angular_acceleration_rad_per_s2,
                ramp_duration_s=ramp_duration_s,
                gaussian_sigma_fraction=gaussian_sigma_fraction,
                cos2_mat=cos2_mat,
                jz_mat=jz_mat,
                states=states,
            )

            psi = _propagate_one_step(psi, H_mid_hz, dt_s)
            record_point(k + 1, t_right, psi)

        return TDSECurve(
            time_s=times_s,
            centrifuge_frequency_hz=centrifuge_frequency_hz,
            raman_frequency_hz=2.0 * centrifuge_frequency_hz,
            intensity_W_per_m2=intensity_W_per_m2,
            p_J=p_J,
            mean_jj1=mean_jj1,
            effective_J_from_mean=effective_J_from_mean,
            peak_J=peak_J,
            average_eff_inverse_I_hz=avg_eff_inverse_I_hz,
            sigma_from_avg_inverse_I=sigma_from_avg_inverse_I,
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
