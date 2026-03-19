from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, TypeAlias

import numpy as np
from numpy import pi

from base_core.math.enums import CartesianAxis
from base_core.quantities.enums import Prefix
from base_core.quantities.constants import BOLTZMANN_J_K, EPS0_F_M, PLANCK_H_J_S, SPEED_OF_LIGHT
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


# ----------------- helper methods ----------------------

BasisState: TypeAlias = tuple[int, int]


def _gaussian_intensity_at_time(
    peak_intensity_W_per_m2: float,
    ramp_duration_s: float,
    time_s: float,
    sigma_fraction: float,
) -> float:
    """
    Gaussian intensity envelope centered at T/2.

    sigma = sigma_fraction * T.
    The default sigma_fraction=1/6 means the pulse is concentrated mostly
    inside the interval [0, T].
    """
    sigma_s = sigma_fraction * ramp_duration_s
    if sigma_s <= 0.0:
        raise ValueError("sigma_fraction * ramp_duration_s must be > 0.")

    center_s = 0.5 * ramp_duration_s
    exponent = -0.5 * ((time_s - center_s) / sigma_s) ** 2
    return peak_intensity_W_per_m2 * float(np.exp(exponent))



def _build_basis_states(j_max: int) -> list[BasisState]:
    if j_max < 0:
        raise ValueError("j_max must be >= 0.")

    return [(J, M) for J in range(j_max + 1) for M in range(-J, J + 1)]



def _cos_theta_up(J: int, M: int) -> float:
    numerator = (J + 1) ** 2 - M**2
    denominator = (2 * J + 1) * (2 * J + 3)
    if numerator <= 0 or denominator <= 0:
        return 0.0
    return float(np.sqrt(numerator / denominator))



def _cos_theta_down(J: int, M: int) -> float:
    if J <= 0:
        return 0.0

    numerator = J**2 - M**2
    denominator = (2 * J - 1) * (2 * J + 1)
    if numerator <= 0 or denominator <= 0:
        return 0.0
    return float(np.sqrt(numerator / denominator))



def _build_rotating_frame_hamiltonian_hz(
    *,
    basis_states: list[BasisState],
    index_by_state: dict[BasisState, int],
    B_hz: float,
    D_hz: float,
    optical_potential_hz: float,
    polarization_rotation_rate_rad_per_s: float,
) -> np.ndarray:
    """
    Build H/h in the field-fixed basis |J, M>_field.

    Model:
        H/h = B J(J+1) - D [J(J+1)]^2
              - u(t) cos^2(theta_field)
              - f_rot(t) j_z

    where
        u(t) = U0(t) / h           [Hz]
        f_rot(t) = Omega(t) / 2pi  [Hz]
        j_z = J_z / hbar           [dimensionless]

    In this basis:
        cos^2(theta_field): Delta J = 0, +/-2 and Delta M = 0
        j_z:                Delta J = 0 and Delta M = +/-1
    """
    n = len(basis_states)
    H = np.zeros((n, n), dtype=float)
    rotation_frequency_hz = polarization_rotation_rate_rad_per_s / (2.0 * pi)

    for col, (J, M) in enumerate(basis_states):
        jj1 = float(J * (J + 1))

        # field-free diagonal term
        H[col, col] += B_hz * jj1 - D_hz * jj1 * jj1

        # optical potential term: -u cos^2(theta)
        a_j = _cos_theta_up(J, M)
        b_j = _cos_theta_down(J, M)

        H[col, col] += -optical_potential_hz * (a_j * a_j + b_j * b_j)

        up_state = (J + 2, M)
        if up_state in index_by_state:
            up_coeff = a_j * _cos_theta_up(J + 1, M)
            H[index_by_state[up_state], col] += -optical_potential_hz * up_coeff

        down_state = (J - 2, M)
        if down_state in index_by_state:
            down_coeff = b_j * _cos_theta_down(J - 1, M)
            H[index_by_state[down_state], col] += -optical_potential_hz * down_coeff

        # rotating-frame term: -f_rot * j_z in the field-fixed basis
        m_plus_state = (J, M + 1)
        if m_plus_state in index_by_state:
            coeff = 0.5 * np.sqrt(jj1 - M * (M + 1))
            H[index_by_state[m_plus_state], col] += -rotation_frequency_hz * coeff

        m_minus_state = (J, M - 1)
        if m_minus_state in index_by_state:
            coeff = 0.5 * np.sqrt(jj1 - M * (M - 1))
            H[index_by_state[m_minus_state], col] += -rotation_frequency_hz * coeff

    # numerical cleanup
    return 0.5 * (H + H.T)



def _adiabatic_dressed_state(
    *,
    B_hz: float,
    D_hz: float,
    delta_alpha_SI: float,
    peak_intensity_W_per_m2: float,
    angular_acceleration_rad_per_s2: float,
    ramp_duration_s: float,
    time_s: float,
    j_max: int,
    n_time_steps: int,
    gaussian_sigma_fraction: float,
) -> tuple[np.ndarray, list[BasisState], float]:
    """
    Follow the instantaneous dressed-state branch that is adiabatically
    connected to |J=0, M=0>_field at t=0.

    Returns
    -------
    state_vector
        Eigenvector coefficients in the field-fixed |J, M> basis.
    basis_states
        Basis-state labels corresponding to the vector entries.
    intensity_at_time_W_per_m2
        Instantaneous intensity at the requested time.
    """
    if ramp_duration_s <= 0.0:
        raise ValueError("ramp_duration_s must be > 0.")
    if time_s < 0.0 or time_s > ramp_duration_s:
        raise ValueError("time_s must satisfy 0 <= time_s <= ramp_duration_s.")
    if n_time_steps < 2:
        raise ValueError("n_time_steps must be >= 2.")

    basis_states = _build_basis_states(j_max)
    index_by_state = {state: i for i, state in enumerate(basis_states)}
    reference_state = np.zeros(len(basis_states), dtype=float)
    reference_state[index_by_state[(0, 0)]] = 1.0

    previous_state: Optional[np.ndarray] = None
    times = np.linspace(0.0, time_s, n_time_steps)
    intensity_at_time = 0.0

    for current_time_s in times:
        intensity_at_time = _gaussian_intensity_at_time(
            peak_intensity_W_per_m2=peak_intensity_W_per_m2,
            ramp_duration_s=ramp_duration_s,
            time_s=current_time_s,
            sigma_fraction=gaussian_sigma_fraction,
        )
        optical_potential_J = intensity_at_time * delta_alpha_SI / (2.0 * SPEED_OF_LIGHT * EPS0_F_M)
        optical_potential_hz = optical_potential_J / PLANCK_H_J_S
        omega_rad_per_s = angular_acceleration_rad_per_s2 * current_time_s

        H_hz = _build_rotating_frame_hamiltonian_hz(
            basis_states=basis_states,
            index_by_state=index_by_state,
            B_hz=B_hz,
            D_hz=D_hz,
            optical_potential_hz=optical_potential_hz,
            polarization_rotation_rate_rad_per_s=omega_rad_per_s,
        )

        _, eigenvectors = np.linalg.eigh(H_hz)

        target_state = reference_state if previous_state is None else previous_state
        overlaps = np.abs(eigenvectors.T @ target_state)
        selected = int(np.argmax(overlaps))
        state = eigenvectors[:, selected]

        if previous_state is not None:
            phase = float(previous_state @ state)
            if phase < 0.0:
                state = -state

        previous_state = state

    if previous_state is None:
        raise RuntimeError("Failed to construct dressed state.")

    return previous_state, basis_states, intensity_at_time



def _mean_JJ_plus_1(state_vector: np.ndarray, basis_states: list[BasisState]) -> float:
    probabilities = np.abs(state_vector) ** 2
    return float(
        sum(probability * J * (J + 1) for probability, (J, _M) in zip(probabilities, basis_states))
    )


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
        peak_intensity: Intensity,
        temperature: Temperature | None,
        angular_acceleration_rad_per_s2: float,
        ramp_duration_s: float,
        time_s: float,
        *,
        j_max: int = 30,
        n_time_steps: int = 200,
        gaussian_sigma_fraction: float = 1.0 / 6.0,
    ) -> float:
        """
        Approximate optical-centrifuge spinnability in an instantaneous
        field-dressed model.

        The method uses a rotating-frame Hamiltonian, follows the dressed-state
        branch adiabatically from |J=0, M=0>, projects that state onto the
        field-free |J, M> basis, and replaces J(J+1) by its expectation value.

        Assumptions
        -----------
        - linear polarization rotates with constant angular acceleration
            Omega(t) = angular_acceleration_rad_per_s2 * t
        - intensity envelope is Gaussian, centered at T/2, with
            sigma = gaussian_sigma_fraction * T
        - field-dressed state is followed adiabatically from |0, 0>
        - droplet B and D are used and are stored in Hz
        - peak_intensity is given in W/m^2
        - ramp_duration_s and time_s are given in seconds
        """
        if angular_acceleration_rad_per_s2 <= 0.0:
            raise ValueError("angular_acceleration_rad_per_s2 must be > 0.")
        if ramp_duration_s <= 0.0:
            raise ValueError("ramp_duration_s must be > 0.")
        if time_s < 0.0 or time_s > ramp_duration_s:
            raise ValueError("time_s must satisfy 0 <= time_s <= ramp_duration_s.")
        if j_max < 0:
            raise ValueError("j_max must be >= 0.")
        if n_time_steps < 2:
            raise ValueError("n_time_steps must be >= 2.")
        if gaussian_sigma_fraction <= 0.0:
            raise ValueError("gaussian_sigma_fraction must be > 0.")
        if self.droplet.B is None or self.droplet.D is None:
            raise ValueError("Droplet B and D must be defined.")
        if self.polarizability.aniso is None:
            raise ValueError("Anisotropic polarizability must be defined.")

        peak_intensity_si = float(peak_intensity)
        B_hz = float(self.droplet.B)
        D_hz = float(self.droplet.D)
        delta_alpha_si = self.polarizability.aniso.to_SI()

        if peak_intensity_si < 0.0:
            raise ValueError("peak_intensity must be >= 0.")
        if B_hz <= 0.0:
            raise ValueError("Droplet B must be > 0.")
        if D_hz < 0.0:
            raise ValueError("Droplet D must be >= 0.")

        state_vector, basis_states, intensity_at_time = _adiabatic_dressed_state(
            B_hz=B_hz,
            D_hz=D_hz,
            delta_alpha_SI=delta_alpha_si,
            peak_intensity_W_per_m2=peak_intensity_si,
            angular_acceleration_rad_per_s2=angular_acceleration_rad_per_s2,
            ramp_duration_s=ramp_duration_s,
            time_s=time_s,
            j_max=j_max,
            n_time_steps=n_time_steps,
            gaussian_sigma_fraction=gaussian_sigma_fraction,
        )

        mean_JJ1 = _mean_JJ_plus_1(state_vector, basis_states)
        eff_inverse_I = B_hz - 2.0 * D_hz * mean_JJ1

        if eff_inverse_I <= 0.0:
            raise ValueError(
                "Effective inverse moment of inertia became <= 0. "
                "The requested time is at or beyond the centrifugal wall in this model."
            )

        optical_potential_J = intensity_at_time * delta_alpha_si / (2.0 * SPEED_OF_LIGHT * EPS0_F_M)
        I_eff_kg_m2 = PLANCK_H_J_S / (8.0 * pi**2 * eff_inverse_I)
        sigma0 = 2.0 * optical_potential_J / (pi * I_eff_kg_m2 * angular_acceleration_rad_per_s2)

        if temperature is None:
            return sigma0

        temperature_K = float(temperature)
        if temperature_K <= 0.0:
            raise ValueError("temperature must be > 0.")

        sigmaT = sigma0 * optical_potential_J / (0.5 * BOLTZMANN_J_K * temperature_K)
        return min(sigma0, sigmaT)


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
                D=InverseLength(0.4 * 10**-7, Prefix.CENTI).to_frequency(),
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
