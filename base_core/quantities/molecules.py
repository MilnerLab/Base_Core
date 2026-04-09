from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, TypeAlias

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

# ----------------- Base molecule class -----------------

@dataclass(frozen=True, slots=True)
class Molecule:
    key: str
    name: str
    formula: str
    cas: Optional[str] = None

    mass: Optional[AtomicMass] = None  # AtomicMass is a Mass in kg, with from_u() :contentReference[oaicite:5]{index=5}

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

        Assumptions
        -----------
        - all quantities are given in SI-compatible units
        - centrifuge_frequency is the mechanical rotation frequency of the
        centrifuge polarization in Hz
        - B and D are stored in frequency units (Hz)
        """

        if angular_acceleration_rad_per_s2 <= 0.0:
            raise ValueError("angular_acceleration_rad_per_s2 must be > 0.")

        centrifuge_frequency = centrifuge_frequency
        
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

        # optical trap depth
        U0_J = intensity_si * self.polarizability.aniso.to_SI() / (2.0 * SPEED_OF_LIGHT * EPS0_F_M)

        # infer current J from the local rotational frequency picture
        J = _find_current_J_from_level_frequency(
            centrifuge_frequency_hz=cfg_freq_hz,
            B_hz=B_hz,
            D_hz=D_hz,
        )

        # slope-based local effective inverse momentum
        eff_inverse_I = B_hz - 2.0 * D_hz * J * (J + 1.0)

        if eff_inverse_I <= 0.0:
            raise ValueError(
                "B_eff_hz became <= 0. The requested frequency is at or beyond the centrifugal wall."
            )

        # local effective moment of inertia
        I_eff_kg_m2 = PLANCK_H_J_S / (8.0 * pi**2 * eff_inverse_I)

        # low-temperature spinnability
        sigma0 = 2.0 * U0_J / (pi * I_eff_kg_m2 * angular_acceleration_rad_per_s2)

        if temperature is None:
            return sigma0

        temperature_K = float(temperature)
        if temperature_K <= 0.0:
            raise ValueError("temperature must be > 0.")

        sigmaT = sigma0 * U0_J / (0.5 * BOLTZMANN_J_K * temperature_K)
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
                reference="MacPhail-Bartley thesis (example)"),
            droplet=RotationalBD(
                B=Frequency(730,Prefix.MEGA),
                D=Frequency(1.2,Prefix.MEGA),
                reference="https://doi.org/10.1103/PhysRevLett.125.013001"),

            polarizability=Polarizability(
                aniso=PolarizabilityVolume.from_angstrom3(8.7)),

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
                B = Frequency(6,Prefix.GIGA),
                D=InverseLength(0.4*10**-7,Prefix.CENTI)
                ),
            droplet=RotationalBD(
                B = Frequency(2.18,Prefix.GIGA),
                D = Frequency(9.5,Prefix.MEGA),
                reference="https://doi.org/10.1103/PhysRevLett.125.013001"),
            
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
                bond_axis=CartesianAxis.Z, #I-I axis
                aniso = PolarizabilityVolume.from_angstrom3(3.7)),
            
            tags=("heavy", "droplets"),
        )