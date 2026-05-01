from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, TypeAlias

from numpy import pi

from base_core.math.enums import CartesianAxis
from base_core.quantities.enums import Prefix

from base_core.quantities.models import Frequency, InverseLength, Length, Temperature
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

# ----------------- Base molecule class -----------------

@dataclass(frozen=True, slots=True)
class Molecule:
    key: str
    name: str
    formula: str
    cas: Optional[str] = None

    mass: Optional[AtomicMass] = None  # AtomicMass is a Mass in kg, with from_u() :contentReference[oaicite:5]{index=5}
    rotational_radius: Optional[Length] = None

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
            


# ----------------- Fixed-value holders -----------------

class CS2(Molecule):
    def __init__(self) -> None:
        super().__init__(
            key="cs2",
            name="Carbon disulfide",
            formula="CS2",
            mass=AtomicMass.from_u(76.14), 
            
            rotational_radius=Length(1.553, Prefix.ANGSTROM), # sulfur

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
            rotational_radius=Length(1.680, Prefix.ANGSTROM), # oxygen 
            
            
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
                bond_axis=CartesianAxis.Z,
                aniso = PolarizabilityVolume.from_angstrom3(3.7)),
            
            tags=("heavy", "droplets"),
        )
        
class DIB(Molecule):
    def __init__(self) -> None:
        super().__init__(
            key="dib",
            name="1,4-Diiodobenzene",
            formula="C6H4I2",
            mass=AtomicMass.from_u(329.9), 
            rotational_radius=Length(3.45, Prefix.ANGSTROM), # iodine

            gasphase=RotationalBD(
                B=Frequency(156.86, Prefix.MEGA),
                reference="10.1021/acs.jpca.9b11433"),

            polarizability=Polarizability(
                tensor=(
                    (
                        PolarizabilityVolume.from_angstrom3(11.307),
                        PolarizabilityVolume.from_angstrom3(0.0),
                        PolarizabilityVolume.from_angstrom3(0.0),
                    ),
                    (
                        PolarizabilityVolume.from_angstrom3(0.0),
                        PolarizabilityVolume.from_angstrom3(16.676),
                        PolarizabilityVolume.from_angstrom3(0.0),
                    ),
                    (
                        PolarizabilityVolume.from_angstrom3(0.0),
                        PolarizabilityVolume.from_angstrom3(0.0),
                        PolarizabilityVolume.from_angstrom3(32.667),
                    ),
                ),
                bond_axis=CartesianAxis.Z, #I-I axis
                aniso = PolarizabilityVolume.from_angstrom3(21.3),
                reference="10.1021/acs.jpca.9b11433"),

            tags=("asymmetric top", "droplets"),
        )