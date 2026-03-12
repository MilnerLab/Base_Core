from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional

from base_core.quantities.enums import Prefix
from base_core.quantities.constants import SPEED_OF_LIGHT

from base_core.quantities.specific_models import AtomicMass, Intensity, PolarizabilityVolume
from models import (
    Frequency,
    InverseLength,
    Temperature,
)


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
    iso: Optional[PolarizabilityVolume] = None
    aniso: Optional[PolarizabilityVolume] = None
    reference: str = ""
    notes: str = ""


@dataclass(frozen=True, slots=True)
class Spinnability:
    """
    Dimensionless spinnability σ, optionally with context (conditions).
    """
    sigma: float
    temperature: Optional[Temperature] = None
    intensity: Optional[Intensity] = None
    reference: str = ""
    notes: str = ""


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
    spinnability: tuple[Spinnability, ...] = ()

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
            mass=AtomicMass.from_u(76.14),  # fill more precisely if you want

            # Example gas-phase values from Ian MacPhail-Bartley thesis:
            # B = 0.109 cm^-1, Δα ≈ 9 Å^3 :contentReference[oaicite:6]{index=6}
            gasphase=RotationalBD(
                B=InverseLength(0.110, Prefix.CENTI).to_frequency(),  # 0.110 / cm
                D=None,
                reference="MacPhail-Bartley thesis (example)",
            ),

            # Droplet-renormalized B/D: fill when you have numbers
            droplet=RotationalBD(
                B=Frequency(730,Prefix.MEGA),
                D=Frequency(1.2,Prefix.MEGA),
                reference="https://doi.org/10.1103/PhysRevLett.125.013001"),

            polarizability=Polarizability(
                iso=None,
                aniso=PolarizabilityVolume.from_angstrom3(4.9)  # check Prefix.ANGSTOM naming :contentReference[oaicite:7]{index=7}
            ),

            spinnability=(),
            tags=("linear", "droplets"),
        )


class OCS(Molecule):
    def __init__(self) -> None:
        super().__init__(
            key="ocs",
            name="Carbonyl sulfide",
            formula="OCS",
            mass=AtomicMass.from_u(60.07),  # optional placeholder
            gasphase=RotationalBD(
                B = Frequency(6,Prefix.GIGA),
                D=InverseLength(0.4*10**-7,Prefix.CENTI)
                ),
            droplet=RotationalBD(
                B = Frequency(2.18,Prefix.GIGA),
                D = Frequency(9.5,Prefix.MEGA),
                reference="https://doi.org/10.1103/PhysRevLett.125.013001"
                ),
            polarizability=Polarizability(
                iso = None,
                aniso = PolarizabilityVolume.from_angstrom3(3.7)),
            spinnability=(),
            tags=("linear", "droplets"),
        )



# Optional: registry convenience
REGISTRY: dict[str, Molecule] = {m.key: m for m in (CS2(), OCS())}

def get_molecule(key: str) -> Molecule:
    return REGISTRY[key]