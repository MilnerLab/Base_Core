import numpy as np

from base_core.framework.serialization.serde import Primitive, PrimitiveSerde
from base_core.quantities.constants import BOHR_RADIUS_M, EPS0_F_M, U_KG
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Mass, Volume


class AtomicMass(Mass):
    @classmethod
    def from_u(cls, value_u: float) -> "Mass":
        return cls(float(value_u) * U_KG)  # already kg

    def value_u(self) -> float:
        return float(self) / U_KG

class PolarizabilityVolume(Volume):
    """
    Polarizability expressed as a "volume" α_vol.
    Internal unit: m^3.

    Common convention in molecular physics:
      α_vol = α_SI / (4π ε0)
      so α_SI = 4π ε0 α_vol

    This is the convention behind quoting polarizability in Å^3.
    """
    @classmethod
    def from_angstrom3(cls, value_A3: float) -> "PolarizabilityVolume":
        return cls(float(value_A3) * (Prefix.ANGSTROM ** 3))

    def value_angstrom3(self) -> float:
        return float(self) / (Prefix.ANGSTROM ** 3)

    @classmethod
    def from_bohr3(cls, value_a03: float) -> "PolarizabilityVolume":
        return cls(float(value_a03) * (BOHR_RADIUS_M ** 3))

    def value_bohr3(self) -> float:
        return float(self) / (BOHR_RADIUS_M ** 3)

    def to_SI(self) -> float:
        return (4.0 * np.pi * EPS0_F_M * float(self))