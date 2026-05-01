import numpy as np

from base_core.framework.serialization.serde import Primitive, PrimitiveSerde
from base_core.quantities.constants import BOHR_RADIUS_M, EPS0_F_M, U_KG
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Frequency, Mass, Power, Time, Volume


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
    
class Intensity(float, PrimitiveSerde):
    """
    Generic (optical) intensity I = Power / Area.
    Internal unit: W / m^2.

    Units are expressed as:
        (power_prefix * W) / (per * m)^2
    """

    def __new__(
        cls,
        value: float,
        *,
        power_prefix: Prefix = Prefix.NONE,
        area_prefix: Prefix = Prefix.NONE,
    ):
        w_per_m2 = float(value) * power_prefix.value / (area_prefix.value ** 2)
        if w_per_m2 < 0:
            raise ValueError("Intensity cannot be negative.")
        return super().__new__(cls, w_per_m2)

    def value(
        self,
        *,
        power_prefix: Prefix = Prefix.NONE,
        per: Prefix = Prefix.NONE,
    ) -> float:
        return float(self) / power_prefix.value * (per.value ** 2)

    def to_primitive(self) -> float:
        return float(self)  # W/m^2

    @classmethod
    def from_primitive(cls, v: Primitive) -> "Intensity":
        return cls(float(v))  # interpret as W/m^2
    
class Energy(float, PrimitiveSerde):
    """
    Generic energy.
    Internal unit: J (Joule) = W*s.

    Build from Power and Time:
        E = P * t
    """

    def __new__(cls, value: float, prefix: Prefix = Prefix.NONE):
        j = float(value) * prefix.value
        if j < 0:
            raise ValueError("Energy cannot be negative.")
        return super().__new__(cls, j)

    def value(self, prefix: Prefix = Prefix.NONE) -> float:
        return float(self) / prefix.value

    @classmethod
    def from_power_time(cls, power: Power, time: Time) -> "Energy":
        # Power is in W, Time is in s (both internally), so product is Joule.
        return cls(float(power) * float(time), prefix=Prefix.NONE)

    def to_primitive(self) -> float:
        return float(self)  # J

    @classmethod
    def from_primitive(cls, v: Primitive) -> "Energy":
        return cls(float(v))  # interpret as J
    
class AngularFrequency(float, PrimitiveSerde):
    """Angular frequency. Internal unit: rad/s."""

    def __new__(cls, value: float, prefix: Prefix = Prefix.NONE):
        return super().__new__(cls, float(value) * prefix.value)

    def value(self, prefix: Prefix = Prefix.NONE) -> float:
        return float(self) / prefix.value

    @classmethod
    def from_frequency(cls, frequency: Frequency) -> "AngularFrequency":
        return cls(2.0 * np.pi * float(frequency))

    def to_frequency(self) -> Frequency:
        return Frequency(float(self) / (2.0 * np.pi))

    def to_primitive(self) -> float:
        return float(self)

    @classmethod
    def from_primitive(cls, v: Primitive) -> "AngularFrequency":
        return cls(float(v))


class AngularChirp(float, PrimitiveSerde):
    """
    Linear angular-frequency chirp. Internal unit: rad/s^2.

    Example:
        AngularChirp(3.7e-2, time_prefix=Prefix.PICO)
        represents 3.7e-2 rad/ps^2.
    """

    def __new__(cls, value: float, time_prefix: Prefix = Prefix.NONE):
        return super().__new__(cls, float(value) / (time_prefix.value ** 2))

    def value(self, time_prefix: Prefix = Prefix.NONE) -> float:
        return float(self) * (time_prefix.value ** 2)

    def to_primitive(self) -> float:
        return float(self)

    @classmethod
    def from_primitive(cls, v: Primitive) -> "AngularChirp":
        return cls(float(v))