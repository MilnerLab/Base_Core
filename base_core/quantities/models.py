
import numpy as np

from base_core.quantities.constants import SPEED_OF_LIGHT
from base_core.quantities.enums import Prefix, PressureUnit, TemperatureUnit
from base_core.framework.serialization.serde import PrimitiveSerde, Primitive


class Length(float, PrimitiveSerde):
    def __new__(cls, value: float, prefix: Prefix = Prefix.NONE):
        meters = float(value) * prefix.value
        return super().__new__(cls, meters)

    def value(self, prefix: Prefix = Prefix.NONE) -> float:
        return float(self) / prefix.value

    def to_primitive(self) -> float:
        return float(self)  # meters

    @classmethod
    def from_primitive(cls, v: Primitive) -> "Length":
        return cls(float(v))  # interpret as meters
    
class Temperature(float, PrimitiveSerde):
    """
    Internally stored as Kelvin.
    - K can be scaled by Prefix (e.g., milliKelvin).
    - C/F have offsets, so Prefix usually does not make sense.
    """

    def __new__(
        cls,
        value: float,
        unit: TemperatureUnit = TemperatureUnit.K,
        prefix: Prefix = Prefix.NONE,
    ):
        v = float(value)

        if unit == TemperatureUnit.K:
            kelvin = v * prefix.value
        elif unit == TemperatureUnit.C:
            if prefix != Prefix.NONE:
                raise ValueError("Prefix is not meaningful for °C (offset unit).")
            kelvin = v + 273.15
        elif unit == TemperatureUnit.F:
            if prefix != Prefix.NONE:
                raise ValueError("Prefix is not meaningful for °F (offset unit).")
            kelvin = (v - 32.0) * (5.0 / 9.0) + 273.15
        else:
            raise ValueError(f"Unknown TemperatureUnit: {unit}")

        return super().__new__(cls, kelvin)

    def value(
        self,
        unit: TemperatureUnit = TemperatureUnit.K,
        prefix: Prefix = Prefix.NONE,
    ) -> float:
        k = float(self)

        if unit == TemperatureUnit.K:
            return k / prefix.value
        elif unit == TemperatureUnit.C:
            if prefix != Prefix.NONE:
                raise ValueError("Prefix is not meaningful for °C (offset unit).")
            return k - 273.15
        elif unit == TemperatureUnit.F:
            if prefix != Prefix.NONE:
                raise ValueError("Prefix is not meaningful for °F (offset unit).")
            return (k - 273.15) * (9.0 / 5.0) + 32.0

        raise ValueError(f"Unknown TemperatureUnit: {unit}")

    def to_primitive(self) -> float:
        return float(self)  # Kelvin

    @classmethod
    def from_primitive(cls, v: Primitive) -> "Temperature":
        return cls(float(v), unit=TemperatureUnit.K)  # interpret as Kelvin


class Time(float, PrimitiveSerde):
    def __new__(cls, value: float, prefix: Prefix = Prefix.NONE):
        seconds = float(value) * prefix.value
        return super().__new__(cls, seconds)

    def value(self, prefix: Prefix = Prefix.NONE) -> float:
        return float(self) / prefix.value

    def to_primitive(self) -> float:
        return float(self)  # seconds

    @classmethod
    def from_primitive(cls, v: Primitive) -> "Time":
        return cls(float(v))  # interpret as seconds


class Frequency(float, PrimitiveSerde):
    def __new__(cls, value: float, prefix: Prefix = Prefix.NONE):
        hz = float(value) * prefix.value
        return super().__new__(cls, hz)

    def value(self, prefix: Prefix = Prefix.NONE) -> float:
        return float(self) / prefix.value

    def to_primitive(self) -> float:
        return float(self)  # Hz

    @classmethod
    def from_primitive(cls, v: Primitive) -> "Frequency":
        return cls(float(v))  # interpret as Hz
    
    def to_inverse_length(self) -> "InverseLength":
        return InverseLength(float(self) / SPEED_OF_LIGHT)


class InverseLength(float, PrimitiveSerde):
    """
    Generic inverse length.
    Internal unit: 1/m.

    Prefix refers to the *length unit* in the denominator:
      - InverseLength(2.0, Prefix.CENTI)  -> 2 / cm
      - InverseLength(100.0, Prefix.MILLI)-> 100 / mm
    """
    def __new__(cls, value: float, per: Prefix = Prefix.NONE):
        per_meter = float(value) / per.value  # (1/(per*m)) = (1/per)/m
        return super().__new__(cls, per_meter)

    def value(self, per: Prefix = Prefix.NONE) -> float:
        return float(self) * per.value

    def to_primitive(self) -> float:
        return float(self)  # 1/m

    @classmethod
    def from_primitive(cls, v: Primitive) -> "InverseLength":
        return cls(float(v))  # interpret as 1/m

    def to_frequency(self) -> Frequency:
        """ν = c * k  (Hz if k is 1/m)."""
        return Frequency(SPEED_OF_LIGHT * float(self))


class Mass(float, PrimitiveSerde):
    """
    Generic mass.
    Internal unit: kg.

    Prefix is relative to kg:
      - Mass(1.0, Prefix.MILLI) == 1 g
    """
    def __new__(cls, value: float, prefix: Prefix = Prefix.NONE):
        kg = float(value) * prefix.value
        return super().__new__(cls, kg)

    def value(self, prefix: Prefix = Prefix.NONE) -> float:
        return float(self) / prefix.value

    def to_primitive(self) -> float:
        return float(self)  # kg

    @classmethod
    def from_primitive(cls, v: Primitive) -> "Mass":
        return cls(float(v))  # interpret as kg


class Volume(float, PrimitiveSerde):
    """
    Generic volume.
    Internal unit: m^3.

    Prefix is relative to meters:
      - Volume(1.0, Prefix.CENTI) == 1 cm^3
    """
    def __new__(cls, value: float, prefix: Prefix = Prefix.NONE):
        m3 = float(value) * (prefix.value ** 3)
        return super().__new__(cls, m3)

    def value(self, prefix: Prefix = Prefix.NONE) -> float:
        return float(self) / (prefix.value ** 3)

    def to_primitive(self) -> float:
        return float(self)  # m^3

    @classmethod
    def from_primitive(cls, v: Primitive) -> "Volume":
        return cls(float(v))  # interpret as m^3


class Pressure(float, PrimitiveSerde):
    """
    Generic pressure.
    Internal unit: Pa.
    """
    def __new__(
        cls,
        value: float,
        unit: PressureUnit = PressureUnit.PA,
        prefix: Prefix = Prefix.NONE,
    ):
        pa = float(value) * prefix.value * float(unit.value)
        return super().__new__(cls, pa)

    def value(
        self,
        unit: PressureUnit = PressureUnit.PA,
        prefix: Prefix = Prefix.NONE,
    ) -> float:
        return float(self) / (float(unit.value) * prefix.value)

    def to_primitive(self) -> float:
        return float(self)  # Pa

    @classmethod
    def from_primitive(cls, v: Primitive) -> "Pressure":
        return cls(float(v), unit=PressureUnit.PA)  # interpret as Pa