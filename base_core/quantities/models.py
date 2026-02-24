
from base_core.quantities.enums import Prefix
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
