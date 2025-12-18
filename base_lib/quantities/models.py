
from base_lib.quantities.enums import Prefix


class Length(float):
    def __new__(cls, value: float, prefix: Prefix = Prefix.NONE):
        meters = float(value) * prefix.value
        return super().__new__(cls, meters)

    def value(self, prefix: Prefix = Prefix.NONE) -> float:
        return float(self) / prefix.value

class Time(float):
    def __new__(cls, value: float, prefix: Prefix = Prefix.NONE):
        seconds = float(value) * prefix.value
        return super().__new__(cls, seconds)

    def value(self, prefix: Prefix = Prefix.NONE) -> float:
        return float(self) / prefix.value


class Frequency(float):
    def __new__(cls, value: float, prefix: Prefix = Prefix.NONE):
        hz = float(value) * prefix.value
        return super().__new__(cls, hz)

    def value(self, prefix: Prefix = Prefix.NONE) -> float:
        return float(self) / prefix.value



