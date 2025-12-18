from enum import Enum


class Prefix(float, Enum):
    NONE = 1.0   
    PICO   = 1e-12
    NANO   = 1e-9
    MICRO  = 1e-6
    MILLI   = 1e-3
    CENTI   = 1e-2
    KILO   = 1e3
    MEGA   = 1e6
    GIGA   = 1e9
    TERA   = 1e12