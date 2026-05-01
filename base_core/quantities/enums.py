from enum import Enum


class Prefix(float, Enum):
    NONE = 1.0   
    PICO   = 1e-12
    ANGSTROM = 1e-10
    NANO   = 1e-9
    MICRO  = 1e-6
    MILLI   = 1e-3
    CENTI   = 1e-2
    KILO   = 1e3
    MEGA   = 1e6
    GIGA   = 1e9
    TERA   = 1e12
    

class TemperatureUnit(Enum):
    K = "K"
    C = "C"
    F = "F"
    

class PressureUnit(Enum):
    """Conversion factor to Pascal (Pa)."""
    PA   = 1.0
    BAR  = 1e5
    ATM  = 101_325.0

    # vacuum / metrology
    TORR = 101_325.0 / 760.0          # = 133.32236842105263... Pa (exact via atm/760)
    MMHG = 133.322387415              # conventional mmHg (very close to Torr)

    # imperial
    PSI  = 6_894.757293168
    INHG = 133.322387415 * 25.4       # = 3386.388640341 Pa (from mmHg)
    
class CircularHandedness(Enum):
    RIGHT = +1
    LEFT = -1