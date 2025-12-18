from enum import Enum
import numpy as np


class AngleUnit(float, Enum):
    RAD = 1.0          
    DEG = np.pi / 180.0    