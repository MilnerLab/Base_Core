from enum import Enum, IntFlag, auto
import numpy as np


class AngleUnit(float, Enum):
    RAD = 1.0          
    DEG = np.pi / 180.0   
    
class CartesianAxis(IntFlag):
    NONE = 0
    X = auto()
    Y = auto()
    Z = auto()

    def axes(self) -> tuple["CartesianAxis", ...]:
        return tuple(a for a in (self.X, self.Y, self.Z) if self & a)

    def require_single(self) -> "CartesianAxis":
        if len(self.axes()) != 1:
            raise ValueError(f"Expected one axis, got {self}")
        return self

    def require_plane(self) -> "CartesianAxis":
        if len(self.axes()) != 2:
            raise ValueError(f"Expected two axes, got {self}")
        return self
    
XY = CartesianAxis.X | CartesianAxis.Y
XZ = CartesianAxis.X | CartesianAxis.Z
YZ = CartesianAxis.Y | CartesianAxis.Z