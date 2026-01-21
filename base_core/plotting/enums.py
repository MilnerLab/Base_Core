from enum import Enum


class PlotColor(str, Enum):
    BLUE   = "b"
    RED    = "r"
    GREEN  = "g"
    PURPLE = "purple"
    GRAY   = "tab:gray"
    BLACK = "k"
    
class PlotColorMap(str, Enum):
    DEFAULT = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    CIVIDIS = "cividis"
    
    