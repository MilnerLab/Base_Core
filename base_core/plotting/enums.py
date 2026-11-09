from enum import Enum


class PlotColor(str, Enum):
    WHITE = "w"
    BLUE   = "b"
    RED    = "r"
    GREEN  = "g"
    PURPLE = "purple"
    GRAY   = "tab:gray"
    ORANGE = "tab:orange"
    BROWN  = "tab:brown"
    CYAN   = "c"
    NAVY   = "navy"
    BLACK = "k"
    
class PlotColorMap(str, Enum):
    DEFAULT = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    CIVIDIS = "cividis"
    GREENS = "Greens"
    