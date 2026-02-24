from xml.dom import ValidationErr
from matplotlib.axes import Axes
from base_core.math.models import Histogram2D
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
import numpy as np



def plot_histogram2d(ax: Axes, data: Histogram2D, is_heatmap: bool = True) -> None:
    
    
    