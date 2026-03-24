from base_core.math.functions import gaussian
from base_core.fitting.models import GaussianFitResult
from matplotlib.axes import Axes
import numpy as np
from base_core.plotting.enums import PlotColor 

def plot_gaussianfit(ax: Axes, fit: GaussianFitResult, N: int = 500, color: PlotColor = PlotColor.BLACK):
    x0 = fit.center
    width = 3*fit.sigma
    x = np.linspace(x0 - width/2, x0 + width/2, N)
    y = gaussian(x,fit.amplitude,x0,fit.sigma,fit.offset)
    label = fr"y =  {fit.amplitude:.2f}\text{{exp}}[-\frac{{1}}{{2}}\left(\frac{{x-{x0:.2f}}}{{{fit.sigma}}}\right)^2] + {fit.offset:.2f}" 
    ax.plot(x,y,label,color)
    