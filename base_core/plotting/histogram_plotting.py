from xml.dom import ValidationErr
from matplotlib.axes import Axes
from base_core.math.models import Histogram2D
from base_core.plotting.enums import PlotColor, PlotColorMap
from base_core.quantities.enums import Prefix
import numpy as np
from matplotlib import cm, contour, collections
from math import log



def plot_histogram2d(ax: Axes, data: Histogram2D) -> collections.QuadMesh:
    #ax.pcolormesh(data.matrix, shading='auto', cmap='viridis')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    norm = cm.colors.LogNorm()
    return ax.pcolormesh(data.x_edges, data.y_edges, data.matrix.T, norm=norm, shading='auto', cmap=PlotColorMap.MAGMA,alpha=1.0) #transposed matrix bc hist2d transposes the output
    
def plot_contour(ax: Axes, data: Histogram2D,*,min_count: int = 50) -> contour.QuadContourSet:
    # To align the contours correctly, we need the bin centers, not the edges
    # We can use the midpoints of the edges
    
    xcenters = (data.x_edges[:-1] + data.x_edges[1:]) / 2
    ycenters = (data.y_edges[:-1] + data.y_edges[1:]) / 2
    X, Y = np.meshgrid(xcenters, ycenters)
    base = 5
    max_count = int(log(np.max(data.matrix),base))
    min_count = int(log(min_count,base))
    #levels = np.linspace(min_count,max_count,100,dtype=int)
   
    levels = np.unique(np.logspace(min_count, max_count, num=15, endpoint=True, base=base,dtype=int))
    norm = cm.colors.LogNorm(vmax=max_count, vmin=min_count)

    # Use the bin centers (X, Y) and counts for the contour data
    #CS = ax.contour(data.x_edges[:-1], data.y_edges[:-1], data.matrix.T, levels=[levels-2,levels],linewidths=1,cmap=PlotColorMap.MAGMA)
    return ax.contour(X,Y, data.matrix.T,levels=levels,linewidths=2,cmap=PlotColorMap.GREENS)

    #ax.clabel(CS,levels,fontsize=10)

    
    