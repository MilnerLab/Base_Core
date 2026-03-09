from xml.dom import ValidationErr
from matplotlib.axes import Axes
from base_core.math.models import Histogram2D
from base_core.plotting.enums import PlotColor, PlotColorMap
from base_core.quantities.enums import Prefix
import numpy as np



def plot_histogram2d(ax: Axes, data: Histogram2D) -> None:
    
    
    ax.pcolormesh(data.x_edges, data.y_edges, data.matrix.T, shading='auto', cmap=PlotColorMap.DEFAULT,alpha=0.5)
    #ax.pcolormesh(data.matrix, shading='auto', cmap='viridis')
    
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
def plot_contour(ax: Axes, data: Histogram2D) -> None:
    # To align the contours correctly, we need the bin centers, not the edges
    # We can use the midpoints of the edges
    xcenters = (data.x_edges[:-1] + data.x_edges[1:]) / 2
    ycenters = (data.y_edges[:-1] + data.y_edges[1:]) / 2
    X, Y = np.meshgrid(xcenters, ycenters)
    #levels = np.arange(0,5)
    max_count = np.max(data.matrix)
    levels = np.arange(max_count-6,max_count,2)
    # Plot the 2D histogram as a heatmap
    # The 'counts.T' is used because histogram2d returns the array in a transposed manner
    #ax.imshow(counts.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='Blues')

    # Overlay the contour lines
    # Use the bin centers (X, Y) and counts for the contour data
    #CS = ax.contour(data.x_edges[:-1], data.y_edges[:-1], data.matrix.T, levels=[levels-2,levels],linewidths=1,cmap=PlotColorMap.MAGMA)
    CS = ax.contour(X,Y, data.matrix.T, levels=levels,linewidths=1,cmap=PlotColorMap.MAGMA)

    ax.clabel(CS,levels[0::2],fontsize=10)
    #ax.contour(data.matrix.T, levels=levels,linewidths=1,cmap=PlotColorMap.MAGMA)
    
    