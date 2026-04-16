from matplotlib import cm, collections
from matplotlib.axes import Axes
import numpy as np

from base_core.math.models import AngularCovariance


def plot_covariance(
    ax: Axes,
    data: AngularCovariance,
    *,
    to_degree: bool = True,
    clip_negative: bool = True,
    shift_deg: float = 90.0,
) -> collections.QuadMesh:
    x_edges = data.theta1_edges
    y_edges = data.theta2_edges
    m = data.matrix

    if clip_negative:
        m = np.maximum(m, 0.0)

    if to_degree:
        x_edges = np.rad2deg(x_edges) - shift_deg
        y_edges = np.rad2deg(y_edges) - shift_deg

    vmax = float(np.max(m)) if m.size else 1.0
    vmax = max(vmax, 1e-12)

    norm = cm.colors.Normalize(vmin=0.0, vmax=vmax)

    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        m.T,
        norm=norm,
        shading="auto",
        cmap="viridis",
        alpha=1.0,
    )

    ax.set_xlabel("Emission Angle (Degrees)" if to_degree else "Emission Angle (Radians)")
    ax.set_ylabel("Emission Angle (Degrees)" if to_degree else "Emission Angle (Radians)")

    if to_degree:
        ax.set_xlim(-90, 270)
        ax.set_ylim(-90, 270)
        ticks = [-90, 0, 90, 180, 270]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    ax.set_aspect("equal")

    return mesh