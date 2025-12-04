
import numpy as np


def gaussian(x, A, x0, sigma, offset):
    """
    1D Gaussian with constant offset.
    """
    return A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) + offset