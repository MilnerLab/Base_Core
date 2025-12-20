from dataclasses import dataclass
from base_core.math.functions import gaussian
import numpy as np


@dataclass
class GaussianFitResult:
    amplitude: float
    center: float
    sigma: float
    offset: float

    amplitude_err: float | None = None
    center_err: float | None = None
    sigma_err: float | None = None
    offset_err: float | None = None

    covariance: np.ndarray | None = None

    def get_curve(self, x):
        """
        Evaluate the fitted Gaussian for arbitrary x values.
        """
        x = np.asarray(x)
        return gaussian(x, self.amplitude, self.center, self.sigma, self.offset)
