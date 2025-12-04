from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy.optimize import curve_fit

from base_lib.functions import gaussian
from base_lib.models import GaussianFitResult



def fit_gaussian(x: Sequence[float], y: Sequence[float]) -> GaussianFitResult:
    """
    Fit a Gaussian to data (x, y) and return parameters + 1σ errors.
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    # simple initial guesses
    A0 = float(y_arr.max() - y_arr.min())
    x0 = float(x_arr[np.argmax(y_arr)])
    sigma0 = float((x_arr.max() - x_arr.min()) / 6)
    offset0 = float(y_arr.min())
    p0 = [A0, x0, sigma0, offset0]

    popt, pcov = curve_fit(gaussian, x_arr, y_arr, p0=p0)

    # 1σ uncertainties of the parameters from covariance matrix
    perr = np.sqrt(np.diag(pcov))

    return GaussianFitResult(
        amplitude=popt[0],
        center=popt[1],
        sigma=popt[2],
        offset=popt[3],
        amplitude_err=perr[0],
        center_err=perr[1],
        sigma_err=perr[2],
        offset_err=perr[3],
        covariance=pcov,
    )
