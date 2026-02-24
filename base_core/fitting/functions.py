from base_core.fitting.models import GaussianFitResult
import numpy as np
from scipy.optimize import least_squares
from base_core.math.functions import gaussian

def fit_gaussian(x, y, *, max_nfev: int = 20_000):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    # initial guesses
    A0 = float(y_arr.max() - y_arr.min())
    x0 = float(x_arr[np.argmax(y_arr)])
    sigma0 = float(max((x_arr.max() - x_arr.min()) / 6, 1e-12))
    offset0 = float(y_arr.min())
    p0 = np.array([A0, x0, sigma0, offset0], dtype=float)

    def resid(p):
        return gaussian(x_arr, p[0], p[1], p[2], p[3]) - y_arr

    # bounds: sigma > 0, center within data range
    lb = np.array([-np.inf, x_arr.min(), 1e-12, -np.inf], dtype=float)
    ub = np.array([ np.inf, x_arr.max(), np.inf,   np.inf], dtype=float)

    res = least_squares(
        resid,
        p0,
        bounds=(lb, ub),
        method="trf",
        max_nfev=max_nfev,
    )

    popt = res.x

    # covariance / 1σ errors (auch wenn nicht "success", oft trotzdem sinnvoll zum Einschätzen)
    m = y_arr.size
    n = popt.size
    if m > n and res.jac is not None:
        JTJ = res.jac.T @ res.jac
        s_sq = (2 * res.cost) / (m - n)
        pcov = np.linalg.pinv(JTJ) * s_sq
        perr = np.sqrt(np.diag(pcov))
    else:
        pcov = np.full((4, 4), np.nan)
        perr = np.full(4, np.nan)
        
    fit = GaussianFitResult(
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

    return fit
