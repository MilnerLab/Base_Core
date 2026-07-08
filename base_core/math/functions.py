from __future__ import annotations

from dataclasses import dataclass, field
from math import factorial, pi
from typing import Mapping, Sequence

from base_core.quantities import constants
import numpy as np
from scipy.special import erf, lpmv
from scipy.optimize import minimize_scalar

from base_core.quantities.constants import SPEED_OF_LIGHT



def gaussian(x: Sequence[float], A, x0, sigma, offset):
    """
    1D Gaussian with constant offset.
    """
    xs = np.array(x, dtype=float)
    
    return A * np.exp(-((xs - x0) ** 2) / (2 * sigma ** 2)) + offset

def spectrum_fit(lam, A, theta0, theta1, theta2, V, offset,
                 lambda0, delta_lambda_fwhm):
    """
    Direct-polynomial interference spectrum vs wavelength.

    Theta(Omega) = theta0 + theta1*Omega + theta2*Omega^2
    I = A * env * (1 + V * cos(Theta)) + offset

    lam               : wavelength axis [nm]
    A                 : mean-fringe amplitude; peak = A*(1+V)+offset
    theta0            : constant phase offset [rad]
    theta1            : linear phase coeff [ps]   (approx -tau)
    theta2            : quadratic phase coeff [ps^2] (approx 0.5*delta_beta)
    V                 : fringe visibility [0, 1]
    offset            : detector background
    lambda0           : central wavelength [nm]   (typically fixed)
    delta_lambda_fwhm : FWHM bandwidth [nm]        (typically fixed)
    """
    # 1e-3 converts SPEED_OF_LIGHT/lam (nm, SI c) into rad/ps, so theta1/theta2 are
    # genuinely ps/ps^2 as documented above (equivalent to converting lam to meters
    # and theta1/theta2 to seconds; this form keeps the ratio Omega/sigma_w, and
    # hence the envelope shape, unaffected).
    omega  = 2.0 * np.pi * SPEED_OF_LIGHT / lam * 1e-3
    omega0 = 2.0 * np.pi * SPEED_OF_LIGHT / lambda0 * 1e-3
    domega_fwhm = 2.0 * np.pi * SPEED_OF_LIGHT / lambda0**2 * delta_lambda_fwhm * 1e-3
    sigma_w = domega_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    Omega = omega - omega0
    env   = A * np.exp(-Omega**2 / (2.0 * sigma_w**2))
    Theta = theta0 + theta1 * Omega + theta2 * Omega**2
    return env * (1.0 + V * np.cos(Theta)) + offset


def spectrum_fit_skew(lam, R, L, theta0, theta1, theta2,
                      alpha_R, epsilon_R, s_R, alpha_L, epsilon_L, s_L, offset,
                      lambda0, delta_lambda_fwhm):
    """
    Two-arm interference spectrum with an independently skewed R and L arm (eq:skewnorm, eq:V_skew).

    Shat_a(Omega) = exp[-(Omega-eps)^2/(2s^2)]*[1+erf(a*(Omega-eps)/(sqrt2*s))] / S0   (eq:skewnorm)
    B(Omega)      = R*Shat_{alpha_R}(Omega) + L*Shat_{alpha_L}(Omega)
    V(Omega)      = 2*sqrt(R*L*Shat_{alpha_R}(Omega)*Shat_{alpha_L}(Omega)) / B(Omega)   (eq:V_skew)
    I             = B(Omega)*(1 + V(Omega)*cos(Theta(Omega))) + offset

    Each arm reduces to a plain Gaussian at alpha=0, so alpha_R=alpha_L=0,
    epsilon_R=epsilon_L=0, s_R=s_L=sigma_w recovers the original single-Gaussian,
    constant-visibility spectrum_fit model exactly.

    lam               : wavelength axis [nm]
    R                 : R-arm (reference/DA) amplitude; peak-normalized Shat_{alpha_R}
    L                 : L-arm (grating/GA) amplitude; peak-normalized Shat_{alpha_L}
    theta0            : constant phase offset [rad]
    theta1            : linear phase coeff [ps]   (approx -tau)
    theta2            : quadratic phase coeff [ps^2] (approx 0.5*delta_beta)
    alpha_R           : R-arm skewness (alpha_R=0 recovers a Gaussian shape)
    epsilon_R         : R-arm skew-normal location [rad/ps]
    s_R               : R-arm skew-normal scale [rad/ps]
    alpha_L           : L-arm skewness (alpha_L=0 recovers a Gaussian shape)
    epsilon_L         : L-arm skew-normal location [rad/ps]
    s_L               : L-arm skew-normal scale [rad/ps]
    offset            : detector background
    lambda0           : central wavelength [nm]   (typically fixed)
    delta_lambda_fwhm : FWHM bandwidth [nm]        (typically fixed)
    """
    omega  = 2.0 * np.pi * SPEED_OF_LIGHT / lam * 1e-3
    omega0 = 2.0 * np.pi * SPEED_OF_LIGHT / lambda0 * 1e-3
    domega_fwhm = 2.0 * np.pi * SPEED_OF_LIGHT / lambda0**2 * delta_lambda_fwhm * 1e-3
    sigma_w = domega_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    Omega = omega - omega0

    # Peak location z* of exp(-z^2/2)*(1+erf(a*z/sqrt2)) depends only on alpha
    # (shift/scale-invariant). Solving its stationarity condition directly is
    # numerically fragile (erf saturates to +-1 in float64 for moderately large
    # |a*z|, making the derivative expression spuriously flat/zero far from the
    # true root), so maximize the shape itself instead: it is unimodal for
    # every real alpha, so a bounded 1D search is robust across the whole
    # range even as z* -> 0 for |a| -> infinity.

    # --- R arm (eq:skewnorm) ---
    if alpha_R == 0.0:
        z_star_R = 0.0
    else:
        def neg_shape_R(z):
            return -(np.exp(-z**2 / 2.0) * (1.0 + erf(alpha_R * z / np.sqrt(2.0))))
        z_star_R = minimize_scalar(neg_shape_R, bounds=(-8.0, 8.0), method="bounded").x
    S0_R = np.exp(-z_star_R**2 / 2.0) * (1.0 + erf(alpha_R * z_star_R / np.sqrt(2.0)))

    z_R = (Omega - epsilon_R) / s_R
    S_hat_R = np.exp(-z_R**2 / 2.0) * (1.0 + erf(alpha_R * z_R / np.sqrt(2.0))) / S0_R

    # --- L arm (eq:skewnorm) ---
    if alpha_L == 0.0:
        z_star_L = 0.0
    else:
        def neg_shape_L(z):
            return -(np.exp(-z**2 / 2.0) * (1.0 + erf(alpha_L * z / np.sqrt(2.0))))
        z_star_L = minimize_scalar(neg_shape_L, bounds=(-8.0, 8.0), method="bounded").x
    S0_L = np.exp(-z_star_L**2 / 2.0) * (1.0 + erf(alpha_L * z_star_L / np.sqrt(2.0)))

    z_L = (Omega - epsilon_L) / s_L
    S_hat_L = np.exp(-z_L**2 / 2.0) * (1.0 + erf(alpha_L * z_L / np.sqrt(2.0))) / S0_L

    B = R * S_hat_R + L * S_hat_L
    B_safe = np.where(B > 0.0, B, 1.0)
    V_omega = 2.0 * np.sqrt(np.clip(R * L * S_hat_R * S_hat_L, 0.0, None)) / B_safe

    Theta = theta0 + theta1 * Omega + theta2 * Omega**2
    return B * (1.0 + V_omega * np.cos(Theta)) + offset


@dataclass(frozen=True, slots=True)
class SphericalHarmonic:
    """
    Single spherical harmonic Y_l^m(theta, phi).

    Convention:
        theta = polar angle from +z axis
        phi   = azimuth angle in xy plane

    Uses the physics convention including the Condon-Shortley phase.
    """

    l: int
    m: int

    def __post_init__(self) -> None:
        if self.l < 0:
            raise ValueError("l must be >= 0")
        if abs(self.m) > self.l:
            raise ValueError("m must satisfy |m| <= l")

    def __call__(self, theta, phi) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64)
        phi = np.asarray(phi, dtype=np.float64)

        if self.m >= 0:
            return self._positive_m(self.l, self.m, theta, phi)

        m_abs = abs(self.m)

        # Y_l^{-m} = (-1)^m conj(Y_l^m)
        return (-1) ** m_abs * np.conjugate(
            self._positive_m(self.l, m_abs, theta, phi)
        )

    @staticmethod
    def _positive_m(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        norm = np.sqrt(
            (2 * l + 1)
            / (4 * pi)
            * factorial(l - m)
            / factorial(l + m)
        )

        return norm * lpmv(m, l, np.cos(theta)) * np.exp(1j * m * phi)