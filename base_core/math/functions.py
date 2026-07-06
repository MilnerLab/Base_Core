from __future__ import annotations

from dataclasses import dataclass, field
from math import factorial, pi
from typing import Mapping, Sequence

from base_core.quantities import constants
import numpy as np
from scipy.special import lpmv

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