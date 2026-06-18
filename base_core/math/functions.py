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

def cfg_spectrum(lam, A, dphi0, delta_z, delta_beta, offset,
                 lambda0, delta_lambda_fwhm):
    """
    Centrifuge interference spectrum as a function of wavelength.

    Inputs
    ------
    lam               : wavelength axis [nm]
    A                 : amplitude (absorbs factor 4 and |E|^2)
    dphi0             : relative phase phi_R0 - phi_L0 [rad]
    delta_z           : arm path-length difference (z_R - z_L) [mm]; tau = delta_z / c
    dbeta2            : spectral GDD difference beta2_R - beta2_L [ps^2]
    offset            : detector background
    lambda0           : central wavelength [nm]   (fix)
    delta_lambda_fwhm : FWHM bandwidth [nm]        (fix)

    The quadratic phase lives in angular frequency, so the wavelength axis is
    converted to omega internally. The spectral envelope is Gaussian in omega
    (not in lambda). Fixing lambda0 and delta_lambda_fwhm leaves A, dphi0,
    delta_z, dbeta2, offset free.
    """
    omega  = 2.0 * np.pi * SPEED_OF_LIGHT / lam       # rad/ps
    omega0 = 2.0 * np.pi * SPEED_OF_LIGHT / lambda0   # rad/ps

    # FWHM bandwidth [nm] -> spectral intensity std [rad/ps]
    domega_fwhm = 2.0 * np.pi * SPEED_OF_LIGHT / lambda0**2 * delta_lambda_fwhm
    sigma_w     = domega_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    Omega = omega - omega0
    tau   = delta_z / SPEED_OF_LIGHT                  # ps
    env   = A * np.exp(-Omega**2 / (2.0 * sigma_w**2))
    Theta = dphi0 - tau * Omega + 0.5 * delta_beta * Omega**2
    return env * np.cos(Theta / 2.0)**2 + offset

def cfCFG_spectrum(lam, A, dphi0, delta_z, offset,
                 lambda0, delta_lambda_fwhm):
    return cfg_spectrum(lam, A, dphi0, delta_z, 0, offset, lambda0, delta_lambda_fwhm)

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