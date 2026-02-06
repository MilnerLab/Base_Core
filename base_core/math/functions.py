
from typing import Sequence
import numpy as np

from base_core.quantities.constants import SPEED_OF_LIGHT



def gaussian(x: Sequence[float], A, x0, sigma, offset):
    """
    1D Gaussian with constant offset.
    """
    xs = np.array(x, dtype=float)
    
    return A * np.exp(-((xs - x0) ** 2) / (2 * sigma ** 2)) + offset

#def erfc(x: Sequence[float], sigma, )

def usCFG_projection(
    wavelengths: Sequence[float],
    carrier_wavelength: float,
    starting_wavelength: float,
    bandwidth: float,
    baseline: float,
    phase: float,
    acceleration: float) -> list[float]:
    wavelengths_np = np.array(wavelengths, dtype=float)
    sigma = bandwidth / np.sqrt(8*np.log(2))
    # maybe not square gaussian
    return baseline + (1-baseline) * (gaussian(wavelengths, 1, carrier_wavelength, sigma, 0) * np.sin(phase + acceleration * (wavelengths_np - starting_wavelength)**2))**2

def cfCFG_projection(
#S = const +  (1-const)*(Gaussian(lambda - carrier,FWHM)*sin(phase + average*(lambda - carrier) + acceleration*(lambda - carrier)^3 )^2 )
    wavelengths: Sequence[float],
    carrier_wavelength: float,
    average_frequency: float,
    bandwidth: float,
    baseline: float,
    phase: float,
    acceleration: float) -> list[float]:
    wavelengths_np = np.array(wavelengths, dtype=float)
    sigma = bandwidth / np.sqrt(8*np.log(2))
    # maybe not square gaussian
    return baseline + (1-baseline) * (gaussian(wavelengths, 1, carrier_wavelength, sigma, 0) * np.sin(phase + average_frequency*(wavelengths_np - carrier_wavelength) + acceleration * (wavelengths_np - carrier_wavelength)**3))**2

def cfg_projection_nu_equal_amplitudes_safe(
    wavelengths_nm: Sequence[float],
    # envelope parameters (your gaussian in λ)
    env_A: float,
    env_x0_nm: float,
    env_sigma_nm: float,
    env_offset: float,
    # measurement/model parameters
    baseline: float,
    phase0: float,
    tau_ps: float,
    a_R_THz_per_ps: float,
    a_L_THz_per_ps: float,
) -> np.ndarray:
    """
    Uses your Gaussian as INTENSITY envelope in λ.
    Computes oscillation phase in ν-domain safely using THz/ps.
    """
    lam_nm = np.asarray(wavelengths_nm, dtype=float)
    baseline = float(np.clip(baseline, 0.0, 0.999999))

    if a_R_THz_per_ps == 0.0 or a_L_THz_per_ps == 0.0:
        raise ValueError("a_R_THz_per_ps and a_L_THz_per_ps must be nonzero.")

    # Intensity envelope vs λ using your gaussian
    I_env = gaussian(lam_nm, env_A, env_x0_nm, env_sigma_nm, env_offset)
    I_env = np.clip(I_env, 0.0, None)  # avoid negative intensities

    # ν in THz (Fourier variable)
    nu_thz = (SPEED_OF_LIGHT / (lam_nm * 1e-9)) * 1e-12
    nu0_thz = (SPEED_OF_LIGHT / (env_x0_nm * 1e-9)) * 1e-12  # reference at envelope center
    dnu_thz = nu_thz - nu0_thz

    # Chirp spectral phases (linear chirp model, no TOD)
    Phi_R = np.pi * (dnu_thz ** 2) / a_R_THz_per_ps
    Phi_L = np.pi * (dnu_thz ** 2) / a_L_THz_per_ps

    # Relative phase controlling the horizontal projection oscillation
    DeltaPhi = (Phi_R - Phi_L) + 2.0 * np.pi * nu_thz * tau_ps + phase0

    modulation = 0.5 * (1.0 + np.cos(DeltaPhi))  # in [0,1]

    I = baseline + (1.0 - baseline) * (I_env * modulation)
    return I
