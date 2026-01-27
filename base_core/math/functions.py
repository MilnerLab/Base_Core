
from typing import Sequence
import numpy as np



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