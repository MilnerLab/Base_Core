from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, TypeAlias

from numpy import pi

from base_core.math.enums import CartesianAxis
from base_core.quantities.enums import Prefix
from base_core.quantities.constants import HBAR_J_S, BOLTZMANN_J_K
from base_core.quantities.models import Frequency, InverseLength, Length, Temperature
from base_core.quantities.specific_models import AtomicMass, Intensity, PolarizabilityVolume, Energy
from base_core.physics.molecules import Molecule

def inertia(B: Frequency = None):
    return HBAR_J_S**2 / (2 * B)

def rot_energy(mol: Molecule = None, j: int = 0, *, droplet: bool = False) -> Energy:
    if droplet: 
        return Energy(mol.droplet.B.value() * j * (j+1) - mol.droplet.D.value() * j**2 * (j+1)**2)
    else:
        if mol.gasphase.D is not None:
            return Energy(mol.gasphase.B.value() * j * (j+1) - mol.gasphase.D.value() * j**2 * (j+1)**2)
        else:
            return Energy(mol.gasphase.B.value() * j * (j+1))

def twophoton(mol: Molecule = None, j_initial: int = 0, droplet: bool = False) -> Frequency:
    """ The two-photon frequency of the delta J = 2 transistion of a molecule.

    Args:
        mol (Molecule, optional): the molecule class is needed for the B and sometimes also D
        j_initial (int, optional): _description_. Defaults to 0.

    Returns:
        Frequency: default prefix set to GIGA
    """
    if droplet: 
        return Frequency(rot_energy(mol,j_initial+2,droplet=True) - rot_energy(mol,j_initial,droplet=True))
    else:
        return Frequency(rot_energy(mol,j_initial+2,droplet=False) - rot_energy(mol,j_initial,droplet=False))
    
'''def boltzmann(mol: Molecule, T: Temperature,tolerance: float = 0.001):
    if tolerance <= 0: 
        raise ValueError("Tolerance must be greater than 0.")
    m = mol.mass.value()
    k = BOLTZMANN_J_K
    C = k * T * log(tolerance)
    j =  
    J = np.arange(0,J_max)
    if T.value() < 1: 
        E = rot_energy(mol,J,droplet=True)
    else:
        E = rot_energy(mol,J,droplet=False)
        
    y = (m/(2*pi*k*T.value()))**1.5 * np.exp(-E / k / T.value())'''