from dataclasses import dataclass
import numpy as np
from enum import Enum
from typing import Any

from numpy.typing import NDArray

from base_core.math.models import Angle
from base_core.quantities.constants import SPEED_OF_LIGHT
from base_core.quantities.enums import CircularHandedness, Prefix
from base_core.quantities.models import Frequency, Length, Length, Time
from base_core.quantities.specific_models import AngularChirp, AngularFrequency




@dataclass(frozen=True)
class CircularChirpedPulse():
    """
    One circularly polarized chirped Gaussian pulse arm.

    Vacuum propagation is implemented via retarded time:
        t_ret = t - z/c

    Envelope:
        A(t_ret) = A0 exp[-t_ret^2/(2T^2)]

    Phase:
        Phi(t_ret) = omega0*t_ret + 0.5*chirp*t_ret^2 + phi0

    Jones vectors:
        RIGHT = (1, +i)/sqrt(2)
        LEFT  = (1, -i)/sqrt(2)
    """

    A0: float
    omega0: AngularFrequency
    chirp: AngularChirp
    phi0: Angle
    width: Time
    handedness: CircularHandedness
    c: float = SPEED_OF_LIGHT

    @staticmethod
    def _to_float_or_array(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.astype(float, copy=False)
        return float(value)

    def retarded_time(self, t: Time | float | NDArray[np.float64], z: Length | float = 0.0) -> Any:
        return self._to_float_or_array(t) - float(z) / self.c

    def envelope(self, t: Time | float | NDArray[np.float64], z: Length | float = 0.0) -> Any:
        tr = self.retarded_time(t, z)
        return self.A0 * np.exp(-(tr**2) / (2.0 * float(self.width) ** 2))

    def phase(self, t: Time | float | NDArray[np.float64], z: Length | float = 0.0) -> Any:
        tr = self.retarded_time(t, z)
        return float(self.omega0) * tr + 0.5 * float(self.chirp) * tr**2 + self.phi0

    def instantaneous_angular_frequency(self, t: Time | float | NDArray[np.float64], z: Length | float = 0.0) -> Any:
        tr = self.retarded_time(t, z)
        return float(self.omega0) + float(self.chirp) * tr

    def instantaneous_frequency(self, t: Time | float | NDArray[np.float64], z: Length | float = 0.0) -> Any:
        return self.instantaneous_angular_frequency(t, z) / (2.0 * np.pi)

    @property
    def jones_vector(self) -> NDArray[np.complex128]:
        sign = self.handedness.value
        return np.array([1.0, 1j * sign], dtype=np.complex128) / np.sqrt(2.0)

    def scalar_field(self, t: Time | float | NDArray[np.float64], z: Length | float = 0.0) -> Any:
        return self.envelope(t, z) * np.exp(1j * self.phase(t, z))

    def electric_field(self, t: Time | float | NDArray[np.float64], z: Length | float = 0.0) -> NDArray[np.complex128]:
        scalar = np.asarray(self.scalar_field(t, z), dtype=np.complex128)
        if scalar.ndim == 0:
            return scalar * self.jones_vector
        return scalar[..., np.newaxis] * self.jones_vector

    def real_electric_field(self, t: Time | float | NDArray[np.float64], z: Length | float = 0.0) -> NDArray[np.float64]:
        return np.real(self.electric_field(t, z))