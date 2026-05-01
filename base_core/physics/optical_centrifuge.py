from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from base_core.math.models import Angle
from base_core.physics.circular_chirped_puls import CircularChirpedPulse
from base_core.quantities.enums import Prefix, CircularHandedness
from base_core.quantities.models import Frequency, Length, Time
from base_core.quantities.specific_models import AngularChirp, AngularFrequency

# Common values from Kevins paper
CENTRAL_FREQUENCY = AngularFrequency.from_frequency(Frequency(375.0, Prefix.TERA)) # roughly 800nm
PHASE_0 = Angle(0)
T = Time(300.0, Prefix.PICO) # FWHM of the pulse
BETA_0 = AngularChirp(3.7e-2, time_prefix=Prefix.PICO) # Base chirp: 3.7e-2 rad / ps^2
DELTA_BETA = AngularChirp(5.9e-4, time_prefix=Prefix.PICO) # Chirp mismatch: 5.9e-4 rad / ps^2

DELTA_DELAY_ARM = Length(1.5, Prefix.MILLI)

@dataclass(frozen=True)
class OpticalCentrifuge:
    """
    Optical centrifuge built from two oppositely circularly polarized chirped arms.

    The two arms are evaluated at possibly different propagation lengths z_R and z_L.
    In vacuum, a relative path length difference is equivalent to a time delay:

        tau_eff = (z_L - z_R) / c

    Therefore this class does not need an explicit tau parameter if the delay is
    represented by z_R and z_L.

    The total Jones field is

        E_CFG(t) = E_R(t, z_R) + E_L(t, z_L)

    with each arm internally using retarded time

        t_ret = t - z / c.
    """

    right_arm = CircularChirpedPulse(1, CENTRAL_FREQUENCY, BETA_0, PHASE_0, T, CircularHandedness.RIGHT)
    left_arm = CircularChirpedPulse(1, CENTRAL_FREQUENCY, AngularChirp(BETA_0 + DELTA_BETA), PHASE_0, T, CircularHandedness.LEFT)

    def __post_init__(self) -> None:
        if self.right_arm.handedness == self.left_arm.handedness:
            raise ValueError(
                "An optical centrifuge requires two arms with opposite circular handedness."
            )

    def electric_field(
        self,
        t: float | NDArray[np.float64],
        z_R: Length | float = Length(0.0),
        z_L: Length | float = Length(0.0),
    ) -> NDArray[np.complex128]:
        """
        Return the total complex Jones field [Ex, Ey].

        If t is scalar:
            returns shape (2,)

        If t is array with shape (N,):
            returns shape (N, 2)
        """
        E_R = self.right_arm.electric_field(t, float(z_R))
        E_L = self.left_arm.electric_field(t, float(z_L))
        return E_R + E_L

    def x_field(
        self,
        t: float | NDArray[np.float64],
        z_R: Length | float = Length(0.0),
        z_L: Length | float = Length(0.0),
    ) -> complex | NDArray[np.complex128]:
        """
        Return the field after projection onto the laboratory x axis.
        """
        E = self.electric_field(t, z_R, z_L)
        return E[..., 0]

    def y_field(
        self,
        t: float | NDArray[np.float64],
        z_R: Length | float = Length(0.0),
        z_L: Length | float = Length(0.0),
    ) -> complex | NDArray[np.complex128]:
        """
        Return the field after projection onto the laboratory y axis.
        """
        E = self.electric_field(t, z_R, z_L)
        return E[..., 1]

    def linear_projection(
        self,
        t: float | NDArray[np.float64],
        angle: float,
        z_R: Length | float = Length(0.0),
        z_L: Length | float = Length(0.0),
    ) -> complex | NDArray[np.complex128]:
        """
        Project the centrifuge field onto a linear polarization axis.

        angle:
            Projection angle in radians, measured from x toward y.

        The projection unit vector is

            e = (cos(angle), sin(angle)).
        """
        E = self.electric_field(t, z_R, z_L)
        axis = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        return E @ axis

    def intensity(
        self,
        t: float | NDArray[np.float64],
        z_R: Length | float = Length(0.0),
        z_L: Length | float = Length(0.0),
    ) -> float | NDArray[np.float64]:
        """
        Return total Jones intensity |Ex|^2 + |Ey|^2.
        """
        E = self.electric_field(t, z_R, z_L)
        return np.real(np.sum(np.abs(E) ** 2, axis=-1))

    def x_intensity(
        self,
        t: float | NDArray[np.float64],
        z_R: Length | float = Length(0.0),
        z_L: Length | float = Length(0.0),
    ) -> float | NDArray[np.float64]:
        """
        Return intensity after an x polarizer:

            I_x(t) = |E_x(t)|^2.
        """
        return np.abs(self.x_field(t, z_R, z_L)) ** 2

    def projected_intensity(
        self,
        t: float | NDArray[np.float64],
        angle: float,
        z_R: Length | float = Length(0.0),
        z_L: Length | float = Length(0.0),
    ) -> float | NDArray[np.float64]:
        """
        Return intensity after projection onto a linear axis.

        angle:
            Projection angle in radians.
        """
        E_proj = self.linear_projection(t, angle, z_R, z_L)
        return np.abs(E_proj) ** 2

    def phase_difference(
        self,
        t: float | NDArray[np.float64],
        z_R: Length | float = Length(0.0),
        z_L: Length | float = Length(0.0),
    ) -> float | NDArray[np.float64]:
        """
        Return the optical phase difference

            Delta Phi(t) = Phi_R(t, z_R) - Phi_L(t, z_L).

        For equal arm amplitudes, this phase difference determines the
        instantaneous linear polarization angle.
        """
        return self.right_arm.phase(t, float(z_R)) - self.left_arm.phase(t, float(z_L))

    def polarization_angle(
        self,
        t: float | NDArray[np.float64],
        z_R: Length | float = Length(0.0),
        z_L: Length | float = Length(0.0),
    ) -> float | NDArray[np.float64]:
        """
        Return the ideal linear polarization angle of the centrifuge field.

        This assumes that the two circular components have equal amplitudes.
        With the convention

            e_R = (1, +i) / sqrt(2)
            e_L = (1, -i) / sqrt(2)

        the total field is proportional to

            (cos(Delta Phi / 2), -sin(Delta Phi / 2)),

        so the polarization angle is

            theta = -Delta Phi / 2.
        """
        return -0.5 * self.phase_difference(t, z_R, z_L)

    def instantaneous_angular_frequency_difference(
        self,
        t: float | NDArray[np.float64],
        z_R: Length | float = Length(0.0),
        z_L: Length | float = Length(0.0),
    ) -> float | NDArray[np.float64]:
        """
        Return

            Delta omega(t) = omega_R(t, z_R) - omega_L(t, z_L)

        in rad/s.
        """
        omega_R = self.right_arm.instantaneous_angular_frequency(t, float(z_R))
        omega_L = self.left_arm.instantaneous_angular_frequency(t, float(z_L))
        return omega_R - omega_L

    def centrifuge_angular_frequency(
        self,
        t: float | NDArray[np.float64],
        z_R: Length | float = Length(0.0),
        z_L: Length | float = Length(0.0),
    ) -> float | NDArray[np.float64]:
        """
        Return the angular rotation frequency of the linear polarization:

            Omega_CFG(t) = Delta omega(t) / 2

        in rad/s.
        """
        return 0.5 * self.instantaneous_angular_frequency_difference(t, z_R, z_L)

    def centrifuge_frequency(
        self,
        t: float | NDArray[np.float64],
        z_R: Length | float = Length(0.0),
        z_L: Length | float = Length(0.0),
    ) -> float | NDArray[np.float64]:
        """
        Return the centrifuge rotation frequency in cycles/s, i.e. Hz:

            f_CFG(t) = Omega_CFG(t) / (2 pi)
                     = Delta omega(t) / (4 pi).
        """
        return self.centrifuge_angular_frequency(t, z_R, z_L) / (2.0 * np.pi)