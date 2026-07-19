"""Reusable drone energy consumption model utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["DroneEnergyModel"]


# Base parameters sourced from cited literature for the default model.
DRONE_BODY_WEIGHT_KG = 37.0
BATTERY_WEIGHT_KG = 4.2
GRAVITATIONAL_ACCEL_N_PER_KG = 9.81
AIR_DENSITY_KG_PER_CUBIC_M = 1.204
SPINNING_BLADE_DISC_AREA_SQ_M = 0.025
ROTOR_COUNT = 8
BATTERY_CAPACITY_KWH = 0.63


@dataclass(frozen=True)
class DroneEnergyModel:
    """Provide power and energy consumption estimates for drone flights.

    The power curve follows

        P(q) = (W + m + q)^(3/2) * sqrt(g^3 / (2 * rho * varsigma * h))

    where ``q`` is the payload weight in kilograms. Flight energy is computed as

        e_ij = P(q) * t_ij

    with ``t_ij`` the flight duration in hours so that results are in kWh.

    Parameters default to values reported in the referenced literature and can
    be overridden per instance when calibration data is available.
    """

    body_weight_kg: float = DRONE_BODY_WEIGHT_KG
    battery_weight_kg: float = BATTERY_WEIGHT_KG
    gravitational_accel: float = GRAVITATIONAL_ACCEL_N_PER_KG
    air_density: float = AIR_DENSITY_KG_PER_CUBIC_M
    disc_area: float = SPINNING_BLADE_DISC_AREA_SQ_M
    rotor_count: int = ROTOR_COUNT
    battery_capacity_kwh: float = BATTERY_CAPACITY_KWH

    def power_kw(self, payload_weight_kg: float) -> float:
        """Return instantaneous power draw for a payload weight in kilograms."""

        if payload_weight_kg < 0:
            raise ValueError("Payload weight must be non-negative.")

        effective_mass = self.body_weight_kg + self.battery_weight_kg + payload_weight_kg
        if effective_mass < 0:
            raise ValueError("Effective mass must remain non-negative.")

        pre_factor = math.sqrt(
            (self.gravitational_accel ** 3)
            / (2.0 * self.air_density * self.disc_area * self.rotor_count)
        )

        return (effective_mass ** 1.5) * pre_factor / 1000.0

    def energy_kwh(self, payload_weight_kg: float, travel_time_hours: float) -> float:
        """Return energy consumed over a flight duration in hours."""

        if travel_time_hours < 0:
            raise ValueError("Travel time must be non-negative.")

        return self.power_kw(payload_weight_kg) * travel_time_hours

    def max_flight_time_hours(self, payload_weight_kg: float) -> float:
        """Return endurance in hours for the configured battery capacity."""

        power_kw = self.power_kw(payload_weight_kg)
        if power_kw <= 0:
            raise ValueError("Power draw must be positive to compute endurance.")
        return self.battery_capacity_kwh / power_kw
