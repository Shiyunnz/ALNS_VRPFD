"""Unit tests for the drone energy consumption model."""

from __future__ import annotations

import math

import pytest

from alns_vrpfd.evaluation.energy import DroneEnergyModel


def test_power_baseline_matches_reference_value():
    model = DroneEnergyModel()

    power_kw = model.power_kw(payload_weight_kg=0.0)

    assert math.isclose(power_kw, 11.708630370784995, rel_tol=1e-9)


def test_energy_and_endurance_scaling():
    model = DroneEnergyModel()

    power_kw = model.power_kw(payload_weight_kg=5.0)
    energy_kwh = model.energy_kwh(payload_weight_kg=5.0, travel_time_hours=0.1)
    endurance = model.max_flight_time_hours(payload_weight_kg=5.0)

    assert math.isclose(power_kw, 13.90347099742954, rel_tol=1e-9)
    assert math.isclose(energy_kwh, 1.3903470997429541, rel_tol=1e-9)
    assert math.isclose(endurance, 0.63 / power_kw, rel_tol=1e-12)


def test_invalid_inputs_raise():
    model = DroneEnergyModel()

    with pytest.raises(ValueError):
        model.power_kw(payload_weight_kg=-0.1)

    with pytest.raises(ValueError):
        model.energy_kwh(payload_weight_kg=0.0, travel_time_hours=-1.0)

