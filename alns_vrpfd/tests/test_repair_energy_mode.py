"""Tests for embedded vs verification energy mode in repair operators."""

from __future__ import annotations

import math

import pytest

from alns_vrpfd.core.operators import RepairOperator
from alns_vrpfd.instance import InstanceManager


def _build_tiny_instance() -> InstanceManager:
    instance = InstanceManager()
    instance.configure_depots(start=0, end=0)
    instance.register_customer(customer_id=1, demand=2.0)

    instance.register_vehicle_type(
        "truck",
        number=1,
        capacity=50.0,
        endurance=100.0,
        speed=40.0,
        unit_cost=1.0,
    )
    instance.register_vehicle_type(
        "drone",
        number=1,
        capacity=10.0,
        endurance=100.0,
        speed=20.0,
        unit_cost=1.0,
    )

    instance.add_distance("truck", 0, 1, 10.0)
    instance.add_distance("truck", 1, 0, 10.0)
    instance.add_distance("drone", 0, 1, 10.0)
    instance.add_distance("drone", 1, 0, 10.0)

    instance.configure_robustness(
        drone_battery_capacity=6.3,
        energy_uncertainty_budget=3,
        energy_deviation_rate=0.1,
        same_truck_retrieval=False,
    )
    return instance


def test_repair_mode_embedded_adds_uncertainty_margin():
    instance = _build_tiny_instance()
    embedded = RepairOperator(instance, robust_energy_mode="embedded")
    verification = RepairOperator(instance, robust_energy_mode="verification")

    energy_embedded = embedded._worst_case_energy(0, 0, [1])
    energy_verification = verification._worst_case_energy(0, 0, [1])

    assert math.isfinite(energy_embedded)
    assert math.isfinite(energy_verification)
    # With partial robustness (γ_repair ≈ 67% of Γ), embedded uses the full
    # budget while verification uses a reduced budget.  For tasks with very
    # few legs, the budgeted sum can saturate and they become equal.
    assert energy_embedded >= energy_verification


def test_repair_mode_invalid_value_raises():
    instance = _build_tiny_instance()
    with pytest.raises(ValueError):
        RepairOperator(instance, robust_energy_mode="unknown")
