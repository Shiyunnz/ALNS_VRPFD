"""Tests for supply-class deprivation costs and deadlines."""

from __future__ import annotations

import math

from alns_vrpfd.deprivation import (
    WANG_SUPPLY_CLASSES,
    deprivation_cost,
    HOLGUIN_INTERCEPT,
    MAX_TARDINESS_HOURS,
)
from alns_vrpfd.evaluation import Evaluator
from alns_vrpfd.instance import InstanceManager, TimeWindowConfig
from alns_vrpfd.model.route import TruckRoute
from alns_vrpfd.model.solution import Solution


def test_water_cost_with_default_params():
    tau = 0.25
    cost = deprivation_cost(tau, "water", cost_lambda=12.0, rho=1.0, normalized=True)
    assert cost > 0.0
    assert cost < 12.0 * 1.35


def test_class_cost_ordering():
    """Classes with same omega have same cost; higher omega > lower omega."""
    tau = 1.0
    medicine = deprivation_cost(tau, "medicine", cost_lambda=30.0, rho=0.2083, normalized=True)
    water = deprivation_cost(tau, "water", cost_lambda=30.0, rho=0.2083, normalized=True)
    food = deprivation_cost(tau, "food", cost_lambda=30.0, rho=0.2083, normalized=True)
    tent = deprivation_cost(tau, "tent", cost_lambda=30.0, rho=0.2083, normalized=True)
    assert math.isclose(medicine, water, rel_tol=1e-2)
    assert medicine > food > tent


def test_zero_tau_returns_zero():
    assert deprivation_cost(0.0, "medicine") == 0.0
    assert deprivation_cost(0.0, "water") == 0.0
    assert deprivation_cost(0.0, "food") == 0.0
    assert deprivation_cost(0.0, "tent") == 0.0


def test_max_tardiness_equals_lambda_times_omega():
    """At tau=MAX_TARDINESS_HOURS, normalized cost should be lambda * omega."""
    for cls, spec in WANG_SUPPLY_CLASSES.items():
        cost = deprivation_cost(MAX_TARDINESS_HOURS, cls, cost_lambda=30.0, rho=0.2083, normalized=True)
        assert math.isclose(cost, 30.0 * spec.omega, rel_tol=1e-10)


def test_non_normalized_increases_with_tau():
    """Non-normalized raw exponential should be strictly increasing."""
    prev = 0.0
    for tau in [0.1, 0.5, 1.0, 2.0, 3.0, MAX_TARDINESS_HOURS]:
        cost = deprivation_cost(tau, "water", cost_lambda=1.0, rho=1.0, normalized=False)
        assert cost > prev
        prev = cost


def test_rho_compression_non_normalized():
    """Raw (non-normalized) cost increases with rho at same tau."""
    tau = 0.5
    low = deprivation_cost(tau, "water", cost_lambda=1.0, rho=0.2083, normalized=False)
    high = deprivation_cost(tau, "water", cost_lambda=1.0, rho=1.0, normalized=False)
    assert high > low


def test_class_based_time_windows_assign_four_supply_classes_and_order_deadlines():
    instance = InstanceManager()
    instance.configure_depots(start=0, end=99)
    for cid in range(1, 9):
        instance.register_customer(cid, demand=float(cid), location_x=float(cid), location_y=0.0)
    instance.register_vehicle_type("truck", number=1, capacity=50.0, endurance=480.0, speed=40.0, unit_cost=2.0)
    instance.register_vehicle_type("drone", number=1, capacity=10.0, endurance=120.0, speed=60.0, unit_cost=3.0)

    for origin in instance.all_node_ids():
        for destination in instance.all_node_ids():
            distance = abs(float(destination - origin)) if origin != destination else 0.0
            instance.add_distance("truck", origin, destination, distance)
            instance.add_distance("drone", origin, destination, distance)

    instance.generate_time_windows(
        strategy="class_based",
        config=TimeWindowConfig(class_seed=3),
    )

    classes = {
        instance.customer_manager.supply_class(cid)
        for cid in instance.customer_manager.customer_ids()
    }
    assert classes == {"medicine", "water", "food", "tent"}

    optimal_by_class = {}
    for cid in instance.customer_manager.customer_ids():
        supply_class = instance.customer_manager.supply_class(cid)
        optimal_by_class.setdefault(supply_class, []).append(
            instance.customer_manager.time_window(cid)[0]
        )

    assert min(optimal_by_class["medicine"]) < min(optimal_by_class["water"])
    assert min(optimal_by_class["water"]) < min(optimal_by_class["food"])
    assert min(optimal_by_class["food"]) < min(optimal_by_class["tent"])


def test_evaluator_uses_customer_supply_class_for_delay_cost():
    instance = InstanceManager()
    instance.configure_depots(start=0, end=0)
    instance.register_customer(1, demand=1.0)
    instance.customer_manager.assign_supply_class(1, "medicine")
    instance.customer_manager.assign_time_window(1, optimal=0.0, latest=10.0)
    instance.register_vehicle_type("truck", number=1, capacity=10.0, endurance=480.0, speed=1.0, unit_cost=0.0)
    instance.register_vehicle_type("drone", number=0, capacity=0.0, endurance=0.0, speed=1.0, unit_cost=0.0)
    instance.add_distance("truck", 0, 1, 2.0)
    instance.add_distance("truck", 1, 0, 2.0)
    instance.add_distance("drone", 0, 1, 2.0)
    instance.add_distance("drone", 1, 0, 2.0)

    solution = Solution(truck_routes=[TruckRoute(route_id=0, nodes=[0, 1, 0], capacity=10.0)])

    details = Evaluator(instance).evaluate_with_details(solution)

    assert math.isclose(
        details.result.delay_penalty,
        deprivation_cost(2.0, "medicine"),
        rel_tol=1e-12,
    )
