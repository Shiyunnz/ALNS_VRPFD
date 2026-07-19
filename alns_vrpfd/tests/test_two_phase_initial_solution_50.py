"""Regression tests for 50-customer two-phase initial solutions."""

from __future__ import annotations

import pytest

from alns_vrpfd.model.initializer import build_two_phase_initial_solution
from alns_vrpfd.utils.config_loader import ALNSConfig
from alns_vrpfd.utils.io_utils import read_instance
from revision.tune_base import (
    ClassWeightedEvaluator,
    TIME_TOLERANCE_HOURS,
    generate_class_based_deadlines,
)


@pytest.mark.parametrize("seed", [101, 102, 103])
@pytest.mark.parametrize("instance_name", [f"R_30_50_{idx}" for idx in range(1, 6)])
def test_two_phase_initial_solution_keeps_oversized_customers_on_trucks(
    instance_name: str,
    seed: int,
):
    config = ALNSConfig("config/alns_config.yaml")
    instance = read_instance(
        f"data/Instance50/{instance_name}.txt",
        strategy="class_based",
        apply_time_windows=False,
    )
    generate_class_based_deadlines(instance, seed=seed)

    solution = build_two_phase_initial_solution(instance)

    drone_capacity = instance.vehicle_specs["drone"].capacity
    demands = instance.customer_manager.demands()
    truck_customers = {
        customer
        for route in solution.truck_routes
        for customer in route.customers()
    }
    oversized = {
        customer for customer, demand in demands.items()
        if demand > drone_capacity
    }

    assert oversized <= truck_customers


@pytest.mark.parametrize("instance_name", [f"R_30_50_{idx}" for idx in range(1, 6)])
def test_two_phase_initial_solution_is_feasible_for_r30_50_seed101(
    instance_name: str,
):
    config = ALNSConfig("config/alns_config.yaml")
    instance = read_instance(
        f"data/Instance50/{instance_name}.txt",
        strategy="class_based",
        apply_time_windows=False,
    )
    instance.vehicle_specs["drone"].endurance = float("inf")
    instance.configure_robustness(
        drone_battery_capacity=config.drone_battery_capacity,
        energy_uncertainty_budget=config.energy_uncertainty_budget,
        energy_deviation_rate=config.energy_deviation_rate,
        same_truck_retrieval=config.same_truck_retrieval,
    )
    classes = generate_class_based_deadlines(instance, seed=101)
    evaluator = ClassWeightedEvaluator(
        instance,
        classes,
        rendezvous_tolerance=config.drone_rendezvous_tolerance,
        forced_drone_customers=config.forced_drone_customers,
        allow_multiple_launch_per_node=config.relax_allow_multiple_launch_per_node,
        cost_lambda=config.cost_lambda,
        cost_rho=config.cost_rho,
        cost_normalized=config.cost_normalized,
        time_tolerance=TIME_TOLERANCE_HOURS,
    )

    solution = build_two_phase_initial_solution(
        instance,
        allow_multiple_launch_per_node=config.relax_allow_multiple_launch_per_node,
    )
    result = evaluator.evaluate_solution(solution)

    assert result.feasible
