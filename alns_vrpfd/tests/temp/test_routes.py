"""Temporary unit tests for TruckRoute and DroneTask scaffolds."""

from __future__ import annotations

import math

from alns_vrpfd.evaluation.energy import DroneEnergyModel
from alns_vrpfd.evaluation.robustness import RobustnessChecker
from alns_vrpfd.model.route import (
    DroneTask,
    DroneTaskContext,
    DroneTaskTiming,
    TruckRoute,
)
from alns_vrpfd.model.solution import Solution


def test_truckroute_customers_editing():
    route = TruckRoute(route_id=1, nodes=[0, 1, 3], capacity=100.0)
    assert route.customers() == [1]
    route.insert_customer(1, 2)
    assert route.customers() == [1, 2]
    route.swap_customers(0, 1)
    assert route.customers() == [2, 1]
    route.remove_customer(1)
    assert route.customers() == [2]


def test_dronetask_payload_resize():
    task = DroneTask(
        task_id=1,
        drone_id=0,
        launch_truck=0,
        launch_node=1,
        customers=[2],
        land_truck=0,
        retrieve_node=3,
        payloads=[0.0, 0.0],
    )
    assert len(task.payloads) == 2
    task.insert_customer(1, 3)
    assert len(task.payloads) == 3


def test_dronetask_feasibility_success():
    task = DroneTask(
        task_id=2,
        drone_id=0,
        launch_truck=0,
        launch_node=1,
        customers=[2],
        land_truck=0,
        retrieve_node=3,
        payloads=[5.0, 0.0],
    )
    context = DroneTaskContext(
        valid_nodes={0, 1, 2, 3},
        valid_trucks={0},
        served_customers=set(),
        truck_routes={0: [0, 1, 3, 0]},
        truck_arrival_times={0: {1: 10.0, 3: 30.0}},
        wait_max=10.0,
        drone_capacity=5.0,
        customer_demands={2: 5.0},
        customer_latest_times={2: 25.0},
        drone_schedule={0: [(0.0, 8.0), (40.0, 60.0)]},
        timing=DroneTaskTiming(
            launch_time=10.0,
            customer_arrival_times={2: 20.0},
            retrieve_time=35.0,
        ),
    )

    assert task.is_feasible(context)
    assert task.feasibility_errors(context) == []


def test_dronetask_requires_customers():
    task = DroneTask(
        task_id=3,
        drone_id=1,
        launch_truck=0,
        launch_node=1,
        customers=[],
        land_truck=0,
        retrieve_node=2,
        payloads=[0.0],
    )

    errors = task.feasibility_errors()
    assert any("at least one customer" in msg for msg in errors)


def test_dronetask_detects_payload_and_timing_violations():
    task = DroneTask(
        task_id=4,
        drone_id=1,
        launch_truck=0,
        launch_node=1,
        customers=[2],
        land_truck=0,
        retrieve_node=3,
        payloads=[4.0, 4.0],
    )
    context = DroneTaskContext(
        valid_nodes={0, 1, 2, 3},
        valid_trucks={0},
        served_customers={2},
        truck_routes={0: [0, 1, 3, 0]},
        truck_arrival_times={0: {1: 10.0, 3: 20.0}},
        wait_max=2.0,
        drone_capacity=3.0,
        customer_demands={2: 1.5},
        customer_latest_times={2: 12.0},
        drone_schedule={1: [(0.0, 12.0)]},
        timing=DroneTaskTiming(
            launch_time=9.0,
            customer_arrival_times={2: 15.0},
            retrieve_time=25.0,
        ),
    )

    errors = task.feasibility_errors(context)
    assert any("already served" in msg for msg in errors)
    assert any("exceeds drone capacity" in msg for msg in errors)
    assert any("Payload drop" in msg for msg in errors)
    assert any("Drone launches before truck arrival" in msg for msg in errors)
    assert any("served after latest time" in msg for msg in errors)
    assert any("rendezvous exceeds wait limit" in msg for msg in errors)
    assert any("overlaps" in msg for msg in errors)


def test_dronetask_validates_truck_route_nodes():
    task = DroneTask(
        task_id=5,
        drone_id=2,
        launch_truck=0,
        launch_node=1,
        customers=[2],
        land_truck=0,
        retrieve_node=3,
        payloads=[2.0, 1.0],
    )
    context = DroneTaskContext(
        truck_routes={0: [0, 1, 0]},
    )

    errors = task.feasibility_errors(context)
    assert any("Retrieve node" in msg for msg in errors)


def test_dronetask_detects_duplicate_customers():
    task = DroneTask(
        task_id=6,
        drone_id=3,
        launch_truck=0,
        launch_node=1,
        customers=[2, 2],
        land_truck=0,
        retrieve_node=3,
        payloads=[1.0, 0.5, 0.0],
    )
    context = DroneTaskContext(served_customers={2})

    errors = task.feasibility_errors(context)
    assert any("Duplicate customers" in msg for msg in errors)
    assert any("already served" in msg for msg in errors)


def test_dronetask_energy_budget_violation():
    energy_model = DroneEnergyModel()
    task = DroneTask(
        task_id=10,
        drone_id=4,
        launch_truck=0,
        launch_node=1,
        customers=[2],
        land_truck=0,
        retrieve_node=3,
        payloads=[2.0, 0.0],
    )

    context = DroneTaskContext(
        drone_capacity=5.0,
        customer_demands={2: 2.0},
        timing=DroneTaskTiming(
            launch_time=0.0,
            customer_arrival_times={2: 0.5},
            retrieve_time=1.0,
        ),
        energy_model=energy_model,
        energy_uncertainty_budget=1.0,
        drone_energy_capacity=12.0,
        energy_deviation_rate=0.1,
        energy_tolerance=1e-6,
        time_tolerance=1e-6,
    )

    errors = task.feasibility_errors(context)
    assert any("Energy budget violation" in msg for msg in errors)


def test_robustness_checker_handles_per_drone_budgets():
    energy_model = DroneEnergyModel()

    task_a = DroneTask(
        task_id=21,
        drone_id=1,
        launch_truck=0,
        launch_node=1,
        customers=[2],
        land_truck=0,
        retrieve_node=3,
        payloads=[2.0, 0.0],
    )
    task_b = DroneTask(
        task_id=22,
        drone_id=2,
        launch_truck=0,
        launch_node=1,
        customers=[2],
        land_truck=0,
        retrieve_node=3,
        payloads=[2.0, 0.0],
    )

    timing_a = DroneTaskTiming(
        launch_time=0.0,
        customer_arrival_times={2: 0.5},
        retrieve_time=1.0,
    )
    timing_b = DroneTaskTiming(
        launch_time=0.0,
        customer_arrival_times={2: 0.5},
        retrieve_time=1.0,
    )

    context_a = DroneTaskContext(
        drone_capacity=5.0,
        customer_demands={2: 2.0},
        timing=timing_a,
        energy_model=energy_model,
        energy_uncertainty_budget={1: 1.0},
        drone_energy_capacity={1: 12.5},
        energy_deviation_rate=0.1,
        energy_tolerance=1e-6,
        time_tolerance=1e-6,
    )

    context_b = DroneTaskContext(
        drone_capacity=5.0,
        customer_demands={2: 2.0},
        timing=timing_b,
        energy_model=energy_model,
        energy_uncertainty_budget={2: 0.0},
        drone_energy_capacity={2: 20.0},
        energy_deviation_rate=0.1,
        energy_tolerance=1e-6,
        time_tolerance=1e-6,
    )

    solution = Solution(drone_tasks=[task_a, task_b])
    checker = RobustnessChecker(
        energy_model=energy_model,
        battery_capacity={1: 12.5, 2: 20.0},
        energy_uncertainty_budget={1: 1.0, 2: 0.0},
    )

    contexts = {task_a.task_id: context_a, task_b.task_id: context_b}
    result = checker.check(solution, contexts=contexts)

    assert result.feasible is False
    breakdown = {entry.drone_id: entry for entry in result.task_breakdown}
    assert breakdown[1].uncertainty_budget == 1.0
    assert breakdown[2].uncertainty_budget == 0.0
    assert breakdown[1].feasible is False
    assert breakdown[2].feasible is True
    assert math.isclose(
        breakdown[2].worst_case_energy, breakdown[2].nominal_energy, rel_tol=1e-9
    )
