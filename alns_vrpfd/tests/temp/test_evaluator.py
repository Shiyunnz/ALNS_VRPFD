"""Tests for the detailed evaluator pipeline."""

from __future__ import annotations

import math

from alns_vrpfd.deprivation import deprivation_cost
from alns_vrpfd.evaluation import Evaluator
from alns_vrpfd.instance import InstanceManager
from alns_vrpfd.model.route import DroneTask, TruckRoute
from alns_vrpfd.model.solution import Solution


def _build_toy_instance() -> InstanceManager:
    instance = InstanceManager()
    instance.configure_depots(start=0, end=0)

    instance.register_customer(customer_id=1, demand=5.0)
    instance.register_customer(customer_id=2, demand=3.0)

    instance.customer_manager.assign_time_window(1, optimal=0.2, latest=1.0)
    instance.customer_manager.assign_time_window(2, optimal=0.35, latest=1.0)

    instance.register_vehicle_type(
        "truck",
        number=1,
        capacity=50.0,
        endurance=480.0,
        speed=40.0,
        unit_cost=2.0,
    )
    instance.register_vehicle_type(
        "drone",
        number=1,
        capacity=10.0,
        endurance=120.0,
        speed=60.0,
        unit_cost=3.0,
    )

    edges = {
        (0, 1): 10.0,
        (1, 0): 10.0,
        (1, 2): 8.0,
        (2, 0): 10.0,
        (2, 1): 8.0,
        (0, 2): 12.0,
    }
    for (origin, destination), distance in edges.items():
        instance.add_distance("truck", origin, destination, distance)
        instance.add_distance("drone", origin, destination, distance)

    return instance


def _build_solution() -> Solution:
    truck_route = TruckRoute(route_id=0, nodes=[0, 1, 0], capacity=50.0)
    drone_task = DroneTask(
        task_id=1,
        drone_id=0,
        launch_truck=0,
        launch_node=1,
        customers=[2],
        land_truck=0,
        retrieve_node=0,
        payloads=[5.0, 0.0],
    )
    solution = Solution(truck_routes=[truck_route], drone_tasks=[drone_task])
    return solution


def test_evaluator_detailed_results():
    instance = _build_toy_instance()
    solution = _build_solution()

    evaluator = Evaluator(
        instance,
        truck_cost_per_km=2.0,
        drone_cost_per_km=3.0,
        rendezvous_tolerance=0.2,
    )

    details = evaluator.evaluate_with_details(solution)

    # Truck arrival at customer 1
    truck_arrival = details.truck_timings[0].arrival_times[1]
    assert math.isclose(truck_arrival, 1.35, rel_tol=1e-9)

    # Drone arrival at customer 2
    drone_arrival = details.drone_timings[1].customer_arrival_times[2]
    assert math.isclose(drone_arrival, 1.4833333333333332, rel_tol=1e-6)

    rendezvous = details.rendezvous_results[1]
    assert rendezvous.feasible
    assert rendezvous.deviation <= 0.2

    delay_penalty = details.delay_breakdown.total_delay
    # Truck delay at node 1: 1.35 - 0.2 = 1.15
    # Drone delay at node 2: 1.4833... - 0.35 ≈ 1.1333
    # Note: solution has hard TW violations (arrival > latest=1.0), so infeasible
    tau_truck = 1.35 - 0.2
    tau_drone = 1.4833333333333332 - 0.35
    expected_delay = deprivation_cost(tau_truck, "water") + deprivation_cost(tau_drone, "water")
    assert math.isclose(delay_penalty, expected_delay, rel_tol=1e-6)

    truck_cost = 2.0 * (10.0 + 10.0)
    drone_cost = 3.0 * (8.0 + 10.0)
    expected_base = truck_cost + drone_cost + expected_delay
    assert math.isclose(details.result.truck_distance_cost, truck_cost, rel_tol=1e-9)
    assert math.isclose(details.result.drone_distance_cost, drone_cost, rel_tol=1e-9)
    assert math.isclose(details.result.delay_penalty, expected_delay, rel_tol=1e-4)
    # Solution is infeasible due to hard time-window violations
    assert not details.result.feasible


def test_evaluator_two_tasks_same_anchor_are_separate():
    """Ensure evaluator returns separate results for tasks that share the same launch/retrieve node.

    Previously, the evaluator keyed tasks by task_id which could collide. The evaluator
    now disambiguates keys when task_id duplicates are present and returns separate
    timings/robustness entries per task.
    """
    instance = _build_toy_instance()
    # Add a third customer for the second drone
    instance.register_customer(customer_id=3, demand=2.0)
    instance.customer_manager.assign_time_window(3, optimal=0.5, latest=2.0)

    # Add distances to/from new node 3
    instance.add_distance('truck', 3, 0, 12.0)
    instance.add_distance('truck', 0, 3, 12.0)
    instance.add_distance('truck', 2, 3, 8.0)
    instance.add_distance('truck', 3, 2, 8.0)
    instance.add_distance('truck', 1, 3, 12.0)
    instance.add_distance('truck', 3, 1, 12.0)
    instance.add_distance('drone', 1, 3, 12.0)
    instance.add_distance('drone', 3, 1, 12.0)
    instance.add_distance('drone', 3, 0, 12.0)
    instance.add_distance('drone', 0, 3, 12.0)
    instance.add_distance('drone', 2, 3, 8.0)
    instance.add_distance('drone', 3, 2, 8.0)

    # Truck route: 0 -> 1 -> 2 -> 3 -> 0
    truck_route = TruckRoute(route_id=0, nodes=[0, 1, 2, 3, 0], capacity=50.0)
    # Two drone tasks launching and retrieving from the same truck node (1)
    dt1 = DroneTask(
        task_id=0,
        drone_id=0,
        launch_truck=0,
        launch_node=1,
        customers=[2],
        land_truck=0,
        retrieve_node=1,
        payloads=[5.0, 0.0],
    )
    dt2 = DroneTask(
        task_id=0,  # intentionally duplicate task_id to trigger collision
        drone_id=1,
        launch_truck=0,
        launch_node=1,
        customers=[3],
        land_truck=0,
        retrieve_node=1,
        payloads=[2.0, 0.0],
    )
    solution = Solution(truck_routes=[truck_route], drone_tasks=[dt1, dt2])

    evaluator = Evaluator(
        instance,
        truck_cost_per_km=2.0,
        drone_cost_per_km=3.0,
        rendezvous_tolerance=0.2,
    )

    details = evaluator.evaluate_with_details(solution)

    # We should have two separate robustness assessments (one per task)
    assert len(details.robustness.task_breakdown) == 2
    # Ensure both tasks are reported individually (energies not aggregated)
    energy_values = [
        ass.nominal_energy for ass in details.robustness.task_breakdown]
    assert len(energy_values) == 2
    assert energy_values[0] != energy_values[1]
    # Drone timings should include an entry for each enumerated task index (0 and 1)
    assert 0 in details.drone_timings
    assert 1 in details.drone_timings
