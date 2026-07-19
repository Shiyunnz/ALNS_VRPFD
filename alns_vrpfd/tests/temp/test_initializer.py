"""Tests for the initial solution construction heuristics."""

from __future__ import annotations

from alns_vrpfd.instance import InstanceManager
from alns_vrpfd.model.initializer import build_initial_solution
from alns_vrpfd.model.initializer import build_two_phase_initial_solution
from alns_vrpfd.evaluation.evaluator import Evaluator
from alns_vrpfd.model.route import DroneTask
import pytest


def _basic_instance() -> InstanceManager:
    instance = InstanceManager()
    instance.configure_depots(start=0, end=0)

    instance.register_customer(customer_id=1, demand=3.0)
    instance.register_customer(customer_id=2, demand=2.0)
    instance.register_customer(customer_id=3, demand=1.5)

    instance.register_vehicle_type(
        "truck",
        number=1,
        capacity=50.0,
        endurance=480.0,
        speed=40.0,
        unit_cost=1.5,
    )
    instance.register_vehicle_type(
        "drone",
        number=1,
        capacity=5.0,
        endurance=120.0,
        speed=60.0,
        unit_cost=2.5,
    )

    edges = {
        (0, 1): 10.0,
        (1, 2): 8.0,
        (2, 3): 6.0,
        (3, 0): 9.0,
        (1, 0): 10.0,
        (2, 1): 8.0,
        (3, 2): 6.0,
        (0, 2): 11.0,
        (0, 3): 12.0,
        (2, 0): 11.0,
        (3, 1): 7.0,
    }

    for (origin, destination), distance in edges.items():
        instance.add_distance("truck", origin, destination, distance)
        instance.add_distance("drone", origin, destination, distance)

    return instance


def test_initial_solution_covers_all_customers_once():
    instance = _basic_instance()
    solution = build_initial_solution(instance)

    assert solution.truck_routes
    assert solution.drone_tasks

    truck_customers = set()
    for route in solution.truck_routes:
        truck_customers.update(route.customers())

    drone_customers = set()
    for task in solution.drone_tasks:
        drone_customers.update(task.customers())

    all_customers = set(instance.customer_manager.customer_ids())
    assert truck_customers.isdisjoint(drone_customers)
    assert truck_customers.union(drone_customers) == all_customers


def test_two_phase_detects_unservable_customers_and_raises():
    instance = InstanceManager()
    instance.configure_depots(start=0, end=0)

    # Create several customers; the last customer is too large for the drone
    instance.register_customer(customer_id=1, demand=10.0)
    instance.register_customer(customer_id=2, demand=10.0)
    instance.register_customer(customer_id=3, demand=10.0)
    instance.register_customer(customer_id=4, demand=59.0)

    # One truck with capacity 50 so some customers will remain unserved by trucks
    instance.register_vehicle_type(
        "truck",
        number=1,
        capacity=50.0,
        endurance=480.0,
        speed=40.0,
        unit_cost=1.0,
    )
    # Drone capacity is 30 which is insufficient for customer 4
    instance.register_vehicle_type(
        "drone",
        number=1,
        capacity=30.0,
        endurance=120.0,
        speed=60.0,
        unit_cost=2.0,
    )

    # Add fully connected distances (unit distances) to keep NN behavior simple
    nodes = [0, 1, 2, 3, 4]
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            instance.add_distance("truck", i, j, 1.0)
            instance.add_distance("drone", i, j, 1.0)

    with pytest.raises(ValueError) as excinfo:
        build_two_phase_initial_solution(instance)

    assert "exceeds the drone capacity" in str(excinfo.value)


def test_build_initial_solution_detects_unservable_forbidden_customers():
    instance = InstanceManager()
    instance.configure_depots(start=0, end=0)

    instance.register_customer(customer_id=1, demand=10.0)
    instance.register_customer(customer_id=2, demand=10.0)
    instance.register_customer(customer_id=3, demand=10.0)
    instance.register_customer(customer_id=4, demand=59.0)

    instance.register_vehicle_type(
        "truck",
        number=2,
        capacity=50.0,
        endurance=480.0,
        speed=40.0,
        unit_cost=1.5,
    )
    instance.register_vehicle_type(
        "drone",
        number=1,
        capacity=30.0,
        endurance=120.0,
        speed=60.0,
        unit_cost=2.5,
    )

    nodes = [0, 1, 2, 3, 4]
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            instance.add_distance("truck", i, j, 1.0)
            instance.add_distance("drone", i, j, 1.0)

    # Mark customer 4 as forbidden to truck; the truck routes will exclude it
    forced_drone_customers = [4]
    with pytest.raises(ValueError) as excinfo:
        build_initial_solution(
            instance, truck_forbidden_customers=forced_drone_customers)

    assert "exceed" in str(excinfo.value)


def test_two_phase_prioritizes_heavy_customers_for_truck_assignment():
    instance = InstanceManager()
    instance.configure_depots(start=0, end=0)

    # Create customers such that a heavy customer would exceed drone cap
    instance.register_customer(customer_id=1, demand=30.0)
    instance.register_customer(customer_id=2, demand=30.0)
    instance.register_customer(customer_id=3, demand=10.0)
    instance.register_customer(customer_id=4, demand=59.0)

    # One truck with capacity 60 so it can carry customer 4 alone
    instance.register_vehicle_type(
        "truck",
        number=1,
        capacity=60.0,
        endurance=480.0,
        speed=40.0,
        unit_cost=1.0,
    )
    # Drone capacity 30: customers 1 and 2 are drone-serviceable, 4 is not
    instance.register_vehicle_type(
        "drone",
        number=2,
        capacity=30.0,
        endurance=120.0,
        speed=60.0,
        unit_cost=2.0,
    )

    nodes = [0, 1, 2, 3, 4]
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            instance.add_distance("truck", i, j, 1.0)
            instance.add_distance("drone", i, j, 1.0)

    # Build the initial solution; should not raise and should assign 4 to truck
    solution = build_two_phase_initial_solution(instance)

    truck_customers = set()
    for route in solution.truck_routes:
        truck_customers.update(route.customers())

    drone_customers = set()
    for task in solution.drone_tasks:
        drone_customers.update(task.customers())

    assert 4 in truck_customers
    assert 4 not in drone_customers


def test_evaluator_allows_anchor_conflict_when_relaxed():
    # Build a simple instance with two drones and single truck node
    instance = _basic_instance()
    instance.register_vehicle_type(
        "truck",
        number=1,
        capacity=50.0,
        endurance=480.0,
        speed=40.0,
        unit_cost=1.5,
    )
    instance.register_vehicle_type(
        "drone",
        number=2,
        capacity=10.0,
        endurance=120.0,
        speed=60.0,
        unit_cost=2.5,
    )

    # Create a solution with two drone tasks launching from same truck node 1
    from alns_vrpfd.model.solution import Solution
    from alns_vrpfd.model.route import TruckRoute
    sol = Solution()
    truck = TruckRoute(route_id=0, nodes=[0, 1, 11], capacity=50.0)
    sol.add_truck_route(truck)

    # Two drone tasks both launching from truck 0 at node 1
    task1 = DroneTask(drone_id=0, launch_truck=0, launch_node=1, customers=[
                      2], land_truck=0, retrieve_node=11, payloads=[1.0, 0.0])
    task2 = DroneTask(drone_id=1, launch_truck=0, launch_node=1, customers=[
                      3], land_truck=0, retrieve_node=11, payloads=[1.0, 0.0])
    sol.add_drone_task(task1)
    sol.add_drone_task(task2)

    # By default anchor conflicts are allowed (legacy behaviour)
    evaluator = Evaluator(instance)
    assert not evaluator._has_drone_anchor_conflicts(sol.drone_tasks)

    # Evaluator should detect anchor conflict when set to disallow multiple launches
    evaluator_strict = Evaluator(
        instance, allow_multiple_launch_per_node=False)
    assert evaluator_strict._has_drone_anchor_conflicts(sol.drone_tasks)
