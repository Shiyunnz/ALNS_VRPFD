"""Tests for destroy operators."""

from __future__ import annotations

import random

from alns_vrpfd.core.operators import (
    DestroyRandom,
    DestroyWorstDistance,
    DestroyShaw,
    UnassignedPool,
)
from alns_vrpfd.instance import InstanceManager
from alns_vrpfd.model import DroneTask, Solution, TruckRoute


def _build_instance() -> InstanceManager:
    instance = InstanceManager()
    instance.configure_depots(start=0, end=0)

    for customer_id, demand in ((1, 10.0), (2, 8.0), (3, 6.0), (4, 5.0), (5, 7.0), (6, 12.0), (7, 14.0)):
        instance.register_customer(customer_id=customer_id, demand=demand)

    instance.register_vehicle_type(
        "truck",
        number=1,
        capacity=200.0,
        endurance=480.0,
        speed=40.0,
        unit_cost=1.0,
    )
    instance.register_vehicle_type(
        "drone",
        number=2,
        capacity=40.0,
        endurance=120.0,
        speed=60.0,
        unit_cost=1.0,
    )

    # Distances: complete graph with simple values
    nodes = instance.all_node_ids()
    for origin in nodes:
        for destination in nodes:
            if origin == destination:
                distance = 0.0
            else:
                distance = abs(origin - destination) + 1.0
            instance.add_distance("truck", origin, destination, distance)
            instance.add_distance("drone", origin, destination, distance)

    instance.configure_robustness(drone_battery_capacity=80.0)
    return instance


def _build_solution(instance: InstanceManager) -> Solution:
    truck_route = TruckRoute(route_id=0, nodes=[0, 1, 2, 3, 0], capacity=200.0)
    truck_route.current_load = sum(instance.customer_manager.demands().get(c, 0.0) for c in truck_route.customers())
    task = DroneTask(
        task_id=1,
        drone_id=0,
        launch_truck=0,
        launch_node=1,
        customers=[4],
        land_truck=0,
        retrieve_node=2,
        payloads=[instance.customer_manager.demands()[4], 0.0],
    )
    solution = Solution(truck_routes=[truck_route], drone_tasks=[task])
    return solution


class _TargetDestroy(DestroyRandom):
    def __init__(self, instance: InstanceManager, target: int, **kwargs) -> None:
        super().__init__(instance, **kwargs)
        self._target = target

    def _select_customers(self, assignments, count):  # type: ignore[override]
        return [self._target]


def test_random_destroy_removes_requested_customers():
    instance = _build_instance()
    solution = _build_solution(instance)
    operator = DestroyRandom(instance, rng=random.Random(5))

    mutated, pool = operator.apply(solution, 2)

    removed = set(pool.customers)
    remaining_truck = set()
    for route in mutated.truck_routes:
        remaining_truck.update(route.customers())
    remaining_drone = set()
    for task in mutated.drone_tasks:
        remaining_drone.update(task.customers())

    assert len(pool.customers) == 2
    assert removed.isdisjoint(remaining_truck.union(remaining_drone))


def test_drop_anchor_removes_linked_tasks():
    instance = _build_instance()
    solution = _build_solution(instance)
    operator = _TargetDestroy(instance, target=1, anchor_strategy="drop_tasks")

    mutated, pool = operator.apply(solution, 1)

    assert mutated.drone_tasks == []
    assert sorted(pool.customers) == [1, 4]
    assert 1 not in mutated.truck_routes[0].customers()


def test_rebase_anchor_updates_tasks():
    instance = _build_instance()
    solution = _build_solution(instance)
    operator = _TargetDestroy(instance, target=1, anchor_strategy="rebase_to_neighbor")

    mutated, pool = operator.apply(solution, 1)

    assert pool.customers == [1]
    assert mutated.drone_tasks[0].launch_node == 0  # re-based to depot
    assert mutated.drone_tasks[0].retrieve_node == 2
    assert mutated.drone_tasks[0].customers() == [4]


def test_shaw_destroy_respects_count():
    instance = _build_instance()
    solution = _build_solution(instance)
    operator = DestroyShaw(instance, rng=random.Random(3), anchor_strategy="rebase_to_neighbor")

    mutated, pool = operator.apply(solution, 3)

    total_remaining = sum(len(route.customers()) for route in mutated.truck_routes)
    total_remaining += sum(len(task.customers()) for task in mutated.drone_tasks)
    original_total = 4  # customers 1,2,3 in truck and customer 4 in drone

    assert len(pool.customers) == 3
    assert total_remaining == original_total - 3


def test_worst_distance_prefers_long_segment():
    instance = _build_instance()
    solution = _build_solution(instance)
    operator = DestroyWorstDistance(instance)

    mutated, pool = operator.apply(solution, 1)

    assert pool.customers == [3]
    assert 3 not in mutated.truck_routes[0].customers()
