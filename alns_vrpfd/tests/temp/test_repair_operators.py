"""Tests for repair operators."""

from __future__ import annotations

import random

from alns_vrpfd.core.operators import (
    DestroyRandom,
    RepairBiasedRandomized,
    RepairCheapest,
    RepairRegret,
)
from alns_vrpfd.instance import InstanceManager
from alns_vrpfd.model import Solution
from alns_vrpfd.model.initializer import build_initial_solution


def _build_instance() -> InstanceManager:
    instance = InstanceManager()
    instance.configure_depots(start=0, end=0)
    for cid, demand in ((1, 10.0), (2, 12.0), (3, 14.0), (4, 16.0), (5, 9.0), (6, 11.0), (7, 8.0)):
        instance.register_customer(customer_id=cid, demand=demand)
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
    nodes = instance.all_node_ids()
    for origin in nodes:
        for dest in nodes:
            distance = 0.0 if origin == dest else abs(origin - dest) + 1.0
            instance.add_distance("truck", origin, dest, distance)
            instance.add_distance("drone", origin, dest, distance)
    instance.configure_robustness(drone_battery_capacity=80.0, energy_uncertainty_budget=2, energy_deviation_rate=0.1)
    return instance


def _remove_customers(solution: Solution, customers: list[int]) -> list[int]:
    removed: list[int] = []
    for customer in customers:
        for route in solution.truck_routes:
            if customer in route.customers():
                route.remove_customer(customer)
                removed.append(customer)
                break
        else:
            for task in list(solution.drone_tasks):
                if customer in task.customers():
                    task.remove_customer(customer)
                    if not task.customers():
                        solution.drone_tasks.remove(task)
                    removed.append(customer)
                    break
    return removed


def _collect_served(solution: Solution) -> set[int]:
    served: set[int] = set()
    for route in solution.truck_routes:
        served.update(route.customers())
    for task in solution.drone_tasks:
        served.update(task.customers())
    return served


def test_cheapest_repair_recovers_customers():
    instance = _build_instance()
    solution = build_initial_solution(instance)
    removed = _remove_customers(solution, [4, 6])
    operator = RepairCheapest(instance)
    repaired = operator.apply(solution, removed)
    served = _collect_served(repaired)
    for cid in removed:
        assert cid in served


def test_regret_repair_handles_anchor():
    instance = _build_instance()
    solution = build_initial_solution(instance)
    removed = _remove_customers(solution, [3, 5])
    operator = RepairRegret(instance, k=3)
    repaired = operator.apply(solution, removed)
    served = _collect_served(repaired)
    for cid in removed:
        assert cid in served


def test_biased_randomized_inserts_all():
    instance = _build_instance()
    solution = build_initial_solution(instance)
    removed = _remove_customers(solution, [2, 7])
    operator = RepairBiasedRandomized(instance, beta=1.5, rng=random.Random(4))
    repaired = operator.apply(solution, removed)
    served = _collect_served(repaired)
    for cid in removed:
        assert cid in served


def test_repair_after_destroy_roundtrip():
    instance = _build_instance()
    solution = build_initial_solution(instance)
    destroy = DestroyRandom(instance, rng=random.Random(5))
    mutated, pool = destroy.apply(solution, 2)
    operator = RepairCheapest(instance)
    repaired = operator.apply(mutated, pool.customers)
    served = _collect_served(repaired)
    for cid in pool.customers:
        assert cid in served
