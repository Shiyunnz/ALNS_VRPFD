"""Tests for changed-subroute robust pre-check verification."""

from __future__ import annotations

from alns_vrpfd.evaluation import Evaluator, SubrouteRobustVerifier
from alns_vrpfd.instance import InstanceManager
from alns_vrpfd.model import DroneTask, Solution, TruckRoute


def _build_instance_for_subroute_test() -> InstanceManager:
    instance = InstanceManager()
    instance.configure_depots(start=0, end=2)
    instance.register_customer(customer_id=1, demand=1.0)

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

    for mode in ("truck", "drone"):
        instance.add_distance(mode, 0, 2, 5.0)
        instance.add_distance(mode, 2, 0, 5.0)
        instance.add_distance(mode, 0, 1, 10.0)
        instance.add_distance(mode, 1, 0, 10.0)
        instance.add_distance(mode, 1, 2, 10.0)
        instance.add_distance(mode, 2, 1, 10.0)

    # Deterministic search instance for candidate details.
    instance.configure_robustness(
        drone_battery_capacity=100.0,
        energy_uncertainty_budget=0,
        energy_deviation_rate=0.1,
        same_truck_retrieval=False,
    )
    return instance


def test_subroute_verifier_rejects_changed_task_under_tight_capacity():
    instance = _build_instance_for_subroute_test()

    base = Solution(
        truck_routes=[TruckRoute(route_id=0, nodes=[0, 2], capacity=50.0)],
        drone_tasks=[],
    )
    candidate = Solution(
        truck_routes=[TruckRoute(route_id=0, nodes=[0, 2], capacity=50.0)],
        drone_tasks=[
            DroneTask(
                drone_id=0,
                launch_truck=0,
                launch_node=0,
                customers=[1],
                land_truck=0,
                retrieve_node=2,
                payloads=[5.0, 0.0],
                task_id=10,
            )
        ],
    )
    details = Evaluator(instance, rendezvous_tolerance=float("inf")).evaluate_with_details(
        candidate)
    assert details.result.feasible

    verifier = SubrouteRobustVerifier(
        instance=instance,
        drone_energy_capacity=1e-9,
        energy_uncertainty_budget=3,
        energy_deviation_rate=0.1,
    )
    ok = verifier.verify_candidate(
        base=base,
        candidate=candidate,
    )
    assert not ok
    assert verifier.last_summary.changed_drone_tasks == 1
    assert verifier.last_summary.failed_drone_tasks >= 1


def test_subroute_verifier_skips_when_no_changes():
    instance = _build_instance_for_subroute_test()
    solution = Solution(
        truck_routes=[TruckRoute(route_id=0, nodes=[0, 2], capacity=50.0)],
        drone_tasks=[
            DroneTask(
                drone_id=0,
                launch_truck=0,
                launch_node=0,
                customers=[1],
                land_truck=0,
                retrieve_node=2,
                payloads=[1.0, 0.0],
                task_id=10,
            )
        ],
    )
    verifier = SubrouteRobustVerifier(
        instance=instance,
        drone_energy_capacity=1000.0,
        energy_uncertainty_budget=3,
        energy_deviation_rate=0.1,
    )
    ok = verifier.verify_candidate(
        base=solution,
        candidate=solution.clone(),
    )
    assert ok
    assert verifier.last_summary.checked_drone_tasks == 0
