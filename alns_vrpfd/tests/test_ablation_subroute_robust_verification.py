"""Tests for the sub-route robust verification ablation helpers."""

from __future__ import annotations

from alns_vrpfd.model import DroneTask, Solution, TruckRoute
from alns_vrpfd.tests.test_subroute_robust_verifier import (
    _build_instance_for_subroute_test,
)
from scripts.ablation_subroute_robust_verification import (
    InstrumentedFullCandidateVerifier,
    InstrumentedSubrouteVerifier,
)


def test_instrumented_subroute_verifier_counts_rejected_candidate():
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
                payloads=[1.0, 0.0],
                task_id=1,
            )
        ],
    )

    verifier = InstrumentedSubrouteVerifier(
        instance=instance,
        drone_energy_capacity=1e-9,
        energy_uncertainty_budget=3,
        energy_deviation_rate=0.1,
    )

    assert verifier.verify_candidate(base=base, candidate=candidate) is False
    assert verifier.calls == 1
    assert verifier.rejections == 1
    assert verifier.checked_drone_tasks == 1
    assert verifier.failed_drone_tasks == 1
    assert verifier.elapsed_sec >= 0.0


def test_instrumented_full_candidate_verifier_counts_full_rejection():
    from alns_vrpfd.evaluation import Evaluator

    instance = _build_instance_for_subroute_test()
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
                payloads=[1.0, 0.0],
                task_id=1,
            )
        ],
    )

    instance.configure_robustness(
        drone_battery_capacity=1e-9,
        energy_uncertainty_budget=3,
        energy_deviation_rate=0.1,
        same_truck_retrieval=False,
    )
    verifier = InstrumentedFullCandidateVerifier(
        Evaluator(instance, rendezvous_tolerance=float("inf"))
    )

    assert verifier.verify_candidate(base=candidate, candidate=candidate) is False
    assert verifier.calls == 1
    assert verifier.rejections == 1
    assert verifier.elapsed_sec >= 0.0
