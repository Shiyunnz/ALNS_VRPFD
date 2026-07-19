"""Tests for shared search-phase evaluator used by TS and GA."""

from __future__ import annotations

from alns_vrpfd.evaluation import Evaluator, SearchEvaluator, SubrouteRobustVerifier
from alns_vrpfd.model import DroneTask, Solution, TruckRoute

from alns_vrpfd.tests.test_subroute_robust_verifier import (
    _build_instance_for_subroute_test,
)


def _truck_only_solution() -> Solution:
    return Solution(
        truck_routes=[TruckRoute(route_id=0, nodes=[0, 1, 2], capacity=50.0)],
        drone_tasks=[],
    )


def _drone_solution() -> Solution:
    return Solution(
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


def test_search_evaluator_matches_canonical_evaluator_for_feasible_solution():
    instance = _build_instance_for_subroute_test()
    evaluator = Evaluator(instance, rendezvous_tolerance=float("inf"))
    search = SearchEvaluator(evaluator)
    solution = _truck_only_solution()

    canonical = evaluator.evaluate_solution(solution)
    result = search.evaluate_solution(solution)

    assert result.feasible == canonical.feasible
    assert result.total_cost == canonical.total_cost
    assert result.truck_distance_cost == canonical.truck_distance_cost
    assert result.drone_distance_cost == canonical.drone_distance_cost
    assert result.delay_penalty == canonical.delay_penalty


def test_search_evaluator_reuses_static_instance_matrices():
    instance = _build_instance_for_subroute_test()
    evaluator = Evaluator(instance, rendezvous_tolerance=float("inf"))
    search = SearchEvaluator(evaluator)
    solution = _truck_only_solution()

    first = search.evaluate_solution(solution)
    second = search.evaluate_solution(solution.clone())

    assert first.feasible and second.feasible
    assert search.matrix_cache_misses > 0
    assert search.matrix_cache_hits > 0


def test_search_evaluator_reuses_robust_feasibility_cache():
    instance = _build_instance_for_subroute_test()
    evaluator = Evaluator(instance, rendezvous_tolerance=float("inf"))
    search = SearchEvaluator(evaluator, robust_cache_size=8)
    solution = _drone_solution()

    assert search.robust_feasible_cached(solution)
    assert search.robust_feasible_cached(solution.clone())

    assert search.robust_eval_calls == 1
    assert search.robust_cache_hits == 1


def test_search_evaluator_uses_subroute_verifier_for_candidate_gate():
    instance = _build_instance_for_subroute_test()
    evaluator = Evaluator(instance, rendezvous_tolerance=float("inf"))
    verifier = SubrouteRobustVerifier(
        instance=instance,
        drone_energy_capacity=1e-9,
        energy_uncertainty_budget=3,
        energy_deviation_rate=0.1,
    )
    search = SearchEvaluator(evaluator, candidate_subroute_verifier=verifier)

    assert not search.verify_candidate(
        base=_truck_only_solution(),
        candidate=_drone_solution(),
    )
    assert search.candidate_rejections == 1
