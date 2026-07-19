"""Regression tests for ALNS feasibility recovery."""

from __future__ import annotations

from types import SimpleNamespace

from alns_vrpfd.core.operators.base import UnassignedPool
from alns_vrpfd.core.sa import SANNCfg, SimulatedAnnealingALNS
from alns_vrpfd.model.route import TruckRoute
from alns_vrpfd.model.solution import Solution


class _CustomerManager:
    def customer_ids(self):
        return list(range(1, 11))


class _Instance:
    customer_manager = _CustomerManager()


class _Evaluator:
    def evaluate_with_details(self, solution):
        return SimpleNamespace(
            result=self.evaluate_solution(solution),
            robustness=SimpleNamespace(feasible=True, task_breakdown=[]),
        )

    def evaluate_solution(self, solution):
        finite = any(2 in route.nodes for route in solution.truck_routes)
        return SimpleNamespace(feasible=finite, total_cost=10.0 if finite else float("inf"))


class _NoopDestroy:
    name = "noop_destroy"

    def apply(self, solution, count):
        return solution.clone(), UnassignedPool()


class _SecondAttemptRepair:
    name = "second_attempt_repair"

    def __init__(self):
        self.calls = 0

    def apply(self, solution, customers):
        self.calls += 1
        repaired = solution.clone()
        if self.calls >= 2:
            repaired.truck_routes[0].nodes.insert(-1, 2)
        return repaired


class _ProgressiveEvaluator:
    def evaluate_with_details(self, solution):
        return SimpleNamespace(
            result=self.evaluate_solution(solution),
            robustness=SimpleNamespace(feasible=True, task_breakdown=[]),
            delay_breakdown=SimpleNamespace(violations=[]),
        )

    def evaluate_solution(self, solution):
        nodes = solution.truck_routes[0].nodes
        if 3 in nodes:
            return SimpleNamespace(
                feasible=True,
                total_cost=10.0,
                truck_distance_cost=10.0,
                drone_distance_cost=0.0,
                delay_penalty=0.0,
                energy_penalty=0.0,
            )
        if 2 in nodes:
            return SimpleNamespace(
                feasible=False,
                total_cost=float("inf"),
                truck_distance_cost=20.0,
                drone_distance_cost=0.0,
                delay_penalty=50.0,
                energy_penalty=0.0,
            )
        return SimpleNamespace(
            feasible=False,
            total_cost=float("inf"),
            truck_distance_cost=30.0,
            drone_distance_cost=0.0,
            delay_penalty=100.0,
            energy_penalty=0.0,
        )


class _ProgressiveRepair:
    name = "progressive_repair"

    def __init__(self):
        self.calls = 0

    def apply(self, solution, customers):
        self.calls += 1
        repaired = solution.clone()
        nodes = repaired.truck_routes[0].nodes
        if 2 in nodes and 3 not in nodes:
            nodes.insert(-1, 3)
        elif 2 not in nodes:
            nodes.insert(-1, 2)
        return repaired


class _TimeWindowPenaltyEvaluator:
    def evaluate_with_details(self, solution):
        nodes = solution.truck_routes[0].nodes
        if 2 in nodes:
            violations = (
                SimpleNamespace(arrival_time=5.0, latest_time=1.0),
                SimpleNamespace(arrival_time=6.0, latest_time=1.0),
            )
        elif 3 in nodes:
            violations = (SimpleNamespace(arrival_time=2.0, latest_time=1.0),)
        else:
            violations = ()
        return SimpleNamespace(
            result=self.evaluate_solution(solution),
            robustness=SimpleNamespace(feasible=True, task_breakdown=[]),
            delay_breakdown=SimpleNamespace(violations=violations),
        )

    def evaluate_solution(self, solution):
        nodes = solution.truck_routes[0].nodes
        if 2 in nodes:
            truck_cost = 1.0
        elif 3 in nodes:
            truck_cost = 1000.0
        else:
            truck_cost = 10.0
        return SimpleNamespace(
            feasible=False,
            total_cost=float("inf"),
            truck_distance_cost=truck_cost,
            drone_distance_cost=0.0,
            delay_penalty=0.0,
            energy_penalty=0.0,
        )


def test_run_extends_search_until_first_feasible_solution_when_initial_is_infeasible():
    initial = Solution(truck_routes=[TruckRoute(route_id=0, nodes=[0, 1, 0], capacity=10.0)])
    repair = _SecondAttemptRepair()
    cfg = SANNCfg(
        iterations=1,
        eta=0.5,
        escape_enabled=False,
        local_search_frequency=0,
        local_search_on_new_best=False,
        intensify_frequency=0,
        path_relinking_prob=0.0,
        dynamic_cooling_enabled=False,
        matheuristic_lns_enabled=False,
        max_non_improve=1,
        measure_operator_time=False,
    )
    alns = SimulatedAnnealingALNS(
        instance=_Instance(),
        destroy_ops=[_NoopDestroy()],
        repair_ops=[repair],
        evaluator=_Evaluator(),
        cfg=cfg,
        verbose=False,
    )

    best = alns.run(initial)

    assert _Evaluator().evaluate_solution(best).feasible
    assert repair.calls == 2
    assert alns.last_run_stats["configured_iterations"] == 1
    assert alns.last_run_stats["executed_iterations"] == 2


def test_run_accepts_infeasible_progress_toward_feasibility():
    initial = Solution(truck_routes=[TruckRoute(route_id=0, nodes=[0, 1, 0], capacity=10.0)])
    repair = _ProgressiveRepair()
    cfg = SANNCfg(
        iterations=1,
        eta=0.5,
        escape_enabled=False,
        local_search_frequency=0,
        local_search_on_new_best=False,
        intensify_frequency=0,
        path_relinking_prob=0.0,
        dynamic_cooling_enabled=False,
        matheuristic_lns_enabled=False,
        max_non_improve=1,
        measure_operator_time=False,
    )
    evaluator = _ProgressiveEvaluator()
    alns = SimulatedAnnealingALNS(
        instance=_Instance(),
        destroy_ops=[_NoopDestroy()],
        repair_ops=[repair],
        evaluator=evaluator,
        cfg=cfg,
        verbose=False,
    )

    best = alns.run(initial)

    assert evaluator.evaluate_solution(best).feasible
    assert repair.calls == 2


def test_search_cost_prioritizes_hard_time_window_feasibility_over_distance():
    evaluator = _TimeWindowPenaltyEvaluator()
    alns = SimulatedAnnealingALNS(
        instance=_Instance(),
        destroy_ops=[_NoopDestroy()],
        repair_ops=[_ProgressiveRepair()],
        evaluator=evaluator,
        cfg=SANNCfg(iterations=1, measure_operator_time=False),
        verbose=False,
    )
    many_violations = Solution(
        truck_routes=[TruckRoute(route_id=0, nodes=[0, 1, 2, 0], capacity=10.0)]
    )
    fewer_violations = Solution(
        truck_routes=[TruckRoute(route_id=0, nodes=[0, 1, 3, 0], capacity=10.0)]
    )

    assert alns._search_cost(
        fewer_violations,
        evaluator.evaluate_solution(fewer_violations),
    ) < alns._search_cost(
        many_violations,
        evaluator.evaluate_solution(many_violations),
    )
