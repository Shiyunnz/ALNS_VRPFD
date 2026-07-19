"""Regression tests for TS time-limit handling."""

from __future__ import annotations

import random
from types import SimpleNamespace
from pathlib import Path

from alns_vrpfd.evaluation import Evaluator
from alns_vrpfd.instance import InstanceManager
from alns_vrpfd.model import DroneTask, Solution, TruckRoute
from heuristics.ga.ga import GeneticAlgorithm, Individual
from heuristics.tabu_search import tabu_search


class DummySolution:
    def __init__(self, name: str):
        self.name = name
        self.drone_tasks = []

    def clone(self):
        return self


def _build_three_customer_instance() -> InstanceManager:
    instance = InstanceManager()
    instance.configure_depots(start=0, end=99)
    coords = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (2.0, 0.0),
        3: (3.0, 0.0),
        99: (4.0, 0.0),
    }
    for customer in (1, 2, 3):
        instance.register_customer(customer, demand=1.0, location_x=coords[customer][0])
        instance.customer_manager.assign_time_window(customer, 0.0, 100.0)
    instance.register_vehicle_type(
        "truck", number=1, capacity=50.0, endurance=100.0, speed=1.0, unit_cost=1.0
    )
    instance.register_vehicle_type(
        "drone", number=1, capacity=10.0, endurance=100.0, speed=1.0, unit_cost=1.0
    )
    for mode in ("truck", "drone"):
        for origin, (ox, oy) in coords.items():
            for destination, (dx, dy) in coords.items():
                distance = ((ox - dx) ** 2 + (oy - dy) ** 2) ** 0.5
                instance.add_distance(mode, origin, destination, distance)
    instance.configure_robustness(
        drone_battery_capacity=100.0,
        energy_uncertainty_budget=0,
        energy_deviation_rate=0.1,
        same_truck_retrieval=False,
    )
    return instance


def _make_ts(evaluator: Evaluator) -> tabu_search.TabuSearch:
    ts = tabu_search.TabuSearch.__new__(tabu_search.TabuSearch)
    ts._evaluator = evaluator
    ts._search_evaluator = None
    ts._vehicle_specs = evaluator._instance.vehicle_specs
    ts._demands = evaluator._instance.customer_manager.demands()
    ts._depot_start = evaluator._instance.customer_manager.depot_start
    ts._depot_end = evaluator._instance.customer_manager.depot_end
    ts._depots = {ts._depot_start, ts._depot_end}
    ts._truck_dist = evaluator._instance.distance_matrix("truck")
    ts._drone_dist = evaluator._instance.distance_matrix("drone")
    ts._node_index = {n: i for i, n in enumerate(evaluator._instance.all_node_ids())}
    ts._alpha_energy = 100.0
    ts._alpha_tw = 50.0
    ts._alpha_coverage = 200.0
    ts._alpha_capacity = 100.0
    ts._alpha_hard = 100_000.0
    ts._moves_per_type = 60
    return ts


def _illegal_backward_drone_solution() -> Solution:
    return Solution(
        truck_routes=[TruckRoute(route_id=0, nodes=[0, 1, 2, 99], capacity=50.0)],
        drone_tasks=[
            DroneTask(
                task_id=1,
                drone_id=0,
                launch_truck=0,
                launch_node=2,
                customers=[3],
                land_truck=0,
                retrieve_node=1,
                payloads=[1.0, 0.0],
            )
        ],
    )


def test_tabu_search_checks_time_limit_before_neighbor_evaluation(monkeypatch):
    ts = tabu_search.TabuSearch.__new__(tabu_search.TabuSearch)
    ts._evaluator = SimpleNamespace(_instance=object())
    ts._max_iterations = 2
    ts._rng = random.Random(1)
    ts._base_tabu_tenure = 3
    ts._max_stagnation = 10
    ts._penalty_adapt_interval = 50
    ts._alpha_energy = 100.0
    ts._alpha_tw = 50.0
    ts._moves_per_type = 10
    ts._max_neighbors = 10
    ts._n_customers = 50

    initial = DummySolution("initial")
    neighbors = [(DummySolution(f"neighbor-{i}"), ("move", i)) for i in range(5)]

    monkeypatch.setattr(ts, "_repair_energy_violations", lambda solution: solution)
    monkeypatch.setattr(ts, "_ensure_all_customers_served", lambda solution: solution)
    monkeypatch.setattr(ts, "_gen_violation_directed_moves", lambda solution: iter(neighbors))
    monkeypatch.setattr(ts, "_gen_drone_moves", lambda solution: iter(()))
    monkeypatch.setattr(ts, "_gen_cross_truck_drone_moves", lambda solution: iter(()))
    monkeypatch.setattr(ts, "_gen_truck_relocate", lambda solution: iter(()))
    monkeypatch.setattr(ts, "_gen_ruin_recreate", lambda solution: iter(()))
    monkeypatch.setattr(ts, "_gen_or_opt", lambda solution: iter(()))
    monkeypatch.setattr(ts, "_perturb", lambda solution: solution)
    monkeypatch.setattr(ts, "_apply_2opt", lambda solution: solution)
    monkeypatch.setattr(ts, "_apply_drone_optimization", lambda solution: solution)

    evaluated = []

    def penalized_cost(solution):
        evaluated.append(solution.name)
        return 1.0, True, {}

    monkeypatch.setattr(ts, "_penalized_cost", penalized_cost)

    times = iter([0.0, 0.0, 1.0, 1.0])
    monkeypatch.setattr(tabu_search.time, "perf_counter", lambda: next(times))

    result = ts.run(initial, time_limit=0.5)

    assert result is initial
    assert evaluated == ["initial"]


def test_tabu_search_infeasible_penalty_uses_full_delay_cost():
    details = SimpleNamespace(
        result=SimpleNamespace(
            feasible=False,
            total_cost=float("inf"),
            delay_penalty=12.0,
        ),
        robustness=SimpleNamespace(task_breakdown=[]),
        delay_breakdown=SimpleNamespace(violations=[]),
    )
    truck_spec = SimpleNamespace(capacity=100.0)
    ts = tabu_search.TabuSearch.__new__(tabu_search.TabuSearch)
    ts._evaluator = SimpleNamespace(evaluate_with_details=lambda solution: details)
    ts._demands = {}
    ts._vehicle_specs = {"truck": truck_spec}
    ts._alpha_energy = 100.0
    ts._alpha_tw = 50.0
    ts._alpha_coverage = 200.0
    ts._alpha_capacity = 100.0
    ts._alpha_hard = 100_000.0

    cost, feasible, violations = ts._penalized_cost(
        SimpleNamespace(truck_routes=[], drone_tasks=[])
    )

    assert feasible is False
    assert cost == 1_000_012.0
    assert violations == {}


def test_tabu_search_non_delay_gate_rejects_drone_location_illegal_solution():
    instance = _build_three_customer_instance()
    evaluator = Evaluator(instance, rendezvous_tolerance=float("inf"))
    solution = _illegal_backward_drone_solution()
    details = evaluator.evaluate_with_details(solution)
    ts = _make_ts(evaluator)

    assert evaluator._has_drone_task_violations(solution)
    assert ts._non_delay_feasible(solution, details) is False


def test_tabu_search_non_delay_gate_allows_time_window_only_violation():
    instance = _build_three_customer_instance()
    instance.customer_manager.assign_time_window(3, 0.0, 1.0)
    evaluator = Evaluator(instance, rendezvous_tolerance=float("inf"), time_tolerance=0.0)
    solution = Solution(
        truck_routes=[TruckRoute(route_id=0, nodes=[0, 1, 2, 3, 99], capacity=50.0)],
        drone_tasks=[],
    )
    details = evaluator.evaluate_with_details(solution)
    ts = _make_ts(evaluator)

    assert details.delay_breakdown.violations
    assert details.result.feasible is False
    assert ts._non_delay_feasible(solution, details) is True


def test_tabu_search_penalizes_drone_task_violation_above_truck_repair():
    instance = _build_three_customer_instance()
    evaluator = Evaluator(instance, rendezvous_tolerance=float("inf"))
    ts = _make_ts(evaluator)
    illegal = _illegal_backward_drone_solution()
    repaired = Solution(
        truck_routes=[TruckRoute(route_id=0, nodes=[0, 1, 2, 3, 99], capacity=50.0)],
        drone_tasks=[],
    )

    illegal_cost, illegal_feasible, illegal_violations = ts._penalized_cost(illegal)
    repaired_cost, _, _ = ts._penalized_cost(repaired)

    assert illegal_feasible is False
    assert illegal_violations["drone_task"] == 1.0
    assert illegal_cost > repaired_cost


def test_tabu_search_truck_backbone_rechain_generates_hard_feasible_delay_fix():
    instance = _build_three_customer_instance()
    instance.customer_manager.assign_time_window(1, 0.0, 1.5)
    evaluator = Evaluator(instance, rendezvous_tolerance=float("inf"), time_tolerance=0.0)
    ts = _make_ts(evaluator)
    solution = Solution(
        truck_routes=[TruckRoute(route_id=0, nodes=[0, 3, 2, 1, 99], capacity=50.0)],
        drone_tasks=[],
    )
    details = evaluator.evaluate_with_details(solution)

    moves = list(ts._gen_truck_backbone_rechain_moves(solution, details))

    assert any(sig[0].startswith("fix_tw_truck_rechain") for _, sig in moves)
    assert all(ts._non_delay_feasible(candidate) for candidate, _ in moves)
    assert any(
        len(evaluator.evaluate_with_details(candidate).delay_breakdown.violations)
        < len(details.delay_breakdown.violations)
        for candidate, _ in moves
    )


def test_tabu_search_anchor_window_repair_reinserts_removed_drone_customers():
    instance = _build_three_customer_instance()
    evaluator = Evaluator(instance, rendezvous_tolerance=float("inf"))
    ts = _make_ts(evaluator)
    solution = Solution(
        truck_routes=[TruckRoute(route_id=0, nodes=[0, 3, 2, 99], capacity=50.0)],
        drone_tasks=[
            DroneTask(
                task_id=1,
                drone_id=0,
                launch_truck=0,
                launch_node=3,
                customers=[1],
                land_truck=0,
                retrieve_node=2,
                payloads=[1.0, 0.0],
            )
        ],
    )

    removed = ts._repair_touched_drone_anchors(solution, {0: {2, 3}})

    assert removed == [1]
    assert solution.drone_tasks == []
    assert any(1 in route.nodes for route in solution.truck_routes)
    assert not evaluator._has_customer_coverage_violation(solution)
    assert not evaluator._has_drone_task_violations(solution)


def test_tabu_search_final_tw_polish_reduces_lateness():
    instance = _build_three_customer_instance()
    instance.customer_manager.assign_time_window(1, 0.0, 1.5)
    evaluator = Evaluator(instance, rendezvous_tolerance=float("inf"), time_tolerance=0.0)
    ts = _make_ts(evaluator)
    ts._n_customers = 3
    ts._reset_stats()
    solution = Solution(
        truck_routes=[TruckRoute(route_id=0, nodes=[0, 3, 2, 1, 99], capacity=50.0)],
        drone_tasks=[],
    )
    before_details = evaluator.evaluate_with_details(solution)
    before_score = ts._delay_repair_score(solution, before_details)

    polished = ts._apply_final_tw_polish(
        solution,
        deadline=tabu_search.time.perf_counter() + 5.0,
    )
    after_details = evaluator.evaluate_with_details(polished)
    after_score = ts._delay_repair_score(polished, after_details)

    assert after_score < before_score
    assert ts._non_delay_feasible(polished, after_details)
    assert ts.stats["tw_polish_iterations"] > 0


def test_tabu_search_final_feasibility_profile_records_late_node_context():
    instance = _build_three_customer_instance()
    instance.customer_manager.assign_time_window(1, 0.0, 1.5)
    evaluator = Evaluator(instance, rendezvous_tolerance=float("inf"), time_tolerance=0.0)
    ts = _make_ts(evaluator)
    solution = Solution(
        truck_routes=[TruckRoute(route_id=0, nodes=[0, 3, 2, 1, 99], capacity=50.0)],
        drone_tasks=[],
    )

    profile = ts._final_feasibility_profile(solution)

    assert profile["hard_ok"] is True
    assert profile["tw_count"] >= 1
    assert profile["late_nodes"][0]["node"] == 1
    assert profile["late_nodes"][0]["window"] == [0, 3, 2, 1, 99]


def test_tabu_search_increases_constraint_penalties_when_no_feasible_neighbors():
    ts = tabu_search.TabuSearch.__new__(tabu_search.TabuSearch)
    ts._alpha_energy = 100.0
    ts._alpha_tw = 50.0
    ts._alpha_coverage = 200.0
    ts._alpha_capacity = 100.0

    ts._adapt_penalties(feasible_count=0, total_checked=40, has_feasible_solution=False)

    assert ts._alpha_energy > 100.0
    assert ts._alpha_tw > 50.0
    assert ts._alpha_coverage > 200.0
    assert ts._alpha_capacity > 100.0


def test_tabu_search_dynamic_drone_task_limit_allows_more_than_four_for_r50():
    ts = tabu_search.TabuSearch.__new__(tabu_search.TabuSearch)
    ts._n_customers = 50
    ts._drone_eligible = set(range(47))
    ts._vehicle_specs = {"drone": SimpleNamespace(number=1)}

    assert ts._max_drone_tasks_allowed() >= 8


def test_tabu_search_generates_drone_insert_for_time_window_violation():
    solution = tabu_search.Solution(
        truck_routes=[
            tabu_search.TruckRoute(route_id=0, nodes=[0, 1, 2, 99], capacity=100.0)
        ],
        drone_tasks=[],
    )
    violation = SimpleNamespace(node_id=2, arrival_time=10.0, latest_time=5.0)
    details = SimpleNamespace(
        robustness=SimpleNamespace(feasible=True, task_breakdown=[]),
        delay_breakdown=SimpleNamespace(violations=[violation]),
    )
    ts = tabu_search.TabuSearch.__new__(tabu_search.TabuSearch)
    ts._evaluator = SimpleNamespace(evaluate_with_details=lambda candidate: details)
    ts._drone_eligible = {2}
    ts._demands = {1: 1.0, 2: 1.0}
    ts._vehicle_specs = {"drone": SimpleNamespace(capacity=10.0, number=1)}
    ts._depot_start = 0
    ts._depot_end = 99
    ts._depots = {0, 99}
    ts._drone_optimizer = SimpleNamespace(_robust_energy_feasible=lambda *args: True)

    moves = list(ts._gen_violation_directed_moves(solution))

    assert any(move_sig[0] == "fix_tw_drone_insert" for _, move_sig in moves)
    neighbor, move_sig = next(
        (candidate, sig) for candidate, sig in moves if sig[0] == "fix_tw_drone_insert"
    )
    assert move_sig[1] == 2
    assert 2 not in neighbor.truck_routes[0].nodes
    assert neighbor.drone_tasks[0].customers() == [2]


def test_tabu_search_penalized_cost_delegates_to_search_evaluator():
    ts = tabu_search.TabuSearch.__new__(tabu_search.TabuSearch)
    solution = DummySolution("candidate")
    calls = []

    class DummySearchEvaluator:
        def penalized_cost(self, candidate):
            calls.append(candidate)
            return 12.0, True, {"delegated": 1.0}

    ts._search_evaluator = DummySearchEvaluator()

    assert ts._penalized_cost(solution) == (12.0, True, {"delegated": 1.0})
    assert calls == [solution]


def test_tabu_search_candidate_gate_skips_rejected_neighbor(monkeypatch):
    ts = tabu_search.TabuSearch.__new__(tabu_search.TabuSearch)
    ts._evaluator = SimpleNamespace(_instance=object())
    ts._max_iterations = 1
    ts._rng = random.Random(1)
    ts._base_tabu_tenure = 3
    ts._max_stagnation = 10
    ts._penalty_adapt_interval = 50
    ts._alpha_energy = 100.0
    ts._alpha_tw = 50.0
    ts._moves_per_type = 10
    ts._max_neighbors = 10
    ts._n_customers = 50

    initial = DummySolution("initial")
    neighbor = DummySolution("rejected")

    class DummySearchEvaluator:
        def __init__(self):
            self.checked = []

        def verify_candidate(self, *, base, candidate):
            self.checked.append((base, candidate))
            return False

    search = DummySearchEvaluator()
    ts._search_evaluator = search

    monkeypatch.setattr(ts, "_repair_energy_violations", lambda solution: solution)
    monkeypatch.setattr(ts, "_ensure_all_customers_served", lambda solution: solution)
    monkeypatch.setattr(ts, "_gen_violation_directed_moves", lambda solution: iter([(neighbor, ("move", 1))]))
    monkeypatch.setattr(ts, "_gen_drone_moves", lambda solution: iter(()))
    monkeypatch.setattr(ts, "_gen_cross_truck_drone_moves", lambda solution: iter(()))
    monkeypatch.setattr(ts, "_gen_truck_relocate", lambda solution: iter(()))
    monkeypatch.setattr(ts, "_gen_ruin_recreate", lambda solution: iter(()))
    monkeypatch.setattr(ts, "_gen_or_opt", lambda solution: iter(()))
    monkeypatch.setattr(ts, "_perturb", lambda solution: solution)
    monkeypatch.setattr(ts, "_apply_2opt", lambda solution: solution)
    monkeypatch.setattr(ts, "_apply_drone_optimization", lambda solution: solution)

    scored = []

    def penalized_cost(solution):
        scored.append(solution.name)
        return 1.0, True, {}

    monkeypatch.setattr(ts, "_penalized_cost", penalized_cost)

    result = ts.run(initial, time_limit=None)

    assert result is initial
    assert search.checked
    assert all(pair == (initial, neighbor) for pair in search.checked)
    assert scored == ["initial"]


def test_tabu_search_prefilter_limits_full_candidate_scoring(monkeypatch):
    ts = tabu_search.TabuSearch.__new__(tabu_search.TabuSearch)
    ts._evaluator = SimpleNamespace(_instance=object())
    ts._max_iterations = 1
    ts._rng = random.Random(1)
    ts._base_tabu_tenure = 3
    ts._max_stagnation = 10
    ts._penalty_adapt_interval = 50
    ts._alpha_energy = 100.0
    ts._alpha_tw = 50.0
    ts._moves_per_type = 10
    ts._max_neighbors = 10
    ts._candidate_eval_limit = 2
    ts._candidate_prefilter_enabled = True
    ts._n_customers = 50
    ts._search_evaluator = None

    initial = DummySolution("initial")
    neighbors = [
        (DummySolution("bad"), ("move", "bad")),
        (DummySolution("best"), ("move", "best")),
        (DummySolution("middle"), ("move", "middle")),
    ]
    cheap_scores = {"bad": 30.0, "best": 1.0, "middle": 2.0}

    monkeypatch.setattr(ts, "_repair_energy_violations", lambda solution: solution)
    monkeypatch.setattr(ts, "_ensure_all_customers_served", lambda solution: solution)
    monkeypatch.setattr(ts, "_gen_violation_directed_moves", lambda solution: iter(neighbors))
    monkeypatch.setattr(ts, "_gen_drone_moves", lambda solution: iter(()))
    monkeypatch.setattr(ts, "_gen_cross_truck_drone_moves", lambda solution: iter(()))
    monkeypatch.setattr(ts, "_gen_truck_relocate", lambda solution: iter(()))
    monkeypatch.setattr(ts, "_gen_ruin_recreate", lambda solution: iter(()))
    monkeypatch.setattr(ts, "_gen_or_opt", lambda solution: iter(()))
    monkeypatch.setattr(ts, "_perturb", lambda solution: solution)
    monkeypatch.setattr(ts, "_apply_2opt", lambda solution: solution)
    monkeypatch.setattr(ts, "_apply_drone_optimization", lambda solution: solution)
    monkeypatch.setattr(
        ts,
        "_cheap_neighbor_score",
        lambda current, neighbor, move_sig: cheap_scores[neighbor.name],
        raising=False,
    )

    scored = []

    def penalized_cost(solution):
        scored.append(solution.name)
        return cheap_scores.get(solution.name, 10.0), True, {}

    monkeypatch.setattr(ts, "_penalized_cost", penalized_cost)

    result = ts.run(initial, time_limit=None)

    assert result.name == "best"
    assert "bad" not in scored


def test_tabu_search_bucket_prefilter_keeps_random_diversity(monkeypatch):
    ts = tabu_search.TabuSearch.__new__(tabu_search.TabuSearch)
    ts._rng = random.Random(7)
    ts._candidate_eval_limit = 4
    ts._candidate_prefilter_enabled = True
    ts._candidate_bucket_enabled = True
    ts._candidate_bucket_shares = {
        "distance_proxy": 0.0,
        "drone_saving": 0.0,
        "violation_fix": 0.0,
        "random_diversity": 1.0,
    }

    current = DummySolution("current")
    moves = [(DummySolution(f"candidate-{i}"), ("relocate", i)) for i in range(10)]
    monkeypatch.setattr(
        ts,
        "_cheap_neighbor_score",
        lambda current, neighbor, move_sig: int(neighbor.name.split("-")[-1]),
        raising=False,
    )

    selected = ts._rank_potential_moves(current, moves)
    selected_names = [neighbor.name for neighbor, _ in selected]

    assert len(selected) == 4
    assert selected_names != ["candidate-0", "candidate-1", "candidate-2", "candidate-3"]
    assert ts._last_selected_by_bucket["random_diversity"] == 4


def test_tabu_search_bucket_prefilter_preserves_violation_moves(monkeypatch):
    ts = tabu_search.TabuSearch.__new__(tabu_search.TabuSearch)
    ts._rng = random.Random(1)
    ts._candidate_eval_limit = 4
    ts._candidate_prefilter_enabled = True
    ts._candidate_bucket_enabled = True
    ts._candidate_bucket_shares = {
        "distance_proxy": 0.5,
        "drone_saving": 0.0,
        "violation_fix": 0.5,
        "random_diversity": 0.0,
    }

    current = DummySolution("current")
    moves = [
        (DummySolution("cheap-1"), ("relocate", 1)),
        (DummySolution("cheap-2"), ("swap", 2)),
        (DummySolution("bad-but-violation"), ("fix_energy_remove", 9, 4)),
        (DummySolution("cheap-3"), ("or_opt", 3)),
        (DummySolution("expensive"), ("relocate", 4)),
    ]
    cheap_scores = {
        "cheap-1": 1.0,
        "cheap-2": 2.0,
        "cheap-3": 3.0,
        "bad-but-violation": 999.0,
        "expensive": 1000.0,
    }
    monkeypatch.setattr(
        ts,
        "_cheap_neighbor_score",
        lambda current, neighbor, move_sig: cheap_scores[neighbor.name],
        raising=False,
    )

    selected = ts._rank_potential_moves(current, moves)

    assert ("fix_energy_remove", 9, 4) in [move_sig for _, move_sig in selected]
    assert ts._last_selected_by_bucket["violation_fix"] == 1


def test_tabu_search_bucket_prefilter_deduplicates_move_signatures():
    ts = tabu_search.TabuSearch.__new__(tabu_search.TabuSearch)
    ts._rng = random.Random(1)
    ts._candidate_eval_limit = 10
    ts._candidate_prefilter_enabled = True
    ts._candidate_bucket_enabled = True
    ts._candidate_bucket_shares = {
        "distance_proxy": 1.0,
        "drone_saving": 0.0,
        "violation_fix": 0.0,
        "random_diversity": 0.0,
    }

    current = DummySolution("current")
    duplicate_sig = ("drone_insert", 1, 0, 99)
    moves = [
        (DummySolution("first"), duplicate_sig),
        (DummySolution("second"), duplicate_sig),
        (DummySolution("third"), ("relocate", 2)),
    ]

    selected = ts._rank_potential_moves(current, moves)

    assert [neighbor.name for neighbor, _ in selected].count("first") == 1
    assert "second" not in [neighbor.name for neighbor, _ in selected]
    assert len(selected) == 2


def test_tabu_search_records_neighborhood_stats():
    ts = tabu_search.TabuSearch.__new__(tabu_search.TabuSearch)
    ts._last_candidate_pool_size = 12
    ts._last_selected_candidate_count = 5
    ts._last_selected_by_bucket = {
        "distance_proxy": 2,
        "drone_saving": 1,
        "violation_fix": 1,
        "random_diversity": 1,
    }
    ts._reset_stats()

    ts._record_stats(
        iteration=3,
        start_time=tabu_search.time.perf_counter(),
        current_cost=10.0,
        best_cost=9.0,
        best_feasible_cost=8.0,
        neighbors_checked=5,
    )

    assert ts.stats["candidate_pool_size"] == [12]
    assert ts.stats["selected_candidate_count"] == [5]
    assert ts.stats["selected_by_bucket"] == [
        {
            "distance_proxy": 2,
            "drone_saving": 1,
            "violation_fix": 1,
            "random_diversity": 1,
        }
    ]
    assert ts.stats["best_feasible_cost"] == [8.0]


def test_ts_neighborhood_benchmark_writes_first_hit_metrics(tmp_path, monkeypatch):
    from scripts import benchmark_ts_neighborhood_quality as bench

    out = tmp_path / "benchmark.json"
    ts_stats = {
        "elapsed_time": [1.0, 2.0, 3.0],
        "best_feasible_cost": [95.0, 88.0, 66.0],
        "candidate_pool_size": [10, 14, 12],
        "selected_candidate_count": [4, 4, 4],
        "selected_by_bucket": [
            {"distance_proxy": 1, "drone_saving": 1, "violation_fix": 1, "random_diversity": 1},
            {"distance_proxy": 1, "drone_saving": 2, "violation_fix": 0, "random_diversity": 1},
            {"distance_proxy": 2, "drone_saving": 1, "violation_fix": 0, "random_diversity": 1},
        ],
    }
    ga_stats = {
        "elapsed_time": [1.0, 2.0],
        "best_feasible_fitness": [80.0, 70.0],
    }

    monkeypatch.setattr(
        bench,
        "run_ts",
        lambda *args, **kwargs: {
            "cost": 66.0,
            "feasible": True,
            "runtime": 3.0,
            "iterations_completed": 3,
            "ts_stats": ts_stats,
        },
    )
    monkeypatch.setattr(
        bench,
        "run_ga",
        lambda *args, **kwargs: {
            "cost": 70.0,
            "feasible": True,
            "runtime": 2.0,
            "generations_completed": 2,
            "ga_stats": ga_stats,
        },
    )

    result = bench.run_benchmark(
        instance="R_30_10_2",
        size=10,
        seeds=[44],
        time_limit=60.0,
        out_path=out,
    )

    saved = bench.json.loads(Path(out).read_text())
    assert result == saved
    assert saved["alns_target_cost"] == 65.72
    assert saved["runs"][0]["ts"]["first_hit"]["cost_le_90"] == 2.0
    assert saved["runs"][0]["ts"]["first_hit"]["alns_plus_5pct"] == 3.0
    assert saved["runs"][0]["ga"]["first_hit"]["cost_le_90"] == 1.0
    assert saved["runs"][0]["ts"]["neighborhood_stats_summary"]["bucket_selection_totals"] == {
        "distance_proxy": 4,
        "drone_saving": 4,
        "violation_fix": 1,
        "random_diversity": 3,
    }


def test_tabu_search_capacity_insert_creates_route_when_existing_routes_full():
    route = tabu_search.TruckRoute(route_id=0, nodes=[0, 1, 99], capacity=10.0)
    solution = SimpleNamespace(truck_routes=[route])
    ts = tabu_search.TabuSearch.__new__(tabu_search.TabuSearch)
    ts._demands = {1: 10.0, 2: 5.0}
    ts._depots = {0, 99}
    ts._depot_start = 0
    ts._depot_end = 99
    ts._node_index = {0: 0, 1: 1, 2: 2, 99: 3}
    ts._truck_dist = [
        [0.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 0.0],
    ]

    inserted = ts._insert_customer_with_capacity(solution, 2, truck_capacity=10.0)

    assert inserted is True
    assert [r.nodes for r in solution.truck_routes] == [[0, 1, 99], [0, 2, 99]]


def test_ga_infeasible_fitness_uses_full_delay_cost():
    details = SimpleNamespace(
        result=SimpleNamespace(
            feasible=False,
            total_cost=float("inf"),
            truck_distance_cost=10.0,
            drone_distance_cost=2.0,
            delay_penalty=12.0,
        ),
        robustness=SimpleNamespace(task_breakdown=[]),
        delay_breakdown=SimpleNamespace(violations=[]),
    )
    ga = GeneticAlgorithm.__new__(GeneticAlgorithm)
    ga.evaluator = SimpleNamespace(evaluate_with_details=lambda solution: details)
    ga.instance = SimpleNamespace(
        customer_manager=SimpleNamespace(demands=lambda: {}),
        vehicle_specs={"truck": SimpleNamespace(capacity=100.0)},
    )
    individual = Individual(
        solution=SimpleNamespace(truck_routes=[], drone_tasks=[])
    )

    ga._evaluate_individual(individual)

    assert individual.feasible is False
    assert individual.delay_penalty == 12.0
    assert individual.fitness == 1_000_012.0


def test_ga_evaluate_individual_delegates_to_search_evaluator():
    ga = GeneticAlgorithm.__new__(GeneticAlgorithm)
    solution = SimpleNamespace(truck_routes=[], drone_tasks=[])
    individual = Individual(solution=solution)
    calls = []

    class DummySearchEvaluator:
        def penalized_cost(self, candidate):
            calls.append(candidate)
            return 42.0, True, {}

        def evaluate_solution(self, candidate):
            return SimpleNamespace(
                truck_distance_cost=20.0,
                drone_distance_cost=2.0,
                delay_penalty=1.0,
            )

    ga.search_evaluator = DummySearchEvaluator()

    ga._evaluate_individual(individual)

    assert calls == [solution]
    assert individual.fitness == 42.0
    assert individual.feasible is True
    assert individual.truck_distance == 20.0
    assert individual.drone_distance == 2.0
    assert individual.delay_penalty == 1.0


def test_ga_evaluate_solution_helper_uses_search_evaluator():
    ga = GeneticAlgorithm.__new__(GeneticAlgorithm)
    solution = SimpleNamespace(truck_routes=[], drone_tasks=[])
    calls = []

    class DummySearchEvaluator:
        def evaluate_solution(self, candidate):
            calls.append(candidate)
            return SimpleNamespace(feasible=True, total_cost=7.0)

    ga.search_evaluator = DummySearchEvaluator()

    result = ga._evaluate_solution(solution)

    assert result.total_cost == 7.0
    assert calls == [solution]


def test_ga_drone_anchor_prefilter_keeps_best_lightweight_candidates(monkeypatch):
    ga = GeneticAlgorithm.__new__(GeneticAlgorithm)
    ga._drone_mutation_candidate_eval_limit = 2

    candidates = [
        ("source", 1, "launch", 10, "retrieve", 20),
        ("source", 2, "launch", 11, "retrieve", 21),
        ("source", 3, "launch", 12, "retrieve", 22),
    ]
    scores = {1: 30.0, 2: 1.0, 3: 2.0}
    monkeypatch.setattr(
        ga,
        "_score_drone_anchor_candidate",
        lambda candidate: scores[candidate[1]],
        raising=False,
    )

    ranked = ga._rank_drone_anchor_candidates(candidates)

    assert [candidate[1] for candidate in ranked] == [2, 3]


def test_ga_aggressive_drone_optimization_respects_top_n_limit():
    ga = GeneticAlgorithm.__new__(GeneticAlgorithm)
    ga.population = [
        Individual(solution=DummySolution(f"sol-{i}"), fitness=100.0, feasible=True)
        for i in range(5)
    ]
    ga._aggressive_drone_top_n = 2
    ga._check_time_limits = lambda: False
    ga.instance = object()
    ga.evaluator = object()
    ga.rng = random.Random(1)
    ga.repair = SimpleNamespace(repair_solution=lambda solution: solution)
    ga._evaluate_solution = lambda solution: SimpleNamespace(
        feasible=False,
        total_cost=float("inf"),
        truck_distance_cost=0.0,
        drone_distance_cost=0.0,
        delay_penalty=0.0,
    )

    processed = []

    class CountingDroneBuilder:
        def optimize_drone_tasks(self, solution):
            processed.append(solution.name)
            return solution

    ga.drone_builder = CountingDroneBuilder()

    ga._apply_aggressive_drone_optimization()

    assert processed == ["sol-0", "sol-1"]


def test_ga_create_population_stops_before_offspring_when_time_limit_reached(monkeypatch):
    ga = GeneticAlgorithm.__new__(GeneticAlgorithm)
    ga.config = SimpleNamespace(
        population_size=3,
        elite_size=1,
        crossover_rate=0.0,
        mutation_rate=0.0,
    )
    ga.population = [
        Individual(solution=DummySolution("elite"), fitness=1.0, feasible=True)
    ]
    ga._check_time_limits = lambda: True
    ga._tournament_selection = lambda: (_ for _ in ()).throw(
        AssertionError("GA continued creating offspring after time limit")
    )

    new_population = ga._create_new_population()

    assert new_population == ga.population


def test_ga_run_returns_best_feasible_individual_over_lower_penalized_infeasible(monkeypatch):
    ga = GeneticAlgorithm.__new__(GeneticAlgorithm)
    ga.config = SimpleNamespace(
        generations=1,
        adaptive_enabled=False,
        adaptation_interval=1,
        max_stagnation=10,
    )
    ga.generation = 0
    ga.stagnation_counter = 0
    ga.start_time = 0.0
    ga.stats = {
        "generations": [],
        "best_fitness": [],
        "avg_fitness": [],
        "feasible_count": [],
        "diversity": [],
        "unique_solutions": [],
    }
    feasible = Individual(
        solution=DummySolution("feasible"),
        fitness=100.0,
        feasible=True,
    )
    infeasible = Individual(
        solution=DummySolution("infeasible"),
        fitness=1.0,
        feasible=False,
    )

    def initialize_population(initial_solution):
        ga.population = [feasible]
        ga.best_individual = feasible

    monkeypatch.setattr(ga, "initialize_population", initialize_population)
    monkeypatch.setattr(ga, "_check_time_limits", lambda: False)
    monkeypatch.setattr(ga, "_create_new_population", lambda: [infeasible, feasible])
    monkeypatch.setattr(ga, "_apply_local_search", lambda: None)
    monkeypatch.setattr(ga, "_apply_aggressive_drone_optimization", lambda: None)
    monkeypatch.setattr(ga, "_record_statistics", lambda: None)

    best = ga.run(DummySolution("initial"))

    assert best.feasible is True
    assert best.solution.name == "feasible"
