"""Tests for incremental checkpoint updates in the three-algorithm runner."""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

from heuristics.ga import ga as ga_module
from revision import validate_alns_ts_ga


def test_run_algorithm_adds_missing_seed_to_existing_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(validate_alns_ts_ga, "OUT_DIR", tmp_path)
    monkeypatch.setattr(validate_alns_ts_ga, "record_run", lambda *args, **kwargs: "run-id")
    monkeypatch.setattr(validate_alns_ts_ga, "update_run", lambda *args, **kwargs: None)

    calls = []

    def run_fn(instance, instance_size, params, seed):
        calls.append((instance, seed))
        return {
            "cost": float(seed),
            "feasible": True,
            "runtime": 0.01,
            "delay_cost": 0.0,
            "truck_cost": float(seed),
            "drone_cost": 0.0,
        }

    validate_alns_ts_ga.run_algorithm(
        "mock", 50, ["R_30_50_1"], [1], {}, run_fn
    )
    validate_alns_ts_ga.run_algorithm(
        "mock", 50, ["R_30_50_1"], [1, 2], {}, run_fn
    )

    assert calls == [("R_30_50_1", 1), ("R_30_50_1", 2)]

    checkpoint = json.loads((tmp_path / "inst50_mock_checkpoint.json").read_text())
    result = checkpoint["results"][0]
    assert result["instance"] == "R_30_50_1"
    assert result["seeds"] == [1, 2]
    assert [entry["seed"] for entry in result["seed_results"]] == [1, 2]


def test_seed_checkpoint_preserves_initial_constructor_metrics():
    seed_result = validate_alns_ts_ga._seed_result_from_run(
        101,
        {
            "cost": 123.0,
            "feasible": True,
            "runtime": 0.5,
            "initial_feasible": True,
            "initial_constructor": "deadline_backbone_drone_repair",
            "initial_constructor_final_violations": 0,
            "iterations_completed": 7,
        },
        runtime=1.0,
    )

    assert seed_result["initial_feasible"] is True
    assert seed_result["initial_constructor"] == "deadline_backbone_drone_repair"
    assert seed_result["initial_constructor_final_violations"] == 0
    assert seed_result["iterations_completed"] == 7


def test_run_algorithm_reruns_infeasible_seed_from_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(validate_alns_ts_ga, "OUT_DIR", tmp_path)
    monkeypatch.setattr(validate_alns_ts_ga, "record_run", lambda *args, **kwargs: "run-id")
    monkeypatch.setattr(validate_alns_ts_ga, "update_run", lambda *args, **kwargs: None)

    checkpoint = {
        "algo": "mock",
        "instance_size": 50,
        "run_tag": "",
        "timestamp": "old",
        "results": [
            {
                "instance": "R_30_50_3",
                "seed": 2,
                "cost": 20.0,
                "feasible": True,
                "runtime": 0.2,
                "seeds": [1, 2],
                "seed_results": [
                    {
                        "seed": 1,
                        "cost": validate_alns_ts_ga.INFEASIBLE_PENALTY,
                        "feasible": False,
                        "runtime": 0.1,
                        "delay_cost": 0.0,
                        "truck_cost": 0.0,
                        "drone_cost": 0.0,
                        "error": "old infeasible",
                    },
                    {
                        "seed": 2,
                        "cost": 20.0,
                        "feasible": True,
                        "runtime": 0.2,
                        "delay_cost": 0.0,
                        "truck_cost": 20.0,
                        "drone_cost": 0.0,
                        "error": None,
                    },
                ],
            }
        ],
        "summary": {},
    }
    (tmp_path / "inst50_mock_checkpoint.json").write_text(json.dumps(checkpoint))

    calls = []

    def run_fn(instance, instance_size, params, seed):
        calls.append((instance, seed))
        return {
            "cost": 10.0,
            "feasible": True,
            "runtime": 0.01,
            "delay_cost": 0.0,
            "truck_cost": 10.0,
            "drone_cost": 0.0,
        }

    validate_alns_ts_ga.run_algorithm(
        "mock", 50, ["R_30_50_3"], [1, 2], {}, run_fn
    )

    assert calls == [("R_30_50_3", 1)]

    updated = json.loads((tmp_path / "inst50_mock_checkpoint.json").read_text())
    result = updated["results"][0]
    assert result["cost"] == 10.0
    assert result["seed"] == 1
    assert [entry["seed"] for entry in result["seed_results"]] == [1, 2]
    assert result["seed_results"][0]["feasible"] is True
    assert result["seed_results"][0]["error"] is None
    assert result["seed_results"][1]["cost"] == 20.0


def test_compute_seed_summary_uses_all_feasible_seed_runs():
    results = [
        {
            "instance": "R_30_50_1",
            "seed": 1,
            "cost": 10.0,
            "feasible": True,
            "runtime": 1.0,
            "seed_results": [
                {"seed": 1, "cost": 10.0, "feasible": True, "runtime": 1.0},
                {"seed": 2, "cost": 20.0, "feasible": True, "runtime": 2.0},
            ],
        },
        {
            "instance": "R_30_50_2",
            "seed": 1,
            "cost": 30.0,
            "feasible": True,
            "runtime": 3.0,
            "seed_results": [
                {"seed": 1, "cost": 30.0, "feasible": True, "runtime": 3.0},
                {
                    "seed": 2,
                    "cost": validate_alns_ts_ga.INFEASIBLE_PENALTY,
                    "feasible": False,
                    "runtime": 4.0,
                },
            ],
        },
    ]

    summary = validate_alns_ts_ga.compute_seed_summary(results)

    assert summary["n_runs"] == 4
    assert summary["n_feasible"] == 3
    assert summary["mean_cost"] == 20.0
    assert summary["per_instance_mean"] == {
        "R_30_50_1": 15.0,
        "R_30_50_2": 30.0,
    }
    assert summary["per_instance_min"] == {
        "R_30_50_1": 10.0,
        "R_30_50_2": 30.0,
    }


def test_update_aggregate_output_upserts_rows_and_recomputes_summary(tmp_path):
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(
        json.dumps(
            {
                "description": "existing aggregate",
                "rows": [
                    {
                        "instance": "R_30_50_1",
                        "alns_mean": 10.0,
                        "alns_best": 9.0,
                        "ts_mean": 20.0,
                        "ts_best": 19.0,
                        "ga_mean": 30.0,
                        "ga_best": 29.0,
                    },
                    {
                        "instance": "R_50_50_1",
                        "alns_mean": 999.0,
                        "alns_best": 999.0,
                        "ts_mean": 999.0,
                        "ts_best": 999.0,
                        "ga_mean": 999.0,
                        "ga_best": 999.0,
                    },
                ],
            }
        )
    )
    output = {
        "instance_size": 50,
        "instances": ["R_50_50_1"],
    }
    alns = [
        {
            "instance": "R_50_50_1",
            "seed_results": [
                {"seed": 101, "cost": 100.0, "feasible": True, "runtime": 1.0},
                {"seed": 102, "cost": 110.0, "feasible": True, "runtime": 3.0},
            ],
        }
    ]
    ts = [
        {
            "instance": "R_50_50_1",
            "seed_results": [
                {"seed": 101, "cost": 200.0, "feasible": True, "runtime": 2.0},
            ],
        }
    ]
    ga = [
        {
            "instance": "R_50_50_1",
            "seed_results": [
                {
                    "seed": 101,
                    "cost": validate_alns_ts_ga.INFEASIBLE_PENALTY,
                    "feasible": False,
                    "runtime": 4.0,
                },
            ],
        }
    ]

    csv_path = validate_alns_ts_ga.update_aggregate_output(
        aggregate_path,
        output,
        alns,
        ts,
        ga,
        run_tag="unit",
    )

    updated = json.loads(aggregate_path.read_text())
    rows = {row["instance"]: row for row in updated["rows"]}
    assert rows["R_50_50_1"]["alns_mean"] == 105.0
    assert rows["R_50_50_1"]["alns_best"] == 100.0
    assert rows["R_50_50_1"]["alns_feasible"] == "2/2"
    assert rows["R_50_50_1"]["ga_mean"] is None
    assert updated["summary"]["R30"]["alns"]["mean_of_instance_means"] == 10.0
    assert updated["summary"]["R50"]["alns"]["mean_of_instance_means"] == 105.0
    assert updated["summary"]["ALL"]["alns"]["mean_of_instance_means"] == 57.5
    assert csv_path.exists()


def test_select_instances_accepts_exact_names_and_classes():
    all_names = [
        "R_30_50_1",
        "R_30_50_2",
        "R_40_50_1",
        "R_50_50_1",
        "R_50_50_2",
    ]

    selected = validate_alns_ts_ga.select_instances(
        all_names,
        instance_size=50,
        max_instances=None,
        instance_prefix=None,
        instance_names="R_50_50_2,R_30_50_1",
        instance_classes=None,
    )
    assert selected == ["R_30_50_1", "R_50_50_2"]

    selected = validate_alns_ts_ga.select_instances(
        all_names,
        instance_size=50,
        max_instances=None,
        instance_prefix=None,
        instance_names=None,
        instance_classes="R40,R50",
    )
    assert selected == ["R_40_50_1", "R_50_50_1", "R_50_50_2"]


def test_parse_algorithms_accepts_all_and_subsets():
    assert validate_alns_ts_ga.parse_algorithms("all") == ["alns", "ts", "ga"]
    assert validate_alns_ts_ga.parse_algorithms("ts,ga") == ["ts", "ga"]
    assert validate_alns_ts_ga.parse_algorithms("GA,TS") == ["ts", "ga"]


def test_update_aggregate_output_preserves_unselected_algorithms(tmp_path):
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(
        json.dumps(
            {
                "description": "existing aggregate",
                "rows": [
                    {
                        "instance": "R_50_50_1",
                        "alns_mean": 111.0,
                        "alns_best": 100.0,
                        "alns_feasible": "3/3",
                        "ts_mean": 999.0,
                        "ts_best": 999.0,
                        "ga_mean": 333.0,
                        "ga_best": 300.0,
                        "ga_feasible": "3/3",
                    },
                ],
            }
        )
    )
    output = {"instance_size": 50, "instances": ["R_50_50_1"]}
    ts = [
        {
            "instance": "R_50_50_1",
            "seed_results": [
                {"seed": 101, "cost": 200.0, "feasible": True, "runtime": 2.0},
                {"seed": 102, "cost": 220.0, "feasible": True, "runtime": 4.0},
            ],
        }
    ]

    validate_alns_ts_ga.update_aggregate_output(
        aggregate_path,
        output,
        [],
        ts,
        [],
        run_tag="fair300",
        selected_algorithms=["ts"],
    )

    row = json.loads(aggregate_path.read_text())["rows"][0]
    assert row["alns_mean"] == 111.0
    assert row["alns_best"] == 100.0
    assert row["ga_mean"] == 333.0
    assert row["ga_best"] == 300.0
    assert row["ts_mean"] == 210.0
    assert row["ts_best"] == 200.0


def test_run_alns_uses_shared_instance_loader(monkeypatch):
    expected_instance = object()
    expected_initial = object()
    expected_solution = object()
    captured = {}

    class DummyEvaluator:
        def evaluate_solution(self, solution):
            captured["evaluated_solution"] = solution
            return SimpleNamespace(
                total_cost=111.0,
                feasible=True,
                delay_penalty=5.0,
                truck_distance_cost=100.0,
                drone_distance_cost=6.0,
            )

    class DummyALNS:
        def __init__(self, **kwargs):
            captured["alns_kwargs"] = kwargs

        def run(self, initial, time_limit=None):
            captured["initial"] = initial
            captured["time_limit"] = time_limit
            return expected_solution

    def shared_loader(instance_name, seed, instance_dir):
        captured["loader_args"] = (instance_name, seed, instance_dir)
        return expected_instance, DummyEvaluator(), {"node": "class"}

    monkeypatch.setattr(validate_alns_ts_ga, "load_instance_for_tuning", shared_loader)
    monkeypatch.setattr(
        validate_alns_ts_ga,
        "read_instance",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("run_alns used the old direct read_instance path")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        validate_alns_ts_ga,
        "build_feasible_initial_solution",
        lambda instance, evaluator: (
            expected_initial,
            SimpleNamespace(
                constructor="test_constructor",
                feasible=True,
                initial_violations=0,
                final_violations=0,
                initial_lateness=0.0,
                final_lateness=0.0,
                initial_delay_cost=0.0,
                final_delay_cost=0.0,
                drone_tasks_added=0,
                iterations=0,
                reason="feasible",
            ),
        ),
    )
    monkeypatch.setattr(validate_alns_ts_ga, "DestroyRandom", lambda *args, **kwargs: object())
    monkeypatch.setattr(validate_alns_ts_ga, "DestroyWorstDistance", lambda *args, **kwargs: object())
    monkeypatch.setattr(validate_alns_ts_ga, "DestroyShaw", lambda *args, **kwargs: object())
    monkeypatch.setattr(validate_alns_ts_ga, "RepairCheapest", lambda *args, **kwargs: object())
    monkeypatch.setattr(validate_alns_ts_ga, "RepairDronePriorityRegret", lambda *args, **kwargs: object())
    monkeypatch.setattr(validate_alns_ts_ga, "RepairTruckFirst", lambda *args, **kwargs: object())
    monkeypatch.setattr(validate_alns_ts_ga, "RepairEqualPriority", lambda *args, **kwargs: object())
    monkeypatch.setattr(validate_alns_ts_ga, "SimulatedAnnealingALNS", DummyALNS)

    result = validate_alns_ts_ga.run_alns(
        "R_30_50_1",
        50,
        seed=101,
        iterations=10,
        time_limit_override=0.1,
    )

    assert captured["loader_args"] == ("R_30_50_1", 101, "Instance50")
    assert captured["initial"] is expected_initial
    assert captured["evaluated_solution"] is expected_solution
    assert captured["alns_kwargs"]["instance"] is expected_instance
    assert captured["alns_kwargs"]["evaluator"].__class__ is DummyEvaluator
    assert result["feasible"] is True
    assert result["cost"] == 111.0
    assert result["initial_constructor"] == "test_constructor"


def test_run_ts_uses_two_phase_initial_solution(monkeypatch):
    expected_initial = object()
    expected_metrics = {
        "initial_cost": 999.0,
        "initial_feasible": False,
        "initial_delay_cost": 9.0,
        "initial_truck_cost": 900.0,
        "initial_drone_cost": 90.0,
        "initial_num_routes": 3,
        "initial_num_drone_tasks": 2,
    }
    captured = {}

    class DummyEvaluator:
        def evaluate_solution(self, solution):
            captured["evaluated_solution"] = solution
            return SimpleNamespace(
                total_cost=123.0,
                feasible=True,
                delay_penalty=1.0,
                truck_distance_cost=120.0,
                drone_distance_cost=2.0,
            )

    class DummyTabuSearch:
        def __init__(self, **kwargs):
            captured["tabu_kwargs"] = kwargs

        def run(self, initial, time_limit=None):
            captured["initial"] = initial
            captured["time_limit"] = time_limit
            return initial

    dummy_module = types.SimpleNamespace(TabuSearch=DummyTabuSearch)
    monkeypatch.setitem(sys.modules, "tabu_search", dummy_module)
    monkeypatch.setattr(
        validate_alns_ts_ga,
        "load_instance_for_tuning",
        lambda *args, **kwargs: (object(), DummyEvaluator(), {}),
    )
    monkeypatch.setattr(
        validate_alns_ts_ga,
        "build_two_phase_initial_solution",
        lambda instance: (_ for _ in ()).throw(
            AssertionError("run_ts bypassed shared initial helper")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        validate_alns_ts_ga,
        "build_shared_initial_solution",
        lambda instance, evaluator: (expected_initial, expected_metrics),
        raising=False,
    )

    result = validate_alns_ts_ga.run_ts(
        "R_30_50_3",
        50,
        {"tabu_tenure": 3, "max_iterations": 5, "max_stagnation": 2},
        seed=101,
        time_limit_override=0.1,
    )

    assert result["feasible"] is True
    assert captured["initial"] is expected_initial
    assert captured["evaluated_solution"] is expected_initial
    assert result["initial_feasible"] is False
    assert result["initial_cost"] == 999.0
    assert result["initial_num_drone_tasks"] == 2


def test_run_ga_returns_final_evaluator_recheck(monkeypatch):
    expected_initial = object()
    expected_solution = object()
    expected_metrics = {
        "initial_cost": 77.0,
        "initial_feasible": True,
        "initial_delay_cost": 1.0,
        "initial_truck_cost": 70.0,
        "initial_drone_cost": 6.0,
        "initial_num_routes": 2,
        "initial_num_drone_tasks": 1,
    }
    captured = {}

    class DummyEvaluator:
        def evaluate_solution(self, solution):
            captured["evaluated_solution"] = solution
            return SimpleNamespace(
                total_cost=42.0,
                feasible=True,
                delay_penalty=3.0,
                truck_distance_cost=37.0,
                drone_distance_cost=2.0,
            )

    class DummyGAConfig:
        def __init__(self, **kwargs):
            captured["ga_config"] = kwargs

    class DummyGeneticAlgorithm:
        def __init__(self, instance, config, evaluator, rng=None):
            captured["ga_init"] = (instance, config, evaluator, rng)

        def run(self, initial):
            captured["initial"] = initial
            return SimpleNamespace(
                solution=expected_solution,
                fitness=999.0,
                feasible=True,
                truck_distance=900.0,
                drone_distance=90.0,
                delay_penalty=9.0,
            )

    monkeypatch.setattr(ga_module, "GAConfig", DummyGAConfig)
    monkeypatch.setattr(ga_module, "GeneticAlgorithm", DummyGeneticAlgorithm)
    monkeypatch.setattr(
        validate_alns_ts_ga,
        "load_instance_for_tuning",
        lambda *args, **kwargs: (object(), DummyEvaluator(), {}),
    )
    monkeypatch.setattr(
        validate_alns_ts_ga,
        "build_two_phase_initial_solution",
        lambda instance: (_ for _ in ()).throw(
            AssertionError("run_ga bypassed shared initial helper")
        ),
    )
    monkeypatch.setattr(
        validate_alns_ts_ga,
        "build_shared_initial_solution",
        lambda instance, evaluator: (expected_initial, expected_metrics),
        raising=False,
    )

    result = validate_alns_ts_ga.run_ga(
        "R_30_50_2",
        50,
        {
            "population_size": 5,
            "generations": 2,
            "tournament_size": 2,
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "elite_size": 1,
            "max_stagnation": 2,
            "truck_route_crossover_rate": 0.7,
            "drone_task_mutation_rate": 0.3,
            "route_segment_swap_rate": 0.4,
        },
        seed=101,
        time_limit_override=0.1,
    )

    assert captured["initial"] is expected_initial
    assert captured["evaluated_solution"] is expected_solution
    assert result["cost"] == 42.0
    assert result["feasible"] is True
    assert result["truck_cost"] == 37.0
    assert result["drone_cost"] == 2.0
    assert result["delay_cost"] == 3.0
    assert result["runtime"] >= 0.0
    assert result["initial_feasible"] is True
    assert result["initial_cost"] == 77.0
    assert result["initial_num_routes"] == 2
