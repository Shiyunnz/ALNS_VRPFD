#!/usr/bin/env python3
"""Validate ALNS, TS, and GA on a given instance size.

Compares all three algorithms on the benchmark set for a specified instance size.
Each algorithm runs on N instances x S seeds.

ALNS: 4000 iterations  (from config/alns_config.yaml)
TS:   tuned config + 300s time limit
GA:   tuned config + 300s time limit

Usage:
    cd code/
    python revision/validate_alns_ts_ga.py --instance-size 25       # all R_*_25_*.txt
    python revision/validate_alns_ts_ga.py --instance-size 25 --instances 5  # first 5 only
    python revision/validate_alns_ts_ga.py --instance-size 10 --seeds 10     # 10 seeds
    python revision/validate_alns_ts_ga.py --instance-size 25 --seeds 101-105
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "heuristics" / "tabu_search"))
sys.path.insert(0, str(PROJECT_ROOT / "heuristics" / "ga"))

from alns_vrpfd.utils.config_loader import ALNSConfig
from alns_vrpfd.evaluation import SearchEvaluator, SubrouteRobustVerifier
from alns_vrpfd.model.feasible_initializer import build_feasible_initial_solution
from alns_vrpfd.model.initializer import build_two_phase_initial_solution
logging.getLogger("alns_vrpfd.model.initializer").setLevel(logging.WARNING)
from alns_vrpfd.core.operators import (
    DestroyRandom, DestroyShaw, DestroyWorstDistance,
    RepairCheapest, RepairDronePriorityRegret, RepairEqualPriority, RepairTruckFirst,
)
from alns_vrpfd.core.sa import SANNCfg, SimulatedAnnealingALNS
from revision.tune_base import (
    load_instance_for_tuning, INFEASIBLE_PENALTY,
)
from revision.manifest import config_hash, record_run, update_run, REVISION_ROOT

OUT_DIR = REVISION_ROOT / "main_tables"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Config cache (loaded once) ──────────────────────────────────────────

ALNS_CONFIG_PATH = PROJECT_ROOT / "config" / "alns_config.yaml"
ALNS_CFG = ALNSConfig(str(ALNS_CONFIG_PATH))

TS_CONFIG_PATH = REVISION_ROOT / "configs" / "ts" / "final_config.json"
GA_CONFIG_PATH = REVISION_ROOT / "configs" / "ga" / "final_config.json"

TS_FINAL_CFG: Dict[str, Any] = json.loads(TS_CONFIG_PATH.read_text())
GA_FINAL_CFG: Dict[str, Any] = json.loads(GA_CONFIG_PATH.read_text())


# ─── Helpers ─────────────────────────────────────────────────────────────

def discover_instances(
    instance_size: int,
    max_instances: Optional[int] = None,
    instance_prefix: str | None = None,
) -> List[str]:
    """Auto-discover instances from data/Instance{size}/."""
    data_dir = PROJECT_ROOT / "data" / f"Instance{instance_size}"
    if not data_dir.exists():
        raise FileNotFoundError(f"Instance directory not found: {data_dir}")
    files = sorted(data_dir.glob("*.txt"))
    names = [f.stem for f in files]
    if instance_prefix:
        names = [name for name in names if name.startswith(instance_prefix)]
    if max_instances:
        names = names[:max_instances]
    return names


def _split_csv(value: str | None) -> List[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _class_prefix(token: str, instance_size: int) -> str:
    value = token.strip().upper()
    if not value:
        raise ValueError("Empty instance class")
    if value.isdigit():
        return f"R_{value}_{instance_size}_"
    if value.startswith("R_"):
        parts = value.split("_")
        if len(parts) == 2 and parts[1].isdigit():
            return f"R_{parts[1]}_{instance_size}_"
        return value if value.endswith("_") else f"{value}_"
    if value.startswith("R") and value[1:].isdigit():
        return f"R_{value[1:]}_{instance_size}_"
    raise ValueError(
        f"Invalid instance class '{token}'. Use forms like R50, 50, R_50, or R_50_{instance_size}."
    )


def select_instances(
    all_names: List[str],
    *,
    instance_size: int,
    max_instances: Optional[int],
    instance_prefix: str | None,
    instance_names: str | None,
    instance_classes: str | None,
) -> List[str]:
    """Select instances by exact names, class prefixes, legacy prefix, and count."""
    available = set(all_names)
    selected: set[str] = set()

    exact_names = _split_csv(instance_names)
    missing = [name for name in exact_names if name not in available]
    if missing:
        raise ValueError(f"Unknown instance name(s): {', '.join(missing)}")
    selected.update(exact_names)

    for token in _split_csv(instance_classes):
        prefix = _class_prefix(token, instance_size)
        selected.update(name for name in all_names if name.startswith(prefix))

    if not exact_names and not _split_csv(instance_classes):
        selected.update(all_names)

    if instance_prefix:
        selected = {name for name in selected if name.startswith(instance_prefix)}

    ordered = [name for name in all_names if name in selected]
    if max_instances:
        ordered = ordered[:max_instances]
    return ordered


def parse_algorithms(value: str | None) -> List[str]:
    requested = [item.lower() for item in _split_csv(value or "all")]
    if not requested or "all" in requested:
        return ["alns", "ts", "ga"]
    allowed = ["alns", "ts", "ga"]
    invalid = [algo for algo in requested if algo not in allowed]
    if invalid:
        raise ValueError(f"Unknown algorithm(s): {', '.join(invalid)}")
    return [algo for algo in allowed if algo in set(requested)]


def make_alns_config_dict() -> Dict[str, Any]:
    return {
        "w_percent": ALNS_CFG.w_percent,
        "cooling_rate_initial": ALNS_CFG.cooling_rate_initial,
        "cooling_rate_final": ALNS_CFG.cooling_rate_final,
        "eta": ALNS_CFG.eta,
        "drone_priority": ALNS_CFG.drone_priority,
        "depot_bonus": ALNS_CFG.drone_bonus["depot_bonus"],
        "multi_customer_bonus": ALNS_CFG.drone_bonus["multi_customer_bonus"],
        "local_search_frequency": ALNS_CFG.local_search_frequency,
        "iterations": 4000,
    }


def make_alns_bonus() -> Dict[str, Any]:
    return {
        "depot_bonus": ALNS_CFG.drone_bonus["depot_bonus"],
        "multi_customer_bonus": ALNS_CFG.drone_bonus["multi_customer_bonus"],
        "multi_customer_threshold": 2,
        "wait_max": 20.0,
        "allow_multiple_launch_per_node": ALNS_CFG.relax_allow_multiple_launch_per_node,
    }


def parse_seed_range(s: str) -> List[int]:
    """Parse '101-105' or '10' or '100,102,104'."""
    if "-" in s:
        parts = s.split("-")
        start, end = int(parts[0]), int(parts[1])
        return list(range(start, end + 1))
    elif "," in s:
        return [int(x) for x in s.split(",")]
    else:
        return [int(s)]


def _build_search_evaluator(instance, evaluator) -> SearchEvaluator | None:
    """Build the shared TS/GA search evaluator when a real instance is available."""
    try:
        robust = instance.robust_config
        verifier = SubrouteRobustVerifier(
            instance=instance,
            drone_energy_capacity=robust.drone_battery_capacity,
            energy_uncertainty_budget=robust.energy_uncertainty_budget,
            energy_deviation_rate=robust.energy_deviation_rate,
        )
        return SearchEvaluator(
            evaluator,
            candidate_subroute_verifier=verifier,
            robust_cache_size=100_000,
        )
    except Exception:
        return None


def _search_stats(search_evaluator: SearchEvaluator | None) -> Dict[str, Any]:
    if search_evaluator is None:
        return {}
    return {
        "search_full_eval_calls": search_evaluator.full_eval_calls,
        "search_detail_eval_calls": search_evaluator.detail_eval_calls,
        "search_robust_eval_calls": search_evaluator.robust_eval_calls,
        "search_robust_cache_hits": search_evaluator.robust_cache_hits,
        "search_candidate_checks": search_evaluator.candidate_checks,
        "search_candidate_rejections": search_evaluator.candidate_rejections,
        "search_matrix_cache_hits": search_evaluator.matrix_cache_hits,
        "search_matrix_cache_misses": search_evaluator.matrix_cache_misses,
    }


def build_shared_initial_solution(instance, evaluator):
    """Build and evaluate the common initial solution used by all algorithms."""
    diagnostics = None
    initial, diagnostics = build_feasible_initial_solution(instance, evaluator)
    try:
        initial_eval = evaluator.evaluate_solution(initial)
        metrics = {
            "initial_cost": initial_eval.total_cost,
            "initial_feasible": initial_eval.feasible,
            "initial_delay_cost": initial_eval.delay_penalty,
            "initial_truck_cost": initial_eval.truck_distance_cost,
            "initial_drone_cost": initial_eval.drone_distance_cost,
        }
    except Exception as exc:
        metrics = {
            "initial_cost": INFEASIBLE_PENALTY,
            "initial_feasible": False,
            "initial_delay_cost": 0.0,
            "initial_truck_cost": 0.0,
            "initial_drone_cost": 0.0,
            "initial_error": str(exc),
        }
    metrics.update({
        "initial_num_routes": len(getattr(initial, "truck_routes", []) or []),
        "initial_num_drone_tasks": len(getattr(initial, "drone_tasks", []) or []),
    })
    if diagnostics is not None:
        metrics.update({
            "initial_constructor": diagnostics.constructor,
            "initial_constructor_feasible": diagnostics.feasible,
            "initial_constructor_initial_violations": diagnostics.initial_violations,
            "initial_constructor_final_violations": diagnostics.final_violations,
            "initial_constructor_initial_lateness": diagnostics.initial_lateness,
            "initial_constructor_final_lateness": diagnostics.final_lateness,
            "initial_constructor_initial_delay_cost": diagnostics.initial_delay_cost,
            "initial_constructor_final_delay_cost": diagnostics.final_delay_cost,
            "initial_constructor_drone_tasks_added": diagnostics.drone_tasks_added,
            "initial_constructor_iterations": diagnostics.iterations,
            "initial_constructor_reason": diagnostics.reason,
        })
    return initial, metrics


# ═══════════════════════════════════════════════════════════════════════════
#  ALNS
# ═══════════════════════════════════════════════════════════════════════════

def run_alns(
    instance_name: str,
    instance_size: int,
    seed: int,
    iterations: int = 4000,
    time_limit_override: float | None = None,
) -> Dict[str, Any]:
    instance_dir = f"Instance{instance_size}"
    instance, evaluator, _ = load_instance_for_tuning(
        instance_name, seed=seed, instance_dir=instance_dir,
    )

    sa_cfg = ALNS_CFG.build_sa_config_dict()
    sa_cfg["iterations"] = iterations
    sa_cfg["size"] = "small"
    sa_cfg["log_operator_metrics"] = False

    rng_alns = random.Random(seed)
    dp = ALNS_CFG.drone_priority
    bonus = make_alns_bonus()

    destroy_ops = [
        DestroyRandom(instance, rng=random.Random(seed + 1000),
                      anchor_strategy="rebase_to_neighbor"),
        DestroyWorstDistance(instance, rng=random.Random(seed + 1004),
                             anchor_strategy="rebase_to_neighbor"),
        DestroyShaw(instance, rng=random.Random(seed + 1002),
                    anchor_strategy="rebase_to_neighbor"),
    ]
    repair_ops = [
        RepairCheapest(instance, rng=random.Random(seed + 2004),
                       drone_priority=dp, robust_energy_mode="embedded", **bonus),
        RepairDronePriorityRegret(instance, rng=random.Random(seed + 2002),
                                  drone_priority=dp, robust_energy_mode="embedded", **bonus),
        RepairTruckFirst(instance, rng=random.Random(seed + 2003),
                         drone_priority=dp, robust_energy_mode="embedded", **bonus),
        RepairEqualPriority(instance, rng=random.Random(seed + 2001),
                            drone_priority=dp, robust_energy_mode="embedded", **bonus),
    ]

    start = time.time()
    try:
        initial, initial_metrics = build_shared_initial_solution(instance, evaluator)
        alns = SimulatedAnnealingALNS(
            instance=instance, destroy_ops=destroy_ops, repair_ops=repair_ops,
            evaluator=evaluator, cfg=SANNCfg(**sa_cfg), rng=rng_alns,
            verbose=False,
        )
        best = alns.run(initial, time_limit=time_limit_override)
    except Exception as e:
        runtime = time.time() - start
        return {"cost": INFEASIBLE_PENALTY, "feasible": False,
                "runtime": runtime, "error": str(e)}
    runtime = time.time() - start

    res = evaluator.evaluate_solution(best)
    return {
        "cost": res.total_cost, "feasible": res.feasible,
        "delay_cost": res.delay_penalty,
        "truck_cost": res.truck_distance_cost,
        "drone_cost": res.drone_distance_cost,
        "runtime": runtime,
        **initial_metrics,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  TS
# ═══════════════════════════════════════════════════════════════════════════

def run_ts(
    instance_name: str,
    instance_size: int,
    params: Dict[str, Any],
    seed: int,
    time_limit_override: float = 300.0,
) -> Dict[str, Any]:
    from tabu_search import TabuSearch

    instance_dir = f"Instance{instance_size}"
    instance, evaluator, _ = load_instance_for_tuning(
        instance_name, seed=seed, instance_dir=instance_dir,
    )
    start = time.time()
    try:
        initial, initial_metrics = build_shared_initial_solution(instance, evaluator)
        search_evaluator = _build_search_evaluator(instance, evaluator)
        ts = TabuSearch(
            evaluator=evaluator,
            tabu_tenure=params["tabu_tenure"],
            max_iterations=params["max_iterations"],
            max_stagnation=params.get("max_stagnation"),
            rng=random.Random(seed),
            search_evaluator=search_evaluator,
        )
        best = ts.run(initial, time_limit=time_limit_override)
        runtime = time.time() - start
        res = evaluator.evaluate_solution(best)
        result = {
            "cost": res.total_cost, "feasible": res.feasible,
            "delay_cost": res.delay_penalty,
            "truck_cost": res.truck_distance_cost,
            "drone_cost": res.drone_distance_cost,
            "runtime": runtime,
            **_search_stats(search_evaluator),
            **initial_metrics,
        }
        ts_stats = getattr(ts, "stats", None)
        if ts_stats is not None:
            result["iterations_completed"] = len(ts_stats.get("iterations", []))
            result["ts_stats"] = ts_stats
        return result
    except Exception as e:
        runtime = time.time() - start
        return {"cost": INFEASIBLE_PENALTY, "feasible": False,
                "runtime": runtime, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
#  GA
# ═══════════════════════════════════════════════════════════════════════════

def run_ga(
    instance_name: str,
    instance_size: int,
    params: Dict[str, Any],
    seed: int,
    time_limit_override: float = 300.0,
) -> Dict[str, Any]:
    from heuristics.ga.ga import GeneticAlgorithm, GAConfig

    instance_dir = f"Instance{instance_size}"
    instance, evaluator, _ = load_instance_for_tuning(
        instance_name, seed=seed, instance_dir=instance_dir,
    )
    start = time.time()
    try:
        initial, initial_metrics = build_shared_initial_solution(instance, evaluator)
        search_evaluator = _build_search_evaluator(instance, evaluator)
        ga_config = GAConfig(
            population_size=params["population_size"],
            generations=params["generations"],
            tournament_size=params["tournament_size"],
            crossover_rate=params["crossover_rate"],
            mutation_rate=params["mutation_rate"],
            elite_size=params["elite_size"],
            max_stagnation=params["max_stagnation"],
            truck_route_crossover_rate=params["truck_route_crossover_rate"],
            drone_task_mutation_rate=params["drone_task_mutation_rate"],
            route_segment_swap_rate=params["route_segment_swap_rate"],
            time_limit=time_limit_override,
            strict_time_budget=True,
        )
        ga_kwargs = {"rng": random.Random(seed)}
        if search_evaluator is not None:
            ga_kwargs["search_evaluator"] = search_evaluator
        ga = GeneticAlgorithm(instance, ga_config, evaluator, **ga_kwargs)
        best_individual = ga.run(initial)
        runtime = time.time() - start
        res = evaluator.evaluate_solution(best_individual.solution)
        result = {
            "cost": res.total_cost, "feasible": res.feasible,
            "truck_cost": res.truck_distance_cost,
            "drone_cost": res.drone_distance_cost,
            "delay_cost": res.delay_penalty,
            "runtime": runtime,
            **_search_stats(search_evaluator),
            **initial_metrics,
        }
        ga_stats = getattr(ga, "stats", None)
        if ga_stats is not None:
            result["generations_completed"] = len(ga_stats.get("generations", []))
            result["ga_stats"] = ga_stats
        return result
    except Exception as e:
        runtime = time.time() - start
        return {"cost": INFEASIBLE_PENALTY, "feasible": False,
                "runtime": runtime, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
#  Checkpoint / Orchestration
# ═══════════════════════════════════════════════════════════════════════════

def _tag_suffix(run_tag: str = "") -> str:
    return f"_{run_tag}" if run_tag else ""


def _ckpt_path(algo: str, instance_size: int, run_tag: str = "") -> Path:
    return OUT_DIR / f"inst{instance_size}_{algo}{_tag_suffix(run_tag)}_checkpoint.json"


def _save_checkpoint(algo: str, instance_size: int,
                     results: List[Dict[str, Any]], run_tag: str = "") -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "algo": algo,
        "instance_size": instance_size,
        "run_tag": run_tag,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
        "summary": compute_summary(results),
    }
    with open(_ckpt_path(algo, instance_size, run_tag), "w") as f:
        json.dump(ckpt, f, indent=2, default=str)


def _load_checkpoint(algo: str, instance_size: int,
                     run_tag: str = "") -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    path = _ckpt_path(algo, instance_size, run_tag)
    if not path.exists():
        return [], {}
    ckpt = json.loads(path.read_text())
    results = ckpt["results"]
    by_instance = {r["instance"]: r for r in results if "instance" in r}
    logger.info(
        f"  [{algo.upper()}] checkpoint: {len(results)} instances present, resuming"
    )
    return results, by_instance


def _completed_seed_set(result: Dict[str, Any] | None) -> set[int]:
    if not result:
        return set()
    if result.get("seed_results"):
        return {
            int(entry["seed"])
            for entry in result["seed_results"]
            if entry.get("feasible", False)
        }
    if result.get("feasible", False):
        return {int(seed) for seed in result.get("seeds", [])}
    return set()


def _seed_result_from_run(seed: int, result: Dict[str, Any], runtime: float) -> Dict[str, Any]:
    cost = result["cost"] if result.get("feasible", False) else INFEASIBLE_PENALTY
    seed_result = {
        "seed": seed,
        "cost": cost,
        "feasible": result.get("feasible", False),
        "runtime": result.get("runtime", runtime),
        "delay_cost": result.get("delay_cost", 0.0),
        "truck_cost": result.get("truck_cost", 0.0),
        "drone_cost": result.get("drone_cost", 0.0),
        "error": result.get("error"),
    }
    for key, value in result.items():
        if key.startswith("initial_"):
            seed_result[key] = value
    if "iterations_completed" in result:
        seed_result["iterations_completed"] = result["iterations_completed"]
    return seed_result


def _merge_instance_result(
    instance: str,
    existing: Dict[str, Any] | None,
    requested_seeds: List[int],
    new_seed_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    seed_results_by_seed = {
        int(entry["seed"]): dict(entry)
        for entry in (existing.get("seed_results", []) if existing else [])
    }
    for entry in new_seed_results:
        seed_results_by_seed[int(entry["seed"])] = entry
    seen = set(seed_results_by_seed)
    seed_results = list(seed_results_by_seed.values())
    seed_results.sort(key=lambda entry: int(entry["seed"]))

    feasible = [entry for entry in seed_results if entry.get("feasible", False)]
    candidates = list(feasible)
    if existing and existing.get("feasible", False):
        candidates.append(existing)
    if candidates:
        best = dict(min(candidates, key=lambda entry: entry["cost"]))
    elif seed_results:
        best = dict(min(seed_results, key=lambda entry: entry["cost"]))
    else:
        best = dict(existing or {})

    best["instance"] = instance
    best["seeds"] = sorted(set(requested_seeds) | seen)
    best["seed_results"] = seed_results
    return best


def run_algorithm(
    algo: str,
    instance_size: int,
    instances: List[str],
    seeds: List[int],
    params: Dict[str, Any],
    run_fn,
    run_tag: str = "",
) -> List[Dict[str, Any]]:
    family = f"inst{instance_size}_{algo}"

    all_results, by_instance = _load_checkpoint(algo, instance_size, run_tag)
    result_order = [r["instance"] for r in all_results if "instance" in r]

    for inst in instances:
        existing = by_instance.get(inst)
        completed = _completed_seed_set(existing)
        missing_seeds = [seed for seed in seeds if seed not in completed]
        if not missing_seeds:
            logger.info(
                f"  [{algo.upper()}] {inst}: all {len(seeds)} requested seeds done, skipping"
            )
            continue

        seed_results = []
        for seed in missing_seeds:
            eid = record_run(algo, inst, seed, params, family,
                             status="started", command="validate_alns_ts_ga")
            t0 = time.time()
            try:
                result = run_fn(inst, instance_size, params, seed)
            except Exception as e:
                result = {
                    "cost": INFEASIBLE_PENALTY,
                    "feasible": False,
                    "runtime": time.time() - t0,
                    "error": str(e),
                }
            runtime = time.time() - t0

            cost = result["cost"] if result.get("feasible", False) else INFEASIBLE_PENALTY
            status = "success" if result.get("feasible", False) else "failed"
            update_run(eid, status=status, runtime_seconds=runtime,
                       notes=f"cost={cost:.2f}")

            seed_results.append(_seed_result_from_run(seed, result, runtime))

        best = _merge_instance_result(inst, existing, seeds, seed_results)
        by_instance[inst] = best
        if inst not in result_order:
            result_order.append(inst)
        all_results = [by_instance[name] for name in result_order if name in by_instance]
        _save_checkpoint(algo, instance_size, all_results, run_tag)

        logger.info(f"  [{algo.upper()}] {inst}: best={best['cost']:.2f} over {len(seeds)} seeds")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
#  Summary / Print / Output
# ═══════════════════════════════════════════════════════════════════════════

def compute_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    feasible = [r for r in results if r["feasible"]]
    costs = [r["cost"] for r in feasible]
    runtimes = [r["runtime"] for r in results]
    n_total = len(results)
    n_feasible = len(feasible)

    if n_feasible == 0:
        return {
            "n_runs": n_total, "n_feasible": 0,
            "mean_cost": float("inf"), "std_cost": 0.0,
            "min_cost": float("inf"), "max_cost": float("inf"),
            "mean_runtime": float(np.mean(runtimes)) if runtimes else 0.0,
            "per_instance_mean": {},
            "per_instance_std": {},
            "per_instance_min": {},
        }

    per_instance: Dict[str, List[float]] = {}
    for r in feasible:
        per_instance.setdefault(r["instance"], []).append(r["cost"])

    return {
        "n_runs": n_total, "n_feasible": n_feasible,
        "mean_cost": float(np.mean(costs)),
        "std_cost": float(np.std(costs)),
        "min_cost": float(np.min(costs)),
        "max_cost": float(np.max(costs)),
        "mean_runtime": float(np.mean(runtimes)),
        "per_instance_mean": {k: float(np.mean(v)) for k, v in per_instance.items()},
        "per_instance_std": {k: float(np.std(v)) for k, v in per_instance.items()},
        "per_instance_min": {k: float(np.min(v)) for k, v in per_instance.items()},
    }


def _iter_seed_records(results: List[Dict[str, Any]]):
    for result in results:
        seed_results = result.get("seed_results") or []
        if seed_results:
            for seed_result in seed_results:
                record = dict(seed_result)
                record["instance"] = result["instance"]
                yield record
        else:
            yield result


def compute_seed_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    records = list(_iter_seed_records(results))
    feasible = [r for r in records if r.get("feasible", False)]
    costs = [r["cost"] for r in feasible]
    runtimes = [r.get("runtime", 0.0) for r in records]
    n_total = len(records)
    n_feasible = len(feasible)

    if n_feasible == 0:
        return {
            "n_runs": n_total, "n_feasible": 0,
            "mean_cost": float("inf"), "std_cost": 0.0,
            "min_cost": float("inf"), "max_cost": float("inf"),
            "mean_runtime": float(np.mean(runtimes)) if runtimes else 0.0,
            "per_instance_mean": {},
            "per_instance_std": {},
            "per_instance_min": {},
        }

    per_instance: Dict[str, List[float]] = {}
    for r in feasible:
        per_instance.setdefault(r["instance"], []).append(r["cost"])

    return {
        "n_runs": n_total, "n_feasible": n_feasible,
        "mean_cost": float(np.mean(costs)),
        "std_cost": float(np.std(costs)),
        "min_cost": float(np.min(costs)),
        "max_cost": float(np.max(costs)),
        "mean_runtime": float(np.mean(runtimes)),
        "per_instance_mean": {k: float(np.mean(v)) for k, v in per_instance.items()},
        "per_instance_std": {k: float(np.std(v)) for k, v in per_instance.items()},
        "per_instance_min": {k: float(np.min(v)) for k, v in per_instance.items()},
    }


def build_final_output(instance_size: int, instances: List[str], seeds: List[int],
                       alns_res, ts_res, ga_res, total_time, run_tag: str = ""):
    return {
        "description": f"Instance{instance_size} benchmark: ALNS vs TS vs GA",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "instance_size": instance_size,
        "run_tag": run_tag,
        "instances": instances,
        "seeds": seeds,
        "alns_config_hash": config_hash(make_alns_config_dict()),
        "alns": {"config": make_alns_config_dict(), "results": alns_res},
        "ts": {"config": TS_FINAL_CFG, "results": ts_res},
        "ga": {"config": GA_FINAL_CFG, "results": ga_res},
        "summary": {
            "alns": compute_seed_summary(alns_res),
            "ts": compute_seed_summary(ts_res),
            "ga": compute_seed_summary(ga_res),
        },
        "best_of_seed_summary": {
            "alns": compute_summary(alns_res),
            "ts": compute_summary(ts_res),
            "ga": compute_summary(ga_res),
        },
        "total_runtime_seconds": total_time,
    }


def _checkpoint_name(algo: str, instance_size: int, run_tag: str = "") -> str:
    return _ckpt_path(algo, instance_size, run_tag).name


def _algo_row_metrics(
    result: Dict[str, Any] | None,
    *,
    source: str,
) -> Dict[str, Any]:
    seed_results = (result or {}).get("seed_results", []) if result else []
    if not seed_results and result:
        seed_results = [result]

    feasible = [entry for entry in seed_results if entry.get("feasible", False)]
    feasible_costs = [float(entry["cost"]) for entry in feasible]
    runtimes = [float(entry.get("runtime", 0.0)) for entry in seed_results]
    total = len(seed_results)

    return {
        "best": float(np.min(feasible_costs)) if feasible_costs else None,
        "mean": float(np.mean(feasible_costs)) if feasible_costs else None,
        "std": float(np.std(feasible_costs)) if feasible_costs else 0.0,
        "feasible": f"{len(feasible)}/{total}",
        "runtime_mean": float(np.mean(runtimes)) if runtimes else 0.0,
        "source": source,
    }


def _comparison_row(
    instance: str,
    instance_size: int,
    alns_res: List[Dict[str, Any]],
    ts_res: List[Dict[str, Any]],
    ga_res: List[Dict[str, Any]],
    run_tag: str = "",
) -> Dict[str, Any]:
    by_algo = {
        "alns": {r["instance"]: r for r in alns_res},
        "ts": {r["instance"]: r for r in ts_res},
        "ga": {r["instance"]: r for r in ga_res},
    }
    row = {"instance": instance}
    for algo in ("alns", "ts", "ga"):
        metrics = _algo_row_metrics(
            by_algo[algo].get(instance),
            source=_checkpoint_name(algo, instance_size, run_tag),
        )
        for key, value in metrics.items():
            row[f"{algo}_{key}"] = value
    return row


def _instance_group(instance: str) -> str:
    parts = instance.split("_")
    if len(parts) >= 2 and parts[0] == "R":
        return f"R{parts[1]}"
    return "OTHER"


def _mean_ignore_missing(values: List[Any]) -> float | None:
    numeric = [float(v) for v in values if v is not None]
    return float(np.mean(numeric)) if numeric else None


def _aggregate_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    groups = sorted({_instance_group(row["instance"]) for row in rows})
    groups.append("ALL")
    for group in groups:
        group_rows = rows if group == "ALL" else [
            row for row in rows if _instance_group(row["instance"]) == group
        ]
        summary[group] = {}
        for algo in ("alns", "ts", "ga"):
            summary[group][algo] = {
                "instances": len(group_rows),
                "mean_of_instance_means": _mean_ignore_missing(
                    [row.get(f"{algo}_mean") for row in group_rows]
                ),
                "mean_of_instance_bests": _mean_ignore_missing(
                    [row.get(f"{algo}_best") for row in group_rows]
                ),
            }
    return summary


def _write_aggregate_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "instance",
        "alns_best", "alns_mean", "alns_std", "alns_feasible",
        "alns_runtime_mean", "alns_source",
        "ts_best", "ts_mean", "ts_std", "ts_feasible",
        "ts_runtime_mean", "ts_source",
        "ga_best", "ga_mean", "ga_std", "ga_feasible",
        "ga_runtime_mean", "ga_source",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def update_aggregate_output(
    aggregate_path: Path,
    output: Dict[str, Any],
    alns_res: List[Dict[str, Any]],
    ts_res: List[Dict[str, Any]],
    ga_res: List[Dict[str, Any]],
    run_tag: str = "",
    selected_algorithms: List[str] | None = None,
) -> Path:
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    if aggregate_path.exists() and aggregate_path.stat().st_size:
        existing = json.loads(aggregate_path.read_text())
        rows_by_instance = {
            row["instance"]: dict(row)
            for row in existing.get("rows", [])
            if "instance" in row
        }
        description = existing.get("description", "Incremental comparison")
    else:
        rows_by_instance = {}
        description = "Incremental comparison"
    if "Instance50" in description or int(output["instance_size"]) == 50:
        description = (
            "Incremental Instance50 comparison. Rows are upserted by instance; "
            "summary is recomputed after each update."
        )

    instance_size = int(output["instance_size"])
    selected = set(selected_algorithms or ["alns", "ts", "ga"])
    for instance in output["instances"]:
        new_row = _comparison_row(
            instance,
            instance_size,
            alns_res,
            ts_res,
            ga_res,
            run_tag,
        )
        if instance in rows_by_instance and selected != {"alns", "ts", "ga"}:
            merged = dict(rows_by_instance[instance])
            merged["instance"] = instance
            for algo in selected:
                for key, value in new_row.items():
                    if key.startswith(f"{algo}_"):
                        merged[key] = value
            rows_by_instance[instance] = merged
        else:
            rows_by_instance[instance] = new_row

    rows = [rows_by_instance[name] for name in sorted(rows_by_instance)]
    aggregate = {
        "description": description,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rows": rows,
        "summary": _aggregate_summary(rows),
        "notes": {
            "update_mode": "upsert_by_instance",
            "last_run_tag": run_tag,
            "last_instances": output["instances"],
        },
    }
    aggregate_path.write_text(json.dumps(aggregate, indent=2, default=str))
    csv_path = aggregate_path.with_suffix(".csv")
    _write_aggregate_csv(csv_path, rows)
    return csv_path


def print_table(instance_size: int, instances: List[str], alns, ts, ga):
    as_ = compute_seed_summary(alns)
    ts_ = compute_seed_summary(ts)
    ga_ = compute_seed_summary(ga)
    print()
    print("=" * 70)
    print(f"  Instance{instance_size} Comparison: ALNS vs TS vs GA (seed-run mean)")
    print("=" * 70)
    print(f"  {'Inst':>12} | {'ALNS mean':>10} | {'TS mean':>10} | {'GA mean':>10}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for inst in instances:
        a = as_["per_instance_mean"].get(inst, float("inf"))
        t = ts_["per_instance_mean"].get(inst, float("inf"))
        g = ga_["per_instance_mean"].get(inst, float("inf"))
        a_str = f"{a:.2f}" if a != float("inf") else "  inf  "
        t_str = f"{t:.2f}" if t != float("inf") else "  inf  "
        g_str = f"{g:.2f}" if g != float("inf") else "  inf  "
        print(f"  {inst:>12} | {a_str:>10} | {t_str:>10} | {g_str:>10}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    print(f"  {'Overall':>12} | "
          f"{as_['mean_cost']:>8.2f}  | "
          f"{ts_['mean_cost']:>8.2f}  | "
          f"{ga_['mean_cost']:>8.2f}")
    print(f"\n  Feasible runs:  ALNS {as_['n_feasible']}/{as_['n_runs']}, "
          f"TS {ts_['n_feasible']}/{ts_['n_runs']}, "
          f"GA {ga_['n_feasible']}/{ga_['n_runs']}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ALNS vs TS vs GA comparison")
    parser.add_argument("--instance-size", type=int, default=25,
                        help="Instance size directory (e.g. 25 for Instance25)")
    parser.add_argument("--instances", type=int, default=None,
                        help="Number of instances to use (default: all)")
    parser.add_argument("--instance-prefix", default=None,
                        help="Only run instances whose names start with this prefix")
    parser.add_argument("--instance-names", default=None,
                        help="Comma-separated exact instance names, e.g. R_50_50_1,R_50_50_2")
    parser.add_argument("--instance-classes", default=None,
                        help="Comma-separated classes such as R30,R40,R50 or 30,40,50")
    parser.add_argument("--algorithms", "--algorithm", default="all",
                        help="Algorithms to run: all, alns, ts, ga, or comma-separated e.g. ts,ga")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Seeds: '101-105', '10', or comma-separated (default: 101-105)")
    parser.add_argument("--seed-start", type=int, default=101,
                        help="First seed (default: 101)")
    parser.add_argument("--seed-count", type=int, default=3,
                        help="Number of seeds (default: 3)")
    parser.add_argument("--alns-iterations", type=int, default=4000,
                        help="ALNS iterations per seed (default: 4000)")
    parser.add_argument("--alns-time-limit", type=float, default=None,
                        help="Optional ALNS wall-clock time limit per seed in seconds")
    parser.add_argument("--heuristic-time-limit", type=float, default=300.0,
                        help="TS/GA time limit in seconds per seed (default: 300)")
    parser.add_argument("--run-tag", default="",
                        help="Optional suffix for checkpoint/output files, e.g. smoke")
    parser.add_argument("--aggregate-output", type=Path, default=None,
                        help="Optional aggregate JSON to upsert with this run's rows")
    args = parser.parse_args()

    all_instances = discover_instances(
        args.instance_size,
        max_instances=None,
        instance_prefix=None,
    )
    instances = select_instances(
        all_instances,
        instance_size=args.instance_size,
        max_instances=args.instances,
        instance_prefix=args.instance_prefix,
        instance_names=args.instance_names,
        instance_classes=args.instance_classes,
    )
    if not instances:
        raise SystemExit("No instances selected")
    algorithms = parse_algorithms(args.algorithms)

    if args.seeds:
        seeds = parse_seed_range(args.seeds)
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.seed_count))

    alns_params = make_alns_config_dict()
    alns_params["iterations"] = args.alns_iterations

    def run_alns_fn(inst, sz, params, seed):
        return run_alns(
            inst, sz, seed,
            iterations=alns_params["iterations"],
            time_limit_override=args.alns_time_limit,
        )

    def run_ts_fn(inst, sz, params, seed):
        return run_ts(
            inst, sz, params, seed,
            time_limit_override=args.heuristic_time_limit,
        )

    def run_ga_fn(inst, sz, params, seed):
        return run_ga(
            inst, sz, params, seed,
            time_limit_override=args.heuristic_time_limit,
        )

    start_global = time.time()

    alns_results: List[Dict[str, Any]] = []
    ts_results: List[Dict[str, Any]] = []
    ga_results: List[Dict[str, Any]] = []

    if "alns" in algorithms:
        alns_results = run_algorithm("alns", args.instance_size, instances, seeds,
                                     alns_params, run_alns_fn, run_tag=args.run_tag)
    if "ts" in algorithms:
        ts_results = run_algorithm("ts", args.instance_size, instances, seeds,
                                   TS_FINAL_CFG, run_ts_fn, run_tag=args.run_tag)
    if "ga" in algorithms:
        ga_results = run_algorithm("ga", args.instance_size, instances, seeds,
                                   GA_FINAL_CFG, run_ga_fn, run_tag=args.run_tag)

    total_time = time.time() - start_global
    output = build_final_output(args.instance_size, instances, seeds,
                                alns_results, ts_results, ga_results, total_time,
                                run_tag=args.run_tag)

    algo_suffix = "" if algorithms == ["alns", "ts", "ga"] else "_" + "_".join(algorithms)
    out_path = OUT_DIR / f"inst{args.instance_size}_comparison{_tag_suffix(args.run_tag)}{algo_suffix}.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print_table(args.instance_size, instances, alns_results, ts_results, ga_results)
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Results saved to {out_path}")
    if args.aggregate_output is not None:
        csv_path = update_aggregate_output(
            args.aggregate_output,
            output,
            alns_results,
            ts_results,
            ga_results,
            run_tag=args.run_tag,
            selected_algorithms=algorithms,
        )
        print(f"  Aggregate updated: {args.aggregate_output}")
        print(f"  Aggregate CSV updated: {csv_path}")
    print()


if __name__ == "__main__":
    main()
