#!/usr/bin/env python3
"""Unified Bayesian tuning for TS and GA using Optuna TPE.

Usage:
    cd code/
    python revision/tune_ts_ga.py --algo ts [--phase 1|2|3|all] [--trials N]
    python revision/tune_ts_ga.py --algo ga [--phase 1|2|3|all] [--trials N]

Phase 1: Coarse search (few instances, few seeds, short budget)
Phase 2: Fine search (more instances, more seeds, medium budget)
Phase 3: Validation (all instances, 10 seeds, full budget)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import optuna
from optuna.samplers import TPESampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "heuristics" / "tabu_search"))
sys.path.insert(0, str(PROJECT_ROOT / "heuristics" / "ga"))

from revision.tune_base import (
    ClassWeightedEvaluator, generate_class_based_deadlines, load_instance_for_tuning,
    INFEASIBLE_PENALTY, stability_objective, save_tuning_result,
    TRAINING_INSTANCES, ALL_INSTANCE10,
    TRAINING_SEEDS_PHASE1, TRAINING_SEEDS_PHASE2, VALIDATION_SEEDS,
    REVISION_ROOT, record_run, update_run,
)
from revision.manifest import config_hash
from alns_vrpfd.evaluation import SearchEvaluator, SubrouteRobustVerifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _build_search_evaluator(instance, evaluator) -> SearchEvaluator | None:
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

# ─── TS parameter search space ──────────────────────────────────────────

def suggest_ts_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "tabu_tenure": trial.suggest_int("tabu_tenure", 5, 40),
        "max_iterations": trial.suggest_int("max_iterations", 500, 3000),
        "max_stagnation": trial.suggest_int("max_stagnation", 30, 200),
        "time_limit_seconds": trial.suggest_float("time_limit_seconds", 60, 600),
        "neighborhood_sample_ratio": trial.suggest_float("neighborhood_sample_ratio", 0.3, 1.0),
    }


# ─── GA parameter search space ──────────────────────────────────────────

def suggest_ga_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "population_size": trial.suggest_int("population_size", 30, 200),
        "generations": trial.suggest_int("generations", 30, 300),
        "tournament_size": trial.suggest_int("tournament_size", 2, 10),
        "crossover_rate": trial.suggest_float("crossover_rate", 0.5, 0.95),
        "mutation_rate": trial.suggest_float("mutation_rate", 0.01, 0.3),
        "elite_size": trial.suggest_int("elite_size", 1, 20),
        "max_stagnation": trial.suggest_int("max_stagnation", 10, 50),
        "truck_route_crossover_rate": trial.suggest_float("truck_route_crossover_rate", 0.5, 0.9),
        "drone_task_mutation_rate": trial.suggest_float("drone_task_mutation_rate", 0.1, 0.5),
        "route_segment_swap_rate": trial.suggest_float("route_segment_swap_rate", 0.1, 0.5),
        "time_limit_seconds": trial.suggest_float("time_limit_seconds", 60, 600),
    }


# ─── Run wrappers ────────────────────────────────────────────────────────

def run_ts_single(instance_name: str, params: Dict[str, Any], seed: int,
                  time_limit_override: Optional[float] = None) -> Dict[str, Any]:
    """Run TS on a single instance with given params."""
    from tabu_search import TabuSearch
    instance, evaluator, classes = load_instance_for_tuning(instance_name, seed=seed)
    from alns_vrpfd.model.initializer import build_initial_solution
    initial = build_initial_solution(instance)

    tl = time_limit_override or params.get("time_limit_seconds", 300)
    search_evaluator = _build_search_evaluator(instance, evaluator)

    ts = TabuSearch(
        evaluator=evaluator,
        tabu_tenure=params["tabu_tenure"],
        max_iterations=params["max_iterations"],
        max_stagnation=params.get("max_stagnation"),
        rng=random.Random(seed),
        search_evaluator=search_evaluator,
    )

    start = time.time()
    try:
        best = ts.run(initial, time_limit=tl)
        runtime = time.time() - start
        res = evaluator.evaluate_solution(best)
        return {
            "cost": res.total_cost, "feasible": res.feasible,
            "delay_cost": res.delay_penalty, "truck_cost": res.truck_distance_cost,
            "drone_cost": res.drone_distance_cost, "runtime": runtime,
            **_search_stats(search_evaluator),
        }
    except Exception as e:
        runtime = time.time() - start
        return {"cost": INFEASIBLE_PENALTY, "feasible": False, "runtime": runtime, "error": str(e)}


def run_ga_single(instance_name: str, params: Dict[str, Any], seed: int,
                  time_limit_override: Optional[float] = None) -> Dict[str, Any]:
    """Run GA on a single instance with given params."""
    from heuristics.ga.ga import GeneticAlgorithm, GAConfig
    instance, evaluator, classes = load_instance_for_tuning(instance_name, seed=seed)
    from alns_vrpfd.model.initializer import build_two_phase_initial_solution
    initial = build_two_phase_initial_solution(instance)

    tl = time_limit_override or params.get("time_limit_seconds", 300)
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
        time_limit=tl,
    )

    ga = GeneticAlgorithm(
        instance,
        ga_config,
        evaluator,
        rng=random.Random(seed),
        search_evaluator=search_evaluator,
    )

    start = time.time()
    try:
        best_individual = ga.run(initial)
        runtime = time.time() - start
        return {
            "cost": best_individual.fitness, "feasible": best_individual.feasible,
            "truck_cost": best_individual.truck_distance,
            "drone_cost": best_individual.drone_distance,
            "delay_cost": best_individual.delay_penalty,
            "runtime": runtime,
            **_search_stats(search_evaluator),
        }
    except Exception as e:
        runtime = time.time() - start
        return {"cost": INFEASIBLE_PENALTY, "feasible": False, "runtime": runtime, "error": str(e)}


# ─── Objective ────────────────────────────────────────────────────────

def create_objective(algo: str, instances: List[str], n_seeds: int,
                     time_limit: float, iterations_budget: int,
                     lambda_std: float = 1.0):
    run_fn = run_ts_single if algo == "ts" else run_ga_single

    def objective(trial: optuna.Trial) -> float:
        if algo == "ts":
            params = suggest_ts_params(trial)
            params["max_iterations"] = iterations_budget
        else:
            params = suggest_ga_params(trial)
            params["generations"] = iterations_budget

        scores = []
        grand_costs = []
        for inst in instances:
            costs = []
            for s_off in range(n_seeds):
                seed = 200 + s_off + trial.number * 11
                result = run_fn(inst, params, seed, time_limit_override=time_limit)
                cost = result["cost"] if result.get("feasible", False) else INFEASIBLE_PENALTY
                costs.append(cost)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            mean_c = np.mean(costs)
            std_c = np.std(costs)
            scores.append(mean_c + lambda_std * std_c)
            grand_costs.extend(costs)

        overall = float(np.mean(scores))
        trial.set_user_attr("mean_cost", float(np.mean(grand_costs)))
        trial.set_user_attr("std_cost", float(np.std(grand_costs)))
        return overall

    return objective


# ─── Phase runners ────────────────────────────────────────────────────

def run_phase1(algo: str, n_trials: int = 20, lambda_std: float = 1.0):
    label = algo.upper()
    print("=" * 80)
    print(f"  PHASE 1 ({label}): Coarse Search")
    print(f"  Instances: 2, Seeds: 2, Time: 120s, λ_std={lambda_std}")
    print("=" * 80)

    instances = TRAINING_INSTANCES[:2]
    sampler = TPESampler(seed=42, n_startup_trials=5)
    study = optuna.create_study(direction="minimize", sampler=sampler,
                                study_name=f"{algo}_phase1")
    objective = create_objective(algo, instances, n_seeds=2,
                                 time_limit=120, iterations_budget=1000 if algo == "ts" else 80,
                                 lambda_std=lambda_std)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\n  Phase 1 Best: score={study.best_value:.2f}")
    print(f"  Best params: {json.dumps(study.best_params, indent=2)}")

    save_tuning_result(algo, "phase1", study.best_params, study.best_value,
                       [{"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
                        for t in study.trials])
    return study


def run_phase2(algo: str, phase1_study: optuna.Study,
               n_trials: int = 15, lambda_std: float = 1.0):
    label = algo.upper()
    print("=" * 80)
    print(f"  PHASE 2 ({label}): Fine Search")
    print(f"  Instances: 3, Seeds: 3, Time: 180s, λ_std={lambda_std}")
    print("=" * 80)

    sampler = TPESampler(seed=123, n_startup_trials=5)
    study = optuna.create_study(direction="minimize", sampler=sampler,
                                study_name=f"{algo}_phase2")
    warm_start = None
    try:
        warm_start = phase1_study.best_params
    except ValueError:
        for t in phase1_study.get_trials(deepcopy=False):
            if t.state == optuna.trial.TrialState.COMPLETE:
                warm_start = t.params
                break
    if warm_start:
        study.enqueue_trial(warm_start)

    objective = create_objective(algo, TRAINING_INSTANCES, n_seeds=3,
                                 time_limit=180, iterations_budget=2000 if algo == "ts" else 150,
                                 lambda_std=lambda_std)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\n  Phase 2 Best: score={study.best_value:.2f}")
    print(f"  Best params: {json.dumps(study.best_params, indent=2)}")

    save_tuning_result(algo, "phase2", study.best_params, study.best_value,
                       [{"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
                        for t in study.trials])
    return study


def run_validation(algo: str, best_params: Dict[str, Any],
                  n_seeds: int = 10, lambda_std: float = 1.5):
    label = algo.upper()
    iterations = 2000 if algo == "ts" else 150
    time_limit = 300.0

    print("=" * 80)
    print(f"  PHASE 3 ({label}): Validation")
    print(f"  All 5 instances, {n_seeds} seeds, {iterations} iters, {time_limit}s limit")
    print("=" * 80)

    run_fn = run_ts_single if algo == "ts" else run_ga_single
    all_results = []

    for inst in ALL_INSTANCE10:
        for seed in VALIDATION_SEEDS[:n_seeds]:
            t0 = time.time()
            result = run_fn(inst, best_params, seed, time_limit_override=time_limit)
            runtime = time.time() - t0
            cost = result["cost"] if result.get("feasible", False) else INFEASIBLE_PENALTY
            all_results.append({
                "instance": inst, "seed": seed, "cost": cost,
                "feasible": result.get("feasible", False),
                "runtime": result.get("runtime", runtime),
            })
            print(f"  {inst} seed={seed}: cost={cost:.2f} feasible={result.get('feasible', False)}")

    costs = [r["cost"] for r in all_results if r["feasible"]]
    mean_cost = np.mean(costs) if costs else float("inf")
    std_cost = np.std(costs) if costs else 0
    print(f"\n  Validation: mean={mean_cost:.2f}, std={std_cost:.2f}, n_feasible={len(costs)}/50")

    save_tuning_result(algo, "validation", best_params, mean_cost, all_results)
    return all_results


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bayesian tuning for TS/GA")
    parser.add_argument("--algo", choices=["ts", "ga"], required=True)
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--all", action="store_true", help="Run all phases sequentially")
    args = parser.parse_args()

    if args.all or args.phase is None:
        p1_trials = args.trials or (20 if args.algo == "ts" else 20)
        p2_trials = args.trials or (15 if args.algo == "ts" else 15)
        phase1_study = run_phase1(args.algo, n_trials=p1_trials)
        phase2_study = run_phase2(args.algo, phase1_study, n_trials=p2_trials)
        run_validation(args.algo, phase2_study.best_params)
    elif args.phase == 1:
        n = args.trials or 20
        study = run_phase1(args.algo, n_trials=n)
    elif args.phase == 2:
        n = args.trials or 15
        logger.info("Phase 2 requires Phase 1 results. Running Phase 1 first with fewer trials.")
        phase1_study = run_phase1(args.algo, n_trials=min(n, 10))
        run_phase2(args.algo, phase1_study, n_trials=n)
    elif args.phase == 3:
        logger.info("Phase 3 (validation) requires best params from Phase 2.")
        logger.info("Please provide best params manually or run --all.")


if __name__ == "__main__":
    main()
