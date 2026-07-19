"""
Bayesian Optimization (Optuna TPE) for ALNS hyperparameter tuning.

Goal: Find the optimal ALNS configuration that stably converges to the best solution.
Uses Optuna's Tree-structured Parzen Estimator for efficient black-box optimization.

Search strategy:
  - Phase 1 (Coarse): 60 trials on 2 representative instances, 1000 iters each
  - Phase 2 (Fine): 30 trials on 3 instances, 2000 iters each with best region refinement
  - Phase 3 (Validation): Run best config on all 5 instances with 4000 iters

Objective: minimize mean cost across instances (infeasible = penalty)

Usage:
    cd code/
    python scripts/bayesian_tune_alns.py [--phase {1,2,3,all}] [--trials N] [--jobs N]
"""

import sys
import os
import json
import time
import random
import datetime
import logging
import argparse
from pathlib import Path
from copy import deepcopy
from typing import Dict, Any, List, Optional

import numpy as np
import optuna
from optuna.samplers import TPESampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from alns_vrpfd.utils.io_utils import read_instance
from alns_vrpfd.deprivation import DEFAULT_SUPPLY_CLASS_SEQUENCE, WANG_SUPPLY_CLASSES
from alns_vrpfd.evaluation.evaluator import Evaluator
from alns_vrpfd.model.initializer import build_two_phase_initial_solution
from alns_vrpfd.core.operators import (
    DestroyRandom, DestroyShaw, DestroyWorstDistance,
    RepairCheapest, RepairDronePriorityRegret,
    RepairEqualPriority, RepairTruckFirst,
    RepairBiasedRandomized, RepairRegret,
)
from alns_vrpfd.core.sa import SANNCfg, SimulatedAnnealingALNS
from alns_vrpfd.utils.config_loader import ALNSConfig

ResultT = Dict[str, Any]

PHASE1_INSTANCES = ["R_30_10_1", "R_30_10_3"]
PHASE2_INSTANCES = ["R_30_10_1", "R_30_10_3", "R_30_10_5"]
ALL_INSTANCES = ["R_30_10_1", "R_30_10_2", "R_30_10_3", "R_30_10_4", "R_30_10_5"]
INFEASIBLE_PENALTY = 1e6


def generate_class_based_deadlines(instance, rng):
    customer_ids = instance.customer_manager.customer_ids()
    truck_speed = instance.vehicle_specs["truck"].speed
    drone_speed = instance.vehicle_specs["drone"].speed
    depot_id = instance.customer_manager.depot_start
    node_list = instance.all_node_ids()
    idx_map = {nid: i for i, nid in enumerate(node_list)}
    depot_idx = idx_map[depot_id]
    dist_truck = instance.distance_matrix("truck")
    dist_drone = instance.distance_matrix("drone")
    classes = {}
    supply_classes = list(DEFAULT_SUPPLY_CLASS_SEQUENCE)
    for offset, cid in enumerate(customer_ids):
        c = supply_classes[offset % len(supply_classes)]
        if offset >= len(supply_classes):
            c = str(rng.choice(supply_classes))
        classes[cid] = c
        params = WANG_SUPPLY_CLASSES[c]
        ci = idx_map[cid]
        r_i = min(dist_truck[depot_idx][ci] / truck_speed,
                   dist_drone[depot_idx][ci] / drone_speed)
        delta_o = float(rng.uniform(*params.deadline_optimal_delta_hours))
        delta_l = float(rng.uniform(*params.deadline_latest_delta_hours))
        instance.customer_manager.assign_supply_class(cid, c)
        instance.customer_manager.assign_time_window(
            cid, r_i + delta_o, r_i + delta_o + delta_l)
    return classes


def suggest_params(trial: optuna.Trial, phase: int = 1) -> Dict[str, Any]:
    """Define search space for ALNS hyperparameters."""
    params = {}

    # === SA Temperature & Cooling ===
    params["w_percent"] = trial.suggest_float("w_percent", 5.0, 40.0)
    params["cooling_rate_initial"] = trial.suggest_float("cooling_rate_initial", 0.990, 0.9999)
    params["cooling_rate_final"] = trial.suggest_float("cooling_rate_final", 0.97, 0.995)
    params["cooling_transition_iters"] = trial.suggest_int("cooling_transition_iters", 200, 1800)

    # === Adaptive Operator Selection ===
    params["eta"] = trial.suggest_float("eta", 0.05, 0.8)
    params["alpha_credit"] = trial.suggest_float("alpha_credit", 0.3, 0.9)
    sigma1 = trial.suggest_float("reward_global", 10.0, 60.0)
    sigma2 = trial.suggest_float("reward_better", 5.0, 30.0)
    sigma3 = trial.suggest_float("reward_slight_better", 3.0, 20.0)
    sigma4 = trial.suggest_float("reward_accepted_worse", 0.1, 5.0)
    params["reward_scale"] = {
        "global": sigma1, "better": sigma2,
        "slight_better": sigma3, "accepted_worse": sigma4,
    }
    params["weight_decay"] = trial.suggest_float("weight_decay", 0.001, 0.1)

    # === Destroy Quota ===
    params["r_lower"] = trial.suggest_float("r_lower", 0.05, 0.30)
    params["r_upper_small"] = trial.suggest_float("r_upper_small", 0.25, 0.60)
    params["quota_base_cap"] = trial.suggest_int("quota_base_cap", 10, 50)

    # === Reheat ===
    params["reheat_stall_trigger"] = trial.suggest_int("reheat_stall_trigger", 100, 800)
    params["reheat_acceptance_window"] = trial.suggest_int("reheat_acceptance_window", 30, 200)
    params["reheat_acceptance_min"] = trial.suggest_float("reheat_acceptance_min", 0.01, 0.20)
    params["reheat_duration"] = trial.suggest_int("reheat_duration", 15, 100)
    params["reheat_temperature_scale"] = trial.suggest_float("reheat_temperature_scale", 0.1, 0.8)
    params["reheat_quota_multiplier"] = trial.suggest_float("reheat_quota_multiplier", 1.5, 4.0)
    params["reheat_shake_probability"] = trial.suggest_float("reheat_shake_probability", 0.3, 0.95)
    params["reheat_random_repair_prob"] = trial.suggest_float("reheat_random_repair_prob", 0.2, 0.8)

    # === Local Search ===
    params["local_search_frequency"] = trial.suggest_int("local_search_frequency", 3, 25)
    params["local_search_on_new_best"] = trial.suggest_categorical("local_search_on_new_best", [True, False])
    params["depot_drone_probability"] = trial.suggest_float("depot_drone_probability", 0.05, 0.4)
    params["intensify_frequency"] = trial.suggest_int("intensify_frequency", 15, 80)

    # === Escape ===
    params["escape_enabled"] = trial.suggest_categorical("escape_enabled", [True, False])
    if params["escape_enabled"]:
        params["escape_trigger_stall"] = trial.suggest_int("escape_trigger_stall", 40, 200)
        params["escape_duration"] = trial.suggest_int("escape_duration", 10, 50)

    # === Convergence Enhancement ===
    params["dynamic_cooling_enabled"] = trial.suggest_categorical("dynamic_cooling_enabled", [True, False])
    if params["dynamic_cooling_enabled"]:
        params["cooling_slowdown_factor"] = trial.suggest_float("cooling_slowdown_factor", 0.990, 0.9999)
        params["cooling_speedup_factor"] = trial.suggest_float("cooling_speedup_factor", 0.96, 0.995)
        params["recent_improvement_window"] = trial.suggest_int("recent_improvement_window", 20, 100)

    params["diversification_enabled"] = trial.suggest_categorical("diversification_enabled", [True, False])
    if params["diversification_enabled"]:
        params["diversification_trigger_stall"] = trial.suggest_int("diversification_trigger_stall", 200, 800)
        params["diversification_restart_best_prob"] = trial.suggest_float("diversification_restart_best_prob", 0.3, 0.9)
        params["diversification_destroy_ratio"] = trial.suggest_float("diversification_destroy_ratio", 0.3, 0.7)

    params["adaptive_quota_enabled"] = trial.suggest_categorical("adaptive_quota_enabled", [True, False])

    # === Drone Bonus ===
    params["drone_priority"] = trial.suggest_float("drone_priority", 1.0, 4.0)
    params["depot_bonus"] = trial.suggest_float("depot_bonus", 0.1, 2.0)
    params["multi_customer_bonus"] = trial.suggest_float("multi_customer_bonus", 1.0, 10.0)

    return params


def build_sa_config(params: Dict[str, Any], iterations: int = 2000) -> SANNCfg:
    """Build SANNCfg from Optuna params, keeping non-searched params at defaults."""
    config = ALNSConfig(str(PROJECT_ROOT / "config" / "alns_config.yaml"))
    base = config.build_sa_config_dict()
    base.update({
        "iterations": iterations,
        "size": "small",
        "log_operator_metrics": False,
        "w_percent": params["w_percent"],
        "cooling_rate_initial": params["cooling_rate_initial"],
        "cooling_rate_final": params["cooling_rate_final"],
        "cooling_transition_iters": params["cooling_transition_iters"],
        "eta": params["eta"],
        "alpha_credit": params["alpha_credit"],
        "reward_scale": params["reward_scale"],
        "weight_decay": params["weight_decay"],
        "r_lower": params["r_lower"],
        "r_upper_small": params["r_upper_small"],
        "quota_base_cap": params["quota_base_cap"],
        "reheat_stall_trigger": params["reheat_stall_trigger"],
        "reheat_acceptance_window": params["reheat_acceptance_window"],
        "reheat_acceptance_min": params["reheat_acceptance_min"],
        "reheat_duration": params["reheat_duration"],
        "reheat_temperature_scale": params["reheat_temperature_scale"],
        "reheat_quota_multiplier": params["reheat_quota_multiplier"],
        "reheat_shake_probability": params["reheat_shake_probability"],
        "reheat_random_repair_prob": params["reheat_random_repair_prob"],
        "local_search_frequency": params["local_search_frequency"],
        "local_search_on_new_best": params["local_search_on_new_best"],
        "depot_drone_probability": params["depot_drone_probability"],
        "intensify_frequency": params["intensify_frequency"],
        "escape_enabled": params["escape_enabled"],
        "dynamic_cooling_enabled": params["dynamic_cooling_enabled"],
        "diversification_enabled": params["diversification_enabled"],
        "adaptive_quota_enabled": params["adaptive_quota_enabled"],
    })
    if params.get("escape_enabled"):
        base["escape_trigger_stall"] = params["escape_trigger_stall"]
        base["escape_duration"] = params["escape_duration"]
    else:
        base["escape_trigger_stall"] = 100
        base["escape_duration"] = 20
    if params.get("dynamic_cooling_enabled"):
        base["improvement_threshold"] = 0.01
        base["cooling_slowdown_factor"] = params.get("cooling_slowdown_factor", 0.998)
        base["cooling_speedup_factor"] = params.get("cooling_speedup_factor", 0.980)
        base["recent_improvement_window"] = params.get("recent_improvement_window", 50)
    if params.get("diversification_enabled"):
        base["diversification_trigger_stall"] = params.get("diversification_trigger_stall", 500)
        base["diversification_restart_best_prob"] = params.get("diversification_restart_best_prob", 0.7)
        base["diversification_destroy_ratio"] = params.get("diversification_destroy_ratio", 0.6)
    return SANNCfg(**base)


def run_single(instance_name: str, params: Dict[str, Any],
               seed: int, iterations: int, use_class: bool = False) -> ResultT:
    """Run ALNS once with given params on one instance."""
    fpath = str(PROJECT_ROOT / "data" / "Instance10" / f"{instance_name}.txt")
    config = ALNSConfig(str(PROJECT_ROOT / "config" / "alns_config.yaml"))

    if use_class:
        rng_tw = np.random.default_rng(seed)
        instance = read_instance(fpath, strategy="class_based", apply_time_windows=False)
    else:
        instance = read_instance(fpath, strategy="class_based")

    instance.vehicle_specs["drone"].endurance = float("inf")
    instance.configure_robustness(
        drone_battery_capacity=config.drone_battery_capacity,
        energy_uncertainty_budget=config.energy_uncertainty_budget,
        energy_deviation_rate=config.energy_deviation_rate,

        same_truck_retrieval=config.same_truck_retrieval,
    )

    if use_class:
        classes = generate_class_based_deadlines(instance, rng_tw)
        evaluator = Evaluator(
            instance,
            rendezvous_tolerance=config.drone_rendezvous_tolerance,
            forced_drone_customers=config.forced_drone_customers,
            allow_multiple_launch_per_node=config.relax_allow_multiple_launch_per_node,
        )
    else:
        classes = {}
        evaluator = Evaluator(
            instance,
            rendezvous_tolerance=config.drone_rendezvous_tolerance,
            forced_drone_customers=config.forced_drone_customers,
            allow_multiple_launch_per_node=config.relax_allow_multiple_launch_per_node,
        )

    sa_cfg = build_sa_config(params, iterations=iterations)
    alns_rng = random.Random(seed)

    drone_bonus_kwargs = {
        "depot_bonus": params.get("depot_bonus", 0.6),
        "multi_customer_bonus": params.get("multi_customer_bonus", 5.5),
        "multi_customer_threshold": 2,
        "wait_max": 20.0,
    }

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
                       drone_priority=params.get("drone_priority", 2.2),
                       robust_energy_mode="embedded", **drone_bonus_kwargs),
        RepairDronePriorityRegret(instance, rng=random.Random(seed + 2002),
                                   drone_priority=params.get("drone_priority", 2.2),
                                   robust_energy_mode="embedded", **drone_bonus_kwargs),
        RepairTruckFirst(instance, rng=random.Random(seed + 2003),
                         drone_priority=params.get("drone_priority", 2.2),
                         robust_energy_mode="embedded", **drone_bonus_kwargs),
        RepairEqualPriority(instance, rng=random.Random(seed + 2001),
                            drone_priority=params.get("drone_priority", 2.2),
                            robust_energy_mode="embedded", **drone_bonus_kwargs),
    ]

    initial_solution = build_two_phase_initial_solution(instance)
    alns = SimulatedAnnealingALNS(
        instance=instance, destroy_ops=destroy_ops, repair_ops=repair_ops,
        evaluator=evaluator, cfg=sa_cfg, rng=alns_rng,
    )

    start = time.time()
    try:
        best_sol = alns.run(initial_solution)
    except Exception as e:
        return {"cost": INFEASIBLE_PENALTY, "feasible": False, "runtime": time.time() - start,
                "delay_cost": 0, "truck_cost": 0, "drone_cost": 0, "error": str(e)}

    runtime = time.time() - start
    eval_res = evaluator.evaluate_solution(best_sol)
    return {
        "cost": eval_res.total_cost,
        "feasible": eval_res.feasible,
        "delay_cost": eval_res.delay_penalty,
        "truck_cost": eval_res.truck_distance_cost,
        "drone_cost": eval_res.drone_distance_cost,
        "runtime": runtime,
    }


def create_objective(instances: List[str], seed_base: int,
                     iterations: int, use_class: bool = False):
    """Create Optuna objective for given instance set."""
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        costs = []
        for i, inst in enumerate(instances):
            seed = seed_base + i + trial.number * 7
            result = run_single(inst, params, seed, iterations, use_class)
            cost = result["cost"]
            if not result["feasible"]:
                cost = INFEASIBLE_PENALTY
            costs.append(cost)
            trial.report(cost, i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        mean_cost = np.mean(costs)
        std_cost = np.std(costs)
        trial.set_user_attr("mean_cost", float(mean_cost))
        trial.set_user_attr("std_cost", float(std_cost))
        trial.set_user_attr("feasible_count", sum(1 for c in costs if c < INFEASIBLE_PENALTY))
        return mean_cost
    return objective


def run_phase1(n_trials: int = 60, use_class: bool = False) -> optuna.Study:
    """Phase 1: Coarse search on 2 instances with 1000 iterations."""
    print("=" * 80)
    print("  PHASE 1: Coarse Bayesian Optimization")
    print("  Instances: 2, Iterations: 1000, Trials:", n_trials)
    print("=" * 80)

    sampler = TPESampler(seed=42, n_startup_trials=15)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=f"alns_phase1_{'class' if use_class else 'demand'}",
    )
    objective = create_objective(PHASE1_INSTANCES, seed_base=42, iterations=1000,
                                  use_class=use_class)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n  Phase 1 Best: cost={study.best_value:.2f}")
    print(f"  Best params keys: {len(study.best_params)}")
    return study


def run_phase2(phase1_study: optuna.Study, n_trials: int = 30, use_class: bool = False) -> optuna.Study:
    """Phase 2: Fine search using Phase 1 best as starting point."""
    print("=" * 80)
    print("  PHASE 2: Fine Bayesian Optimization")
    print("  Instances: 3, Iterations: 2000, Trials:", n_trials)
    print("=" * 80)

    sampler = TPESampler(seed=123, n_startup_trials=5)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=f"alns_phase2_{'class' if use_class else 'demand'}",
    )
    # Enqueue Phase 1 best as starting point
    study.enqueue_trial(phase1_study.best_params)

    objective = create_objective(PHASE2_INSTANCES, seed_base=100, iterations=2000,
                                  use_class=use_class)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n  Phase 2 Best: cost={study.best_value:.2f}")
    return study


def run_validation(best_params: Dict[str, Any], use_class: bool = False) -> List[ResultT]:
    """Phase 3: Validate best config on all instances with full iterations."""
    print("=" * 80)
    print("  PHASE 3: Validation with Best Config")
    print("  All 5 instances, 4000 iterations, 3 seeds each")
    print("=" * 80)
    print(f"  Best params: {json.dumps(best_params, indent=2, default=str)[:500]}...")

    results = []
    for inst in ALL_INSTANCES:
        for seed_off in range(3):
            seed = 200 + seed_off
            result = run_single(inst, best_params, seed=seed, iterations=4000,
                                use_class=use_class)
            result["instance"] = inst
            result["seed"] = seed
            results.append(result)
            status = "OK" if result["feasible"] else "INFEASIBLE"
            print(f"  {inst} seed={seed}: cost={result['cost']:.2f} [{status}] "
                  f"({result['runtime']:.1f}s)")

    feasible = [r for r in results if r["feasible"]]
    if feasible:
        mean = np.mean([r["cost"] for r in feasible])
        std = np.std([r["cost"] for r in feasible])
        print(f"\n  Validation: mean={mean:.2f}, std={std:.2f}, "
              f"feasible={len(feasible)}/{len(results)}")
    return results


def save_results(phase: str, study: optuna.Study, results_dir: Path):
    """Save study results."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    df = study.trials_dataframe()
    df.to_csv(results_dir / f"bayesian_{phase}_{timestamp}.csv", index=False)

    # Save best params
    best = study.best_params
    best["reward_scale"] = {
        "global": best.pop("reward_global"),
        "better": best.pop("reward_better"),
        "slight_better": best.pop("reward_slight_better"),
        "accepted_worse": best.pop("reward_accepted_worse"),
    }
    # Re-add subclass fields if present
    for key in list(best.keys()):
        if key.startswith("_"):
            best.pop(key)

    with open(results_dir / f"best_params_{phase}_{timestamp}.json", "w") as f:
        json.dump(best, f, indent=2, default=str)

    print(f"\n  Results saved to {results_dir}")
    print(f"  Best params → best_params_{phase}_{timestamp}.json")
    return best, timestamp


def config_from_best_params(best_params: Dict[str, Any]) -> str:
    """Generate a YAML config snippet from best params."""
    p = best_params
    lines = [
        "# === Auto-tuned ALNS Configuration ===",
        "# Generated by Bayesian Optimization (Optuna TPE)",
        "",
        "simulated_annealing:",
        f"  w_percent: {p.get('w_percent', 15.0):.4f}",
        f"  temperature_min: 1.0e-4",
        "  cooling:",
        f"    rate_initial: {p.get('cooling_rate_initial', 0.9995):.6f}",
        f"    rate_final: {p.get('cooling_rate_final', 0.985):.6f}",
        f"    rate_default: {(p.get('cooling_rate_initial', 0.9995) + p.get('cooling_rate_final', 0.985)) / 2:.6f}",
        f"    transition_iters: {p.get('cooling_transition_iters', 800)}",
        "",
        "adaptive_selection:",
        f"  eta: {p.get('eta', 0.35):.4f}",
        f"  alpha_credit: {p.get('alpha_credit', 0.60):.4f}",
        "  rewards:",
        f"    global: {p['reward_scale']['global']:.2f}" if isinstance(p.get('reward_scale'), dict) else f"    global: {p.get('reward_global', 33.0):.2f}",
        f"    better: {p['reward_scale']['better']:.2f}" if isinstance(p.get('reward_scale'), dict) else f"    better: {p.get('reward_better', 13.0):.2f}",
        f"    slight_better: {p['reward_scale']['slight_better']:.2f}" if isinstance(p.get('reward_scale'), dict) else f"    slight_better: {p.get('reward_slight_better', 9.0):.2f}",
        f"    accepted_worse: {p['reward_scale']['accepted_worse']:.2f}" if isinstance(p.get('reward_scale'), dict) else f"    accepted_worse: {p.get('reward_accepted_worse', 1.0):.2f}",
        f"  probability_floor: {p.get('probability_floor', 0.03):.4f}",
        f"  weight_decay: {p.get('weight_decay', 0.015):.4f}",
        "",
        "destroy_quota:",
        f"  r_lower: {p.get('r_lower', 0.15):.4f}",
        f"  r_upper_small: {p.get('r_upper_small', 0.50):.4f}",
        f"  base_cap: {p.get('quota_base_cap', 30)}",
        "",
        "reheat:",
        f"  stall_trigger: {p.get('reheat_stall_trigger', 450)}",
        f"  acceptance_window: {p.get('reheat_acceptance_window', 100)}",
        f"  acceptance_min: {p.get('reheat_acceptance_min', 0.08):.4f}",
        f"  duration: {p.get('reheat_duration', 50)}",
        f"  temperature_scale: {p.get('reheat_temperature_scale', 0.3):.4f}",
        f"  quota_multiplier: {p.get('reheat_quota_multiplier', 3.0):.4f}",
        f"  shake_probability: {p.get('reheat_shake_probability', 0.9):.4f}",
        f"  random_repair_prob: {p.get('reheat_random_repair_prob', 0.6):.4f}",
        "",
        "local_search:",
        f"  frequency: {p.get('local_search_frequency', 10)}",
        f"  on_new_best: {p.get('local_search_on_new_best', True)}",
        f"  depot_drone_probability: {p.get('depot_drone_probability', 0.18):.4f}",
        f"  intensify_frequency: {p.get('intensify_frequency', 35)}",
        "",
        "escape:",
        f"  enabled: {p.get('escape_enabled', False)}",
        f"  trigger_stall: {p.get('escape_trigger_stall', 80)}",
        f"  duration: {p.get('escape_duration', 30)}",
        "",
        "convergence_enhancement:",
        "  dynamic_cooling:",
        f"    enabled: {p.get('dynamic_cooling_enabled', True)}",
        f"    slowdown_factor: {p.get('cooling_slowdown_factor', 0.998):.6f}",
        f"    speedup_factor: {p.get('cooling_speedup_factor', 0.980):.4f}",
        "  diversification:",
        f"    enabled: {p.get('diversification_enabled', True)}",
        f"    trigger_stall: {p.get('diversification_trigger_stall', 500)}",
        f"    restart_best_prob: {p.get('diversification_restart_best_prob', 0.7):.2f}",
        f"    random_destroy_ratio: {p.get('diversification_destroy_ratio', 0.6):.2f}",
        "  adaptive_quota:",
        f"    enabled: {p.get('adaptive_quota_enabled', True)}",
        "",
        "drone:",
        f"  priority: {p.get('drone_priority', 2.2):.2f}",
        "  bonus:",
        f"    depot: {p.get('depot_bonus', 0.6):.2f}",
        f"    multi_customer: {p.get('multi_customer_bonus', 5.5):.2f}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian Optimization for ALNS")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=None,
                        help="Phase to run (1=coarse, 2=fine, 3=validation)")
    parser.add_argument("--trials", type=int, default=None,
                        help="Number of trials (phase1=60, phase2=30 default)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Number of parallel jobs (use with caution)")
    parser.add_argument("--use-demand", action="store_true",
                        help="Use demand-based deadlines (default: class-based)")
    parser.add_argument("--all", action="store_true",
                        help="Run all phases sequentially")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    args = parser.parse_args()

    if args.all:
        args.phase = None

    use_class = not args.use_demand

    results_dir = PROJECT_ROOT / "results" / ("bayesian_tuning_class" if use_class else "bayesian_tuning")
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = results_dir / f"tuning_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Bayesian tuning started: phase={args.phase}, use_class={use_class}")

    if args.phase == 1 or args.all or args.phase is None:
        n = args.trials or 60
        logger.info(f"Running Phase 1 with {n} trials...")
        study1 = run_phase1(n_trials=n, use_class=use_class)
        best_p1, ts1 = save_results("phase1", study1, results_dir)
        logger.info(f"Phase 1 complete. Best value: {study1.best_value:.2f}")

        if args.phase == 1:
            yaml_out = config_from_best_params(best_p1)
            out_path = results_dir / f"best_config_phase1_{ts1}.yaml"
            with open(out_path, "w") as f:
                f.write(yaml_out)
            logger.info(f"Best config saved → {out_path}")

    if args.phase == 2 or args.all:
        n2 = args.trials or 30
        if args.phase == 2 and not hasattr(locals(), 'study1'):
            # Load latest Phase 1 study
            import glob
            csvs = sorted(glob.glob(str(results_dir / "bayesian_phase1_*.csv")))
            if not csvs:
                logger.error("No Phase 1 data found. Run Phase 1 first.")
                sys.exit(1)
            latest = csvs[-1]
            study1 = optuna.load_study(
                study_name=f"alns_phase1_{'class' if use_class else 'demand'}",
                storage=f"sqlite:///{results_dir / 'optuna.db'}",
            )
        logger.info(f"Running Phase 2 with {n2} trials...")
        study2 = run_phase2(study1, n_trials=n2, use_class=use_class)
        best_p2, ts2 = save_results("phase2", study2, results_dir)

        yaml_out = config_from_best_params(best_p2)
        out_path = results_dir / f"best_config_phase2_{ts2}.yaml"
        with open(out_path, "w") as f:
            f.write(yaml_out)
        logger.info(f"Best config saved → {out_path}")

    if args.phase == 3:
        import glob as _glob
        jsons = sorted(_glob.glob(str(results_dir / "best_params_phase2_*.json")))
        if not jsons:
            jsons = sorted(_glob.glob(str(results_dir / "best_params_phase1_*.json")))
        if not jsons:
            logger.error("No best params file found. Run Phase 1/2 first.")
            sys.exit(1)
        with open(jsons[-1]) as f:
            best_params = json.load(f)
        logger.info(f"Loaded best params from {jsons[-1]}")
        # Handle reward_scale format
        if "reward_scale" not in best_params and "reward_global" in best_params:
            best_params["reward_scale"] = {
                "global": best_params.pop("reward_global"),
                "better": best_params.pop("reward_better"),
                "slight_better": best_params.pop("reward_slight_better"),
                "accepted_worse": best_params.pop("reward_accepted_worse"),
            }
        logger.info("Running Phase 3 validation...")
        val_results = run_validation(best_params, use_class=use_class)

        val_path = results_dir / f"validation_{timestamp}.json"
        with open(val_path, "w") as f:
            json.dump(val_results, f, indent=2, default=str)
        logger.info(f"Validation results → {val_path}")

    logger.info("Bayesian tuning complete!")
