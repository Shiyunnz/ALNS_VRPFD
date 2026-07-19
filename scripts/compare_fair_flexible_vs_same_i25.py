#!/usr/bin/env python3
"""Fair ALNS same-truck vs flexible docking comparison for Instance25.

Supports incremental runs (skips already-completed instance/seed pairs).
Finds the best initial solution across multiple tries to reduce infeasibility.

Usage:
    # Run all
    python scripts/compare_fair_flexible_vs_same_i25.py

    # Only specific instances (comma-separated)
    python scripts/compare_fair_flexible_vs_same_i25.py --instances R_30_25_4,R_40_25_1

    # Run with seeds (comma-separated)
    python scripts/compare_fair_flexible_vs_same_i25.py --instances ALL --seeds 100,101
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from alns_vrpfd.core.operators import (
    DestroyRandom, DestroyShaw, DestroyWorstDistance,
    RepairCheapest, RepairDronePriorityRegret, RepairEqualPriority, RepairTruckFirst,
)
from alns_vrpfd.core.sa import SANNCfg, SimulatedAnnealingALNS
from alns_vrpfd.evaluation import Evaluator
from alns_vrpfd.model.initializer import build_two_phase_initial_solution
from alns_vrpfd.model.solution import Solution
from alns_vrpfd.utils.config_loader import ALNSConfig
from alns_vrpfd.utils.io_utils import read_instance

DEFAULT_SEEDS = [100, 101, 102, 103, 104]
ALL_INSTANCES = [
    "R_30_25_1", "R_30_25_2", "R_30_25_3", "R_30_25_4", "R_30_25_5",
    "R_40_25_1", "R_40_25_2", "R_40_25_3", "R_40_25_4", "R_40_25_5",
    "R_50_25_1", "R_50_25_2", "R_50_25_3", "R_50_25_4", "R_50_25_5",
]
INSTANCE_DIR = "Instance25"
ALNS_ITERATIONS = 4000
ALNS_TIME_LIMIT = 300.0
OUTPUT_DIR = PROJECT_ROOT / "results/revision_experiments/fair_flexible_vs_same_i25"
CFG_PATH = PROJECT_ROOT / "config" / "alns_config.yaml"
RESULTS_PATH = OUTPUT_DIR / "comparison_results.json"


def _region_of(name: str) -> int:
    return int(name.split("_")[1])


def _load_instance(instance_name: str):
    fpath = str(PROJECT_ROOT / "data" / INSTANCE_DIR / f"{instance_name}.txt")
    cfg = ALNSConfig(str(CFG_PATH))
    instance = read_instance(fpath, strategy="demand_based")
    if "drone" in instance.vehicle_specs:
        instance.vehicle_specs["drone"].endurance = float("inf")
    instance.configure_robustness(
        drone_battery_capacity=cfg.drone_battery_capacity,
        energy_uncertainty_budget=cfg.energy_uncertainty_budget,
        energy_deviation_rate=cfg.energy_deviation_rate,
        same_truck_retrieval=False,
    )
    evaluator = Evaluator(
        instance,
        rendezvous_tolerance=cfg.drone_rendezvous_tolerance,
        forced_drone_customers=cfg.forced_drone_customers,
        allow_multiple_launch_per_node=cfg.relax_allow_multiple_launch_per_node,
        cost_lambda=cfg.cost_lambda,
        cost_rho=cfg.cost_rho,
        cost_normalized=cfg.cost_normalized,
    )
    return instance, evaluator


def _build_alns(instance, evaluator, seed: int) -> SimulatedAnnealingALNS:
    cfg = ALNSConfig(str(CFG_PATH))
    sa_cfg = cfg.build_sa_config_dict()
    sa_cfg["iterations"] = ALNS_ITERATIONS
    sa_cfg["size"] = "small"
    sa_cfg["log_operator_metrics"] = False
    sa_cfg["track_no_cross_truck"] = True

    dp = cfg.drone_priority
    bonus = {
        "depot_bonus": cfg.drone_bonus["depot_bonus"],
        "multi_customer_bonus": cfg.drone_bonus["multi_customer_bonus"],
        "multi_customer_threshold": 2,
        "wait_max": 20.0,
        "allow_multiple_launch_per_node": cfg.relax_allow_multiple_launch_per_node,
    }

    destroy_ops = [
        DestroyRandom(instance, rng=random.Random(seed + 1000), anchor_strategy="rebase_to_neighbor"),
        DestroyWorstDistance(instance, rng=random.Random(seed + 1004), anchor_strategy="rebase_to_neighbor"),
        DestroyShaw(instance, rng=random.Random(seed + 1002), anchor_strategy="rebase_to_neighbor"),
    ]
    repair_ops = [
        RepairCheapest(instance, rng=random.Random(seed + 2004), drone_priority=dp, robust_energy_mode="embedded", **bonus),
        RepairDronePriorityRegret(instance, rng=random.Random(seed + 2002), drone_priority=dp, robust_energy_mode="embedded", **bonus),
        RepairTruckFirst(instance, rng=random.Random(seed + 2003), drone_priority=dp, robust_energy_mode="embedded", **bonus),
        RepairEqualPriority(instance, rng=random.Random(seed + 2001), drone_priority=dp, robust_energy_mode="embedded", **bonus),
    ]

    return SimulatedAnnealingALNS(
        instance=instance, destroy_ops=destroy_ops, repair_ops=repair_ops,
        evaluator=evaluator, cfg=SANNCfg(**sa_cfg), rng=random.Random(seed),
        verbose=False,
    )


def _count_cross_truck(solution: Solution) -> int:
    return sum(
        1 for task in solution.drone_tasks
        if task.launch_truck is not None and task.land_truck is not None
        and task.launch_truck != task.land_truck
    )


def _try_build_initial(instance, evaluator, max_attempts: int = 5):
    """Try multiple initial solution strategies, return the best feasible one."""
    best_cost = float("inf")
    best_solution = None

    # Try 1: two-phase
    for attempt in range(max_attempts):
        seed_offset = attempt * 1000
        try:
            rng_seed = 42 + seed_offset
            rng = random.Random(rng_seed)
            from alns_vrpfd.model.initializer import build_two_phase_initial_solution
            sol = build_two_phase_initial_solution(instance)
            ev = evaluator.evaluate_solution(sol)
            if ev.feasible and ev.total_cost < best_cost:
                best_cost = ev.total_cost
                best_solution = sol
                break
        except Exception:
            pass

    # Try 2: basic initializer
    if best_solution is None:
        try:
            from alns_vrpfd.model.feasible_initializer import build_feasible_initial_solution
            sol, _ = build_feasible_initial_solution(instance, evaluator)
            ev = evaluator.evaluate_solution(sol)
            if ev.feasible and ev.total_cost < best_cost:
                best_cost = ev.total_cost
                best_solution = sol
        except Exception:
            pass

    # Try 3: truck-only fallback
    if best_solution is None:
        try:
            from alns_vrpfd.model.initializer import _build_truck_routes
            truck_routes = _build_truck_routes(instance, time_limit=10.0)
            sol = Solution.empty()
            for route in truck_routes:
                sol.add_truck_route(route)
            best_solution = sol
        except Exception:
            pass

    return best_solution


def run_instance(instance_name: str, seed: int) -> Dict[str, Any]:
    instance, evaluator = _load_instance(instance_name)
    alns = _build_alns(instance, evaluator, seed)

    initial = _try_build_initial(instance, evaluator)
    if initial is None:
        return {
            "instance": instance_name,
            "region": _region_of(instance_name),
            "seed": seed,
            "same_cost": None, "flex_cost": None,
            "same_feasible": False, "flex_feasible": False,
            "same_cross_truck": 0, "flex_cross_truck": 0,
            "has_true_flex": False, "runtime": 0.0,
        }

    t0 = time.time()
    best = alns.run(initial, time_limit=ALNS_TIME_LIMIT)
    runtime = time.time() - t0

    same_solution = alns._best_no_cross_truck_solution or best
    flex_solution = best

    same_eval = evaluator.evaluate_solution(same_solution)
    flex_eval = evaluator.evaluate_solution(flex_solution)
    same_cross = _count_cross_truck(same_solution)
    flex_cross = _count_cross_truck(flex_solution)

    return {
        "instance": instance_name,
        "region": _region_of(instance_name),
        "seed": seed,
        "same_cost": same_eval.total_cost if same_eval.feasible else None,
        "flex_cost": flex_eval.total_cost if flex_eval.feasible else None,
        "same_feasible": same_eval.feasible,
        "flex_feasible": flex_eval.feasible,
        "same_cross_truck": same_cross,
        "flex_cross_truck": flex_cross,
        "has_true_flex": flex_cross > 0,
        "runtime": round(runtime, 2),
    }


def load_checkpoint() -> Dict[str, Dict[str, Any]]:
    """Load existing results keyed by 'instance|seed'."""
    if not RESULTS_PATH.exists():
        return {}
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    return {f"{r['instance']}|{r['seed']}": r for r in data}


def save_results(results: List[Dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    # Write summary CSV for plot
    feasible = [r for r in results if r["same_feasible"] and r["flex_feasible"]]
    by_inst: Dict[str, List] = {}
    for r in feasible:
        by_inst.setdefault(r["instance"], []).append(r)
    rows = []
    for inst in sorted(by_inst):
        runs = by_inst[inst]
        region = _region_of(inst)
        same_costs = [r["same_cost"] for r in runs]
        flex_costs = [r["flex_cost"] for r in runs]
        avg_same = float(np.mean(same_costs))
        avg_flex = float(np.mean(flex_costs))
        saving = (avg_same - avg_flex) / avg_same * 100 if avg_same > 0 else 0
        n_true = sum(1 for r in runs if r["has_true_flex"])
        rows.append({
            "region": region, "group": inst,
            "avg_same_cost": round(avg_same, 2),
            "avg_flexible_cost": round(avg_flex, 2),
            "avg_flexible_saving_vs_same": round(saving, 4),
            "n_seeds": len(runs), "n_true_flex": n_true,
        })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUTPUT_DIR / "summary_for_plot.csv", index=False)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", default=None,
                        help="Comma-separated instance names, or 'ALL'")
    parser.add_argument("--seeds", default=None,
                        help="Comma-separated seed values")
    args = parser.parse_args()

    if args.instances and args.instances.upper() != "ALL":
        instances = [s.strip() for s in args.instances.split(",")]
    else:
        instances = ALL_INSTANCES

    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds = DEFAULT_SEEDS

    checkpoint = load_checkpoint()
    results = list(checkpoint.values())

    for inst in instances:
        for seed in seeds:
            key = f"{inst}|{seed}"
            if key in checkpoint:
                prev = checkpoint[key]
                if prev.get("same_feasible") and prev.get("flex_feasible"):
                    print(f"  {inst} seed={seed}... SKIP (already done)")
                    continue
                print(f"  {inst} seed={seed}... RERUN (previously infeasible)", end=" ", flush=True)
            else:
                print(f"  {inst} seed={seed}...", end=" ", flush=True)

            try:
                r = run_instance(inst, seed)
                r["_key"] = key
                if key in checkpoint:
                    # Update in place
                    for i, existing in enumerate(results):
                        if existing.get("_key") == key:
                            results[i] = r
                            break
                else:
                    results.append(r)

                if r["same_feasible"] and r["flex_feasible"]:
                    saving = (r["same_cost"] - r["flex_cost"]) / r["same_cost"] * 100
                    print(f"same={r['same_cost']:.2f} flex={r['flex_cost']:.2f} saving={saving:.2f}% cross={r['flex_cross_truck']}")
                else:
                    print(f"INFEASIBLE")
                save_results(results)
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback; traceback.print_exc()

    save_results(results)
    print(f"\nSaved {len(results)} results to {RESULTS_PATH}")

    # Print summary
    feasible = [r for r in results if r["same_feasible"] and r["flex_feasible"]]
    print(f"\n{'Instance':>12} | {'Same':>8} | {'Flex':>8} | {'Saving':>7} | {'TrueFlex':>8}")
    print("-" * 55)
    by_inst: Dict[str, List] = {}
    for r in feasible:
        by_inst.setdefault(r["instance"], []).append(r)
    for inst, runs in sorted(by_inst.items()):
        same_m = np.mean([r["same_cost"] for r in runs])
        flex_m = np.mean([r["flex_cost"] for r in runs])
        cross_count = sum(1 for r in runs if r["has_true_flex"])
        saving = (same_m - flex_m) / same_m * 100 if same_m > 0 else 0
        print(f"  {inst:>12} | {same_m:>8.2f} | {flex_m:>8.2f} | {saving:>+6.2f}% | {cross_count}/{len(runs)}")


if __name__ == "__main__":
    main()
