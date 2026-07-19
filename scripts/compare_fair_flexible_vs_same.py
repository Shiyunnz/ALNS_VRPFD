#!/usr/bin/env python3
"""Fair ALNS same-truck vs flexible docking comparison (single run, dual tracking).

Runs ALNS once with `same_truck_retrieval=False` and tracks:
- best_no_cross_truck_solution  → same-truck baseline (no cross-truck sorties)
- best (global)                 → flexible result (may include cross-truck)

This eliminates the search-space confound: both results come from the exact same
search trajectory, so any cost difference is truly attributable to the presence
of cross-truck sorties, not to different search paths.

Usage:
    cd code/
    python scripts/compare_fair_flexible_vs_same.py
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from alns_vrpfd.core.operators import (
    DestroyRandom, DestroyShaw, DestroyWorstDistance,
    RepairCheapest, RepairDronePriorityRegret, RepairEqualPriority, RepairTruckFirst,
)
from alns_vrpfd.core.sa import SANNCfg, SimulatedAnnealingALNS
from alns_vrpfd.evaluation import Evaluator
from alns_vrpfd.model.solution import Solution
from alns_vrpfd.utils.config_loader import ALNSConfig
from alns_vrpfd.utils.io_utils import read_instance
from scripts.analyze_flexible_docking_operational_metrics import MetricRow, _extract_metrics


SEEDS = [42, 43, 44, 45, 46]
INSTANCES = [
    "R_30_10_1", "R_30_10_2", "R_30_10_3", "R_30_10_4", "R_30_10_5",
    "R_40_10_1", "R_40_10_2", "R_40_10_3", "R_40_10_4", "R_40_10_5",
    "R_50_10_1", "R_50_10_2", "R_50_10_3", "R_50_10_4", "R_50_10_5",
]
INSTANCE_DIR = "Instance10"
ALNS_ITERATIONS = 4000
ALNS_TIME_LIMIT = 600.0  # seconds
OUTPUT_DIR = PROJECT_ROOT / "results/revision_experiments/fair_flexible_vs_same_i10"

CFG_PATH = PROJECT_ROOT / "config" / "alns_config.yaml"


def _load_instance(instance_name: str, seed: int):
    """Load and configure an instance (no time windows, flexible docking)."""
    fpath = str(PROJECT_ROOT / "data" / INSTANCE_DIR / f"{instance_name}.txt")
    cfg = ALNSConfig(str(CFG_PATH))
    instance = read_instance(fpath, strategy="class_based", apply_time_windows=False)
    instance.vehicle_specs["drone"].endurance = float("inf")
    instance.configure_robustness(
        drone_battery_capacity=cfg.drone_battery_capacity,
        energy_uncertainty_budget=cfg.energy_uncertainty_budget,
        energy_deviation_rate=cfg.energy_deviation_rate,
        same_truck_retrieval=False,  # ← always flexible
    )
    # Generate class-based deadlines
    from revision.tune_base import generate_class_based_deadlines
    classes = generate_class_based_deadlines(instance, seed=seed)
    evaluator = Evaluator(
        instance,
        rendezvous_tolerance=cfg.drone_rendezvous_tolerance,
        forced_drone_customers=cfg.forced_drone_customers,
        allow_multiple_launch_per_node=cfg.relax_allow_multiple_launch_per_node,
        cost_lambda=cfg.cost_lambda,
        cost_rho=cfg.cost_rho,
        cost_normalized=cfg.cost_normalized,
    )
    return instance, evaluator, classes


def _build_alns(instance, evaluator, seed: int) -> SimulatedAnnealingALNS:
    cfg = ALNSConfig(str(CFG_PATH))
    sa_cfg = cfg.build_sa_config_dict()
    sa_cfg["iterations"] = ALNS_ITERATIONS
    sa_cfg["size"] = "small"
    sa_cfg["log_operator_metrics"] = False
    sa_cfg["track_no_cross_truck"] = True  # ← enable dual tracking

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


@dataclass
class RunResult:
    instance: str
    seed: int
    same_cost: float
    flex_cost: float
    same_feasible: bool
    flex_feasible: bool
    same_cross_truck: int
    flex_cross_truck: int
    same_drone_cust: int
    flex_drone_cust: int
    runtime: float
    has_true_flex: bool  # flex solution uses cross-truck sorties


def run(instance_name: str, seed: int) -> RunResult:
    instance, evaluator, classes = _load_instance(instance_name, seed)
    alns = _build_alns(instance, evaluator, seed)

    from alns_vrpfd.model.initializer import build_two_phase_initial_solution
    initial = build_two_phase_initial_solution(instance)

    t0 = time.time()
    best = alns.run(initial, time_limit=ALNS_TIME_LIMIT)
    runtime = time.time() - t0

    # Extract dual results
    same_solution = alns._best_no_cross_truck_solution or best
    flex_solution = best

    same_eval = evaluator.evaluate_solution(same_solution)
    flex_eval = evaluator.evaluate_solution(flex_solution)

    same_cross = _count_cross_truck(same_solution)
    flex_cross = _count_cross_truck(flex_solution)
    same_drone = len({c for t in same_solution.drone_tasks for c in t.customers()})
    flex_drone = len({c for t in flex_solution.drone_tasks for c in t.customers()})

    return RunResult(
        instance=instance_name,
        seed=seed,
        same_cost=same_eval.total_cost if same_eval.feasible else float("inf"),
        flex_cost=flex_eval.total_cost if flex_eval.feasible else float("inf"),
        same_feasible=same_eval.feasible,
        flex_feasible=flex_eval.feasible,
        same_cross_truck=same_cross,
        flex_cross_truck=flex_cross,
        same_drone_cust=same_drone,
        flex_drone_cust=flex_drone,
        runtime=runtime,
        has_true_flex=flex_cross > 0,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, Any]] = []

    for inst in INSTANCES:
        for seed in SEEDS:
            print(f"  {inst} seed={seed}...", end=" ", flush=True)
            try:
                r = run(inst, seed)
                all_results.append({
                    "instance": inst, "seed": seed,
                    "same_cost": r.same_cost, "flex_cost": r.flex_cost,
                    "same_feasible": r.same_feasible, "flex_feasible": r.flex_feasible,
                    "same_cross_truck": r.same_cross_truck,
                    "flex_cross_truck": r.flex_cross_truck,
                    "same_drone_cust": r.same_drone_cust,
                    "flex_drone_cust": r.flex_drone_cust,
                    "saving_pct": (
                        (r.same_cost - r.flex_cost) / r.same_cost * 100
                        if r.same_feasible and r.flex_feasible and r.same_cost > 0
                        else None
                    ),
                    "has_true_flex": r.has_true_flex,
                    "runtime": r.runtime,
                })
                saving = all_results[-1]["saving_pct"]
                print(f"same={r.same_cost:.2f} flex={r.flex_cost:.2f} "
                      f"saving={saving or 'N/A'}% cross={r.flex_cross_truck}")
            except Exception as e:
                print(f"ERROR: {e}")

    # Save raw results
    out_path = OUTPUT_DIR / "comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved raw results to {out_path}")

    # Compute summary
    feasible = [r for r in all_results if r["same_feasible"] and r["flex_feasible"]]
    n_total = len(all_results)
    n_feasible = len(feasible)

    print("\n" + "=" * 80)
    print(f"  FAIR COMPARISON: ALNS single-run dual-tracking (Instance10)")
    print(f"  {n_feasible}/{n_total} feasible pairs")
    print("=" * 80)

    # True-flexible vs same-truck-only comparison
    true_flex = [r for r in feasible if r["has_true_flex"]]
    no_flex = [r for r in feasible if not r["has_true_flex"]]

    if true_flex:
        same_mean = sum(r["same_cost"] for r in true_flex) / len(true_flex)
        flex_mean = sum(r["flex_cost"] for r in true_flex) / len(true_flex)
        savings = [r["saving_pct"] for r in true_flex if r["saving_pct"] is not None]
        avg_saving = sum(savings) / len(savings) if savings else 0
        print(f"\n  TRUE FLEXIBLE (cross-truck > 0): {len(true_flex)} runs")
        print(f"    Same-truck mean: {same_mean:.2f}")
        print(f"    Flexible mean:   {flex_mean:.2f}")
        print(f"    Avg saving:      {avg_saving:.2f}%")
    else:
        print("\n  No runs with cross-truck sorties found.")

    if no_flex:
        same_mean = sum(r["same_cost"] for r in no_flex) / len(no_flex)
        flex_mean = sum(r["flex_cost"] for r in no_flex) / len(no_flex)
        print(f"\n  SAME-TRUCK ONLY (cross-truck = 0): {len(no_flex)} runs")
        print(f"    Same-truck mean: {same_mean:.2f}")
        print(f"    Flexible mean:   {flex_mean:.2f}")
        print(f"    Same cost (no-cross-truck baseline) vs global best within same search")
        print(f"    → differences reflect stochastic variation within same trajectory")

    # Aggregate by instance
    print(f"\n  {'Instance':>12} | {'SameMean':>8} | {'FlexMean':>8} | {'Saving':>7} | {'Cross>0':>6}")
    print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*6}")
    by_inst: Dict[str, List] = {}
    for r in feasible:
        by_inst.setdefault(r["instance"], []).append(r)
    for inst, runs in sorted(by_inst.items()):
        same_m = sum(r["same_cost"] for r in runs) / len(runs)
        flex_m = sum(r["flex_cost"] for r in runs) / len(runs)
        cross_count = sum(1 for r in runs if r["has_true_flex"])
        saving = (same_m - flex_m) / same_m * 100 if same_m > 0 else 0
        print(f"  {inst:>12} | {same_m:>8.2f} | {flex_m:>8.2f} | {saving:>+6.2f}% | {cross_count:>3}/{len(runs)}")
    print()


if __name__ == "__main__":
    main()
