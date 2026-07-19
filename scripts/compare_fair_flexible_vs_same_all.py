#!/usr/bin/env python3
"""Fair ALNS comparison for any instance size with checkpointing.

Usage:
    # Run n=50 with 5 seeds, flexible time limit
    python scripts/compare_fair_flexible_vs_same_all.py --size 50 --seeds 100,101,102

    # Run n=75 with 3 seeds
    python scripts/compare_fair_flexible_vs_same_all.py --size 75 --seeds 200,201,202

    # Run all sizes
    python scripts/compare_fair_flexible_vs_same_all.py --size all
"""

from __future__ import annotations

import json, math, random, sys, time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from alns_vrpfd.core.operators import (
    DestroyRandom, DestroyShaw, DestroyWorstDistance,
    RepairCheapest, RepairDronePriorityRegret, RepairEqualPriority, RepairTruckFirst,
)
from alns_vrpfd.core.sa import SANNCfg, SimulatedAnnealingALNS
from alns_vrpfd.evaluation import Evaluator
from alns_vrpfd.model.solution import Solution
from alns_vrpfd.utils.config_loader import ALNSConfig
from alns_vrpfd.utils.io_utils import read_instance

CFG_PATH = PROJECT_ROOT / "config" / "alns_config.yaml"
BASE_OUTPUT = PROJECT_ROOT / "results/revision_experiments"

# Instance sizes and their configs
SIZE_CONFIGS = {
    50: {"time_limit": 600, "instances": [f"R_{r}_50_{i}" for r in [30,40,50] for i in range(1,6)]},
    75: {"time_limit": 600, "instances": [f"R_{r}_75_{i}" for r in [30,40,50] for i in range(1,6)]},
    100: {"time_limit": 600, "instances": [f"R_{r}_100_{i}" for r in [30,40,50] for i in range(1,6)]},
}


def _region_of(name: str) -> int:
    return int(name.split("_")[1])


def load_instance(instance_name: str, size: int):
    fpath = str(PROJECT_ROOT / "data" / f"Instance{size}" / f"{instance_name}.txt")
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
        cost_lambda=cfg.cost_lambda, cost_rho=cfg.cost_rho, cost_normalized=cfg.cost_normalized,
    )
    return instance, evaluator


def build_alns(instance, evaluator, seed: int, size: int) -> SimulatedAnnealingALNS:
    cfg = ALNSConfig(str(CFG_PATH))
    sa_cfg = cfg.build_sa_config_dict()
    sa_cfg["iterations"] = 4000
    sa_cfg["size"] = "large" if size >= 50 else "small"
    sa_cfg["log_operator_metrics"] = False
    sa_cfg["track_no_cross_truck"] = True
    dp = cfg.drone_priority
    bonus = {"depot_bonus": cfg.drone_bonus["depot_bonus"], "multi_customer_bonus": cfg.drone_bonus["multi_customer_bonus"],
             "multi_customer_threshold": 2, "wait_max": 20.0, "allow_multiple_launch_per_node": cfg.relax_allow_multiple_launch_per_node}
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
    return SimulatedAnnealingALNS(instance=instance, destroy_ops=destroy_ops, repair_ops=repair_ops,
                                   evaluator=evaluator, cfg=SANNCfg(**sa_cfg), rng=random.Random(seed), verbose=False)


def _count_cross_truck(solution: Solution) -> int:
    return sum(1 for t in solution.drone_tasks
               if t.launch_truck is not None and t.land_truck is not None and t.launch_truck != t.land_truck)


def run_instance(instance_name: str, size: int, seed: int) -> Dict[str, Any]:
    instance, evaluator = load_instance(instance_name, size)
    alns = build_alns(instance, evaluator, seed, size)
    from alns_vrpfd.model.initializer import build_two_phase_initial_solution
    initial = build_two_phase_initial_solution(instance)
    t0 = time.time()
    best = alns.run(initial, time_limit=SIZE_CONFIGS[size]["time_limit"])
    runtime = time.time() - t0
    same_sol = alns._best_no_cross_truck_solution or best
    flex_sol = best
    same_eval = evaluator.evaluate_solution(same_sol)
    flex_eval = evaluator.evaluate_solution(flex_sol)
    return {"instance": instance_name, "region": _region_of(instance_name), "seed": seed,
            "same_cost": same_eval.total_cost if same_eval.feasible else None,
            "flex_cost": flex_eval.total_cost if flex_eval.feasible else None,
            "same_feasible": same_eval.feasible, "flex_feasible": flex_eval.feasible,
            "same_cross_truck": _count_cross_truck(same_sol),
            "flex_cross_truck": _count_cross_truck(flex_sol),
            "has_true_flex": _count_cross_truck(flex_sol) > 0, "runtime": round(runtime, 2)}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=str, default="all", help="Instance size: 50, 75, 100, or 'all'")
    parser.add_argument("--seeds", type=str, default="100,101,102,103,104", help="Comma-separated seeds")
    args = parser.parse_args()

    if args.size == "all":
        sizes = [50, 75, 100]
    else:
        sizes = [int(args.size)]

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    for size in sizes:
        print(f"\n{'='*60}")
        print(f"  Running Instance{size} with {len(seeds)} seeds")
        print(f"{'='*60}")

        output_dir = BASE_OUTPUT / f"fair_flexible_vs_same_i{size}"
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "comparison_results.json"

        # Load checkpoint
        existing = {}
        if results_path.exists():
            with open(results_path) as f:
                for r in json.load(f):
                    existing[f"{r['instance']}|{r['seed']}"] = r

        instances = SIZE_CONFIGS[size]["instances"]
        results = list(existing.values())

        for inst in instances:
            for seed in seeds:
                key = f"{inst}|{seed}"
                if key in existing and existing[key].get("flex_feasible"):
                    print(f"  {inst} seed={seed}... SKIP")
                    continue
                print(f"  {inst} seed={seed}...", end=" ", flush=True)
                try:
                    r = run_instance(inst, size, seed)
                    r["_key"] = key
                    # update or append
                    found = False
                    for i, existing_r in enumerate(results):
                        if existing_r.get("_key") == key:
                            results[i] = r
                            found = True
                            break
                    if not found:
                        results.append(r)
                    if r["flex_feasible"]:
                        saving = (r["same_cost"] - r["flex_cost"]) / r["same_cost"] * 100 if r["same_cost"] else 0
                        print(f"flex={r['flex_cost']:.2f} saving={saving:.2f}% cross={r['flex_cross_truck']}")
                    else:
                        print("INFEASIBLE")
                    with open(results_path, "w") as f:
                        json.dump(results, f, indent=2, default=str)
                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback; traceback.print_exc()

        # Summary
        feasible = [r for r in results if r["flex_feasible"]]
        by_inst = {}
        for r in feasible:
            by_inst.setdefault(r["instance"], []).append(r)
        print(f"\n  Summary Instance{size}:")
        print(f"  {'Instance':>12} | {'FlexBest':>8} | {'FlexMean':>8} | {'Saving':>7} | {'TrueFlex':>8}")
        print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}")
        rows = []
        for inst in sorted(by_inst):
            runs = by_inst[inst]
            flex_costs = [r["flex_cost"] for r in runs]
            same_costs = [r["same_cost"] for r in runs]
            best = min(flex_costs)
            mean = np.mean(flex_costs)
            same_mean = np.mean(same_costs)
            saving = (same_mean - mean) / same_mean * 100 if same_mean > 0 else 0
            n_true = sum(1 for r in runs if r["has_true_flex"])
            rows.append({"inst": inst, "best": best, "mean": mean, "saving": saving, "n_true": n_true})
            print(f"  {inst:>12} | {best:>8.2f} | {mean:>8.2f} | {saving:>+6.2f}% | {n_true}/{len(runs)}")

        # Save best-of-5-seeds summary
        summary_rows = [{"instance": r["inst"], "best_flex": round(r["best"], 2),
                         "mean_flex": round(r["mean"], 2), "saving_pct": round(r["saving"], 2),
                         "n_true_flex": r["n_true"], "n_seeds": len(by_inst[r["inst"]])}
                        for r in rows]
        with open(output_dir / "best_summary.json", "w") as f:
            json.dump(summary_rows, f, indent=2)
        print(f"\n  Saved best summary to {output_dir / 'best_summary.json'}")

        overall_avg = np.mean([r["saving"] for r in rows]) if rows else 0
        print(f"  Overall average saving: {overall_avg:.2f}%")


if __name__ == "__main__":
    main()
