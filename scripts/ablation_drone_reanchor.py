#!/usr/bin/env python3
"""Ablation experiment: measure contribution of Step 6/7 drone re-anchor operators.

Four configurations on R_40_10_1:
  1. baseline:  Step6=off, Step7=off
  2. step6:    Step6=on,  Step7=off  (DroneTaskSplitMergeLS)
  3. step7:    Step6=off, Step7=on   (DroneTaskReanchorRepair)
  4. both:     Step6=on,  Step7=on   (both)

Each configuration runs 10 seeds (42..51), 4000 iterations.
Reports: cost, feasibility, runtime, drone task structure.
"""

import sys
import time
import json
import random
from pathlib import Path
from copy import deepcopy

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from alns_vrpfd.utils.io_utils import read_instance
from alns_vrpfd.utils.config_loader import ALNSConfig
from alns_vrpfd.evaluation.evaluator import Evaluator
from alns_vrpfd.core.sa import SANNCfg, SimulatedAnnealingALNS
from run_alns import build_operators, infer_size
from alns_vrpfd.model.initializer import build_two_phase_initial_solution

INSTANCE = "data/Instance10/R_40_10_1.txt"
ITERS = 4000
SEEDS = list(range(42, 52))
CONFIGS = {
    "baseline": {"step6": False, "step7": False},
    "step6":    {"step6": True,  "step7": False},
    "step7":    {"step6": False, "step7": True},
    "both":     {"step6": True,  "step7": True},
}


def run_once(instance, evaluator, config, seed, step6, step7):
    cfg = config
    sa_config = cfg.build_sa_config_dict()
    sa_config["iterations"] = ITERS
    sa_config["size"] = infer_size(instance)
    sa_config["drone_reanchor_ls_enabled"] = step6
    sa_config["drone_composite_reanchor_enabled"] = step7
    sa_cfg = SANNCfg(**sa_config)
    rng = random.Random(seed)

    destroy_ops, repair_ops = build_operators(
        instance, seed,
        drone_priority=cfg.drone_priority,
        repair_set="all",
        enable_composite=True,
        drone_bonus_kwargs=cfg.drone_bonus,
        forced_drone_customers=cfg.forced_drone_customers,
        robust_energy_mode="embedded",
    )

    initial_solution = build_two_phase_initial_solution(
        instance,
        truck_forbidden_customers=cfg.forced_drone_customers,
        allow_multiple_launch_per_node=cfg.relax_allow_multiple_launch_per_node,
    )

    alns = SimulatedAnnealingALNS(
        instance=instance,
        destroy_ops=destroy_ops,
        repair_ops=repair_ops,
        evaluator=evaluator,
        cfg=sa_cfg,
        rng=rng,
    )

    start = time.perf_counter()
    best_sol = alns.run(initial_solution)
    runtime = time.perf_counter() - start
    return best_sol, runtime, alns


def main():
    print("=" * 80)
    print("ABLATION EXPERIMENT: Step 6/7 drone re-anchor operators")
    print("=" * 80)

    config = ALNSConfig("config/alns_config.yaml")
    instance = read_instance(INSTANCE, strategy=config.time_window_strategy)
    if "drone" in instance.vehicle_specs:
        instance.vehicle_specs["drone"].endurance = float("inf")
    instance.configure_robustness(
        drone_battery_capacity=config.drone_battery_capacity,
        energy_uncertainty_budget=config.energy_uncertainty_budget,
        energy_deviation_rate=config.energy_deviation_rate,
        same_truck_retrieval=config.same_truck_retrieval,
    )

    evaluator = Evaluator(
        instance,
        rendezvous_tolerance=config.drone_rendezvous_tolerance,
        forced_drone_customers=config.forced_drone_customers,
        allow_multiple_launch_per_node=config.relax_allow_multiple_launch_per_node,
        cost_lambda=config.cost_lambda,
        cost_rho=config.cost_rho,
        cost_normalized=config.cost_normalized,
    )

    results = {}

    for cfg_name, flags in CONFIGS.items():
        step6 = flags["step6"]
        step7 = flags["step7"]
        print(f"\n--- Configuration: {cfg_name} (Step6={step6}, Step7={step7}) ---")
        cfg_results = []

        for seed in SEEDS:
            print(f"  Seed {seed}...", end=" ", flush=True)
            best_sol, runtime, alns = run_once(
                instance, evaluator, config, seed, step6, step7,
            )
            details = evaluator.evaluate_with_details(best_sol)
            cost = details.result.total_cost
            feasible = details.robustness.feasible

            drone_tasks_str = []
            for t in best_sol.drone_tasks:
                launch_str = f"T{t.launch_truck}@{t.launch_node}" if t.launch_truck is not None else f"depot@{t.launch_node}"
                retrieve_str = f"T{t.land_truck}@{t.retrieve_node}" if t.land_truck is not None else f"depot@{t.retrieve_node}"
                drone_tasks_str.append(f"D{t.drone_id}: [{launch_str} -> {t.customers()} -> {retrieve_str}]")

            result = {
                "seed": seed,
                "cost": cost,
                "feasible": feasible,
                "runtime": round(runtime, 2),
                "trucks": len(best_sol.truck_routes),
                "drones": len({t.drone_id for t in best_sol.drone_tasks}),
                "drone_tasks": drone_tasks_str,
            }
            cfg_results.append(result)
            print(f"cost={cost:.2f}, feasible={feasible}, time={runtime:.2f}s")

        costs = [r["cost"] for r in cfg_results]
        avg_cost = sum(costs) / len(costs)
        min_cost = min(costs)
        max_cost = max(costs)
        results[cfg_name] = {
            "results": cfg_results,
            "avg_cost": round(avg_cost, 2),
            "min_cost": round(min_cost, 2),
            "max_cost": round(max_cost, 2),
            "avg_runtime": round(sum(r["runtime"] for r in cfg_results) / len(cfg_results), 2),
        }
        print(f"  Avg={avg_cost:.2f}, Min={min_cost:.2f}, Max={max_cost:.2f}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Config':<12} {'Avg Cost':>10} {'Min Cost':>10} {'Max Cost':>10} {'Avg Time':>10} {'Gap% vs baseline':>18}")
    print("-" * 80)
    baseline_avg = results["baseline"]["avg_cost"]
    for cfg_name in CONFIGS:
        r = results[cfg_name]
        gap = (r["avg_cost"] - baseline_avg) / baseline_avg * 100 if baseline_avg > 0 else 0
        print(f"{cfg_name:<12} {r['avg_cost']:>10.2f} {r['min_cost']:>10.2f} {r['max_cost']:>10.2f} {r['avg_runtime']:>10.2f}s {gap:>17.2f}%")

    milp_cost = 97.18
    print(f"\nMILP reference cost: {milp_cost:.2f}")
    print(f"{'Config':<12} {'Gap% vs MILP':>14}")
    print("-" * 30)
    for cfg_name in CONFIGS:
        r = results[cfg_name]
        gap_milp = (r["avg_cost"] - milp_cost) / milp_cost * 100
        print(f"{cfg_name:<12} {gap_milp:>13.2f}%")

    output_path = Path("results") / "ablation_drone_reanchor.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()