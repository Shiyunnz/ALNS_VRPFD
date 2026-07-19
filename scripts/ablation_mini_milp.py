#!/usr/bin/env python3
"""Ablation experiment: Step6 vs Step6+miniMILP on R_40_10_1.

Tests whether a mini-MILP truck route polisher (fixing drone tasks,
re-optimizing truck routes via Gurobi) can break through the 97.60
local optimum barrier.

Two configurations:
  1. step6:       Step6=on,  miniMILP=off (best known config)
  2. step6+milp:  Step6=on,  miniMILP=on  (30s time limit, 0.5% gap)

Each runs 10 seeds (42..51), 4000 iterations + final polish.
"""

import sys
import time
import json
import random
import statistics
from pathlib import Path

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
    "step6":      {"step6": True,  "milp": False},
    "step6+milp": {"step6": True,  "milp": True},
}


def run_once(instance, evaluator, config, seed, flags):
    cfg = config
    sa_config = cfg.build_sa_config_dict()
    sa_config["iterations"] = ITERS
    sa_config["size"] = infer_size(instance)
    sa_config["drone_reanchor_ls_enabled"] = flags["step6"]
    sa_config["drone_composite_reanchor_enabled"] = False
    sa_config["drone_sortie_constructor_enabled"] = False
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
    if flags["milp"]:
        best_sol = alns.run_with_milp_polish(
            initial_solution, milp_time_limit=30.0, milp_gap=0.005)
    else:
        best_sol = alns.run(initial_solution)
    runtime = time.perf_counter() - start

    ev = evaluator.evaluate_solution(best_sol)
    drone_tasks = []
    for t in best_sol.drone_tasks:
        lt = f"T{t.launch_truck}" if t.launch_truck is not None else "depot"
        ldt = f"T{t.land_truck}" if t.land_truck is not None else "depot"
        drone_tasks.append(f"D{t.drone_id}: [{lt}@{t.launch_node} -> {t.customers()} -> {ldt}@{t.retrieve_node}]")

    return {
        "seed": seed,
        "cost": ev.total_cost,
        "feasible": ev.feasible,
        "runtime": round(runtime, 2),
        "trucks": [f"Truck {r.id}: {r.nodes}" for r in best_sol.truck_routes],
        "drones": drone_tasks,
    }


def main():
    instance = read_instance(INSTANCE, strategy="class_based")
    evaluator = Evaluator(instance)
    config = ALNSConfig()

    results = {}
    for cfg_name, flags in CONFIGS.items():
        print(f"\n=== Configuration: {cfg_name} ===")
        run_results = []
        for seed in SEEDS:
            print(f"  Seed {seed}...", end=" ", flush=True)
            r = run_once(instance, evaluator, config, seed, flags)
            run_results.append(r)
            cost_str = f"{r['cost']:.2f}" if r['feasible'] else "infeasible"
            print(f"cost={cost_str}  time={r['runtime']}s")
        costs = [r["cost"] for r in run_results if r["feasible"]]
        results[cfg_name] = {
            "results": run_results,
            "avg_cost": sum(costs) / len(costs) if costs else float("inf"),
            "min_cost": min(costs) if costs else float("inf"),
            "max_cost": max(costs) if costs else float("inf"),
            "avg_runtime": sum(r["runtime"] for r in run_results) / len(run_results),
        }

    out = Path("results/ablation_mini_milp.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")

    milp_cost = 97.18
    print("\n=== Summary ===")
    for cfg_name, data in results.items():
        costs = [r["cost"] for r in data["results"] if r["feasible"]]
        at_best = sum(1 for c in costs if abs(c - 97.60) < 0.01)
        below_98 = sum(1 for c in costs if c < 98.0)
        print(f"{cfg_name:12s}: avg={statistics.mean(costs):.2f}  med={statistics.median(costs):.2f}  "
              f"min={min(costs):.2f}  max={max(costs):.2f}  best={at_best}/10  <98={below_98}/10  "
              f"gap={min(costs)-milp_cost:.2f}")


if __name__ == "__main__":
    main()