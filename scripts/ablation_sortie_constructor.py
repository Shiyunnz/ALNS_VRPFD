#!/usr/bin/env python3
"""Ablation experiment: measure contribution of multi-customer sortie constructor.

Three configurations on R_40_10_1:
  1. baseline:  Step6=on, Step7=off, Step8=off
  2. step8:    Step6=on,  Step7=off, Step8=on  (MultiCustomerSortieConstructor)
  3. step6+8:  Step6=on,  Step7=off, Step8=on

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
    "baseline": {"step6": True, "step7": False, "step8": False},
    "step8":    {"step6": False, "step7": False, "step8": True},
    "step6+8":  {"step6": True,  "step7": False, "step8": True},
}


def run_once(instance, evaluator, config, seed, step6, step7, step8):
    cfg = config
    sa_config = cfg.build_sa_config_dict()
    sa_config["iterations"] = ITERS
    sa_config["size"] = infer_size(instance)
    sa_config["drone_reanchor_ls_enabled"] = step6
    sa_config["drone_composite_reanchor_enabled"] = step7
    sa_config["drone_sortie_constructor_enabled"] = step8
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
            r = run_once(
                instance, evaluator, config, seed,
                step6=flags["step6"],
                step7=flags["step7"],
                step8=flags["step8"],
            )
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

    out = Path("results/ablation_sortie_constructor.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")

    print("\n=== Summary ===")
    milp_cost = 97.18
    for cfg_name, data in results.items():
        costs = [r["cost"] for r in data["results"] if r["feasible"]]
        print(f"{cfg_name:12s}: avg={data['avg_cost']:.2f}  min={data['min_cost']:.2f}  "
              f"max={data['max_cost']:.2f}  gap_to_milp={data['min_cost']-milp_cost:.2f}  "
              f"runtime={data['avg_runtime']:.1f}s")


if __name__ == "__main__":
    main()