#!/usr/bin/env python3
"""Run ALNS with dual-tracking, serialize solutions, output metrics CSV.

Output CSV is compatible with analyze_flexible_docking_operational_metrics.py.

Usage:
    # Run n=25 with 2 seeds
    python scripts/run_flexible_docking_full_metrics.py --size 25 --seeds 100,101

    # Run all sizes
    python scripts/run_flexible_docking_full_metrics.py --size all --seeds 100,101
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
import time
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
from alns_vrpfd.model.route import DroneTask, TruckRoute
from alns_vrpfd.model.solution import Solution
from alns_vrpfd.utils.config_loader import ALNSConfig
from alns_vrpfd.utils.io_utils import read_instance

CFG_PATH = PROJECT_ROOT / "config" / "alns_config.yaml"
BASE_OUTPUT = PROJECT_ROOT / "results/revision_experiments/flexible_docking_full_metrics"

SIZE_CONFIGS = {
    10: {"time_limit": 300, "instances": [f"R_{r}_10_{i}" for r in [30, 40, 50] for i in range(1, 6)]},
    25: {"time_limit": 600, "instances": [f"R_{r}_25_{i}" for r in [30, 40, 50] for i in range(1, 6)]},
    50: {"time_limit": 600, "instances": [f"R_{r}_50_{i}" for r in [30, 40, 50] for i in range(1, 6)]},
    75: {"time_limit": 600, "instances": [f"R_{r}_75_{i}" for r in [30, 40, 50] for i in range(1, 6)]},
    100: {"time_limit": 600, "instances": [f"R_{r}_100_{i}" for r in [30, 40, 50] for i in range(1, 6)]},
}

OUTPUT_FIELDS = [
    "instance", "instance_name", "region",
    "same_seed", "flex_seed",
    "same_cost", "flexible_cost", "flexible_saving_vs_same",
    "same_best_drone_customers", "flex_best_drone_customers",
    "same_truck_routes", "same_drone_tasks",
    "flexible_truck_routes", "flexible_drone_tasks",
]


def _region_of(name: str) -> int:
    return int(name.split("_")[1])


def _serialize_truck_routes(solution: Solution) -> str:
    payload = []
    for route in solution.truck_routes:
        payload.append({"truck_id": route.id, "nodes": list(route.nodes)})
    return json.dumps(payload, ensure_ascii=False)


def _serialize_drone_tasks(solution: Solution) -> str:
    payload = []
    for task in solution.drone_tasks:
        payload.append({
            "drone_id": task.drone_id,
            "launch_truck": task.launch_truck,
            "launch_node": task.launch_node,
            "customers": list(task.customers()),
            "land_truck": task.land_truck,
            "retrieve_node": task.retrieve_node,
        })
    return json.dumps(payload, ensure_ascii=False)


def _count_drone_customers(solution: Solution) -> int:
    return len({node for task in solution.drone_tasks for node in task.customers()})


def _count_cross_truck(solution: Solution) -> int:
    return sum(1 for t in solution.drone_tasks
               if t.launch_truck is not None and t.land_truck is not None
               and t.launch_truck != t.land_truck)


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
    bonus = {"depot_bonus": cfg.drone_bonus["depot_bonus"],
             "multi_customer_bonus": cfg.drone_bonus["multi_customer_bonus"],
             "multi_customer_threshold": 2, "wait_max": 20.0,
             "allow_multiple_launch_per_node": cfg.relax_allow_multiple_launch_per_node}
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


def _try_build_initial(instance, evaluator):
    """Try to build an initial solution. ALNS can handle infeasible starts."""
    for attempt in range(5):
        try:
            from alns_vrpfd.model.initializer import build_two_phase_initial_solution
            sol = build_two_phase_initial_solution(instance)
            return sol
        except Exception:
            pass
    try:
        from alns_vrpfd.model.feasible_initializer import build_feasible_initial_solution
        sol, _ = build_feasible_initial_solution(instance, evaluator)
        return sol
    except Exception:
        pass
    try:
        from alns_vrpfd.model.initializer import _build_truck_routes
        from alns_vrpfd.model.solution import Solution
        truck_routes = _build_truck_routes(instance, time_limit=10.0)
        sol = Solution.empty()
        for route in truck_routes:
            sol.add_truck_route(route)
        return sol
    except Exception:
        pass
    return None


def run_instance(instance_name: str, size: int, seed: int) -> Dict[str, Any]:
    instance, evaluator = load_instance(instance_name, size)
    alns = build_alns(instance, evaluator, seed, size)
    initial = _try_build_initial(instance, evaluator)
    if initial is None:
        return {"instance": f"data/Instance{size}/{instance_name}.txt",
                "instance_name": instance_name, "region": _region_of(instance_name),
                "same_seed": seed, "flex_seed": seed,
                "same_cost": None, "flexible_cost": None, "flexible_saving_vs_same": 0.0,
                "same_best_drone_customers": 0, "flex_best_drone_customers": 0,
                "same_truck_routes": "[]", "same_drone_tasks": "[]",
                "flexible_truck_routes": "[]", "flexible_drone_tasks": "[]"}
    t0 = time.time()
    best = alns.run(initial, time_limit=SIZE_CONFIGS[size]["time_limit"])
    runtime = time.time() - t0

    same_sol = alns._best_no_cross_truck_solution or best
    flex_sol = best

    same_eval = evaluator.evaluate_solution(same_sol)
    flex_eval = evaluator.evaluate_solution(flex_sol)

    same_cost = same_eval.total_cost if same_eval.feasible else None
    flex_cost = flex_eval.total_cost if flex_eval.feasible else None
    saving = (same_cost - flex_cost) / same_cost * 100 if (same_cost and flex_cost and same_cost > 0) else 0.0

    instance_path = f"data/Instance{size}/{instance_name}.txt"

    return {
        "instance": instance_path,
        "instance_name": instance_name,
        "region": _region_of(instance_name),
        "same_seed": seed,
        "flex_seed": seed,
        "same_cost": same_cost,
        "flexible_cost": flex_cost,
        "flexible_saving_vs_same": round(saving, 6),
        "same_best_drone_customers": _count_drone_customers(same_sol),
        "flex_best_drone_customers": _count_drone_customers(flex_sol),
        "same_truck_routes": _serialize_truck_routes(same_sol),
        "same_drone_tasks": _serialize_drone_tasks(same_sol),
        "flexible_truck_routes": _serialize_truck_routes(flex_sol),
        "flexible_drone_tasks": _serialize_drone_tasks(flex_sol),
    }


def verify_serialization(row: Dict[str, Any]) -> bool:
    """Verify that the serialized solutions can be deserialized and match costs."""
    try:
        truck_json = json.loads(row["flexible_truck_routes"])
        drone_json = json.loads(row["flexible_drone_tasks"])

        fpath = str(PROJECT_ROOT / row["instance"])
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

        sol = Solution.empty()
        truck_cap = float(instance.vehicle_specs["truck"].capacity)
        for route_data in truck_json:
            nodes = [int(n) for n in route_data["nodes"]]
            sol.add_truck_route(TruckRoute(
                route_id=int(route_data["truck_id"]),
                nodes=nodes,
                capacity=truck_cap,
            ))
        for task_data in drone_json:
            sol.add_drone_task(DroneTask(
                drone_id=int(task_data.get("drone_id", 0)),
                launch_truck=task_data.get("launch_truck"),
                launch_node=int(task_data["launch_node"]),
                customers=[int(n) for n in task_data.get("customers", [])],
                land_truck=task_data.get("land_truck"),
                retrieve_node=int(task_data["retrieve_node"]),
            ))

        result = evaluator.evaluate_solution(sol)
        if not result.feasible:
            print(f"    ❌ Verif: rebuilt solution INFEASIBLE")
            return False

        orig_cost = float(row["flexible_cost"])
        rebuilt_cost = result.total_cost
        if abs(orig_cost - rebuilt_cost) > 1e-4:
            print(f"    ❌ Verif: cost mismatch {orig_cost} vs {rebuilt_cost}")
            return False

        return True
    except Exception as e:
        print(f"    ❌ Verif: exception {e}")
        return False


def _read_existing_csv(csv_path: Path) -> Dict[str, Dict[str, Any]]:
    """Read existing CSV into a dict keyed by 'instance_name|flex_seed'."""
    if not csv_path.exists():
        return {}
    existing = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['instance_name']}|{row['flex_seed']}"
            existing[key] = row
    return existing


def _write_csv(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=str, default="all", help="Instance size: 10, 25, 50, 75, 100, or 'all'")
    parser.add_argument("--seeds", type=str, default="100,101", help="Comma-separated seeds")
    parser.add_argument("--verify", action="store_true", default=True, help="Verify serialization after each run")
    args = parser.parse_args()

    if args.size == "all":
        sizes = [10, 25, 50, 75, 100]
    else:
        sizes = [int(args.size)]

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    for size in sizes:
        print(f"\n{'='*60}")
        print(f"  Running Instance{size} with {len(seeds)} seeds")
        print(f"{'='*60}")

        output_dir = BASE_OUTPUT
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"i{size}_best_by_instance.csv"

        existing = _read_existing_csv(csv_path)
        print(f"  Existing entries: {len(existing)}")

        instances = SIZE_CONFIGS[size]["instances"]
        all_rows = list(existing.values())

        for inst in instances:
            for seed in seeds:
                key = f"{inst}|{seed}"
                if key in existing:
                    print(f"  {inst} seed={seed}... SKIP")
                    continue

                print(f"  {inst} seed={seed}...", end=" ", flush=True)
                try:
                    row = run_instance(inst, size, seed)
                    if row["flexible_cost"] is not None:
                        print(f"flex={row['flexible_cost']:.2f} "
                              f"saving={row['flexible_saving_vs_same']:.2f}%",
                              end="")
                    else:
                        print("INFEASIBLE", end="")

                    if args.verify and row["flexible_cost"] is not None:
                        ok = verify_serialization(row)
                        if ok:
                            print(" ✓ser", end="")
                        else:
                            print(" ❌SER", end="")

                    # Persist after each run (incremental)
                    found = False
                    for i, existing_row in enumerate(all_rows):
                        if existing_row.get("instance_name") == inst and existing_row.get("flex_seed") == str(seed):
                            all_rows[i] = row
                            found = True
                            break
                    if not found:
                        all_rows.append(row)
                    _write_csv(csv_path, all_rows)
                    print()
                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback
                    traceback.print_exc()

        # Summary
        feasible = [r for r in all_rows if r.get("flexible_cost") and float(r["flexible_cost"]) > 0]
        by_inst: Dict[str, List[Dict[str, Any]]] = {}
        for r in feasible:
            by_inst.setdefault(r["instance_name"], []).append(r)

        print(f"\n  Summary Instance{size}:")
        print(f"  {'Instance':>12} | {'FlexBest':>8} | {'Saving':>7} | {'Seeds':>5}")
        print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*7}-+-{'-'*5}")
        flex_bests = []
        for inst in sorted(by_inst):
            flex_costs = [float(r["flexible_cost"]) for r in by_inst[inst]]
            same_costs = [float(r["same_cost"]) for r in by_inst[inst]]
            best = min(flex_costs)
            flex_bests.append(best)
            same_mean = np.mean(same_costs)
            saving = (same_mean - best) / same_mean * 100 if same_mean > 0 else 0
            print(f"  {inst:>12} | {best:>8.2f} | {saving:>+6.2f}% | {len(flex_costs):>3}/{len(seeds)}")

        if flex_bests:
            print(f"\n  Overall avg best: {np.mean(flex_bests):.2f}")
        print(f"\n  Saved to {csv_path}")


if __name__ == "__main__":
    main()
