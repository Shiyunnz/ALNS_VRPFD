#!/usr/bin/env python3
"""ALNS Parameter Sweep: test key parameters on R_30_25_1 with 3 seeds."""
import csv
import math
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from alns_vrpfd.core.sa import SANNCfg, SimulatedAnnealingALNS
from alns_vrpfd.evaluation import Evaluator
from alns_vrpfd.model.initializer import build_two_phase_initial_solution
from alns_vrpfd.utils.config_loader import ALNSConfig
from alns_vrpfd.utils.io_utils import read_instance
from run_alns import build_operators, infer_size

INSTANCE_FILE = "data/Instance25/R_30_25_1.txt"
BATTERY = 6.3
ITERATIONS = 2000
SEEDS = [42, 123, 777]

# ── Configurations to test ──────────────────────────────────────────────
# Format: (name, {param_overrides})
CONFIGS = [
    # 0. Current P0 baseline
    ("baseline_p0", {}),

    # ── Local search frequency ──
    ("ls_freq_10", {"local_search_frequency": 10}),
    ("ls_freq_15", {"local_search_frequency": 15}),
    ("ls_freq_20", {"local_search_frequency": 20}),
    ("ls_off", {"local_search_frequency": 0}),

    # ── Intensify frequency ──
    ("intensify_50", {"intensify_frequency": 50}),
    ("intensify_100", {"intensify_frequency": 100}),
    ("intensify_off", {"intensify_frequency": 0}),

    # ── Path relinking ──
    ("pr_0.10", {"path_relinking_prob": 0.10}),
    ("pr_off", {"path_relinking_prob": 0.0}),

    # ── Cross exchange ──
    ("cx_off", {"cross_exchange_prob": 0.0}),

    # ── Combined speed configs ──
    ("fast_A", {
        "local_search_frequency": 15,
        "intensify_frequency": 100,
        "path_relinking_prob": 0.0,
        "cross_exchange_prob": 0.0,
    }),
    ("fast_B", {
        "local_search_frequency": 10,
        "intensify_frequency": 50,
        "path_relinking_prob": 0.10,
        "cross_exchange_prob": 0.0,
    }),
    ("fast_C", {
        "local_search_frequency": 20,
        "intensify_frequency": 0,
        "path_relinking_prob": 0.0,
        "cross_exchange_prob": 0.0,
    }),

    # ── Reheat tuning ──
    ("reheat_stall_300", {"reheat_stall_trigger": 300, "reheat_cooldown": 120}),
    ("reheat_stall_600", {"reheat_stall_trigger": 600, "reheat_cooldown": 250}),

    # ── LS on new best ──
    ("ls_no_best", {"local_search_on_new_best": False}),
]


def run_single(config_name: str, overrides: dict, seed: int) -> dict:
    config = ALNSConfig("config/alns_config.yaml")
    instance = read_instance(INSTANCE_FILE, strategy="class_based")
    if "drone" in instance.vehicle_specs:
        instance.vehicle_specs["drone"].endurance = float("inf")
    instance.configure_robustness(
        drone_battery_capacity=BATTERY,
        energy_uncertainty_budget=config.energy_uncertainty_budget,
        energy_deviation_rate=config.energy_deviation_rate,

        same_truck_retrieval=config.same_truck_retrieval,
    )
    evaluator = Evaluator(
        instance,
        rendezvous_tolerance=config.drone_rendezvous_tolerance,
        forced_drone_customers=config.forced_drone_customers,
        allow_multiple_launch_per_node=config.relax_allow_multiple_launch_per_node,
    )
    master_rng = random.Random(seed)
    destroy_ops, repair_ops = build_operators(
        instance, seed, config.drone_priority,
        repair_set="new", enable_composite=False,
        master_rng=master_rng, drone_bonus_kwargs=None,
        forced_drone_customers=config.forced_drone_customers,
    )
    sa_cfg_dict = config.build_sa_config_dict(
        size=infer_size(instance))
    sa_cfg_dict["iterations"] = ITERATIONS
    sa_cfg_dict["log_operator_metrics"] = False

    # Apply overrides
    for k, v in overrides.items():
        sa_cfg_dict[k] = v

    sa_cfg = SANNCfg(**sa_cfg_dict)
    initial_solution = build_two_phase_initial_solution(
        instance,
        allow_multiple_launch_per_node=config.relax_allow_multiple_launch_per_node,
    )

    t0 = time.perf_counter()
    try:
        sa = SimulatedAnnealingALNS(
            instance=instance,
            destroy_ops=destroy_ops,
            repair_ops=repair_ops,
            evaluator=evaluator,
            cfg=sa_cfg,
            rng=master_rng,
        )
        best = sa.run(initial_solution)
        details = evaluator.evaluate_with_details(best)
        cost = float(details.result.total_cost)
        feasible = bool(details.result.feasible)
        drone_custs = int(sum(len(t.customers()) for t in best.drone_tasks))
    except Exception as exc:
        cost = float("inf")
        feasible = False
        drone_custs = 0
    elapsed = time.perf_counter() - t0

    return {
        "config": config_name,
        "seed": seed,
        "cost": cost,
        "feasible": feasible,
        "drone_customers": drone_custs,
        "runtime": round(elapsed, 2),
    }


def main():
    results_dir = Path("sensitivity/results_new/param_sweep")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "sweep_results.csv"
    fieldnames = ["config", "seed", "cost", "feasible", "drone_customers", "runtime"]

    total = len(CONFIGS) * len(SEEDS)
    print(f"Parameter sweep: {len(CONFIGS)} configs × {len(SEEDS)} seeds = {total} runs")
    print(f"Instance: {INSTANCE_FILE}, battery={BATTERY}, iters={ITERATIONS}")
    print("=" * 80)

    all_results = []
    for cfg_idx, (name, overrides) in enumerate(CONFIGS):
        for s_idx, seed in enumerate(SEEDS):
            run_num = cfg_idx * len(SEEDS) + s_idx + 1
            print(f"[{run_num}/{total}] {name} seed={seed} ... ", end="", flush=True)
            result = run_single(name, overrides, seed)
            all_results.append(result)
            print(f"cost={result['cost']:.2f}  time={result['runtime']}s  drones={result['drone_customers']}")

    # Write CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Config':<20} {'Avg Cost':>10} {'Min Cost':>10} {'Max Cost':>10} {'Spread':>8} {'Avg Time':>10}")
    print("-" * 80)

    from collections import defaultdict
    by_config = defaultdict(list)
    for r in all_results:
        by_config[r["config"]].append(r)

    for name, _ in CONFIGS:
        runs = by_config[name]
        costs = [r["cost"] for r in runs if math.isfinite(r["cost"])]
        times = [r["runtime"] for r in runs]
        if costs:
            avg_c = sum(costs) / len(costs)
            min_c = min(costs)
            max_c = max(costs)
            spread = max_c - min_c
        else:
            avg_c = min_c = max_c = spread = float("inf")
        avg_t = sum(times) / len(times)
        print(f"{name:<20} {avg_c:>10.2f} {min_c:>10.2f} {max_c:>10.2f} {spread:>8.2f} {avg_t:>10.2f}s")

    print(f"\nResults saved to: {out_csv}")


if __name__ == "__main__":
    main()
