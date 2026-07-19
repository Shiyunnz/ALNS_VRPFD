#!/usr/bin/env python3
"""Run ALNS on one or more instances and produce convergence/operator analysis plots.

Usage:
    # Single instance
    python run_convergence_analysis.py data/Instance10/R_30_10_1.txt

    # Multiple instances
    python run_convergence_analysis.py data/Instance10/R_30_10_1.txt data/Instance25/R_30_25_1.txt

    # With custom output dir & iterations
    python run_convergence_analysis.py data/Instance10/R_30_10_1.txt --iterations 3000 --output-dir results/convergence
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from run_alns import build_operators, infer_size
from alns_vrpfd.core.sa import SANNCfg, SimulatedAnnealingALNS
from alns_vrpfd.evaluation import Evaluator
from alns_vrpfd.model.initializer import build_two_phase_initial_solution
from alns_vrpfd.utils.config_loader import ALNSConfig
from alns_vrpfd.utils.io_utils import read_instance


def export_convergence_csv(history: list[dict], path: Path) -> None:
    if not history:
        return
    keys = ["iteration", "current_cost", "best_cost", "temperature", "destroy", "repair"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)
    print(f"  Convergence CSV: {path}")


def export_usage_csv(usage: dict[str, int], path: Path) -> None:
    rows = [{"operator": k, "uses": v} for k, v in sorted(usage.items(), key=lambda x: -x[1])]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["operator", "uses"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Usage CSV: {path}")


def export_weights_csv(weight_history: list[dict], path: Path) -> None:
    if not weight_history:
        return
    keys = list(weight_history[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(weight_history)
    print(f"  Weights CSV: {path}")


def run_single(instance_path: str, config: ALNSConfig, iterations: int,
               seed: int, output_dir: Path, max_plot_iter: int | None = None) -> None:
    inst_name = Path(instance_path).stem
    inst_dir = output_dir / inst_name
    inst_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running ALNS on {inst_name}  (iterations={iterations}, seed={seed})")
    print(f"{'='*60}")

    instance = read_instance(instance_path, strategy=config.time_window_strategy)
    if "drone" in instance.vehicle_specs:
        instance.vehicle_specs["drone"].endurance = float("inf")

    search_gamma = config.energy_uncertainty_budget
    instance.configure_robustness(
        drone_battery_capacity=config.drone_battery_capacity,
        energy_uncertainty_budget=search_gamma,
        energy_deviation_rate=config.energy_deviation_rate,

        same_truck_retrieval=config.same_truck_retrieval,
    )

    forced_drone_customers = config.forced_drone_customers or []
    evaluator = Evaluator(
        instance,
        rendezvous_tolerance=config.drone_rendezvous_tolerance,
        forced_drone_customers=forced_drone_customers,
        allow_multiple_launch_per_node=config.relax_allow_multiple_launch_per_node,
    )

    sa_config_dict = config.build_sa_config_dict()
    sa_config_dict["iterations"] = iterations
    sa_config_dict["size"] = infer_size(instance)
    sa_config_dict["log_operator_metrics"] = False
    sa_cfg = SANNCfg(**sa_config_dict)

    rng = random.Random(seed)
    destroy_ops, repair_ops = build_operators(
        instance, seed,
        drone_priority=config.drone_priority,
        repair_set="all",
        enable_composite=True,
        drone_bonus_kwargs=config.drone_bonus,
        forced_drone_customers=forced_drone_customers,
        robust_energy_mode="embedded",
    )

    initial_solution = build_two_phase_initial_solution(
        instance,
        truck_forbidden_customers=forced_drone_customers,
        allow_multiple_launch_per_node=config.relax_allow_multiple_launch_per_node,
    )

    alns = SimulatedAnnealingALNS(
        instance=instance,
        destroy_ops=destroy_ops,
        repair_ops=repair_ops,
        evaluator=evaluator,
        cfg=sa_cfg,
        rng=rng,
    )

    t0 = time.time()
    best_sol = alns.run(initial_solution)
    runtime = time.time() - t0

    eval_res = evaluator.evaluate_solution(best_sol)
    print(f"\nResult: cost={eval_res.total_cost:.2f}, feasible={eval_res.feasible}, time={runtime:.1f}s")

    # Export CSVs
    conv_csv = inst_dir / "convergence.csv"
    usage_csv = inst_dir / "usage.csv"
    weights_csv = inst_dir / "weights.csv"
    export_convergence_csv(alns.convergence_history, conv_csv)
    export_usage_csv(alns.operator_usage, usage_csv)
    export_weights_csv(alns.operator_weight_history, weights_csv)

    # Generate plots
    from sensitivity.plotter.plot_alns_convergence import (
        plot_convergence, plot_operator_frequency, plot_weight_evolution,
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from collections import Counter

    conv_df = pd.read_csv(conv_csv)
    weights_df = pd.read_csv(weights_csv)

    # Truncate to max_plot_iter if set
    if max_plot_iter is not None and max_plot_iter > 0:
        conv_df = conv_df[conv_df["iteration"] <= max_plot_iter].copy()
        weights_df = weights_df[weights_df["iteration"] <= max_plot_iter].copy()

    # Recompute operator usage from (possibly truncated) convergence data
    if "destroy" in conv_df.columns and "repair" in conv_df.columns:
        counter: Counter[str] = Counter()
        counter.update(conv_df["destroy"])
        counter.update(conv_df["repair"])
        usage_df = pd.DataFrame(
            [{"operator": k, "uses": v} for k, v in counter.most_common()]
        )
    else:
        usage_df = pd.read_csv(usage_csv)

    fig = plt.figure(figsize=(16, 14), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1])

    ax_conv = fig.add_subplot(gs[0, :])
    plot_convergence(conv_df, ax_conv)

    ax_freq = fig.add_subplot(gs[1, :])
    plot_operator_frequency(usage_df, ax_freq)

    ax_wd = fig.add_subplot(gs[2, 0])
    ax_wr = fig.add_subplot(gs[2, 1])
    plot_weight_evolution(weights_df, ax_wd, ax_wr)

    fig.suptitle(f"ALNS Convergence Analysis — {inst_name}",
                 fontsize=16, fontweight="bold", y=1.01)

    tag = f"_{max_plot_iter}iter" if max_plot_iter else ""
    for ext in ("pdf", "png"):
        out = inst_dir / f"alns_analysis{tag}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {inst_dir / f'alns_analysis{tag}.pdf'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ALNS convergence analysis")
    parser.add_argument("instances", nargs="+", help="Instance file paths")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Override iteration count")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-plot-iter", type=int, default=None,
                        help="Truncate plots to first N iterations")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/convergence_analysis"))
    args = parser.parse_args()

    config = ALNSConfig("config/alns_config.yaml")
    iterations = args.iterations or config.iterations_default

    for inst_path in args.instances:
        if not os.path.exists(inst_path):
            print(f"Warning: {inst_path} not found, skipping.")
            continue
        run_single(inst_path, config, iterations, args.seed, args.output_dir,
                   max_plot_iter=args.max_plot_iter)

    print("\nAll done.")


if __name__ == "__main__":
    main()
