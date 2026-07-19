#!/usr/bin/env python3
"""Plot ALNS convergence curve, operator frequency table, and operator weight evolution.

Usage:
    python plot_alns_convergence.py --convergence-csv conv.csv --weights-csv weights.csv --usage-csv usage.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- TikZ-consistent colour palette (same as sensitivity plots) ----
_BLUE_BORDER = '#3886C2'
_BLUE_FILL = '#CFEEF6'
_RED_BORDER = '#E38D83'
_RED_FILL = '#F6D8E6'
_GREEN = '#2ca02c'
_ORANGE = '#D4A76A'

# Readable operator name mapping
OPERATOR_LABELS: dict[str, str] = {
    # Destroy
    "DestroyRandom": "Random",
    "DestroyWorstDistance": "Worst Distance",
    "DestroyShaw": "Shaw",
    "DestroySegmentShuffle": "Segment Shuffle",
    "DestroyLargeRandom": "Large Random",
    # Repair
    "RepairCheapest": "Cheapest Insert",
    "RepairRegret": "Regret-k",
    "RepairBiasedRandomized": "Biased Random",
    "RepairDronePriorityRegret": "Drone Priority",
    "RepairTruckFirst": "Truck First",
    "RepairEqualPriority": "Equal Priority",
}


def _label(name: str) -> str:
    return OPERATOR_LABELS.get(name, name)


# ------------------------------------------------------------------ #
#  1) Convergence Curve  (best cost only)
# ------------------------------------------------------------------ #
def plot_convergence(df: pd.DataFrame, ax: plt.Axes) -> None:
    iters = df["iteration"]
    best = df["best_cost"]

    ax.plot(iters, best, color=_BLUE_BORDER,
            linewidth=3, label="Best Cost")

    ax.set_xlabel("Iteration", fontsize=14, fontweight="bold")
    ax.set_ylabel("Best Cost", color=_BLUE_BORDER, fontsize=14, fontweight="bold")
    ax.tick_params(axis="y", labelcolor=_BLUE_BORDER, labelsize=12)
    ax.tick_params(axis="x", labelsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Vertical headroom
    best_init = best.iloc[0]
    best_final = best.iloc[-1]
    span = max(best_init - best_final, 1e-9)
    ax.set_ylim(best_final - span * 0.08, best_init + span * 0.15)

    # Secondary y-axis for temperature
    ax2 = ax.twinx()
    ax2.plot(iters, df["temperature"], color=_RED_BORDER,
             linewidth=3, linestyle="--", label="Temperature")
    ax2.set_ylabel("Temperature", color=_RED_BORDER, fontsize=14, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=_RED_BORDER, labelsize=12)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              loc="upper right", frameon=True, shadow=True, fontsize=12)


# ------------------------------------------------------------------ #
#  2) Operator Frequency Bar Chart
# ------------------------------------------------------------------ #
def plot_operator_frequency(usage_df: pd.DataFrame, ax: plt.Axes) -> None:
    usage_df = usage_df.sort_values("uses", ascending=True)
    names = [_label(n) for n in usage_df["operator"]]
    uses = usage_df["uses"].values
    total = uses.sum()
    pct = uses / total * 100

    is_destroy = [n.startswith("Destroy") for n in usage_df["operator"]]
    colors = [_BLUE_BORDER if d else _GREEN for d in is_destroy]
    edge_colors = [_BLUE_BORDER if d else _GREEN for d in is_destroy]
    fill_colors = [_BLUE_FILL if d else '#d5f5e3' for d in is_destroy]

    bars = ax.barh(names, uses, color=fill_colors, edgecolor=edge_colors, linewidth=1.5)
    max_uses = uses.max()
    for bar, p in zip(bars, pct):
        w = bar.get_width()
        ax.text(w - max_uses * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{p:.1f}%", va="center", ha="right", fontsize=11, fontweight="bold")

    ax.set_xlabel("Number of Uses", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(True, axis="x", linestyle="--", alpha=0.6)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=_BLUE_FILL, edgecolor=_BLUE_BORDER, linewidth=1.5, label="Destroy"),
                       Patch(facecolor='#d5f5e3', edgecolor=_GREEN, linewidth=1.5, label="Repair")],
              loc="lower right", frameon=True, shadow=True, fontsize=12)


# ------------------------------------------------------------------ #
#  3) Operator Weight Evolution
# ------------------------------------------------------------------ #
def plot_weight_evolution(weights_df: pd.DataFrame, ax_d: plt.Axes, ax_r: plt.Axes) -> None:
    iters = weights_df["iteration"]
    destroy_cols = [c for c in weights_df.columns if c.startswith("d_")]
    repair_cols = [c for c in weights_df.columns if c.startswith("r_")]

    d_colors = plt.cm.Set2(np.linspace(0, 0.8, max(len(destroy_cols), 1)))
    r_colors = plt.cm.Set1(np.linspace(0, 0.8, max(len(repair_cols), 1)))

    for i, col in enumerate(destroy_cols):
        ax_d.plot(iters, weights_df[col], label=_label(col[2:]),
                  linewidth=2, color=d_colors[i])
    ax_d.set_xlabel("Iteration", fontsize=14, fontweight="bold")
    ax_d.set_ylabel("Weight", fontsize=14, fontweight="bold")
    ax_d.set_title("Destroy Operator Weights", fontsize=14, fontweight="bold")
    ax_d.legend(fontsize=10, loc="best", frameon=True, shadow=True)
    ax_d.tick_params(labelsize=12)
    ax_d.grid(True, linestyle="--", alpha=0.6)

    for i, col in enumerate(repair_cols):
        ax_r.plot(iters, weights_df[col], label=_label(col[2:]),
                  linewidth=2, color=r_colors[i])
    ax_r.set_xlabel("Iteration", fontsize=14, fontweight="bold")
    ax_r.set_ylabel("Weight", fontsize=14, fontweight="bold")
    ax_r.set_title("Repair Operator Weights", fontsize=14, fontweight="bold")
    ax_r.legend(fontsize=10, loc="best", frameon=True, shadow=True)
    ax_r.tick_params(labelsize=12)
    ax_r.grid(True, linestyle="--", alpha=0.6)


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #
def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ALNS convergence & operator analysis.")
    parser.add_argument("--convergence-csv", type=Path, required=True,
                        help="CSV with iteration, current_cost, best_cost, temperature columns.")
    parser.add_argument("--usage-csv", type=Path, required=True,
                        help="CSV with operator, uses columns.")
    parser.add_argument("--weights-csv", type=Path, required=True,
                        help="CSV with iteration and d_*/r_* weight columns.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output file path (PDF/PNG). Default: next to convergence CSV.")
    parser.add_argument("--instance-name", type=str, default="",
                        help="Instance name for plot suptitle.")
    args = parser.parse_args()

    conv_df = pd.read_csv(args.convergence_csv)
    usage_df = pd.read_csv(args.usage_csv)
    weights_df = pd.read_csv(args.weights_csv)

    fig = plt.figure(figsize=(16, 14), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1])

    ax_conv = fig.add_subplot(gs[0, :])
    plot_convergence(conv_df, ax_conv)

    ax_freq = fig.add_subplot(gs[1, :])
    plot_operator_frequency(usage_df, ax_freq)

    ax_wd = fig.add_subplot(gs[2, 0])
    ax_wr = fig.add_subplot(gs[2, 1])
    plot_weight_evolution(weights_df, ax_wd, ax_wr)

    if args.instance_name:
        fig.suptitle(f"ALNS Analysis — {args.instance_name}",
                     fontsize=16, fontweight="bold", y=1.01)

    out = args.output or args.convergence_csv.parent / "alns_convergence_analysis.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    png_out = out.with_suffix(".png")
    fig.savefig(png_out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}  &  {png_out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
