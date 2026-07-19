#!/usr/bin/env python3
"""Single-panel publication-style figure for docking flexibility (instance-level, t5)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot instance-level docking flexibility in one panel (bars + saving points)."
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="sensitivity/results_new/drone_flexibility/docking_flexibility_region_vertical_t5_summary.csv",
        help="Instance-level summary CSV path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sensitivity/results_new/drone_flexibility/docking_flexibility_instance_singlepanel_t5.pdf",
        help="Output PDF path (PNG with same stem).",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="region_group",
        choices=["region_group", "saving_desc"],
        help="Instance display order.",
    )
    return parser.parse_args()


def expand_axis_limits(ax, values, *, pad_ratio: float = 0.12, min_pad: float = 0.5) -> None:
    vals = [float(v) for v in values]
    if not vals:
        return
    vmin = min(vals)
    vmax = max(vals)
    span = max(vmax - vmin, 1e-9)
    pad = max(span * pad_ratio, min_pad)
    ax.set_ylim(vmin - pad, vmax + pad)


def annotate_points_smart(ax, x_vals, y_vals, labels, color: str) -> None:
    offsets = [10, -12, 14, -16, 18, -20]
    sorted_idx = sorted(range(len(y_vals)), key=lambda i: y_vals[i], reverse=True)
    for i, (x, y, lbl) in enumerate(zip(x_vals, y_vals, labels)):
        rank = sorted_idx.index(i)
        if rank == 0 and y > 18:
            off = -18
            ha = "center"
        elif rank <= 2 and y > 15:
            off = offsets[i % len(offsets)]
            ha = "center" if abs(off) < 16 else "center"
        else:
            off = offsets[i % len(offsets)]
            ha = "center"
        va = "bottom" if off > 0 else "top"
        ax.annotate(
            lbl,
            xy=(x, y),
            xytext=(20 if rank == 0 and y > 18 else 0, off),
            textcoords="offset points",
            ha="left" if rank == 0 and y > 18 else ha,
            va=va,
            fontsize=9,
            fontweight="bold",
            color=color,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.2),
        )


def main() -> int:
    args = parse_args()

    summary_csv = Path(args.summary_csv)
    output_pdf = Path(args.output)
    output_png = output_pdf.with_suffix(".png")

    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    df = pd.read_csv(summary_csv)
    required = {
        "region",
        "group",
        "avg_same_cost",
        "avg_flexible_cost",
        "avg_flexible_saving_vs_same",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in summary CSV: {sorted(missing)}")

    if args.sort_by == "saving_desc":
        df = df.sort_values(["avg_flexible_saving_vs_same", "region", "group"], ascending=[False, True, True]).reset_index(drop=True)
    else:
        df = df.sort_values(["region", "group"]).reset_index(drop=True)

    x = np.arange(len(df))
    labels = df["group"].tolist()
    same_cost = df["avg_same_cost"].to_numpy(dtype=float)
    flex_cost = df["avg_flexible_cost"].to_numpy(dtype=float)
    saving = df["avg_flexible_saving_vs_same"].to_numpy(dtype=float)

    # Global red/blue palette: deep border + light fill.
    red_border = "#E38DB3"
    red_fill = "#F6DBE6"
    blue_border = "#3886C2"
    blue_fill = "#CFECF6"
    # Saving markers/lines and average reference line use the same auxiliary palette color.
    color_save = "#4EAC97"
    color_avg = "#4EAC97"

    fig, ax1 = plt.subplots(figsize=(16, 6.8))
    width = 0.38

    bars_same = ax1.bar(
        x - width / 2,
        same_cost,
        width=width,
        color=red_fill,
        edgecolor=red_border,
        linewidth=1.8,
        alpha=1.0,
        label="Fixed Docking",
    )
    bars_flex = ax1.bar(
        x + width / 2,
        flex_cost,
        width=width,
        color=blue_fill,
        edgecolor=blue_border,
        linewidth=1.8,
        alpha=1.0,
        label="Flexible Docking",
    )
    ax1.set_xlabel("Instance", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Average Cost", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax1.tick_params(axis="y", labelsize=11)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    if args.sort_by == "region_group":
        # Region separators (R30 | R40 | R50), each has 5 instances.
        for xpos in (4.5, 9.5):
            ax1.axvline(x=xpos, color="#999999", linestyle="--", linewidth=1.0, alpha=0.75)
        # Region labels on top partition axis (avoid overlap with instance labels).
        ax_top = ax1.secondary_xaxis("top")
        ax_top.set_xticks([2, 7, 12])
        ax_top.set_xticklabels(["R30", "R40", "R50"])
        ax_top.tick_params(axis="x", labelsize=11, pad=8, length=0)
        for tick in ax_top.get_xticklabels():
            tick.set_fontweight("bold")

    ax2 = ax1.twinx()
    stems = ax2.vlines(
        x,
        0.0,
        saving,
        color=color_save,
        linewidth=1.2,
        alpha=0.6,
        label="Cost Saving (%)",
    )
    ax2.scatter(
        x,
        saving,
        edgecolors=color_save,
        c=color_save,
        marker="o",
        s=52,
        facecolors="white",
        linewidths=1.8,
        zorder=3,
    )
    ax2.set_ylabel("Cost Saving (%)", color=color_save, fontsize=14, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_save, labelsize=11)
    ax2.axhline(0, color="#777777", linestyle=":", linewidth=1.0, alpha=0.8)
    avg_saving = float(np.mean(saving))
    avg_line = ax2.axhline(
        avg_saving,
        color=color_avg,
        linestyle="--",
        linewidth=2.0,
        alpha=0.85,
        label="Average Saving",
    )

    expand_axis_limits(ax1, np.r_[same_cost, flex_cost], pad_ratio=0.10, min_pad=8.0)
    expand_axis_limits(ax2, saving, pad_ratio=0.14, min_pad=1.5)
    annotate_points_smart(ax2, x, saving, [f"{v:.1f}%" for v in saving], color=color_save)
    ax2.annotate(
        f"Avg {avg_saving:.2f}%",
        xy=(len(x) - 0.15, avg_saving),
        xytext=(-8, 6),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color=color_avg,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=0.2),
    )

    # Reserve extra whitespace at upper-left for legend and labels.
    y1_min, y1_max = ax1.get_ylim()
    ax1.set_ylim(y1_min, y1_max + (y1_max - y1_min) * 0.18)
    y2_min, y2_max = ax2.get_ylim()
    ax2.set_ylim(y2_min, y2_max + (y2_max - y2_min) * 0.15)

    # Combined legend in upper-left, single panel style.
    handles = [bars_same, bars_flex, stems, avg_line]
    labels_legend = [h.get_label() for h in handles]
    ax1.legend(
        handles,
        labels_legend,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.975),
        borderaxespad=0.6,
        frameon=True,
        shadow=True,
        fontsize=11,
    )

    plt.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_pdf}")
    print(f"Plot saved to: {output_png}")
    print(f"Average cost saving: {avg_saving:.4f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
