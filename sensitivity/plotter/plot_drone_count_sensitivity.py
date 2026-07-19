#!/usr/bin/env python3
"""Plot drone count sensitivity as a single chart (battery-style)."""

from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot drone count sensitivity (single figure)")
    parser.add_argument(
        "--scale",
        type=str,
        default="Instance25",
        help="Scale to plot, e.g. Instance25 / Instance10 / temp_instance",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="Custom summary CSV path (default: results_new/drone_count/drone_count_summary.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PDF path (PNG will be saved with same stem)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom plot title (default: no title)",
    )
    return parser.parse_args()


def annotate_smart(
    ax,
    x_vals,
    y_vals,
    labels,
    *,
    color: str,
    used_positions: list[tuple[float, float]],
    dpi: float,
) -> None:
    """Annotate points with simple overlap-avoidance in display coordinates."""
    offsets_pt = [10, -12, 16, -18, 22, -24]
    x_gap_px = 22
    y_gap_px = 14
    px_per_pt = dpi / 72.0

    for x, y, label in zip(x_vals, y_vals, labels):
        x_px, y_px = ax.transData.transform((x, y))
        chosen_offset = offsets_pt[-1]

        for off in offsets_pt:
            candidate_y_px = y_px + off * px_per_pt
            collision = any(
                abs(x_px - ux) < x_gap_px and abs(candidate_y_px - uy) < y_gap_px
                for ux, uy in used_positions
            )
            if not collision:
                chosen_offset = off
                break

        va = "bottom" if chosen_offset > 0 else "top"
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(0, chosen_offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=10,
            fontweight="bold",
            color=color,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.2),
        )
        used_positions.append((x_px, y_px + chosen_offset * px_per_pt))


def expand_axis_limits(ax, values, *, pad_ratio: float = 0.12, min_pad: float = 0.5) -> None:
    """Add vertical headroom/footroom so annotations stay inside plot area."""
    vals = [float(v) for v in values]
    if not vals:
        return
    vmin = min(vals)
    vmax = max(vals)
    span = max(vmax - vmin, 1e-9)
    pad = max(span * pad_ratio, min_pad)
    ax.set_ylim(vmin - pad, vmax + pad)


def main() -> int:
    args = parse_args()

    sensitivity_dir = Path(__file__).resolve().parent.parent
    summary_csv = (
        Path(args.summary_csv)
        if args.summary_csv
        else sensitivity_dir / "results_new" / "drone_count" / "drone_count_summary.csv"
    )
    output_pdf = (
        Path(args.output)
        if args.output
        else sensitivity_dir / "results_new" / "drone_count" / "drone_count_sensitivity_plot_large.pdf"
    )
    output_png = output_pdf.with_suffix(".png")

    if not summary_csv.exists():
        print(f"Error: Data file not found at {summary_csv}")
        return 1

    try:
        df = pd.read_csv(summary_csv)
    except Exception as exc:
        print(f"Error reading CSV: {exc}")
        return 1

    available_scales = sorted(df["scale"].dropna().unique().tolist())
    if args.scale not in available_scales:
        print(f"Error: scale '{args.scale}' not found. Available: {available_scales}")
        return 1

    data = df[df["scale"] == args.scale].copy().sort_values("drone_count")
    if data.empty:
        print(f"Error: no rows found for scale '{args.scale}'")
        return 1

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Match battery plotting style
    bar_red_fill = "#F6D8E6"
    bar_red_border = "#E38D83"
    bar_blue_fill = "#CFEEF6"
    bar_blue_border = "#3886C2"
    color_cost = bar_blue_border
    color_drone = bar_red_border

    x_vals = data["drone_count"]
    y_cost = data["avg_cost_saving_vs_baseline"]
    y_drone = data["avg_best_drone_customers"]

    ax1.set_xlabel("Number of Drones", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Avg Cost Saving vs Baseline (%)", color=color_cost, fontsize=14, fontweight="bold")
    line1 = ax1.plot(
        x_vals,
        y_cost,
        color=color_cost,
        marker="s",
        markerfacecolor=bar_blue_fill,
        markeredgewidth=2,
        markersize=10,
        linewidth=3,
        label="Cost Saving (%)",
    )
    ax1.tick_params(axis="y", labelcolor=color_cost, labelsize=12)
    ax1.tick_params(axis="x", labelsize=12)
    ax1.set_xticks(x_vals)
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Avg Drone Served Customers", color=color_drone, fontsize=14, fontweight="bold")
    line2 = ax2.plot(
        x_vals,
        y_drone,
        color=color_drone,
        marker="o",
        markerfacecolor=bar_red_fill,
        markeredgewidth=2,
        markersize=10,
        linewidth=3,
        linestyle="--",
        label="Drone Customers",
    )
    ax2.tick_params(axis="y", labelcolor=color_drone, labelsize=12)

    # Slightly expand both y-axes to prevent label clipping at top/bottom.
    expand_axis_limits(ax1, y_cost, pad_ratio=0.12, min_pad=0.8)
    expand_axis_limits(ax2, y_drone, pad_ratio=0.12, min_pad=0.6)

    # Draw first so transforms are stable, then place labels with collision avoidance
    fig.canvas.draw()
    used_positions: list[tuple[float, float]] = []
    annotate_smart(
        ax1,
        x_vals,
        y_cost,
        [f"{y:.2f}%" for y in y_cost],
        color=color_cost,
        used_positions=used_positions,
        dpi=fig.dpi,
    )
    annotate_smart(
        ax2,
        x_vals,
        y_drone,
        [f"{y:.2f}" for y in y_drone],
        color=color_drone,
        used_positions=used_positions,
        dpi=fig.dpi,
    )

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", frameon=True, shadow=True, fontsize=12)

    if args.title:
        ax1.set_title(args.title, fontsize=16, pad=15)

    plt.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_pdf, dpi=300)
    plt.savefig(output_png, dpi=300)
    print(f"Plot saved to: {output_pdf}")
    print(f"Plot saved to: {output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
