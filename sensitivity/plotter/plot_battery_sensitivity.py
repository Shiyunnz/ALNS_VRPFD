#!/usr/bin/env python3
"""Plot battery sensitivity from unified summary CSV."""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


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

        va = 'bottom' if chosen_offset > 0 else 'top'
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(0, chosen_offset),
            textcoords='offset points',
            ha='center',
            va=va,
            fontsize=10,
            fontweight='bold',
            color=color,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=0.2),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot battery sensitivity from summary CSV.")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path(
            "sensitivity/results_new/battery_sensitivity_rerun_in_25/battery_sensitivity_15inst_summary.csv"
        ),
        help="Input summary CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "sensitivity/results_new/battery_sensitivity_rerun_in_25/battery_sensitivity_15inst_final.pdf"
        ),
        help="Output PDF path; PNG will be written with same basename.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_csv = args.summary_csv
    output_path = args.output
    output_png = output_path.with_suffix(".png")

    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    df = pd.read_csv(summary_csv)

    if "battery_capacity" in df.columns:
        x_col = "battery_capacity"
    elif "battery" in df.columns:
        x_col = "battery"
    else:
        raise KeyError("Battery column not found. Expected 'battery_capacity' or 'battery'.")

    if "avg_cost_saving_vs_baseline" in df.columns:
        y_cost_col = "avg_cost_saving_vs_baseline"
    elif "saving_pct" in df.columns:
        y_cost_col = "saving_pct"
    else:
        raise KeyError("Cost-saving column not found. Expected 'avg_cost_saving_vs_baseline' or 'saving_pct'.")

    if "avg_best_drone_customers" in df.columns:
        y_drone_col = "avg_best_drone_customers"
    elif "drones" in df.columns:
        y_drone_col = "drones"
    else:
        raise KeyError("Drone-customers column not found. Expected 'avg_best_drone_customers' or 'drones'.")

    df = df.sort_values(x_col)

    # Setup plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Define colors from TikZ style (User template)
    barRedFill = '#F6D8E6'
    barRedBorder = '#E38D83'
    barBlueFill = '#CFEEF6'
    barBlueBorder = '#3886C2'

    # Map to variables (Cost Saving = Blue, Drone Nodes = Red)
    color_cost = barBlueBorder
    color_drone = barRedBorder

    x_vals = df[x_col]
    x_vals = x_vals.astype(float)

    # Unified x-axis display format:
    # - If data is 4.3/5.3/... then display as 0.43/0.53/...
    # - If data is already 0.43/0.53/... keep it unchanged.
    if float(x_vals.max()) > 2.0:
        x_tick_labels = [f"{x / 10.0:.2f}" for x in x_vals]
    else:
        x_tick_labels = [f"{x:.2f}" for x in x_vals]

    # --- Axis 1 (Left): Cost Saving ---
    ax1.set_xlabel('Battery Capacity', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Avg Cost Saving vs Baseline (%)', color=color_cost, fontsize=14, fontweight='bold')

    # Plot Cost Saving (Square markers, Blue)
    l1 = ax1.plot(x_vals, df[y_cost_col], color=color_cost, marker='s',
                  markerfacecolor=barBlueFill, markeredgewidth=2, markersize=10,
                  linewidth=3, label='Cost Saving (%)')

    ax1.tick_params(axis='y', labelcolor=color_cost, labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.set_xticks(x_vals)  # Explicit ticks
    ax1.set_xticklabels(x_tick_labels)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Axis 2 (Right): Drone Nodes ---
    ax2 = ax1.twinx()
    ax2.set_ylabel('Avg Drone Served Nodes', color=color_drone, fontsize=14, fontweight='bold')

    # Plot Drone Nodes (Circle markers, Red, Dashed)
    l2 = ax2.plot(x_vals, df[y_drone_col], color=color_drone, marker='o',
                  markerfacecolor=barRedFill, markeredgewidth=2, markersize=10,
                  linewidth=3, linestyle='--', label='Drone Nodes')

    ax2.tick_params(axis='y', labelcolor=color_drone, labelsize=12)

    # Expand y-limits to keep labels inside frame
    expand_axis_limits(ax1, df[y_cost_col], pad_ratio=0.12, min_pad=0.8)
    expand_axis_limits(ax2, df[y_drone_col], pad_ratio=0.12, min_pad=0.6)

    # Draw first so transforms are stable, then place labels with collision avoidance
    fig.canvas.draw()
    used_positions: list[tuple[float, float]] = []
    annotate_smart(
        ax1,
        x_vals,
        df[y_cost_col],
        [f"{y:.2f}%" for y in df[y_cost_col]],
        color=color_cost,
        used_positions=used_positions,
        dpi=fig.dpi,
    )
    annotate_smart(
        ax2,
        x_vals,
        df[y_drone_col],
        [f"{y:.2f}" for y in df[y_drone_col]],
        color=color_drone,
        used_positions=used_positions,
        dpi=fig.dpi,
    )

    # --- Title & Legend ---
    # Combine legends
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', frameon=True, shadow=True, fontsize=12)

    # Layout adjustment
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.savefig(output_png, dpi=300)
    print(f"Plot saved to: {output_path}")
    print(f"Plot saved to: {output_png}")


if __name__ == "__main__":
    main()
