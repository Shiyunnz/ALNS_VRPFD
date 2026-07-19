#!/usr/bin/env python3
"""Plot theta (energy deviation rate) sensitivity from summary CSV."""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Set paths
current_dir = Path(__file__).parent.parent  # Go up to sensitivity folder
data_path = current_dir / "results_new" / "theta_sensitivity" / "theta_summary.csv"
output_pdf = current_dir / "results_new" / "theta_sensitivity" / "theta_sensitivity_plot.pdf"
output_png = output_pdf.with_suffix(".png")

# Check file
if not data_path.exists():
    print(f"Error: Data file not found at {data_path}")
    sys.exit(1)

# Load data
df = pd.read_csv(data_path)
df.sort_values("theta", inplace=True)

# Use Instance25 data
df_plot = df[df['scale'] == 'Instance25'].copy()
if df_plot.empty:
    df_plot = df.copy()

# Convert cost_increase to cost_saving (negate: higher theta = higher cost = negative saving)
df_plot["avg_cost_saving_vs_baseline"] = -df_plot["avg_cost_increase_vs_baseline"]


def annotate_smart(ax, x_vals, y_vals, labels, *, color, used_positions, dpi):
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


def expand_axis_limits(ax, values, *, pad_ratio=0.12, min_pad=0.5):
    """Add vertical headroom/footroom so annotations stay inside plot area."""
    vals = [float(v) for v in values]
    if not vals:
        return
    vmin = min(vals)
    vmax = max(vals)
    span = max(vmax - vmin, 1e-9)
    pad = max(span * pad_ratio, min_pad)
    ax.set_ylim(vmin - pad, vmax + pad)


# Define colors (matching battery sensitivity style)
barRedFill = '#F6D8E6'
barRedBorder = '#E38D83'
barBlueFill = '#CFEEF6'
barBlueBorder = '#3886C2'

color_cost = barBlueBorder
color_drone = barRedBorder

# Setup plot — same figsize as battery sensitivity
fig, ax1 = plt.subplots(figsize=(10, 6))

x_vals = df_plot["theta"]

# --- Axis 1 (Left): Cost Saving ---
ax1.set_xlabel('Theta (Energy Deviation Rate)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Avg Cost Saving vs Baseline (%)', color=color_cost, fontsize=14, fontweight='bold')

l1 = ax1.plot(x_vals, df_plot["avg_cost_saving_vs_baseline"],
              color=color_cost, marker='s',
              markerfacecolor=barBlueFill, markeredgewidth=2, markersize=10,
              linewidth=3, label='Cost Saving (%)')

ax1.tick_params(axis='y', labelcolor=color_cost, labelsize=12)
ax1.tick_params(axis='x', labelsize=12)
ax1.set_xticks(x_vals)
ax1.grid(True, linestyle='--', alpha=0.6)

# --- Axis 2 (Right): Drone Customers ---
ax2 = ax1.twinx()
ax2.set_ylabel('Avg Drone Served Customers', color=color_drone, fontsize=14, fontweight='bold')

l2 = ax2.plot(x_vals, df_plot["avg_best_drone_customers"],
              color=color_drone, marker='o',
              markerfacecolor=barRedFill, markeredgewidth=2, markersize=10,
              linewidth=3, linestyle='--', label='Drone Customers')

ax2.tick_params(axis='y', labelcolor=color_drone, labelsize=12)

# Expand y-limits to keep labels inside frame
expand_axis_limits(ax1, df_plot["avg_cost_saving_vs_baseline"], pad_ratio=0.12, min_pad=0.8)
expand_axis_limits(ax2, df_plot["avg_best_drone_customers"], pad_ratio=0.12, min_pad=0.6)

# Draw first so transforms are stable, then place labels with collision avoidance
fig.canvas.draw()
used_positions = []
annotate_smart(
    ax1, x_vals, df_plot["avg_cost_saving_vs_baseline"],
    [f"{y:.2f}%" for y in df_plot["avg_cost_saving_vs_baseline"]],
    color=color_cost, used_positions=used_positions, dpi=fig.dpi,
)
annotate_smart(
    ax2, x_vals, df_plot["avg_best_drone_customers"],
    [f"{y:.2f}" for y in df_plot["avg_best_drone_customers"]],
    color=color_drone, used_positions=used_positions, dpi=fig.dpi,
)

# --- Legend (upper left, no title) ---
lines = l1 + l2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', frameon=True, shadow=True, fontsize=12)

# Layout adjustment
plt.tight_layout()

# Save
output_pdf.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_pdf, dpi=300)
plt.savefig(output_png, dpi=300)
print(f"Plot saved to: {output_pdf}")
print(f"Plot saved to: {output_png}")
