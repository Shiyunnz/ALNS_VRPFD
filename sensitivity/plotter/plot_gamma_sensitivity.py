import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse

# Set paths
current_dir = Path(__file__).parent.parent  # Go up to sensitivity folder
data_path = current_dir / "results_new" / "gamma_sensitivity" / "gamma_summary.csv"
output_path = current_dir / "results_new" / "gamma_sensitivity" / "gamma_sensitivity_plot_large.pdf"

# Check file
if not data_path.exists():
    print(f"Error: Data file not found at {data_path}")
    sys.exit(1)

# Load data
try:
    df = pd.read_csv(data_path)
except Exception as e:
    print(f"Error reading CSV: {e}")
    sys.exit(1)

# Ensure data is sorted
df.sort_values("gamma", inplace=True)

# Separate dataframes by scale
df_10 = df[df['scale'] == 'Instance10'].copy()
df_25 = df[df['scale'] == 'Instance25'].copy()

# Filter data points: gamma >= 0 and <= 3
df_10 = df_10[(df_10['gamma'] >= 0) & (df_10['gamma'] <= 3)]
df_25 = df_25[(df_25['gamma'] >= 0) & (df_25['gamma'] <= 3)]

# Font sizes (Consolidated to 3 sizes)
FONT_TITLE = 18       # Large
FONT_LABEL = 15       # Medium
FONT_TICK = 12        # Small
FONT_ANNOTATION = 12  # Small
FONT_LEGEND = 12      # Small

# Setup Plot: Horizontal Layout (1 row, 2 cols)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Define colors
barRedFill = '#F6D8E6'
barRedBorder = '#E38D83'
barBlueFill = '#CFEEF6'
barBlueBorder = '#3886C2'

color_cost = barBlueBorder
color_drone = barRedBorder

def plot_sensitivity(ax, data, title_suffix):
    x_vals = data["gamma"]

    # Axis 1 (Left): Cost Increase
    ax.set_xlabel('Gamma (Energy Uncertainty Budget)', fontsize=FONT_LABEL, fontweight='bold')
    ax.set_ylabel('Avg Cost Increase vs Baseline (%)',
                  color=color_cost, fontsize=FONT_LABEL, fontweight='bold')
    l1 = ax.plot(x_vals, data["avg_cost_increase_vs_baseline"],
                 color=color_cost,
                 marker='s',
                 markerfacecolor=barBlueFill,
                 markeredgewidth=2,
                 markersize=8,
                 linewidth=2.5,
                 label='Cost Increase (%)')
    ax.tick_params(axis='y', labelcolor=color_cost, labelsize=FONT_TICK)
    ax.tick_params(axis='x', labelsize=FONT_TICK)
    ax.set_xticks(x_vals)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title(f'Instance {title_suffix}', fontsize=FONT_TITLE, pad=10, fontweight='bold')

    # Axis 2 (Right): Drone Customers
    ax2 = ax.twinx()
    ax2.set_ylabel('Avg Drone Served Customers', color=color_drone,
                   fontsize=FONT_LABEL, fontweight='bold')
    l2 = ax2.plot(x_vals, data["avg_best_drone_customers"],
                  color=color_drone,
                  marker='o',
                  markerfacecolor=barRedFill,
                  markeredgewidth=2,
                  markersize=8,
                  linewidth=2.5,
                  linestyle='--',
                  label='Drone Customers')
    ax2.tick_params(axis='y', labelcolor=color_drone, labelsize=FONT_TICK)
    
    # Adjust right Y-axis range
    y_min, y_max = ax2.get_ylim()
    ax2.set_ylim(y_min, y_max * 1.05)

    return l1 + l2

# --- Plot: Instance10 (Left) ---
if not df_10.empty:
    lines1 = plot_sensitivity(axes[0], df_10, "10")
    labels1 = [l.get_label() for l in lines1]
    axes[0].legend(lines1, labels1, loc='upper left',
                   frameon=True, shadow=True, fontsize=FONT_LEGEND)
else:
    axes[0].text(0.5, 0.5, "No Data for Instance 10",
                 ha='center', va='center', fontsize=FONT_TITLE)
    axes[0].set_title('Instance 10', fontsize=FONT_TITLE, pad=10)

# --- Plot: Instance25 (Right) ---
if not df_25.empty:
    lines2 = plot_sensitivity(axes[1], df_25, "25")
    labels2 = [l.get_label() for l in lines2]
    axes[1].legend(lines2, labels2, loc='upper left',
                   frameon=True, shadow=True, fontsize=FONT_LEGEND)
else:
    axes[1].text(0.5, 0.5, "No Data for Instance 25",
                 ha='center', va='center', fontsize=FONT_TITLE)
    axes[1].set_title('Instance 25', fontsize=FONT_TITLE, pad=10)

# Layout adjustment
plt.tight_layout()

# Save
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300)
print(f"Plot saved to: {output_path}")
