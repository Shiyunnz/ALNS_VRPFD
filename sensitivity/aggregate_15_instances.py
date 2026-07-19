import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Define the paths to the 3 result files
files = [
    "sensitivity/results_new/battery_sensitivity_rerun_in_25/battery_R30_low/bat_low_all_trials.csv",
    "sensitivity/results_new/battery_sensitivity_rerun_in_25/battery_R40_25_low/bat_R40_low_all_trials.csv",
    "sensitivity/results_new/battery_sensitivity_rerun_in_25/battery_R50_25_low/bat_R50_25_low_all_trials.csv",
    "sensitivity/results_new/battery_sensitivity_rerun_in_25/battery_R30_low/bat_low_fix_2_all_trials.csv",
    "sensitivity/results_new/battery_sensitivity_rerun_in_25/battery_R30_low/bat_low_fix_4_all_trials.csv",
    "sensitivity/results_new/battery_sensitivity_rerun_in_25/battery_R30_low/bat_low_fix_5_all_trials.csv"
]

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.2f}'.format)

all_data = []

print(f"{'Scale':<10} {'Instance':<15} {'Bat':<5} {'BaseCost':<10} {'MinCost':<10} {'Saving%':<10} {'Drones':<5}")
print("-" * 80)

# Process all files together
dfs = []
for f in files:
    if not os.path.exists(f):
        print(f"File not found: {f}")
        continue
    dfs.append(pd.read_csv(f))

if not dfs:
    print("No data found.")
    exit(1)

df = pd.concat(dfs, ignore_index=True)

# Implement "Best of 5" Methodology:
# Sort by Cost (asc) then Drones (desc) to break ties in favor of more drones
df_sorted = df.sort_values(['best_cost', 'best_drone_customers'], ascending=[True, False])

# For each instance and battery capacity, keep only the BEST (minimum cost) trial
best_configs = df_sorted.drop_duplicates(subset=['scale', 'instance_name', 'battery_capacity'], keep='first')

# Calculate stats
global_stats = []
instances = best_configs['instance_name'].unique()

for inst in instances:
    inst_data = best_configs[best_configs['instance_name'] == inst]
    
    # Get baseline (4.3)
    baseline_row = inst_data[np.isclose(inst_data['battery_capacity'], 4.3)]
    
    if baseline_row.empty:
        # Fallback
        min_bat = inst_data['battery_capacity'].min()
        baseline_row = inst_data[inst_data['battery_capacity'] == min_bat]
        
    baseline_cost = baseline_row.iloc[0]['best_cost']
    
    # Calculate stats for all rows
    for _, row in inst_data.iterrows():
        bat = row['battery_capacity']
        cost = row['best_cost']
        drones = row['best_drone_customers']
        
        saving = (baseline_cost - cost) / baseline_cost * 100
        
        global_stats.append({
            'scale': row['scale'],
            'instance': inst,
            'battery': bat,
            'min_cost': cost,
            'saving_pct': saving,
            'drones': drones
        })

# Create DataFrame from all stats
res_df = pd.DataFrame(global_stats)

# Define relevant battery levels to 1 decimal place to grouping
res_df['battery'] = res_df['battery'].round(1)
levels = [4.3, 5.3, 6.3, 7.3, 8.3]
res_df = res_df[res_df['battery'].isin(levels)]

# Group by battery level and compute averages across all 15 instances
summary = res_df.groupby('battery').agg({
    'saving_pct': 'mean',
    'drones': 'mean',
    'min_cost': 'count'
}).reset_index()

print("\nAggregate Results (15 instances: R30, R40, R50 - All 25 customers):")
print(summary)

# Export summary to CSV for plotting.
# Canonical location follows the active battery_sensitivity directory layout.
summary_csv_path = "sensitivity/results_new/battery_sensitivity_rerun_in_25/battery_sensitivity_15inst_summary.csv"
os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
summary.to_csv(summary_csv_path, index=False)
print(f"Exported summary data to: {summary_csv_path}")

# Backward-compatibility export for older scripts that still read the legacy root path.
legacy_summary_csv_path = "sensitivity/results_new/battery_sensitivity_15inst_summary.csv"
summary.to_csv(legacy_summary_csv_path, index=False)
print(f"Also exported compatibility copy to: {legacy_summary_csv_path}")

# Also breakdown by scale
print("\nBreakdown by Scale:")
summary_scale = res_df.groupby(['scale', 'battery']).agg({
    'saving_pct': 'mean',
    'drones': 'mean'
}).reset_index()

# Plotting
sns.set_style("whitegrid")
output_dir = "sensitivity/results_new"

# 1. Cost Saving Plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=summary, x='battery', y='saving_pct', marker='o', linewidth=2.5, markersize=8)
plt.title("Average Cost Saving vs. Battery Capacity (15 Instances)", fontsize=14)
plt.xlabel("Battery Capacity (kWh)", fontsize=12)
plt.ylabel("Cost Saving (%)", fontsize=12)
plt.xticks(levels)
plt.grid(True, linestyle='--', alpha=0.7)

# Add value labels
for x, y in zip(summary['battery'], summary['saving_pct']):
    plt.text(x, y + 0.3, f"{y:.2f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

plot_path_base = os.path.join(output_dir, "battery_15inst_cost_saving")
plt.savefig(f"{plot_path_base}.pdf", bbox_inches='tight')
plt.savefig(f"{plot_path_base}.png", bbox_inches='tight', dpi=300)
print(f"Saved plot to {plot_path_base}.pdf/png")
plt.close()

# 2. Drone Count Plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=summary, x='battery', y='drones', marker='D', color='orange', linewidth=2.5, markersize=8)
plt.title("Average Drone Customers vs. Battery Capacity (15 Instances)", fontsize=14)
plt.xlabel("Battery Capacity (kWh)", fontsize=12)
plt.ylabel("Average Drone Customers", fontsize=12)
plt.xticks(levels)
plt.grid(True, linestyle='--', alpha=0.7)

# Add value labels
for x, y in zip(summary['battery'], summary['drones']):
    plt.text(x, y + 0.1, f"{y:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

plot_path_drones = os.path.join(output_dir, "battery_15inst_drones")
plt.savefig(f"{plot_path_drones}.pdf", bbox_inches='tight')
plt.savefig(f"{plot_path_drones}.png", bbox_inches='tight', dpi=300)
print(f"Saved plot to {plot_path_drones}.pdf/png")
plt.close()
