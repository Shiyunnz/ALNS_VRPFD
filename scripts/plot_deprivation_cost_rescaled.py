import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

# Parameters
T = 8.0  # operation horizon
b_orig = 7.032  # original b from Holguin-Veras
b = b_orig / T  # rescaled: 0.879

# Class definitions: a_k = a_base + ln(kappa)
# a_base = 1.5031 (from Holguin-Veras water calibration)
a_base = 1.5031

classes = {
    'Class 1\n(Medicine)': {'kappa': 3.0, 'color': '#d62728', 'ls': '-'},
    'Class 2\n(Water)':    {'kappa': 2.0, 'color': '#ff7f0e', 'ls': '-'},
    'Class 3\n(Food)':     {'kappa': 1.0, 'color': '#2ca02c', 'ls': '-'},
    'Class 4\n(Shelter)':  {'kappa': 0.4, 'color': '#1f77b4', 'ls': '-'},
}

tau = np.linspace(0, 5, 500)  # delay in hours

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# --- Left panel: full range ---
ax = axes[0]
for name, info in classes.items():
    a_k = a_base + math.log(info['kappa'])
    cost = np.exp(a_k + b * tau) - np.exp(a_k)
    label = f'{name.replace(chr(10), " ")}  ($\\kappa$={info["kappa"]}, $a_k$={a_k:.2f})'
    ax.plot(tau, cost, color=info['color'], ls=info['ls'], lw=2.2, label=label)

ax.set_xlabel('Delay $\\tau$ (hours)', fontsize=13)
ax.set_ylabel('Deprivation cost $f_k(\\tau)$', fontsize=13)
ax.set_title('(a) Rescaled exponential deprivation cost', fontsize=13)
ax.legend(fontsize=9.5, loc='upper left')
ax.set_xlim(0, 5)
ax.set_ylim(0, 120)
ax.axhline(y=30, color='gray', ls=':', lw=0.8, alpha=0.5)
ax.axhline(y=90, color='gray', ls=':', lw=0.8, alpha=0.5)
ax.text(4.8, 31, 'Typical truck cost $\\sim$30', fontsize=8, color='gray', ha='right')
ax.text(4.8, 91, 'Upper truck cost $\\sim$90', fontsize=8, color='gray', ha='right')
ax.grid(True, alpha=0.3)

# --- Right panel: comparison with original (unrescaled) ---
ax2 = axes[1]
for name, info in classes.items():
    a_k = a_base + math.log(info['kappa'])
    cost_new = np.exp(a_k + b * tau) - np.exp(a_k)
    label_short = f'{name.replace(chr(10), " ")}'
    ax2.plot(tau, cost_new, color=info['color'], ls='-', lw=2.2, label=label_short)

# Original unrescaled (only Class 3 for illustration)
cost_orig = np.exp(a_base + b_orig * tau) - np.exp(a_base)
tau_orig_vis = tau[cost_orig <= 5000]
cost_orig_vis = cost_orig[cost_orig <= 5000]
ax2.plot(tau_orig_vis, cost_orig_vis, color='black', ls='--', lw=1.5,
         label='Original (unrescaled)\n$e^{1.50+7.03\\tau}-e^{1.50}$')

ax2.set_xlabel('Delay $\\tau$ (hours)', fontsize=13)
ax2.set_ylabel('Deprivation cost', fontsize=13)
ax2.set_title('(b) Comparison with original scale', fontsize=13)
ax2.legend(fontsize=9, loc='upper left')
ax2.set_xlim(0, 5)
ax2.set_ylim(0, 500)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/minz/Desktop/ResearchProject/code/figures/deprivation_cost_rescaled.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/minz/Desktop/ResearchProject/code/figures/deprivation_cost_rescaled.pdf', bbox_inches='tight')
print("Saved to figures/deprivation_cost_rescaled.png and .pdf")

# Print numeric table
print("\n=== Numeric values: f_k(tau) = exp(a_k + (b/T)*tau) - exp(a_k) ===")
print(f"b/T = {b:.4f}, T = {T}")
print(f"{'tau':>6}", end="")
for name in classes:
    print(f"  {name.replace(chr(10), ' '):>12}", end="")
print()

for t in [0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
    print(f"{t:6.1f}", end="")
    for info in classes.values():
        a_k = a_base + math.log(info['kappa'])
        cost = math.exp(a_k + b * t) - math.exp(a_k)
        print(f"  {cost:12.2f}", end="")
    print()