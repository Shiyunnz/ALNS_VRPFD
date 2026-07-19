"""。"""
"""绘制能耗函数和剥夺成本函数的原函数与分段线性近似对比图。"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ----------  ----------
rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 11,
    "figure.dpi": 150,
})

# ===================================================================
# 1.   P(ω) = (W + m + ω)^1.5 × C   (kW)
# ===================================================================
W = 37.0        # kg
m = 4.2         # kg
g = 9.81
rho = 1.204
sigma = 0.025
h = 8

C = math.sqrt(g**3 / (2.0 * rho * sigma * h)) / 1000.0  # kW

def power_func(omega):
    return (W + m + omega) ** 1.5 * C

omega_max = 10.0  # kg
K_energy = 3      # YAML


bp_omega = [k * omega_max / K_energy for k in range(K_energy + 1)]
bp_power = [power_func(w) for w in bp_omega]


omega_cont = np.linspace(0, omega_max, 500)
power_cont = np.array([power_func(w) for w in omega_cont])


def pwl_interp(x, xpts, ypts):
    """。"""
    """简单分段线性插值。"""
    if x <= xpts[0]:
        return ypts[0]
    if x >= xpts[-1]:
        return ypts[-1]
    for i in range(len(xpts) - 1):
        if xpts[i] <= x <= xpts[i + 1]:
            t = (x - xpts[i]) / (xpts[i + 1] - xpts[i])
            return ypts[i] + t * (ypts[i + 1] - ypts[i])
    return ypts[-1]

power_pwl = np.array([pwl_interp(w, bp_omega, bp_power) for w in omega_cont])
power_error = np.abs(power_cont - power_pwl)
power_rel_error = power_error / power_cont * 100

# ===================================================================
# 2.   f(τ) = exp(1.5031 + 7.032τ) − exp(1.5031)
# ===================================================================
def delay_func(tau):
    return math.exp(1.5031 + 7.032 * tau) - math.exp(1.5031)

max_delay = 3.0
K_delay = 3        # YAML

bp_tau = [k * max_delay / K_delay for k in range(K_delay + 1)]
bp_delay = [delay_func(t) for t in bp_tau]

tau_cont = np.linspace(0, max_delay, 500)
delay_cont = np.array([delay_func(t) for t in tau_cont])
delay_pwl = np.array([pwl_interp(t, bp_tau, bp_delay) for t in tau_cont])
delay_error = np.abs(delay_cont - delay_pwl)

delay_rel_error = np.where(delay_cont > 1e-6, delay_error / delay_cont * 100, 0.0)

# ===================================================================

# ===================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- (a)  ---
ax = axes[0, 0]
ax.plot(omega_cont, power_cont, "b-", lw=2, label="Original: $(W{+}m{+}\\omega)^{1.5} \\cdot C$")
ax.plot(omega_cont, power_pwl, "r--", lw=1.8, label=f"PWL approx (K={K_energy})")
ax.plot(bp_omega, bp_power, "ro", ms=7, zorder=5, label="Breakpoints")

ax.fill_between(omega_cont, power_cont, power_pwl, alpha=0.15, color="red")
ax.set_xlabel("Payload $\\omega$ (kg)")
ax.set_ylabel("Power $P(\\omega)$ (kW)")
ax.set_title("(a) Energy Consumption Function")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

# --- (b)  ---
ax = axes[0, 1]
ax.plot(omega_cont, power_rel_error, "b-", lw=1.5)
ax.set_xlabel("Payload $\\omega$ (kg)")
ax.set_ylabel("Relative Error (%)")
ax.set_title(f"(b) Energy PWL Relative Error (K={K_energy})")
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color="gray", lw=0.5)

max_idx = np.argmax(power_rel_error)
ax.annotate(
    f"Max = {power_rel_error[max_idx]:.3f}%",
    xy=(omega_cont[max_idx], power_rel_error[max_idx]),
    xytext=(omega_cont[max_idx] + 1, power_rel_error[max_idx] * 0.8),
    arrowprops=dict(arrowstyle="->", color="red"),
    fontsize=11, color="red",
)

# --- (c)  ---
ax = axes[1, 0]
ax.plot(tau_cont, delay_cont, "b-", lw=2, label="Original: $e^{1.5031+7.032\\tau}-e^{1.5031}$")
ax.plot(tau_cont, delay_pwl, "r--", lw=1.8, label=f"PWL approx (K={K_delay})")
ax.plot(bp_tau, bp_delay, "ro", ms=7, zorder=5, label="Breakpoints")
ax.fill_between(tau_cont, delay_cont, delay_pwl, alpha=0.15, color="red")
ax.set_xlabel("Delay $\\tau$ (hours)")
ax.set_ylabel("Deprivation Cost $f(\\tau)$")
ax.set_title("(c) Deprivation Cost Function")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)
ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

# --- (d)  ---
ax = axes[1, 1]
# τ > 0.05 （）
mask = tau_cont > 0.05
ax.plot(tau_cont[mask], delay_rel_error[mask], "b-", lw=1.5)
ax.set_xlabel("Delay $\\tau$ (hours)")
ax.set_ylabel("Relative Error (%)")
ax.set_title(f"(d) Deprivation Cost PWL Relative Error (K={K_delay})")
ax.grid(True, alpha=0.3)

valid_errors = delay_rel_error[mask]
valid_taus = tau_cont[mask]
max_idx2 = np.argmax(valid_errors)
ax.annotate(
    f"Max = {valid_errors[max_idx2]:.1f}%",
    xy=(valid_taus[max_idx2], valid_errors[max_idx2]),
    xytext=(valid_taus[max_idx2] + 0.3, valid_errors[max_idx2] * 0.85),
    arrowprops=dict(arrowstyle="->", color="red"),
    fontsize=11, color="red",
)

plt.tight_layout(pad=2.0)
output_path = "/Users/minz/Downloads/ALNS_VRPFD/results/pwl_approximation_comparison.png"
plt.savefig(output_path, bbox_inches="tight")
plt.close()
print(f"Figure saved to {output_path}")


print("\n===== 能耗函数 PWL 断点 =====")
for i, (x, y) in enumerate(zip(bp_omega, bp_power)):
    print(f"  断点 {i}: ω = {x:.2f} kg, P = {y:.4f} kW")
print(f"  最大相对误差: {np.max(power_rel_error):.4f}%")
print(f"  平均相对误差: {np.mean(power_rel_error):.4f}%")

print("\n===== 剥夺成本函数 PWL 断点 =====")
for i, (x, y) in enumerate(zip(bp_tau, bp_delay)):
    print(f"  断点 {i}: τ = {x:.2f} h, f(τ) = {y:.2f}")
print(f"  最大相对误差 (τ>0.05): {np.max(valid_errors):.2f}%")
print(f"  平均相对误差 (τ>0.05): {np.mean(valid_errors):.2f}%")
