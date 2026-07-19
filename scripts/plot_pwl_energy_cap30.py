"""能耗函数 P(ω) 在 ω ∈ [0, 30] kg 的 3 段 PWL 近似对比。"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 11,
    "figure.dpi": 150,
})

# ---- 能耗模型参数 ----
W = 37.0    # 机体重量 kg
m = 4.2     # 电池重量 kg
g = 9.81
rho = 1.204
sigma = 0.025
h = 8
C = math.sqrt(g**3 / (2.0 * rho * sigma * h)) / 1000.0  # kW

def power_func(omega):
    return (W + m + omega) ** 1.5 * C

def power_func_vec(omega):
    return (W + m + omega) ** 1.5 * C

omega_max = 30.0  # 实际无人机容量
K = 3

# PWL 插值
def pwl_interp_vec(x_arr, xpts, ypts):
    result = np.zeros_like(x_arr)
    for idx, x in enumerate(x_arr):
        if x <= xpts[0]:
            result[idx] = ypts[0]
        elif x >= xpts[-1]:
            result[idx] = ypts[-1]
        else:
            for i in range(len(xpts) - 1):
                if xpts[i] <= x <= xpts[i + 1]:
                    t = (x - xpts[i]) / (xpts[i + 1] - xpts[i])
                    result[idx] = ypts[i] + t * (ypts[i + 1] - ypts[i])
                    break
    return result

# 均匀 3 段
bp = [k * omega_max / K for k in range(K + 1)]  # [0, 10, 20, 30]
vals = [power_func(w) for w in bp]

# 连续曲线
omega_cont = np.linspace(0, omega_max, 1000)
p_cont = power_func_vec(omega_cont)
p_pwl = pwl_interp_vec(omega_cont, bp, vals)

p_error = np.abs(p_cont - p_pwl)
p_rel_error = p_error / p_cont * 100

# ---- 绘图 ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# (a) 函数对比
ax = axes[0]
ax.plot(omega_cont, p_cont, "b-", lw=2, label="Original: $(W{+}m{+}\\omega)^{1.5} \\cdot C$")
ax.plot(omega_cont, p_pwl, "r--", lw=1.8, label=f"PWL approx (K={K})")
ax.plot(bp, vals, "ro", ms=8, zorder=5, label="Breakpoints")
ax.fill_between(omega_cont, p_cont, p_pwl, alpha=0.15, color="red")
for i, (bx, by) in enumerate(zip(bp, vals)):
    ax.annotate(f"({bx:.0f}kg, {by:.2f}kW)", xy=(bx, by),
                xytext=(bx + 0.5, by - 0.8), fontsize=9, color="darkred")
ax.set_xlabel("Payload $\\omega$ (kg)")
ax.set_ylabel("Power $P(\\omega)$ (kW)")
ax.set_title(f"(a) Energy: $\\omega \\in [0, {omega_max:.0f}]$ kg (drone capacity={omega_max:.0f}kg)")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

# (b) 相对误差
ax = axes[1]
ax.plot(omega_cont, p_rel_error, "b-", lw=1.5)
ax.set_xlabel("Payload $\\omega$ (kg)")
ax.set_ylabel("Relative Error (%)")
ax.set_title(f"(b) PWL Relative Error (K={K}, $\\omega_{{max}}$={omega_max:.0f}kg)")
ax.grid(True, alpha=0.3)

max_idx = np.argmax(p_rel_error)
ax.annotate(
    f"Max = {p_rel_error[max_idx]:.3f}%\nat ω={omega_cont[max_idx]:.1f}kg",
    xy=(omega_cont[max_idx], p_rel_error[max_idx]),
    xytext=(omega_cont[max_idx] + 3, p_rel_error[max_idx] * 0.85),
    arrowprops=dict(arrowstyle="->", color="red"),
    fontsize=11, color="red",
)

plt.tight_layout(pad=2.0)
output_path = "/Users/minz/Downloads/ALNS_VRPFD/results/pwl_energy_cap30.png"
plt.savefig(output_path, bbox_inches="tight")
plt.close()
print(f"Figure saved to {output_path}")

# 打印结果
print(f"\n===== 能耗函数 PWL (ω_max={omega_max}kg, K={K}) =====")
print(f"断点:   {bp}")
print(f"功率值: {[f'{v:.4f}' for v in vals]} kW")
print(f"最大相对误差: {np.max(p_rel_error):.4f}%")
print(f"平均相对误差: {np.mean(p_rel_error):.4f}%")

print(f"\n===== 分段线性表达式 =====")
for i in range(len(bp) - 1):
    x0, x1 = bp[i], bp[i + 1]
    y0, y1 = vals[i], vals[i + 1]
    slope = (y1 - y0) / (x1 - x0)
    print(f"  段 {i+1}: ω ∈ [{x0:.0f}, {x1:.0f}] kg")
    print(f"         P(ω) = {slope:.6f} × (ω - {x0:.0f}) + {y0:.4f}  kW")
    print(f"         斜率 = {slope:.6f} kW/kg")
    print()

# 对比 ω_max=10 vs 30
print("===== 对比: ω_max=10 vs ω_max=30 =====")
bp10 = [k * 10.0 / 3 for k in range(4)]
vals10 = [power_func(w) for w in bp10]
omega10 = np.linspace(0, 10, 500)
p10_cont = power_func_vec(omega10)
p10_pwl = pwl_interp_vec(omega10, bp10, vals10)
rel10 = np.abs(p10_cont - p10_pwl) / p10_cont * 100
print(f"  ω_max=10: 最大相对误差 = {np.max(rel10):.4f}%")
print(f"  ω_max=30: 最大相对误差 = {np.max(p_rel_error):.4f}%")
