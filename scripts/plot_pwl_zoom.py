"""剥夺成本函数在 τ ∈ [0, 0.5h] 区间的放大对比图。"""

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

# 剥夺成本函数 (τ 单位: 小时)
def delay_func(tau):
    return math.exp(1.5031 + 7.032 * tau) - math.exp(1.5031)

# PWL 插值
def pwl_interp(x, xpts, ypts):
    if x <= xpts[0]:
        return ypts[0]
    if x >= xpts[-1]:
        return ypts[-1]
    for i in range(len(xpts) - 1):
        if xpts[i] <= x <= xpts[i + 1]:
            t = (x - xpts[i]) / (xpts[i + 1] - xpts[i])
            return ypts[i] + t * (ypts[i + 1] - ypts[i])
    return ypts[-1]

# ---- 参数 ----
max_delay = 3.0  # PWL 总范围 (小时)
K = 3            # 分段数
zoom_max = 0.5   # 放大区间上限 (小时)

bp_tau = [k * max_delay / K for k in range(K + 1)]  # [0, 1, 2, 3]
bp_delay = [delay_func(t) for t in bp_tau]

tau_cont = np.linspace(0, zoom_max, 500)
delay_cont = np.array([delay_func(t) for t in tau_cont])
delay_pwl = np.array([pwl_interp(t, bp_tau, bp_delay) for t in tau_cont])
delay_error = np.abs(delay_cont - delay_pwl)
delay_rel_error = np.where(delay_cont > 1e-6, delay_error / delay_cont * 100, 0.0)

# ---- 绘图 ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) 函数对比
ax = axes[0]
ax.plot(tau_cont, delay_cont, "b-", lw=2, label="Original: $e^{1.5031+7.032\\tau}-e^{1.5031}$")
ax.plot(tau_cont, delay_pwl, "r--", lw=1.8, label=f"PWL approx (K={K}, breakpoints at 0,1,2,3h)")
# 标出在此区间内的断点
for i, (bx, by) in enumerate(zip(bp_tau, bp_delay)):
    if bx <= zoom_max:
        ax.plot(bx, by, "ro", ms=8, zorder=5, label="Breakpoint" if i == 0 else None)
ax.fill_between(tau_cont, delay_cont, delay_pwl, alpha=0.15, color="red", label="Approximation error")
ax.set_xlabel("Delay $\\tau$ (hours)")
ax.set_ylabel("Deprivation Cost $f(\\tau)$")
ax.set_title(f"(a) Deprivation Cost: $\\tau \\in [0, {zoom_max}]$ h  (= [0, {int(zoom_max*60)}] min)")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

# 标注关键值
ax.annotate(
    f"f({zoom_max}h) = {delay_func(zoom_max):.1f}",
    xy=(zoom_max, delay_func(zoom_max)),
    xytext=(zoom_max * 0.5, delay_func(zoom_max) * 1.3),
    arrowprops=dict(arrowstyle="->", color="blue"),
    fontsize=10, color="blue",
)
ax.annotate(
    f"PWL({zoom_max}h) = {pwl_interp(zoom_max, bp_tau, bp_delay):.1f}",
    xy=(zoom_max, pwl_interp(zoom_max, bp_tau, bp_delay)),
    xytext=(zoom_max * 0.35, pwl_interp(zoom_max, bp_tau, bp_delay) * 2.5),
    arrowprops=dict(arrowstyle="->", color="red"),
    fontsize=10, color="red",
)

# (b) 相对误差
ax = axes[1]
mask = tau_cont > 0.01
ax.plot(tau_cont[mask], delay_rel_error[mask], "b-", lw=1.5)
ax.set_xlabel("Delay $\\tau$ (hours)")
ax.set_ylabel("Relative Error (%)")
ax.set_title(f"(b) PWL Relative Error in $\\tau \\in [0, {zoom_max}]$ h")
ax.grid(True, alpha=0.3)

# 标注最大误差
valid_errors = delay_rel_error[mask]
valid_taus = tau_cont[mask]
max_idx = np.argmax(valid_errors)
ax.annotate(
    f"Max = {valid_errors[max_idx]:.1f}%\nat τ={valid_taus[max_idx]:.3f}h ({valid_taus[max_idx]*60:.1f}min)",
    xy=(valid_taus[max_idx], valid_errors[max_idx]),
    xytext=(valid_taus[max_idx] + 0.05, valid_errors[max_idx] * 0.75),
    arrowprops=dict(arrowstyle="->", color="red"),
    fontsize=10, color="red",
)

plt.tight_layout(pad=2.0)
output_path = "/Users/minz/Downloads/ALNS_VRPFD/results/pwl_deprivation_zoom_0_0.5h.png"
plt.savefig(output_path, bbox_inches="tight")
plt.close()
print(f"Figure saved to {output_path}")

# 打印关键数值
print(f"\n===== τ ∈ [0, {zoom_max}h] = [0, {int(zoom_max*60)}min] 区间分析 =====")
print(f"PWL 断点: {bp_tau}  (仅第一个断点 0 在此区间内)")
print(f"此区间完全落在第一段 [0, 1h] 内，PWL 为单一直线段")
print(f"\n真实值 f({zoom_max}h) = {delay_func(zoom_max):.2f}")
print(f"PWL值 PWL({zoom_max}h) = {pwl_interp(zoom_max, bp_tau, bp_delay):.2f}")
print(f"绝对误差 = {abs(delay_func(zoom_max) - pwl_interp(zoom_max, bp_tau, bp_delay)):.2f}")
print(f"相对误差 = {abs(delay_func(zoom_max) - pwl_interp(zoom_max, bp_tau, bp_delay)) / delay_func(zoom_max) * 100:.1f}%")
print(f"\nPWL 第一段斜率 (线性) = {(bp_delay[1] - bp_delay[0]) / (bp_tau[1] - bp_tau[0]):.2f}")
print(f"真实导数 f'(0) = {7.032 * math.exp(1.5031):.2f}")
print(f"真实导数 f'({zoom_max}) = {7.032 * math.exp(1.5031 + 7.032 * zoom_max):.2f}")
