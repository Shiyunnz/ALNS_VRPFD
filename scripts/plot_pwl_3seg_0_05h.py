"""在 τ ∈ [0, 0.5h] 区间构建 3 段 PWL 近似并可视化。"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
# 纯网格搜索，不依赖 scipy

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

def delay_func_vec(tau):
    return np.exp(1.5031 + 7.032 * tau) - np.exp(1.5031)

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

# ---- 参数 ----
tau_max = 0.5  # 30 分钟
K = 3

# 方案 1: 均匀间距
bp_uniform = [k * tau_max / K for k in range(K + 1)]
val_uniform = [delay_func(t) for t in bp_uniform]

# 方案 2: 最优断点 — 最小化最大相对误差
tau_dense = np.linspace(0.005, tau_max, 2000)
f_dense = delay_func_vec(tau_dense)

def max_rel_error(inner_bps):
    """给定 2 个内部断点，计算最大相对误差。"""
    b1, b2 = sorted(inner_bps)
    if b1 <= 0.001 or b2 <= b1 + 0.001 or b2 >= tau_max - 0.001:
        return 1e10
    xpts = [0.0, float(b1), float(b2), tau_max]
    ypts = [delay_func(x) for x in xpts]
    pwl_vals = pwl_interp_vec(tau_dense, xpts, ypts)
    rel_err = np.abs(pwl_vals - f_dense) / np.maximum(f_dense, 1e-10) * 100
    return np.max(rel_err)

# 网格搜索 + 局部优化
best_err = 1e10
best_bps = None
for b1 in np.linspace(0.02, 0.3, 30):
    for b2 in np.linspace(b1 + 0.02, 0.48, 30):
        err = max_rel_error([b1, b2])
        if err < best_err:
            best_err = err
            best_bps = [b1, b2]

# 细化网格搜索
for _ in range(3):
    b1c, b2c = best_bps
    step = 0.005
    for b1 in np.linspace(max(0.005, b1c - 0.03), b1c + 0.03, 40):
        for b2 in np.linspace(max(b1 + 0.005, b2c - 0.03), min(tau_max - 0.005, b2c + 0.03), 40):
            err = max_rel_error([b1, b2])
            if err < best_err:
                best_err = err
                best_bps = [b1, b2]

bp_opt = [0.0, float(best_bps[0]), float(best_bps[1]), tau_max]
val_opt = [delay_func(t) for t in bp_opt]

# ---- 连续曲线 ----
tau_cont = np.linspace(0, tau_max, 1000)
f_cont = delay_func_vec(tau_cont)

pwl_uniform = pwl_interp_vec(tau_cont, bp_uniform, val_uniform)
pwl_opt = pwl_interp_vec(tau_cont, bp_opt, val_opt)

rel_err_uniform = np.where(f_cont > 0.1, np.abs(pwl_uniform - f_cont) / f_cont * 100, 0)
rel_err_opt = np.where(f_cont > 0.1, np.abs(pwl_opt - f_cont) / f_cont * 100, 0)

# ---- 绘图 ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# (a) 均匀断点
ax = axes[0]
ax.plot(tau_cont * 60, f_cont, "b-", lw=2, label="Original $f(\\tau)$")
ax.plot(tau_cont * 60, pwl_uniform, "r--", lw=1.8, label=f"PWL uniform (K={K})")
ax.plot([t * 60 for t in bp_uniform], val_uniform, "ro", ms=8, zorder=5)
ax.fill_between(tau_cont * 60, f_cont, pwl_uniform, alpha=0.12, color="red")
for i, (bx, by) in enumerate(zip(bp_uniform, val_uniform)):
    ax.annotate(f"({bx*60:.0f}min, {by:.1f})", xy=(bx * 60, by),
                xytext=(bx * 60 + 1, by + 12), fontsize=8, color="darkred")
ax.set_xlabel("Delay (minutes)")
ax.set_ylabel("Deprivation Cost $f(\\tau)$")
ax.set_title("(a) Uniform Spacing")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

# (b) 最优断点
ax = axes[1]
ax.plot(tau_cont * 60, f_cont, "b-", lw=2, label="Original $f(\\tau)$")
ax.plot(tau_cont * 60, pwl_opt, "g--", lw=1.8, label=f"PWL optimized (K={K})")
ax.plot([t * 60 for t in bp_opt], val_opt, "go", ms=8, zorder=5)
ax.fill_between(tau_cont * 60, f_cont, pwl_opt, alpha=0.12, color="green")
for i, (bx, by) in enumerate(zip(bp_opt, val_opt)):
    ax.annotate(f"({bx*60:.1f}min, {by:.1f})", xy=(bx * 60, by),
                xytext=(bx * 60 + 0.5, by + 12), fontsize=8, color="darkgreen")
ax.set_xlabel("Delay (minutes)")
ax.set_ylabel("Deprivation Cost $f(\\tau)$")
ax.set_title("(b) Optimized Spacing (min-max error)")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

# (c) 误差对比
ax = axes[2]
mask = tau_cont > 0.005
ax.plot(tau_cont[mask] * 60, rel_err_uniform[mask], "r-", lw=1.5, label="Uniform")
ax.plot(tau_cont[mask] * 60, rel_err_opt[mask], "g-", lw=1.5, label="Optimized")
ax.set_xlabel("Delay (minutes)")
ax.set_ylabel("Relative Error (%)")
ax.set_title("(c) Relative Error Comparison")
ax.legend()
ax.grid(True, alpha=0.3)

# 标注最大误差
max_idx_u = np.argmax(rel_err_uniform[mask])
max_idx_o = np.argmax(rel_err_opt[mask])
ax.annotate(f"Max = {rel_err_uniform[mask][max_idx_u]:.1f}%",
            xy=(tau_cont[mask][max_idx_u] * 60, rel_err_uniform[mask][max_idx_u]),
            xytext=(tau_cont[mask][max_idx_u] * 60 + 3, rel_err_uniform[mask][max_idx_u] * 0.85),
            arrowprops=dict(arrowstyle="->", color="red"), fontsize=10, color="red")
ax.annotate(f"Max = {rel_err_opt[mask][max_idx_o]:.1f}%",
            xy=(tau_cont[mask][max_idx_o] * 60, rel_err_opt[mask][max_idx_o]),
            xytext=(tau_cont[mask][max_idx_o] * 60 + 3, rel_err_opt[mask][max_idx_o] + 5),
            arrowprops=dict(arrowstyle="->", color="green"), fontsize=10, color="green")

plt.tight_layout(pad=2.0)
output_path = "/Users/minz/Downloads/ALNS_VRPFD/results/pwl_deprivation_3seg_0_05h.png"
plt.savefig(output_path, bbox_inches="tight")
plt.close()
print(f"Figure saved to {output_path}")

# ---- 打印结果 ----
print("\n===== 均匀间距 3 段 PWL =====")
print(f"断点 (分钟): {[f'{t*60:.1f}' for t in bp_uniform]}")
print(f"函数值:      {[f'{v:.2f}' for v in val_uniform]}")
print(f"最大相对误差: {np.max(rel_err_uniform):.2f}%")

print("\n===== 最优间距 3 段 PWL =====")
print(f"断点 (分钟): {[f'{t*60:.2f}' for t in bp_opt]}")
print(f"函数值:      {[f'{v:.2f}' for v in val_opt]}")
print(f"最大相对误差: {np.max(rel_err_opt):.2f}%")

# 输出分段线性函数的显式表达式
print("\n===== 最优 3 段 PWL 显式表达式 =====")
print(f"f(τ) 的分段线性近似 (τ 单位: 小时, 对应分钟如下):\n")
for i in range(len(bp_opt) - 1):
    x0, x1 = bp_opt[i], bp_opt[i + 1]
    y0, y1 = val_opt[i], val_opt[i + 1]
    slope = (y1 - y0) / (x1 - x0)
    print(f"  段 {i+1}: τ ∈ [{x0*60:.2f}, {x1*60:.2f}] min  (= [{x0:.4f}, {x1:.4f}] h)")
    print(f"         γ(τ) = {slope:.4f} × (τ - {x0:.4f}) + {y0:.4f}")
    print(f"         斜率 = {slope:.4f}")
    print()
