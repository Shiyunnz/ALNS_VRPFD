# Embedded vs Precheck_Guarded：现状说明

## 1. 目标与范围

本文档用于记录当前仓库中两种鲁棒策略的**最新对比现状**，仅覆盖：

- `embedded`
- `precheck_guarded`

不再讨论已下线的 `verification/paper/...` 旧策略。

更新时间：2026-02-22

---

## 2. 两种策略定义（当前代码）

### 2.1 embedded

- 搜索阶段使用鲁棒 `gamma`（`gamma(search)=gamma(robust)`）。
- 修复算子使用 `robust_energy_mode="embedded"`。
- 等价于“鲁棒性嵌入搜索过程”。

关键代码：

- `/Users/minz/Downloads/ALNS_VRPFD/run_alns.py:364`
- `/Users/minz/Downloads/ALNS_VRPFD/run_alns.py:397`
- `/Users/minz/Downloads/ALNS_VRPFD/sensitivity/robust_verification_strategy_compare.py:50`

### 2.2 precheck_guarded

- 搜索阶段使用确定性 `gamma(search)=0`。
- 候选先过 changed-subroute 鲁棒前筛（轻量）。
- 仅在候选即将成为 new-best 时做一次精确鲁棒校验（guarded）。
- 修复算子使用 `robust_energy_mode="verification"`。

关键代码：

- `/Users/minz/Downloads/ALNS_VRPFD/run_alns.py:364`
- `/Users/minz/Downloads/ALNS_VRPFD/run_alns.py:397`
- `/Users/minz/Downloads/ALNS_VRPFD/run_alns.py:444`
- `/Users/minz/Downloads/ALNS_VRPFD/run_alns.py:527`
- `/Users/minz/Downloads/ALNS_VRPFD/alns_vrpfd/evaluation/subroute_robust_verifier.py:1`
- `/Users/minz/Downloads/ALNS_VRPFD/sensitivity/robust_verification_strategy_compare.py:57`

---

## 3. 实验口径（本次对比）

- 数据集：`data/Instance10` 全 15 个算例
- seed：3 个（`20260222~20260224`）
- 迭代：`2000`
- `gamma=3`
- 方法：`embedded, precheck_guarded`
- 输出前缀：`instance10_all_t3_i2000_embedded_vs_precheck_guarded`

结果文件：

- `/Users/minz/Downloads/ALNS_VRPFD/sensitivity/results_new/verification_strategy_compare/instance10_all_t3_i2000_embedded_vs_precheck_guarded_trials.csv`
- `/Users/minz/Downloads/ALNS_VRPFD/sensitivity/results_new/verification_strategy_compare/instance10_all_t3_i2000_embedded_vs_precheck_guarded_method_summary.csv`
- `/Users/minz/Downloads/ALNS_VRPFD/sensitivity/results_new/verification_strategy_compare/instance10_all_t3_i2000_embedded_vs_precheck_guarded_pair_summary.csv`

---

## 4. 总体结果

### 4.1 运行时间与平均成本（45 次）

| Method | Mean Runtime (s) | Median Runtime (s) | Mean Robust Cost | Median Robust Cost | Robust Feasible Ratio |
|---|---:|---:|---:|---:|---:|
| embedded | 2.2241 | 2.0218 | 123.1493 | 110.9600 | 1.000 |
| precheck_guarded | 2.2883 | 2.1432 | 125.1096 | 110.9600 | 1.000 |

结论：两者鲁棒可行率都为 1.0；平均运行时间 `embedded` 更快（约 2.9%）。

### 4.2 成本/时间波动（全样本）

| Method | Runtime Std | Runtime CV | Robust Cost Std | Robust Cost CV |
|---|---:|---:|---:|---:|
| embedded | 0.5102 | 0.2294 | 49.7173 | 0.4037 |
| precheck_guarded | 0.6045 | 0.2642 | 49.8093 | 0.3981 |

结论：全样本层面，`precheck_guarded` 时间波动更大；成本 CV 两者接近。

### 4.3 同一算例跨 seed 稳定性（更关键）

| Method | Mean Instance Runtime Std | Median Instance Runtime Std | Mean Instance RobustCost Std | Median Instance RobustCost Std |
|---|---:|---:|---:|---:|
| embedded | 0.1276 | 0.0945 | 0.6619 | 0.4673 |
| precheck_guarded | 0.2307 | 0.1953 | 2.1586 | 0.8126 |

结论：在“同算例多次运行一致性”上，`embedded` 显著更稳。

---

## 5. 配对比较（45 对 instance-seed）

以 `method_a=embedded`、`method_b=precheck_guarded`：

- `b_faster_ratio = 0.4444`
- `b_robust_noninferior_ratio = 0.5556`
- `b_effective_acceleration_ratio = 0.2667`
- `mean_robust_cost_delta_b_minus_a = +1.9602`

解释：`precheck_guarded` 只有约 26.7% 的样本同时实现“更快且鲁棒不劣”。

---

## 6. 分区表现（R30/R40/R50）

| Region | Embedded Runtime Mean | Precheck Runtime Mean | Embedded Robust Cost Mean | Precheck Robust Cost Mean |
|---|---:|---:|---:|---:|
| R30 | 1.9560 | 2.1669 | 73.5317 | 75.6653 |
| R40 | 2.0275 | 1.9024 | 114.7627 | 116.1120 |
| R50 | 2.6887 | 2.7957 | 181.1536 | 183.5513 |

结论：`precheck_guarded` 主要在 R40 区间有速度优势；R30 和 R50 目前不占优。

---

## 7. 当前阶段结论

在当前实现与参数下：

1. 两个策略都能保持 100% 鲁棒可行率。
2. `embedded` 在总体平均时间和跨 seed 稳定性上更优。
3. `precheck_guarded` 具备局部场景优势（尤其 R40），但尚未形成全局优势。
4. 若目标是“让 `precheck_guarded` 全局领先”，需要继续降低 guarded 精验触发成本并抑制跨 seed 成本波动。

