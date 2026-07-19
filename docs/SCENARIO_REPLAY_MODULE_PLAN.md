# 场景重放与鲁棒性评估模块方案（Scenario Replay & Robustness Evaluation）

## 1. 背景与目标

本文新增模块用于回应审稿意见：
- 仅报告鲁棒优化模型内部目标值不足以证明方案在随机能耗下的实际可执行性。
- 需要仿照 zotero文献库中的Jeong et al. (2024) Section 5.3，构造外样本不确定场景，对不同 `gamma` 下的解进行统一重放比较。

本模块目标：
1. 在多种不确定能耗分布下评估不同 `gamma` 解的稳定性与可服务性。
2. 将“成本”与“服务失败（未服务客户数）”分开记录，避免对 infeasible 成本做不可靠估计。
3. 形成可复现实验流程、结果表和论文可直接引用的分析结论。

---

## 2. 模块范围与不做事项

### 2.1 模块范围
- 输入：已求解的各 `gamma` 方案（卡车路径、无人机任务分配、时序计划等）。
- 输入：标称能耗参数（弧/任务层面的能耗基线）。
- 处理：构造随机场景、逐场景重放、记录成本和服务状态指标。
- 输出：类似 Table 6 风格的统计表 + 图（可选）+ 摘要结论。

### 2.2 不做事项
- 不在此模块内重新优化路径（除非显式扩展为 recourse re-optimization）。
- 不强行将 infeasible 场景映射成“伪成本”；该问题改由未服务指标单独表达。

---

## 3. 关键定义与符号

- `G`: gamma 集合，如 `{0, 2, 4, 6, 8, 10, 12}`。
- `x^g`: 在 `gamma = g` 下求得的固定解。
- `A`: 能耗参数单元集合（可定义为弧、飞行段或任务段）。
- `h_bar[a]`: 单元 `a` 的标称能耗。
- `S`: 每种分布的场景数（建议 `1000`）。
- `h[a,s]`: 场景 `s` 下单元 `a` 的实现能耗。
- `Cost[g,s]`: `x^g` 在场景 `s` 的目标函数成本（仅计算可执行部分）。
- `Unserved[g,s]`: 场景 `s` 未服务客户数。
- `NoTakeoff[g,s]`: 因能耗不足而未起飞任务数。
- `AbortReturn[g,s]`: 飞行中提前返航任务数。

---

## 4. 场景构建方案

> 结论：场景需要自行构建（若无历史实测数据）；若有历史数据，优先做数据驱动采样。

### 4.1 基础采样框架
对每个场景 `s`、每个单元 `a` 采样随机扰动 `xi[a,s]`，并构造：

`h[a,s] = h_bar[a] * xi[a,s]`

为了物理合理性，建议加截断：

`h[a,s] = clip(h[a,s], h_min[a], h_max[a])`

例如 `h_min[a]=0.7*h_bar[a]`, `h_max[a]=1.5*h_bar[a]`。

### 4.2 推荐分布（与5.3对齐）
1. ND（独立正态）：
   - `xi[a,s] ~ N(1, cv^2)`，建议 `cv=0.1`。
2. UD（独立均匀）：
   - `xi[a,s] ~ U(1-delta, 1+delta)`，建议 `delta=0.1`。
3. NDC（相关正态）：
   - 先采样 `z_s ~ N(0, Sigma)`，再 `xi[a,s] = 1 + cv * z[a,s]`。
   - `Sigma` 用空间邻近或同走廊弧段相关规则构造（相近单元相关更高）。

### 4.3 扩展“特殊分布”（可选，用于加强审稿回应）
- Lognormal（右尾重，反映偶发高耗）：保持 `E[xi]=1` 校准参数。
- Student-t（重尾冲击）：自由度 `nu` 小于 10 时更保守。
- Mixture（混合分布）：普通工况 + 拥堵/逆风工况。

### 4.4 采样一致性原则（非常重要）
- 对所有 `gamma` 使用同一批场景（共同随机数 CRN），保证横向比较公平。
- 训练/求解与评估分离：评估场景随机种子独立于求解过程。

---

## 5. 场景重放规则（无人机执行逻辑）

每个客户服务任务按以下顺序判断：

1. 起飞前电量检查：
   - 若当前电量 `<`（去程 + 回程 + 安全余量）需求，则 `NoTakeoff += 1`，客户记未服务。
2. 飞行中动态检查：
   - 若执行中发现剩余电量无法保证返航，触发提前返航：`AbortReturn += 1`，客户记未服务。
3. 正常完成：
   - 客户记为已服务。

约束解释：
- “无人机是否服务客户”由场景实现能耗决定，不再是确定性常量。
- 无人机失败后不得虚构完成；必须显式记录为未服务。

---

## 6. 成本与服务指标设计

鉴于 infeasible 成本难估，采用“双通道评估”：

### 6.1 成本通道
记录目标函数对应成本（例如能耗成本、运营成本等）：
- `AvgCost[g] = mean_s Cost[g,s]`
- `StdCost[g] = std_s Cost[g,s]`
- `MaxCost[g], MinCost[g]`

### 6.2 服务通道
单独记录服务失效：
- `AvgUnserved[g] = mean_s Unserved[g,s]`
- `P0[g] = P(Unserved[g,s] = 0)`（全服务概率）
- `AvgNoTakeoff[g]`
- `AvgAbortReturn[g]`

建议主文至少报告：`AvgCost/StdCost/Max/Min + AvgUnserved + P0`。

---

## 7. 实验流程（端到端）

1. 设定 `gamma` 集合 `G`。
2. 对每个 `g in G` 求解并保存 `x^g`。
3. 为每种分布生成 `S` 个场景。
4. 对每个 `g`、每个场景 `s` 执行重放模拟。
5. 输出场景级记录明细（便于复核）。
6. 聚合为统计表和图。
7. 进行跨 `gamma` 比较并写出管理含义。

---

## 8. 结果表模板（Table 6 风格）

建议每种分布一张表：

| Instance | gamma | AvgCost | StdCost | MaxCost | MinCost | AvgUnserved | P(Unserved=0)% |
|---|---:|---:|---:|---:|---:|---:|---:|
| I1 | 0  | ... | ... | ... | ... | ... | ... |
| I1 | 6  | ... | ... | ... | ... | ... | ... |
| I1 | 12 | ... | ... | ... | ... | ... | ... |

### 8.1 列含义说明（与 Jeong et al., 2024 Table 6 对齐）

Jeong 原文 Table 6（ND 分布）列名为：`Inst., nv, Γ, Avg., Std., Max, Min, Feasibility(%)`。  
在本模块中，可映射为以下含义：

- `Inst.` / `Instance`：算例编号（如 `C103C40`）。
- `nv`：该行对应解使用的车辆数（EV 数量）。用于公平比较（有时鲁棒解会多用 1 辆车）。
- `Γ` / `gamma`：鲁棒预算参数；`Γ=0` 通常表示 deterministic 基线，`Γ>0` 为鲁棒解。
- `Avg` / `AvgCost`：在该分布下、该 `(instance, gamma)` 的场景平均成本，公式：  
  `AvgCost = (1/S) * Σ_s Cost[g,s]`。
- `Std` / `StdCost`：场景成本标准差（波动性/稳定性指标），越小通常表示越稳健。
- `Max` / `MaxCost`：场景成本最大值（最差表现）。
- `Min` / `MinCost`：场景成本最小值（最好表现）。
- `Feasibility(%)` / `P(Unserved=0)%`：全服务场景占比（可行率/可靠性），公式：  
  `Feasibility = 100 * P(Unserved[g,s] = 0)`。

补充说明（建议在文中明确）：

- 上述 `Avg/Std/Max/Min` 是对同一分布下 `S` 个外样本场景聚合得到。
- `Feasibility(%)` 与 `AvgUnserved` 建议同时报告：前者反映“是否全服务”，后者反映“失败程度”。
- 若采用本项目当前 replay 输出字段，则对应关系为：  
  `avg_cost/std_cost/max_cost/min_cost/avg_unserved/p0_all_served/avg_no_takeoff/avg_abort_return`。

可选附表：
- `AvgNoTakeoff`, `AvgAbortReturn`
- CVaR 类指标（如果你后续需要风险尾部分析）

---

## 9. 图表建议（可选但强烈建议）

1. `gamma`-`AvgCost` 曲线（成本随鲁棒性变化）。
2. `gamma`-`P(Unserved=0)` 曲线（可靠性提升）。
3. `gamma`-`AvgUnserved` 曲线（服务损失下降）。
4. `Cost` 箱线图（不同 `gamma` 波动性比较，类似 5.3 思路）。

---

## 10. 论文写作建议（可直接改写）

### 10.1 方法描述
- “我们在求解后进行外样本 Monte Carlo 评估，不重新优化，仅重放固定解。”
- “由于 infeasible 成本难以可靠估计，我们将成本与未服务客户数分离报告。”

### 10.2 预期结果叙述模板
- “随着 `gamma` 增大，成本可能上升，但 `P(Unserved=0)` 提升、`AvgUnserved` 下降，体现成本-鲁棒性权衡。”
- “在重尾/相关分布下，鲁棒解相对优势更明显。”

### 10.3 审稿意见回应点
- 已实现多分布随机场景（含特殊分布）评估。
- 已提供类似 Table 6 的统一统计表。
- 已显式报告服务失败，不再依赖不可解释的 infeasible 罚成本。

---

## 11. 伪代码

```text
Input:
  Gamma set G
  Fixed solutions {x^g | g in G}
  Nominal energy h_bar[a]
  Distributions D = {ND, UD, NDC, ...}
  Scenario count S

For each distribution d in D:
  Generate scenarios Omega_d = {h[a,s]} for s=1..S

  For each gamma g in G:
    For each scenario s in 1..S:
      state <- initialize by x^g
      For each drone task t in execution order:
        if preflight_energy_not_enough(t, h[:,s], state):
          mark unserved(t), NoTakeoff += 1
          continue
        execute t with realized energy h[:,s]
        if cannot_finish_and_return(state):
          abort_and_return(), mark unserved(t), AbortReturn += 1
          continue
        mark served(t)

      Cost[g,s] <- objective_cost_under_scenario(state)
      Unserved[g,s] <- count_unserved_customers(state)

    Aggregate metrics for gamma g under distribution d:
      Avg/Std/Max/Min of Cost
      AvgUnserved
      P(Unserved=0)
      AvgNoTakeoff, AvgAbortReturn (optional)

Output tables and figures.
```

---

## 12. 数据与复现规范

- 固定随机种子并公开：`seed_main`, `seed_dist`。
- 对比使用同一场景池（CRN）。
- 保存场景明细（CSV/Parquet）：
  - `instance, distribution, gamma, scenario_id, cost, unserved, no_takeoff, abort_return`。
- 保存参数配置文件（YAML/JSON）：
  - 分布参数、截断区间、安全余量、电池容量等。

---水平）。
