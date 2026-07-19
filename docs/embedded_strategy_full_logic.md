# Embedded 策略完整逻辑说明（当前代码版本）

本文档描述当前仓库中 `embedded` 策略的实际执行逻辑。目标是把从命令入口到最终结果输出的完整数据流、控制流、约束检查与公式都说清楚，并明确与 `precheck_guarded` 的边界。

## 1. 入口与策略选择

入口脚本为 `run_alns.py`，通过 `--robust-strategy` 选择策略：

- `embedded`
- `precheck_guarded`

在 `embedded` 下：

- `search_gamma = config.energy_uncertainty_budget`（搜索阶段直接使用鲁棒 Gamma）
- `repair_mode = "embedded"`
- 不创建 `robust_evaluator`
- 不创建 `candidate_subroute_verifier`
- `robust_check_on_new_best = False`

也就是说，`embedded` 的鲁棒性不是“后验门控”，而是直接注入搜索 evaluator 与 repair 候选生成过程。

## 2. 配置与实例初始化

主要配置来源：`config/alns_config.yaml`，通过 `ALNSConfig` 读取。

关键鲁棒参数：

- `drone_battery_capacity`（默认 6.3 kWh）
- `energy_uncertainty_budget`（默认 3）
- `energy_deviation_rate`（默认 0.1）
- `same_truck_retrieval`（当前默认 false）

`embedded` 下会调用 `instance.configure_robustness(...)`，其中 `energy_uncertainty_budget=search_gamma=Gamma`。这意味着后续 evaluator 与 repair 算子拿到的实例鲁棒配置已是“完整鲁棒预算”。

## 3. 初始解构造

由 `initial_solution.two_phase` 决定：

- `true`: `build_two_phase_initial_solution(...)`
- `false`: `build_initial_solution(...)`

若配置 `forced_drone_customers`，这些节点在初始解阶段即禁止卡车访问（只允许无人机）。

## 4. 算子池构建（embedded）

### 4.1 Destroy 算子

当前 `build_operators(...)` 中默认 destroy 集合为：

- `DestroyRandom`
- `DestroyWorstDistance`
- `DestroyShaw`

这三个算子都在 `DestroyOperator` 基类框架下工作：先采样移除客户，再执行客户移除与连带一致性修复（包括无人机任务锚点关联）。

### 4.2 Repair 算子

当前 `repair_set="all"` 时包含：

- `RepairCheapest`
- `RepairRegret`
- `RepairBiasedRandomized`
- `RepairEqualPriority`
- `RepairDronePriorityRegret`
- `RepairTruckFirst`

所有 repair 算子均继承 `RepairOperator`，共享候选生成与约束过滤逻辑；差异主要在“客户选择规则”或“候选选择规则”。

关键点：`embedded` 传入 `robust_energy_mode="embedded"`。

## 5. Repair 详细流程（embedded 的核心）

`RepairOperator.apply(solution, unassigned)` 的单客户插入循环：

1. 从未分配池中选择一个客户（按策略，如 regret、biased random 等）。
2. 为该客户生成所有候选插入位置：
   - 卡车插入候选（truck route + position）
   - 插入已有无人机任务候选
   - 新建无人机任务候选（truck launch/retrieve 或 depot launch/retrieve）
3. 对候选进行可行性过滤。
4. 对剩余候选打分并选最优候选。
5. 应用该候选到解上。

### 5.1 候选可行性过滤

过滤维度包括：

- 载重约束（truck/drone capacity）
- 时间约束（retrieve 同步窗口、time slack）
- 路径合法性（launch/retrieve 节点必须在对应卡车路径或 depot）
- 无人机能量约束（鲁棒最坏能耗不超过电池容量）

### 5.2 embedded 的能量计算公式

对于任务 `launch -> customers -> retrieve`：

1. 分段名义能耗：
   - `e_k = energy_model.energy_kwh(payload_k, time_k)`
2. 分段偏差：
   - `d_k = e_k * deviation_rate`
3. 最坏能耗：
   - `worst = sum(e_k) + budgeted_sum({d_k}, Gamma)`

其中 `embedded` 直接取 `Gamma = instance.robust_config.energy_uncertainty_budget`。

`budgeted_sum` 实现为对偏差降序后取前 `floor(Gamma)` 个全额偏差，若有小数部分则加上下一项的比例偏差（当前 Gamma 通常为整数）。

### 5.3 候选评分（normalise + score）

候选评分由三部分归一化线性组合：

- 距离成本项（truck/drone 分别乘 unit cost）
- 能耗增量项
- 延误增量项

公式（简化）：

- `score = w1*c_norm + w2*e_norm + w3*l_norm`

随后若是 drone 候选，减去 drone bonus（`drone_priority`、depot bonus、多客户 bonus、高电量利用率 bonus 等）。

最后叠加时间窗 slack 罚项。

结论：`embedded` 的 repair 在打分层就偏好“低鲁棒能耗风险”的无人机方案，不是先出解再筛。

## 6. Evaluator 详细流程（embedded 的目标函数与可行性）

`SimulatedAnnealingALNS` 对候选解调用 `evaluator.evaluate_solution(candidate)`，其内部是 `evaluate_with_details(...)`。

评估主流程：

1. 计算 truck timing。
2. 计算 drone timing 与 rendezvous。
3. 进行最多两轮 truck/drone 同步迭代，确保卡车离开时刻不早于无人机回收时刻。
4. 最终重算 drone timing（与最终 truck 时刻一致）。
5. 计算延迟成本、truck/drone 距离成本。
6. 执行结构性可行性检查：
   - 时间窗硬约束
   - anchor 冲突（当前可放宽）
   - 无人机数量/序列一致性约束
   - 客户覆盖约束
   - forced-drone 约束
7. 执行鲁棒能耗检查（`RobustnessChecker`）。
8. 若任一约束失败：`total_cost = inf`；否则 `total_cost = base_cost`。

### 6.1 延迟成本函数

对每个延迟节点使用指数型惩罚，再乘全局系数：

- `f(tau) = exp(1.5031 + 7.032*tau) - exp(1.5031)`
- `total_delay = delay_penalty_cost * sum(f(tau_i))`

### 6.2 RobustnessChecker 的 Gamma 逻辑

`RobustnessChecker.check(...)` 对每个 drone task 执行能耗鲁棒评估：

- 先从 task timing 得到分段飞行时长
- 用 `DroneEnergyModel` 计算分段能耗
- 偏差率为 `energy_deviation_rate`
- 预算 `Gamma` 为整数（非整数会抛错）
- 计算最坏能耗并与 `battery_capacity` 比较
- 聚合为 `robustness.feasible` 与 margin

在 `embedded` 中，这个 checker 使用的正是搜索实例上的配置（即完整 Gamma），因此候选成本本身已经是“鲁棒可行性强约束后的有效成本”。

## 7. SA 主循环（embedded）

`SimulatedAnnealingALNS.run(initial)` 主要步骤：

1. `current = best = initial.clone()`；评估初始成本。
2. 初始化温度 `T0`：
   - 若成本有限：`delta = (w_percent/100)*|cost|`
   - `T0 = -delta / ln(0.5)`（下限 `temperature_min`）
   - 若初始不可行，给默认较高温度。
3. 每轮迭代：
   - AOS 选择 destroy 与 repair
   - 采样销毁配额 `beta`
   - destroy + repair 生成 candidate
   - 直接 evaluator 评估 candidate 成本
   - 用 SA 接受准则决定是否接纳 candidate 为 current
   - 若 candidate 优于 best，则更新 best
   - 周期性 local search / intensify / path relinking
   - 更新算子权重（AOS）
   - 降温 + 触发 reheat / diversification（按配置）

### 7.1 销毁配额采样

基础范围：

- `lower = max(3, r_L * |C|)`
- `upper = min(base_cap, r_U * |C|)`

重热阶段会按 `quota_multiplier` 等参数放大上界与采样值。

### 7.2 SA 接受规则

设 `delta = candidate_cost - current_cost`：

- 若 candidate 成为全局更优：直接接受（reward=`global`）
- 若 `delta <= 0`：接受（按改善幅度分 `better` 或 `slight_better`）
- 若 `delta > 0`：按 `exp(-delta / T)` 概率接受（reward=`accepted_worse`）
- 否则拒绝

### 7.3 AOS 权重更新

destroy/repair 各自维护权重，按奖励与时间归一化更新：

- `new_w = (1-eta)*old_w + eta*target_reward`

并支持历史平滑与衰减。

## 8. embedded 中“没有发生”的事情（很重要）

在 `embedded` 下，以下机制默认不参与：

- `candidate_subroute_verifier`（changed-subroute 预检查）不启用
- `robust_verifier` 不启用
- `robust_check_on_new_best` 不启用
- `robust_check_every` 固定 0，不做周期性额外鲁棒二次评估

这意味着 embedded 是“单路径评估”：候选只走一次 evaluator 路径，不走 precheck/new-best 额外门控路径。

## 9. 结果输出

`run_alns.py` 对 `embedded` 输出：

- `Cost`（搜索 evaluator 总成本）
- 运行时间
- `Feasible`
- truck routes
- drone tasks

注意：在 `embedded` 模式下不会单独打印 “Conservative Robust Cost”，因为没有构造独立 `robust_evaluator`。

## 10. 与 precheck_guarded 的本质区别（对照结论）

`embedded`：

- 搜索中直接用完整 Gamma
- repair 候选能量计算也用完整 Gamma
- evaluator 每轮直接按完整鲁棒约束给出成本/不可行
- 单路径评估

`precheck_guarded`：

- 搜索 evaluator 用 `gamma=0`（确定性）
- repair 用 `verification` 模式（当前实现约 0.67*Gamma 的引导）
- 候选阶段可走 changed-subroute 鲁棒预筛
- new-best 阶段再做鲁棒可行性门控

因此两者从搜索早期就处在不同目标面与不同剪枝机制上，不可能保证“轨迹一致”或“成本完全复刻”。

## 11. 一句话总结

当前代码里的 `embedded` 不是“后验验证型鲁棒”，而是把鲁棒预算 Gamma 直接嵌入 repair 候选生成与 evaluator 可行性判定，形成全程鲁棒驱动的 ALNS 搜索。

## 12. Local Search / Intensify / Path Relinking 细节（embedded 实际会执行）

`embedded` 与 `precheck_guarded` 在 SA 框架上共用这套增强模块；区别只在鲁棒判定路径。当前配置下（`config/alns_config.yaml`）：

- `local_search.frequency = 10`
- `local_search.on_new_best = true`
- `local_search.intensify_frequency = 35`
- `local_search.path_relinking_prob = 0.0`（因此路径重连分支通常不会触发）

### 12.1 `on_new_best` 本地搜索

当候选成为新 best 时会触发 `self._local_search(best.clone())`，只在改进且可行时接纳。

### 12.2 周期性本地搜索

每 `local_search_frequency` 轮触发 `_local_search(current.clone())`，改进后更新 `current`，必要时可更新 `best`。

### 12.3 周期性强化搜索

每 `intensify_frequency` 轮触发 `_intensify_search(best.clone())`。该函数执行：

1. 完整 `_local_search`
2. `_cross_exchange`（跨路径片段交换）
3. `_string_relocate`（字符串重定位）
4. 再次 `_optimize_truck_route`
5. 两轮 `_try_depot_drone_tasks`

### 12.4 Path Relinking

若 `path_relinking_prob > 0` 且随机命中，执行 `_path_relinking(current, best)`。当前默认配置为 0，因此一般不会启用。

## 13. 默认参数快照（当前仓库配置）

下列参数直接影响 embedded 搜索行为：

- 迭代：`small=2000`, `large=4000`
- SA 温度：`w_percent=15.0`, `temperature_min=1e-4`
- 冷却：`rate_initial=0.9995`, `rate_final=0.985`, `transition_iters=800`
- 重热：`stall_trigger=450`, `duration=50`, `temperature_scale=0.3`
- 配额：`r_lower=0.15`, `r_upper_small=0.5`, `r_upper_large=0.3`, `base_cap=30`
- AOS：`eta=0.35`, `alpha_credit=0.60`, `probability_floor=0.03`
- Local search：`frequency=10`, `intensify_frequency=35`
- 鲁棒：`battery=6.3`, `Gamma=3`, `deviation=0.1`, `same_truck_retrieval=false`
- 放宽：`allow_multiple_launch_per_node=true`, `allow_anchor_conflict=true`

## 14. 代码锚点索引（按调用链）

### 14.1 入口与策略分叉

- `run_alns.py`：策略切换与 `gamma(search)` 设置
- `run_alns.py`：`embedded` 不启用 `robust_evaluator/subroute_verifier`
- `run_alns.py`：`SimulatedAnnealingALNS(...)` 构造参数

### 14.2 算子构建

- `run_alns.py`：`build_operators(...)` 声明与算子池定义
- `run_alns.py`：destroy 集合（Random / WorstDistance / Shaw）
- `run_alns.py`：repair 集合（Cheapest / Regret / Biased / Equal / DronePriorityRegret / TruckFirst）

### 14.3 Repair 关键逻辑

- `alns_vrpfd/core/operators/repair.py`：`_generate_candidates(...)`
- `alns_vrpfd/core/operators/repair.py`：`_filter_valid_candidates(...)`
- `alns_vrpfd/core/operators/repair.py`：`_normalise(...)` 与 score 计算
- `alns_vrpfd/core/operators/repair.py`：`_worst_case_energy(...)`（embedded 取完整 Gamma）
- `alns_vrpfd/core/operators/repair.py`：`_budgeted_sum(...)`

### 14.4 SA 主循环

- `alns_vrpfd/core/sa.py`：构造函数参数与鲁棒开关
- `alns_vrpfd/core/sa.py`：主循环中 candidate 生成与 evaluator 调用
- `alns_vrpfd/core/sa.py`：`_acceptance(...)`
- `alns_vrpfd/core/sa.py`：`_sample_quota(...)`
- `alns_vrpfd/core/sa.py`：周期性 local search/intensify/path relinking 触发点

### 14.5 Evaluator 与 Robustness

- `alns_vrpfd/evaluation/evaluator.py`：`evaluate_with_details(...)` 主流程
- `alns_vrpfd/evaluation/evaluator.py`：可行性聚合与 `total_cost=inf` 规则
- `alns_vrpfd/evaluation/robustness.py`：`RobustnessChecker.check(...)`
- `alns_vrpfd/evaluation/robustness.py`：`assess_drone_task_energy(...)` 与 Gamma 分层计算
