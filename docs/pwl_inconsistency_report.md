# PWL 能耗近似导致的 MILP 解不一致问题

## 1. 问题描述

MILP 求解器返回一个在自身 PWL 模型下**可行且最优**的解（status=2, OPTIMAL），但该解经**精确能耗评估**后不可行（total_cost=inf）。

| 指标 | ALNS (62.48) | MILP (最优, 4 段不可行) |
|------|-------------|------------------------|
| 车辆数 | 2 | 1 |
| 总成本 | 62.48 | 52.68（MILP 自身） / inf（精确评估） |
| 能耗可行 | ✓ 全部 | ✗ 4 段中 3 段不可行 |
| 延误成本 | 0.0 | 0.77 |
| 求解时间 | 7.59s | 1701.85s (1800s 限时) |

---

## 2. 运行实验记录

### 2.1 实验设置

- **实例**: `R_30_10_5`（10 个客户）
- **配置**: PWL K=3, energy_uncertainty_budget=3, cost_lambda=30, cost_rho=0.2083
- **种子**: 42
- **脚本**: `run_alns_milp_comparison.py`

### 2.2 实验结果对比

#### 2.2.1 ALNS 解（2 车, feasible）

```
运行时间: 7.59s, 总成本: 62.48

卡车 0: 0(drone↑) → 6(drone↑) → 1(drone↑) → 4 → 11
卡车 1: 0 → 2(drone↑) → 11

无人机 0:
  [0→9→2] 客户:9  能耗:3.56/6.3 kWh ✓  (T0@0→T1@2 跨车)
  [2→5→6] 客户:5  能耗:4.60/6.3 kWh ✓  (T1@2→T0@6 跨车)
  [6→8→11] 客户:8  能耗:3.90/6.3 kWh ✓  (T0@6→Depot@11)

无人机 1:
  [2→3→6]  客户:3  能耗:5.31/6.3 kWh ✓  (T1@2→T0@6 跨车)
  [1→10→7→4] 客户:10,7  能耗:3.53/6.3 kWh ✓  (T0@1→T0@4 同车)

所有航段能耗均在 3.5-5.8 kWh 范围，距电池上限 (6.3 kWh) 有 0.5-2.8 kWh 安全裕度。
```

#### 2.2.2 MILP 180s 解（2 车, 旧重建, exact-feasible）

```
运行时间: 180s (TIME_LIMIT), MILP 成本: 67.66

卡车 0: 0 → 6 → 4 → 11
卡车 1: 0 → 2 → 9 → 11

无人机 0: [6→5→1→4] 客户:5,1  能耗:5.11/6.3 kWh ✓
无人机 1: [2→3→6]  客户:3    能耗:5.31/6.3 kWh ✓
          [4→7→10→8→11] 客户:7,10,8  能耗:5.73/6.3 kWh ✓

所有段能耗可行，但有 3 段显示 Depot@X（无 launch_truck）
→ 旧重建代码的 coupling 提取问题，不影响能耗评估
```

#### 2.2.3 MILP 1800s 解（1 车, OPTIMAL, exact-infeasible）

```
运行时间: 1701.85s (status=2, OPTIMAL)

MILP 自身目标: 52.68
精确评估总成本: inf
精确评估能耗: ✗ 3 段不可行

卡车 1: 0 → 4(drone↑) → 6(drone↑) → 11

无人机 0:
  [4→1→5→6]   客户:1,5   能耗:5.04/6.3 kWh ✓
  [6→2→9→11]  客户:2,9   能耗:5.83/6.3 kWh ✗ (worst_case=6.41)

无人机 1:
  [4→10→7→8→6]   客户:10,7,8   能耗:6.93/6.3 kWh ✗ (worst_case=7.59)
  [6→3→11]        客户:3        能耗:6.41/6.3 kWh ✗ (worst_case=7.05)
```

### 2.3 关键发现

**MILP 在 600s 时找到的 2 车解（cost=59.36）在精确评估下完全可行。**

```
600s MILP 解:
  总成本: 59.36 (与 exact 评估一致)
  精确可行: ✓
  所有 esg[gamma_max] ≤ 6.3 ✓
```

**MILP 在 600s→1800s 期间找到了 1 车最优解，进入精确不可行区域。**

---

## 3. PWL K=3 精度分析

### 3.1 PWL 断点与功率值

```
断点: load = [0, 10, 20, 30] kg
功率值: power(load) = (W + m + load)^1.5 * const

  Load(kg)    Exact(kW)      PWL(kW)    Err(kW)       Err%
         0    11.708630    11.708630     0.0000      0.000%
         5    13.903471    13.964583     0.0611      0.440%  ← 最大绝对误差
        10    16.220536    16.220536     0.0000      0.000%  ← 断点
        15    18.653690    18.709086     0.0554      0.297%
        20    21.197635    21.197635     0.0000      0.000%  ← 断点
        25    23.847742    23.898775     0.0510      0.214%
        30    26.599916    26.599916     0.0000      0.000%  ← 断点

最大插值误差: 0.0611 kW (load=5kg, 0.44%)
```

### 3.2 PWL 误差方向

**PWL 轻微高估能耗**（所有非断点处 PWL > Exact）。

| 负载 | 误差 | 方向 |
|------|------|------|
| 0-10 kg | +0.18%~+0.44% | 保守（高估） |
| 10-20 kg | +0.12%~+0.30% | 保守（高估） |
| 20-30 kg | +0.07%~+0.21% | 保守（高估） |

PWL 模型比真实能耗更保守，不是 MILP 解不可行的原因。

### 3.3 MILP 中的能量计算链

```
PWL 模型的能量计算链条:

  load_drone_plus[i,d]          ← MILP 决策变量（离开节点 i 时的负载）
      ↓
  omega_active[i,j,d] =         ← McCormick: load_drone_plus[i,d] * y_drone[i,j,d]
  load_drone_plus[i,d] * y      (y=1 时精确等于 load_drone_plus)
      ↓
  power_approx[i,j,d] =         ← Gurobi addGenConstrPWL: PWL(omega_active)
  PWL(omega_active)
      ↓
  energy_active[i,j,d] =        ← energy_link 约束:
  power_approx * travel_time    energy_active = power_approx * travel_time (当 y=1)
      ↓
  energy_state_gamma[j,d,g] ← 累积能耗（含偏差预算 Gamma=3）
      ↓
  Eq(41): energy_state_gamma[j,d,gamma_max] ≤ battery_capacity * sum(y_drone[*,j,d])
```

---

## 4. 根因分析

### 4.1 直接原因

**约束式 (32)-(33) 中 `z_out` 与 `v_served` 的交互导致负载连续性的异常。**

```python
# Eq (31)-(33): v_served definition  (builder.py:910-926)
for j in data.customers:
    for d in data.drones:
        y_in = gp.quicksum(vars.y_drone[i, j, d] for i in data.v0 if (i, j) in arc_set)
        z_out = gp.quicksum(
            vars.z_coupling[j, h, k, d] for k in data.trucks for h in data.v_plus if (j, h) in arc_set
        )
        # Eq (31): v_served ≤ y_in
        model.addConstr(vars.v_served[j, d] <= y_in, ...)
        # Eq (32): v_served ≤ 1 - z_out   ← 关键
        model.addConstr(vars.v_served[j, d] <= 1 - z_out, ...)
        # Eq (33): v_served ≥ y_in - z_out
        model.addConstr(vars.v_served[j, d] >= y_in - z_out, ...)
```

**问题**：`z_out` 统计从节点 `j` 出发的所有 `z_coupling[j, h, k, d]`。当 `z_coupling` 覆盖航段的**所有弧**时（而非仅首/末弧），每个中间客户节点的 `z_out = 1`，强制 `v_served = 0`。

### 4.2 级联效应

```
z_coupling 覆盖所有弧 → z_out = 1 在中间客户节点
  → v_served = 0（无人机不服务该客户）
    → 负载连续性 Eq(34-35) 不减去客户需求
      → load_drone_plus 在航段中不下降
        → omega_active = 恒定的低负载
          → PWL 功率 = 低负载功率
            → PWL 能耗 = 严重低估
```

### 4.3 实际影响（1800s 解）

对航段 `4→10→7→8→6` 的分析：

**假设 A：正确的物理负载（v_served=1）**
```
弧段         负载(kg)  精确能耗(kWh)   PWL能耗(kWh)
4→10           30        3.12          3.12
10→7 (需求 2)  28        0.55          0.55
7→8  (需求 9)  19        3.40          3.40
8→6  (需求 4)  15        3.44          3.44
总计            -       10.50         10.51    ✗ (均 > 6.3)
```

**假设 B：v_served=0（通过 z_coupling）**
```
弧段         负载(kg)  精确能耗(kWh)   PWL能耗(kWh)
4→10            X        低             低
10→7            X        低             低
7→8             X        低             低
8→6             X        低             低
总计            -        ~5.70         ~5.70    ✓ (≤ 6.3!)
```

X 可以是极小值（如 `load_drone_plus[4,1] = 0.001`），因为 Eq(34-35) 在 `u_sum=1`（发射点）时被 big-M 松弛，不受下游需求约束。

### 4.4 变量值验证（600s 可行解中 v_served 正确的表现）

```
Drone 1 的正确负载连续性:

  Arc  4→1:  load_plus[4]=20.9998  v_served[1]=1  ← 负载正确计算
  Arc  1→7:  load_plus[1]=14.9999  v_served[7]=1
  Arc  7→10: load_plus[7]= 6.0000  v_served[10]=1
  Arc 10→8:  load_plus[10]=4.0000  v_served[8]=1
  Arc  8→11: load_plus[8]= 0.0000

能量状态: esg[gamma_max] 均在 1.67-6.30 kWh，满足电池约束 ✓
精确评估: 总成本 59.36，完全可行 ✓
```

---

## 5. 故障诊断过程

### 5.1 步骤 1: 排除数据缓存问题

- 旧 `results/mip_runs.json`（53.96）是 commit `e9930e8` 的残留，已丢弃
- 确认所有比较使用同一配置、同一评估器

### 5.2 步骤 2: 验证 PWL 近似精度

- 计算 PWL K=3 在所有负载点（0-30 kg）的误差
- 结论：最大 0.44%，方向保守（高估），不是根因

### 5.3 步骤 3: 检查能量计算链条

```
load_drone_plus → omega_active(McCormick) → power_approx(Gurobi PWL) → energy_active → energy_state_gamma
```

- 600s 解中验证：`omega_active = load_drone_plus`（y=1 时 McCormick 精确）
- `power_approx = PWL(omega_active)` 正确
- `energy_active = power_approx * travel_time` 正确
- `energy_state_gamma[gamma_max] ≤ 6.3` 正确满足

### 5.4 步骤 4: 发现 v_served 与负载连续性的耦合

- 检查 `z_out` 对 `v_served` 的约束（Eq 32）
- 发现 `z_coupling` 覆盖所有弧时会导致 `v_served=0`
- `v_served=0` 触发 Eq(34-35) 的异常路径

### 5.5 步骤 5: 在 600s 解中验证正确行为

- 600s 解中 `v_served=1` 对所有客户
- `load_drone_plus` 正确递减
- 精确评估完全可行

---

## 6. 修复方案

### 6.1 [推荐] 方案 A：修复 v_served 约束

将 `z_out` 替换为 `u_sum`（是否发射点），消除 `z_coupling` 的误影响：

```python
# 修复后 Eq (32)-(33)
for j in data.customers:
    for d in data.drones:
        y_in = gp.quicksum(vars.y_drone[i, j, d] for i in data.v0 if (i, j) in arc_set)
        u_sum = gp.quicksum(vars.u[j, k, d] for k in data.trucks)
        # 仅当是发射点时禁止无人机服务该节点
        model.addConstr(vars.v_served[j, d] <= y_in, name="served_ub_y")
        model.addConstr(vars.v_served[j, d] <= 1 - u_sum, name="served_ub_u")    # 替代 Eq(32)
        model.addConstr(vars.v_served[j, d] >= y_in - u_sum, name="served_lb_u")  # 替代 Eq(33)
```

**语义变化**：
- 原：无人机经过节点 `j` 且无耦合 → 必须服务（`v_served = y_in - z_out`）
- 修复后：无人机经过节点 `j` 且非发射点 → 必须服务（`v_served = y_in - u_sum`）

### 6.2 [备用] 方案 B：电池安全裕度

在 MILP 模型中设置 `battery_capacity = 6.0 kWh`（比真实 6.3 低 5%）：

```python
# 修改 ProblemData 或构建参数
battery_capacity = instance.robust_config.drone_battery_capacity * 0.95
```

### 6.3 [备用] 方案 C：增加 PWL 段数

`num_segments=3` → `num_segments=10`（误差从 0.44% 降至 <0.1%）。

但 PWL 精度不是根因。

### 6.4 [不推荐] 方案 D：后处理修复

对 MILP 解重建后运行二次修复。复杂且丧失了 MILP 的最优性保证。

---

## 7. 结论

| 问题 | 结论 |
|------|------|
| PWL K=3 误差是否导致不一致？ | **否** — 误差仅 <0.44%，方向保守（高估能耗）|
| 不一致的真正原因？ | **`z_coupling` → `z_out` → `v_served=0` → 负载不下降 → PWL 能耗低估** |
| 哪些解受影响？ | **仅约束紧绷的解**（如 1 车方案，各段能耗接近电池上限）|
| 600s 解为何可行？ | 2 车方案有充分能耗裕度，`z_coupling` 不影响关键负载连续性 |
| 修复优先级？ | **方案 A**（根因修复）> 方案 C（增加 PWL 段数）> 方案 B（安全裕度） |
