# Deadline 与剥夺成本修改方案

## 核心思路

原算例仅有 `q_i`（需求量），无物资类别。**`q_i` 仅用于载重/能耗计算，不再表征紧迫性。** 另外生成物资类别 `c_i`，由 `c_i` 决定 deadline 和剥夺成本权重。

---

## 1. 物资类别 `c_i`

| 类别 | 名称 | 示例 | 紧迫性 |
|------|------|------|--------|
| C1 | Health / life-critical | 药品、急救包、疫苗 | 最高 |
| C2 | WASH / survival-critical | 饮用水、净水片 | 高 |
| C3 | Food / basic relief | 食品、营养包 | 中 |
| C4 | Shelter / bulk relief | 帐篷、毛毯、睡袋 | 低 |

分类依据：Sphere Handbook (2018) 将人道主义响应标准按 WASH、Food security、Shelter、Health 划分；Lin et al. (2020) 按 medicine > water > food 设优先级；Huang & Rafiei (2015) 区分 urgent items (water) 和 non-urgent items (tent)；Wang et al. (2017) 分别估计不同物资的 deprivation level functions。

**类别生成**：独立于 `q_i`，均匀随机分配（各25%）。不用 `q_i` 分位数推断——Shao et al. (2020) 指出 demand type 和 demand quantity 是两个独立维度，不应混为一谈。

**敏感性分析**：另测 Life-critical-heavy (C1:35%, C2:30%, C3:25%, C4:10%) 和 Bulk-relief-heavy (C1:10%, C2:20%, C3:35%, C4:35%) 两组情景。

---

## 2. Deadline

$$r_i = \min\{t_{0i}^T, t_{0i}^D\}, \quad o_i = r_i + \Delta^o_{c_i}, \quad l_i = o_i + \Delta^l_{c_i}$$

| 类别 | $\Delta^o_c$ (h) | $\Delta^l_c$ (h) |
|------|-------------------|-------------------|
| C1 | U(0.10, 0.30) | U(0.20, 0.50) |
| C2 | U(0.30, 0.70) | U(0.50, 1.00) |
| C3 | U(0.70, 1.30) | U(1.00, 1.80) |
| C4 | U(1.20, 2.00) | U(1.50, 2.50) |

`r_i` 保证 deadline 物理可达；类越紧急 → `Δ^o`/`Δ^l` 越小 → 窗口越窄。

---

## 3. 剥夺成本

$$f_i(\tau) = \kappa_{c_i}\left(e^{1.5031+7.032\tau} - e^{1.5031}\right)$$

| 类别 | $\kappa_c$ |
|------|-----------|
| C1 | 3.0 |
| C2 | 2.0 |
| C3 | 1.0 |
| C4 | 0.4 |

底层函数保留 Holguín-Veras et al. (2013) 的凸形式；$\kappa_c$ 引入类别差异：Wang et al. (2017) 不同物资不同函数；Huang & Rafiei (2015) water 紧 tent 松；Lin et al. (2020) medicine > water > food 优先级。

---

## 4. 论文修改位置

- **Section 3.1 (L220–224)**：删去 `q_i` 决定 deadline 的描述，引入 `c_i` 及类别加权剥夺成本
- **Objective function (L330)**：`f(τ)` → `f_i(τ)`
- **Section 4.1 (L588)**：删去 demand-based deadline 生成策略，改为类别生成方案
- **Sensitivity analysis**：新增类别占比敏感性分析

---

## 5. 新增参考文献

| cite key | 文献 | 用途 |
|----------|------|------|
| `shao2020multiobjective` | Shao et al. (2020) | demand type ≠ demand quantity |
| `wang2017deprivation` | Wang et al. (2017) | 不同物资不同 deprivation 函数 |
| `huangModelingMultipleHumanitarian2015` | Huang & Rafiei (2015) | water urgent / tent non-urgent |
| `lin2020multiobjective` | Lin et al. (2020) | medicine > water > food 优先级 |
| `sphere2018handbook` | Sphere Handbook (2018) | WASH/Food/Shelter/Health 分类标准 |