# 仓库执行原则（Repository Execution Principles）

## 1) 定位与目的

本文件是仓库级执行标准，统一约束：

- 代码与实验如何运行；
- 敏感性实验如何聚合；
- 图表与结论如何对应；
- 论文目录同步的操作顺序。

本文件已替代原 `sensitivity/data_mapping.md`。

---

## 2) 仓库结构（按执行视角）

### 2.1 入口脚本

- `run_alns.py`：统一 ALNS 入口（敏感性脚本复用 `build_operators`）。
- `setup_path.py`：统一导入路径配置。
- `sensitivity/*.py`：当前主实验入口（battery / drone_count / speed / payload / gamma / flexibility / replay）。

### 2.2 核心模块

- `alns_vrpfd/core`：算子、退火、搜索逻辑。
- `alns_vrpfd/model`：解结构与构造。
- `alns_vrpfd/instance`：算例与领域对象。
- `alns_vrpfd/mip`：精确模型相关。
- `alns_vrpfd/evaluation`：可行性与成本评估。
- `alns_vrpfd/utils`：配置与 IO。

### 2.3 实验与分析

- `sensitivity`：敏感性脚本、replay、plotter。
- `sensitivity/results_new`：新实验结果主目录。
- `sensitivity/results_new/*/legacy`：历史旧版本结果（归档，不作为当前主口径输入）。
- `sensitivity/plotter`：绘图脚本。
- `sensitivity/plotter` 仅保留当前有效绘图脚本（不再保留 `legacy` 子目录）。

### 2.4 数据与结果

- `data/Instance*`：基准算例。
- `results/`：优化输出、基线结果。

---

## 3) 环境与导入规则

- 新脚本若导入 `alns_vrpfd`，必须先写：

```python
import setup_path  # noqa
```

- 实验与绘图默认环境（强制）：

```bash
source /Users/minz/anaconda3/bin/activate base
```

- 后续执行敏感性实验、replay、绘图脚本时，均默认先激活上述 `base` 环境。

- 建议环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

- 测试命令：

```bash
python -m pytest alns_vrpfd/tests
```

---

## 4) 运行规范

### 4.1 基础运行

```bash
python run_alns.py data/Instance10/R_50_10_4.txt --iterations 500
```

### 4.2 实验模块运行

```bash
python -m alns_vrpfd.experiments.<module_name>
```

### 4.3 敏感性脚本统一参数

统一支持：

- `--instance-dir`（可重复）
- `--instance-scope all|region|single`
- `--regions 30,40,50`（scope=region）
- `--instance-name`（scope=single）
- `--trials`
- `--skip-baseline`
- `--append`

---

## 5) 敏感性统计口径（强制）

### 5.1 仅一个平均层级

只允许在“跨算例”层做平均。

每个参数水平的流程：

1. 每个 `(instance, level)` 跑 `k` 次；
2. 在组内取 best-of-k：`best_cost` 最小；并列时 `best_drone_customers` 更大者优先；
3. 用这条记录计算指标；
4. 对同一水平跨算例平均，输出 `summary.csv`。

### 5.2 绘图数据来源

- 绘图脚本只读 `summary.csv`；
- 禁止在绘图脚本硬编码数据；
- 数据更新顺序：先跑实验生成 summary，再绘图。

---

## 6) 当前敏感性数据映射

| 类别 | Summary CSV | 绘图脚本 | 输出图 |
| :--- | :--- | :--- | :--- |
| Gamma | `sensitivity/results_new/gamma_sensitivity/gamma_summary.csv` | `sensitivity/plotter/plot_gamma_sensitivity.py` | `sensitivity/results_new/gamma_sensitivity/gamma_sensitivity_plot_large.pdf` |
| Battery | `sensitivity/results_new/battery_sensitivity_rerun_in_25/battery_sensitivity_15inst_summary.csv` | `sensitivity/plotter/plot_battery_sensitivity.py` | `sensitivity/results_new/battery_sensitivity_rerun_in_25/battery_sensitivity_15inst_final.pdf` |
| Drone Count | `sensitivity/results_new/drone_count/drone_count_summary.csv` | `sensitivity/plotter/plot_drone_count_sensitivity.py` | `sensitivity/results_new/drone_count/drone_count_sensitivity_plot_large.pdf` |
| Drone Speed | `sensitivity/results_new/drone_speed/drone_speed_summary.csv` | `sensitivity/plotter/plot_drone_speed_sensitivity.py` | `sensitivity/results_new/drone_speed/drone_speed_sensitivity_plot_large.pdf` |
| Payload | `sensitivity/results_new/drone_payload/drone_payload_summary.csv` | `sensitivity/plotter/plot_drone_payload_sensitivity.py` | `sensitivity/results_new/drone_payload/drone_payload_sensitivity_plot_large.pdf` |
| Flexibility | `sensitivity/results_new/drone_flexibility/docking_flexibility_region_vertical_t5_summary.csv` | `sensitivity/plotter/plot_docking_flexibility.py` | `sensitivity/results_new/drone_flexibility/docking_flexibility_instance_singlepanel_t5.pdf` |

### 6.1 新旧结果分层（强制）

- 当前主口径结果：各元素目录根层（例如 `sensitivity/results_new/drone_speed/*.csv`）。
- 历史旧版结果：统一放在对应元素下 `legacy/` 子目录。
- 论文与主分析默认只读取根层当前结果，不读取 `legacy/`。

### 6.2 Plotter 脚本唯一入口（强制）

- `sensitivity/plotter` 顶层每个元素仅保留一个脚本入口：
  - `plot_battery_sensitivity.py`
  - `plot_drone_count_sensitivity.py`
  - `plot_drone_speed_sensitivity.py`
  - `plot_drone_payload_sensitivity.py`
  - `plot_gamma_sensitivity.py`
  - `plot_docking_flexibility.py`
- 不再保留 `sensitivity/plotter/legacy/`，旧版绘图脚本已移除。

---

## 7) Same vs Flex（Instance25）执行标准

### 7.1 范围一致性（强制）

- 若结论声明为 `Instance25`，图和文字必须都来自 `Instance25`。
- 禁止用 `Instance10` 图支持 `Instance25` 结论。

### 7.2 脚本与参数

脚本：`sensitivity/docking_flexibility_comparison.py`

支持：

- `--instance-scope all|region|single`
- `--regions 30,40,50`
- `--instance-name`
- `--trials`、`--seed-start`、`--seeds`
- `--dry-run`

建议按区域逐步执行：

```bash
# 先 dry-run 检查筛选
python3 sensitivity/docking_flexibility_comparison.py \
  --instance-dir data/Instance25 \
  --instance-scope region \
  --regions 40 \
  --trials 10 \
  --seed-start 20260221 \
  --output-prefix docking_flexibility_i25_r40 \
  --dry-run

# 正式运行
python3 sensitivity/docking_flexibility_comparison.py \
  --instance-dir data/Instance25 \
  --instance-scope region \
  --regions 40 \
  --trials 10 \
  --seed-start 20260221 \
  --output-prefix docking_flexibility_i25_r40
```

### 7.3 输出文件策略（强制）

每个 prefix 仅保留两类输出：

- 完整明细：`<prefix>_results.csv`
- 绘图汇总：`<prefix>_summary.csv`

其中 `flex` 的 `summary.csv` 采用**主口径**：

- 在同一算例的多次 trial 中按 `saving` 取最大值作为代表记录；
- 再在该口径下输出 instance / region / ALL 的统计汇总。

---

## 8) 绘图规则

- 与 battery/drone count 视觉规范保持一致；
- 标注需避免重叠（必要时做偏移与避让）；
- 纵轴预留上下边界避免文字出框；
- 至少输出 PDF（PNG 视需求追加）。

### 8.1 全局统一配色（强制）

仅使用红蓝双色系（深色描边 + 浅色填充）：

- `barRedBorder = #E38DB3`
- `barRedFill = #F6DBE6`
- `barBlueBorder = #3886C2`
- `barBlueFill = #CFECF6`

使用规则：

- Baseline / Same-truck 柱：`barRedFill` 填充，`barRedBorder` 描边；
- 改进方案（如 Flexible）柱：`barBlueFill` 填充，`barBlueBorder` 描边；
- Cost-saving 点/线默认用 `barBlueBorder`；
- 平均参考线（如 Average saving）用 `barRedBorder` 虚线；
- 禁止引入第三主色（如橙/绿）作为主编码色。

---

## 9) 基线文件约定

- MIP 基线目录：`results/MIPresult/`
- 文件格式：`output_{customers}_{fleet}_{instance}.txt`

---

## 10) 论文目录联动规则（强制）

论文目录：

- `/Users/minz/Desktop/Manuscript-ALNS-June2025`
- 论文仓库路径（固定）：`/Users/minz/Desktop/Manuscript-ALNS-June2025`
- 主文稿文件：`/Users/minz/Desktop/Manuscript-ALNS-June2025/main.tex`

将任何新结果写入论文目录前必须：

1. 先提醒打开 Overleaf 同步；
2. 仅在明确确认后执行覆盖/复制。

---

## 11) 变更同步原则

若实验口径、绘图标准、输出文件策略发生变更，必须在同一变更中同步更新本文件。
