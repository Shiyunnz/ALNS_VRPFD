# Flexible Docking Full Metrics — 重跑方案

## 1. 背景与问题

### 1.1 当前状态

Appendix B 中 `tab:flexible_docking_per_instance` 目前只有 **n=25** 的 15 行数据，来自旧 sensitivity 实验（`sensitivity/results_new/drone_flexibility/docking_flexibility_i25_system_t3_best_by_instance.csv`）。

**三个问题：**
1. **缺规模**：n=50、75、100 没有数据
2. **Same/Flex 不一致**：旧实验是两次独立 ALNS 运行（不同 seed），导致 `flex_cross_truck=0` 时 `same_cost ≠ flex_cost`（如 R_50_25_3 差 135）
3. **没存解结构**：`comparison_results.json` 只存了 cost 和 cross_truck 计数，无法提取运营指标（truck wait、drone util、makespan 等）

### 1.2 已有基础设施

- **Dual-tracking**：`SA.cfg.track_no_cross_truck=True`，单次 ALNS 运行同时追踪：
  - `alns.best_solution` → 全局最优（可能有 cross-truck）
  - `alns._best_no_cross_truck_solution` → 最优零跨车解
  - 当 `flex_cross_truck=0` 时，两者是同一个解 → `same_cost == flex_cost` **自动保证**
- **解序列化函数**（在 `sensitivity/docking_flexibility_comparison.py:393-416`）：
  - `serialize_truck_routes(solution)` → JSON string
  - `serialize_drone_tasks(solution)` → JSON string
- **运营指标提取**（`alns_vrpfd/evaluation/` 中的 `Evaluator.evaluate_with_details()`）
- **指标→LaTeX 表**（`scripts/analyze_flexible_docking_operational_metrics.py`）

---

## 2. 新建脚本：`scripts/run_flexible_docking_full_metrics.py`

### 2.1 脚本目标

对指定的 instance size 和 seeds，运行 ALNS（dual-tracking），输出一个 CSV，每行包含：

| 字段 | 类型 | 来源 |
|------|------|------|
| `instance` | str | 实例文件路径，如 `data/Instance50/R_30_50_1.txt` |
| `instance_name` | str | 实例名，如 `R_30_50_1` |
| `region` | int | 区域大小，如 30 |
| `same_seed` | int | 同 seed |
| `flex_seed` | int | 同 seed（dual-tracking 单次运行） |
| `same_cost` | float | `evaluator.evaluate_solution(same_sol).total_cost` |
| `flexible_cost` | float | `evaluator.evaluate_solution(flex_sol).total_cost` |
| `flexible_saving_vs_same` | float | `(same_cost - flex_cost) / same_cost * 100` |
| `same_best_drone_customers` | int | `len(same_sol.drone_customers())` 或等价计算 |
| `flex_best_drone_customers` | int | 同上 |
| `same_truck_routes` | str | `json.dumps([{"truck_id":r.id,"nodes":list(r.nodes)} for r in same_sol.truck_routes])` |
| `same_drone_tasks` | str | `json.dumps([{...} for t in same_sol.drone_tasks])` |
| `flexible_truck_routes` | str | 同上，flex 版本 |
| `flexible_drone_tasks` | str | 同上，flex 版本 |

### 2.2 从哪个文件复制代码

从 `compare_fair_flexible_vs_same_all.py` 复制以下部分，不做改动：

- **导入**（第 1-34 行）：所有 import
- **CFG_PATH / BASE_OUTPUT**（第 36-37 行）
- **SIZE_CONFIGS**（第 40-44 行）：n=25 也要加进去
- **`_region_of()`**（第 47-48 行）
- **`load_instance()`**（第 51-70 行）：完全不变
- **`build_alns()`**（第 73-95 行）：完全不变
- **`_count_cross_truck()`**（第 98-100 行）：完全不变
- **main() 的循环结构**（第 138-218 行）：checkpoint 逻辑、SKIP 逻辑、进度打印

### 2.3 改写的部分

**`run_instance()` 函数**——这是核心改动。

```python
def run_instance(instance_name: str, size: int, seed: int) -> Dict[str, Any]:
    instance, evaluator = load_instance(instance_name, size)
    alns = build_alns(instance, evaluator, seed, size)
    from alns_vrpfd.model.initializer import build_two_phase_initial_solution
    initial = build_two_phase_initial_solution(instance)
    t0 = time.time()
    best = alns.run(initial, time_limit=TIME_LIMITS[size])
    runtime = time.time() - t0
    
    # --- Dual-tracking: 提取 same 和 flex 解 ---
    same_sol = alns._best_no_cross_truck_solution or best
    flex_sol = best
    
    # --- 评估 ---
    same_eval = evaluator.evaluate_solution(same_sol)
    flex_eval = evaluator.evaluate_solution(flex_sol)
    
    # --- 序列化解结构 ---
    # 使用 sensitivity/docking_flexibility_comparison.py 中的格式:
    # same_truck_routes = json.dumps([{"truck_id": r.id, "nodes": list(r.nodes)}
    #                                   for r in same_sol.truck_routes])
    # same_drone_tasks = json.dumps([{"drone_id": t.drone_id,
    #                                  "launch_truck": t.launch_truck,
    #                                  "launch_node": t.launch_node,
    #                                  "customers": list(t.customers()),
    #                                  "land_truck": t.land_truck,
    #                                  "retrieve_node": t.retrieve_node}
    #                                 for t in same_sol.drone_tasks])
    # (flex 同理)
    
    return {
        "instance": str(INSTANCE_DIRS[size] / f"{instance_name}.txt"),
        "instance_name": instance_name,
        "region": _region_of(instance_name),
        "same_seed": seed,
        "flex_seed": seed,
        "same_cost": same_eval.total_cost if same_eval.feasible else None,
        "flexible_cost": flex_eval.total_cost if flex_eval.feasible else None,
        "flexible_saving_vs_same": (same_eval.total_cost - flex_eval.total_cost) / same_eval.total_cost * 100
            if same_eval.feasible and flex_eval.feasible and same_eval.total_cost else 0.0,
        "same_best_drone_customers": _count_drone_customers(same_sol),
        "flex_best_drone_customers": _count_drone_customers(flex_sol),
        "same_truck_routes": _serialize_truck_routes(same_sol),
        "same_drone_tasks": _serialize_drone_tasks(same_sol),
        "flexible_truck_routes": _serialize_truck_routes(flex_sol),
        "flexible_drone_tasks": _serialize_drone_tasks(flex_sol),
    }
```

**新增辅助函数：**

```python
def _serialize_truck_routes(solution) -> str:
    """序列化卡车路径为 JSON 字符串。"""
    payload = []
    for route in solution.truck_routes:
        payload.append({"truck_id": route.id, "nodes": list(route.nodes)})
    return json.dumps(payload, ensure_ascii=False)

def _serialize_drone_tasks(solution) -> str:
    """序列化无人机任务为 JSON 字符串。"""
    payload = []
    for task in solution.drone_tasks:
        payload.append({
            "drone_id": task.drone_id,
            "launch_truck": task.launch_truck,
            "launch_node": task.launch_node,
            "customers": list(task.customers()),
            "land_truck": task.land_truck,
            "retrieve_node": task.retrieve_node,
        })
    return json.dumps(payload, ensure_ascii=False)

def _count_drone_customers(solution) -> int:
    """统计无人机服务的客户数。"""
    return len({node for task in solution.drone_tasks for node in task.customers()})
```

**新增常量：**

```python
SIZE_CONFIGS = {
    25: {"time_limit": 600, "instances": [f"R_{r}_25_{i}" for r in [30,40,50] for i in range(1,6)]},
    50: {"time_limit": 600, "instances": [f"R_{r}_50_{i}" for r in [30,40,50] for i in range(1,6)]},
    75: {"time_limit": 600, "instances": [f"R_{r}_75_{i}" for r in [30,40,50] for i in range(1,6)]},
    100: {"time_limit": 600, "instances": [f"R_{r}_100_{i}" for r in [30,40,50] for i in range(1,6)]},
}

INSTANCE_DIRS = {
    25: PROJECT_ROOT / "data" / "Instance25",
    50: PROJECT_ROOT / "data" / "Instance50",
    75: PROJECT_ROOT / "data" / "Instance75",
    100: PROJECT_ROOT / "data" / "Instance100",
}
```

**输出格式：**

CSV 文件保存到 `results/revision_experiments/flexible_docking_full_metrics/i{size}_best_by_instance.csv`，用 `csv.DictWriter` 写入，用 `csv.QUOTE_NONNUMERIC` 处理 JSON 字段中的逗号。

```python
import csv
csv_path = output_dir / f"i{size}_best_by_instance.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELD_NAMES, quoting=csv.QUOTE_NONNUMERIC)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
```

其中 `FIELD_NAMES`：
```python
FIELD_NAMES = [
    "instance", "instance_name", "region",
    "same_seed", "flex_seed",
    "same_cost", "flexible_cost", "flexible_saving_vs_same",
    "same_best_drone_customers", "flex_best_drone_customers",
    "same_truck_routes", "same_drone_tasks",
    "flexible_truck_routes", "flexible_drone_tasks",
]
```

### 2.4 不变的部分

- **并行 checkpoint 逻辑**：`main()` 里 `existing` 字典 + `SKIP` 逻辑完全复用
- **SA 配置**：`build_alns()` 里的 `track_no_cross_truck=True`、4000 iterations、`size="large"/"small"` 等
- **Instance 加载**：`load_instance()` 中的 `strategy="demand_based"`、`endurance=inf`、`same_truck_retrieval=False`

### 2.5 依赖的已有代码结构

```
SimulatedAnnealingALNS
├─ .run(initial, time_limit=N) → Solution
├─ .best_solution: Solution (全局最优)
├─ ._best_no_cross_truck_solution: Optional[Solution] (最优零跨车解)
└─ cfg.track_no_cross_truck: bool (=True 开启追踪)

Solution
├─ .truck_routes: List[TruckRoute]
├─ .drone_tasks: List[DroneTask]

TruckRoute
├─ .id: int
├─ .nodes: Sequence[int]
├─ .customers() → List[int] (排除 depot)

DroneTask
├─ .drone_id: int
├─ .launch_truck: Optional[int]
├─ .launch_node: int
├─ .land_truck: Optional[int]
├─ .retrieve_node: int
├─ .customers() → List[int]

Evaluator
├─ .evaluate_solution(sol) → EvalResult (.total_cost, .feasible)
├─ .evaluate_with_details(sol) → DetailedEval (truck_timings, drone_timings, delay_breakdown)
  └─ 在 analyze_flexible_docking_operational_metrics.py 的 _extract_metrics() 中使用
```

---

## 3. 运行计划

### 3.1 Seeds

| Size | Seeds | 理由 |
|------|-------|------|
| n=25 | 100, 101 | 2 seeds 足够生成运营指标 |
| n=50 | 100, 101 | 2 seeds |
| n=75 | 100, 101 | 2 seeds |
| n=100 | 100, 101 | 2 seeds |

### 3.2 预估时间

| Size | 算例数 | Seeds | Runs | 每轮 (4000 iters) | 合计 | tmux id |
|------|--------|-------|------|-------------------|------|---------|
| 25 | 15 | 2 | 30 | ~180s | ~90min | `fd25` |
| 50 | 15 | 2 | 30 | ~350s | ~175min | `fd50` |
| 75 | 15 | 2 | 30 | ~500s | ~250min | `fd75` |
| 100 | 15 | 2 | 30 | ~600s | ~300min | `fd100` |

**并行运行**：4 个 tmux session 同时跑，总 wall-clock ≈ 5 小时（受限于 n=100）。

### 3.3 启动命令

```bash
# tmux new -s fd25
cd /Users/minz/Desktop/ResearchProject/code
/Users/minz/anaconda3/bin/python scripts/run_flexible_docking_full_metrics.py --size 25 --seeds 100,101 2>&1 | tee results/revision_experiments/flexible_docking_full_metrics/fd25_run.log

# tmux new -s fd50 (同上, --size 50)
# tmux new -s fd75 (同上, --size 75)
# tmux new -s fd100 (同上, --size 100)
```

### 3.4 输出目录结构

```
results/revision_experiments/flexible_docking_full_metrics/
├── fd25_run.log
├── fd50_run.log
├── fd75_run.log
├── fd100_run.log
├── i25_best_by_instance.csv       # 30 rows (15×2)
├── i50_best_by_instance.csv       # 30 rows
├── i75_best_by_instance.csv       # 30 rows
└── i100_best_by_instance.csv      # 30 rows (可含 infeasible)
```

---

## 4. 后处理：运营指标与 LaTeX 表

### 4.1 运行 metrics 脚本

```bash
# 需要修改 analyze_flexible_docking_operational_metrics.py
# 使其接受 --input-csv 参数指向新生成的 CSV

for size in 25 50 75 100; do
    python scripts/analyze_flexible_docking_operational_metrics.py \
        --input-csv results/revision_experiments/flexible_docking_full_metrics/i${size}_best_by_instance.csv \
        --output-dir results/revision_experiments/flexible_docking_full_metrics/analysis_i${size}
done
```

### 4.2 `analyze_flexible_docking_operational_metrics.py` 需要的最小修改

该脚本的 `run()` 函数（第 586 行）需要一个 `list[csv.DictReader Row]`，每行需要有：

```
instance, instance_name, region, same_seed, flex_seed,
same_cost, flexible_cost, flexible_saving_vs_same,
same_best_drone_customers, flex_best_drone_customers,
same_truck_routes, same_drone_tasks,
flexible_truck_routes, flexible_drone_tasks
```

而新脚本输出的 CSV 字段名一致，所以 **不需要改字段名**。

需要确保 `_read_paired_rows()` 里 `row["instance"]` 指向实例文件路径（新脚本输出就是路径）。

### 4.3 `_extract_metrics()` 的调用

```python
metrics.append(
    _extract_metrics(
        instance_path=PROJECT_ROOT / row["instance"],
        instance_name=row["instance_name"],
        region=int(row["region"]),
        mode="same_truck",
        cost=float(row["same_cost"]),
        saving_pct=0.0,
        truck_json=row["same_truck_routes"],
        drone_json=row["same_drone_tasks"],
        same_truck_retrieval=True,
        cfg=cfg,
    )
)
metrics.append(
    _extract_metrics(
        instance_path=PROJECT_ROOT / row["instance"],
        instance_name=row["instance_name"],
        region=int(row["region"]),
        mode="flexible",
        cost=float(row["flexible_cost"]),
        saving_pct=float(row["flexible_saving_vs_same"]),
        truck_json=row["flexible_truck_routes"],
        drone_json=row["flexible_drone_tasks"],
        same_truck_retrieval=False,
        cfg=cfg,
    )
)
```

注意：`_extract_metrics` 内部会调用 `read_instance()` 重新加载实例文件 + `_solution_from_json()` 重建解对象。

### 4.4 生成的 LaTeX 表

`_write_latex_tables()` 会生成两个表：
1. **`tab:flexible_docking_operational_metrics`**：汇总指标（same vs flex 对比）
2. **`tab:flexible_docking_instance_characteristics`**：分区域 saving + 实例特征

但 Appendix B 需要的是 **per-instance 详细表**，`_write_latex_tables()` 生成的是汇总表。所以需要：

**新增函数** `_write_per_instance_table()`，生成如下格式：

```latex
\begin{sidewaystable}[htbp]
\centering
\caption{Per-instance operational metrics of flexible docking solutions.}
\label{tab:flexible_docking_per_instance}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrrrr}
\toprule
& & \multicolumn{4}{c}{n=25} & \multicolumn{4}{c}{n=50} \\
\cmidrule(lr){3-6} \cmidrule(lr){7-10}
Instance & Region & Cost & Cross & Wait & Util & Cost & Cross & Wait & Util \\
\midrule
...
\end{tabular}
\end{sidewaystable}
```

或者更实用的：**每个 size 一个子表**，用 subtable 排版。

**简化方案**：因为只有 Flex 版本（去掉 Same），每 size 的列数减少，60 行可以放进一个 sidewaystable。

列结构（只展示 Flex 数据）：

```
Instance, Region, Cost, Cross-truck, Truck wait(h), Drone util(%), Makespan(h), Tardiness(h), Delayed cust.
```

按 size 分组（n=25 → n=50 → n=75 → n=100），每组内按 region 排序。

---

## 5. 验证清单

### 5.1 运行中验证

| 检查点 | 方法 | 预期 |
|--------|------|------|
| Checkpoint SKIP | 中断后重跑 | 跳过已有 entry |
| 解可还原 | 从 CSV 中读 JSON→`_solution_from_json()`→`evaluate_solution()` | cost 一致 |
| Dual-tracking 一致性 | `flex_cross_truck=0` 的 entry | `same_cost == flex_cost` |
| 运营指标稳定 | 同一 instance/seed 重跑 | 指标完全一致 |

### 5.2 运行后验证

```bash
# 1. 检查所有结果可行
python -c "
import csv
for s in [25,50,75,100]:
    with open(f'results/revision_experiments/flexible_docking_full_metrics/i{s}_best_by_instance.csv') as f:
        rows = list(csv.DictReader(f))
    feasible = sum(1 for r in rows if r['flexible_cost'] and float(r['flexible_cost']) > 0)
    print(f'n={s}: {feasible}/{len(rows)} feasible')
"

# 2. 检查 same==flex 一致性
python -c "
import csv
issues = 0
for s in [25,50,75,100]:
    with open(f'results/revision_experiments/flexible_docking_full_metrics/i{s}_best_by_instance.csv') as f:
        for r in csv.DictReader(f):
            if r['flexible_cost'] is None: continue
            # Check cross-truck = 0 case
            # (need to deserialize and count cross-truck)
print(f'Issues: {issues}')
"
```

---

## 6. Paper 更新步骤

### 6.1 替换 Appendix B 表

```latex
% 旧表（n=25 only，有 Same+Flex）
% \begin{sidewaystable}[htbp] ... \end{sidewaystable}

% 新表（n=25~100，仅 Flex）
\begin{sidewaystable}[htbp]
\centering
\caption{Per-instance operational metrics of flexible docking solutions ($n=25,50,75,100$).}
\label{tab:flexible_docking_per_instance}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrrrrrrrr}
\toprule
& & \multicolumn{4}{c}{$n=25$} & \multicolumn{4}{c}{$n=50$} \\
...
\end{tabular}
\end{sidewaystable}
```

### 6.2 更新表前文字

第 1276 行的描述文字从仅描述 n=25 改为覆盖所有规模。

### 6.3 字段计算说明

| 字段 | 来源 |
|------|------|
| Cost | `evaluate_solution().total_cost` |
| Cross-truck | `count_cross_truck(solution)` |
| Truck wait | `evaluate_with_details()` 中的 truck_timings |
| Drone util | `total_drone_flight / (available_drones * makespan)` |
| Makespan | `max(truck_durations)` |
| Tardiness | `delay_breakdown.nodes[].delay` 之和 |
| Delayed cust. | `delay_breakdown.nodes` 中 delay>0 的数量 |

---

## 7. 文件索引

| 文件 | 用途 | 需要修改? |
|------|------|-----------|
| `scripts/compare_fair_flexible_vs_same_all.py` | 参考代码——复制其结构和导入 | ❌ 不改 |
| `scripts/run_flexible_docking_full_metrics.py` | **新建**——主脚本 | ✅ 新建 |
| `sensitivity/docking_flexibility_comparison.py:393-416` | 参考——序列化函数 | ❌ 不改，但可复制代码 |
| `scripts/analyze_flexible_docking_operational_metrics.py` | 提取运营指标→LaTeX | ✅ 需要添加 per-instance 表生成 |
| `alns_vrpfd/evaluation/` | Evaluator + evaluate_with_details | ❌ 不改 |
| `alns_vrpfd/model/solution.py` | Solution dataclass | ❌ 不改 |
| `alns_vrpfd/model/route.py` | TruckRoute, DroneTask | ❌ 不改 |
| `alns_vrpfd/core/sa.py` | SimulatedAnnealingALNS (已有 dual-tracking) | ❌ 不改 |
| `paper/main.tex` | 论文正文 | ✅ 更新 Appendix B 表 + 文字 |
