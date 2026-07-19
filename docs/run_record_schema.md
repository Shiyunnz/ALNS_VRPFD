# Run Record Schema Design

## Goal

每次运行 ALNS 或 MILP 后，将完整的最终解信息统一记录为 JSON 文件，保证可读性优先，便于人工查看和后续对比。

## File Organization

```
results/
  records/
    R_30_10_5/
      alns_2026-06-07_14-30-00.json
      mip_2026-06-07_15-00-00.json
    R_30_10_6/
      alns_2026-06-07_16-00-00.json
```

路径格式: `results/records/{instance}/{algorithm}_{timestamp}.json`

## Data Types

| 类型 | 说明 |
|------|------|
| `hours` | 浮点数，以小时为单位的时间 |
| `cost_unit` | 浮点数，成本单位 |
| `kwh` | 浮点数，千瓦时 |
| `kg` | 浮点数，千克 |
| `string` | 字符串 |
| `array` | 列表 |
| `object` | 嵌套对象 |

## Schema

### Root

```json
{
  "run": { ... },
  "summary": { ... },
  "routes": [ ... ],
  "drone_flights": [ ... ],
  "details": { ... }
}
```

### `run`

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | `R_30_10_5_alns_2026-06-07_14-30-00` |
| `algorithm` | string | `"alns"` 或 `"mip"` |
| `instance` | string | `"R_30_10_5"` |
| `seed` | int | 随机种子（ALNS 特有） |
| `timestamp` | string | ISO 8601 |
| `runtime_seconds` | float | 求解耗时 |
| `config` | object | 参数快照 |

### `summary`

| 字段 | 类型 | 说明 |
|------|------|------|
| `total_cost` | cost_unit | 总成本（运输+延误） |
| `transport_cost` | cost_unit | 运输成本 |
| `delay_cost` | cost_unit | 延误成本 |
| `service_completion_hour` | hours | 最后任务完成时间 |
| `energy_feasible` | bool | 所有无人机段是否均在电池容量内 |
| `num_trucks` | int | 使用的车辆数 |
| `num_drones` | int | 使用的无人机数 |
| `num_customers_served` | int | 服务的客户数 |
| `delay_violations` | int | 迟到客户数 |

### `routes[]`

每条卡车路线一条记录。

| 字段 | 类型 | 说明 |
|------|------|------|
| `truck` | int | 车辆编号 |
| `path` | string | 可读路径，如 `"0 → 4(drone↑) → 1(drone↑) → 6(drone↑) → 11"` |
| `node_sequence` | array[int] | 节点序列 |
| `arrival_times` | array[hours] | 各节点到达时间 |
| `drone_launches` | array[int] | 在此车哪些节点发射了无人机 |
| `drone_launch_count` | int | 发射次数 |

### `drone_flights[]`

每条无人机一条记录，包含其所有飞行段。

| 字段 | 类型 | 说明 |
|------|------|------|
| `drone` | int | 无人机编号 |
| `parent_truck` | int | 所属卡车 |
| `segments` | array[string] | 每段的可读描述 |

每段字符串示例:
```
"4→7→10→8→1  (customers: 7,10,8,  launch→retrieve: 0.6h→1.1h,  energy: 5.74/6.3 kWh ✓)"
```

#### 能耗标志

| 符号 | 含义 |
|------|------|
| `✓` | 能耗 ≤ 电池容量 |
| `✗` | 能耗 > 电池容量 |

### `details`

次要数据，供需时深入查看。

#### `details.customers[]`

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | int | 客户编号 |
| `class` | string | 物资类别 (C1/C2/C3/C4) |
| `demand` | kg | 需求量 |
| `drone` | int | 由哪架无人机服务 |
| `parent_truck` | int | 关联卡车 |
| `arrival` | hours | 到达时间 |
| `due` | hours | 最早时间窗 |
| `deadline` | hours | 最晚时间窗 |
| `delay` | hours | 延误时长 |
| `cost` | cost_unit | 延误成本 |
| `early` | bool | 是否在 due 之前到达 |

#### `details.energy_per_segment[]`

| 字段 | 类型 | 说明 |
|------|------|------|
| `drone` | int | 无人机编号 |
| `path` | string | 路径描述 |
| `kwh` | float | 该段总能耗 |
| `battery` | float | 电池容量 |
| `feasible` | bool | 是否可行 |
| `loads` | array[kg] | 每段弧的负载变化 |

#### `details.timing`

| 字段 | 类型 | 说明 |
|------|------|------|
| `total_travel_hours` | hours | 总行驶时间 |
| `total_delay_hours` | hours | 总延误时间 |
| `max_delay_hours` | hours | 最大客户延误 |
| `time_window_violations` | int | 迟到客户数 |

## Example

```json
{
  "run": {
    "id": "R_30_10_5_mip_2026-06-07_15-00-00",
    "algorithm": "mip",
    "instance": "R_30_10_5",
    "seed": null,
    "timestamp": "2026-06-07T15:00:00",
    "runtime_seconds": 1.353,
    "config": {
      "drone_battery_capacity": 6.3,
      "energy_uncertainty_budget": 3,
      "cost_lambda": 30,
      "cost_rho": 0.2083
    }
  },

  "summary": {
    "total_cost": 59.36,
    "transport_cost": 55.00,
    "delay_cost": 4.36,
    "service_completion_hour": 2.30,
    "energy_feasible": true,
    "num_trucks": 1,
    "num_drones": 2,
    "num_customers_served": 7,
    "delay_violations": 1
  },

  "routes": [
    {
      "truck": 0,
      "path": "0 → 4(drone↑) → 1(drone↑) → 6(drone↑) → 11",
      "node_sequence": [0, 4, 1, 6, 11],
      "arrival_times": [0.0, 0.5, 1.2, 1.8, 2.3],
      "drone_launches": [4, 1, 6],
      "drone_launch_count": 3
    }
  ],

  "drone_flights": [
    {
      "drone": 0,
      "parent_truck": 0,
      "segments": [
        "4→7→10→8→1  (customers: 7,10,8,  launch→retrieve: 0.6h→1.1h,  energy: 5.74/6.3 kWh ✓)",
        "1→5→6        (customers: 5,       launch→retrieve: 1.3h→1.6h,  energy: 5.42/6.3 kWh ✓)"
      ]
    },
    {
      "drone": 1,
      "parent_truck": 0,
      "segments": [
        "6→2→9→11     (customers: 2,9,      launch→retrieve: 1.9h→2.2h,  energy: 4.85/6.3 kWh ✓)"
      ]
    }
  ],

  "details": {
    "customers": [
      {"id": 7, "class": "C1", "demand": 9, "drone": 0, "parent_truck": 0,
       "arrival": 0.7, "due": 1.8, "deadline": 3.0, "delay": 0.0, "cost": 0.0, "early": true},
      {"id": 10, "class": "C1", "demand": 2, "drone": 0, "parent_truck": 0,
       "arrival": 0.8, "due": 1.5, "deadline": 2.5, "delay": 0.0, "cost": 0.0, "early": true}
    ],
    "energy_per_segment": [
      {"drone": 0, "path": "4→7→10→8→1", "kwh": 5.74, "battery": 6.3, "feasible": true, "loads": [15, 6, 4, 0]},
      {"drone": 0, "path": "1→5→6",       "kwh": 5.42, "battery": 6.3, "feasible": true, "loads": [12, 0]},
      {"drone": 1, "path": "6→2→9→11",    "kwh": 4.85, "battery": 6.3, "feasible": true, "loads": [13, 2, 0]}
    ],
    "timing": {
      "total_travel_hours": 2.3,
      "total_delay_hours": 0.15,
      "max_delay_hours": 0.15,
      "time_window_violations": 1
    }
  }
}
```

## Recording Function

核心函数签名:

```python
def save_run_record(
    instance: InstanceManager,
    algorithm: str,
    solution: Solution,
    evaluation: EvaluationResult,
    robustness: RobustnessResult,
    runtime_seconds: float,
    config: dict,
    seed: int | None = None,
) -> str:
    """Build and save a run record JSON file.
    
    Returns the file path of the saved record.
    """
```

该函数从现有的数据结构中提取信息，组装成 schema 格式，写入 JSON。

## Committing

在 `alns_vrpfd/experiments/run_experiments.py` 和 `alns_vrpfd/mip/run_mip.py` 完成后的返回点调用 `save_run_record()`。
同时在 `run_alns.py` 主入口也调用。
