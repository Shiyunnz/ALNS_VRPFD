"""
Theta (能耗偏差率) 参数敏感性分析

分析不同 theta (energy_deviation_rate) 值对以下指标的影响:
1. 总成本 (Total Cost)
2. 相对基线成本变化百分比 (Cost Change vs Baseline %)
3. 无人机服务的客户点数量 (Drone-served Customers)
4. 可行性 (Feasibility)

基准: theta=0.05 作为 baseline（最小不确定性水平）
"""

from __future__ import annotations
from typing import Any, Dict, Iterable, List
from collections import defaultdict
import time
import random
import math
import csv
import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path before other imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent
for _p in [Path(__file__).resolve().parent] + list(Path(__file__).resolve().parents):
    if (_p / 'run_alns.py').exists():
        _project_root = _p
        break
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
del _p, _project_root

from alns_vrpfd.evaluation import Evaluator
import alns_vrpfd.model.initializer as initializer
from alns_vrpfd.utils.config_loader import ALNSConfig
from alns_vrpfd.utils.io_utils import read_instance
from alns_vrpfd.core.sa import SANNCfg, SimulatedAnnealingALNS
from run_alns import build_operators
from sensitivity.instance_selector import collect_instance_paths_with_scope


def _infer_size(instance) -> str:
    """Infer 'small'|'medium'|'large' based on customer count."""
    num_customers = len(instance.customer_manager.customer_ids())
    if num_customers <= 15:
        return "small"
    if num_customers <= 50:
        return "medium"
    return "large"


def _safe_mean(values: List[float]) -> float:
    """Compute mean of values, returning 0.0 if list is empty."""
    filtered = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return sum(filtered) / len(filtered) if filtered else 0.0


# Load default configuration from YAML
_default_config = ALNSConfig()


# ==========================================================================
# 实验配置
# ==========================================================================

# Theta 值水平用于实验横坐标
THETA_LEVELS = [0.05, 0.1, 0.15, 0.2]
BASELINE_THETA = 0.05

# 默认算例目录
DEFAULT_INSTANCE_DIRS = [Path("data/Instance10")]

# ALNS 运行参数 - 固定为 2000 次迭代
ITERATIONS = 2000
TIME_LIMIT = _default_config.time_limit
SEED = _default_config.seed
DRONE_PRIORITY = _default_config.drone_priority
REPAIR_SET = "all"

# Output configuration
OUTPUT_DIR = Path(__file__).parent / "results_new" / "theta_sensitivity"
OUTPUT_CSV = OUTPUT_DIR / "theta_sensitivity_results.csv"


def _build_sa_config(instance) -> SANNCfg:
    """从YAML配置构建SANNCfg"""
    sa_config_dict = _default_config.build_sa_config_dict()
    sa_config_dict['size'] = _infer_size(instance)
    sa_config_dict['iterations'] = ITERATIONS
    return SANNCfg(**sa_config_dict)


# ==========================================================================
# Parameter parsing and utility functions
# ==========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Theta (能耗偏差率) 参数敏感性分析"
    )
    parser.add_argument(
        "--instance-dir",
        action="append",
        dest="instance_dirs",
        help="算例目录路径，支持多次指定，例如 --instance-dir data/Instance25",
    )
    parser.add_argument(
        "--instance-scope",
        type=str,
        choices=["all", "region", "single"],
        default="all",
        help="算例选择范围: all(全量) / region(按30,40,50) / single(单个算例)",
    )
    parser.add_argument(
        "--regions",
        type=str,
        default="30,40,50",
        help="按区域筛选时使用，逗号分隔，例如 '30' 或 '30,40'",
    )
    parser.add_argument(
        "--instance-name",
        type=str,
        default=None,
        help="单算例模式下指定算例名或路径，例如 R_30_10_1 或 data/Instance10/R_30_10_1.txt",
    )
    parser.add_argument(
        "--theta-values",
        type=str,
        default=None,
        help="Theta值水平，逗号分隔，例如 '0.05,0.1,0.15,0.2'。如果不指定，使用默认列表。",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="跳过运行基线（theta=0.05）实验，假设已运行并从 CSV 读取。",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="将结果追加写入已有 CSV（若文件不存在则自动创建）",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="每个算例+参数组合独立运行次数，按 Best-of-k 聚合（默认: 5）",
    )
    return parser.parse_args()


def collect_instance_paths(
    instance_dirs: Iterable[str | Path],
    *,
    instance_scope: str,
    regions: str | None,
    instance_name: str | None,
) -> List[str]:
    """根据目录列表收集算例文件路径。"""
    return collect_instance_paths_with_scope(
        instance_dirs,
        scope=instance_scope,
        regions_text=regions,
        instance_name=instance_name,
    )


# ==========================================================================
# 辅助函数
# ==========================================================================

def count_drone_served_customers(solution) -> int:
    """统计无人机服务的客户点数量。"""
    drone_customers = set()
    for task in solution.drone_tasks:
        drone_customers.update(task.customers())
    return len(drone_customers)


def run_single_experiment(
    instance_path: str,
    theta: float,
    *,
    same_truck_retrieval: bool = False,
) -> Dict[str, Any]:
    """运行单个theta配置实验。"""

    print(f"  Running: theta={theta}, same_truck={same_truck_retrieval}")

    instance = read_instance(instance_path, strategy="class_based")

    # Align drone endurance to infinity for standard comparison (MIP assumption)
    if 'drone' in instance.vehicle_specs:
        instance.vehicle_specs['drone'].endurance = float('inf')

    # Apply robustness configuration with theta parameter
    instance.configure_robustness(
        drone_battery_capacity=_default_config.drone_battery_capacity,
        energy_uncertainty_budget=_default_config.energy_uncertainty_budget,
        energy_deviation_rate=theta,  # This is the theta parameter

        same_truck_retrieval=same_truck_retrieval,
    )

    evaluator = Evaluator(
        instance,
        rendezvous_tolerance=_default_config.drone_rendezvous_tolerance,
        forced_drone_customers=_default_config.forced_drone_customers,
        allow_multiple_launch_per_node=_default_config.relax_allow_multiple_launch_per_node,
    )

    # Initial Solution Strategy matching run_alns.py
    use_two_phase = _default_config.raw.get(
        "initial_solution", {}).get("two_phase", True)
    forced_drone_customers = _default_config.forced_drone_customers

    initial_cost = math.inf
    initial_drone_customers = 0

    try:
        if use_two_phase:
            initial_solution = initializer.build_two_phase_initial_solution(
                instance,
                truck_forbidden_customers=forced_drone_customers,
                allow_multiple_launch_per_node=_default_config.relax_allow_multiple_launch_per_node,
            )
        else:
            initial_solution = initializer.build_initial_solution(
                instance,
                truck_forbidden_customers=forced_drone_customers,
                allow_multiple_launch_per_node=_default_config.relax_allow_multiple_launch_per_node,
            )

        initial_eval = evaluator.evaluate_solution(initial_solution)
        initial_cost = initial_eval.total_cost
        initial_drone_customers = count_drone_served_customers(initial_solution)

    except Exception as exc:
        print(f"    ! Initial solution failed: {exc}. Trying fallback...")
        try:
            truck_routes = initializer._build_truck_routes(instance, time_limit=5.0)
            initial_solution = initializer.Solution.empty()
            for route in truck_routes:
                initial_solution.add_truck_route(route)
            initial_eval = evaluator.evaluate_solution(initial_solution)
            initial_cost = initial_eval.total_cost
            initial_drone_customers = 0
        except Exception as fallback_exc:
            print(f"    ! Fallback failed: {fallback_exc}. Marking infeasible.")
            return {
                "instance": instance_path,
                "theta": theta,
                "same_truck_retrieval": same_truck_retrieval,
                "initial_cost": math.inf,
                "best_cost": math.inf,
                "cost_reduction_percent": math.nan,
                "initial_drone_customers": 0,
                "best_drone_customers": 0,
                "drone_customer_change": 0,
                "feasible": False,
                "run_time": 0.0,
                "truck_distance_cost": math.nan,
                "drone_distance_cost": math.nan,
                "error": f"initial_failed: {fallback_exc}",
            }

    local_seed = SEED if SEED is not None else int(time.time())

    drone_bonus_kwargs = _default_config.drone_bonus
    destroy_ops, repair_ops = build_operators(
        instance,
        seed=local_seed,
        drone_priority=DRONE_PRIORITY,
        repair_set="all",
        enable_composite=True,
        drone_bonus_kwargs=drone_bonus_kwargs,
        forced_drone_customers=forced_drone_customers
    )

    cfg = _build_sa_config(instance)
    alns = SimulatedAnnealingALNS(
        instance=instance,
        destroy_ops=destroy_ops,
        repair_ops=repair_ops,
        evaluator=evaluator,
        cfg=cfg,
        rng=random.Random(local_seed),
    )

    start_time = time.perf_counter()
    best_solution = alns.run(initial_solution, time_limit=TIME_LIMIT)
    run_time = time.perf_counter() - start_time

    try:
        best_eval = evaluator.evaluate_solution(best_solution)
        best_drone_customers = count_drone_served_customers(best_solution)
    except Exception as exc:
        print(f"    ! Best solution evaluation failed: {exc}.")
        return {
            "instance": instance_path,
            "theta": theta,
            "same_truck_retrieval": same_truck_retrieval,
            "initial_cost": initial_cost,
            "best_cost": math.inf,
            "cost_reduction_percent": math.nan,
            "initial_drone_customers": initial_drone_customers,
            "best_drone_customers": 0,
            "drone_customer_change": -initial_drone_customers,
            "feasible": False,
            "run_time": run_time,
            "truck_distance_cost": math.nan,
            "drone_distance_cost": math.nan,
            "error": f"best_eval_failed: {exc}",
        }

    cost_reduction = math.nan
    if (math.isfinite(initial_cost)
            and math.isfinite(best_eval.total_cost)
            and initial_cost > 0):
        cost_reduction = (initial_cost - best_eval.total_cost) / initial_cost * 100

    return {
        "instance": instance_path,
        "theta": theta,
        "same_truck_retrieval": same_truck_retrieval,
        "initial_cost": initial_cost,
        "best_cost": best_eval.total_cost,
        "cost_reduction_percent": cost_reduction,
        "initial_drone_customers": initial_drone_customers,
        "best_drone_customers": best_drone_customers,
        "drone_customer_change": best_drone_customers - initial_drone_customers,
        "feasible": best_eval.feasible,
        "run_time": run_time,
        "truck_distance_cost": best_eval.truck_distance_cost,
        "drone_distance_cost": best_eval.drone_distance_cost,
    }


def _choose_best_result(rows: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """按 best_cost 最小 + best_drone_customers 最大（平局）选择最佳记录。"""
    if not rows:
        return None
    best_row = None
    for row in rows:
        if best_row is None:
            best_row = row
            continue
        row_cost = row.get("best_cost", math.inf)
        best_cost = best_row.get("best_cost", math.inf)
        if row_cost < best_cost:
            best_row = row
            continue
        if row_cost == best_cost and row.get("best_drone_customers", 0) > best_row.get("best_drone_customers", 0):
            best_row = row
    return best_row.copy() if best_row is not None else None


def load_baseline_from_csv(
    instance_paths: List[str],
) -> tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    """从现有 CSV 加载基线成本与基线最佳记录。"""
    baseline_costs: Dict[str, float] = {path: math.inf for path in instance_paths}
    baseline_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if not OUTPUT_CSV.exists():
        print("  Warning: No existing CSV found, will use inf for baselines.")
        return baseline_costs, {}

    with open(OUTPUT_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            instance = row.get("instance")
            if instance not in instance_paths:
                continue
            try:
                theta = float(row.get("theta", -1))
                cost = float(row.get("best_cost", "inf"))
            except (TypeError, ValueError):
                continue
            strategy = row.get("strategy", "")
            if strategy == "Baseline" or abs(theta - BASELINE_THETA) < 1e-9:
                baseline_rows[instance].append({
                    "instance": instance,
                    "strategy": "Baseline",
                    "theta": theta,
                    "best_cost": cost,
                    "best_drone_customers": float(row.get("best_drone_customers", 0) or 0),
                    "trial": row.get("trial", "csv"),
                })

    baseline_results_map: Dict[str, Dict[str, Any]] = {}
    for instance in instance_paths:
        best_row = _choose_best_result(baseline_rows.get(instance, []))
        if best_row is None:
            continue
        best_row["baseline_best_cost"] = best_row.get("best_cost", math.inf)
        best_row["cost_increase_vs_baseline"] = 0.0
        baseline_costs[instance] = best_row.get("best_cost", math.inf)
        baseline_results_map[instance] = best_row
    return baseline_costs, baseline_results_map


def _extract_scale_label(instance_path: str) -> str:
    path = Path(instance_path)
    try:
        return path.parent.name or "unknown"
    except IndexError:
        return "unknown"


def run_theta_sensitivity_analysis(
    instance_paths: List[str],
    theta_levels: List[float] | None = None,
    skip_baseline: bool = False,
    trials: int = 5,
) -> List[Dict[str, Any]]:
    """运行theta参数敏感度分析并返回结果列表。"""

    if theta_levels is None:
        theta_levels = THETA_LEVELS

    print("=" * 80)
    print("Theta (能耗偏差率) 参数敏感性分析")
    print("=" * 80)
    print(f"测试算例: {len(instance_paths)} 个")
    print(f"Theta水平: {theta_levels}")
    print(f"Baseline Theta: {BASELINE_THETA}")
    print(f"迭代次数: {ITERATIONS}")
    print(f"时间限制: {TIME_LIMIT}s")
    print(f"随机种子: {SEED}")
    print(f"独立试验次数 (Trials): {trials}")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results: list[Dict[str, Any]] = []
    baseline_costs = {}  # instance_path -> best_cost
    baseline_results_map = {}  # instance_path -> result dict

    # 1. 运行 Baseline (theta=0.05) 获取基准 Costs
    print(f"\nPhase 1: Handling Baseline (Theta={BASELINE_THETA})...")

    if skip_baseline:
        print("  Skipping baseline run, reading from existing CSV...")
        baseline_costs, baseline_results_map = load_baseline_from_csv(instance_paths)
    else:
        print(f"  Running Baseline for all instances ({trials} trials per instance)...")

        for idx, instance_path in enumerate(instance_paths, 1):
            print(f"  [{idx}/{len(instance_paths)}] Baseline: {Path(instance_path).stem}")
            best_baseline_res = None
            for t in range(trials):
                res = run_single_experiment(
                    instance_path=instance_path,
                    theta=BASELINE_THETA,
                    same_truck_retrieval=False
                )
                res["trial"] = t
                all_results.append(res)
                if best_baseline_res is None or res.get("best_cost", math.inf) < best_baseline_res.get("best_cost", math.inf):
                    best_baseline_res = res.copy()
                elif res.get("best_cost", math.inf) == best_baseline_res.get("best_cost", math.inf) and res.get("best_drone_customers", 0) > best_baseline_res.get("best_drone_customers", 0):
                    best_baseline_res = res.copy()

            if best_baseline_res is None:
                continue

            best_baseline_res["strategy"] = "Baseline"
            best_baseline_res["baseline_best_cost"] = best_baseline_res.get("best_cost", math.inf)
            best_baseline_res["cost_increase_vs_baseline"] = 0.0
            baseline_costs[instance_path] = best_baseline_res.get("best_cost", math.inf)
            baseline_results_map[instance_path] = best_baseline_res

    # 2. Iterate through Theta values
    print(f"\nPhase 2: Running Sensitivity Tests...")

    for theta in theta_levels:
        print(f"\n>>> Testing Theta: {theta} <<<")

        current_level_results = []

        for idx, instance_path in enumerate(instance_paths, 1):
            instance_name = Path(instance_path).stem

            if abs(theta - BASELINE_THETA) < 1e-9:
                cached_baseline = baseline_results_map.get(instance_path)
                if cached_baseline is not None:
                    print(f"  [{idx}/{len(instance_paths)}] {instance_name} (Using Cached Baseline)")
                    result = cached_baseline.copy()
                    current_level_results.append(result)
                    all_results.append(result)
                    continue
                print(
                    f"  [{idx}/{len(instance_paths)}] {instance_name} (Baseline not cached, rerun {trials} trials)")
                best_baseline_res = None
                for t in range(trials):
                    res = run_single_experiment(
                        instance_path=instance_path,
                        theta=theta,
                        same_truck_retrieval=False
                    )
                    res["trial"] = t
                    res["strategy"] = "Baseline"
                    res["baseline_best_cost"] = math.nan
                    res["cost_increase_vs_baseline"] = math.nan
                    all_results.append(res)
                    if best_baseline_res is None or res.get("best_cost", math.inf) < best_baseline_res.get("best_cost", math.inf):
                        best_baseline_res = res.copy()
                    elif res.get("best_cost", math.inf) == best_baseline_res.get("best_cost", math.inf) and res.get("best_drone_customers", 0) > best_baseline_res.get("best_drone_customers", 0):
                        best_baseline_res = res.copy()
                if best_baseline_res is None:
                    continue
                best_baseline_res["baseline_best_cost"] = best_baseline_res.get("best_cost", math.inf)
                best_baseline_res["cost_increase_vs_baseline"] = 0.0
                baseline_costs[instance_path] = best_baseline_res.get("best_cost", math.inf)
                baseline_results_map[instance_path] = best_baseline_res.copy()
                result = best_baseline_res
            else:
                print(f"  [{idx}/{len(instance_paths)}] {instance_name} ({trials} trials)")
                base_cost = baseline_costs.get(instance_path, math.inf)
                best_test_res = None
                for t in range(trials):
                    res = run_single_experiment(
                        instance_path=instance_path,
                        theta=theta,
                        same_truck_retrieval=False
                    )
                    res["trial"] = t
                    res["strategy"] = "Test"
                    res["baseline_best_cost"] = base_cost
                    cur_cost = res.get("best_cost", math.inf)
                    if math.isfinite(base_cost) and base_cost > 0 and math.isfinite(cur_cost):
                        res["cost_increase_vs_baseline"] = (cur_cost - base_cost) / base_cost * 100.0
                    else:
                        res["cost_increase_vs_baseline"] = math.nan
                    all_results.append(res)
                    if best_test_res is None or res.get("best_cost", math.inf) < best_test_res.get("best_cost", math.inf):
                        best_test_res = res.copy()
                    elif res.get("best_cost", math.inf) == best_test_res.get("best_cost", math.inf) and res.get("best_drone_customers", 0) > best_test_res.get("best_drone_customers", 0):
                        best_test_res = res.copy()
                if best_test_res is None:
                    continue
                result = best_test_res

            # Calculate Cost Increase vs Baseline for the best result
            base_cost = baseline_costs.get(instance_path, math.inf)
            current_cost = result.get("best_cost", math.inf)

            result["baseline_best_cost"] = base_cost

            if math.isfinite(base_cost) and base_cost > 0 and math.isfinite(current_cost):
                increase = (current_cost - base_cost) / base_cost * 100.0
                result["cost_increase_vs_baseline"] = increase
            else:
                result["cost_increase_vs_baseline"] = math.nan

            current_level_results.append(result)

        # Compute Averages for this Theta Level
        valid_increases = [r["cost_increase_vs_baseline"] for r in current_level_results
                         if math.isfinite(r.get("cost_increase_vs_baseline", math.nan))]
        valid_drones = [r["best_drone_customers"] for r in current_level_results
                        if math.isfinite(r.get("best_drone_customers", math.nan))]
        valid_feasible = [1 if r.get("feasible", False) else 0 for r in current_level_results]

        avg_increase = sum(valid_increases) / len(valid_increases) if valid_increases else 0.0
        avg_drone = sum(valid_drones) / len(valid_drones) if valid_drones else 0.0
        feasibility_rate = sum(valid_feasible) / len(valid_feasible) * 100 if valid_feasible else 0.0

        print(f"--- Summary for Theta {theta} ---")
        print(f"  Avg Cost Increase vs Baseline: {avg_increase:.2f}%")
        print(f"  Avg Drone Served Customers:  {avg_drone:.2f}")
        print(f"  Feasibility Rate:            {feasibility_rate:.1f}%")

    print(f"\n{'=' * 80}")
    print("实验完成")
    print(f"{'=' * 80}")

    return all_results


def write_summary_csv(results: Iterable[Dict[str, Any]], out_path: Path) -> None:
    """Write a compact summary CSV grouped by scale and theta."""
    # 1. Collapse to best-of-k per (instance, theta)
    by_instance_level: dict[tuple[str, float], list[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        inst = r.get("instance")
        theta = r.get("theta")
        if not isinstance(inst, str) or not isinstance(theta, (int, float)):
            continue
        by_instance_level[(inst, float(theta))].append(r)

    best_rows: list[Dict[str, Any]] = []
    for _, rows in by_instance_level.items():
        best_row = _choose_best_result(rows)
        if best_row is not None:
            best_rows.append(best_row)

    # 2. Process new results into summary rows
    grouped: dict[tuple[str, float], list[Dict[str, Any]]] = defaultdict(list)
    for r in best_rows:
        inst = r.get("instance")
        if isinstance(inst, str):
            scale = _extract_scale_label(inst)
        else:
            scale = "unknown"
        theta = r.get("theta")
        if not isinstance(theta, (int, float)):
            continue
        grouped[(scale, float(theta))].append(r)

    new_rows = []
    keys_to_update = set()
    for (scale, theta), items in sorted(grouped.items()):
        increases = [it.get("cost_increase_vs_baseline") for it in items
                   if isinstance(it.get("cost_increase_vs_baseline"), (int, float))]
        drones = [it.get("best_drone_customers") for it in items
                  if isinstance(it.get("best_drone_customers"), (int, float))]
        feasible_flags = [1 if it.get("feasible", False) else 0 for it in items]

        row = {
            "scale": scale,
            "theta": theta,
            "avg_cost_increase_vs_baseline": _safe_mean(increases),
            "avg_best_drone_customers": _safe_mean(drones),
            "feasibility_rate": (sum(feasible_flags) / len(feasible_flags) * 100) if feasible_flags else 0.0,
        }
        new_rows.append(row)
        keys_to_update.add((str(scale), float(theta)))

    # 3. Read existing data
    existing_rows = []
    fieldnames = ["scale", "theta", "avg_cost_increase_vs_baseline", "avg_best_drone_customers", "feasibility_rate"]

    if out_path.exists():
        try:
            with out_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    for row in reader:
                        try:
                            r_scale = str(row["scale"])
                            r_theta = float(row["theta"])
                            if (r_scale, r_theta) not in keys_to_update:
                                existing_rows.append(row)
                        except (ValueError, KeyError):
                            pass
        except Exception as e:
            print(f"Warning: Could not read existing summary file {out_path}: {e}. Starting fresh.")
            existing_rows = []

    # 4. Combine and Sort
    all_rows = existing_rows + new_rows
    all_rows.sort(key=lambda x: (str(x.get("scale", "")), float(x.get("theta", 0))))

    # 5. Write back
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"  汇总结果已更新: {out_path}")


def write_results(
    results: Iterable[Dict[str, Any]],
    *,
    append: bool,
) -> None:
    results_list = list(results)
    csv_headers = [
        "instance",
        "strategy",
        "theta",
        "baseline_best_cost",
        "cost_increase_vs_baseline",
        "initial_cost",
        "best_cost",
        "cost_reduction_percent",
        "initial_drone_customers",
        "best_drone_customers",
        "drone_customer_change",
        "feasible",
        "run_time",
        "truck_distance_cost",
        "drone_distance_cost",
    ]

    mode = "a" if append else "w"
    write_header = not append or not OUTPUT_CSV.exists()

    with OUTPUT_CSV.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        if write_header:
            writer.writeheader()
        for result in results_list:
            writer.writerow({key: result.get(key, math.nan) for key in csv_headers})

    action = "追加" if append else "写入"
    print(f"  详细结果已{action}: {OUTPUT_CSV}")


def main() -> None:
    args = parse_args()

    instance_dirs = args.instance_dirs or DEFAULT_INSTANCE_DIRS
    instance_paths = collect_instance_paths(
        instance_dirs,
        instance_scope=args.instance_scope,
        regions=args.regions,
        instance_name=args.instance_name,
    )

    # Parse theta values if provided
    theta_levels = None
    if args.theta_values:
        try:
            theta_levels = [float(x.strip()) for x in args.theta_values.split(',')]
        except ValueError:
            print("Error: Invalid theta values format. Use comma-separated floats.")
            return

    print("选择的算例目录:")
    for directory in instance_dirs:
        print(f"  - {Path(directory)}")
    print(f"实例筛选模式: {args.instance_scope}")
    if args.instance_scope == "region":
        print(f"区域过滤: {args.regions}")
    if args.instance_scope == "single":
        print(f"单算例: {args.instance_name}")

    print(f"共收集到 {len(instance_paths)} 个算例文件。")

    results = run_theta_sensitivity_analysis(
        instance_paths, theta_levels, skip_baseline=args.skip_baseline, trials=args.trials)
    write_results(results, append=args.append)

    # Extra summary CSV for plotting
    summary_path = OUTPUT_DIR / "theta_summary.csv"
    write_summary_csv(results, summary_path)

    print("\n  Theta 敏感度分析全部完成!")


if __name__ == "__main__":
    main()
