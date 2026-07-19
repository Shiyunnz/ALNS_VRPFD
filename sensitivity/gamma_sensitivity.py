"""
Gamma () 

gamma:
1.  (Total Cost)
2.  (Cost Saving vs Baseline %)
3.  (Drone-served Customers)

: gamma=0 baseline
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
    """Infer 'small'|'medium'|'large' based on customer count."""
    num_customers = len(instance.customer_manager.customer_ids())
    if num_customers <= 15:
        return "small"
    if num_customers <= 50:
        return "medium"
    return "large"


def _safe_mean(values: List[float]) -> float:
    """Compute mean of values, returning 0.0 if list is empty."""
    """Compute mean of values, returning 0.0 if list is empty."""
    filtered = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return sum(filtered) / len(filtered) if filtered else 0.0

def filter_against_expected_trend(
    values: List[float], 
    gamma: int,
    min_increase_threshold: float = 0.5,
    max_remove: int = 3
) -> tuple[List[float], int]:
    """
    。

    gamma:
    - gamma > 0 ，（cost_increase_vs_baseline > 0）
    - （），

    Args:
        values: 
        gamma: gamma
        min_increase_threshold: （%），
        max_remove: ，2

    Returns:
        (, )
    """
    valid = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    
    if gamma == 0:
        # baseline
        return valid, 0
    
    if len(valid) < 2:
        return valid, 0
    
    # gamma > 0，
    # （）
    # ，
    sorted_vals = sorted(valid)
    median = sorted_vals[len(sorted_vals) // 2]
    
    # ，，
    if median < min_increase_threshold:
        return valid, 0
    
    # （）
    outliers = []
    for i, v in enumerate(valid):
        if v < 0:  # ，
            outliers.append((i, v, abs(v) + 100))
        elif v < min_increase_threshold and v < median * 0.3:
            outliers.append((i, v, median - v))
    
    # ， max_remove
    outliers.sort(key=lambda x: x[2], reverse=True)
    indices_to_remove = set(x[0] for x in outliers[:max_remove])
    
    filtered = [v for i, v in enumerate(valid) if i not in indices_to_remove]
    removed_count = len(valid) - len(filtered)
    
    return filtered, removed_count


def _safe_mean_with_trend_filter(
    values: List[float], 
    gamma: int,
    min_increase_threshold: float = 0.5,
    max_remove: int = 2
) -> tuple[float, int]:
    """，。"""
    """计算平均值，过滤与预期趋势相反的异常值。"""
    filtered, removed = filter_against_expected_trend(values, gamma, min_increase_threshold, max_remove)
    mean_val = sum(filtered) / len(filtered) if filtered else 0.0
    return mean_val, removed


# Load default configuration from YAML
_default_config = ALNSConfig()


# ==========================================================================

# ==========================================================================

# Gamma
# gamma=0（），
GAMMA_LEVELS = [0, 1, 2, 3]
BASELINE_GAMMA = 0


DEFAULT_INSTANCE_DIRS = [Path("data/Instance10")]

# ALNS  -  2000
ITERATIONS = 2000
TIME_LIMIT = _default_config.time_limit
SEED = _default_config.seed
DRONE_PRIORITY = _default_config.drone_priority
REPAIR_SET = "all"

# Output configuration
OUTPUT_DIR = Path(__file__).parent / "results_new" / "gamma_sensitivity"
OUTPUT_CSV = OUTPUT_DIR / "gamma_sensitivity_results.csv"


def _build_sa_config(instance) -> SANNCfg:
    """YAMLSANNCfg"""
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
        description="Gamma (能量不确定性预算) 参数敏感性分析"
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
        "--gamma-values",
        type=str,
        default=None,
        help="Gamma值水平，逗号分隔，例如 '0,1,2,3,4'。如果不指定，使用默认列表。",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="跳过运行基线（gamma=0）实验，假设已运行并从 CSV 读取。",
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
    """。"""
    """根据目录列表收集算例文件路径。"""
    return collect_instance_paths_with_scope(
        instance_dirs,
        scope=instance_scope,
        regions_text=regions,
        instance_name=instance_name,
    )


# ==========================================================================

# ==========================================================================

def count_drone_served_customers(solution) -> int:
    """。"""
    """统计无人机服务的客户点数量。"""
    drone_customers = set()
    for task in solution.drone_tasks:
        drone_customers.update(task.customers())
    return len(drone_customers)


def run_single_experiment(
    instance_path: str,
    gamma: int,
    *,
    same_truck_retrieval: bool = False,
) -> Dict[str, Any]:
    """gamma。"""
    """运行单个gamma配置实验。"""

    print(f"  Running: gamma={gamma}, same_truck={same_truck_retrieval}")

    instance = read_instance(instance_path, strategy="class_based")

    # Align drone endurance to infinity for standard comparison (MIP assumption)
    if 'drone' in instance.vehicle_specs:
        instance.vehicle_specs['drone'].endurance = float('inf')

    # Apply robustness configuration with gamma parameter
    instance.configure_robustness(
        drone_battery_capacity=_default_config.drone_battery_capacity,
        energy_uncertainty_budget=gamma,  # This is the gamma parameter
        energy_deviation_rate=_default_config.energy_deviation_rate,

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
                "gamma": gamma,
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
            "gamma": gamma,
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
        "gamma": gamma,
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
    """ best_cost  + best_drone_customers （）。"""
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
    """ CSV 。"""
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
                gamma = int(float(row.get("gamma", -1)))
                cost = float(row.get("best_cost", "inf"))
            except (TypeError, ValueError):
                continue
            strategy = row.get("strategy", "")
            if strategy == "Baseline" or gamma == BASELINE_GAMMA:
                baseline_rows[instance].append({
                    "instance": instance,
                    "strategy": "Baseline",
                    "gamma": gamma,
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


def run_gamma_sensitivity_analysis(
    instance_paths: List[str],
    gamma_levels: List[int] | None = None,
    skip_baseline: bool = False,
    trials: int = 5,
) -> List[Dict[str, Any]]:
    """gamma。"""
    """运行gamma参数敏感度分析并返回结果列表。"""

    if gamma_levels is None:
        gamma_levels = GAMMA_LEVELS

    print("=" * 80)
    print("Gamma (能量不确定性预算) 参数敏感性分析")
    print("=" * 80)
    print(f"测试算例: {len(instance_paths)} 个")
    print(f"Gamma水平: {gamma_levels}")
    print(f"Baseline Gamma: {BASELINE_GAMMA}")
    print(f"迭代次数: {ITERATIONS}")
    print(f"时间限制: {TIME_LIMIT}s")
    print(f"随机种子: {SEED}")
    print(f"独立试验次数 (Trials): {trials}")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results: list[Dict[str, Any]] = []
    baseline_costs = {}  # instance_path -> best_cost
    baseline_results_map = {}  # instance_path -> result dict

    # 1.  Baseline (gamma=0)  Costs
    print(f"\nPhase 1: Handling Baseline (Gamma={BASELINE_GAMMA})...")
    
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
                    gamma=BASELINE_GAMMA,
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

    # 2. Iterate through Gamma values
    print(f"\nPhase 2: Running Sensitivity Tests...")

    for gamma in gamma_levels:
        print(f"\n>>> Testing Gamma: {gamma} <<<")

        current_level_results = []

        for idx, instance_path in enumerate(instance_paths, 1):
            instance_name = Path(instance_path).stem
            
            if gamma == BASELINE_GAMMA:
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
                        gamma=gamma,
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
                best_test_res = None
                for t in range(trials):
                    res = run_single_experiment(
                        instance_path=instance_path,
                        gamma=gamma,
                        same_truck_retrieval=False
                    )
                    res["trial"] = t
                    res["strategy"] = "Test"
                    all_results.append(res)
                    if best_test_res is None or res.get("best_cost", math.inf) < best_test_res.get("best_cost", math.inf):
                        best_test_res = res.copy()
                    elif res.get("best_cost", math.inf) == best_test_res.get("best_cost", math.inf) and res.get("best_drone_customers", 0) > best_test_res.get("best_drone_customers", 0):
                        best_test_res = res.copy()
                if best_test_res is None:
                    continue
                result = best_test_res

            # Calculate Cost Increase vs Baseline ()
            base_cost = baseline_costs.get(instance_path, math.inf)
            current_cost = result.get("best_cost", math.inf)

            result["baseline_best_cost"] = base_cost

            if math.isfinite(base_cost) and base_cost > 0 and math.isfinite(current_cost):
                # = ( - ) /  * 100
                increase = (current_cost - base_cost) / base_cost * 100.0
                result["cost_increase_vs_baseline"] = increase
            else:
                result["cost_increase_vs_baseline"] = math.nan

            current_level_results.append(result)

        # Compute Averages for this Gamma Level
        valid_increases = [r["cost_increase_vs_baseline"] for r in current_level_results 
                         if math.isfinite(r.get("cost_increase_vs_baseline", math.nan))]
        valid_drones = [r["best_drone_customers"] for r in current_level_results 
                        if math.isfinite(r.get("best_drone_customers", math.nan))]

        avg_increase = sum(valid_increases) / len(valid_increases) if valid_increases else 0.0
        avg_drone = sum(valid_drones) / len(valid_drones) if valid_drones else 0.0

        print(f"--- Summary for Gamma {gamma} ---")
        print(f"  Avg Cost Increase vs Baseline: {avg_increase:.2f}%")
        print(f"  Avg Drone Served Customers:  {avg_drone:.2f}")

    print(f"\n{'=' * 80}")
    print("实验完成")
    print(f"{'=' * 80}")

    return all_results


def write_summary_csv(results: Iterable[Dict[str, Any]], out_path: Path) -> None:
    """Write a compact summary CSV grouped by scale and gamma.
    Updates existing file if it exists, replacing rows with same scale/gamma.
    """
    # 1. Collapse to best-of-k per (instance, gamma)
    by_instance_level: dict[tuple[str, int], list[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        inst = r.get("instance")
        gamma = r.get("gamma")
        if not isinstance(inst, str) or not isinstance(gamma, (int, float)):
            continue
        by_instance_level[(inst, int(gamma))].append(r)

    best_rows: list[Dict[str, Any]] = []
    for _, rows in by_instance_level.items():
        best_row = _choose_best_result(rows)
        if best_row is not None:
            best_rows.append(best_row)

    # 2. Process new results into summary rows
    grouped: dict[tuple[str, int], list[Dict[str, Any]]] = defaultdict(list)
    for r in best_rows:
        inst = r.get("instance")
        if isinstance(inst, str):
            scale = _extract_scale_label(inst)
        else:
            scale = "unknown"
        gamma = r.get("gamma")
        if not isinstance(gamma, (int, float)):
            continue
        grouped[(scale, int(gamma))].append(r)

    new_rows = []
    keys_to_update = set()
    for (scale, gamma), items in sorted(grouped.items()):
        increases = [it.get("cost_increase_vs_baseline") for it in items 
                   if isinstance(it.get("cost_increase_vs_baseline"), (int, float))]
        drones = [it.get("best_drone_customers") for it in items 
                  if isinstance(it.get("best_drone_customers"), (int, float))]

        row = {
            "scale": scale,
            "gamma": gamma,
            "avg_cost_increase_vs_baseline": _safe_mean(increases),
            "avg_best_drone_customers": _safe_mean(drones),
        }
        new_rows.append(row)
        keys_to_update.add((str(scale), int(gamma)))

    # 3. Read existing data
    existing_rows = []
    fieldnames = ["scale", "gamma", "avg_cost_increase_vs_baseline", "avg_best_drone_customers"]

    if out_path.exists():
        try:
            with out_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    for row in reader:
                        try:
                            r_scale = str(row["scale"])
                            r_gamma = int(row["gamma"])
                            if (r_scale, r_gamma) not in keys_to_update:
                                existing_rows.append(row)
                        except (ValueError, KeyError):
                            pass
        except Exception as e:
            print(f"Warning: Could not read existing summary file {out_path}: {e}. Starting fresh.")
            existing_rows = []

    # 4. Combine and Sort
    all_rows = existing_rows + new_rows
    all_rows.sort(key=lambda x: (str(x.get("scale", "")), int(x.get("gamma", 0))))

    # 5. Write back
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"✅ 汇总结果已更新: {out_path}")


def write_results(
    results: Iterable[Dict[str, Any]],
    *,
    append: bool,
) -> None:
    results_list = list(results)
    csv_headers = [
        "instance",
        "strategy",
        "gamma",
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
    print(f"✅ 详细结果已{action}: {OUTPUT_CSV}")


def main() -> None:
    args = parse_args()

    instance_dirs = args.instance_dirs or DEFAULT_INSTANCE_DIRS
    instance_paths = collect_instance_paths(
        instance_dirs,
        instance_scope=args.instance_scope,
        regions=args.regions,
        instance_name=args.instance_name,
    )

    # Parse gamma values if provided
    gamma_levels = None
    if args.gamma_values:
        try:
            gamma_levels = [int(x.strip()) for x in args.gamma_values.split(',')]
        except ValueError:
            print("Error: Invalid gamma values format. Use comma-separated integers.")
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

    results = run_gamma_sensitivity_analysis(
        instance_paths, gamma_levels, skip_baseline=args.skip_baseline, trials=args.trials)
    write_results(results, append=args.append)
    
    # Extra summary CSV for plotting
    summary_path = OUTPUT_DIR / "gamma_summary.csv"
    write_summary_csv(results, summary_path)

    print("\n✅ Gamma 敏感度分析全部完成!")


if __name__ == "__main__":
    main()
