"""
时间窗宽度敏感性分析

以缩放因子 scale 等比例调整 (min_window_width, max_window_width),
分析不同时间窗宽度对以下指标的影响:
1. 总成本 (Total Cost)
2. 相对基线成本变化百分比 (Cost Saving vs Baseline %)
3. 无人机服务的客户点数量 (Drone-served Customers)
4. 可行性 (Feasibility)

基准: scale=1.0 (默认时间窗 min=0.33, max=2.0)
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
from alns_vrpfd.instance.time_windows import TimeWindowConfig
from alns_vrpfd.core.sa import SANNCfg, SimulatedAnnealingALNS
from run_alns import build_operators
from sensitivity.instance_selector import collect_instance_paths_with_scope


def _infer_size(instance) -> str:
    num_customers = len(instance.customer_manager.customer_ids())
    if num_customers <= 15:
        return "small"
    if num_customers <= 50:
        return "medium"
    return "large"


def _safe_mean(values: List[float]) -> float:
    filtered = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return sum(filtered) / len(filtered) if filtered else 0.0


# Load default configuration from YAML
_default_config = ALNSConfig()

# Default TimeWindowConfig for reference
_default_tw = TimeWindowConfig()
BASE_MIN_WIDTH = _default_tw.min_window_width  # 0.33
BASE_MAX_WIDTH = _default_tw.max_window_width  # 2.0

# ==========================================================================
# 实验配置
# ==========================================================================

SCALE_LEVELS = [0.5, 1.0, 1.5, 2.0]
BASELINE_SCALE = 1.0

DEFAULT_INSTANCE_DIRS = [Path("data/Instance10")]

ITERATIONS = 2000
TIME_LIMIT = _default_config.time_limit
SEED = _default_config.seed
DRONE_PRIORITY = _default_config.drone_priority
REPAIR_SET = "all"

OUTPUT_DIR = Path(__file__).parent / "results_new" / "time_window_sensitivity"
OUTPUT_CSV = OUTPUT_DIR / "time_window_sensitivity_results.csv"


def _build_sa_config(instance) -> SANNCfg:
    sa_config_dict = _default_config.build_sa_config_dict()
    sa_config_dict['size'] = _infer_size(instance)
    sa_config_dict['iterations'] = ITERATIONS
    return SANNCfg(**sa_config_dict)


def _build_tw_config(scale: float) -> TimeWindowConfig:
    """Build a TimeWindowConfig with scaled window widths."""
    return TimeWindowConfig(
        operation_horizon=_default_tw.operation_horizon,
        min_window_width=BASE_MIN_WIDTH * scale,
        max_window_width=BASE_MAX_WIDTH * scale,
        service_time=_default_tw.service_time,
        shift_factor=_default_tw.shift_factor,
        road_condition_factor=_default_tw.road_condition_factor,
        priority_levels=_default_tw.priority_levels,
        latest_time_slack=_default_tw.latest_time_slack,
    )


# ==========================================================================
# Parameter parsing
# ==========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="时间窗宽度敏感性分析"
    )
    parser.add_argument(
        "--instance-dir", action="append", dest="instance_dirs",
        help="算例目录路径，支持多次指定",
    )
    parser.add_argument(
        "--instance-scope", type=str, choices=["all", "region", "single"],
        default="all", help="算例选择范围",
    )
    parser.add_argument(
        "--regions", type=str, default="30,40,50",
        help="按区域筛选时使用，逗号分隔",
    )
    parser.add_argument(
        "--instance-name", type=str, default=None,
        help="单算例模式下指定算例名",
    )
    parser.add_argument(
        "--scale-values", type=str, default=None,
        help="缩放因子水平，逗号分隔，例如 '0.5,1.0,1.5,2.0'",
    )
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="跳过运行基线实验，从 CSV 读取",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="将结果追加写入已有 CSV",
    )
    parser.add_argument(
        "--trials", type=int, default=5,
        help="每个算例+参数组合独立运行次数（默认: 5）",
    )
    return parser.parse_args()


def collect_instance_paths(
    instance_dirs: Iterable[str | Path], *,
    instance_scope: str, regions: str | None, instance_name: str | None,
) -> List[str]:
    return collect_instance_paths_with_scope(
        instance_dirs, scope=instance_scope,
        regions_text=regions, instance_name=instance_name,
    )


# ==========================================================================
# 辅助函数
# ==========================================================================

def count_drone_served_customers(solution) -> int:
    drone_customers = set()
    for task in solution.drone_tasks:
        drone_customers.update(task.customers())
    return len(drone_customers)


def run_single_experiment(
    instance_path: str,
    scale: float,
    *,
    same_truck_retrieval: bool = False,
) -> Dict[str, Any]:
    """运行单个时间窗配置实验。"""

    tw_config = _build_tw_config(scale)
    print(f"  Running: scale={scale} (min_w={tw_config.min_window_width:.2f}, max_w={tw_config.max_window_width:.2f})")

    # Read instance with custom time window config
    instance = read_instance(instance_path, strategy="demand_based", config=tw_config)

    if 'drone' in instance.vehicle_specs:
        instance.vehicle_specs['drone'].endurance = float('inf')

    instance.configure_robustness(
        drone_battery_capacity=_default_config.drone_battery_capacity,
        energy_uncertainty_budget=_default_config.energy_uncertainty_budget,
        energy_deviation_rate=_default_config.energy_deviation_rate,

        same_truck_retrieval=same_truck_retrieval,
    )

    evaluator = Evaluator(
        instance,
        rendezvous_tolerance=_default_config.drone_rendezvous_tolerance,
        forced_drone_customers=_default_config.forced_drone_customers,
        allow_multiple_launch_per_node=_default_config.relax_allow_multiple_launch_per_node,
    )

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
                "scale": scale,
                "min_window_width": tw_config.min_window_width,
                "max_window_width": tw_config.max_window_width,
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
        instance, seed=local_seed, drone_priority=DRONE_PRIORITY,
        repair_set="all", enable_composite=True,
        drone_bonus_kwargs=drone_bonus_kwargs,
        forced_drone_customers=forced_drone_customers,
    )

    cfg = _build_sa_config(instance)
    alns = SimulatedAnnealingALNS(
        instance=instance, destroy_ops=destroy_ops, repair_ops=repair_ops,
        evaluator=evaluator, cfg=cfg, rng=random.Random(local_seed),
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
            "scale": scale,
            "min_window_width": tw_config.min_window_width,
            "max_window_width": tw_config.max_window_width,
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
    if (math.isfinite(initial_cost) and math.isfinite(best_eval.total_cost) and initial_cost > 0):
        cost_reduction = (initial_cost - best_eval.total_cost) / initial_cost * 100

    return {
        "instance": instance_path,
        "scale": scale,
        "min_window_width": tw_config.min_window_width,
        "max_window_width": tw_config.max_window_width,
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
                scale = float(row.get("scale", -1))
                cost = float(row.get("best_cost", "inf"))
            except (TypeError, ValueError):
                continue
            strategy = row.get("strategy", "")
            if strategy == "Baseline" or abs(scale - BASELINE_SCALE) < 1e-9:
                baseline_rows[instance].append({
                    "instance": instance, "strategy": "Baseline",
                    "scale": scale, "best_cost": cost,
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
    return path.parent.name or "unknown"


def run_time_window_sensitivity_analysis(
    instance_paths: List[str],
    scale_levels: List[float] | None = None,
    skip_baseline: bool = False,
    trials: int = 5,
) -> List[Dict[str, Any]]:

    if scale_levels is None:
        scale_levels = SCALE_LEVELS

    print("=" * 80)
    print("时间窗宽度敏感性分析")
    print("=" * 80)
    print(f"测试算例: {len(instance_paths)} 个")
    print(f"缩放水平: {scale_levels}")
    print(f"Baseline Scale: {BASELINE_SCALE}")
    print(f"基础窗口: min={BASE_MIN_WIDTH}, max={BASE_MAX_WIDTH}")
    print(f"迭代次数: {ITERATIONS}")
    print(f"时间限制: {TIME_LIMIT}s")
    print(f"随机种子: {SEED}")
    print(f"独立试验次数 (Trials): {trials}")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results: list[Dict[str, Any]] = []
    baseline_costs = {}
    baseline_results_map = {}

    # Phase 1: Baseline (scale=1.0)
    print(f"\nPhase 1: Handling Baseline (Scale={BASELINE_SCALE})...")

    if skip_baseline:
        print("  Skipping baseline run, reading from existing CSV...")
        baseline_costs, baseline_results_map = load_baseline_from_csv(instance_paths)
    else:
        print(f"  Running Baseline for all instances ({trials} trials per instance)...")
        for idx, instance_path in enumerate(instance_paths, 1):
            print(f"  [{idx}/{len(instance_paths)}] Baseline: {Path(instance_path).stem}")
            best_baseline_res = None
            for t in range(trials):
                res = run_single_experiment(instance_path=instance_path, scale=BASELINE_SCALE)
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

    # Phase 2: Sensitivity Tests
    print(f"\nPhase 2: Running Sensitivity Tests...")

    for scale in scale_levels:
        tw_cfg = _build_tw_config(scale)
        print(f"\n>>> Testing Scale: {scale} (min_w={tw_cfg.min_window_width:.2f}, max_w={tw_cfg.max_window_width:.2f}) <<<")

        current_level_results = []

        for idx, instance_path in enumerate(instance_paths, 1):
            instance_name = Path(instance_path).stem

            if abs(scale - BASELINE_SCALE) < 1e-9:
                cached_baseline = baseline_results_map.get(instance_path)
                if cached_baseline is not None:
                    print(f"  [{idx}/{len(instance_paths)}] {instance_name} (Using Cached Baseline)")
                    result = cached_baseline.copy()
                    current_level_results.append(result)
                    all_results.append(result)
                    continue
                print(f"  [{idx}/{len(instance_paths)}] {instance_name} (Baseline not cached, rerun {trials} trials)")
                best_baseline_res = None
                for t in range(trials):
                    res = run_single_experiment(instance_path=instance_path, scale=scale)
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
                    res = run_single_experiment(instance_path=instance_path, scale=scale)
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
                result["cost_increase_vs_baseline"] = (current_cost - base_cost) / base_cost * 100.0
            else:
                result["cost_increase_vs_baseline"] = math.nan

            current_level_results.append(result)

        # Summary
        valid_increases = [r["cost_increase_vs_baseline"] for r in current_level_results
                         if math.isfinite(r.get("cost_increase_vs_baseline", math.nan))]
        valid_drones = [r["best_drone_customers"] for r in current_level_results
                        if math.isfinite(r.get("best_drone_customers", math.nan))]
        valid_feasible = [1 if r.get("feasible", False) else 0 for r in current_level_results]

        avg_increase = sum(valid_increases) / len(valid_increases) if valid_increases else 0.0
        avg_drone = sum(valid_drones) / len(valid_drones) if valid_drones else 0.0
        feasibility_rate = sum(valid_feasible) / len(valid_feasible) * 100 if valid_feasible else 0.0

        print(f"--- Summary for Scale {scale} ---")
        print(f"  Avg Cost Increase vs Baseline: {avg_increase:.2f}%")
        print(f"  Avg Drone Served Customers:  {avg_drone:.2f}")
        print(f"  Feasibility Rate:            {feasibility_rate:.1f}%")

    print(f"\n{'=' * 80}")
    print("实验完成")
    print(f"{'=' * 80}")

    return all_results


def write_summary_csv(results: Iterable[Dict[str, Any]], out_path: Path) -> None:
    by_instance_level: dict[tuple[str, float], list[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        inst = r.get("instance")
        scale = r.get("scale")
        if not isinstance(inst, str) or not isinstance(scale, (int, float)):
            continue
        by_instance_level[(inst, float(scale))].append(r)

    best_rows: list[Dict[str, Any]] = []
    for _, rows in by_instance_level.items():
        best_row = _choose_best_result(rows)
        if best_row is not None:
            best_rows.append(best_row)

    grouped: dict[tuple[str, float], list[Dict[str, Any]]] = defaultdict(list)
    for r in best_rows:
        inst = r.get("instance")
        inst_scale = _extract_scale_label(inst) if isinstance(inst, str) else "unknown"
        scale = r.get("scale")
        if not isinstance(scale, (int, float)):
            continue
        grouped[(inst_scale, float(scale))].append(r)

    new_rows = []
    keys_to_update = set()
    for (inst_scale, scale), items in sorted(grouped.items()):
        increases = [it.get("cost_increase_vs_baseline") for it in items
                   if isinstance(it.get("cost_increase_vs_baseline"), (int, float))]
        drones = [it.get("best_drone_customers") for it in items
                  if isinstance(it.get("best_drone_customers"), (int, float))]
        feasible_flags = [1 if it.get("feasible", False) else 0 for it in items]

        row = {
            "instance_scale": inst_scale,
            "tw_scale": scale,
            "min_window_width": BASE_MIN_WIDTH * scale,
            "max_window_width": BASE_MAX_WIDTH * scale,
            "avg_cost_increase_vs_baseline": _safe_mean(increases),
            "avg_best_drone_customers": _safe_mean(drones),
            "feasibility_rate": (sum(feasible_flags) / len(feasible_flags) * 100) if feasible_flags else 0.0,
        }
        new_rows.append(row)
        keys_to_update.add((str(inst_scale), float(scale)))

    existing_rows = []
    fieldnames = ["instance_scale", "tw_scale", "min_window_width", "max_window_width",
                  "avg_cost_increase_vs_baseline", "avg_best_drone_customers", "feasibility_rate"]

    if out_path.exists():
        try:
            with out_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    for row in reader:
                        try:
                            r_scale_label = str(row["instance_scale"])
                            r_tw_scale = float(row["tw_scale"])
                            if (r_scale_label, r_tw_scale) not in keys_to_update:
                                existing_rows.append(row)
                        except (ValueError, KeyError):
                            pass
        except Exception:
            existing_rows = []

    all_rows = existing_rows + new_rows
    all_rows.sort(key=lambda x: (str(x.get("instance_scale", "")), float(x.get("tw_scale", 0))))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"  汇总结果已更新: {out_path}")


def write_results(results: Iterable[Dict[str, Any]], *, append: bool) -> None:
    results_list = list(results)
    csv_headers = [
        "instance", "strategy", "scale", "min_window_width", "max_window_width",
        "baseline_best_cost", "cost_increase_vs_baseline",
        "initial_cost", "best_cost", "cost_reduction_percent",
        "initial_drone_customers", "best_drone_customers", "drone_customer_change",
        "feasible", "run_time", "truck_distance_cost", "drone_distance_cost",
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
        instance_dirs, instance_scope=args.instance_scope,
        regions=args.regions, instance_name=args.instance_name,
    )

    scale_levels = None
    if args.scale_values:
        try:
            scale_levels = [float(x.strip()) for x in args.scale_values.split(',')]
        except ValueError:
            print("Error: Invalid scale values format. Use comma-separated floats.")
            return

    print("选择的算例目录:")
    for directory in instance_dirs:
        print(f"  - {Path(directory)}")
    print(f"实例筛选模式: {args.instance_scope}")
    print(f"共收集到 {len(instance_paths)} 个算例文件。")

    results = run_time_window_sensitivity_analysis(
        instance_paths, scale_levels, skip_baseline=args.skip_baseline, trials=args.trials)
    write_results(results, append=args.append)

    summary_path = OUTPUT_DIR / "time_window_summary.csv"
    write_summary_csv(results, summary_path)

    print("\n  时间窗宽度敏感度分析全部完成!")


if __name__ == "__main__":
    main()
