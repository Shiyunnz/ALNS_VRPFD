"""


，：
1.  (Total Cost)
2.  (Cost Saving vs Baseline %)
3.  (Drone-served Customers)

：
-  (instance, battery_level)  k （--trials）
-  best_cost  best-of-k（ best_drone_customers ）
-  best-of-k 
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List
import argparse
import csv
import math
import random
import sys
import time

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

from run_alns import build_operators

from alns_vrpfd.core.sa import SANNCfg, SimulatedAnnealingALNS
from alns_vrpfd.evaluation import Evaluator
import alns_vrpfd.model.initializer as initializer
from alns_vrpfd.utils.config_loader import ALNSConfig
from alns_vrpfd.utils.io_utils import read_instance
from sensitivity.instance_selector import collect_instance_paths_with_scope


# Load default configuration from YAML
_default_config = ALNSConfig()


# ==========================================================================

# ==========================================================================

BATTERY_LEVELS = [4.3, 5.3, 6.3, 7.3, 8.3]
BASELINE_BATTERY = 4.3

DEFAULT_INSTANCE_DIRS = [Path("data/Instance25")]

ITERATIONS = 2000
TIME_LIMIT = _default_config.time_limit
SEED = _default_config.seed
DRONE_PRIORITY = _default_config.drone_priority
REPAIR_SET = "new"

OUTPUT_DIR = Path(__file__).parent / "results_new" / "battery_sensitivity_rerun_in_25"
OUTPUT_CSV = OUTPUT_DIR / "battery_sensitivity_results.csv"


# ==========================================================================

# ==========================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="电池容量敏感度分析")
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
        help="单算例模式下指定算例名或路径，例如 R_30_25_1 或 data/Instance25/R_30_25_1.txt",
    )
    parser.add_argument(
        "--battery-levels",
        type=str,
        default=None,
        help="电池容量水平（kWh），逗号分隔，例如 '4.3,5.3,6.3,7.3,8.3'。",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="跳过运行基线（最低电池容量）实验，假设已运行并从 CSV 读取。",
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


def _extract_scale_label(instance_path: str) -> str:
    path = Path(instance_path)
    try:
        return path.parent.name or "unknown"
    except IndexError:
        return "unknown"


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


def count_drone_served_customers(solution) -> int:
    drone_customers = set()
    for task in solution.drone_tasks:
        drone_customers.update(task.customers())
    return len(drone_customers)


def _build_sa_config(instance) -> SANNCfg:
    sa_config_dict = _default_config.build_sa_config_dict()
    sa_config_dict["size"] = _infer_size(instance)
    sa_config_dict["iterations"] = ITERATIONS
    return SANNCfg(**sa_config_dict)


# ==========================================================================

# ==========================================================================


def run_single_experiment(
    instance_path: str,
    battery_capacity: float,
    *,
    same_truck_retrieval: bool = False,
    seed: int | None = None,
) -> Dict[str, Any]:
    """。"""
    """运行单个电池容量配置实验。"""

    print(
        f"  Running: battery_capacity={battery_capacity:.2f}, same_truck={same_truck_retrieval}")

    instance = read_instance(instance_path, strategy="class_based")

    if "drone" in instance.vehicle_specs:
        instance.vehicle_specs["drone"].endurance = float("inf")

    instance.configure_robustness(
        drone_battery_capacity=battery_capacity,
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

    use_two_phase = _default_config.raw.get("initial_solution", {}).get("two_phase", True)
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
                "battery_capacity": battery_capacity,
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

    local_seed = seed if seed is not None else (SEED if SEED is not None else int(time.time()))

    drone_bonus_kwargs = _default_config.drone_bonus
    destroy_ops, repair_ops = build_operators(
        instance,
        seed=local_seed,
        drone_priority=DRONE_PRIORITY,
        repair_set=REPAIR_SET,
        enable_composite=True,
        drone_bonus_kwargs=drone_bonus_kwargs,
        forced_drone_customers=forced_drone_customers,
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
            "battery_capacity": battery_capacity,
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
    if (
        math.isfinite(initial_cost)
        and math.isfinite(best_eval.total_cost)
        and initial_cost > 0
    ):
        cost_reduction = ((initial_cost - best_eval.total_cost) / initial_cost) * 100

    return {
        "instance": instance_path,
        "battery_capacity": battery_capacity,
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
        "error": "",
    }


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
                battery = float(row.get("battery_capacity", 0))
                cost = float(row.get("best_cost", "inf"))
            except (TypeError, ValueError):
                continue
            strategy = row.get("strategy", "")
            if strategy == "Baseline" or math.isclose(battery, BASELINE_BATTERY):
                baseline_rows[instance].append({
                    "instance": instance,
                    "strategy": "Baseline",
                    "battery_capacity": battery,
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
        best_row["cost_saving_vs_baseline"] = 0.0
        baseline_costs[instance] = best_row.get("best_cost", math.inf)
        baseline_results_map[instance] = best_row
    return baseline_costs, baseline_results_map


def run_battery_sensitivity_analysis(
    instance_paths: List[str],
    battery_levels: List[float] | None = None,
    skip_baseline: bool = False,
    trials: int = 5,
) -> List[Dict[str, Any]]:
    """。"""
    """运行电池容量敏感度分析并返回结果列表。"""

    if battery_levels is None:
        battery_levels = BATTERY_LEVELS

    baseline_battery = float(battery_levels[0])

    print("=" * 80)
    print("电池容量敏感度分析")
    print("=" * 80)
    print(f"测试算例: {len(instance_paths)} 个")
    print(f"电池容量水平: {battery_levels}")
    print(f"Baseline Battery: {baseline_battery}")
    print(f"迭代次数: {ITERATIONS}")
    print(f"时间限制: {TIME_LIMIT}s")
    print(f"随机种子: {SEED}")
    print(f"独立试验次数 (Trials): {trials}")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results: list[Dict[str, Any]] = []
    baseline_costs: Dict[str, float] = {}
    baseline_results_map: Dict[str, Dict[str, Any]] = {}

    print(f"\nPhase 1: Handling Baseline (Battery={baseline_battery})...")

    if skip_baseline:
        print("  Skipping baseline run, reading from existing CSV...")
        baseline_costs, baseline_results_map = load_baseline_from_csv(instance_paths)
    else:
        print(f"  Running Baseline for all instances ({trials} trials per instance)...")

        for idx, instance_path in enumerate(instance_paths, 1):
            print(f"  [{idx}/{len(instance_paths)}] Baseline: {Path(instance_path).stem}")
            best_baseline_res = None
            for t in range(trials):
                trial_seed = SEED + t if SEED is not None else int(time.time()) + t
                res = run_single_experiment(
                    instance_path=instance_path,
                    battery_capacity=baseline_battery,
                    same_truck_retrieval=False,
                    seed=trial_seed,
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
            best_baseline_res["cost_saving_vs_baseline"] = 0.0

            baseline_costs[instance_path] = best_baseline_res.get("best_cost", math.inf)
            baseline_results_map[instance_path] = best_baseline_res

    print("\nPhase 2: Running Sensitivity Tests...")

    for battery in battery_levels:
        print(f"\n>>> Testing Battery Capacity: {battery:.2f} <<<")

        current_level_results = []

        for idx, instance_path in enumerate(instance_paths, 1):
            instance_name = Path(instance_path).stem

            if math.isclose(battery, baseline_battery):
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
                    trial_seed = SEED + t if SEED is not None else int(time.time()) + t
                    res = run_single_experiment(
                        instance_path=instance_path,
                        battery_capacity=battery,
                        same_truck_retrieval=False,
                        seed=trial_seed,
                    )
                    res["trial"] = t
                    res["strategy"] = "Baseline"
                    res["baseline_best_cost"] = math.nan
                    res["cost_saving_vs_baseline"] = math.nan
                    all_results.append(res)
                    if best_baseline_res is None or res.get("best_cost", math.inf) < best_baseline_res.get("best_cost", math.inf):
                        best_baseline_res = res.copy()
                    elif res.get("best_cost", math.inf) == best_baseline_res.get("best_cost", math.inf) and res.get("best_drone_customers", 0) > best_baseline_res.get("best_drone_customers", 0):
                        best_baseline_res = res.copy()
                if best_baseline_res is None:
                    continue
                best_baseline_res["baseline_best_cost"] = best_baseline_res.get("best_cost", math.inf)
                best_baseline_res["cost_saving_vs_baseline"] = 0.0
                baseline_costs[instance_path] = best_baseline_res.get("best_cost", math.inf)
                baseline_results_map[instance_path] = best_baseline_res.copy()
                result = best_baseline_res
            else:
                print(f"  [{idx}/{len(instance_paths)}] {instance_name} ({trials} trials)")
                best_test_res = None
                for t in range(trials):
                    trial_seed = SEED + t if SEED is not None else int(time.time()) + t
                    res = run_single_experiment(
                        instance_path=instance_path,
                        battery_capacity=battery,
                        same_truck_retrieval=False,
                        seed=trial_seed,
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

            base_cost = baseline_costs.get(instance_path, math.inf)
            current_cost = result.get("best_cost", math.inf)

            result["baseline_best_cost"] = base_cost

            if math.isfinite(base_cost) and base_cost > 0 and math.isfinite(current_cost):
                result["cost_saving_vs_baseline"] = ((base_cost - current_cost) / base_cost) * 100.0
            else:
                result["cost_saving_vs_baseline"] = math.nan

            current_level_results.append(result)

        valid_savings = [r.get("cost_saving_vs_baseline") for r in current_level_results
                         if isinstance(r.get("cost_saving_vs_baseline"), (int, float))
                         and math.isfinite(r.get("cost_saving_vs_baseline", math.nan))]
        valid_drones = [r.get("best_drone_customers") for r in current_level_results
                        if isinstance(r.get("best_drone_customers"), (int, float))
                        and math.isfinite(r.get("best_drone_customers", math.nan))]

        avg_saving = sum(valid_savings) / len(valid_savings) if valid_savings else 0.0
        avg_drone = sum(valid_drones) / len(valid_drones) if valid_drones else 0.0

        print(f"--- Summary for Battery {battery:.2f} ---")
        print(f"  Avg Cost Saving vs Baseline: {avg_saving:.2f}%")
        print(f"  Avg Drone Served Customers:  {avg_drone:.2f}")

    print(f"\n{'=' * 80}")
    print("实验完成")
    print(f"{'=' * 80}")

    return all_results


# ==========================================================================

# ==========================================================================


def write_summary_csv(results: Iterable[Dict[str, Any]], out_path: Path) -> None:
    """Write compact summary grouped by scale and battery_capacity."""
    """Write compact summary grouped by scale and battery_capacity."""

    # 1. Collapse to best-of-k per (instance, battery_capacity)
    by_instance_level: dict[tuple[str, float], list[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        inst = r.get("instance")
        battery = r.get("battery_capacity")
        if not isinstance(inst, str) or not isinstance(battery, (int, float)):
            continue
        by_instance_level[(inst, float(battery))].append(r)

    best_rows: list[Dict[str, Any]] = []
    for _, rows in by_instance_level.items():
        best_row = _choose_best_result(rows)
        if best_row is not None:
            best_rows.append(best_row)

    # 1.1 Recompute saving from best-of-k rows directly.
    # This avoids depending on trial rows that may not carry pre-filled saving fields.
    baseline_by_instance: dict[str, float] = {}
    for row in best_rows:
        inst = row.get("instance")
        battery = row.get("battery_capacity")
        best_cost = row.get("best_cost")
        if not isinstance(inst, str) or not isinstance(battery, (int, float)):
            continue
        if not isinstance(best_cost, (int, float)) or not math.isfinite(best_cost):
            continue
        if math.isclose(float(battery), BASELINE_BATTERY):
            baseline_by_instance[inst] = float(best_cost)

    # Fallback: if configured baseline is missing for an instance, use the minimum battery level found.
    rows_by_instance: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in best_rows:
        inst = row.get("instance")
        if isinstance(inst, str):
            rows_by_instance[inst].append(row)
    for inst, rows in rows_by_instance.items():
        if inst in baseline_by_instance:
            continue
        candidate_rows = [
            r for r in rows
            if isinstance(r.get("battery_capacity"), (int, float))
            and isinstance(r.get("best_cost"), (int, float))
            and math.isfinite(float(r.get("best_cost")))
        ]
        if not candidate_rows:
            continue
        candidate_rows.sort(key=lambda r: float(r.get("battery_capacity")))
        baseline_by_instance[inst] = float(candidate_rows[0].get("best_cost"))

    for row in best_rows:
        inst = row.get("instance")
        best_cost = row.get("best_cost")
        base_cost = baseline_by_instance.get(inst) if isinstance(inst, str) else None
        if (
            isinstance(base_cost, (int, float))
            and isinstance(best_cost, (int, float))
            and math.isfinite(float(base_cost))
            and math.isfinite(float(best_cost))
            and float(base_cost) > 0
        ):
            row["baseline_best_cost"] = float(base_cost)
            row["cost_saving_vs_baseline"] = ((float(base_cost) - float(best_cost)) / float(base_cost)) * 100.0
        else:
            row["baseline_best_cost"] = math.nan
            row["cost_saving_vs_baseline"] = math.nan

    # 2. Aggregate across instances
    grouped: dict[tuple[str, float], list[Dict[str, Any]]] = defaultdict(list)
    for r in best_rows:
        inst = r.get("instance")
        if isinstance(inst, str):
            scale = _extract_scale_label(inst)
        else:
            scale = "unknown"
        battery = r.get("battery_capacity")
        if not isinstance(battery, (int, float)):
            continue
        grouped[(scale, float(battery))].append(r)

    new_rows = []
    keys_to_update = set()

    for (scale, battery), items in sorted(grouped.items()):
        savings = [it.get("cost_saving_vs_baseline") for it in items if isinstance(it.get("cost_saving_vs_baseline"), (int, float))]
        drones = [it.get("best_drone_customers") for it in items if isinstance(it.get("best_drone_customers"), (int, float))]

        row = {
            "scale": scale,
            "battery_capacity": battery,
            "avg_cost_saving_vs_baseline": _safe_mean(savings),
            "avg_best_drone_customers": _safe_mean(drones),
        }
        new_rows.append(row)
        keys_to_update.add((str(scale), float(battery)))

    # 3. Read existing data and keep rows not updated
    existing_rows = []
    fieldnames = ["scale", "battery_capacity", "avg_cost_saving_vs_baseline", "avg_best_drone_customers"]

    if out_path.exists():
        try:
            with out_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    for row in reader:
                        try:
                            r_scale = str(row["scale"])
                            r_battery = float(row["battery_capacity"])
                            if (r_scale, r_battery) not in keys_to_update:
                                existing_rows.append(row)
                        except (ValueError, KeyError):
                            pass
        except Exception as e:
            print(f"Warning: Could not read existing summary file {out_path}: {e}. Starting fresh.")
            existing_rows = []

    # 4. Combine and sort
    all_rows = existing_rows + new_rows
    all_rows.sort(key=lambda x: (str(x.get("scale", "")), float(x.get("battery_capacity", 0))))

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
        "trial",
        "battery_capacity",
        "baseline_best_cost",
        "cost_saving_vs_baseline",
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
        "error",
    ]

    mode = "a" if append else "w"
    write_header = not append or not OUTPUT_CSV.exists()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        if write_header:
            writer.writeheader()
        for result in results_list:
            writer.writerow({key: result.get(key, math.nan) for key in csv_headers})

    action = "追加" if append else "写入"
    print(f"✅ 详细结果已{action}: {OUTPUT_CSV}")


# ==========================================================================
# Main
# ==========================================================================


def main() -> None:
    args = parse_args()

    instance_dirs = args.instance_dirs or DEFAULT_INSTANCE_DIRS
    instance_paths = collect_instance_paths(
        instance_dirs,
        instance_scope=args.instance_scope,
        regions=args.regions,
        instance_name=args.instance_name,
    )

    battery_levels = None
    if args.battery_levels:
        try:
            battery_levels = [float(x.strip()) for x in args.battery_levels.split(",")]
        except ValueError:
            print("Error: Invalid battery levels format. Use comma-separated floats.")
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

    results = run_battery_sensitivity_analysis(
        instance_paths,
        battery_levels,
        skip_baseline=args.skip_baseline,
        trials=args.trials,
    )

    write_results(results, append=args.append)

    summary_path = OUTPUT_DIR / "battery_sensitivity_summary.csv"
    write_summary_csv(results, summary_path)

    print("\n✅ 电池容量敏感度分析全部完成!")


if __name__ == "__main__":
    main()
