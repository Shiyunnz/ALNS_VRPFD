"""
 vs  

 ALNS ：
1.  (Flexible):  (same_truck_retrieval=False)
2.  (Same-Truck):  (same_truck_retrieval=True)

:  (Same-Truck)
"""

from __future__ import annotations
from typing import Any, Dict, Iterable, List
from collections import defaultdict
import time
import random
import math
import csv
import json
import argparse
import sys
import re
import statistics
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


# Load default configuration from YAML
_default_config = ALNSConfig()


# ==========================================================================

# ==========================================================================

# : baseline  same_truck
DOCKING_MODES = {
    "same_truck": {
        "name": "Same-Truck",
        "same_truck_retrieval": True,
        "is_baseline": True,
    },
    "flexible": {
        "name": "Flexible",
        "same_truck_retrieval": False,
        "is_baseline": False,
    },
}

# （：Instance25）
DEFAULT_INSTANCE_DIRS = [Path("data/Instance25")]

# ALNS
ITERATIONS = 2000
TIME_LIMIT = _default_config.time_limit
SEED = _default_config.seed
DRONE_PRIORITY = _default_config.drone_priority
DRONE_BONUS_OVERRIDES: Dict[str, Any] = {}
REPAIR_WEIGHTS_OVERRIDE: tuple[float, float, float] | None = None
OPERATOR_PROFILE = "lite"
FLEX_TWO_STAGE_WARMUP_ITERS = 0
FLEX_TWO_STAGE_SEED_OFFSET = 1000003
DEFAULT_TRIALS = 10
DEFAULT_SEED_START = int(SEED) if SEED is not None else 20250212
TIE_EPSILON_PERCENT = 0.1

# Output configuration (defaults; can be overridden via CLI)
OUTPUT_DIR = Path(__file__).parent / "results_new" / "drone_flexibility"
OUTPUT_PREFIX = "docking_flexibility"
OUTPUT_CSV = OUTPUT_DIR / f"{OUTPUT_PREFIX}_results.csv"
SUMMARY_CSV = OUTPUT_DIR / f"{OUTPUT_PREFIX}_summary.csv"


def _build_sa_config(instance) -> SANNCfg:
    """YAMLSANNCfg"""
    """从YAML配置构建SANNCfg"""
    sa_config_dict = _default_config.build_sa_config_dict()
    sa_config_dict['size'] = _infer_size(instance)
    sa_config_dict['iterations'] = ITERATIONS
    return SANNCfg(**sa_config_dict)


def _current_drone_bonus_kwargs() -> Dict[str, Any]:
    bonus = dict(_default_config.drone_bonus)
    if DRONE_BONUS_OVERRIDES:
        bonus.update(DRONE_BONUS_OVERRIDES)
    if REPAIR_WEIGHTS_OVERRIDE is not None:
        bonus["weights"] = REPAIR_WEIGHTS_OVERRIDE
    return bonus


def _configure_instance_robustness(instance, *, same_truck_retrieval: bool) -> None:
    """Apply robustness settings with the requested docking mode."""
    """Apply robustness settings with the requested docking mode."""
    instance.configure_robustness(
        drone_battery_capacity=_default_config.drone_battery_capacity,
        energy_uncertainty_budget=_default_config.energy_uncertainty_budget,
        energy_deviation_rate=_default_config.energy_deviation_rate,

        same_truck_retrieval=same_truck_retrieval,
    )


def _build_evaluator(instance) -> Evaluator:
    return Evaluator(
        instance,
        rendezvous_tolerance=_default_config.drone_rendezvous_tolerance,
        forced_drone_customers=_default_config.forced_drone_customers,
        allow_multiple_launch_per_node=_default_config.relax_allow_multiple_launch_per_node,
    )


def _run_alns_stage(
    *,
    instance,
    initial_solution,
    same_truck_retrieval: bool,
    seed: int,
    iterations: int,
):
    """Run one ALNS stage under a fixed docking mode."""
    """Run one ALNS stage under a fixed docking mode."""
    _configure_instance_robustness(
        instance, same_truck_retrieval=same_truck_retrieval)
    evaluator = _build_evaluator(instance)
    forced_drone_customers = _default_config.forced_drone_customers
    drone_bonus_kwargs = _current_drone_bonus_kwargs()
    destroy_ops, repair_ops = build_operators(
        instance,
        seed=seed,
        drone_priority=DRONE_PRIORITY,
        repair_set="all",
        enable_composite=True,
        operator_profile=OPERATOR_PROFILE,
        drone_bonus_kwargs=drone_bonus_kwargs,
        forced_drone_customers=forced_drone_customers,
    )

    cfg = _build_sa_config(instance)
    cfg.iterations = int(iterations)
    alns = SimulatedAnnealingALNS(
        instance=instance,
        destroy_ops=destroy_ops,
        repair_ops=repair_ops,
        evaluator=evaluator,
        cfg=cfg,
        rng=random.Random(seed),
    )

    start_time = time.perf_counter()
    best_solution = alns.run(initial_solution, time_limit=TIME_LIMIT)
    run_time = time.perf_counter() - start_time
    run_stats = getattr(alns, "last_run_stats", {})
    return best_solution, evaluator, run_time, run_stats


# ==========================================================================
# Parameter parsing and utility functions
# ==========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="灵活起降 vs 同车回收 对比分析"
    )
    parser.add_argument(
        "--instance-dir",
        action="append",
        dest="instance_dirs",
        help="算例目录路径，支持多次指定",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="跳过运行基线（same_truck）实验，假设已运行并从 CSV 读取。",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="将结果追加写入已有 CSV（若文件不存在则自动创建）",
    )
    parser.add_argument(
        "--instance-pattern",
        type=str,
        default=None,
        help="仅保留匹配正则的算例文件名 (匹配 basename，例如 'R_50_50_')",
    )
    parser.add_argument(
        "--instance-scope",
        choices=["all", "region", "single"],
        default="all",
        help="算例范围：all(全部) | region(按区域) | single(单算例)",
    )
    parser.add_argument(
        "--regions",
        type=str,
        default="30,40,50",
        help="区域列表，仅 instance-scope=region 时生效，如 30,40,50",
    )
    parser.add_argument(
        "--instance-name",
        type=str,
        default=None,
        help="单算例名称或完整路径，仅 instance-scope=single 时生效",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="结果输出目录（默认: sensitivity/results_new/drone_flexibility）",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="结果文件名前缀（默认: docking_flexibility）",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="随机种子列表（逗号分隔），如: 20250212,20250213",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help=f"试验次数；当未显式给 --seeds 时，将从 seed-start 连续生成（默认: {DEFAULT_TRIALS}）",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=DEFAULT_SEED_START,
        help=f"未给 --seeds 时的起始种子（默认: {DEFAULT_SEED_START}）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印筛选后的算例与种子，不执行实验。",
    )
    return parser.parse_args()


def collect_instance_paths(instance_dirs: Iterable[str | Path]) -> List[str]:
    """。"""
    """根据目录列表收集算例文件路径。"""
    collected: list[str] = []
    for directory in instance_dirs:
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"算例目录不存在: {dir_path}")

        for instance_file in sorted(dir_path.glob("*.txt")):
            collected.append(str(instance_file))

    if not collected:
        raise ValueError(
            "未在指定目录中找到任何 .txt 算例文件，请检查目录路径是否正确。"
        )

    return collected


def filter_instance_paths(instance_paths: List[str], pattern: str | None) -> List[str]:
    """Filter instance paths by regex pattern applied to basename."""
    """Filter instance paths by regex pattern applied to basename."""
    if not pattern:
        return instance_paths
    regex = re.compile(pattern)
    filtered = [p for p in instance_paths if regex.search(Path(p).name)]
    if not filtered:
        raise ValueError(f"没有算例匹配 pattern: {pattern}")
    return filtered


def parse_seed_values(
    seeds_text: str | None,
    *,
    trials: int,
    seed_start: int,
) -> List[int]:
    """Parse paired seeds. If --seeds not given, generate consecutive seeds."""
    """Parse paired seeds. If --seeds not given, generate consecutive seeds."""
    if seeds_text:
        seeds = []
        for part in seeds_text.split(","):
            token = part.strip()
            if not token:
                continue
            seeds.append(int(token))
        if not seeds:
            raise ValueError("参数 --seeds 解析后为空，请提供至少一个整数种子。")
        return seeds

    if trials <= 0:
        raise ValueError("--trials 必须为正整数。")

    return [int(seed_start) + i for i in range(int(trials))]


def _extract_instance_stem(instance_path: str) -> str:
    return Path(instance_path).stem


def _extract_region_id(instance_path: str) -> int | None:
    """Parse region id from name like R_40_25_1."""
    """Parse region id from name like R_40_25_1."""
    name = _extract_instance_stem(instance_path)
    match = re.match(r"R_(\d+)_", name)
    if not match:
        return None
    return int(match.group(1))


def parse_regions(text: str) -> set[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("--regions 解析为空，请提供如 30,40,50。")
    regions = {int(p) for p in parts}
    invalid = [r for r in regions if r not in {30, 40, 50}]
    if invalid:
        raise ValueError(f"--regions 包含非法值: {sorted(invalid)}，仅支持 30/40/50。")
    return regions


def apply_instance_scope(
    instance_paths: List[str],
    *,
    scope: str,
    regions_text: str,
    instance_name: str | None,
) -> List[str]:
    """Apply instance scope selection."""
    """Apply instance scope selection."""
    if scope == "all":
        return instance_paths

    if scope == "region":
        target_regions = parse_regions(regions_text)
        filtered = [
            p for p in instance_paths
            if _extract_region_id(p) in target_regions
        ]
        if not filtered:
            raise ValueError(f"instance-scope=region 未匹配到区域 {sorted(target_regions)} 的算例。")
        return filtered

    # single
    if not instance_name:
        raise ValueError("instance-scope=single 时必须提供 --instance-name。")
    target = instance_name.strip()
    target_stem = Path(target).stem
    filtered = [
        p for p in instance_paths
        if Path(p).name == Path(target).name or _extract_instance_stem(p) == target_stem
    ]
    if not filtered:
        raise ValueError(f"instance-scope=single 未匹配到算例: {instance_name}")
    return [filtered[0]]


# ==========================================================================

# ==========================================================================

def count_drone_served_customers(solution) -> int:
    """。"""
    """统计无人机服务的客户点数量。"""
    drone_customers = set()
    for task in solution.drone_tasks:
        drone_customers.update(task.customers())
    return len(drone_customers)


def serialize_truck_routes(solution) -> str:
    """Serialize truck routes to JSON string."""
    """Serialize truck routes to JSON string."""
    payload = []
    for route in solution.truck_routes:
        payload.append({
            "truck_id": route.id,
            "nodes": list(route.nodes),
        })
    return json.dumps(payload, ensure_ascii=False)


def serialize_drone_tasks(solution) -> str:
    """Serialize drone tasks to JSON string."""
    """Serialize drone tasks to JSON string."""
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


def run_single_experiment(
    instance_path: str,
    mode_key: str,
    seed: int,
) -> Dict[str, Any]:
    """。"""
    """运行单个模式配置实验。"""

    mode_config = DOCKING_MODES[mode_key]
    print(
        "  Running: "
        f"{mode_config['name']} (same_truck={mode_config['same_truck_retrieval']}, seed={seed})"
    )

    instance = read_instance(instance_path, strategy="class_based")

    # Align drone endurance to infinity for standard comparison
    if 'drone' in instance.vehicle_specs:
        instance.vehicle_specs['drone'].endurance = float('inf')

    _configure_instance_robustness(
        instance, same_truck_retrieval=mode_config["same_truck_retrieval"])
    evaluator = _build_evaluator(instance)

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
                "seed": seed,
                "mode": mode_key,
                "mode_name": mode_config["name"],
                "same_truck_retrieval": mode_config["same_truck_retrieval"],
                "initial_cost": math.inf,
                "best_cost": math.inf,
                "cost_reduction_percent": math.nan,
                "initial_drone_customers": 0,
                "best_drone_customers": 0,
                "feasible": False,
                "run_time": 0.0,
                "run_iterations": 0,
                "configured_iterations": ITERATIONS,
                "termination_reason": "initial_failed",
                "operator_profile": OPERATOR_PROFILE,
                "truck_distance_cost": math.nan,
                "drone_distance_cost": math.nan,
                "truck_routes": "",
                "drone_tasks": "",
                "error": f"initial_failed: {fallback_exc}",
            }

    local_seed = seed

    two_stage_enabled = (
        mode_key == "flexible"
        and int(FLEX_TWO_STAGE_WARMUP_ITERS) > 0
        and int(ITERATIONS) > 1
    )
    if two_stage_enabled:
        warmup_iters = min(int(FLEX_TWO_STAGE_WARMUP_ITERS), int(ITERATIONS) - 1)
        final_iters = int(ITERATIONS) - warmup_iters
        warm_seed = int(local_seed) + int(FLEX_TWO_STAGE_SEED_OFFSET)

        warm_solution, _warm_evaluator, warm_time, warm_stats = _run_alns_stage(
            instance=instance,
            initial_solution=initial_solution,
            same_truck_retrieval=True,
            seed=warm_seed,
            iterations=warmup_iters,
        )
        best_solution, evaluator, final_time, final_stats = _run_alns_stage(
            instance=instance,
            initial_solution=warm_solution,
            same_truck_retrieval=False,
            seed=local_seed,
            iterations=final_iters,
        )
        run_time = warm_time + final_time
        run_iterations = int(warm_stats.get("executed_iterations", 0) or 0) + int(
            final_stats.get("executed_iterations", 0) or 0
        )
        configured_iterations = int(warm_stats.get("configured_iterations", warmup_iters) or warmup_iters) + int(
            final_stats.get("configured_iterations", final_iters) or final_iters
        )
        warm_reason = str(warm_stats.get("termination_reason", "unknown"))
        final_reason = str(final_stats.get("termination_reason", "unknown"))
        termination_reason = f"two_stage[{warm_reason}+{final_reason}]"
    else:
        best_solution, evaluator, run_time, run_stats = _run_alns_stage(
            instance=instance,
            initial_solution=initial_solution,
            same_truck_retrieval=mode_config["same_truck_retrieval"],
            seed=local_seed,
            iterations=int(ITERATIONS),
        )
        run_iterations = int(run_stats.get("executed_iterations", 0) or 0)
        configured_iterations = int(run_stats.get("configured_iterations", ITERATIONS) or ITERATIONS)
        termination_reason = str(run_stats.get("termination_reason", "unknown"))

    try:
        best_eval = evaluator.evaluate_solution(best_solution)
        best_drone_customers = count_drone_served_customers(best_solution)
    except Exception as exc:
        print(f"    ! Best solution evaluation failed: {exc}.")
        return {
            "instance": instance_path,
            "seed": seed,
            "mode": mode_key,
            "mode_name": mode_config["name"],
            "same_truck_retrieval": mode_config["same_truck_retrieval"],
            "initial_cost": initial_cost,
            "best_cost": math.inf,
            "cost_reduction_percent": math.nan,
            "initial_drone_customers": initial_drone_customers,
            "best_drone_customers": 0,
            "feasible": False,
            "run_time": run_time,
            "run_iterations": run_iterations,
            "configured_iterations": configured_iterations,
            "termination_reason": termination_reason,
            "operator_profile": OPERATOR_PROFILE,
            "truck_distance_cost": math.nan,
            "drone_distance_cost": math.nan,
            "truck_routes": "",
            "drone_tasks": "",
            "error": f"best_eval_failed: {exc}",
        }

    cost_reduction = math.nan
    if (math.isfinite(initial_cost)
            and math.isfinite(best_eval.total_cost)
            and initial_cost > 0):
        cost_reduction = (initial_cost - best_eval.total_cost) / initial_cost * 100

    return {
        "instance": instance_path,
        "seed": seed,
        "mode": mode_key,
        "mode_name": mode_config["name"],
        "same_truck_retrieval": mode_config["same_truck_retrieval"],
        "initial_cost": initial_cost,
        "best_cost": best_eval.total_cost,
        "cost_reduction_percent": cost_reduction,
        "initial_drone_customers": initial_drone_customers,
        "best_drone_customers": best_drone_customers,
        "feasible": best_eval.feasible,
        "run_time": run_time,
        "run_iterations": run_iterations,
        "configured_iterations": configured_iterations,
        "termination_reason": termination_reason,
        "operator_profile": OPERATOR_PROFILE,
        "truck_distance_cost": best_eval.truck_distance_cost,
        "drone_distance_cost": best_eval.drone_distance_cost,
        "truck_routes": serialize_truck_routes(best_solution),
        "drone_tasks": serialize_drone_tasks(best_solution),
    }


def load_baseline_from_csv(
    instance_paths: List[str],
    output_csv: Path,
    seeds: List[int],
) -> Dict[tuple[str, int], float]:
    """ CSV  (same_truck )。"""
    """从现有 CSV 加载基线成本 (same_truck 模式)。"""
    baseline_costs: dict[tuple[str, int], float] = {}
    if not output_csv.exists():
        print("  Warning: No existing CSV found, will use inf for baselines.")
        return {(path, seed): math.inf for path in instance_paths for seed in seeds}

    with open(output_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            instance = row.get("instance")
            mode = row.get("mode", "")
            if instance not in instance_paths or mode != "same_truck":
                continue

            seed_value = row.get("seed")
            if seed_value is None or seed_value == "":
                continue

            try:
                parsed_seed = int(seed_value)
                cost = float(row.get("best_cost", "inf"))
            except ValueError:
                continue

            if parsed_seed in seeds:
                baseline_costs[(instance, parsed_seed)] = cost

    for path in instance_paths:
        for seed in seeds:
            baseline_costs.setdefault((path, seed), math.inf)
    return baseline_costs


def _extract_scale_label(instance_path: str) -> str:
    path = Path(instance_path)
    try:
        return path.parent.name or "unknown"
    except IndexError:
        return "unknown"


def run_docking_comparison(
    instance_paths: List[str],
    seeds: List[int],
    skip_baseline: bool = False,
    baseline_csv: Path | None = None,
) -> List[Dict[str, Any]]:
    """。"""
    """运行对比实验并返回结果列表。"""

    print("=" * 80)
    print("灵活起降 vs 同车回收 对比分析")
    print("=" * 80)
    print(f"测试算例: {len(instance_paths)} 个")
    print(f"对比模式: Same-Truck (baseline) vs Flexible")
    print(f"种子列表: {seeds}")
    print(f"迭代次数: {ITERATIONS}")
    print(f"时间限制: {TIME_LIMIT}s")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results: list[Dict[str, Any]] = []
    baseline_costs: dict[tuple[str, int], float] = {}

    if skip_baseline:
        print("\n跳过基线运行：将从已有 CSV 加载 same_truck 成本。")
        baseline_costs = load_baseline_from_csv(instance_paths, baseline_csv or OUTPUT_CSV, seeds)

    print("\n按算例逐个执行：每个 seed 先 Same-Truck，再 Flexible，并做对比")

    for idx, instance_path in enumerate(instance_paths, 1):
        instance_name = Path(instance_path).stem
        print(f"\n[{idx}/{len(instance_paths)}] {instance_name}")

        for trial_index, seed in enumerate(seeds, 1):
            print(f"  Seed [{trial_index}/{len(seeds)}]: {seed}")
            base_cost = baseline_costs.get((instance_path, seed), math.inf)

            if not skip_baseline:
                baseline_result = run_single_experiment(
                    instance_path=instance_path,
                    mode_key="same_truck",
                    seed=seed,
                )
                base_cost = baseline_result.get("best_cost", math.inf)
                baseline_costs[(instance_path, seed)] = base_cost

                baseline_result["trial_index"] = trial_index
                baseline_result["strategy"] = "Baseline"
                baseline_result["baseline_best_cost"] = base_cost
                baseline_result["cost_saving_vs_baseline"] = 0.0
                all_results.append(baseline_result)
            elif not math.isfinite(base_cost):
                print("    Warning: 未找到该算例该 seed 的 same_truck 成本，节省率将为 NaN。")

            flexible_result = run_single_experiment(
                instance_path=instance_path,
                mode_key="flexible",
                seed=seed,
            )
            flexible_result["trial_index"] = trial_index
            flexible_result["strategy"] = "Test"
            flexible_result["baseline_best_cost"] = base_cost

            current_cost = flexible_result.get("best_cost", math.inf)
            if math.isfinite(base_cost) and base_cost > 0 and math.isfinite(current_cost):
                saving = (base_cost - current_cost) / base_cost * 100.0
                flexible_result["cost_saving_vs_baseline"] = saving
                print(
                    f"    对比结果: same_truck={base_cost:.2f}, "
                    f"flexible={current_cost:.2f}, saving={saving:.2f}%"
                )
            else:
                flexible_result["cost_saving_vs_baseline"] = math.nan
                print("    对比结果: 成本不可用，saving=NaN")

            all_results.append(flexible_result)

    # Summary
    valid_savings = [r["cost_saving_vs_baseline"] for r in all_results 
                     if r.get("mode") == "flexible" and math.isfinite(r.get("cost_saving_vs_baseline", math.nan))]
    
    avg_saving = sum(valid_savings) / len(valid_savings) if valid_savings else 0.0

    print(f"\n{'=' * 80}")
    print("实验完成")
    print(f"平均成本节省 (Flexible vs Same-Truck): {avg_saving:.2f}%")
    print(f"{'=' * 80}")

    return all_results


def write_summary_csv(results: Iterable[Dict[str, Any]], out_path: Path) -> None:
    """Write summary with secondary caliber: mode-independent best-of-k."""
    """Write summary with secondary caliber: mode-independent best-of-k."""
    best_rows = build_best_instance_rows(results)

    def _to_float(v: Any) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return math.nan

    summary_rows: list[Dict[str, Any]] = []

    # Instance-level rows (secondary caliber base rows)
    for row in best_rows:
        same_cost = _to_float(row.get("same_cost"))
        flex_cost = _to_float(row.get("flexible_cost"))
        saving = _to_float(row.get("flexible_saving_vs_same"))
        same_drone = _to_float(row.get("same_best_drone_customers"))
        flex_drone = _to_float(row.get("flex_best_drone_customers"))
        summary_rows.append({
            "summary_level": "instance",
            "group": row.get("instance_name", ""),
            "region": row.get("region", ""),
            "num_instances": 1,
            "avg_same_cost": same_cost,
            "avg_flexible_cost": flex_cost,
            "avg_flexible_saving_vs_same": saving,
            "std_flexible_saving_vs_same": 0.0,
            "min_flexible_saving_vs_same": saving,
            "max_flexible_saving_vs_same": saving,
            "avg_same_best_drone_customers": same_drone,
            "avg_flex_best_drone_customers": flex_drone,
        })

    # Region-level aggregation
    by_region: dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in best_rows:
        reg = row.get("region")
        if isinstance(reg, int):
            by_region[reg].append(row)

    for reg in sorted(by_region.keys()):
        rows = by_region[reg]
        same_costs = [_to_float(r.get("same_cost")) for r in rows]
        flex_costs = [_to_float(r.get("flexible_cost")) for r in rows]
        savings = [_to_float(r.get("flexible_saving_vs_same")) for r in rows]
        same_drones = [_to_float(r.get("same_best_drone_customers")) for r in rows]
        flex_drones = [_to_float(r.get("flex_best_drone_customers")) for r in rows]
        finite_savings = [s for s in savings if math.isfinite(s)]
        summary_rows.append({
            "summary_level": "region",
            "group": f"R{reg}",
            "region": reg,
            "num_instances": len(rows),
            "avg_same_cost": _safe_mean(same_costs),
            "avg_flexible_cost": _safe_mean(flex_costs),
            "avg_flexible_saving_vs_same": _safe_mean(savings),
            "std_flexible_saving_vs_same": _safe_std(savings),
            "min_flexible_saving_vs_same": min(finite_savings) if finite_savings else math.nan,
            "max_flexible_saving_vs_same": max(finite_savings) if finite_savings else math.nan,
            "avg_same_best_drone_customers": _safe_mean(same_drones),
            "avg_flex_best_drone_customers": _safe_mean(flex_drones),
        })

    # Overall aggregation
    if best_rows:
        same_costs = [_to_float(r.get("same_cost")) for r in best_rows]
        flex_costs = [_to_float(r.get("flexible_cost")) for r in best_rows]
        savings = [_to_float(r.get("flexible_saving_vs_same")) for r in best_rows]
        same_drones = [_to_float(r.get("same_best_drone_customers")) for r in best_rows]
        flex_drones = [_to_float(r.get("flex_best_drone_customers")) for r in best_rows]
        finite_savings = [s for s in savings if math.isfinite(s)]
        summary_rows.append({
            "summary_level": "all",
            "group": "ALL",
            "region": "",
            "num_instances": len(best_rows),
            "avg_same_cost": _safe_mean(same_costs),
            "avg_flexible_cost": _safe_mean(flex_costs),
            "avg_flexible_saving_vs_same": _safe_mean(savings),
            "std_flexible_saving_vs_same": _safe_std(savings),
            "min_flexible_saving_vs_same": min(finite_savings) if finite_savings else math.nan,
            "max_flexible_saving_vs_same": max(finite_savings) if finite_savings else math.nan,
            "avg_same_best_drone_customers": _safe_mean(same_drones),
            "avg_flex_best_drone_customers": _safe_mean(flex_drones),
        })

    fieldnames = [
        "summary_level",
        "group",
        "region",
        "num_instances",
        "avg_same_cost",
        "avg_flexible_cost",
        "avg_flexible_saving_vs_same",
        "std_flexible_saving_vs_same",
        "min_flexible_saving_vs_same",
        "max_flexible_saving_vs_same",
        "avg_same_best_drone_customers",
        "avg_flex_best_drone_customers",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"✅ 汇总结果已保存（主口径 max-saving）: {out_path}")


def _safe_std(values: List[float]) -> float:
    filtered = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if len(filtered) < 2:
        return 0.0
    return float(statistics.stdev(filtered))


def _select_mode_best(rows: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """Select best-of-k by min cost; tie-break by max drone-served customers."""
    """Select best-of-k by min cost; tie-break by max drone-served customers."""
    valid = [
        r for r in rows
        if isinstance(r.get("best_cost"), (int, float))
        and math.isfinite(float(r["best_cost"]))
    ]
    if not valid:
        return None
    return min(
        valid,
        key=lambda r: (
            float(r.get("best_cost", math.inf)),
            -float(r.get("best_drone_customers", 0.0)),
        ),
    )


def build_best_instance_rows(results: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build primary-caliber rows: select the maximum saving trial per instance."""
    """Build primary-caliber rows: select the maximum saving trial per instance."""
    paired_rows = build_paired_trial_rows(results)
    grouped: dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in paired_rows:
        inst = row.get("instance")
        if isinstance(inst, str):
            grouped[inst].append(row)

    rows_out: list[Dict[str, Any]] = []
    for instance in sorted(grouped.keys()):
        rows = grouped[instance]
        valid_rows = [
            r for r in rows
            if isinstance(r.get("delta_pct_flex_vs_same"), (int, float))
            and math.isfinite(float(r["delta_pct_flex_vs_same"]))
        ]
        if valid_rows:
            # ： saving ； flexible cost。
            selected = max(
                valid_rows,
                key=lambda r: (
                    float(r["delta_pct_flex_vs_same"]),
                    -float(r.get("flexible_cost", math.inf)),
                ),
            )
        else:
            # saving， flexible_cost 。
            selected = min(rows, key=lambda r: float(r.get("flexible_cost", math.inf)))

        rows_out.append({
            "instance": instance,
            "instance_name": selected.get("instance_name", _extract_instance_stem(instance)),
            "region": selected.get("region", _extract_region_id(instance)),
            "same_best_seed": selected.get("seed"),
            "flex_best_seed": selected.get("seed"),
            "num_same_trials": len(rows),
            "num_flex_trials": len(rows),
            "same_cost": selected.get("same_cost", math.nan),
            "flexible_cost": selected.get("flexible_cost", math.nan),
            "flexible_saving_vs_same": selected.get("delta_pct_flex_vs_same", math.nan),
            "same_best_drone_customers": selected.get("same_best_drone_customers", math.nan),
            "flex_best_drone_customers": selected.get("flex_best_drone_customers", math.nan),
            "same_truck_routes": "",
            "same_drone_tasks": "",
            "flexible_truck_routes": "",
            "flexible_drone_tasks": "",
        })
    return rows_out


def write_best_instance_csv(rows: Iterable[Dict[str, Any]], out_path: Path) -> None:
    """Write final per-instance rows using best-of-k for each mode."""
    """Write final per-instance rows using best-of-k for each mode."""
    rows_list = list(rows)
    fieldnames = [
        "instance",
        "instance_name",
        "region",
        "same_best_seed",
        "flex_best_seed",
        "num_same_trials",
        "num_flex_trials",
        "same_cost",
        "flexible_cost",
        "flexible_saving_vs_same",
        "same_best_drone_customers",
        "flex_best_drone_customers",
        "same_truck_routes",
        "same_drone_tasks",
        "flexible_truck_routes",
        "flexible_drone_tasks",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_list:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(f"✅ 最终结果(best-of-k, 按模式独立取最优)已保存: {out_path}")


def build_paired_trial_rows(results: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build paired rows per (instance, seed): flexible vs same-truck."""
    """Build paired rows per (instance, seed): flexible vs same-truck."""
    paired: dict[tuple[str, int], dict[str, Dict[str, Any]]] = defaultdict(dict)
    for item in results:
        instance = item.get("instance")
        seed = item.get("seed")
        mode = item.get("mode")
        if not isinstance(instance, str) or not isinstance(seed, int) or not isinstance(mode, str):
            continue
        if mode not in {"same_truck", "flexible"}:
            continue
        paired[(instance, seed)][mode] = item

    out: list[Dict[str, Any]] = []
    for (instance, seed), mode_rows in sorted(paired.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        same_row = mode_rows.get("same_truck")
        flex_row = mode_rows.get("flexible")
        if same_row is None or flex_row is None:
            continue
        same_cost = float(same_row.get("best_cost", math.inf))
        flex_cost = float(flex_row.get("best_cost", math.inf))
        if same_cost > 0 and math.isfinite(same_cost) and math.isfinite(flex_cost):
            delta_pct = (same_cost - flex_cost) / same_cost * 100.0
            delta_cost = same_cost - flex_cost
        else:
            delta_pct = math.nan
            delta_cost = math.nan
        same_drone = float(same_row.get("best_drone_customers", math.nan))
        flex_drone = float(flex_row.get("best_drone_customers", math.nan))
        delta_drone = flex_drone - same_drone if math.isfinite(same_drone) and math.isfinite(flex_drone) else math.nan
        out.append({
            "instance": instance,
            "instance_name": _extract_instance_stem(instance),
            "region": _extract_region_id(instance),
            "seed": seed,
            "same_cost": same_cost,
            "flexible_cost": flex_cost,
            "delta_cost": delta_cost,
            "delta_pct_flex_vs_same": delta_pct,
            "same_best_drone_customers": same_drone,
            "flex_best_drone_customers": flex_drone,
            "delta_drone_customers": delta_drone,
            "same_feasible": same_row.get("feasible", False),
            "flex_feasible": flex_row.get("feasible", False),
        })
    return out


def write_paired_trial_csv(rows: Iterable[Dict[str, Any]], out_path: Path) -> None:
    rows_list = list(rows)
    fieldnames = [
        "instance",
        "instance_name",
        "region",
        "seed",
        "same_cost",
        "flexible_cost",
        "delta_cost",
        "delta_pct_flex_vs_same",
        "same_best_drone_customers",
        "flex_best_drone_customers",
        "delta_drone_customers",
        "same_feasible",
        "flex_feasible",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_list)
    print(f"✅ 配对试验结果已保存: {out_path}")


def build_instance_stats_rows(
    paired_rows: Iterable[Dict[str, Any]],
    *,
    tie_epsilon_pct: float = TIE_EPSILON_PERCENT,
) -> List[Dict[str, Any]]:
    grouped: dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in paired_rows:
        inst = row.get("instance")
        if isinstance(inst, str):
            grouped[inst].append(row)

    out: list[Dict[str, Any]] = []
    for inst in sorted(grouped.keys()):
        rows = grouped[inst]
        deltas = [
            float(r.get("delta_pct_flex_vs_same", math.nan))
            for r in rows
            if isinstance(r.get("delta_pct_flex_vs_same"), (int, float))
            and math.isfinite(float(r.get("delta_pct_flex_vs_same", math.nan)))
        ]
        drone_deltas = [
            float(r.get("delta_drone_customers", math.nan))
            for r in rows
            if isinstance(r.get("delta_drone_customers"), (int, float))
            and math.isfinite(float(r.get("delta_drone_customers", math.nan)))
        ]
        n = len(deltas)
        if n == 0:
            continue
        win = sum(1 for d in deltas if d > tie_epsilon_pct)
        tie = sum(1 for d in deltas if abs(d) <= tie_epsilon_pct)
        loss = sum(1 for d in deltas if d < -tie_epsilon_pct)
        out.append({
            "instance": inst,
            "instance_name": _extract_instance_stem(inst),
            "region": _extract_region_id(inst),
            "num_paired_trials": n,
            "mean_delta_pct_flex_vs_same": _safe_mean(deltas),
            "std_delta_pct_flex_vs_same": _safe_std(deltas),
            "min_delta_pct_flex_vs_same": min(deltas),
            "max_delta_pct_flex_vs_same": max(deltas),
            "win_rate_pct": (win / n) * 100.0,
            "tie_rate_pct": (tie / n) * 100.0,
            "loss_rate_pct": (loss / n) * 100.0,
            "mean_delta_drone_customers": _safe_mean(drone_deltas),
        })
    return out


def write_instance_stats_csv(rows: Iterable[Dict[str, Any]], out_path: Path) -> None:
    rows_list = list(rows)
    fieldnames = [
        "instance",
        "instance_name",
        "region",
        "num_paired_trials",
        "mean_delta_pct_flex_vs_same",
        "std_delta_pct_flex_vs_same",
        "min_delta_pct_flex_vs_same",
        "max_delta_pct_flex_vs_same",
        "win_rate_pct",
        "tie_rate_pct",
        "loss_rate_pct",
        "mean_delta_drone_customers",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_list)
    print(f"✅ 算例级统计已保存: {out_path}")


def build_region_stats_rows(instance_rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_region: dict[str, List[Dict[str, Any]]] = defaultdict(list)
    rows_list = [r for r in instance_rows]
    for row in rows_list:
        reg = row.get("region")
        key = str(reg) if isinstance(reg, int) else "unknown"
        by_region[key].append(row)

    out: list[Dict[str, Any]] = []
    def _aggregate(name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        mean_vals = [float(r["mean_delta_pct_flex_vs_same"]) for r in rows]
        win_vals = [float(r["win_rate_pct"]) for r in rows]
        tie_vals = [float(r["tie_rate_pct"]) for r in rows]
        loss_vals = [float(r["loss_rate_pct"]) for r in rows]
        drone_vals = [float(r["mean_delta_drone_customers"]) for r in rows]
        return {
            "region": name,
            "num_instances": len(rows),
            "avg_instance_mean_delta_pct": _safe_mean(mean_vals),
            "std_instance_mean_delta_pct": _safe_std(mean_vals),
            "avg_instance_win_rate_pct": _safe_mean(win_vals),
            "avg_instance_tie_rate_pct": _safe_mean(tie_vals),
            "avg_instance_loss_rate_pct": _safe_mean(loss_vals),
            "avg_instance_mean_delta_drone_customers": _safe_mean(drone_vals),
        }

    for reg in sorted(by_region.keys(), key=lambda x: (x == "unknown", x)):
        out.append(_aggregate(reg, by_region[reg]))
    if rows_list:
        out.append(_aggregate("ALL", rows_list))
    return out


def write_region_stats_csv(rows: Iterable[Dict[str, Any]], out_path: Path) -> None:
    rows_list = list(rows)
    fieldnames = [
        "region",
        "num_instances",
        "avg_instance_mean_delta_pct",
        "std_instance_mean_delta_pct",
        "avg_instance_win_rate_pct",
        "avg_instance_tie_rate_pct",
        "avg_instance_loss_rate_pct",
        "avg_instance_mean_delta_drone_customers",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_list)
    print(f"✅ 区域级统计已保存: {out_path}")


def write_results(
    results: Iterable[Dict[str, Any]],
    *,
    append: bool,
    output_csv: Path,
) -> None:
    results_list = list(results)
    csv_headers = [
        "instance",
        "seed",
        "trial_index",
        "strategy",
        "mode",
        "mode_name",
        "same_truck_retrieval",
        "baseline_best_cost",
        "cost_saving_vs_baseline",
        "initial_cost",
        "best_cost",
        "cost_reduction_percent",
        "initial_drone_customers",
        "best_drone_customers",
        "feasible",
        "run_time",
        "run_iterations",
        "configured_iterations",
        "termination_reason",
        "operator_profile",
        "truck_distance_cost",
        "drone_distance_cost",
        "truck_routes",
        "drone_tasks",
    ]

    mode = "a" if append else "w"
    write_header = not append or not output_csv.exists()

    with output_csv.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        if write_header:
            writer.writeheader()
        for result in results_list:
            writer.writerow({key: result.get(key, math.nan) for key in csv_headers})

    action = "追加" if append else "写入"
    print(f"✅ 详细结果已{action}: {output_csv}")


def main() -> None:
    args = parse_args()

    instance_dirs = args.instance_dirs or DEFAULT_INSTANCE_DIRS
    instance_paths = collect_instance_paths(instance_dirs)
    instance_paths = filter_instance_paths(instance_paths, args.instance_pattern)
    instance_paths = apply_instance_scope(
        instance_paths,
        scope=args.instance_scope,
        regions_text=args.regions,
        instance_name=args.instance_name,
    )
    seeds = parse_seed_values(
        args.seeds,
        trials=args.trials,
        seed_start=args.seed_start,
    )

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_prefix = args.output_prefix or OUTPUT_PREFIX
    output_csv = output_dir / f"{output_prefix}_results.csv"
    summary_csv = output_dir / f"{output_prefix}_summary.csv"

    print("选择的算例目录:")
    for directory in instance_dirs:
        print(f"  - {Path(directory)}")
    print(f"算例范围: {args.instance_scope}")
    if args.instance_scope == "region":
        print(f"区域: {args.regions}")
    if args.instance_scope == "single":
        print(f"单算例: {args.instance_name}")

    print(f"共收集到 {len(instance_paths)} 个算例文件。")
    print(f"将使用 {len(seeds)} 个 seed: {seeds}")
    if args.dry_run:
        print("\nDry-run 模式：仅展示将执行的算例列表。")
        for idx, path in enumerate(instance_paths, 1):
            print(f"  [{idx:02d}] {Path(path).stem}")
        print("\n✅ Dry-run 完成，未执行任何实验。")
        return

    results = run_docking_comparison(
        instance_paths,
        seeds=seeds,
        skip_baseline=args.skip_baseline,
        baseline_csv=output_csv,
    )
    write_results(results, append=args.append, output_csv=output_csv)

    # Generate summary CSV for plotting
    write_summary_csv(results, summary_csv)

    print("\n✅ 对比分析完成!")


if __name__ == "__main__":
    main()
