"""
Gamma multi-run + scenario replay pipeline.

Workflow:
1. For each selected instance and gamma level, run ALNS multiple times.
2. Pick the best solution per gamma and export its truck/drone routes.
3. Feed the best-per-gamma fixed solutions into scenario replay and export
   scenario-level and aggregated robustness metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
import argparse
import csv
import json
import math
import random
import sys
import time

# Ensure project root is in sys.path for CLI + VSCode run button.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

import alns_vrpfd.model.initializer as initializer
from alns_vrpfd.core.sa import SANNCfg, SimulatedAnnealingALNS
from alns_vrpfd.evaluation import (
    Evaluator,
    GammaSolutionInput,
    ScenarioDistributionConfig,
    ScenarioReplayConfig,
    run_scenario_replay,
)
from alns_vrpfd.model.solution import Solution
from alns_vrpfd.utils.config_loader import ALNSConfig
from alns_vrpfd.utils.io_utils import read_instance
from sensitivity.instance_selector import collect_instance_paths_with_scope


# Keep aligned with sensitivity/gamma_sensitivity.py
DEFAULT_GAMMA_LEVELS = [0, 1, 2, 3]
DEFAULT_INSTANCE_DIRS = [Path("data/Instance10")]
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results_new" / "scenario_replay"


@dataclass(frozen=True)
class TrialResult:
    instance: str
    gamma: int
    seed: int
    feasible: bool
    best_cost: float
    initial_cost: float
    run_time: float
    best_drone_customers: int
    truck_distance_cost: float
    drone_distance_cost: float
    truck_routes: str
    drone_tasks: str
    error: str = ""
    solution: Solution | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gamma multi-run and scenario replay."
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
        help="单算例模式下指定算例名或路径，例如 R_40_25_1 或 data/Instance25/R_40_25_1.txt",
    )
    parser.add_argument(
        "--gamma-values",
        type=str,
        default=",".join(str(g) for g in DEFAULT_GAMMA_LEVELS),
        help="Comma-separated gamma list, e.g. 0,1,2,3.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="逗号分隔随机种子列表（显式指定时覆盖 --trials/--seed-base）",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="每个算例+gamma 的独立运行次数（默认: 5）",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=20260218,
        help="自动生成种子时的起始值（共生成 trials 个）",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="ALNS iterations override; default uses config iterations.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="ALNS runtime limit per run (seconds); default uses config.",
    )
    parser.add_argument(
        "--scenario-count",
        type=int,
        default=1000,
        help="Scenario count per distribution for replay.",
    )
    parser.add_argument(
        "--replay-seed",
        type=int,
        default=2024,
        help="Random seed for scenario generation.",
    )
    parser.add_argument(
        "--distributions",
        type=str,
        default="ND,UD,NDC",
        help="Comma-separated replay distributions from ND,UD,NDC,LOGNORMAL,STUDENT_T,MIXTURE,DETERMINISTIC.",
    )
    parser.add_argument(
        "--nd-cv",
        type=float,
        default=0.1,
        help="ND distribution CV parameter.",
    )
    parser.add_argument(
        "--ud-delta",
        type=float,
        default=0.1,
        help="UD distribution half-width delta parameter.",
    )
    parser.add_argument(
        "--ndc-cv",
        type=float,
        default=0.1,
        help="NDC distribution CV parameter.",
    )
    parser.add_argument(
        "--ndc-correlation",
        type=float,
        default=0.3,
        help="NDC distribution correlation parameter in [0, 1).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="instance10_gamma",
        help="全局汇总文件前缀，例如 instance25_r40_gamma",
    )
    parser.add_argument(
        "--energy-deviation-rate",
        type=float,
        default=None,
        help="Override energy deviation rate (theta), e.g. 0.2.",
    )
    parser.add_argument(
        "--repair-weights",
        type=str,
        default=None,
        help="Optional repair candidate weights as 'cost,energy,delay', e.g. 1,0,0.",
    )
    parser.add_argument(
        "--skip-replay",
        action="store_true",
        help="Skip scenario replay and only export trials/best solution bank.",
    )
    return parser.parse_args()


def _parse_int_list(text: str, name: str) -> List[int]:
    values: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError(f"{name} list is empty after parsing.")
    return values


def _parse_dist_list(text: str) -> List[str]:
    values = []
    for token in text.split(","):
        token = token.strip().upper()
        if token:
            values.append(token)
    if not values:
        raise ValueError("Distribution list is empty after parsing.")
    return values


def _parse_float_triplet(text: str | None, name: str) -> tuple[float, float, float] | None:
    if text is None:
        return None
    tokens = [x.strip() for x in text.split(",") if x.strip()]
    if len(tokens) != 3:
        raise ValueError(f"{name} must contain exactly three comma-separated floats.")
    vals = tuple(float(x) for x in tokens)
    return vals  # type: ignore[return-value]


def _infer_size(instance) -> str:
    n = len(instance.customer_manager.customer_ids())
    if n <= 15:
        return "small"
    if n <= 50:
        return "medium"
    return "large"


def _count_drone_served_customers(solution: Solution) -> int:
    served = set()
    for task in solution.drone_tasks:
        served.update(task.customers())
    return len(served)


def _serialize_truck_routes(solution: Solution) -> str:
    payload = [{"truck_id": r.id, "nodes": list(r.nodes)} for r in solution.truck_routes]
    return json.dumps(payload, ensure_ascii=False)


def _serialize_drone_tasks(solution: Solution) -> str:
    payload = []
    for task in solution.drone_tasks:
        payload.append(
            {
                "drone_id": task.drone_id,
                "launch_truck": task.launch_truck,
                "launch_node": task.launch_node,
                "customers": list(task.customers()),
                "land_truck": task.land_truck,
                "retrieve_node": task.retrieve_node,
            }
        )
    return json.dumps(payload, ensure_ascii=False)


def _build_sa_cfg(instance, cfg_obj: ALNSConfig, iterations_override: int | None) -> SANNCfg:
    d = cfg_obj.build_sa_config_dict(size=_infer_size(instance))
    d["iterations"] = iterations_override if iterations_override is not None else cfg_obj.iterations
    return SANNCfg(**d)


def _build_initial_solution(instance, cfg_obj: ALNSConfig) -> Solution:
    use_two_phase = cfg_obj.raw.get("initial_solution", {}).get("two_phase", True)
    forced = cfg_obj.forced_drone_customers
    if use_two_phase:
        return initializer.build_two_phase_initial_solution(
            instance,
            truck_forbidden_customers=forced,
            allow_multiple_launch_per_node=cfg_obj.relax_allow_multiple_launch_per_node,
        )
    return initializer.build_initial_solution(
        instance,
        truck_forbidden_customers=forced,
        allow_multiple_launch_per_node=cfg_obj.relax_allow_multiple_launch_per_node,
    )


def _run_one_trial(
    *,
    instance_path: str,
    gamma: int,
    seed: int,
    cfg_obj: ALNSConfig,
    iterations_override: int | None,
    time_limit_override: float | None,
    energy_deviation_rate_override: float | None = None,
    repair_weights: tuple[float, float, float] | None = None,
) -> TrialResult:
    start = time.perf_counter()
    try:
        instance = read_instance(instance_path, strategy="class_based")
        if "drone" in instance.vehicle_specs:
            instance.vehicle_specs["drone"].endurance = float("inf")

        deviation_rate = (
            float(energy_deviation_rate_override)
            if energy_deviation_rate_override is not None
            else float(cfg_obj.energy_deviation_rate)
        )
        instance.configure_robustness(
            drone_battery_capacity=cfg_obj.drone_battery_capacity,
            energy_uncertainty_budget=gamma,
            energy_deviation_rate=deviation_rate,

            same_truck_retrieval=cfg_obj.same_truck_retrieval,
        )

        evaluator = Evaluator(
            instance,
            rendezvous_tolerance=cfg_obj.drone_rendezvous_tolerance,
            forced_drone_customers=cfg_obj.forced_drone_customers,
            allow_multiple_launch_per_node=cfg_obj.relax_allow_multiple_launch_per_node,
        )
        initial_solution = _build_initial_solution(instance, cfg_obj)
        initial_cost = evaluator.evaluate_solution(initial_solution).total_cost

        destroy_ops, repair_ops = build_operators(
            instance=instance,
            seed=seed,
            drone_priority=cfg_obj.drone_priority,
            repair_set="all",
            enable_composite=True,
            drone_bonus_kwargs=cfg_obj.drone_bonus,
            forced_drone_customers=cfg_obj.forced_drone_customers,
            repair_weights=repair_weights,
        )
        sa_cfg = _build_sa_cfg(instance, cfg_obj, iterations_override)
        alns = SimulatedAnnealingALNS(
            instance=instance,
            destroy_ops=destroy_ops,
            repair_ops=repair_ops,
            evaluator=evaluator,
            cfg=sa_cfg,
            rng=random.Random(seed),
        )
        best_solution = alns.run(
            initial_solution,
            time_limit=time_limit_override if time_limit_override is not None else cfg_obj.time_limit,
        )
        best_eval = evaluator.evaluate_solution(best_solution)
        elapsed = time.perf_counter() - start
        return TrialResult(
            instance=instance_path,
            gamma=gamma,
            seed=seed,
            feasible=bool(best_eval.feasible),
            best_cost=float(best_eval.total_cost),
            initial_cost=float(initial_cost),
            run_time=elapsed,
            best_drone_customers=_count_drone_served_customers(best_solution),
            truck_distance_cost=float(best_eval.truck_distance_cost),
            drone_distance_cost=float(best_eval.drone_distance_cost),
            truck_routes=_serialize_truck_routes(best_solution),
            drone_tasks=_serialize_drone_tasks(best_solution),
            solution=best_solution.clone(),
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return TrialResult(
            instance=instance_path,
            gamma=gamma,
            seed=seed,
            feasible=False,
            best_cost=math.inf,
            initial_cost=math.inf,
            run_time=elapsed,
            best_drone_customers=0,
            truck_distance_cost=math.nan,
            drone_distance_cost=math.nan,
            truck_routes="",
            drone_tasks="",
            error=str(exc),
            solution=None,
        )


def _pick_best_trial(trials: Sequence[TrialResult]) -> TrialResult:
    feasible = [t for t in trials if t.feasible and math.isfinite(t.best_cost)]
    if feasible:
        return min(feasible, key=lambda t: t.best_cost)
    return min(trials, key=lambda t: t.best_cost)


def _build_distribution_configs(
    names: Sequence[str],
    *,
    nd_cv: float = 0.1,
    ud_delta: float = 0.1,
    ndc_cv: float = 0.1,
    ndc_correlation: float = 0.3,
) -> List[ScenarioDistributionConfig]:
    configs: list[ScenarioDistributionConfig] = []
    for name in names:
        if name == "ND":
            configs.append(ScenarioDistributionConfig(name="ND", kind="ND", cv=nd_cv))
        elif name == "UD":
            configs.append(ScenarioDistributionConfig(name="UD", kind="UD", delta=ud_delta))
        elif name == "NDC":
            configs.append(
                ScenarioDistributionConfig(
                    name="NDC",
                    kind="NDC",
                    cv=ndc_cv,
                    correlation=ndc_correlation,
                )
            )
        elif name == "LOGNORMAL":
            configs.append(ScenarioDistributionConfig(name="LOGNORMAL", kind="LOGNORMAL", cv=0.1))
        elif name == "STUDENT_T":
            configs.append(
                ScenarioDistributionConfig(
                    name="STUDENT_T",
                    kind="STUDENT_T",
                    cv=0.1,
                    degrees_of_freedom=6,
                )
            )
        elif name == "MIXTURE":
            configs.append(
                ScenarioDistributionConfig(
                    name="MIXTURE",
                    kind="MIXTURE",
                    cv=0.05,
                    mixture_probability=0.2,
                    stress_mean=1.2,
                    stress_cv=0.1,
                )
            )
        elif name == "DETERMINISTIC":
            configs.append(
                ScenarioDistributionConfig(
                    name="DETERMINISTIC",
                    kind="DETERMINISTIC",
                    deterministic_multiplier=1.0,
                )
            )
        else:
            raise ValueError(f"Unsupported distribution name: {name}")
    return configs


def _write_trial_csv(path: Path, rows: Sequence[TrialResult]) -> None:
    fields = [
        "instance",
        "gamma",
        "seed",
        "feasible",
        "best_cost",
        "initial_cost",
        "run_time",
        "best_drone_customers",
        "truck_distance_cost",
        "drone_distance_cost",
        "truck_routes",
        "drone_tasks",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "instance": r.instance,
                    "gamma": r.gamma,
                    "seed": r.seed,
                    "feasible": int(r.feasible),
                    "best_cost": r.best_cost,
                    "initial_cost": r.initial_cost,
                    "run_time": r.run_time,
                    "best_drone_customers": r.best_drone_customers,
                    "truck_distance_cost": r.truck_distance_cost,
                    "drone_distance_cost": r.drone_distance_cost,
                    "truck_routes": r.truck_routes,
                    "drone_tasks": r.drone_tasks,
                    "error": r.error,
                }
            )


def _write_best_csv(path: Path, rows: Sequence[TrialResult]) -> None:
    fields = [
        "instance",
        "gamma",
        "best_seed",
        "best_cost",
        "feasible",
        "best_drone_customers",
        "truck_distance_cost",
        "drone_distance_cost",
        "truck_routes",
        "drone_tasks",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "instance": r.instance,
                    "gamma": r.gamma,
                    "best_seed": r.seed,
                    "best_cost": r.best_cost,
                    "feasible": int(r.feasible),
                    "best_drone_customers": r.best_drone_customers,
                    "truck_distance_cost": r.truck_distance_cost,
                    "drone_distance_cost": r.drone_distance_cost,
                    "truck_routes": r.truck_routes,
                    "drone_tasks": r.drone_tasks,
                    "error": r.error,
                }
            )


def _write_replay_records_csv(path: Path, instance_name: str, records) -> None:
    fields = [
        "instance",
        "distribution",
        "gamma",
        "scenario_id",
        "cost",
        "unserved",
        "no_takeoff",
        "abort_return",
        "served_customers",
        "total_customers",
        "all_served",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "instance": instance_name,
                    "distribution": r.distribution,
                    "gamma": r.gamma,
                    "scenario_id": r.scenario_id,
                    "cost": r.cost,
                    "unserved": r.unserved,
                    "no_takeoff": r.no_takeoff,
                    "abort_return": r.abort_return,
                    "served_customers": r.served_customers,
                    "total_customers": r.total_customers,
                    "all_served": int(r.all_served),
                }
            )


def _write_replay_summary_csv(path: Path, instance_name: str, summaries) -> None:
    fields = [
        "instance",
        "distribution",
        "gamma",
        "scenario_count",
        "avg_cost",
        "std_cost",
        "max_cost",
        "min_cost",
        "avg_unserved",
        "p0_all_served",
        "avg_no_takeoff",
        "avg_abort_return",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in summaries:
            writer.writerow(
                {
                    "instance": instance_name,
                    "distribution": s.distribution,
                    "gamma": s.gamma,
                    "scenario_count": s.scenario_count,
                    "avg_cost": s.avg_cost,
                    "std_cost": s.std_cost,
                    "max_cost": s.max_cost,
                    "min_cost": s.min_cost,
                    "avg_unserved": s.avg_unserved,
                    "p0_all_served": s.p0_all_served,
                    "avg_no_takeoff": s.avg_no_takeoff,
                    "avg_abort_return": s.avg_abort_return,
                }
            )


def main() -> None:
    args = parse_args()
    cfg_obj = ALNSConfig()

    gamma_levels = _parse_int_list(args.gamma_values, "gamma")
    if args.seeds:
        seeds = _parse_int_list(args.seeds, "seed")
    else:
        if args.trials <= 0:
            raise ValueError("--trials must be positive.")
        seeds = [args.seed_base + i for i in range(args.trials)]
    dist_names = _parse_dist_list(args.distributions)
    dist_configs = _build_distribution_configs(
        dist_names,
        nd_cv=args.nd_cv,
        ud_delta=args.ud_delta,
        ndc_cv=args.ndc_cv,
        ndc_correlation=args.ndc_correlation,
    )
    repair_weights = _parse_float_triplet(args.repair_weights, "repair_weights")

    instance_dirs = args.instance_dirs or [str(p) for p in DEFAULT_INSTANCE_DIRS]
    instance_paths = collect_instance_paths_with_scope(
        instance_dirs,
        scope=args.instance_scope,
        regions_text=args.regions,
        instance_name=args.instance_name,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("Gamma multi-run + scenario replay")
    print("=" * 96)
    print(f"instance dirs: {instance_dirs}")
    print(f"instance scope: {args.instance_scope}")
    if args.instance_scope == "region":
        print(f"regions: {args.regions}")
    if args.instance_scope == "single":
        print(f"instance name: {args.instance_name}")
    print(f"instances: {len(instance_paths)}")
    print(f"gamma levels: {gamma_levels}")
    print(f"seeds: {seeds}")
    print(f"distributions: {dist_names}")
    print(
        "distribution params: "
        f"nd_cv={args.nd_cv}, ud_delta={args.ud_delta}, "
        f"ndc_cv={args.ndc_cv}, ndc_correlation={args.ndc_correlation}"
    )
    print(f"scenario_count: {args.scenario_count}")
    print(f"output_dir: {output_dir}")

    all_trial_rows: list[TrialResult] = []
    all_best_rows: list[TrialResult] = []
    all_replay_summary_rows: list[dict[str, Any]] = []

    for instance_idx, instance_path in enumerate(instance_paths, 1):
        instance_name = Path(instance_path).stem
        print(f"\n[{instance_idx}/{len(instance_paths)}] {instance_name}")
        per_instance_trials: list[TrialResult] = []
        per_instance_best: list[TrialResult] = []

        for gamma in gamma_levels:
            gamma_trials: list[TrialResult] = []
            for seed in seeds:
                print(f"  gamma={gamma} seed={seed} ...", end=" ", flush=True)
                trial = _run_one_trial(
                    instance_path=instance_path,
                    gamma=gamma,
                    seed=seed,
                    cfg_obj=cfg_obj,
                    iterations_override=args.iterations,
                    time_limit_override=args.time_limit,
                    energy_deviation_rate_override=args.energy_deviation_rate,
                    repair_weights=repair_weights,
                )
                gamma_trials.append(trial)
                per_instance_trials.append(trial)
                all_trial_rows.append(trial)
                if trial.feasible and math.isfinite(trial.best_cost):
                    print(f"cost={trial.best_cost:.4f}")
                else:
                    print(f"failed ({trial.error or 'infeasible'})")

            best = _pick_best_trial(gamma_trials)
            per_instance_best.append(best)
            all_best_rows.append(best)
            print(
                f"  -> best gamma={gamma}: seed={best.seed}, "
                f"cost={best.best_cost:.4f}, feasible={best.feasible}"
            )

        if args.skip_replay:
            print("  replay skipped by --skip-replay.")
        else:
            # Run scenario replay on best-per-gamma fixed solutions for this instance.
            replay_inputs: list[GammaSolutionInput] = []
            for best in per_instance_best:
                if best.solution is None or not math.isfinite(best.best_cost):
                    continue
                replay_inputs.append(
                    GammaSolutionInput(
                        gamma=best.gamma,
                        solution=best.solution,
                        base_cost=best.best_cost,
                    )
                )

            if replay_inputs:
                replay_instance = read_instance(instance_path, strategy="class_based")
                replay_instance.configure_robustness(
                    drone_battery_capacity=cfg_obj.drone_battery_capacity,
                    energy_uncertainty_budget=0,
                    energy_deviation_rate=cfg_obj.energy_deviation_rate,

                    same_truck_retrieval=cfg_obj.same_truck_retrieval,
                )
                replay_result = run_scenario_replay(
                    instance=replay_instance,
                    gamma_solutions=replay_inputs,
                    distributions=dist_configs,
                    config=ScenarioReplayConfig(
                        scenario_count=args.scenario_count,
                        seed=args.replay_seed,
                        include_base_cost=True,
                    ),
                )
                rec_path = output_dir / f"{instance_name}_replay_records.csv"
                sum_path = output_dir / f"{instance_name}_replay_summary.csv"
                _write_replay_records_csv(rec_path, instance_name, replay_result.records)
                _write_replay_summary_csv(sum_path, instance_name, replay_result.summaries)
                print(f"  replay records saved: {rec_path}")
                print(f"  replay summary saved: {sum_path}")

                for s in replay_result.summaries:
                    all_replay_summary_rows.append(
                        {
                            "instance": instance_name,
                            "distribution": s.distribution,
                            "gamma": s.gamma,
                            "scenario_count": s.scenario_count,
                            "avg_cost": s.avg_cost,
                            "std_cost": s.std_cost,
                            "max_cost": s.max_cost,
                            "min_cost": s.min_cost,
                            "avg_unserved": s.avg_unserved,
                            "p0_all_served": s.p0_all_served,
                            "avg_no_takeoff": s.avg_no_takeoff,
                            "avg_abort_return": s.avg_abort_return,
                        }
                    )
            else:
                print("  replay skipped: no valid best solutions.")

        # Per-instance outputs.
        _write_trial_csv(output_dir / f"{instance_name}_trials.csv", per_instance_trials)
        _write_best_csv(output_dir / f"{instance_name}_best.csv", per_instance_best)

    # Global outputs.
    _write_trial_csv(output_dir / f"{args.output_prefix}_trials_all.csv", all_trial_rows)
    _write_best_csv(output_dir / f"{args.output_prefix}_best_all.csv", all_best_rows)
    if all_replay_summary_rows:
        replay_summary_path = output_dir / f"{args.output_prefix}_replay_summary_all.csv"
        with replay_summary_path.open("w", newline="", encoding="utf-8") as f:
            fields = [
                "instance",
                "distribution",
                "gamma",
                "scenario_count",
                "avg_cost",
                "std_cost",
                "max_cost",
                "min_cost",
                "avg_unserved",
                "p0_all_served",
                "avg_no_takeoff",
                "avg_abort_return",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_replay_summary_rows)
        print(f"\nglobal replay summary saved: {replay_summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
