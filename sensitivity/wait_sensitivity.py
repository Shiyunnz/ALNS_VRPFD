#!/usr/bin/env python3
"""Sensitivity analysis for truck-drone rendezvous waiting tolerance."""

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


_default_config = ALNSConfig()

WAIT_TOLERANCE_LEVELS = [0.0, 5.0, 10.0, 20.0, 40.0, 80.0, 100.0]
BASELINE_WAIT_TOLERANCE = 20.0

DEFAULT_INSTANCE_DIRS = [Path("data/Instance25")]

ITERATIONS = 2000
TIME_LIMIT = _default_config.time_limit
SEED = _default_config.seed
DRONE_PRIORITY = _default_config.drone_priority
REPAIR_SET = "new"

OUTPUT_DIR = Path(__file__).parent / "results_new" / "wait_sensitivity"
OUTPUT_CSV = OUTPUT_DIR / "wait_sensitivity_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Truck-drone waiting tolerance sensitivity analysis")
    parser.add_argument("--instance-dir", action="append", dest="instance_dirs")
    parser.add_argument("--instance-scope", choices=["all", "region", "single"], default="all")
    parser.add_argument("--regions", default="30,40,50")
    parser.add_argument("--instance-name", default=None)
    parser.add_argument("--wait-levels", default=None, help="Comma-separated waiting tolerance levels.")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--append", action="store_true")
    return parser.parse_args()


def _infer_size(instance) -> str:
    num_customers = len(instance.customer_manager.customer_ids())
    if num_customers <= 15:
        return "small"
    if num_customers <= 50:
        return "medium"
    return "large"


def _safe_mean(values: Iterable[float]) -> float:
    filtered = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return sum(filtered) / len(filtered) if filtered else math.nan


def _extract_scale_label(instance_path: str) -> str:
    return Path(instance_path).parent.name or "unknown"


def _choose_best_result(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any] | None:
    valid = [
        row
        for row in rows
        if isinstance(row.get("best_cost"), (int, float)) and math.isfinite(float(row["best_cost"]))
    ]
    if not valid:
        return None
    return min(
        valid,
        key=lambda row: (
            float(row.get("best_cost", math.inf)),
            -float(row.get("best_drone_customers", 0.0) or 0.0),
        ),
    ).copy()


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


def _rendezvous_stats(details) -> tuple[float, float, int]:
    deviations = [
        float(result.deviation)
        for result in details.rendezvous_results.values()
        if result is not None and math.isfinite(float(result.deviation))
    ]
    if not deviations:
        return 0.0, 0.0, 0
    return max(deviations), sum(deviations) / len(deviations), len(deviations)


def run_single_experiment(
    instance_path: str,
    wait_tolerance: float,
    *,
    same_truck_retrieval: bool = False,
    seed: int | None = None,
) -> Dict[str, Any]:
    """Run one ALNS trial with a synchronized waiting tolerance."""

    wait_tolerance = float(wait_tolerance)
    print(f"  Running: wait_tolerance={wait_tolerance:.2f}, same_truck={same_truck_retrieval}")

    instance = read_instance(instance_path, strategy="class_based")

    if "drone" in instance.vehicle_specs:
        instance.vehicle_specs["drone"].endurance = float("inf")

    instance.configure_robustness(
        drone_battery_capacity=_default_config.drone_battery_capacity,
        energy_uncertainty_budget=_default_config.energy_uncertainty_budget,
        energy_deviation_rate=_default_config.energy_deviation_rate,
        same_truck_retrieval=same_truck_retrieval,
    )

    evaluator = Evaluator(
        instance,
        rendezvous_tolerance=wait_tolerance,
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
                "wait_tolerance": wait_tolerance,
                "repair_wait_max": wait_tolerance,
                "rendezvous_tolerance": wait_tolerance,
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
                "max_rendezvous_deviation": math.nan,
                "avg_rendezvous_deviation": math.nan,
                "rendezvous_count": 0,
                "error": f"initial_failed: {fallback_exc}",
            }

    local_seed = seed if seed is not None else (SEED if SEED is not None else int(time.time()))

    drone_bonus_kwargs = dict(_default_config.drone_bonus)
    drone_bonus_kwargs["wait_max"] = wait_tolerance

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
        best_details = evaluator.evaluate_with_details(best_solution)
        best_eval = best_details.result
        best_drone_customers = count_drone_served_customers(best_solution)
        max_wait, avg_wait, rendezvous_count = _rendezvous_stats(best_details)
    except Exception as exc:
        print(f"    ! Best solution evaluation failed: {exc}.")
        return {
            "instance": instance_path,
            "wait_tolerance": wait_tolerance,
            "repair_wait_max": wait_tolerance,
            "rendezvous_tolerance": wait_tolerance,
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
            "max_rendezvous_deviation": math.nan,
            "avg_rendezvous_deviation": math.nan,
            "rendezvous_count": 0,
            "error": f"best_eval_failed: {exc}",
        }

    cost_reduction = math.nan
    if math.isfinite(initial_cost) and math.isfinite(best_eval.total_cost) and initial_cost > 0:
        cost_reduction = ((initial_cost - best_eval.total_cost) / initial_cost) * 100

    return {
        "instance": instance_path,
        "wait_tolerance": wait_tolerance,
        "repair_wait_max": wait_tolerance,
        "rendezvous_tolerance": wait_tolerance,
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
        "max_rendezvous_deviation": max_wait,
        "avg_rendezvous_deviation": avg_wait,
        "rendezvous_count": rendezvous_count,
        "error": "",
    }


def write_summary_csv(results: Iterable[Dict[str, Any]], out_path: Path) -> None:
    """Write compact best-of-k summary grouped by scale and waiting tolerance."""

    by_instance_level: dict[tuple[str, float], list[Dict[str, Any]]] = defaultdict(list)
    for row in results:
        inst = row.get("instance")
        wait = row.get("wait_tolerance")
        if isinstance(inst, str) and isinstance(wait, (int, float)):
            by_instance_level[(inst, float(wait))].append(row)

    best_rows: list[Dict[str, Any]] = []
    for rows in by_instance_level.values():
        best = _choose_best_result(rows)
        if best is not None:
            best_rows.append(best)

    baseline_by_instance: dict[str, float] = {}
    rows_by_instance: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in best_rows:
        inst = row.get("instance")
        if isinstance(inst, str):
            rows_by_instance[inst].append(row)
        wait = row.get("wait_tolerance")
        cost = row.get("best_cost")
        if (
            isinstance(inst, str)
            and isinstance(wait, (int, float))
            and isinstance(cost, (int, float))
            and math.isfinite(float(cost))
            and math.isclose(float(wait), BASELINE_WAIT_TOLERANCE)
        ):
            baseline_by_instance[inst] = float(cost)

    for inst, rows in rows_by_instance.items():
        if inst in baseline_by_instance:
            continue
        finite_rows = [
            row
            for row in rows
            if isinstance(row.get("wait_tolerance"), (int, float))
            and isinstance(row.get("best_cost"), (int, float))
            and math.isfinite(float(row["best_cost"]))
        ]
        if finite_rows:
            finite_rows.sort(key=lambda row: abs(float(row["wait_tolerance"]) - BASELINE_WAIT_TOLERANCE))
            baseline_by_instance[inst] = float(finite_rows[0]["best_cost"])

    for row in best_rows:
        inst = row.get("instance")
        cost = row.get("best_cost")
        base_cost = baseline_by_instance.get(inst) if isinstance(inst, str) else None
        if (
            isinstance(base_cost, (int, float))
            and isinstance(cost, (int, float))
            and math.isfinite(float(base_cost))
            and math.isfinite(float(cost))
            and float(base_cost) > 0
        ):
            row["baseline_best_cost"] = float(base_cost)
            row["cost_saving_vs_baseline"] = ((float(base_cost) - float(cost)) / float(base_cost)) * 100.0
        else:
            row["baseline_best_cost"] = math.nan
            row["cost_saving_vs_baseline"] = math.nan

    grouped: dict[tuple[str, float], list[Dict[str, Any]]] = defaultdict(list)
    for row in best_rows:
        inst = row.get("instance")
        wait = row.get("wait_tolerance")
        if isinstance(inst, str) and isinstance(wait, (int, float)):
            grouped[(_extract_scale_label(inst), float(wait))].append(row)

    fieldnames = [
        "scale",
        "wait_tolerance",
        "avg_cost_saving_vs_baseline",
        "avg_best_cost",
        "avg_best_drone_customers",
        "avg_max_rendezvous_deviation",
        "avg_rendezvous_deviation",
        "avg_rendezvous_count",
        "n_instances",
    ]

    rows_out = []
    for (scale, wait), items in sorted(grouped.items()):
        rows_out.append(
            {
                "scale": scale,
                "wait_tolerance": wait,
                "avg_cost_saving_vs_baseline": _safe_mean(
                    row.get("cost_saving_vs_baseline") for row in items
                ),
                "avg_best_cost": _safe_mean(row.get("best_cost") for row in items),
                "avg_best_drone_customers": _safe_mean(row.get("best_drone_customers") for row in items),
                "avg_max_rendezvous_deviation": _safe_mean(
                    row.get("max_rendezvous_deviation") for row in items
                ),
                "avg_rendezvous_deviation": _safe_mean(
                    row.get("avg_rendezvous_deviation") for row in items
                ),
                "avg_rendezvous_count": _safe_mean(row.get("rendezvous_count") for row in items),
                "n_instances": len(items),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"✅ 汇总结果已更新: {out_path}")


def main() -> None:
    args = parse_args()
    instance_dirs = args.instance_dirs or DEFAULT_INSTANCE_DIRS
    instances = collect_instance_paths_with_scope(
        instance_dirs,
        scope=args.instance_scope,
        regions_text=args.regions,
        instance_name=args.instance_name,
    )
    levels = (
        [float(item.strip()) for item in args.wait_levels.split(",") if item.strip()]
        if args.wait_levels
        else WAIT_TOLERANCE_LEVELS
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not args.append or not OUTPUT_CSV.exists()
    mode = "a" if args.append else "w"
    fieldnames = [
        "instance",
        "trial",
        "wait_tolerance",
        "repair_wait_max",
        "rendezvous_tolerance",
        "initial_cost",
        "best_cost",
        "best_drone_customers",
        "feasible",
        "truck_distance_cost",
        "drone_distance_cost",
        "max_rendezvous_deviation",
        "avg_rendezvous_deviation",
        "rendezvous_count",
        "run_time",
        "error",
    ]
    rows: list[Dict[str, Any]] = []
    with OUTPUT_CSV.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for instance in instances:
            for wait in levels:
                for trial in range(args.trials):
                    seed = (SEED if SEED is not None else int(time.time())) + trial
                    row = run_single_experiment(instance, wait, seed=seed)
                    row["trial"] = trial
                    writer.writerow(row)
                    handle.flush()
                    rows.append(row)

    write_summary_csv(rows, OUTPUT_DIR / "wait_sensitivity_summary.csv")


if __name__ == "__main__":
    main()
