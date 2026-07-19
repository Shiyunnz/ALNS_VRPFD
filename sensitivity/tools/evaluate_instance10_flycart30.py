#!/usr/bin/env python3
"""Evaluate Instance10 with FlyCart30 battery settings.

This script reuses sensitivity.docking_flexibility_comparison and only changes
energy-related defaults in-process for this run:
1) robust battery capacity -> FlyCart30 capacity
2) DroneEnergyModel default battery weight/capacity -> FlyCart30 values
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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

import sensitivity.docking_flexibility_comparison as dock
from alns_vrpfd.evaluation.energy import DroneEnergyModel as _DroneEnergyModel


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _safe_mean(values: Iterable[float]) -> float:
    vals = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not vals:
        return math.nan
    return sum(vals) / len(vals)


def _parse_customer_count(instance_path: str) -> int:
    stem = Path(instance_path).stem
    match = re.match(r"R_\d+_(\d+)_\d+$", stem)
    if match:
        return int(match.group(1))
    return 10


def _parse_task_list(payload: str) -> List[Dict[str, Any]]:
    if not payload:
        return []
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _count_cross_truck_tasks(tasks: List[Dict[str, Any]]) -> int:
    count = 0
    for task in tasks:
        launch = task.get("launch_truck")
        land = task.get("land_truck")
        if land is None:
            continue
        if launch != land:
            count += 1
    return count


def _select_mode_best(rows: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    valid = [
        row for row in rows
        if math.isfinite(_safe_float(row.get("best_cost")))
    ]
    if not valid:
        return None
    return min(
        valid,
        key=lambda row: (
            _safe_float(row.get("best_cost")),
            -_safe_float(row.get("best_drone_customers", 0.0)),
        ),
    )


def build_instance_metrics(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in results:
        instance = row.get("instance")
        mode = row.get("mode")
        if isinstance(instance, str) and isinstance(mode, str):
            grouped[(instance, mode)].append(row)

    by_instance: Dict[str, Dict[str, Any]] = {}
    for (instance, mode), rows in grouped.items():
        best = _select_mode_best(rows)
        if best is None:
            continue
        by_instance.setdefault(instance, {})[mode] = best

    out: List[Dict[str, Any]] = []
    for instance in sorted(by_instance.keys()):
        pack = by_instance[instance]
        same = pack.get("same_truck")
        flex = pack.get("flexible")
        if same is None or flex is None:
            continue

        n_customers = _parse_customer_count(instance)
        same_cost = _safe_float(same.get("best_cost"))
        flex_cost = _safe_float(flex.get("best_cost"))
        saving_abs = same_cost - flex_cost if math.isfinite(same_cost) and math.isfinite(flex_cost) else math.nan
        saving_pct = (saving_abs / same_cost * 100.0) if math.isfinite(saving_abs) and same_cost > 0 else math.nan

        same_drone_customers = _safe_float(same.get("best_drone_customers"))
        flex_drone_customers = _safe_float(flex.get("best_drone_customers"))
        same_util = (same_drone_customers / n_customers) if n_customers > 0 and math.isfinite(same_drone_customers) else math.nan
        flex_util = (flex_drone_customers / n_customers) if n_customers > 0 and math.isfinite(flex_drone_customers) else math.nan

        same_tasks = _parse_task_list(str(same.get("drone_tasks", "")))
        flex_tasks = _parse_task_list(str(flex.get("drone_tasks", "")))
        flex_cross_count = _count_cross_truck_tasks(flex_tasks)
        flex_cross_ratio = (flex_cross_count / len(flex_tasks)) if flex_tasks else 0.0

        out.append({
            "instance": instance,
            "instance_name": Path(instance).stem,
            "region": dock._extract_region_id(instance),
            "num_customers": n_customers,
            "same_best_seed": same.get("seed"),
            "flex_best_seed": flex.get("seed"),
            "same_cost": same_cost,
            "flex_cost": flex_cost,
            "saving_abs": saving_abs,
            "saving_pct": saving_pct,
            "same_drone_customers": same_drone_customers,
            "flex_drone_customers": flex_drone_customers,
            "same_uav_customer_utilization": same_util,
            "flex_uav_customer_utilization": flex_util,
            "delta_uav_customer_utilization": flex_util - same_util if math.isfinite(same_util) and math.isfinite(flex_util) else math.nan,
            "same_drone_tasks": len(same_tasks),
            "flex_drone_tasks": len(flex_tasks),
            "flex_cross_truck_tasks": flex_cross_count,
            "flex_cross_truck_ratio": flex_cross_ratio,
        })
    return out


def build_overall_summary(instance_rows: List[Dict[str, Any]], tie_eps_pct: float = 0.1) -> List[Dict[str, Any]]:
    if not instance_rows:
        return []

    savings = [_safe_float(r.get("saving_pct")) for r in instance_rows]
    same_util = [_safe_float(r.get("same_uav_customer_utilization")) for r in instance_rows]
    flex_util = [_safe_float(r.get("flex_uav_customer_utilization")) for r in instance_rows]
    delta_util = [_safe_float(r.get("delta_uav_customer_utilization")) for r in instance_rows]
    cross_ratio = [_safe_float(r.get("flex_cross_truck_ratio")) for r in instance_rows]

    win = sum(1 for v in savings if math.isfinite(v) and v > tie_eps_pct)
    tie = sum(1 for v in savings if math.isfinite(v) and abs(v) <= tie_eps_pct)
    loss = sum(1 for v in savings if math.isfinite(v) and v < -tie_eps_pct)
    with_cross = [
        _safe_float(r.get("saving_pct"))
        for r in instance_rows
        if _safe_float(r.get("flex_cross_truck_tasks")) > 0
    ]
    without_cross = [
        _safe_float(r.get("saving_pct"))
        for r in instance_rows
        if _safe_float(r.get("flex_cross_truck_tasks")) <= 0
    ]

    return [{
        "group": "ALL",
        "num_instances": len(instance_rows),
        "avg_saving_pct": _safe_mean(savings),
        "win_instances": win,
        "tie_instances": tie,
        "loss_instances": loss,
        "avg_same_uav_customer_utilization": _safe_mean(same_util),
        "avg_flex_uav_customer_utilization": _safe_mean(flex_util),
        "avg_delta_uav_customer_utilization": _safe_mean(delta_util),
        "instances_with_cross_truck_tasks": sum(1 for r in instance_rows if _safe_float(r.get("flex_cross_truck_tasks")) > 0),
        "avg_flex_cross_truck_ratio": _safe_mean(cross_ratio),
        "avg_saving_pct_when_cross_truck_used": _safe_mean(with_cross),
        "avg_saving_pct_when_no_cross_truck": _safe_mean(without_cross),
    }]


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@contextmanager
def patch_flycart_energy_defaults(*, battery_weight_kg: float, battery_capacity_kwh: float):
    original_init = _DroneEnergyModel.__init__

    def patched_init(self, *args, **kwargs):
        if not args:
            kwargs.setdefault("battery_weight_kg", battery_weight_kg)
            kwargs.setdefault("battery_capacity_kwh", battery_capacity_kwh)
        return original_init(self, *args, **kwargs)

    _DroneEnergyModel.__init__ = patched_init
    try:
        yield
    finally:
        _DroneEnergyModel.__init__ = original_init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Instance10 using FlyCart30 battery settings.")
    parser.add_argument("--instance-dir", type=str, default="data/Instance10")
    parser.add_argument("--output-dir", type=str, default="sensitivity/results_new/drone_flexibility_flycart30")
    parser.add_argument("--output-prefix", type=str, default="docking_flexibility_i10_flycart30")
    parser.add_argument("--battery-weight-kg", type=float, default=22.6)
    parser.add_argument("--battery-capacity-kwh", type=float, default=3.9688)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--seed-start", type=int, default=20260329)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_prefix = args.output_prefix
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reuse existing script runtime config.
    dock.ITERATIONS = int(args.iterations)
    dock._default_config.raw.setdefault("robustness", {})["drone_battery_capacity"] = float(args.battery_capacity_kwh)

    instance_paths = dock.collect_instance_paths([Path(args.instance_dir)])
    seeds = dock.parse_seed_values(args.seeds, trials=args.trials, seed_start=args.seed_start)

    print(f"Running {len(instance_paths)} instances with seeds={seeds}")
    print(f"FlyCart30 params: battery_weight={args.battery_weight_kg} kg, battery_capacity={args.battery_capacity_kwh} kWh")

    with patch_flycart_energy_defaults(
        battery_weight_kg=float(args.battery_weight_kg),
        battery_capacity_kwh=float(args.battery_capacity_kwh),
    ):
        results = dock.run_docking_comparison(
            instance_paths=instance_paths,
            seeds=seeds,
            skip_baseline=False,
            baseline_csv=None,
        )

    results_csv = output_dir / f"{output_prefix}_results.csv"
    paired_csv = output_dir / f"{output_prefix}_paired_trials.csv"
    instance_csv = output_dir / f"{output_prefix}_instance_metrics.csv"
    summary_csv = output_dir / f"{output_prefix}_summary.csv"

    dock.write_results(results, append=False, output_csv=results_csv)
    dock.write_paired_trial_csv(dock.build_paired_trial_rows(results), paired_csv)

    instance_metrics = build_instance_metrics(results)
    summary = build_overall_summary(instance_metrics)
    write_csv(instance_csv, instance_metrics)
    write_csv(summary_csv, summary)

    print(f"Saved: {results_csv}")
    print(f"Saved: {paired_csv}")
    print(f"Saved: {instance_csv}")
    print(f"Saved: {summary_csv}")

    if summary:
        row = summary[0]
        print(
            "Summary(ALL): "
            f"avg_saving_pct={row['avg_saving_pct']:.3f}, "
            f"avg_same_util={row['avg_same_uav_customer_utilization']:.3f}, "
            f"avg_flex_util={row['avg_flex_uav_customer_utilization']:.3f}, "
            f"cross_task_instances={int(row['instances_with_cross_truck_tasks'])}"
        )


if __name__ == "__main__":
    main()
