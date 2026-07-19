#!/usr/bin/env python3
"""Incremental rerunner for sensitivity experiments after deadline/cost changes."""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import math
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sensitivity.instance_selector import collect_instance_paths_with_scope


META_FIELDNAMES = [
    "task_key",
    "analysis",
    "level_name",
    "level_value",
    "trial",
    "seed",
    "instance_name",
]

RESULT_FIELDNAMES = [
    "instance",
    "strategy",
    "mode",
    "mode_name",
    "same_truck_retrieval",
    "battery_capacity",
    "drone_speed",
    "drone_count",
    "payload_capacity",
    "gamma",
    "theta",
    "scale",
    "min_window_width",
    "max_window_width",
    "wait_tolerance",
    "repair_wait_max",
    "rendezvous_tolerance",
    "max_rendezvous_deviation",
    "avg_rendezvous_deviation",
    "rendezvous_count",
    "baseline_best_cost",
    "cost_saving_vs_baseline",
    "cost_increase_vs_baseline",
    "initial_cost",
    "best_cost",
    "cost_reduction_percent",
    "initial_drone_customers",
    "best_drone_customers",
    "drone_customer_change",
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
    "error",
]

TRIAL_FIELDNAMES = META_FIELDNAMES + RESULT_FIELDNAMES
_META_FIELDS = set(META_FIELDNAMES)


@dataclass(frozen=True)
class TrialTask:
    analysis: str
    instance: str
    level_name: str
    level_value: Any
    trial: int
    seed: int

    def key(self) -> str:
        return "|".join(
            [
                self.analysis,
                self.instance,
                self.level_name,
                _format_value(self.level_value),
                str(self.trial),
                str(self.seed),
            ]
        )


@dataclass(frozen=True)
class AnalysisSpec:
    name: str
    module_name: str
    level_name: str
    levels_attr: str
    baseline_attr: str | None
    summary_filename: str
    metric: str
    cast_level: Callable[[str], Any]


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _parse_scalar(value: str) -> Any:
    if value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        number = float(value)
    except ValueError:
        return value
    if math.isfinite(number) and number.is_integer():
        return int(number)
    return number


def append_trial_row(path: Path, task: TrialTask, result: dict[str, Any]) -> None:
    """Append one completed trial to CSV immediately."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    row = {
        "task_key": task.key(),
        "analysis": task.analysis,
        "level_name": task.level_name,
        "level_value": _format_value(task.level_value),
        "trial": task.trial,
        "seed": task.seed,
        "instance_name": Path(task.instance).stem,
    }
    row.update(result)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({field: _csv_value(row.get(field, "")) for field in TRIAL_FIELDNAMES})
        handle.flush()


def completed_task_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row["task_key"] for row in reader if row.get("task_key")}


def pending_tasks(tasks: Sequence[TrialTask], completed: set[str], *, force: bool) -> list[TrialTask]:
    if force:
        return list(tasks)
    return [task for task in tasks if task.key() not in completed]


def load_trial_rows(
    path: Path,
    *,
    analysis: str | None = None,
    include_task_metadata: bool = False,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            if analysis is not None and raw.get("analysis") != analysis:
                continue
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if key in _META_FIELDS and not include_task_metadata:
                    continue
                if value in (None, ""):
                    continue
                parsed = _parse_scalar(value)
                if parsed is not None:
                    row[key] = parsed
            rows.append(row)
    return rows


def _analysis_specs() -> dict[str, AnalysisSpec]:
    return {
        "flexibility": AnalysisSpec(
            name="flexibility",
            module_name="sensitivity.docking_flexibility_comparison",
            level_name="mode",
            levels_attr="DOCKING_MODES",
            baseline_attr=None,
            summary_filename="docking_flexibility_summary.csv",
            metric="flex_saving",
            cast_level=str,
        ),
        "battery": AnalysisSpec(
            name="battery",
            module_name="sensitivity.battery_sensitivity",
            level_name="battery_capacity",
            levels_attr="BATTERY_LEVELS",
            baseline_attr="BASELINE_BATTERY",
            summary_filename="battery_sensitivity_summary.csv",
            metric="saving",
            cast_level=float,
        ),
        "speed": AnalysisSpec(
            name="speed",
            module_name="sensitivity.drone_speed_sensitivity",
            level_name="drone_speed",
            levels_attr="DRONE_SPEED_LEVELS",
            baseline_attr="BASELINE_DRONE_SPEED",
            summary_filename="drone_speed_summary.csv",
            metric="saving",
            cast_level=float,
        ),
        "count": AnalysisSpec(
            name="count",
            module_name="sensitivity.drone_count_sensitivity",
            level_name="drone_count",
            levels_attr="DEFAULT_DRONE_COUNT_LEVELS",
            baseline_attr=None,
            summary_filename="drone_count_summary.csv",
            metric="saving",
            cast_level=int,
        ),
        "payload": AnalysisSpec(
            name="payload",
            module_name="sensitivity.drone_payload_sensitivity",
            level_name="payload_capacity",
            levels_attr="DRONE_PAYLOAD_LEVELS",
            baseline_attr="BASELINE_PAYLOAD",
            summary_filename="drone_payload_summary.csv",
            metric="saving",
            cast_level=float,
        ),
        "gamma": AnalysisSpec(
            name="gamma",
            module_name="sensitivity.gamma_sensitivity",
            level_name="gamma",
            levels_attr="GAMMA_LEVELS",
            baseline_attr="BASELINE_GAMMA",
            summary_filename="gamma_summary.csv",
            metric="increase",
            cast_level=int,
        ),
        "theta": AnalysisSpec(
            name="theta",
            module_name="sensitivity.theta_sensitivity",
            level_name="theta",
            levels_attr="THETA_LEVELS",
            baseline_attr="BASELINE_THETA",
            summary_filename="theta_summary.csv",
            metric="increase",
            cast_level=float,
        ),
        "timewindow": AnalysisSpec(
            name="timewindow",
            module_name="sensitivity.time_window_sensitivity",
            level_name="scale",
            levels_attr="SCALE_LEVELS",
            baseline_attr="BASELINE_SCALE",
            summary_filename="time_window_summary.csv",
            metric="increase",
            cast_level=float,
        ),
        "wait": AnalysisSpec(
            name="wait",
            module_name="sensitivity.wait_sensitivity",
            level_name="wait_tolerance",
            levels_attr="WAIT_TOLERANCE_LEVELS",
            baseline_attr="BASELINE_WAIT_TOLERANCE",
            summary_filename="wait_sensitivity_summary.csv",
            metric="saving",
            cast_level=float,
        ),
    }


def _levels_for(spec: AnalysisSpec, module: Any, instance_path: str, override: list[Any] | None) -> list[Any]:
    if override is not None:
        levels = list(override)
    elif spec.name == "flexibility":
        levels = list(getattr(module, spec.levels_attr).keys())
    elif spec.name == "count":
        scale = module._extract_scale_label(instance_path)
        levels = list(getattr(module, "DRONE_COUNT_LEVELS_BY_SCALE", {}).get(scale, getattr(module, spec.levels_attr)))
    else:
        levels = list(getattr(module, spec.levels_attr))

    baseline = _baseline_for(spec, module, instance_path)
    if baseline in levels:
        return [baseline] + [level for level in levels if level != baseline]
    return levels


def _baseline_for(spec: AnalysisSpec, module: Any, instance_path: str) -> Any:
    if spec.name == "flexibility":
        return "same_truck"
    if spec.name == "count":
        scale = module._extract_scale_label(instance_path)
        return getattr(module, "BASELINE_BY_SCALE", {}).get(scale, getattr(module, "DEFAULT_BASELINE"))
    if spec.baseline_attr is None:
        return None
    return getattr(module, spec.baseline_attr)


def _parse_levels(spec: AnalysisSpec, text: str | None) -> list[Any] | None:
    if text is None:
        return None
    return [spec.cast_level(item.strip()) for item in text.split(",") if item.strip()]


def _collect_instances(module: Any, args: argparse.Namespace) -> list[str]:
    instance_dirs = args.instance_dirs or getattr(module, "DEFAULT_INSTANCE_DIRS")
    return collect_instance_paths_with_scope(
        instance_dirs,
        scope=args.instance_scope,
        regions_text=args.regions,
        instance_name=args.instance_name,
    )


def build_tasks(spec: AnalysisSpec, module: Any, args: argparse.Namespace) -> list[TrialTask]:
    level_override = _parse_levels(spec, args.levels)
    instances = _collect_instances(module, args)
    tasks: list[TrialTask] = []
    seed_start = args.seed
    if seed_start is None:
        seed_start = int(getattr(module, "DEFAULT_SEED_START", getattr(module, "SEED", 42) or 42))

    for instance in instances:
        for level in _levels_for(spec, module, instance, level_override):
            for trial in range(args.trials):
                tasks.append(
                    TrialTask(
                        analysis=spec.name,
                        instance=str(instance),
                        level_name=spec.level_name,
                        level_value=level,
                        trial=trial,
                        seed=int(seed_start) + trial,
                    )
                )
    return tasks


def _is_baseline(spec: AnalysisSpec, module: Any, row: dict[str, Any]) -> bool:
    instance = row.get("instance")
    if not isinstance(instance, str):
        return False
    baseline = _baseline_for(spec, module, instance)
    value = row.get(spec.level_name)
    if isinstance(value, float) or isinstance(baseline, float):
        try:
            return math.isclose(float(value), float(baseline), rel_tol=0.0, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return value == baseline


def _choose_best(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
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
    )


def enrich_baseline_metrics(
    spec: AnalysisSpec,
    module: Any,
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched = [dict(row) for row in rows]
    baselines: dict[str, float] = {}
    by_instance: dict[str, list[dict[str, Any]]] = {}
    for row in enriched:
        instance = row.get("instance")
        if isinstance(instance, str):
            by_instance.setdefault(instance, []).append(row)

    for instance, instance_rows in by_instance.items():
        best = _choose_best(row for row in instance_rows if _is_baseline(spec, module, row))
        if best is not None:
            baselines[instance] = float(best["best_cost"])

    for row in enriched:
        instance = row.get("instance")
        cost = row.get("best_cost")
        base_cost = baselines.get(instance) if isinstance(instance, str) else None
        if _is_baseline(spec, module, row):
            row.setdefault("strategy", "Baseline")
        elif spec.name != "flexibility":
            row.setdefault("strategy", "Test")
        if isinstance(base_cost, (int, float)) and isinstance(cost, (int, float)) and base_cost > 0:
            row["baseline_best_cost"] = base_cost
            if spec.metric in {"saving", "flex_saving"}:
                row["cost_saving_vs_baseline"] = (base_cost - float(cost)) / base_cost * 100.0
            elif spec.metric == "increase":
                row["cost_increase_vs_baseline"] = (float(cost) - base_cost) / base_cost * 100.0
        elif _is_baseline(spec, module, row) and isinstance(cost, (int, float)) and math.isfinite(float(cost)):
            row["baseline_best_cost"] = float(cost)
            if spec.metric in {"saving", "flex_saving"}:
                row["cost_saving_vs_baseline"] = 0.0
            elif spec.metric == "increase":
                row["cost_increase_vs_baseline"] = 0.0
    return enriched


def run_task(spec: AnalysisSpec, module: Any, task: TrialTask, *, iterations: int | None) -> dict[str, Any]:
    if iterations is not None and hasattr(module, "ITERATIONS"):
        setattr(module, "ITERATIONS", int(iterations))

    if hasattr(module, "SEED"):
        setattr(module, "SEED", int(task.seed))

    if spec.name == "flexibility":
        return module.run_single_experiment(task.instance, str(task.level_value), int(task.seed))

    signature = inspect.signature(module.run_single_experiment)
    kwargs: dict[str, Any] = {
        "instance_path": task.instance,
        spec.level_name: task.level_value,
    }
    if "seed" in signature.parameters:
        kwargs["seed"] = int(task.seed)
    if iterations is not None and "iterations" in signature.parameters:
        kwargs["iterations"] = int(iterations)
    return module.run_single_experiment(**kwargs)


def refresh_summary(spec: AnalysisSpec, module: Any, trial_csv: Path, output_dir: Path) -> None:
    include_meta = spec.name == "flexibility"
    rows = load_trial_rows(trial_csv, analysis=spec.name, include_task_metadata=include_meta)
    rows = enrich_baseline_metrics(spec, module, rows)
    summary_path = output_dir / spec.summary_filename
    module.write_summary_csv(rows, summary_path)

    if spec.name == "flexibility":
        paired = module.build_paired_trial_rows(rows)
        module.write_paired_trial_csv(paired, output_dir / "docking_flexibility_paired_trials.csv")
        best_rows = module.build_best_instance_rows(rows)
        module.write_best_instance_csv(best_rows, output_dir / "docking_flexibility_best_by_instance.csv")


def parse_args() -> argparse.Namespace:
    specs = _analysis_specs()
    parser = argparse.ArgumentParser(description="Rerun sensitivity experiments with incremental checkpoints.")
    parser.add_argument("--only", nargs="*", choices=sorted(specs), default=None)
    parser.add_argument("--skip", nargs="*", choices=sorted(specs), default=[])
    parser.add_argument("--instance-dir", action="append", dest="instance_dirs")
    parser.add_argument("--instance-scope", choices=["all", "region", "single"], default="all")
    parser.add_argument("--regions", default="30,40,50")
    parser.add_argument("--instance-name", default=None)
    parser.add_argument("--levels", default=None, help="Comma-separated levels; valid only when one analysis is selected.")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--limit-runs", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Rerun tasks even if their task_key exists.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sensitivity/results_new") / f"deadline_cost_rerun_{date.today():%Y%m%d}",
    )
    args = parser.parse_args()
    selected = args.only or sorted(specs)
    selected = [name for name in selected if name not in set(args.skip)]
    if args.levels is not None and len(selected) != 1:
        parser.error("--levels can only be used with exactly one selected analysis.")
    args.selected = selected
    return args


def main() -> None:
    args = parse_args()
    specs = _analysis_specs()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_csv = output_dir / "trials.csv"

    print(f"Output directory: {output_dir}")
    print(f"Checkpoint CSV:   {trial_csv}")

    total_done = 0
    for analysis_name in args.selected:
        spec = specs[analysis_name]
        module = importlib.import_module(spec.module_name)
        tasks = build_tasks(spec, module, args)
        todo = pending_tasks(tasks, completed_task_keys(trial_csv), force=args.force)
        if args.limit_runs is not None:
            todo = todo[: args.limit_runs]

        print(f"\n[{analysis_name}] total={len(tasks)} pending={len(todo)}")
        for index, task in enumerate(todo, 1):
            print(
                f"  ({index}/{len(todo)}) {Path(task.instance).stem} "
                f"{task.level_name}={task.level_value} trial={task.trial} seed={task.seed}",
                flush=True,
            )
            result = run_task(spec, module, task, iterations=args.iterations)
            result["instance"] = task.instance
            result[task.level_name] = task.level_value
            result["trial"] = task.trial
            if spec.name == "flexibility":
                result["seed"] = task.seed

            existing = load_trial_rows(trial_csv, analysis=spec.name, include_task_metadata=(spec.name == "flexibility"))
            current_rows = enrich_baseline_metrics(spec, module, existing + [result])
            current_result = current_rows[-1]
            append_trial_row(trial_csv, task, current_result)
            refresh_summary(spec, module, trial_csv, output_dir)
            total_done += 1

        if not todo:
            refresh_summary(spec, module, trial_csv, output_dir)

    print(f"\nCompleted new runs: {total_done}")
    print(f"Results saved incrementally under: {output_dir}")


if __name__ == "__main__":
    main()
