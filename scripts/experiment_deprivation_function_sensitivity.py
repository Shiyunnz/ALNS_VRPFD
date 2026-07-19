#!/usr/bin/env python3
"""Sensitivity analysis for deprivation-cost form and time-scale factor.

Conditions
----------
* exponential_rho_0.15
* exponential_rho_0.208333 (paper baseline)
* exponential_rho_0.30
* linear
* quadratic

The linear and quadratic functions preserve the same class-specific endpoint
cost, lambda * omega_c, at H_tau=4.4947 h. Runs use paired instance/seed
combinations and are saved incrementally for safe resumption.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alns_vrpfd.core.operators import (  # noqa: E402
    DestroyRandom,
    DestroyShaw,
    DestroyWorstDistance,
    RepairCheapest,
    RepairDronePriorityRegret,
    RepairEqualPriority,
    RepairTruckFirst,
)
from alns_vrpfd.core.sa import SANNCfg, SimulatedAnnealingALNS  # noqa: E402
from alns_vrpfd.deprivation import (  # noqa: E402
    MAX_TARDINESS_HOURS,
    WANG_SUPPLY_CLASSES,
    deprivation_cost as exponential_deprivation_cost,
)
from revision import tune_base  # noqa: E402
from revision.validate_alns_ts_ga import (  # noqa: E402
    ALNS_CFG,
    build_shared_initial_solution,
    make_alns_bonus,
)


CONDITIONS = {
    "exponential_rho_0.15": ("exponential", 0.15),
    "exponential_rho_0.208333": ("exponential", 0.20833333333333334),
    "exponential_rho_0.30": ("exponential", 0.30),
    "linear": ("linear", None),
    "quadratic": ("quadratic", None),
}
BASELINE = "exponential_rho_0.208333"
FIELDS = [
    "task_key",
    "condition",
    "function_form",
    "rho",
    "instance",
    "seed",
    "iterations",
    "time_limit",
    "feasible",
    "total_cost",
    "truck_cost",
    "drone_cost",
    "delay_cost",
    "delayed_nodes",
    "total_delay_hours",
    "mean_delay_hours",
    "max_delay_hours",
    "drone_customers",
    "runtime_sec",
    "termination_reason",
    "class_delay_hours",
    "class_delay_cost",
    "error",
]


def normalize_class(value: str | None) -> str:
    key = str(value or "water").strip().lower()
    aliases = {
        "health": "medicine",
        "medical": "medicine",
        "med": "medicine",
        "wash": "water",
        "drinking_water": "water",
        "shelter": "tent",
    }
    key = aliases.get(key, key)
    return key if key in WANG_SUPPLY_CLASSES else "water"


def condition_cost(
    form: str,
    selected_rho: float | None,
    tau_hours: float,
    supply_class: str | None = "water",
    *,
    cost_lambda: float = 30.0,
    rho: float = 0.20833333333333334,
    normalized: bool = True,
) -> float:
    del rho, normalized
    tau = max(0.0, float(tau_hours))
    key = normalize_class(supply_class)
    omega = WANG_SUPPLY_CLASSES[key].omega
    if form == "exponential":
        return exponential_deprivation_cost(
            tau,
            key,
            cost_lambda=cost_lambda,
            rho=float(selected_rho),
            normalized=True,
        )
    ratio = tau / MAX_TARDINESS_HOURS
    if form == "linear":
        shape = ratio
    elif form == "quadratic":
        shape = ratio**2
    else:
        raise ValueError(f"Unknown deprivation form: {form}")
    return cost_lambda * omega * shape


def task_key(condition: str, instance: str, seed: int) -> str:
    return f"{condition}|{instance}|{seed}"


def parse_seeds(value: str) -> list[int]:
    if "-" in value and "," not in value:
        start, end = value.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(part) for part in value.split(",") if part.strip()]


def all_instances() -> list[str]:
    return [
        f"R_{region}_25_{idx}"
        for region in (30, 40, 50)
        for idx in range(1, 6)
    ]


def parse_instances(value: str | None) -> list[str]:
    if not value:
        return all_instances()
    return [item.strip() for item in value.split(",") if item.strip()]


def run_task(payload: tuple[str, str, int, int, float]) -> dict[str, Any]:
    condition, instance_name, seed, iterations, time_limit = payload
    form, selected_rho = CONDITIONS[condition]

    def active_cost(*args, **kwargs):
        return condition_cost(form, selected_rho, *args, **kwargs)

    tune_base.deprivation_cost = active_cost
    start = time.time()
    row: dict[str, Any] = {
        "task_key": task_key(condition, instance_name, seed),
        "condition": condition,
        "function_form": form,
        "rho": "" if selected_rho is None else selected_rho,
        "instance": instance_name,
        "seed": seed,
        "iterations": iterations,
        "time_limit": time_limit,
    }
    try:
        instance, evaluator, classes = tune_base.load_instance_for_tuning(
            instance_name,
            seed=seed,
            instance_dir="Instance25",
        )
        sa_dict = ALNS_CFG.build_sa_config_dict()
        sa_dict["iterations"] = iterations
        sa_dict["size"] = "small"
        sa_dict["log_operator_metrics"] = False
        drone_priority = ALNS_CFG.drone_priority
        bonus = make_alns_bonus()
        destroy_ops = [
            DestroyRandom(instance, rng=random.Random(seed + 1000), anchor_strategy="rebase_to_neighbor"),
            DestroyWorstDistance(instance, rng=random.Random(seed + 1004), anchor_strategy="rebase_to_neighbor"),
            DestroyShaw(instance, rng=random.Random(seed + 1002), anchor_strategy="rebase_to_neighbor"),
        ]
        repair_ops = [
            RepairCheapest(instance, rng=random.Random(seed + 2004), drone_priority=drone_priority, robust_energy_mode="embedded", **bonus),
            RepairDronePriorityRegret(instance, rng=random.Random(seed + 2002), drone_priority=drone_priority, robust_energy_mode="embedded", **bonus),
            RepairTruckFirst(instance, rng=random.Random(seed + 2003), drone_priority=drone_priority, robust_energy_mode="embedded", **bonus),
            RepairEqualPriority(instance, rng=random.Random(seed + 2001), drone_priority=drone_priority, robust_energy_mode="embedded", **bonus),
        ]
        initial, _ = build_shared_initial_solution(instance, evaluator)
        alns = SimulatedAnnealingALNS(
            instance=instance,
            destroy_ops=destroy_ops,
            repair_ops=repair_ops,
            evaluator=evaluator,
            cfg=SANNCfg(**sa_dict),
            rng=random.Random(seed),
            verbose=False,
        )
        best = alns.run(initial, time_limit=time_limit)
        details = evaluator.evaluate_with_details(best)
        result = details.result
        delays = list(details.delay_breakdown.nodes)
        class_delay_hours = {key: 0.0 for key in WANG_SUPPLY_CLASSES}
        class_delay_cost = {key: 0.0 for key in WANG_SUPPLY_CLASSES}
        for delay in delays:
            key = normalize_class(classes.get(delay.node_id))
            class_delay_hours[key] += float(delay.delay)
            class_delay_cost[key] += active_cost(
                delay.delay,
                key,
                cost_lambda=ALNS_CFG.cost_lambda,
                rho=ALNS_CFG.cost_rho,
                normalized=ALNS_CFG.cost_normalized,
            )
        total_delay_hours = sum(float(delay.delay) for delay in delays)
        drone_customers = sum(len(task.nodes) for task in best.drone_tasks)
        stats = getattr(alns, "last_run_stats", {}) or {}
        row.update(
            {
                "feasible": result.feasible,
                "total_cost": result.total_cost,
                "truck_cost": result.truck_distance_cost,
                "drone_cost": result.drone_distance_cost,
                "delay_cost": result.delay_penalty,
                "delayed_nodes": len(delays),
                "total_delay_hours": total_delay_hours,
                "mean_delay_hours": total_delay_hours / len(delays) if delays else 0.0,
                "max_delay_hours": max((float(delay.delay) for delay in delays), default=0.0),
                "drone_customers": drone_customers,
                "runtime_sec": time.time() - start,
                "termination_reason": stats.get("termination_reason", ""),
                "class_delay_hours": json.dumps(class_delay_hours, sort_keys=True),
                "class_delay_cost": json.dumps(class_delay_cost, sort_keys=True),
                "error": "",
            }
        )
    except Exception as exc:
        row.update(
            {
                "feasible": False,
                "runtime_sec": time.time() - start,
                "error": repr(exc),
            }
        )
    return row


def read_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["task_key"] for row in csv.DictReader(handle) if row.get("task_key")}


def append_row(path: Path, row: dict[str, Any]) -> None:
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def summarize(trials_path: Path, output_path: Path) -> None:
    with trials_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    usable = [
        row
        for row in rows
        if row.get("feasible") == "True"
        and row.get("total_cost")
        and math.isfinite(float(row["total_cost"]))
    ]
    baseline = {
        (row["instance"], int(row["seed"])): float(row["total_cost"])
        for row in usable
        if row["condition"] == BASELINE
    }
    summary_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        group = [row for row in usable if row["condition"] == condition]
        costs = [float(row["total_cost"]) for row in group]
        delays = [float(row["delay_cost"]) for row in group]
        delay_hours = [float(row["total_delay_hours"]) for row in group]
        savings = [
            (baseline[(row["instance"], int(row["seed"]))] - float(row["total_cost"]))
            / baseline[(row["instance"], int(row["seed"]))]
            * 100.0
            for row in group
            if (row["instance"], int(row["seed"])) in baseline
        ]
        summary_rows.append(
            {
                "condition": condition,
                "function_form": CONDITIONS[condition][0],
                "rho": "" if CONDITIONS[condition][1] is None else CONDITIONS[condition][1],
                "runs": len(group),
                "mean_total_cost": sum(costs) / len(costs) if costs else "",
                "std_total_cost": float(np_std(costs)) if costs else "",
                "mean_delay_cost": sum(delays) / len(delays) if delays else "",
                "mean_total_delay_hours": sum(delay_hours) / len(delay_hours) if delay_hours else "",
                "mean_paired_saving_vs_baseline_pct": sum(savings) / len(savings) if savings else "",
            }
        )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)


def np_std(values: list[float]) -> float:
    if not values:
        return float("nan")
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="43-47")
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--instances", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    selected = [item.strip() for item in args.conditions.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}")

    output_dir = args.output_dir or (
        PROJECT_ROOT / "results" / "revision_experiments" / "deprivation_sensitivity"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_path = output_dir / "trials.csv"
    completed = read_completed(trials_path)
    tasks = [
        (condition, instance, seed, args.iterations, args.time_limit)
        for condition in selected
        for instance in parse_instances(args.instances)
        for seed in parse_seeds(args.seeds)
        if task_key(condition, instance, seed) not in completed
    ]
    print(f"Pending runs: {len(tasks)}; completed: {len(completed)}")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_task, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            append_row(trials_path, row)
            print(
                f"[{index}/{len(tasks)}] {row['task_key']} "
                f"feasible={row.get('feasible')} cost={row.get('total_cost', '')} "
                f"error={row.get('error', '')}"
            )
    summarize(trials_path, output_dir / "summary.csv")
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
