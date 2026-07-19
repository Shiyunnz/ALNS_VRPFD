#!/usr/bin/env python3
"""Ablation of the greedy initial drone assignment under limited fleets.

For each paired instance/seed and drone fleet size, the script records:

1. the feasible initial assignment produced by the construction and repair
   procedure that contains the benefit-based greedy assignment in Eq. (49);
2. the final solution after the complete ALNS search.

The experiment quantifies whether the final search remains tied to the greedy
assignment by measuring objective improvement and changes in drone-served
customer sets.
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
from revision.tune_base import load_instance_for_tuning  # noqa: E402
from revision.validate_alns_ts_ga import (  # noqa: E402
    ALNS_CFG,
    build_shared_initial_solution,
    make_alns_bonus,
)


FIELDS = [
    "task_key",
    "instance",
    "seed",
    "drone_count",
    "iterations",
    "time_limit",
    "initial_feasible",
    "initial_cost",
    "initial_truck_cost",
    "initial_drone_cost",
    "initial_delay_cost",
    "initial_drone_customers",
    "initial_drone_customer_ids",
    "initial_constructor",
    "final_feasible",
    "final_cost",
    "final_truck_cost",
    "final_drone_cost",
    "final_delay_cost",
    "final_drone_customers",
    "final_drone_customer_ids",
    "cost_improvement_from_initial_pct",
    "assignment_jaccard",
    "customers_removed_from_initial",
    "customers_added_after_search",
    "runtime_sec",
    "termination_reason",
    "error",
]


def parse_seeds(value: str) -> list[int]:
    if "-" in value and "," not in value:
        start, end = value.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(item) for item in value.split(",") if item.strip()]


def parse_counts(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


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


def task_key(instance: str, seed: int, drone_count: int) -> str:
    return f"{instance}|{seed}|drones={drone_count}"


def drone_customer_set(solution: Any) -> set[int]:
    return {
        int(customer)
        for task in getattr(solution, "drone_tasks", []) or []
        for customer in task.nodes
    }


def safe_evaluate(evaluator: Any, solution: Any) -> dict[str, Any]:
    try:
        result = evaluator.evaluate_solution(solution)
        return {
            "feasible": bool(result.feasible),
            "cost": float(result.total_cost),
            "truck_cost": float(result.truck_distance_cost),
            "drone_cost": float(result.drone_distance_cost),
            "delay_cost": float(result.delay_penalty),
        }
    except Exception as exc:
        return {
            "feasible": False,
            "cost": float("inf"),
            "truck_cost": float("nan"),
            "drone_cost": float("nan"),
            "delay_cost": float("nan"),
            "evaluation_error": repr(exc),
        }


def build_feasible_greedy_assignment(
    truck_backbone: Any,
    evaluator: Any,
    greedy_operator: RepairTruckFirst,
) -> tuple[Any, dict[str, Any]]:
    """Apply one benefit-based migration at a time and keep feasible gains."""
    current = truck_backbone.clone()
    current_eval = safe_evaluate(evaluator, current)
    customers = [
        int(node)
        for route in truck_backbone.truck_routes
        for node in route.nodes[1:-1]
    ]
    for customer in customers:
        candidate = current.clone()
        greedy_operator._migrate_to_drones(candidate, [customer])
        candidate_eval = safe_evaluate(evaluator, candidate)
        if (
            candidate_eval["feasible"]
            and math.isfinite(float(candidate_eval["cost"]))
            and float(candidate_eval["cost"]) < float(current_eval["cost"]) - 1e-9
        ):
            current = candidate
            current_eval = candidate_eval
    return current, current_eval


def run_task(payload: tuple[str, int, int, int, float]) -> dict[str, Any]:
    instance_name, seed, drone_count, iterations, time_limit = payload
    start = time.time()
    row: dict[str, Any] = {
        "task_key": task_key(instance_name, seed, drone_count),
        "instance": instance_name,
        "seed": seed,
        "drone_count": drone_count,
        "iterations": iterations,
        "time_limit": time_limit,
    }
    try:
        instance, evaluator, _ = load_instance_for_tuning(
            instance_name,
            seed=seed,
            instance_dir="Instance25",
        )
        instance.vehicle_specs["drone"].number = drone_count

        truck_backbone, initial_metrics = build_shared_initial_solution(instance, evaluator)
        sa_dict = ALNS_CFG.build_sa_config_dict()
        sa_dict["iterations"] = iterations
        sa_dict["size"] = "small"
        sa_dict["log_operator_metrics"] = False
        drone_priority = ALNS_CFG.drone_priority
        bonus = make_alns_bonus()
        greedy_operator = RepairTruckFirst(
            instance,
            rng=random.Random(seed + 9000),
            drone_priority=drone_priority,
            robust_energy_mode="embedded",
            **bonus,
        )
        initial, initial_eval = build_feasible_greedy_assignment(
            truck_backbone,
            evaluator,
            greedy_operator,
        )
        initial_customers = drone_customer_set(initial)
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
        alns = SimulatedAnnealingALNS(
            instance=instance,
            destroy_ops=destroy_ops,
            repair_ops=repair_ops,
            evaluator=evaluator,
            cfg=SANNCfg(**sa_dict),
            rng=random.Random(seed),
            verbose=False,
        )
        final = alns.run(initial, time_limit=time_limit)
        final_eval = safe_evaluate(evaluator, final)
        final_customers = drone_customer_set(final)
        union = initial_customers | final_customers
        intersection = initial_customers & final_customers
        initial_cost = float(initial_eval["cost"])
        final_cost = float(final_eval["cost"])
        improvement = (
            (initial_cost - final_cost) / initial_cost * 100.0
            if math.isfinite(initial_cost) and initial_cost != 0.0 and math.isfinite(final_cost)
            else ""
        )
        stats = getattr(alns, "last_run_stats", {}) or {}
        row.update(
            {
                "initial_feasible": initial_eval["feasible"],
                "initial_cost": initial_eval["cost"],
                "initial_truck_cost": initial_eval["truck_cost"],
                "initial_drone_cost": initial_eval["drone_cost"],
                "initial_delay_cost": initial_eval["delay_cost"],
                "initial_drone_customers": len(initial_customers),
                "initial_drone_customer_ids": json.dumps(sorted(initial_customers)),
                "initial_constructor": "benefit_greedy_migration",
                "final_feasible": final_eval["feasible"],
                "final_cost": final_eval["cost"],
                "final_truck_cost": final_eval["truck_cost"],
                "final_drone_cost": final_eval["drone_cost"],
                "final_delay_cost": final_eval["delay_cost"],
                "final_drone_customers": len(final_customers),
                "final_drone_customer_ids": json.dumps(sorted(final_customers)),
                "cost_improvement_from_initial_pct": improvement,
                "assignment_jaccard": len(intersection) / len(union) if union else 1.0,
                "customers_removed_from_initial": json.dumps(sorted(initial_customers - final_customers)),
                "customers_added_after_search": json.dumps(sorted(final_customers - initial_customers)),
                "runtime_sec": time.time() - start,
                "termination_reason": stats.get("termination_reason", ""),
                "error": "",
            }
        )
    except Exception as exc:
        row.update(
            {
                "final_feasible": False,
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


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def std(values: list[float]) -> float:
    if not values:
        return float("nan")
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def summarize(trials_path: Path, output_path: Path) -> None:
    with trials_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    summaries: list[dict[str, Any]] = []
    for drone_count in sorted({int(row["drone_count"]) for row in rows if row.get("drone_count")}):
        group = [
            row
            for row in rows
            if int(row["drone_count"]) == drone_count
            and row.get("final_feasible") == "True"
            and row.get("final_cost")
        ]
        improvements = [
            float(row["cost_improvement_from_initial_pct"])
            for row in group
            if row.get("cost_improvement_from_initial_pct") not in ("", None)
        ]
        jaccards = [float(row["assignment_jaccard"]) for row in group if row.get("assignment_jaccard")]
        changed = [value < 1.0 - 1e-12 for value in jaccards]
        summaries.append(
            {
                "drone_count": drone_count,
                "runs": len(group),
                "initial_feasible_runs": sum(row.get("initial_feasible") == "True" for row in group),
                "mean_initial_cost": mean(
                    [
                        float(row["initial_cost"])
                        for row in group
                        if row.get("initial_cost") and math.isfinite(float(row["initial_cost"]))
                    ]
                ),
                "mean_final_cost": mean([float(row["final_cost"]) for row in group]),
                "mean_improvement_from_initial_pct": mean(improvements),
                "std_improvement_from_initial_pct": std(improvements),
                "mean_assignment_jaccard": mean(jaccards),
                "assignment_changed_share": sum(changed) / len(changed) if changed else float("nan"),
                "mean_initial_drone_customers": mean([float(row["initial_drone_customers"]) for row in group]),
                "mean_final_drone_customers": mean([float(row["final_drone_customers"]) for row in group]),
            }
        )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="43-47")
    parser.add_argument("--drone-counts", default="1,2,3")
    parser.add_argument("--instances", default=None)
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or (
        PROJECT_ROOT / "results" / "revision_experiments" / "greedy_assignment_ablation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_path = output_dir / "trials.csv"
    completed = read_completed(trials_path)
    tasks = [
        (instance, seed, count, args.iterations, args.time_limit)
        for instance in parse_instances(args.instances)
        for seed in parse_seeds(args.seeds)
        for count in parse_counts(args.drone_counts)
        if task_key(instance, seed, count) not in completed
    ]
    print(f"Pending runs: {len(tasks)}; completed: {len(completed)}")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_task, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            append_row(trials_path, row)
            print(
                f"[{index}/{len(tasks)}] {row['task_key']} "
                f"initial={row.get('initial_cost', '')} final={row.get('final_cost', '')} "
                f"error={row.get('error', '')}"
            )
    summarize(trials_path, output_dir / "summary.csv")
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
