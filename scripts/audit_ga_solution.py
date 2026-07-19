#!/usr/bin/env python3
"""Run one GA case, save its full solution, and audit it with the ALNS evaluator."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from alns_vrpfd.evaluation.run_record import build_run_record
from revision.tune_base import load_instance_for_tuning
from revision.validate_alns_ts_ga import (
    GA_FINAL_CFG,
    _build_search_evaluator,
    build_shared_initial_solution,
)
from heuristics.ga.ga import GAConfig, GeneticAlgorithm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save and independently audit a complete GA solution."
    )
    parser.add_argument("--instance-size", type=int, default=50)
    parser.add_argument("--instance", default="R_50_50_4")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "revision_experiments"
        / "ga_solution_audit"
        / "R_50_50_4_seed101.json",
    )
    return parser.parse_args()


def solution_dict(solution) -> dict[str, Any]:
    return {
        "truck_routes": [
            {
                "route_id": route.id,
                "nodes": list(route.nodes),
                "capacity": route.capacity,
                "current_load": route.current_load,
            }
            for route in solution.truck_routes
        ],
        "drone_tasks": [
            {
                "task_id": task.task_id,
                "drone_id": task.drone_id,
                "launch_truck": task.launch_truck,
                "launch_node": task.launch_node,
                "customers": list(task.customers()),
                "land_truck": task.land_truck,
                "retrieve_node": task.retrieve_node,
                "payloads": list(task.payloads),
            }
            for task in solution.drone_tasks
        ],
    }


def audit_solution(instance, evaluator, solution, details) -> dict[str, Any]:
    route_ids = [route.id for route in solution.truck_routes]
    duplicate_route_ids = sorted(
        route_id for route_id in set(route_ids) if route_ids.count(route_id) > 1
    )
    task_ids = [
        task.task_id
        for task in solution.drone_tasks
        if task.task_id is not None
    ]
    duplicate_task_ids = sorted(
        task_id for task_id in set(task_ids) if task_ids.count(task_id) > 1
    )
    demands = instance.customer_manager.demands()
    route_loads = []
    capacity_violation = False
    for route in solution.truck_routes:
        load = sum(float(demands.get(customer, 0.0)) for customer in route.customers())
        # Match SearchEvaluator: current_load mirrors the assigned route load
        # in many constructors and must not be added a second time.
        excess = max(0.0, load - float(route.capacity))
        route_loads.append(
            {
                "route_id": route.id,
                "load": load,
                "capacity": route.capacity,
                "excess": excess,
            }
        )
        capacity_violation = capacity_violation or excess > 1e-9

    checks = {
        "duplicate_route_ids": duplicate_route_ids,
        "duplicate_task_ids": duplicate_task_ids,
        "depot_start_retrieve_violation": evaluator._has_depot_start_retrieve_violation(
            solution
        ),
        "anchor_conflict": evaluator._has_drone_anchor_conflicts(
            solution.drone_tasks
        ),
        "drone_limit_violation": evaluator._has_drone_limit_violations(solution),
        "drone_task_violation": evaluator._has_drone_task_violations(solution),
        "customer_coverage_violation": evaluator._has_customer_coverage_violation(
            solution
        ),
        "forced_drone_violation": evaluator._has_forced_drone_violation(solution),
        "capacity_violation": capacity_violation,
        "hard_time_window_violation_count": len(
            details.delay_breakdown.violations
        ),
        "robust_energy_feasible": details.robustness.feasible,
    }
    energy = [
        {
            "task_id": assessment.task_id,
            "drone_id": assessment.drone_id,
            "nominal_energy": assessment.nominal_energy,
            "worst_case_energy": assessment.worst_case_energy,
            "capacity": assessment.capacity,
            "margin": assessment.margin,
            "feasible": assessment.feasible,
        }
        for assessment in details.robustness.task_breakdown
    ]
    cost_components = {
        "truck_cost": details.result.truck_distance_cost,
        "drone_cost": details.result.drone_distance_cost,
        "delay_cost": details.result.delay_penalty,
        "total_cost": details.result.total_cost,
        "component_sum": (
            details.result.truck_distance_cost
            + details.result.drone_distance_cost
            + details.result.delay_penalty
        ),
    }
    return {
        "evaluator_feasible": details.result.feasible,
        "all_structural_checks_pass": (
            not duplicate_route_ids
            and not duplicate_task_ids
            and not any(
                checks[key]
                for key in (
                    "depot_start_retrieve_violation",
                    "anchor_conflict",
                    "drone_limit_violation",
                    "drone_task_violation",
                    "customer_coverage_violation",
                    "forced_drone_violation",
                    "capacity_violation",
                )
            )
            and checks["hard_time_window_violation_count"] == 0
            and checks["robust_energy_feasible"]
        ),
        "checks": checks,
        "route_loads": route_loads,
        "energy_tasks": energy,
        "minimum_energy_margin": min(
            (assessment["margin"] for assessment in energy), default=None
        ),
        "cost_components": cost_components,
        "cost_components_consistent": (
            math.isfinite(cost_components["total_cost"])
            and abs(
                cost_components["total_cost"] - cost_components["component_sum"]
            )
            <= 1e-8
        ),
    }


def normalized_route_id_audit(instance, evaluator, solution) -> dict[str, Any]:
    """Re-evaluate after assigning unique route IDs and remapping drone anchors."""
    normalized = solution.clone()
    for route_index, route in enumerate(normalized.truck_routes):
        route.id = route_index

    for task in normalized.drone_tasks:
        launch_matches = [
            index
            for index, route in enumerate(normalized.truck_routes)
            if task.launch_node in route.nodes
        ]
        retrieve_matches = [
            index
            for index, route in enumerate(normalized.truck_routes)
            if task.retrieve_node in route.nodes
        ]
        if task.launch_truck is not None and len(launch_matches) == 1:
            task.launch_truck = launch_matches[0]
        if task.land_truck is not None and len(retrieve_matches) == 1:
            task.land_truck = retrieve_matches[0]

    details = evaluator.evaluate_with_details(normalized)
    result = audit_solution(instance, evaluator, normalized, details)
    result["normalized_solution"] = solution_dict(normalized)
    result["hard_time_window_violations"] = [
        {
            "node_id": violation.node_id,
            "arrival_time": violation.arrival_time,
            "latest_time": violation.latest_time,
            "served_by": violation.served_by,
            "route_id": violation.route_id,
        }
        for violation in details.delay_breakdown.violations
    ]
    return result


def main() -> None:
    args = parse_args()
    instance, evaluator, _ = load_instance_for_tuning(
        args.instance,
        seed=args.seed,
        instance_dir=f"Instance{args.instance_size}",
    )
    initial, initial_metrics = build_shared_initial_solution(instance, evaluator)
    search_evaluator = _build_search_evaluator(instance, evaluator)
    config = GAConfig(
        population_size=GA_FINAL_CFG["population_size"],
        generations=GA_FINAL_CFG["generations"],
        tournament_size=GA_FINAL_CFG["tournament_size"],
        crossover_rate=GA_FINAL_CFG["crossover_rate"],
        mutation_rate=GA_FINAL_CFG["mutation_rate"],
        elite_size=GA_FINAL_CFG["elite_size"],
        max_stagnation=GA_FINAL_CFG["max_stagnation"],
        truck_route_crossover_rate=GA_FINAL_CFG["truck_route_crossover_rate"],
        drone_task_mutation_rate=GA_FINAL_CFG["drone_task_mutation_rate"],
        route_segment_swap_rate=GA_FINAL_CFG["route_segment_swap_rate"],
        time_limit=args.time_limit,
        strict_time_budget=True,
    )
    ga = GeneticAlgorithm(
        instance,
        config,
        evaluator,
        rng=random.Random(args.seed),
        search_evaluator=search_evaluator,
    )

    start = time.time()
    best = ga.run(initial)
    runtime = time.time() - start

    # This is intentionally a fresh final call through the evaluator shared with ALNS.
    details = evaluator.evaluate_with_details(best.solution)
    record = build_run_record(
        instance=instance,
        algorithm="ga_audit",
        solution=best.solution,
        details=details,
        runtime_seconds=runtime,
        config={**GA_FINAL_CFG, "audit_time_limit": args.time_limit},
        seed=args.seed,
        instance_name=args.instance,
    )
    original_audit = audit_solution(instance, evaluator, best.solution, details)
    output = {
        "instance": args.instance,
        "seed": args.seed,
        "runtime_seconds": runtime,
        "ga_fitness": best.fitness,
        "ga_internal_feasible": best.feasible,
        "generations_completed": len(ga.stats.get("generations", [])),
        "initial_metrics": initial_metrics,
        "solution": solution_dict(best.solution),
        "independent_evaluator_audit": original_audit,
        "unique_route_id_reaudit": normalized_route_id_audit(
            instance, evaluator, best.solution
        ),
        "run_record": record,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(json.dumps(output["independent_evaluator_audit"], indent=2))
    print(f"Saved full audit to {args.output}")


if __name__ == "__main__":
    main()
