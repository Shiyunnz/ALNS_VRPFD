#!/usr/bin/env python3
"""Run quick convergence traces for ALNS, TS, and GA termination analysis.

The script is intentionally lightweight: it runs a small set of instance/seed
pairs, records each algorithm's best-found trace, and derives rough no-improve
termination ranges from the observed first-hit and tail-stagnation behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "heuristics" / "tabu_search"))
sys.path.insert(0, str(PROJECT_ROOT / "heuristics" / "ga"))

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
from revision.tune_base import INFEASIBLE_PENALTY, load_instance_for_tuning  # noqa: E402
from revision.validate_alns_ts_ga import (  # noqa: E402
    ALNS_CFG,
    GA_FINAL_CFG,
    TS_FINAL_CFG,
    build_shared_initial_solution,
    make_alns_bonus,
    run_ga,
    run_ts,
)


def finite(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def parse_instances(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_seeds(value: str) -> List[int]:
    if "," in value:
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    if "-" in value:
        start, end = value.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(value)]


def run_alns_trace(
    instance_name: str,
    instance_size: int,
    seed: int,
    *,
    iterations: int,
    time_limit: float,
) -> Dict[str, Any]:
    instance_dir = f"Instance{instance_size}"
    instance, evaluator, _ = load_instance_for_tuning(
        instance_name, seed=seed, instance_dir=instance_dir
    )

    sa_cfg = ALNS_CFG.build_sa_config_dict()
    sa_cfg["iterations"] = iterations
    sa_cfg["size"] = "small"
    sa_cfg["log_operator_metrics"] = False

    rng_alns = random.Random(seed)
    dp = ALNS_CFG.drone_priority
    bonus = make_alns_bonus()

    destroy_ops = [
        DestroyRandom(instance, rng=random.Random(seed + 1000), anchor_strategy="rebase_to_neighbor"),
        DestroyWorstDistance(instance, rng=random.Random(seed + 1004), anchor_strategy="rebase_to_neighbor"),
        DestroyShaw(instance, rng=random.Random(seed + 1002), anchor_strategy="rebase_to_neighbor"),
    ]
    repair_ops = [
        RepairCheapest(
            instance,
            rng=random.Random(seed + 2004),
            drone_priority=dp,
            robust_energy_mode="embedded",
            **bonus,
        ),
        RepairDronePriorityRegret(
            instance,
            rng=random.Random(seed + 2002),
            drone_priority=dp,
            robust_energy_mode="embedded",
            **bonus,
        ),
        RepairTruckFirst(
            instance,
            rng=random.Random(seed + 2003),
            drone_priority=dp,
            robust_energy_mode="embedded",
            **bonus,
        ),
        RepairEqualPriority(
            instance,
            rng=random.Random(seed + 2001),
            drone_priority=dp,
            robust_energy_mode="embedded",
            **bonus,
        ),
    ]

    start = time.time()
    try:
        initial, initial_metrics = build_shared_initial_solution(instance, evaluator)
        alns = SimulatedAnnealingALNS(
            instance=instance,
            destroy_ops=destroy_ops,
            repair_ops=repair_ops,
            evaluator=evaluator,
            cfg=SANNCfg(**sa_cfg),
            rng=rng_alns,
            verbose=False,
        )
        best = alns.run(initial, time_limit=time_limit)
        runtime = time.time() - start
        res = evaluator.evaluate_solution(best)
        stats = getattr(alns, "last_run_stats", {}) or {}
        trace = [
            {
                "unit": row.get("iteration"),
                "best_cost": row.get("best_cost"),
                "elapsed_time": None,
            }
            for row in getattr(alns, "convergence_history", []) or []
        ]
        return {
            "cost": res.total_cost,
            "feasible": res.feasible,
            "delay_cost": res.delay_penalty,
            "truck_cost": res.truck_distance_cost,
            "drone_cost": res.drone_distance_cost,
            "runtime": runtime,
            "termination_reason": stats.get("termination_reason"),
            "units_completed": stats.get("executed_iterations", len(trace)),
            "trace": trace,
            **initial_metrics,
        }
    except Exception as exc:
        return {
            "cost": INFEASIBLE_PENALTY,
            "feasible": False,
            "runtime": time.time() - start,
            "error": str(exc),
            "trace": [],
        }


def normalize_trace(algo: str, result: Dict[str, Any]) -> List[Dict[str, Any]]:
    if algo == "alns":
        return result.get("trace", []) or []

    stats_key = "ts_stats" if algo == "ts" else "ga_stats"
    stats = result.get(stats_key, {}) or {}
    unit_key = "iterations" if algo == "ts" else "generations"
    cost_key = "best_feasible_cost" if algo == "ts" else "best_feasible_fitness"
    units = stats.get(unit_key, []) or []
    costs = stats.get(cost_key, []) or []
    elapsed = stats.get("elapsed_time", []) or []
    return [
        {"unit": unit, "best_cost": cost, "elapsed_time": t}
        for unit, cost, t in zip(units, costs, elapsed)
    ]


def first_hit(trace: Iterable[Dict[str, Any]], threshold: float) -> Dict[str, Any] | None:
    for row in trace:
        cost = finite(row.get("best_cost"))
        if cost is not None and cost <= threshold:
            return {
                "unit": row.get("unit"),
                "elapsed_time": row.get("elapsed_time"),
                "best_cost": cost,
            }
    return None


def trace_summary(trace: List[Dict[str, Any]], final_cost: Any) -> Dict[str, Any]:
    valid = [
        row
        for row in trace
        if finite(row.get("best_cost")) is not None and finite(row.get("unit")) is not None
    ]
    if not valid:
        return {
            "trace_points": 0,
            "first_final_hit": None,
            "first_within_1pct_final": None,
            "first_within_5pct_final": None,
            "last_improvement_unit": None,
            "tail_no_improve_units": None,
            "improvement_count": 0,
        }

    best_seen = float("inf")
    improvements: List[Dict[str, Any]] = []
    for row in valid:
        cost = finite(row.get("best_cost"))
        if cost is not None and cost < best_seen - 1e-9:
            best_seen = cost
            improvements.append(row)

    final = finite(final_cost)
    target = final if final is not None else best_seen
    last_unit = int(valid[-1]["unit"])
    last_imp_unit = int(improvements[-1]["unit"]) if improvements else None
    return {
        "trace_points": len(valid),
        "first_final_hit": first_hit(valid, target + 1e-9),
        "first_within_1pct_final": first_hit(valid, target * 1.01),
        "first_within_5pct_final": first_hit(valid, target * 1.05),
        "last_improvement_unit": last_imp_unit,
        "tail_no_improve_units": (
            last_unit - last_imp_unit if last_imp_unit is not None else None
        ),
        "improvement_count": len(improvements),
    }


def run_one(
    instance: str,
    size: int,
    seed: int,
    *,
    time_limit: float,
    alns_iterations: int,
) -> Dict[str, Any]:
    print(f"[run] {instance} seed={seed} time_limit={time_limit}s")
    alns_result = run_alns_trace(
        instance,
        size,
        seed,
        iterations=alns_iterations,
        time_limit=time_limit,
    )
    ts_result = run_ts(instance, size, TS_FINAL_CFG, seed, time_limit_override=time_limit)
    ga_result = run_ga(instance, size, GA_FINAL_CFG, seed, time_limit_override=time_limit)

    results = {
        "alns": alns_result,
        "ts": ts_result,
        "ga": ga_result,
    }
    compact: Dict[str, Any] = {}
    for algo, result in results.items():
        trace = normalize_trace(algo, result)
        summary = trace_summary(trace, result.get("cost"))
        units_completed = (
            result.get("units_completed")
            or result.get("iterations_completed")
            or result.get("generations_completed")
            or summary.get("trace_points")
        )
        compact[algo] = {
            "cost": result.get("cost"),
            "feasible": result.get("feasible"),
            "runtime": result.get("runtime"),
            "units_completed": units_completed,
            "summary": summary,
            "trace": trace,
            "error": result.get("error"),
        }
    return {"instance": instance, "seed": seed, "algorithms": compact}


def recommend_patience(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    recommendations: Dict[str, Any] = {}
    for algo in ("alns", "ts", "ga"):
        tails: List[int] = []
        first_hits: List[int] = []
        completed: List[int] = []
        for run in runs:
            data = run["algorithms"].get(algo, {})
            summary = data.get("summary", {})
            tail = summary.get("tail_no_improve_units")
            hit = (summary.get("first_final_hit") or {}).get("unit")
            units = data.get("units_completed")
            if isinstance(tail, (int, float)) and math.isfinite(tail):
                tails.append(int(tail))
            if isinstance(hit, (int, float)) and math.isfinite(hit):
                first_hits.append(int(hit))
            if isinstance(units, (int, float)) and math.isfinite(units):
                completed.append(int(units))

        if not completed:
            recommendations[algo] = {"status": "no usable runs"}
            continue

        max_units = max(completed)
        base = max(1, round(max_units * 0.25))
        tail_median = round(statistics.median(tails)) if tails else None
        hit_median = round(statistics.median(first_hits)) if first_hits else None
        # Conservative quick-experiment recommendation: at least 20% of the
        # observed run length, capped by 35%, and not smaller than the median
        # observed post-best tail when available.
        lower = max(round(max_units * 0.20), 1)
        upper = max(round(max_units * 0.35), lower)
        if tail_median is not None:
            lower = max(lower, min(tail_median, upper))
        recommendations[algo] = {
            "observed_units": completed,
            "observed_first_final_hits": first_hits,
            "observed_tail_no_improve_units": tails,
            "median_first_final_hit": hit_median,
            "median_tail_no_improve": tail_median,
            "suggested_no_improve_range": [lower, upper],
            "fallback_25pct_rule": base,
        }
    return recommendations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", default="R_30_10_2,R_30_50_1")
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--seeds", default="44")
    parser.add_argument("--time-limit", type=float, default=45.0)
    parser.add_argument("--alns-iterations", type=int, default=4000)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "results" / "termination_analysis" / "quick_termination_analysis.json",
    )
    args = parser.parse_args()

    runs = [
        run_one(
            instance,
            args.size,
            seed,
            time_limit=args.time_limit,
            alns_iterations=args.alns_iterations,
        )
        for instance in parse_instances(args.instances)
        for seed in parse_seeds(args.seeds)
    ]
    output = {
        "instances": parse_instances(args.instances),
        "size": args.size,
        "seeds": parse_seeds(args.seeds),
        "time_limit": args.time_limit,
        "alns_iterations": args.alns_iterations,
        "runs": runs,
        "recommendations": recommend_patience(runs),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"[saved] {args.out}")
    print(json.dumps(output["recommendations"], indent=2, default=str))


if __name__ == "__main__":
    main()
