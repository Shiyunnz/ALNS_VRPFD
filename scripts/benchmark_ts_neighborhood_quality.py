#!/usr/bin/env python3
"""Benchmark TS neighborhood quality against unchanged GA baseline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from revision.validate_alns_ts_ga import (  # noqa: E402
    GA_FINAL_CFG,
    TS_FINAL_CFG,
    run_ga,
    run_ts,
)

ALNS_TARGET_COST = 65.72


def _finite_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def first_hit_time(
    costs: Iterable[Any],
    times: Iterable[Any],
    threshold: float,
) -> float | None:
    """Return the first elapsed time where cost is at or below threshold."""
    for cost_raw, time_raw in zip(costs, times):
        cost = _finite_or_none(cost_raw)
        elapsed = _finite_or_none(time_raw)
        if cost is not None and elapsed is not None and cost <= threshold:
            return elapsed
    return None


def first_hit_metrics(stats: Dict[str, Any], cost_key: str) -> Dict[str, float | None]:
    costs = stats.get(cost_key, []) or []
    times = stats.get("elapsed_time", []) or []
    return {
        "alns_exact": first_hit_time(costs, times, ALNS_TARGET_COST),
        "alns_plus_1pct": first_hit_time(costs, times, ALNS_TARGET_COST * 1.01),
        "alns_plus_5pct": first_hit_time(costs, times, ALNS_TARGET_COST * 1.05),
        "cost_le_90": first_hit_time(costs, times, 90.0),
    }


def summarize_ts_neighborhood_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    pools = stats.get("candidate_pool_size", []) or []
    selected = stats.get("selected_candidate_count", []) or []
    by_bucket = stats.get("selected_by_bucket", []) or []
    totals = {
        "distance_proxy": 0,
        "drone_saving": 0,
        "violation_fix": 0,
        "random_diversity": 0,
    }
    for row in by_bucket:
        if not isinstance(row, dict):
            continue
        for bucket in totals:
            totals[bucket] += int(row.get(bucket, 0) or 0)

    trace = stats.get("best_feasible_cost", []) or []
    if len(trace) <= 20:
        sample = trace
    else:
        sample = trace[:10] + trace[-10:]

    return {
        "average_candidate_pool_size": (
            sum(pools) / len(pools) if pools else 0.0
        ),
        "average_selected_candidate_count": (
            sum(selected) / len(selected) if selected else 0.0
        ),
        "bucket_selection_totals": totals,
        "final_best_feasible_cost_trace_sample": sample,
    }


def _run_one_seed(
    *,
    instance: str,
    size: int,
    seed: int,
    time_limit: float,
) -> Dict[str, Any]:
    ts_result = run_ts(
        instance,
        size,
        TS_FINAL_CFG,
        seed,
        time_limit_override=time_limit,
    )
    ga_result = run_ga(
        instance,
        size,
        GA_FINAL_CFG,
        seed,
        time_limit_override=time_limit,
    )

    ts_stats = ts_result.get("ts_stats", {}) or {}
    ga_stats = ga_result.get("ga_stats", {}) or {}

    return {
        "seed": seed,
        "ts": {
            "final_cost": ts_result.get("cost"),
            "feasible": ts_result.get("feasible"),
            "runtime": ts_result.get("runtime"),
            "iterations_completed": ts_result.get("iterations_completed"),
            "first_hit": first_hit_metrics(ts_stats, "best_feasible_cost"),
            "neighborhood_stats_summary": summarize_ts_neighborhood_stats(ts_stats),
        },
        "ga": {
            "final_cost": ga_result.get("cost"),
            "feasible": ga_result.get("feasible"),
            "runtime": ga_result.get("runtime"),
            "generations_completed": ga_result.get("generations_completed"),
            "first_hit": first_hit_metrics(ga_stats, "best_feasible_fitness"),
        },
    }


def run_benchmark(
    *,
    instance: str,
    size: int,
    seeds: List[int],
    time_limit: float,
    out_path: Path,
) -> Dict[str, Any]:
    runs = [
        _run_one_seed(
            instance=instance,
            size=size,
            seed=seed,
            time_limit=time_limit,
        )
        for seed in seeds
    ]
    result = {
        "instance": instance,
        "size": size,
        "seeds": seeds,
        "time_limit": time_limit,
        "alns_target_cost": ALNS_TARGET_COST,
        "thresholds": {
            "alns_exact": ALNS_TARGET_COST,
            "alns_plus_1pct": ALNS_TARGET_COST * 1.01,
            "alns_plus_5pct": ALNS_TARGET_COST * 1.05,
            "cost_le_90": 90.0,
        },
        "runs": runs,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def parse_seeds(value: str) -> List[int]:
    if "," in value:
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    if "-" in value:
        start, end = value.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(value)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = run_benchmark(
        instance=args.instance,
        size=args.size,
        seeds=parse_seeds(args.seeds),
        time_limit=args.time_limit,
        out_path=args.out,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
