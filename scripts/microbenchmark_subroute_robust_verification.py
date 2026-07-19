#!/usr/bin/env python3
"""Microbenchmark sub-route robust verification on identical ALNS candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any, Sequence
import argparse
import csv
import json
import math
import random
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alns_vrpfd.evaluation import Evaluator, SubrouteRobustVerifier
from alns_vrpfd.model.initializer import (
    build_initial_solution,
    build_two_phase_initial_solution,
)
from alns_vrpfd.utils.config_loader import ALNSConfig
from alns_vrpfd.utils.io_utils import read_instance
from run_alns import build_operators


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "microbenchmark_subroute_robust_verification"


@dataclass(frozen=True)
class BenchmarkRow:
    instance: str
    seed: int
    candidate_index: int
    full_time_sec: float
    subroute_time_sec: float
    full_feasible: int
    subroute_feasible: int
    checked_drone_tasks: int
    failed_drone_tasks: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark full robust checks vs changed-subroute checks on identical candidates."
    )
    parser.add_argument(
        "--instance-dir",
        type=str,
        default="data/Instance10",
        help="Directory used for instance stems.",
    )
    parser.add_argument(
        "--instances",
        type=str,
        default="R_30_10_1,R_30_10_2,R_30_10_3,R_30_10_4,R_30_10_5",
        help="Comma-separated instance stems or paths.",
    )
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--candidates", type=int, default=250)
    parser.add_argument("--max-attempt-factor", type=int, default=8)
    parser.add_argument("--gamma", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
    )
    return parser.parse_args()


def _parse_ints(text: str) -> list[int]:
    values = [int(token.strip()) for token in text.split(",") if token.strip()]
    if not values:
        raise ValueError("Empty integer list.")
    return values


def _resolve_instances(instances_text: str, instance_dir: Path) -> list[str]:
    resolved: list[str] = []
    for token in instances_text.split(","):
        value = token.strip()
        if not value:
            continue
        path = Path(value)
        if path.suffix != ".txt":
            path = instance_dir / f"{value}.txt"
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Instance not found: {path}")
        resolved.append(str(path))
    return sorted(dict.fromkeys(resolved))


def _configure_instance(instance, cfg: ALNSConfig, gamma: int) -> None:
    if "drone" in instance.vehicle_specs:
        instance.vehicle_specs["drone"].endurance = float("inf")
    instance.configure_robustness(
        drone_battery_capacity=cfg.drone_battery_capacity,
        energy_uncertainty_budget=gamma,
        energy_deviation_rate=cfg.energy_deviation_rate,
        same_truck_retrieval=cfg.same_truck_retrieval,
    )


def _make_evaluator(instance, cfg: ALNSConfig) -> Evaluator:
    return Evaluator(
        instance,
        rendezvous_tolerance=cfg.drone_rendezvous_tolerance,
        forced_drone_customers=cfg.forced_drone_customers,
        allow_multiple_launch_per_node=cfg.relax_allow_multiple_launch_per_node,
        cost_lambda=cfg.cost_lambda,
        cost_rho=cfg.cost_rho,
        cost_normalized=cfg.cost_normalized,
    )


def _build_initial_solution(instance, cfg: ALNSConfig):
    if cfg.raw.get("initial_solution", {}).get("two_phase", True):
        return build_two_phase_initial_solution(
            instance,
            truck_forbidden_customers=cfg.forced_drone_customers,
            allow_multiple_launch_per_node=cfg.relax_allow_multiple_launch_per_node,
        )
    return build_initial_solution(
        instance,
        truck_forbidden_customers=cfg.forced_drone_customers,
        allow_multiple_launch_per_node=cfg.relax_allow_multiple_launch_per_node,
    )


def _sample_quota(rng: random.Random, num_customers: int) -> int:
    lo = max(1, int(round(0.15 * num_customers)))
    hi = max(lo, int(round(0.30 * num_customers)))
    return rng.randint(lo, hi)


def run_instance(
    *,
    instance_path: str,
    seed: int,
    cfg: ALNSConfig,
    target_candidates: int,
    max_attempt_factor: int,
    gamma: int,
) -> list[BenchmarkRow]:
    search_instance = read_instance(instance_path, strategy=cfg.time_window_strategy)
    robust_instance = read_instance(instance_path, strategy=cfg.time_window_strategy)
    _configure_instance(search_instance, cfg, gamma=0)
    _configure_instance(robust_instance, cfg, gamma=gamma)

    search_evaluator = _make_evaluator(search_instance, cfg)
    robust_evaluator = _make_evaluator(robust_instance, cfg)
    verifier = SubrouteRobustVerifier(
        instance=search_instance,
        drone_energy_capacity=robust_instance.robust_config.drone_battery_capacity,
        energy_uncertainty_budget=robust_instance.robust_config.energy_uncertainty_budget,
        energy_deviation_rate=robust_instance.robust_config.energy_deviation_rate,
    )
    destroy_ops, repair_ops = build_operators(
        instance=search_instance,
        seed=seed,
        drone_priority=cfg.drone_priority,
        repair_set="all",
        enable_composite=True,
        drone_bonus_kwargs=cfg.drone_bonus,
        forced_drone_customers=cfg.forced_drone_customers,
        robust_energy_mode="verification",
    )

    rng = random.Random(seed)
    current = _build_initial_solution(search_instance, cfg)
    rows: list[BenchmarkRow] = []
    attempts = 0
    max_attempts = max(target_candidates, target_candidates * max_attempt_factor)
    n_customers = len(search_instance.customer_manager.customer_ids())

    while len(rows) < target_candidates and attempts < max_attempts:
        attempts += 1
        destroy = rng.choice(destroy_ops)
        repair = rng.choice(repair_ops)
        quota = _sample_quota(rng, n_customers)
        try:
            destroyed, pool = destroy.apply(current, quota)
            candidate = repair.apply(destroyed, pool.customers)
            deterministic = search_evaluator.evaluate_solution(candidate)
        except Exception:
            continue
        if not math.isfinite(deterministic.total_cost):
            continue

        full_start = time.perf_counter()
        full_eval = robust_evaluator.evaluate_solution(candidate)
        full_time = time.perf_counter() - full_start

        sub_start = time.perf_counter()
        sub_ok = verifier.verify_candidate(base=current, candidate=candidate)
        sub_time = time.perf_counter() - sub_start
        summary = verifier.last_summary

        rows.append(
            BenchmarkRow(
                instance=Path(instance_path).stem,
                seed=seed,
                candidate_index=len(rows),
                full_time_sec=full_time,
                subroute_time_sec=sub_time,
                full_feasible=int(math.isfinite(full_eval.total_cost)),
                subroute_feasible=int(sub_ok),
                checked_drone_tasks=summary.checked_drone_tasks,
                failed_drone_tasks=summary.failed_drone_tasks,
            )
        )

        if sub_ok and rng.random() < 0.5:
            current = candidate

    return rows


def summarize(rows: Sequence[BenchmarkRow]) -> list[dict[str, Any]]:
    by_instance: dict[str, list[BenchmarkRow]] = {}
    for row in rows:
        by_instance.setdefault(row.instance, []).append(row)
    by_instance["overall"] = list(rows)

    output: list[dict[str, Any]] = []
    for scope, items in sorted(by_instance.items()):
        if not items:
            continue
        full_times = [row.full_time_sec for row in items]
        sub_times = [row.subroute_time_sec for row in items]
        checked = [row.checked_drone_tasks for row in items]
        output.append(
            {
                "scope": scope,
                "candidate_count": len(items),
                "mean_full_time_sec": fmean(full_times),
                "median_full_time_sec": median(full_times),
                "mean_subroute_time_sec": fmean(sub_times),
                "median_subroute_time_sec": median(sub_times),
                "per_check_speedup_full_over_subroute": (
                    fmean(full_times) / fmean(sub_times)
                    if fmean(sub_times) > 0
                    else math.nan
                ),
                "full_feasible_ratio": sum(r.full_feasible for r in items) / len(items),
                "subroute_feasible_ratio": sum(r.subroute_feasible for r in items) / len(items),
                "agreement_ratio": (
                    sum(r.full_feasible == r.subroute_feasible for r in items) / len(items)
                ),
                "mean_checked_drone_tasks": fmean(checked),
            }
        )
    return output


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    cfg = ALNSConfig(PROJECT_ROOT / "config" / "alns_config.yaml")
    instances = _resolve_instances(args.instances, Path(args.instance_dir))
    seeds = _parse_ints(args.seeds)

    rows: list[BenchmarkRow] = []
    print("Running sub-route robust verification microbenchmark")
    print(
        f"instances={len(instances)} seeds={seeds} "
        f"candidates={args.candidates} gamma={args.gamma}"
    )
    for instance_path in instances:
        for seed in seeds:
            part = run_instance(
                instance_path=instance_path,
                seed=seed,
                cfg=cfg,
                target_candidates=args.candidates,
                max_attempt_factor=args.max_attempt_factor,
                gamma=args.gamma,
            )
            rows.extend(part)
            print(f"{Path(instance_path).stem} seed={seed}: candidates={len(part)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_path = output_dir / "trials.csv"
    summary_path = output_dir / "summary.csv"
    json_path = output_dir / "trials.json"

    _write_csv(trial_path, [asdict(row) for row in rows])
    _write_csv(summary_path, summarize(rows))
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(row) for row in rows], handle, indent=2)

    print(f"Trials: {trial_path}")
    print(f"Summary: {summary_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
