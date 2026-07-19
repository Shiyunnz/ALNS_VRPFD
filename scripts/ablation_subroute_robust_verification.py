#!/usr/bin/env python3
"""Small ablation for ALNS sub-route robust verification.

The script is intentionally lightweight. It compares three search-time gates:

1. no_check: deterministic ALNS; robust feasibility is checked only at the end.
2. full_check: deterministic ALNS; every finite candidate is checked by the
   full robust Evaluator.
3. full_candidate_check: deterministic ALNS; every candidate is filtered by
   the full robust Evaluator at the same gate used by subroute_check.
4. subroute_check: deterministic ALNS; every candidate is filtered by
   SubrouteRobustVerifier.verify_candidate() before the full Evaluator.

Default settings run a short Instance10 smoke experiment suitable for revision
analysis without long solver runs.
"""

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

from alns_vrpfd.core.sa import SANNCfg, SimulatedAnnealingALNS
from alns_vrpfd.evaluation import Evaluator, SubrouteRobustVerifier
from alns_vrpfd.model.initializer import (
    build_initial_solution,
    build_two_phase_initial_solution,
)
from alns_vrpfd.utils.config_loader import ALNSConfig
from alns_vrpfd.utils.io_utils import read_instance
from run_alns import build_operators, infer_size


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "ablation_subroute_robust_verification"
DEFAULT_INSTANCE_DIR = PROJECT_ROOT / "data" / "Instance10"
METHODS = ("no_check", "full_check", "full_candidate_check", "subroute_check")


class InstrumentedSubrouteVerifier(SubrouteRobustVerifier):
    """Sub-route verifier with counters for ablation reporting."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.calls = 0
        self.rejections = 0
        self.checked_drone_tasks = 0
        self.failed_drone_tasks = 0
        self.changed_drone_tasks = 0
        self.changed_truck_routes = 0
        self.elapsed_sec = 0.0

    def verify_candidate(self, *, base, candidate) -> bool:
        start = time.perf_counter()
        ok = super().verify_candidate(base=base, candidate=candidate)
        self.elapsed_sec += time.perf_counter() - start
        summary = self.last_summary
        self.calls += 1
        self.changed_truck_routes += summary.changed_truck_routes
        self.changed_drone_tasks += summary.changed_drone_tasks
        self.checked_drone_tasks += summary.checked_drone_tasks
        self.failed_drone_tasks += summary.failed_drone_tasks
        if not ok:
            self.rejections += 1
        return ok

    def verify_all_tasks(self, solution) -> bool:
        start = time.perf_counter()
        ok = super().verify_all_tasks(solution)
        self.elapsed_sec += time.perf_counter() - start
        summary = self.last_summary
        self.calls += 1
        self.changed_drone_tasks += summary.changed_drone_tasks
        self.checked_drone_tasks += summary.checked_drone_tasks
        self.failed_drone_tasks += summary.failed_drone_tasks
        if not ok:
            self.rejections += 1
        return ok


class InstrumentedFullRobustEvaluator:
    """Proxy that times full robust Evaluator.evaluate_solution() calls."""

    def __init__(self, evaluator: Evaluator) -> None:
        self._evaluator = evaluator
        self.calls = 0
        self.rejections = 0
        self.elapsed_sec = 0.0

    def evaluate_solution(self, solution):
        start = time.perf_counter()
        result = self._evaluator.evaluate_solution(solution)
        self.elapsed_sec += time.perf_counter() - start
        self.calls += 1
        if not math.isfinite(result.total_cost):
            self.rejections += 1
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._evaluator, name)


class InstrumentedFullCandidateVerifier:
    """Full robust evaluator used at the same candidate gate as subroute_check."""

    def __init__(self, evaluator: Evaluator) -> None:
        self._evaluator = evaluator
        self.calls = 0
        self.rejections = 0
        self.elapsed_sec = 0.0
        self.last_summary = None

    def verify_candidate(self, *, base, candidate) -> bool:
        del base
        start = time.perf_counter()
        result = self._evaluator.evaluate_solution(candidate)
        self.elapsed_sec += time.perf_counter() - start
        self.calls += 1
        ok = math.isfinite(result.total_cost)
        if not ok:
            self.rejections += 1
        return ok


@dataclass(frozen=True)
class TrialRow:
    instance: str
    seed: int
    method: str
    gamma: int
    iterations: int
    runtime_sec: float
    search_cost: float
    search_feasible: int
    final_robust_cost: float
    final_robust_feasible: int
    executed_iterations: int
    termination_reason: str
    subroute_calls: int
    subroute_rejections: int
    subroute_checked_tasks: int
    subroute_failed_tasks: int
    subroute_avg_checked_tasks: float
    subroute_time_sec: float
    full_check_calls: int
    full_check_rejections: int
    full_check_time_sec: float
    full_candidate_calls: int
    full_candidate_rejections: int
    full_candidate_time_sec: float
    robust_cache_hits: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small ablation for ALNS sub-route robust verification."
    )
    parser.add_argument(
        "--instances",
        type=str,
        default="R_30_10_1,R_30_10_2,R_30_10_5",
        help="Comma-separated instance stems or .txt paths.",
    )
    parser.add_argument(
        "--instance-dir",
        type=str,
        default=str(DEFAULT_INSTANCE_DIR),
        help="Directory used for instance stems.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43",
        help="Comma-separated random seeds.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="Short ALNS iteration budget per run.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=60.0,
        help="Short per-run time limit in seconds.",
    )
    parser.add_argument(
        "--gamma",
        type=int,
        default=3,
        help="Robust energy uncertainty budget for final and gate checks.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="no_check,full_check,subroute_check",
        help=f"Comma-separated methods from {METHODS}.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for CSV/JSON results.",
    )
    return parser.parse_args()


def _parse_csv_ints(text: str) -> list[int]:
    values = [int(token.strip()) for token in text.split(",") if token.strip()]
    if not values:
        raise ValueError("Seed list cannot be empty.")
    return values


def _parse_methods(text: str) -> list[str]:
    methods = [token.strip() for token in text.split(",") if token.strip()]
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}. Allowed: {METHODS}")
    if not methods:
        raise ValueError("Method list cannot be empty.")
    return methods


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
    if not resolved:
        raise ValueError("No instances selected.")
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


def _build_initial_solution(instance, cfg: ALNSConfig):
    use_two_phase = cfg.raw.get("initial_solution", {}).get("two_phase", True)
    if use_two_phase:
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


def _build_sa_cfg(instance, cfg: ALNSConfig, iterations: int) -> SANNCfg:
    size = infer_size(instance)
    raw = cfg.build_sa_config_dict(size=size, iterations=iterations)
    raw["size"] = size
    raw["iterations"] = iterations
    return SANNCfg(**raw)


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


def run_trial(
    *,
    instance_path: str,
    seed: int,
    method: str,
    gamma: int,
    iterations: int,
    time_limit: float,
    cfg: ALNSConfig,
) -> TrialRow:
    search_instance = read_instance(instance_path, strategy=cfg.time_window_strategy)
    robust_instance = read_instance(instance_path, strategy=cfg.time_window_strategy)

    # Keep the search deterministic for all three methods; this isolates the
    # role of candidate robust verification instead of changing the objective.
    _configure_instance(search_instance, cfg, gamma=0)
    _configure_instance(robust_instance, cfg, gamma=gamma)

    search_evaluator = _make_evaluator(search_instance, cfg)
    robust_evaluator = _make_evaluator(robust_instance, cfg)
    initial_solution = _build_initial_solution(search_instance, cfg)

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

    subroute_verifier = None
    full_verifier = None
    full_candidate_verifier = None
    robust_check_every = 0
    if method == "subroute_check":
        subroute_verifier = InstrumentedSubrouteVerifier(
            instance=search_instance,
            drone_energy_capacity=robust_instance.robust_config.drone_battery_capacity,
            energy_uncertainty_budget=robust_instance.robust_config.energy_uncertainty_budget,
            energy_deviation_rate=robust_instance.robust_config.energy_deviation_rate,
        )
    elif method == "full_check":
        full_verifier = InstrumentedFullRobustEvaluator(robust_evaluator)
        robust_check_every = 1
    elif method == "full_candidate_check":
        full_candidate_verifier = InstrumentedFullCandidateVerifier(robust_evaluator)
    elif method != "no_check":
        raise ValueError(f"Unknown method: {method}")

    alns = SimulatedAnnealingALNS(
        instance=search_instance,
        destroy_ops=destroy_ops,
        repair_ops=repair_ops,
        evaluator=search_evaluator,
        cfg=_build_sa_cfg(search_instance, cfg, iterations),
        rng=random.Random(seed),
        verbose=False,
        robust_verifier=full_verifier,
        robust_check_every=robust_check_every,
        robust_check_on_new_best=False,
        candidate_subroute_verifier=subroute_verifier or full_candidate_verifier,
    )

    start = time.perf_counter()
    best = alns.run(initial_solution, time_limit=time_limit)
    runtime = time.perf_counter() - start

    search_eval = search_evaluator.evaluate_solution(best)
    robust_eval = robust_evaluator.evaluate_solution(best)
    stats = getattr(alns, "last_run_stats", {})

    subroute_calls = subroute_verifier.calls if subroute_verifier else 0
    checked_tasks = subroute_verifier.checked_drone_tasks if subroute_verifier else 0
    avg_checked = checked_tasks / subroute_calls if subroute_calls else 0.0

    return TrialRow(
        instance=Path(instance_path).stem,
        seed=seed,
        method=method,
        gamma=gamma,
        iterations=iterations,
        runtime_sec=runtime,
        search_cost=float(search_eval.total_cost),
        search_feasible=int(bool(search_eval.feasible)),
        final_robust_cost=float(robust_eval.total_cost),
        final_robust_feasible=int(bool(robust_eval.feasible)),
        executed_iterations=int(stats.get("executed_iterations", 0)),
        termination_reason=str(stats.get("termination_reason", "unknown")),
        subroute_calls=subroute_calls,
        subroute_rejections=subroute_verifier.rejections if subroute_verifier else 0,
        subroute_checked_tasks=checked_tasks,
        subroute_failed_tasks=subroute_verifier.failed_drone_tasks if subroute_verifier else 0,
        subroute_avg_checked_tasks=avg_checked,
        subroute_time_sec=subroute_verifier.elapsed_sec if subroute_verifier else 0.0,
        full_check_calls=full_verifier.calls if full_verifier else 0,
        full_check_rejections=full_verifier.rejections if full_verifier else 0,
        full_check_time_sec=full_verifier.elapsed_sec if full_verifier else 0.0,
        full_candidate_calls=full_candidate_verifier.calls if full_candidate_verifier else 0,
        full_candidate_rejections=(
            full_candidate_verifier.rejections if full_candidate_verifier else 0
        ),
        full_candidate_time_sec=(
            full_candidate_verifier.elapsed_sec if full_candidate_verifier else 0.0
        ),
        robust_cache_hits=int(stats.get("robust_cache_hits", 0)),
    )


def summarize(rows: Sequence[TrialRow]) -> list[dict[str, Any]]:
    by_method: dict[str, list[TrialRow]] = {}
    for row in rows:
        by_method.setdefault(row.method, []).append(row)

    summaries: list[dict[str, Any]] = []
    for method, items in sorted(by_method.items()):
        runtimes = [row.runtime_sec for row in items]
        robust_costs = [
            row.final_robust_cost
            for row in items
            if math.isfinite(row.final_robust_cost)
        ]
        summaries.append(
            {
                "method": method,
                "runs": len(items),
                "mean_runtime_sec": fmean(runtimes) if runtimes else math.nan,
                "median_runtime_sec": median(runtimes) if runtimes else math.nan,
                "robust_feasible_ratio": (
                    sum(row.final_robust_feasible for row in items) / len(items)
                    if items
                    else math.nan
                ),
                "mean_robust_cost": fmean(robust_costs) if robust_costs else math.nan,
                "median_robust_cost": median(robust_costs) if robust_costs else math.nan,
                "total_subroute_calls": sum(row.subroute_calls for row in items),
                "total_subroute_rejections": sum(row.subroute_rejections for row in items),
                "mean_subroute_avg_checked_tasks": fmean(
                    [row.subroute_avg_checked_tasks for row in items]
                ),
                "total_subroute_time_sec": sum(row.subroute_time_sec for row in items),
                "total_full_check_calls": sum(row.full_check_calls for row in items),
                "total_full_check_rejections": sum(row.full_check_rejections for row in items),
                "total_full_check_time_sec": sum(row.full_check_time_sec for row in items),
                "total_full_candidate_calls": sum(
                    row.full_candidate_calls for row in items
                ),
                "total_full_candidate_rejections": sum(
                    row.full_candidate_rejections for row in items
                ),
                "total_full_candidate_time_sec": sum(
                    row.full_candidate_time_sec for row in items
                ),
            }
        )
    return summaries


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
    seeds = _parse_csv_ints(args.seeds)
    methods = _parse_methods(args.methods)

    rows: list[TrialRow] = []
    print("Running sub-route robust verification ablation")
    print(
        f"instances={len(instances)} seeds={seeds} methods={methods} "
        f"iterations={args.iterations} gamma={args.gamma}"
    )

    for instance_path in instances:
        for seed in seeds:
            method_order = list(methods)
            random.Random(seed ^ 0x515151).shuffle(method_order)
            for method in method_order:
                row = run_trial(
                    instance_path=instance_path,
                    seed=seed,
                    method=method,
                    gamma=args.gamma,
                    iterations=args.iterations,
                    time_limit=args.time_limit,
                    cfg=cfg,
                )
                rows.append(row)
                print(
                    f"{row.instance} seed={seed} method={method} "
                    f"t={row.runtime_sec:.2f}s robust_feasible={row.final_robust_feasible} "
                    f"robust_cost={row.final_robust_cost:.2f} "
                    f"subrej={row.subroute_rejections} fullcalls={row.full_check_calls}"
                )

    summary_rows = summarize(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_path = output_dir / "trials.csv"
    summary_path = output_dir / "summary.csv"
    json_path = output_dir / "trials.json"

    _write_csv(trial_path, [asdict(row) for row in rows])
    _write_csv(summary_path, summary_rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(row) for row in rows], handle, indent=2)

    print(f"Trials: {trial_path}")
    print(f"Summary: {summary_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
