"""
Compare embedded robustness vs post-check verification on runtime and cost.

This script intentionally excludes scenario replay. It only measures in-search
speed and final objective value under paired seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any, Sequence
import argparse
import csv
import math
import random
import sys
import time

# Ensure project root is in sys.path before other imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
from sensitivity.instance_selector import collect_instance_paths_with_scope


DEFAULT_INSTANCE_DIRS = [Path("data/Instance10")]
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results_new" / "embedding_vs_verification"
METHODS = ("embedded", "verification")


@dataclass(frozen=True)
class TrialRow:
    instance: str
    seed: int
    gamma: int
    method: str
    runtime_sec: float
    best_cost: float
    feasible: int
    iterations_executed: int
    termination_reason: str


@dataclass(frozen=True)
class PairRow:
    instance: str
    seed: int
    gamma: int
    runtime_embedded: float
    runtime_verification: float
    best_cost_embedded: float
    best_cost_verification: float
    speedup_verification_over_embedded: float
    cost_delta_verification_minus_embedded: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark embedded robustness vs verification on speed/cost."
    )
    parser.add_argument(
        "--instance-dir",
        action="append",
        dest="instance_dirs",
        help="Instance directory (repeatable).",
    )
    parser.add_argument(
        "--instance-scope",
        type=str,
        choices=["all", "region", "single"],
        default="all",
        help="Instance selection scope.",
    )
    parser.add_argument(
        "--regions",
        type=str,
        default="30,40,50",
        help="Comma-separated regions for --instance-scope region.",
    )
    parser.add_argument(
        "--instance-name",
        type=str,
        default=None,
        help="Instance name/path for --instance-scope single.",
    )
    parser.add_argument(
        "--gamma",
        type=int,
        default=3,
        help="Fixed energy uncertainty budget used in both methods.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override ALNS iterations.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Per-run ALNS time limit in seconds.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Explicit comma-separated seeds. Overrides --trials/--seed-base.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of paired seeds when --seeds is not provided.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=20260222,
        help="Seed start for automatic paired seeds.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="embedding_vs_verification",
        help="Output filename prefix.",
    )
    return parser.parse_args()


def _parse_seed_list(seeds_text: str | None, trials: int, seed_base: int) -> list[int]:
    if seeds_text:
        values: list[int] = []
        for token in seeds_text.split(","):
            token = token.strip()
            if token:
                values.append(int(token))
        if not values:
            raise ValueError("Seed list is empty after parsing --seeds.")
        return values
    if trials <= 0:
        raise ValueError("--trials must be positive.")
    return [seed_base + i for i in range(trials)]


def _build_sa_cfg(instance, cfg_obj, iterations_override: int | None):
    from run_alns import infer_size
    from alns_vrpfd.core.sa import SANNCfg

    size = infer_size(instance)
    config_dict = cfg_obj.build_sa_config_dict(size=size)
    config_dict["iterations"] = (
        iterations_override if iterations_override is not None else cfg_obj.iterations_for(size)
    )
    return SANNCfg(**config_dict)


def _build_initial_solution(instance, cfg_obj):
    from alns_vrpfd.model.initializer import build_initial_solution, build_two_phase_initial_solution

    use_two_phase = cfg_obj.raw.get("initial_solution", {}).get("two_phase", True)
    forced = cfg_obj.forced_drone_customers
    if use_two_phase:
        return build_two_phase_initial_solution(
            instance,
            truck_forbidden_customers=forced,
            allow_multiple_launch_per_node=cfg_obj.relax_allow_multiple_launch_per_node,
        )
    return build_initial_solution(
        instance,
        truck_forbidden_customers=forced,
        allow_multiple_launch_per_node=cfg_obj.relax_allow_multiple_launch_per_node,
    )


def run_trial(
    *,
    instance_path: str,
    method: str,
    seed: int,
    gamma: int,
    cfg_obj,
    iterations_override: int | None,
    time_limit_override: float | None,
) -> TrialRow:
    from run_alns import build_operators
    from alns_vrpfd.core.sa import SimulatedAnnealingALNS
    from alns_vrpfd.evaluation import Evaluator
    from alns_vrpfd.utils.io_utils import read_instance

    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'.")

    instance = read_instance(instance_path, strategy=cfg_obj.time_window_strategy)
    if "drone" in instance.vehicle_specs:
        instance.vehicle_specs["drone"].endurance = float("inf")

    instance.configure_robustness(
        drone_battery_capacity=cfg_obj.drone_battery_capacity,
        energy_uncertainty_budget=gamma,
        energy_deviation_rate=cfg_obj.energy_deviation_rate,

        same_truck_retrieval=cfg_obj.same_truck_retrieval,
    )

    evaluator = Evaluator(
        instance,
        rendezvous_tolerance=cfg_obj.drone_rendezvous_tolerance,
        forced_drone_customers=cfg_obj.forced_drone_customers,
        allow_multiple_launch_per_node=cfg_obj.relax_allow_multiple_launch_per_node,
    )
    initial_solution = _build_initial_solution(instance, cfg_obj)

    destroy_ops, repair_ops = build_operators(
        instance=instance,
        seed=seed,
        drone_priority=cfg_obj.drone_priority,
        repair_set="all",
        enable_composite=True,
        drone_bonus_kwargs=cfg_obj.drone_bonus,
        forced_drone_customers=cfg_obj.forced_drone_customers,
        robust_energy_mode=method,
    )
    sa_cfg = _build_sa_cfg(instance, cfg_obj, iterations_override)
    alns = SimulatedAnnealingALNS(
        instance=instance,
        destroy_ops=destroy_ops,
        repair_ops=repair_ops,
        evaluator=evaluator,
        cfg=sa_cfg,
        rng=random.Random(seed),
        verbose=False,
    )

    start = time.perf_counter()
    best = alns.run(
        initial_solution,
        time_limit=time_limit_override if time_limit_override is not None else cfg_obj.time_limit,
    )
    elapsed = time.perf_counter() - start
    best_eval = evaluator.evaluate_solution(best)
    stats = getattr(alns, "last_run_stats", {})

    return TrialRow(
        instance=instance_path,
        seed=seed,
        gamma=gamma,
        method=method,
        runtime_sec=float(elapsed),
        best_cost=float(best_eval.total_cost),
        feasible=int(bool(best_eval.feasible)),
        iterations_executed=int(stats.get("executed_iterations", sa_cfg.iterations_for())),
        termination_reason=str(stats.get("termination_reason", "unknown")),
    )


def build_pair_rows(rows: Sequence[TrialRow]) -> list[PairRow]:
    keyed: dict[tuple[str, int, int], dict[str, TrialRow]] = {}
    for row in rows:
        key = (row.instance, row.seed, row.gamma)
        bucket = keyed.setdefault(key, {})
        bucket[row.method] = row

    pairs: list[PairRow] = []
    for (instance, seed, gamma), bucket in sorted(keyed.items()):
        embedded = bucket.get("embedded")
        verification = bucket.get("verification")
        if embedded is None or verification is None:
            continue
        if verification.runtime_sec <= 0:
            continue
        pairs.append(
            PairRow(
                instance=instance,
                seed=seed,
                gamma=gamma,
                runtime_embedded=embedded.runtime_sec,
                runtime_verification=verification.runtime_sec,
                best_cost_embedded=embedded.best_cost,
                best_cost_verification=verification.best_cost,
                speedup_verification_over_embedded=embedded.runtime_sec / verification.runtime_sec,
                cost_delta_verification_minus_embedded=(
                    verification.best_cost - embedded.best_cost
                ),
            )
        )
    return pairs


def summarize_pairs(pairs: Sequence[PairRow]) -> list[dict[str, Any]]:
    def _summary(scope: str, items: Sequence[PairRow]) -> dict[str, Any]:
        speedups = [p.speedup_verification_over_embedded for p in items if math.isfinite(p.speedup_verification_over_embedded)]
        cost_deltas = [p.cost_delta_verification_minus_embedded for p in items if math.isfinite(p.cost_delta_verification_minus_embedded)]
        return {
            "scope": scope,
            "pair_count": len(items),
            "mean_speedup_verification_over_embedded": fmean(speedups) if speedups else math.nan,
            "median_speedup_verification_over_embedded": median(speedups) if speedups else math.nan,
            "verification_faster_ratio": (
                sum(1 for v in speedups if v > 1.0) / len(speedups) if speedups else math.nan
            ),
            "mean_cost_delta_verification_minus_embedded": fmean(cost_deltas) if cost_deltas else math.nan,
            "median_cost_delta_verification_minus_embedded": median(cost_deltas) if cost_deltas else math.nan,
            "verification_lower_cost_ratio": (
                sum(1 for v in cost_deltas if v < 0.0) / len(cost_deltas) if cost_deltas else math.nan
            ),
        }

    result: list[dict[str, Any]] = []
    if pairs:
        result.append(_summary("overall", pairs))
    by_instance: dict[str, list[PairRow]] = {}
    for item in pairs:
        by_instance.setdefault(item.instance, []).append(item)
    for instance, items in sorted(by_instance.items()):
        result.append(_summary(instance, items))
    return result


def _write_trials(path: Path, rows: Sequence[TrialRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "instance",
                "seed",
                "gamma",
                "method",
                "runtime_sec",
                "best_cost",
                "feasible",
                "iterations_executed",
                "termination_reason",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "instance": row.instance,
                    "seed": row.seed,
                    "gamma": row.gamma,
                    "method": row.method,
                    "runtime_sec": row.runtime_sec,
                    "best_cost": row.best_cost,
                    "feasible": row.feasible,
                    "iterations_executed": row.iterations_executed,
                    "termination_reason": row.termination_reason,
                }
            )


def _write_pairs(path: Path, rows: Sequence[PairRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "instance",
                "seed",
                "gamma",
                "runtime_embedded",
                "runtime_verification",
                "best_cost_embedded",
                "best_cost_verification",
                "speedup_verification_over_embedded",
                "cost_delta_verification_minus_embedded",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "instance": row.instance,
                    "seed": row.seed,
                    "gamma": row.gamma,
                    "runtime_embedded": row.runtime_embedded,
                    "runtime_verification": row.runtime_verification,
                    "best_cost_embedded": row.best_cost_embedded,
                    "best_cost_verification": row.best_cost_verification,
                    "speedup_verification_over_embedded": row.speedup_verification_over_embedded,
                    "cost_delta_verification_minus_embedded": row.cost_delta_verification_minus_embedded,
                }
            )


def _write_summary(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scope",
                "pair_count",
                "mean_speedup_verification_over_embedded",
                "median_speedup_verification_over_embedded",
                "verification_faster_ratio",
                "mean_cost_delta_verification_minus_embedded",
                "median_cost_delta_verification_minus_embedded",
                "verification_lower_cost_ratio",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    try:
        from alns_vrpfd.utils.config_loader import ALNSConfig
    except ModuleNotFoundError as exc:
        if exc.name == "yaml":
            raise SystemExit(
                "Missing dependency 'pyyaml'. Install it before running this benchmark."
            ) from exc
        raise
    cfg_obj = ALNSConfig()

    seed_list = _parse_seed_list(args.seeds, args.trials, args.seed_base)
    instance_dirs = [Path(p) for p in args.instance_dirs] if args.instance_dirs else list(DEFAULT_INSTANCE_DIRS)
    instances = collect_instance_paths_with_scope(
        instance_dirs,
        scope=args.instance_scope,
        regions_text=args.regions,
        instance_name=args.instance_name,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[TrialRow] = []
    print("Running embedded vs verification benchmark")
    print(f"Instances: {len(instances)}, Seeds: {len(seed_list)}, Gamma: {args.gamma}")
    for instance_path in instances:
        for seed in seed_list:
            method_order = list(METHODS)
            random.Random(seed ^ 0xA5A5A5A5).shuffle(method_order)
            for method in method_order:
                row = run_trial(
                    instance_path=instance_path,
                    method=method,
                    seed=seed,
                    gamma=args.gamma,
                    cfg_obj=cfg_obj,
                    iterations_override=args.iterations,
                    time_limit_override=args.time_limit,
                )
                rows.append(row)
                print(
                    f"[{Path(instance_path).name}] seed={seed} method={method} "
                    f"runtime={row.runtime_sec:.3f}s cost={row.best_cost:.3f}"
                )

    pair_rows = build_pair_rows(rows)
    summary_rows = summarize_pairs(pair_rows)

    prefix = args.output_prefix
    trials_path = output_dir / f"{prefix}_trials.csv"
    pairs_path = output_dir / f"{prefix}_pairs.csv"
    summary_path = output_dir / f"{prefix}_summary.csv"
    _write_trials(trials_path, rows)
    _write_pairs(pairs_path, pair_rows)
    _write_summary(summary_path, summary_rows)

    print(f"Trials written: {trials_path}")
    print(f"Pairs written: {pairs_path}")
    print(f"Summary written: {summary_path}")


if __name__ == "__main__":
    main()
