"""
Compare robust-injection strategies on runtime and robust cost.

Strategies:
1. embedded:
   Robustness is embedded in repair + search-time evaluator (gamma).
2. precheck_guarded:
   Deterministic search (gamma=0) + changed-subroute robust pre-check +
   exact robust check only when candidate is about to become new best.
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
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results_new" / "verification_strategy_compare"


@dataclass(frozen=True)
class MethodProfile:
    name: str
    repair_mode: str  # "embedded" or "verification"
    search_uses_robust_gamma: bool
    periodic_robust_check: bool = False
    robust_check_on_new_best: bool = False
    use_subroute_delta_verifier: bool = False
    use_robust_route_pool: bool = False


METHOD_PROFILES: dict[str, MethodProfile] = {
    "embedded": MethodProfile(
        name="embedded",
        repair_mode="embedded",
        search_uses_robust_gamma=True,
        periodic_robust_check=False,
        robust_check_on_new_best=False,
    ),
    "precheck_guarded": MethodProfile(
        name="precheck_guarded",
        repair_mode="verification",
        search_uses_robust_gamma=False,
        periodic_robust_check=False,
        robust_check_on_new_best=True,
        use_subroute_delta_verifier=True,
        use_robust_route_pool=False,
    ),
}


@dataclass(frozen=True)
class TrialRow:
    instance: str
    seed: int
    gamma: int
    method: str
    runtime_sec: float
    final_check_sec: float
    search_cost: float
    search_feasible: int
    robust_cost: float
    robust_feasible: int
    iterations_executed: int
    termination_reason: str


@dataclass(frozen=True)
class PairRow:
    instance: str
    seed: int
    gamma: int
    method_a: str
    method_b: str
    runtime_a: float
    runtime_b: float
    robust_cost_a: float
    robust_cost_b: float
    robust_feasible_a: int
    robust_feasible_b: int
    speedup_b_over_a: float
    robust_cost_delta_b_minus_a: float
    b_faster: int
    b_robust_noninferior: int
    b_effective_acceleration: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare embedded vs precheck_guarded strategies."
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
        help="Fixed robust energy uncertainty budget used for final robust check.",
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
        default=5,
        help="Number of paired seeds when --seeds is not provided.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=20260222,
        help="Seed start for automatic paired seeds.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="embedded,precheck_guarded",
        help=(
            "Comma-separated methods from: embedded, precheck_guarded"
        ),
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
        default="instance10_embedded_vs_precheck_guarded",
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


def _parse_methods(methods_text: str) -> list[str]:
    values = [token.strip() for token in methods_text.split(",") if token.strip()]
    if not values:
        raise ValueError("--methods cannot be empty.")
    unknown = [m for m in values if m not in METHOD_PROFILES]
    if unknown:
        raise ValueError(f"Unknown methods in --methods: {unknown}")
    return values


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


def _configure_instance_robustness(instance, cfg_obj, gamma: int) -> None:
    instance.configure_robustness(
        drone_battery_capacity=cfg_obj.drone_battery_capacity,
        energy_uncertainty_budget=gamma,
        energy_deviation_rate=cfg_obj.energy_deviation_rate,

        same_truck_retrieval=cfg_obj.same_truck_retrieval,
    )


def _is_robust_noninferior(a: TrialRow, b: TrialRow, tol: float = 1e-9) -> bool:
    if b.robust_feasible > a.robust_feasible:
        return True
    if b.robust_feasible < a.robust_feasible:
        return False
    if math.isfinite(a.robust_cost) and math.isfinite(b.robust_cost):
        return b.robust_cost <= a.robust_cost + tol
    if (not math.isfinite(a.robust_cost)) and math.isfinite(b.robust_cost):
        return True
    if math.isfinite(a.robust_cost) and (not math.isfinite(b.robust_cost)):
        return False
    return True


def run_trial(
    *,
    instance_path: str,
    method: str,
    profile: MethodProfile,
    seed: int,
    gamma: int,
    cfg_obj,
    iterations_override: int | None,
    time_limit_override: float | None,
) -> TrialRow:
    from run_alns import build_operators
    from alns_vrpfd.core.operators import RepairDronePriorityRegret
    from alns_vrpfd.core.sa import SimulatedAnnealingALNS
    from alns_vrpfd.evaluation import Evaluator, SubrouteRobustVerifier
    from alns_vrpfd.utils.io_utils import read_instance

    instance = read_instance(instance_path, strategy=cfg_obj.time_window_strategy)
    robust_instance = read_instance(instance_path, strategy=cfg_obj.time_window_strategy)
    if "drone" in instance.vehicle_specs:
        instance.vehicle_specs["drone"].endurance = float("inf")
    if "drone" in robust_instance.vehicle_specs:
        robust_instance.vehicle_specs["drone"].endurance = float("inf")

    search_gamma = gamma if profile.search_uses_robust_gamma else 0
    _configure_instance_robustness(instance, cfg_obj, search_gamma)
    _configure_instance_robustness(robust_instance, cfg_obj, gamma)

    search_evaluator = Evaluator(
        instance,
        rendezvous_tolerance=cfg_obj.drone_rendezvous_tolerance,
        forced_drone_customers=cfg_obj.forced_drone_customers,
        allow_multiple_launch_per_node=cfg_obj.relax_allow_multiple_launch_per_node,
    )
    robust_evaluator = Evaluator(
        robust_instance,
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
        robust_energy_mode=profile.repair_mode,
    )
    if method == "precheck_guarded":
        _, embedded_repairs = build_operators(
            instance=instance,
            seed=seed + 7919,
            drone_priority=cfg_obj.drone_priority,
            repair_set="all",
            enable_composite=True,
            drone_bonus_kwargs=cfg_obj.drone_bonus,
            forced_drone_customers=cfg_obj.forced_drone_customers,
            robust_energy_mode="embedded",
        )
        embedded_regret = next(
            (
                op
                for op in embedded_repairs
                if isinstance(op, RepairDronePriorityRegret)
            ),
            None,
        )
        if embedded_regret is not None:
            repair_ops = list(repair_ops) + [embedded_regret]
    sa_cfg = _build_sa_cfg(instance, cfg_obj, iterations_override)
    subroute_verifier = (
        SubrouteRobustVerifier(
            instance=instance,
            drone_energy_capacity=robust_instance.robust_config.drone_battery_capacity,
            energy_uncertainty_budget=robust_instance.robust_config.energy_uncertainty_budget,
            energy_deviation_rate=robust_instance.robust_config.energy_deviation_rate,
        )
        if profile.use_subroute_delta_verifier
        else None
    )
    alns = SimulatedAnnealingALNS(
        instance=instance,
        destroy_ops=destroy_ops,
        repair_ops=repair_ops,
        evaluator=search_evaluator,
        cfg=sa_cfg,
        rng=random.Random(seed),
        verbose=False,
        robust_verifier=(
            robust_evaluator if profile.robust_check_on_new_best else None
        ),
        robust_check_every=0,
        robust_check_on_new_best=profile.robust_check_on_new_best,
        candidate_subroute_verifier=subroute_verifier,
        conservative_cost_evaluator=robust_evaluator if profile.use_robust_route_pool else None,
        collect_robust_route_pool=profile.use_robust_route_pool,
    )

    start = time.perf_counter()
    best = alns.run(
        initial_solution,
        time_limit=time_limit_override if time_limit_override is not None else cfg_obj.time_limit,
    )
    elapsed = time.perf_counter() - start

    search_eval = search_evaluator.evaluate_solution(best)

    # Final robust check at requested gamma for all methods (unified comparison target).
    check_start = time.perf_counter()
    robust_eval = robust_evaluator.evaluate_solution(best)
    check_elapsed = time.perf_counter() - check_start

    stats = getattr(alns, "last_run_stats", {})
    return TrialRow(
        instance=instance_path,
        seed=seed,
        gamma=gamma,
        method=method,
        runtime_sec=float(elapsed),
        final_check_sec=float(check_elapsed),
        search_cost=float(search_eval.total_cost),
        search_feasible=int(bool(search_eval.feasible)),
        robust_cost=float(robust_eval.total_cost),
        robust_feasible=int(bool(robust_eval.feasible)),
        iterations_executed=int(stats.get("executed_iterations", sa_cfg.iterations_for())),
        termination_reason=str(stats.get("termination_reason", "unknown")),
    )


def build_pair_rows(rows: Sequence[TrialRow], methods: Sequence[str]) -> list[PairRow]:
    keyed: dict[tuple[str, int, int], dict[str, TrialRow]] = {}
    for row in rows:
        keyed.setdefault((row.instance, row.seed, row.gamma), {})[row.method] = row

    candidate_pairs: list[tuple[str, str]] = []
    preferred = [("embedded", "precheck_guarded")]
    for a, b in preferred:
        if a in methods and b in methods:
            candidate_pairs.append((a, b))
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            pair = (a, b)
            if pair not in candidate_pairs:
                candidate_pairs.append(pair)

    pairs: list[PairRow] = []
    for (instance, seed, gamma), bucket in sorted(keyed.items()):
        for method_a, method_b in candidate_pairs:
            row_a = bucket.get(method_a)
            row_b = bucket.get(method_b)
            if row_a is None or row_b is None:
                continue
            if row_b.runtime_sec <= 0:
                continue
            cost_delta = row_b.robust_cost - row_a.robust_cost
            b_faster = int(row_b.runtime_sec < row_a.runtime_sec)
            b_noninferior = int(_is_robust_noninferior(row_a, row_b))
            pairs.append(
                PairRow(
                    instance=instance,
                    seed=seed,
                    gamma=gamma,
                    method_a=method_a,
                    method_b=method_b,
                    runtime_a=row_a.runtime_sec,
                    runtime_b=row_b.runtime_sec,
                    robust_cost_a=row_a.robust_cost,
                    robust_cost_b=row_b.robust_cost,
                    robust_feasible_a=row_a.robust_feasible,
                    robust_feasible_b=row_b.robust_feasible,
                    speedup_b_over_a=row_a.runtime_sec / row_b.runtime_sec,
                    robust_cost_delta_b_minus_a=cost_delta,
                    b_faster=b_faster,
                    b_robust_noninferior=b_noninferior,
                    b_effective_acceleration=int(b_faster and b_noninferior),
                )
            )
    return pairs


def summarize_methods(rows: Sequence[TrialRow]) -> list[dict[str, Any]]:
    by_method: dict[str, list[TrialRow]] = {}
    for row in rows:
        by_method.setdefault(row.method, []).append(row)

    result: list[dict[str, Any]] = []
    for method, items in sorted(by_method.items()):
        runtimes = [x.runtime_sec for x in items if math.isfinite(x.runtime_sec)]
        robust_costs = [x.robust_cost for x in items if math.isfinite(x.robust_cost)]
        search_costs = [x.search_cost for x in items if math.isfinite(x.search_cost)]
        checks = [x.final_check_sec for x in items if math.isfinite(x.final_check_sec)]
        result.append(
            {
                "method": method,
                "trial_count": len(items),
                "mean_runtime_sec": fmean(runtimes) if runtimes else math.nan,
                "median_runtime_sec": median(runtimes) if runtimes else math.nan,
                "mean_final_check_sec": fmean(checks) if checks else math.nan,
                "mean_search_cost_finite": fmean(search_costs) if search_costs else math.nan,
                "median_search_cost_finite": median(search_costs) if search_costs else math.nan,
                "mean_robust_cost_finite": fmean(robust_costs) if robust_costs else math.nan,
                "median_robust_cost_finite": median(robust_costs) if robust_costs else math.nan,
                "robust_feasible_ratio": (
                    sum(1 for x in items if x.robust_feasible > 0) / len(items) if items else math.nan
                ),
            }
        )
    return result


def summarize_pairs(pairs: Sequence[PairRow]) -> list[dict[str, Any]]:
    by_pair: dict[tuple[str, str], list[PairRow]] = {}
    for item in pairs:
        by_pair.setdefault((item.method_a, item.method_b), []).append(item)

    result: list[dict[str, Any]] = []
    for (method_a, method_b), items in sorted(by_pair.items()):
        speedups = [x.speedup_b_over_a for x in items if math.isfinite(x.speedup_b_over_a)]
        cost_deltas = [x.robust_cost_delta_b_minus_a for x in items if math.isfinite(x.robust_cost_delta_b_minus_a)]
        result.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "pair_count": len(items),
                "mean_speedup_b_over_a": fmean(speedups) if speedups else math.nan,
                "median_speedup_b_over_a": median(speedups) if speedups else math.nan,
                "b_faster_ratio": (
                    sum(1 for x in items if x.b_faster > 0) / len(items) if items else math.nan
                ),
                "mean_robust_cost_delta_b_minus_a": fmean(cost_deltas) if cost_deltas else math.nan,
                "median_robust_cost_delta_b_minus_a": median(cost_deltas) if cost_deltas else math.nan,
                "b_robust_noninferior_ratio": (
                    sum(1 for x in items if x.b_robust_noninferior > 0) / len(items) if items else math.nan
                ),
                "b_effective_acceleration_ratio": (
                    sum(1 for x in items if x.b_effective_acceleration > 0) / len(items) if items else math.nan
                ),
            }
        )
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
                "final_check_sec",
                "search_cost",
                "search_feasible",
                "robust_cost",
                "robust_feasible",
                "iterations_executed",
                "termination_reason",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _write_pairs(path: Path, rows: Sequence[PairRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "instance",
                "seed",
                "gamma",
                "method_a",
                "method_b",
                "runtime_a",
                "runtime_b",
                "robust_cost_a",
                "robust_cost_b",
                "robust_feasible_a",
                "robust_feasible_b",
                "speedup_b_over_a",
                "robust_cost_delta_b_minus_a",
                "b_faster",
                "b_robust_noninferior",
                "b_effective_acceleration",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _write_summary(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
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

    methods = _parse_methods(args.methods)
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
    print("Running verification-strategy benchmark")
    print(
        f"Instances: {len(instances)}, Seeds: {len(seed_list)}, Gamma: {args.gamma}, Methods: {methods}"
    )
    for instance_path in instances:
        for seed in seed_list:
            method_order = list(methods)
            random.Random(seed ^ 0xABCD1357).shuffle(method_order)
            for method in method_order:
                profile = METHOD_PROFILES[method]
                row = run_trial(
                    instance_path=instance_path,
                    method=method,
                    profile=profile,
                    seed=seed,
                    gamma=args.gamma,
                    cfg_obj=cfg_obj,
                    iterations_override=args.iterations,
                    time_limit_override=args.time_limit,
                )
                rows.append(row)
                print(
                    f"[{Path(instance_path).name}] seed={seed} method={method} "
                    f"runtime={row.runtime_sec:.3f}s robust_cost={row.robust_cost:.3f} robust_feasible={row.robust_feasible}"
                )

    pair_rows = build_pair_rows(rows, methods)
    method_summary = summarize_methods(rows)
    pair_summary = summarize_pairs(pair_rows)

    prefix = args.output_prefix
    trials_path = output_dir / f"{prefix}_trials.csv"
    pairs_path = output_dir / f"{prefix}_pairs.csv"
    method_summary_path = output_dir / f"{prefix}_method_summary.csv"
    pair_summary_path = output_dir / f"{prefix}_pair_summary.csv"
    _write_trials(trials_path, rows)
    _write_pairs(pairs_path, pair_rows)
    _write_summary(
        method_summary_path,
        method_summary,
        fieldnames=[
            "method",
            "trial_count",
            "mean_runtime_sec",
            "median_runtime_sec",
            "mean_final_check_sec",
            "mean_search_cost_finite",
            "median_search_cost_finite",
            "mean_robust_cost_finite",
            "median_robust_cost_finite",
            "robust_feasible_ratio",
        ],
    )
    _write_summary(
        pair_summary_path,
        pair_summary,
        fieldnames=[
            "method_a",
            "method_b",
            "pair_count",
            "mean_speedup_b_over_a",
            "median_speedup_b_over_a",
            "b_faster_ratio",
            "mean_robust_cost_delta_b_minus_a",
            "median_robust_cost_delta_b_minus_a",
            "b_robust_noninferior_ratio",
            "b_effective_acceleration_ratio",
        ],
    )

    print(f"Trials written: {trials_path}")
    print(f"Pairs written: {pairs_path}")
    print(f"Method summary written: {method_summary_path}")
    print(f"Pair summary written: {pair_summary_path}")


if __name__ == "__main__":
    main()
