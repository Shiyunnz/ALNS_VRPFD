"""Batch experiment entry comparing ALNS and MILP on small instances."""

from __future__ import annotations

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
import argparse
import csv
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

try:  # Optional dependency required for MILP baseline
    import gurobipy as gp  # type: ignore
    from gurobipy import GRB  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime when Gurobi is missing
    gp = None  # type: ignore
    GRB = None  # type: ignore

from run_alns import build_operators, infer_size  # reuse CLI helpers

from alns_vrpfd.core.sa import SANNCfg, SimulatedAnnealingALNS
from alns_vrpfd.evaluation import Evaluator
from alns_vrpfd.model.initializer import (
    build_initial_solution,
    build_two_phase_initial_solution,
)
from alns_vrpfd.mip.builder import build_mip_model
from alns_vrpfd.utils.config_loader import ALNSConfig
from alns_vrpfd.utils.io_utils import read_instance


@dataclass
class RunMetrics:
    cost: float | None
    time_seconds: float | None
    status: str
    iterations: int | None = None


@dataclass
class ExperimentRecord:
    instance: str
    customers: int
    alns_cost: float | None
    alns_time: float | None
    alns_iterations: int | None
    mip_cost: float | None
    mip_time: float | None
    mip_status: str
    gap_percent: float | None


STATUS_LABELS: dict[int, str] = {}
if GRB is not None:  # pragma: no branch - simple constant mapping
    STATUS_LABELS = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
    }

EXPORT_COLUMNS = [
    "instance",
    "customers",
    "alns_cost",
    "alns_time",
    "alns_iterations",
    "mip_cost",
    "mip_time",
    "mip_status",
    "gap_percent",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ALNS and MILP in a loop over small instances and record the gap.",
    )
    parser.add_argument(
        "--instance-dir",
        type=Path,
        default=Path("data/Instance10"),
        help="Directory containing small instance files.",
    )
    parser.add_argument(
        "--instances",
        type=Path,
        nargs="*",
        help="Explicit list of instance files to run (overrides --instance-dir).",
    )
    parser.add_argument(
        "--pattern",
        default="*.txt",
        help="Filename pattern used when scanning --instance-dir.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional cap on the number of instances to process.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for the ALNS heuristic.",
    )
    parser.add_argument(
        "--alns-iterations",
        type=int,
        help="Iteration budget for ALNS (defaults follow SANNCfg heuristics).",
    )
    parser.add_argument(
        "--alns-time-limit",
        type=float,
        help="Hard time limit (seconds) for each ALNS run.",
    )
    parser.add_argument(
        "--repair-set",
        choices=["new", "legacy", "all"],
        default="all",
        help="Repair operator portfolio passed to the ALNS run.",
    )
    parser.add_argument(
        "--drone-priority",
        type=float,
        default=0.3,
        help="Drone priority weight used by repair operators.",
    )
    parser.add_argument(
        "--time-window-strategy",
        default="demand_based",
        help="Strategy passed to read_instance for time window construction.",
    )
    parser.add_argument(
        "--mip-time-limit",
        type=float,
        help="Time limit (seconds) forwarded to the MILP solver.",
    )
    parser.add_argument(
        "--mip-gap",
        type=float,
        help="Relative MIP gap tolerance forwarded to the solver.",
    )
    parser.add_argument(
        "--mip-threads",
        type=int,
        help="Thread count for the MILP solver (Threads parameter).",
    )
    parser.add_argument(
        "--mip-tardiness-weight",
        type=float,
        default=2.0,  # Tardiness weight for MIP objective
        help="Weight applied to tardiness penalties in the MILP objective.",
    )
    parser.add_argument(
        "--bigm-time",
        type=float,
        default=1000.0,
        help="Big-M constant used in MILP timing constraints.",
    )
    parser.add_argument(
        "--bigm-load",
        type=float,
        default=1000.0,
        help="Big-M constant used in MILP load constraints.",
    )
    parser.add_argument(
        "--bigm-energy",
        type=float,
        default=20.0,
        help="Big-M constant used in MILP energy constraints.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV path to export aggregated results.",
    )
    parser.add_argument(
        "--skip-mip",
        action="store_true",
        help="Skip MILP solves (useful when Gurobi is not available).",
    )
    return parser.parse_args(argv)


def collect_instances(
    provided: Sequence[Path] | None,
    directory: Path,
    pattern: str,
    limit: int | None,
) -> list[Path]:
    if provided:
        existing = [path for path in provided if path.exists()
                    and path.is_file()]
        missing = [path for path in provided if not path.exists()
                   or not path.is_file()]
        for path in missing:
            print(f"Skipping missing instance: {path}", file=sys.stderr)
        return existing[:limit] if limit is not None else existing

    if not directory.exists():
        print(f"Instance directory not found: {directory}", file=sys.stderr)
        return []

    matches = sorted(directory.glob(pattern))
    instances = [path for path in matches if path.is_file()]
    return instances[:limit] if limit is not None else instances


def run_alns(
    instance,
    *,
    seed: int,
    iterations: int | None,
    time_limit: float | None,
    drone_priority: float,
    repair_set: str,
) -> RunMetrics:
    config = ALNSConfig("config/alns_config.yaml")
    rng = random.Random(seed)

    # Align with run_alns.py defaults for manuscript experiments.
    if "drone" in instance.vehicle_specs:
        instance.vehicle_specs["drone"].endurance = float("inf")
    instance.configure_robustness(
        drone_battery_capacity=config.drone_battery_capacity,
        energy_uncertainty_budget=config.energy_uncertainty_budget,
        energy_deviation_rate=config.energy_deviation_rate,

        same_truck_retrieval=config.same_truck_retrieval,
    )
    forced_drone_customers = config.forced_drone_customers

    destroy_ops, repair_ops = build_operators(
        instance,
        seed,
        drone_priority,
        repair_set,
        enable_composite=True,
        drone_bonus_kwargs=config.drone_bonus,
        forced_drone_customers=forced_drone_customers,
        robust_energy_mode="embedded",
    )

    sa_config_dict = config.build_sa_config_dict()
    if iterations is not None:
        sa_config_dict["iterations"] = iterations
    sa_config_dict["size"] = infer_size(instance)
    cfg = SANNCfg(**sa_config_dict)

    evaluator = Evaluator(
        instance,
        rendezvous_tolerance=config.drone_rendezvous_tolerance,
        forced_drone_customers=forced_drone_customers,
        allow_multiple_launch_per_node=config.relax_allow_multiple_launch_per_node,
        cost_lambda=config.cost_lambda,
        cost_rho=config.cost_rho,
        cost_normalized=config.cost_normalized,
    )

    use_two_phase = config.raw.get("initial_solution", {}).get("two_phase", True)
    if use_two_phase:
        initial_solution = build_two_phase_initial_solution(
            instance,
            truck_forbidden_customers=forced_drone_customers,
            allow_multiple_launch_per_node=config.relax_allow_multiple_launch_per_node,
        )
    else:
        initial_solution = build_initial_solution(
            instance,
            truck_forbidden_customers=forced_drone_customers,
            allow_multiple_launch_per_node=config.relax_allow_multiple_launch_per_node,
        )
    alns = SimulatedAnnealingALNS(
        instance=instance,
        destroy_ops=destroy_ops,
        repair_ops=repair_ops,
        evaluator=evaluator,
        cfg=cfg,
        rng=rng,
    )

    start = time.perf_counter()
    best_solution = alns.run(initial_solution, time_limit=time_limit)
    elapsed = time.perf_counter() - start
    best_eval = evaluator.evaluate_solution(best_solution)
    return RunMetrics(
        cost=best_eval.total_cost,
        time_seconds=elapsed,
        status="ok",
        iterations=cfg.iterations_for(),
    )


def run_mip(
    instance,
    *,
    time_limit: float | None,
    mip_gap: float | None,
    threads: int | None,
    tardiness_weight: float,
    bigm_time: float,
    bigm_load: float,
    bigm_energy: float,
    pwl_delay_segments: int | None = None,
) -> RunMetrics:
    if gp is None or GRB is None:
        return RunMetrics(cost=None, time_seconds=None, status="gurobi_unavailable")

    try:
        from alns_vrpfd.utils.config_loader import ALNSConfig
        cfg = ALNSConfig()
        segments = cfg.piecewise_delay_segments if pwl_delay_segments is None else pwl_delay_segments
        artifacts = build_mip_model(
            instance,
            big_m_time=bigm_time,
            big_m_load=bigm_load,
            big_m_energy=bigm_energy,
            tardiness_weight=tardiness_weight,
            pwl_delay_segments=segments,
        )
    except RuntimeError as exc:  # triggered when gurobipy is missing
        return RunMetrics(cost=None, time_seconds=None, status=f"error:{exc}")

    model = artifacts.model
    if mip_gap is not None:
        model.setParam("MIPGap", float(mip_gap))
    if time_limit is not None:
        model.setParam("TimeLimit", float(time_limit))
    if threads is not None and threads > 0:
        model.setParam("Threads", int(threads))

    start = time.perf_counter()
    model.optimize()
    elapsed = time.perf_counter() - start

    status = model.Status
    status_label = STATUS_LABELS.get(status, str(status))
    solution_available = getattr(model, "SolCount", 0) > 0
    objective = model.ObjVal if solution_available else None
    return RunMetrics(cost=objective, time_seconds=elapsed, status=status_label)


def compute_gap(alns_cost: float | None, mip_cost: float | None) -> float | None:
    if alns_cost is None or mip_cost in (None, 0.0):
        return None
    return (alns_cost - mip_cost) / mip_cost * 100.0


def export_csv(records: Sequence[ExperimentRecord], destination: Path) -> None:
    if not records:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPORT_COLUMNS)
        writer.writeheader()
        for record in records:
            row = {
                key: ("" if value is None else value)
                for key, value in asdict(record).items()
            }
            writer.writerow(row)


def mean(values: Iterable[float | None]) -> float | None:
    data = [value for value in values if value is not None]
    if not data:
        return None
    return sum(data) / len(data)


def format_cost(cost: float | None) -> str:
    return f"{cost:.2f}" if cost is not None else "--"


def format_time(seconds: float | None) -> str:
    return f"{seconds:.2f}" if seconds is not None else "--"


def format_gap(gap: float | None) -> str:
    return f"{gap:.2f}%" if gap is not None else "--"


def run_batch(args: argparse.Namespace) -> list[ExperimentRecord]:
    instances = collect_instances(
        args.instances,
        args.instance_dir,
        args.pattern,
        args.limit,
    )
    if not instances:
        print("No instances found for the experiment run.", file=sys.stderr)
        return []

    header = (
        f"{'Instance':30}  {'Cust':>4}  {'ALNS Cost':>12}  {'ALNS Time':>10}  "
        f"{'MILP Cost':>12}  {'MILP Time':>10}  {'Gap%':>8}  {'MILP Status'}"
    )
    print(header)
    print("-" * len(header))

    records: list[ExperimentRecord] = []
    for instance_path in instances:
        instance = read_instance(
            str(instance_path), strategy=args.time_window_strategy)
        customers = len(instance.customer_manager.customer_ids())

        alns_metrics = run_alns(
            instance,
            seed=args.seed,
            iterations=args.alns_iterations,
            time_limit=args.alns_time_limit,
            drone_priority=args.drone_priority,
            repair_set=args.repair_set,
        )

        if args.skip_mip:
            mip_metrics = RunMetrics(
                cost=None, time_seconds=None, status="skipped")
        else:
            mip_metrics = run_mip(
                instance,
                time_limit=args.mip_time_limit,
                mip_gap=args.mip_gap,
                threads=args.mip_threads,
                tardiness_weight=args.mip_tardiness_weight,
                bigm_time=args.bigm_time,
                bigm_load=args.bigm_load,
                bigm_energy=args.bigm_energy,
                pwl_delay_segments=ALNSConfig().piecewise_delay_segments,
            )

        gap = compute_gap(alns_metrics.cost, mip_metrics.cost)
        record = ExperimentRecord(
            instance=str(instance_path),
            customers=customers,
            alns_cost=alns_metrics.cost,
            alns_time=alns_metrics.time_seconds,
            alns_iterations=alns_metrics.iterations,
            mip_cost=mip_metrics.cost,
            mip_time=mip_metrics.time_seconds,
            mip_status=mip_metrics.status,
            gap_percent=gap,
        )
        records.append(record)

        print(
            f"{instance_path.name:30}  {customers:4d}  "
            f"{format_cost(record.alns_cost):>12}  {format_time(record.alns_time):>10}  "
            f"{format_cost(record.mip_cost):>12}  {format_time(record.mip_time):>10}  "
            f"{format_gap(record.gap_percent):>8}  {record.mip_status}",
        )

    avg_gap = mean(record.gap_percent for record in records)
    avg_alns_time = mean(record.alns_time for record in records)
    avg_mip_time = mean(record.mip_time for record in records)
    print("-" * len(header))
    print(
        f"Averages: ALNS time {format_time(avg_alns_time)} s, "
        f"MILP time {format_time(avg_mip_time)} s, gap {format_gap(avg_gap)}",
    )

    return records


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    records = run_batch(args)
    if args.output and records:
        export_csv(records, args.output)
        print(f"Exported results to {args.output}")


if __name__ == "__main__":
    main()
