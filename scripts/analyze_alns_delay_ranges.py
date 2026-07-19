"""Analyze delay ranges in current ALNS best solutions."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from alns_vrpfd.core.sa import SANNCfg, SimulatedAnnealingALNS  # noqa: E402
from alns_vrpfd.evaluation.evaluator import Evaluator  # noqa: E402
from alns_vrpfd.model.initializer import build_two_phase_initial_solution  # noqa: E402
from alns_vrpfd.utils.config_loader import ALNSConfig  # noqa: E402
from alns_vrpfd.utils.io_utils import read_instance  # noqa: E402
from run_alns import build_operators, infer_size  # noqa: E402


def _run_once(instance_name: str, seed: int, iterations: int) -> dict:
    config = ALNSConfig(str(PROJECT_ROOT / "config" / "alns_config.yaml"))
    instance_path = PROJECT_ROOT / "data" / "Instance10" / f"{instance_name}.txt"
    instance = read_instance(str(instance_path), strategy=config.time_window_strategy)
    if "drone" in instance.vehicle_specs:
        instance.vehicle_specs["drone"].endurance = float("inf")
    instance.configure_robustness(
        drone_battery_capacity=config.drone_battery_capacity,
        energy_uncertainty_budget=config.energy_uncertainty_budget,
        energy_deviation_rate=config.energy_deviation_rate,
        same_truck_retrieval=config.same_truck_retrieval,
    )

    evaluator = Evaluator(
        instance,
        rendezvous_tolerance=config.drone_rendezvous_tolerance,
        forced_drone_customers=config.forced_drone_customers,
        allow_multiple_launch_per_node=config.relax_allow_multiple_launch_per_node,
    )
    destroy_ops, repair_ops = build_operators(
        instance,
        seed,
        drone_priority=config.drone_priority,
        repair_set="all",
        enable_composite=True,
        drone_bonus_kwargs=config.drone_bonus,
        forced_drone_customers=config.forced_drone_customers,
        robust_energy_mode="embedded",
    )

    sa_config = config.build_sa_config_dict()
    sa_config["iterations"] = iterations
    sa_config["size"] = infer_size(instance)
    sa_config["log_operator_metrics"] = False
    cfg = SANNCfg(**sa_config)
    rng = random.Random(seed)
    initial = build_two_phase_initial_solution(
        instance,
        truck_forbidden_customers=config.forced_drone_customers,
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

    started = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        best = alns.run(initial)
    runtime = time.time() - started
    details = evaluator.evaluate_with_details(best)

    delays = []
    for delay in details.delay_breakdown.nodes:
        customer = evaluator._customer_lookup.get(delay.node_id)
        supply_class = customer.supply_class if customer else "water"
        delays.append({
            "node": delay.node_id,
            "supply_class": supply_class,
            "delay_hours": delay.delay,
            "arrival_time": delay.arrival_time,
            "reference_time": delay.reference_time,
            "served_by": delay.served_by,
        })

    return {
        "instance": instance_name,
        "seed": seed,
        "iterations": iterations,
        "runtime_sec": runtime,
        "feasible": details.result.feasible,
        "total_cost": details.result.total_cost,
        "truck_cost": details.result.truck_distance_cost,
        "drone_cost": details.result.drone_distance_cost,
        "delay_cost": details.result.delay_penalty,
        "positive_delay_count": len(delays),
        "max_delay_hours": max((d["delay_hours"] for d in delays), default=0.0),
        "sum_delay_hours": sum(d["delay_hours"] for d in delays),
        "delays": delays,
    }


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "count": len(values),
        "min": min(values),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def summarize(records: list[dict]) -> dict:
    positive_delays = [
        delay["delay_hours"]
        for record in records
        for delay in record["delays"]
    ]
    by_class: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for delay in record["delays"]:
            by_class[delay["supply_class"]].append(delay["delay_hours"])

    return {
        "runs": len(records),
        "feasible_runs": sum(1 for record in records if record["feasible"]),
        "runs_with_delay": sum(1 for record in records if record["positive_delay_count"] > 0),
        "positive_delay_distribution_hours": _percentiles(positive_delays),
        "max_delay_per_run_hours": _percentiles([record["max_delay_hours"] for record in records]),
        "sum_delay_per_run_hours": _percentiles([record["sum_delay_hours"] for record in records]),
        "positive_delay_count_per_run": _percentiles([record["positive_delay_count"] for record in records]),
        "positive_delay_count_by_class": dict(Counter(
            delay["supply_class"]
            for record in records
            for delay in record["delays"]
        )),
        "positive_delay_distribution_by_class_hours": {
            supply_class: _percentiles(values)
            for supply_class, values in sorted(by_class.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", nargs="+", default=[f"R_30_10_{i}" for i in range(1, 6)])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results" / "delay_range_analysis.json"))
    args = parser.parse_args()

    records = []
    for instance_name in args.instances:
        for seed in args.seeds:
            record = _run_once(instance_name, seed, args.iterations)
            records.append(record)
            print(
                f"{instance_name} seed={seed} feasible={record['feasible']} "
                f"cost={record['total_cost']:.3f} "
                f"delay_nodes={record['positive_delay_count']} "
                f"max_delay={record['max_delay_hours']:.3f}h"
            )

    payload = {
        "instances": args.instances,
        "seeds": args.seeds,
        "iterations": args.iterations,
        "summary": summarize(records),
        "records": records,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print(json.dumps(payload["summary"], indent=2))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
