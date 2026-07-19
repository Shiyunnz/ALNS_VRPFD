#!/usr/bin/env python3
"""Download/convert/evaluate Solomon instances with current ALNS defaults."""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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

from alns_vrpfd.evaluation import Evaluator
from alns_vrpfd.model.initializer import build_initial_solution, build_two_phase_initial_solution
from alns_vrpfd.utils.io_utils import read_instance
from alns_vrpfd.utils.config_loader import ALNSConfig
from alns_vrpfd.core.sa import SANNCfg, SimulatedAnnealingALNS
from run_alns import build_operators, infer_size


RAW_DIR = PROJECT_ROOT / "data" / "Solomon" / "raw"
CONVERTED_DIR = PROJECT_ROOT / "data" / "Solomon" / "converted"
RESULT_DIR = PROJECT_ROOT / "sensitivity" / "results_new" / "solomon_eval"


@dataclass
class SolomonCustomer:
    cid: int
    x: float
    y: float
    demand: float
    ready: float
    due: float
    service: float


@dataclass
class SolomonInstance:
    name: str
    vehicle_number: int
    vehicle_capacity: float
    customers: List[SolomonCustomer]


def parse_solomon(path: Path) -> SolomonInstance:
    lines = [ln.rstrip("\n") for ln in path.read_text(encoding="utf-8").splitlines()]
    name = lines[0].strip()

    vehicle_number = None
    vehicle_capacity = None
    customers: List[SolomonCustomer] = []

    i = 0
    while i < len(lines):
        token = lines[i].strip().upper()
        if token == "NUMBER     CAPACITY":
            parts = lines[i + 1].split()
            vehicle_number = int(parts[0])
            vehicle_capacity = float(parts[1])
            i += 2
            continue
        if token.startswith("CUST NO."):
            j = i + 1
            while j < len(lines):
                row = lines[j].strip()
                if not row:
                    j += 1
                    continue
                parts = row.split()
                if len(parts) < 7:
                    break
                customers.append(
                    SolomonCustomer(
                        cid=int(parts[0]),
                        x=float(parts[1]),
                        y=float(parts[2]),
                        demand=float(parts[3]),
                        ready=float(parts[4]),
                        due=float(parts[5]),
                        service=float(parts[6]),
                    )
                )
                j += 1
            break
        i += 1

    if vehicle_number is None or vehicle_capacity is None or not customers:
        raise ValueError(f"Failed to parse Solomon file: {path}")

    return SolomonInstance(
        name=name,
        vehicle_number=vehicle_number,
        vehicle_capacity=vehicle_capacity,
        customers=customers,
    )


def euclidean(a: SolomonCustomer, b: SolomonCustomer) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def convert_to_repo_format(
    src: SolomonInstance,
    dst_path: Path,
    *,
    drone_number: int,
    drone_capacity: float,
    truck_speed: float,
    drone_speed: float,
    truck_unit_cost: float,
    drone_unit_cost: float,
) -> Dict[int, tuple[float, float]]:
    """Write a repository-compatible instance and return TW map for customer ids."""
    by_id = {c.cid: c for c in src.customers}
    if 0 not in by_id:
        raise ValueError(f"{src.name}: depot(0) missing")
    depot = by_id[0]
    n_customers = len(src.customers) - 1
    end_depot_id = n_customers + 1

    # Node mapping: Solomon 0..n -> repo 0..n ; add end depot n+1.
    tw_map: Dict[int, tuple[float, float]] = {}
    for c in src.customers:
        if c.cid == 0:
            continue
        tw_map[c.cid] = (c.ready, c.due)

    nodes: Dict[int, SolomonCustomer] = {c.cid: c for c in src.customers}
    nodes[end_depot_id] = SolomonCustomer(
        cid=end_depot_id,
        x=depot.x,
        y=depot.y,
        demand=0.0,
        ready=depot.ready,
        due=depot.due,
        service=0.0,
    )

    ids = sorted(nodes.keys())
    lines: List[str] = []
    lines.append("VEHICLE INFORMATION")
    lines.append("Type\tNumber\tCapacity\tEndurance\t Speed\t Unit cost")
    lines.append(
        f"Truck\t{int(src.vehicle_number)}\t{src.vehicle_capacity:.1f}\t8.00\t{truck_speed:.1f}\t{truck_unit_cost:.2f}"
    )
    lines.append(
        f"Drone\t{int(drone_number)}\t{drone_capacity:.1f}\t1.00\t{drone_speed:.1f}\t{drone_unit_cost:.2f}"
    )
    lines.append("")

    lines.append("CUSTOMER INFORMATION")
    lines.append("Id \tX\tY\tDemand_D\tDemand_P")
    for nid in ids:
        c = nodes[nid]
        lines.append(f"{nid}\t{c.x:.2f}\t{c.y:.2f}\t{c.demand:.2f}\t0.00")
    lines.append("")

    lines.append("Distance For Drone")
    for i in ids:
        for j in ids:
            if i == j:
                continue
            d = euclidean(nodes[i], nodes[j])
            lines.append(f"{i}\t{j}\t{d:.4f}")
    lines.append("")

    lines.append("Distance For Truck")
    for i in ids:
        for j in ids:
            if i == j:
                continue
            d = euclidean(nodes[i], nodes[j])
            lines.append(f"{i}\t{j}\t{d:.4f}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tw_map


def count_drone_served_customers(solution) -> int:
    served = set()
    for t in solution.drone_tasks:
        served.update(t.customers())
    return len(served)


def apply_time_windows(instance, tw_map: Dict[int, tuple[float, float]], tw_scale: float) -> None:
    for cid, (ready, due) in tw_map.items():
        instance.customer_manager.assign_time_window(
            cid,
            optimal=float(ready) * tw_scale,
            latest=float(due) * tw_scale,
        )


def run_single(
    instance_path: Path,
    tw_map: Dict[int, tuple[float, float]],
    tw_scale: float,
    seed: int,
    iterations: int | None,
    use_two_phase: bool,
):
    cfg = ALNSConfig("config/alns_config.yaml")
    instance = read_instance(str(instance_path), strategy="demand_based", apply_time_windows=False)

    # Preserve run_alns behavior used in your main experiments.
    if "drone" in instance.vehicle_specs:
        instance.vehicle_specs["drone"].endurance = float("inf")

    apply_time_windows(instance, tw_map, tw_scale=tw_scale)
    instance.configure_robustness(
        drone_battery_capacity=cfg.drone_battery_capacity,
        energy_uncertainty_budget=cfg.energy_uncertainty_budget,
        energy_deviation_rate=cfg.energy_deviation_rate,

        same_truck_retrieval=cfg.same_truck_retrieval,
    )

    evaluator = Evaluator(
        instance,
        rendezvous_tolerance=cfg.drone_rendezvous_tolerance,
        forced_drone_customers=cfg.forced_drone_customers,
        allow_multiple_launch_per_node=cfg.relax_allow_multiple_launch_per_node,
    )

    sa_cfg_dict = cfg.build_sa_config_dict()
    if iterations is not None:
        sa_cfg_dict["iterations"] = int(iterations)
    sa_cfg_dict["size"] = infer_size(instance)
    sa_cfg = SANNCfg(**sa_cfg_dict)

    destroy_ops, repair_ops = build_operators(
        instance=instance,
        seed=seed,
        drone_priority=cfg.drone_priority,
        repair_set="all",
        enable_composite=True,
        drone_bonus_kwargs=cfg.drone_bonus,
        forced_drone_customers=cfg.forced_drone_customers,
        robust_energy_mode="embedded",
    )

    if use_two_phase:
        initial = build_two_phase_initial_solution(
            instance,
            truck_forbidden_customers=cfg.forced_drone_customers,
            allow_multiple_launch_per_node=cfg.relax_allow_multiple_launch_per_node,
        )
    else:
        initial = build_initial_solution(
            instance,
            truck_forbidden_customers=cfg.forced_drone_customers,
            allow_multiple_launch_per_node=cfg.relax_allow_multiple_launch_per_node,
        )

    alns = SimulatedAnnealingALNS(
        instance=instance,
        destroy_ops=destroy_ops,
        repair_ops=repair_ops,
        evaluator=evaluator,
        cfg=sa_cfg,
        rng=random.Random(seed),
    )

    start = time.perf_counter()
    best = alns.run(initial)
    runtime = time.perf_counter() - start
    ev = evaluator.evaluate_solution(best)

    return {
        "best_cost": ev.total_cost,
        "feasible": ev.feasible,
        "runtime_sec": runtime,
        "truck_distance_cost": ev.truck_distance_cost,
        "drone_distance_cost": ev.drone_distance_cost,
        "delay_penalty": ev.delay_penalty,
        "drone_served_customers": count_drone_served_customers(best),
        "drone_tasks": len(best.drone_tasks),
        "truck_routes": len(best.truck_routes),
    }


def write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate current ALNS defaults on Solomon instances.")
    p.add_argument("--instances", type=str, default="C101,R101,RC101")
    p.add_argument("--seed", type=int, default=20260329)
    p.add_argument("--iterations", type=int, default=None, help="Override ALNS iterations. Default uses config.")
    p.add_argument("--tw-scale", type=float, default=1.0 / 60.0, help="Scale Solomon READY/DUE to internal time unit.")
    p.add_argument("--drone-number", type=int, default=2)
    p.add_argument("--drone-capacity", type=float, default=30.0)
    p.add_argument("--truck-speed", type=float, default=35.0)
    p.add_argument("--drone-speed", type=float, default=70.0)
    p.add_argument("--truck-unit-cost", type=float, default=1.0)
    p.add_argument("--drone-unit-cost", type=float, default=0.2)
    p.add_argument(
        "--use-two-phase",
        action="store_true",
        help="Use two-phase initializer (can be very slow on Solomon 100-customer instances).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    names = [x.strip() for x in args.instances.split(",") if x.strip()]
    rows: List[Dict[str, object]] = []

    for name in names:
        raw_path = RAW_DIR / f"{name}.txt"
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw Solomon file missing: {raw_path}")
        print(f"[{name}] parsing and converting...")
        s = parse_solomon(raw_path)
        converted_path = CONVERTED_DIR / f"{name}_n{len(s.customers)-1}.txt"
        tw_map = convert_to_repo_format(
            s,
            converted_path,
            drone_number=args.drone_number,
            drone_capacity=args.drone_capacity,
            truck_speed=args.truck_speed,
            drone_speed=args.drone_speed,
            truck_unit_cost=args.truck_unit_cost,
            drone_unit_cost=args.drone_unit_cost,
        )

        print(f"[{name}] running ALNS (seed={args.seed}, iterations={args.iterations or 'config-default'}) ...")
        metrics = run_single(
            instance_path=converted_path,
            tw_map=tw_map,
            tw_scale=args.tw_scale,
            seed=args.seed,
            iterations=args.iterations,
            use_two_phase=args.use_two_phase,
        )
        row: Dict[str, object] = {
            "instance": name,
            "raw_file": str(raw_path),
            "converted_file": str(converted_path),
            "seed": args.seed,
            "iterations": args.iterations if args.iterations is not None else "config-default",
            "tw_scale": args.tw_scale,
            "truck_num": s.vehicle_number,
            "truck_capacity": s.vehicle_capacity,
            "drone_num": args.drone_number,
            "drone_capacity": args.drone_capacity,
            **metrics,
        }
        rows.append(row)
        print(
            f"[{name}] done: cost={metrics['best_cost']:.3f}, "
            f"drone_customers={metrics['drone_served_customers']}, runtime={metrics['runtime_sec']:.1f}s"
        )

    out_csv = RESULT_DIR / "solomon_c101_r101_rc101_results.csv"
    write_rows(out_csv, rows)
    print(f"\nSaved results: {out_csv}")


if __name__ == "__main__":
    main()
