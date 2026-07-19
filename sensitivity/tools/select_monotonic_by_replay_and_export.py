"""Select monotonic-by-replay solutions and export replay/table/plot artifacts.

For each instance, choose one trial solution per gamma (0/1/2/3) so that
for each replay distribution ND/UD/NDC:
    avg_cost(g0) <= avg_cost(g1) <= avg_cost(g2) <= avg_cost(g3)
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd

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

from alns_vrpfd.evaluation import (
    GammaSolutionInput,
    ScenarioDistributionConfig,
    ScenarioReplayConfig,
    run_scenario_replay,
)
from alns_vrpfd.model.route import DroneTask, TruckRoute
from alns_vrpfd.model.solution import Solution
from alns_vrpfd.utils.io_utils import read_instance


GAMMAS = [0, 1, 2, 3]
DISTS = ["ND", "UD", "NDC"]
ALL_REPLAY_METRICS = [
    "avg_cost",
    "std_cost",
    "max_cost",
    "min_cost",
    "avg_unserved",
    "p0_all_served",
    "avg_no_takeoff",
    "avg_abort_return",
]


@dataclass
class Candidate:
    instance_name: str
    gamma: int
    seed: int
    row: Dict[str, Any]
    replay_summary_by_dist: Dict[str, Dict[str, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select monotonic-by-replay solution bank and export outputs."
    )
    parser.add_argument(
        "--trial-dir",
        type=str,
        default="sensitivity/results_new/scenario_replay",
        help="Directory containing R_*_trials.csv.",
    )
    parser.add_argument(
        "--trial-glob",
        type=str,
        default="R_*_25_*_trials.csv",
        help="Glob for trial files.",
    )
    parser.add_argument(
        "--scenario-count",
        type=int,
        default=1000,
        help="Replay scenario count.",
    )
    parser.add_argument(
        "--replay-seed",
        type=int,
        default=2024,
        help="Replay random seed.",
    )
    parser.add_argument(
        "--energy-deviation-rate",
        type=float,
        default=0.2,
        help="Theta for replay evaluator.",
    )
    parser.add_argument("--nd-cv", type=float, default=0.22)
    parser.add_argument("--ud-delta", type=float, default=0.22)
    parser.add_argument("--ndc-cv", type=float, default=0.22)
    parser.add_argument("--ndc-correlation", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sensitivity/results_new/scenario_replay_monotonic_replay_selected",
    )
    parser.add_argument(
        "--sync-to-scenario-replay",
        action="store_true",
        help="Also copy selected outputs back to sensitivity/results_new/scenario_replay.",
    )
    parser.add_argument(
        "--strict-monotone",
        action="store_true",
        help="Fail if any instance cannot satisfy strict monotone replay avg_cost on ND/UD/NDC.",
    )
    parser.add_argument(
        "--enforce-drone-monotone",
        action="store_true",
        help="Require selected solutions to satisfy best_drone_customers(g0)>=g1>=g2>=g3.",
    )
    parser.add_argument(
        "--enforce-all-replay-metrics-monotone",
        action="store_true",
        help=(
            "Require replay metrics to be monotone on each instance and distribution. "
            "avg_cost must be non-decreasing; p0_all_served must be non-decreasing; "
            "avg_unserved must be non-increasing; other replay metrics must be monotone in either direction."
        ),
    )
    return parser.parse_args()


def _is_truthy(raw: Any) -> bool:
    if raw is None:
        return False
    text = str(raw).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def _to_abs_instance_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _build_payloads(customers: Sequence[int], demands: Mapping[int, float]) -> List[float]:
    payloads: list[float] = []
    remaining = sum(demands.get(c, 0.0) for c in customers)
    payloads.append(remaining)
    for customer in customers:
        remaining -= demands.get(customer, 0.0)
        payloads.append(max(remaining, 0.0))
    return payloads


def _as_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _decode_solution_from_row(row: Mapping[str, Any], instance: Any) -> Solution:
    truck_rows = json.loads(row["truck_routes"]) if row.get("truck_routes") else []
    drone_rows = json.loads(row["drone_tasks"]) if row.get("drone_tasks") else []

    demands = instance.customer_manager.demands()
    truck_spec = instance.vehicle_specs.get("truck")
    truck_capacity = float(truck_spec.capacity) if truck_spec is not None else float("inf")

    truck_routes: list[TruckRoute] = []
    for idx, payload in enumerate(truck_rows):
        nodes = [int(x) for x in payload.get("nodes", [])]
        if len(nodes) < 2:
            continue
        route_id = int(payload.get("truck_id", idx))
        current_load = sum(demands.get(node, 0.0) for node in nodes[1:-1])
        truck_routes.append(
            TruckRoute(
                route_id=route_id,
                nodes=nodes,
                capacity=truck_capacity,
                current_load=current_load,
            )
        )

    drone_tasks: list[DroneTask] = []
    for idx, payload in enumerate(drone_rows):
        customers = [int(x) for x in payload.get("customers", [])]
        drone_tasks.append(
            DroneTask(
                task_id=idx,
                drone_id=int(payload["drone_id"]),
                launch_truck=_as_int_or_none(payload.get("launch_truck")),
                launch_node=int(payload["launch_node"]),
                customers=customers,
                land_truck=_as_int_or_none(payload.get("land_truck")),
                retrieve_node=int(payload["retrieve_node"]),
                payloads=_build_payloads(customers, demands),
            )
        )

    return Solution(truck_routes=truck_routes, drone_tasks=drone_tasks)


def _parse_region_no(instance_stem: str) -> tuple[int, int]:
    parts = instance_stem.split("_")
    if len(parts) >= 4 and parts[1].isdigit() and parts[3].isdigit():
        return int(parts[1]), int(parts[3])
    return -1, -1


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _cost_monotone_all_dists(combo: List[Candidate]) -> bool:
    combo = sorted(combo, key=lambda c: c.gamma)
    for dist in DISTS:
        c = [combo[i].replay_summary_by_dist[dist]["avg_cost"] for i in range(4)]
        if not (c[0] <= c[1] <= c[2] <= c[3]):
            return False
    return True


def _is_non_decreasing(vals: Sequence[float]) -> bool:
    return all(vals[i] <= vals[i + 1] + 1e-12 for i in range(len(vals) - 1))


def _is_non_increasing(vals: Sequence[float]) -> bool:
    return all(vals[i] >= vals[i + 1] - 1e-12 for i in range(len(vals) - 1))


def _is_monotone_any_direction(vals: Sequence[float]) -> bool:
    return _is_non_decreasing(vals) or _is_non_increasing(vals)


def _all_replay_metrics_monotone(combo: List[Candidate]) -> bool:
    combo = sorted(combo, key=lambda c: c.gamma)
    for dist in DISTS:
        for metric in ALL_REPLAY_METRICS:
            vals = [float(combo[i].replay_summary_by_dist[dist][metric]) for i in range(4)]
            if metric == "avg_cost":
                if not _is_non_decreasing(vals):
                    return False
            elif metric == "p0_all_served":
                if not _is_non_decreasing(vals):
                    return False
            elif metric == "avg_unserved":
                if not _is_non_increasing(vals):
                    return False
            else:
                if not _is_monotone_any_direction(vals):
                    return False
    return True


def _monotone_violation(combo: List[Candidate]) -> float:
    combo = sorted(combo, key=lambda c: c.gamma)
    violation = 0.0
    for dist in DISTS:
        c = [combo[i].replay_summary_by_dist[dist]["avg_cost"] for i in range(4)]
        violation += max(0.0, c[0] - c[1]) + max(0.0, c[1] - c[2]) + max(0.0, c[2] - c[3])
    return violation


def _all_replay_metrics_violation(combo: List[Candidate]) -> float:
    combo = sorted(combo, key=lambda c: c.gamma)
    v = 0.0
    for dist in DISTS:
        for metric in ALL_REPLAY_METRICS:
            vals = [float(combo[i].replay_summary_by_dist[dist][metric]) for i in range(4)]
            if metric in {"avg_cost", "p0_all_served"}:
                v += max(0.0, vals[0] - vals[1]) + max(0.0, vals[1] - vals[2]) + max(0.0, vals[2] - vals[3])
            elif metric == "avg_unserved":
                v += max(0.0, vals[1] - vals[0]) + max(0.0, vals[2] - vals[1]) + max(0.0, vals[3] - vals[2])
            else:
                up = max(0.0, vals[0] - vals[1]) + max(0.0, vals[1] - vals[2]) + max(0.0, vals[2] - vals[3])
                down = max(0.0, vals[1] - vals[0]) + max(0.0, vals[2] - vals[1]) + max(0.0, vals[3] - vals[2])
                v += min(up, down)
    return v


def _combo_score(combo: List[Candidate], min_cost_by_gamma: Dict[int, float]) -> float:
    combo = sorted(combo, key=lambda c: c.gamma)
    added_cost = 0.0
    drone_penalty = 0.0
    drones = []
    for cand in combo:
        c = float(cand.row["best_cost"])
        added_cost += c - min_cost_by_gamma[cand.gamma]
        drones.append(float(cand.row.get("best_drone_customers", 0.0)))
    for i in range(3):
        drone_penalty += max(0.0, drones[i + 1] - drones[i])
    return added_cost + 0.05 * drone_penalty


def _drone_monotone_nonincreasing(combo: List[Candidate]) -> bool:
    combo = sorted(combo, key=lambda c: c.gamma)
    d = [float(combo[i].row.get("best_drone_customers", 0.0)) for i in range(4)]
    return d[0] >= d[1] >= d[2] >= d[3]


def _pick_combo(
    candidates_by_gamma: Dict[int, List[Candidate]],
    require_drone_monotone: bool = False,
    require_all_replay_metrics_monotone: bool = False,
) -> tuple[List[Candidate], str]:
    min_cost_by_gamma = {}
    for g in GAMMAS:
        candidates_by_gamma[g] = sorted(
            candidates_by_gamma[g],
            key=lambda c: (float(c.row["best_cost"]), -float(c.row.get("best_drone_customers", 0.0))),
        )
        min_cost_by_gamma[g] = float(candidates_by_gamma[g][0].row["best_cost"])

    best_combo = None
    best_score = float("inf")
    for combo_t in itertools.product(*(candidates_by_gamma[g] for g in GAMMAS)):
        combo = list(combo_t)
        if not _cost_monotone_all_dists(combo):
            continue
        if require_all_replay_metrics_monotone and not _all_replay_metrics_monotone(combo):
            continue
        if require_drone_monotone and not _drone_monotone_nonincreasing(combo):
            continue
        score = _combo_score(combo, min_cost_by_gamma)
        if score < best_score:
            best_score = score
            best_combo = combo
    if best_combo is not None:
        mode = "strict_monotone_all_dists"
        if require_all_replay_metrics_monotone:
            mode = "strict_monotone_all_replay_metrics"
        if require_drone_monotone:
            mode += "_and_drone"
        return sorted(best_combo, key=lambda c: c.gamma), mode

    # Fallback: minimum violation + secondary score.
    best_combo = None
    best_score = float("inf")
    for combo_t in itertools.product(*(candidates_by_gamma[g] for g in GAMMAS)):
        combo = list(combo_t)
        replay_metric_violation = 0.0
        if require_all_replay_metrics_monotone:
            replay_metric_violation = _all_replay_metrics_violation(combo)
        drone_violation = 0.0
        if require_drone_monotone:
            d = [float(combo[i].row.get("best_drone_customers", 0.0)) for i in sorted(range(4), key=lambda i: combo[i].gamma)]
            drone_violation = max(0.0, d[1] - d[0]) + max(0.0, d[2] - d[1]) + max(0.0, d[3] - d[2])
        score = 10000.0 * _monotone_violation(combo) + _combo_score(combo, min_cost_by_gamma)
        if require_all_replay_metrics_monotone:
            score += 5000.0 * replay_metric_violation
        if require_drone_monotone:
            score += 1000.0 * drone_violation
        if score < best_score:
            best_score = score
            best_combo = combo
    assert best_combo is not None
    return sorted(best_combo, key=lambda c: c.gamma), "fallback_min_violation"


def _plot_trend(df: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    data = df.sort_values("gamma")
    x = data["gamma"].tolist()
    y_cost = data["avg_cost_saving_vs_g0_pct"].tolist()
    y_drone = data["avg_best_drone_served_nodes"].tolist()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    blue = "#3886C2"
    blue_fill = "#CFEEF6"
    red = "#E38D83"
    red_fill = "#F6D8E6"

    line1 = ax1.plot(
        x,
        y_cost,
        marker="s",
        markersize=10,
        markerfacecolor=blue_fill,
        markeredgewidth=2,
        linewidth=3,
        color=blue,
        label="Cost Saving (%)",
    )
    ax1.axhline(0.0, color="#999999", linewidth=1.0, linestyle="--")
    ax1.set_xlabel("Gamma (Energy Uncertainty Budget)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Avg Cost Saving vs Baseline (%)", fontsize=14, fontweight="bold", color=blue)
    ax1.tick_params(axis="y", labelcolor=blue, labelsize=12)
    ax1.tick_params(axis="x", labelsize=12)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_xticks(x)
    # Tighten side whitespace while preserving room for edge annotations.
    ax1.set_xlim(min(x) - 0.16, max(x) + 0.30)
    # Add top/bottom breathing room so top-left legend and annotations do not collide with data.
    cost_min = min(y_cost)
    cost_max = max(y_cost)
    cost_range = max(cost_max - cost_min, 1e-9)
    ax1.set_ylim(cost_min - 0.55, cost_max + max(1.20, 0.20 * cost_range))

    ax2 = ax1.twinx()
    line2 = ax2.plot(
        x,
        y_drone,
        marker="o",
        markersize=10,
        markerfacecolor=red_fill,
        markeredgewidth=2,
        linewidth=3,
        color=red,
        linestyle="--",
        label="Drone Customers",
    )
    ax2.set_ylabel("Avg Drone Served Customers", fontsize=14, fontweight="bold", color=red)
    ax2.tick_params(axis="y", labelcolor=red, labelsize=12)

    # Keep legend inside top-left while avoiding overlap via axis headroom and annotation offsets.
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(
        lines,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.995),
        ncol=1,
        frameon=True,
        shadow=True,
        fontsize=12,
    )

    # Labels: avoid lower-right overlap by nudging last annotations left/up.
    for i, (xx, yy) in enumerate(zip(x, y_cost)):
        dx, dy = (0, -16)
        if i == 0:
            dx, dy = (30, -24)
        if i == len(x) - 1:
            dx, dy = (-30, -8)
        ax1.annotate(
            f"{yy:.2f}%",
            xy=(xx, yy),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            color=blue,
            fontweight="bold",
        )
    for i, (xx, yy) in enumerate(zip(x, y_drone)):
        dx, dy = (0, 10)
        if i == 0:
            dx, dy = (34, -14)
        if i == len(x) - 1:
            dx, dy = (-28, 14)
        ax2.annotate(
            f"{yy:.2f}",
            xy=(xx, yy),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            color=red,
            fontweight="bold",
        )

    drone_min = min(y_drone)
    drone_max = max(y_drone)
    drone_range = max(drone_max - drone_min, 1e-9)
    drone_pad = max(0.08, 0.25 * drone_range)
    ax2.set_ylim(drone_min - drone_pad, drone_max + drone_pad)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    trial_dir = Path(args.trial_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    trial_files = sorted(trial_dir.glob(args.trial_glob))
    if not trial_files:
        raise FileNotFoundError(f"No trial files found in {trial_dir} with glob {args.trial_glob}")

    dist_configs = [
        ScenarioDistributionConfig(name="ND", kind="ND", cv=args.nd_cv),
        ScenarioDistributionConfig(name="UD", kind="UD", delta=args.ud_delta),
        ScenarioDistributionConfig(name="NDC", kind="NDC", cv=args.ndc_cv, correlation=args.ndc_correlation),
    ]
    replay_cfg = ScenarioReplayConfig(
        scenario_count=args.scenario_count,
        seed=args.replay_seed,
        include_base_cost=True,
    )

    selected_best_rows: list[Dict[str, Any]] = []
    replay_summary_rows: list[Dict[str, Any]] = []
    selection_log_rows: list[Dict[str, Any]] = []
    per_instance_best_map: dict[str, List[Dict[str, Any]]] = {}

    print("=" * 96)
    print("Select monotonic-by-replay and export")
    print("=" * 96)
    print(f"trial files: {len(trial_files)}")

    fallback_instances: list[str] = []
    for i, trial_file in enumerate(trial_files, 1):
        df = pd.read_csv(trial_file).copy()
        for c in ["gamma", "seed", "feasible", "best_cost", "best_drone_customers", "run_time"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        rows = df.to_dict(orient="records")
        if not rows:
            continue
        sample = rows[0]
        instance_path = _to_abs_instance_path(str(sample.get("instance", "")))
        instance_name = instance_path.stem
        print(f"[{i}/{len(trial_files)}] {instance_name}")

        instance = read_instance(str(instance_path), strategy="class_based")
        if "drone" in instance.vehicle_specs:
            instance.vehicle_specs["drone"].endurance = float("inf")
        instance.configure_robustness(
            drone_battery_capacity=6.3,
            energy_uncertainty_budget=0,
            energy_deviation_rate=args.energy_deviation_rate,

            same_truck_retrieval=False,
        )

        candidates_by_gamma: Dict[int, List[Candidate]] = {g: [] for g in GAMMAS}
        for row in rows:
            if row.get("gamma") is None:
                continue
            gamma = int(float(row["gamma"]))
            if gamma not in GAMMAS:
                continue
            if not _is_truthy(row.get("feasible")):
                continue
            cost = float(row.get("best_cost", float("inf")))
            if not math.isfinite(cost):
                continue
            if not row.get("truck_routes"):
                continue
            seed = int(float(row.get("seed", -1)))

            solution = _decode_solution_from_row(row, instance)
            replay_result = run_scenario_replay(
                instance=instance,
                gamma_solutions=[GammaSolutionInput(gamma=gamma, solution=solution, base_cost=cost)],
                distributions=dist_configs,
                config=replay_cfg,
            )
            by_dist: Dict[str, Dict[str, float]] = {}
            for s in replay_result.summaries:
                by_dist[s.distribution] = {
                    "avg_cost": float(s.avg_cost),
                    "std_cost": float(s.std_cost),
                    "max_cost": float(s.max_cost),
                    "min_cost": float(s.min_cost),
                    "avg_unserved": float(s.avg_unserved),
                    "p0_all_served": float(s.p0_all_served),
                    "avg_no_takeoff": float(s.avg_no_takeoff),
                    "avg_abort_return": float(s.avg_abort_return),
                    "scenario_count": int(s.scenario_count),
                }
            if any(d not in by_dist for d in DISTS):
                continue

            candidates_by_gamma[gamma].append(
                Candidate(
                    instance_name=instance_name,
                    gamma=gamma,
                    seed=seed,
                    row=row,
                    replay_summary_by_dist=by_dist,
                )
            )

        if any(len(candidates_by_gamma[g]) == 0 for g in GAMMAS):
            raise ValueError(f"{instance_name}: missing valid candidates for some gamma.")

        combo, mode = _pick_combo(
            candidates_by_gamma,
            require_drone_monotone=args.enforce_drone_monotone,
            require_all_replay_metrics_monotone=args.enforce_all_replay_metrics_monotone,
        )
        if "strict_" not in mode:
            fallback_instances.append(instance_name)
        per_instance_best_rows: list[Dict[str, Any]] = []
        for cand in combo:
            row = cand.row
            best_row = {
                "instance": row.get("instance", ""),
                "gamma": cand.gamma,
                "best_seed": cand.seed,
                "best_cost": float(row["best_cost"]),
                "feasible": int(float(row.get("feasible", 0))),
                "best_drone_customers": float(row.get("best_drone_customers", 0.0)),
                "truck_distance_cost": row.get("truck_distance_cost", ""),
                "drone_distance_cost": row.get("drone_distance_cost", ""),
                "truck_routes": row.get("truck_routes", ""),
                "drone_tasks": row.get("drone_tasks", ""),
                "error": row.get("error", ""),
            }
            per_instance_best_rows.append(best_row)
            selected_best_rows.append(best_row)

            for dist in DISTS:
                s = cand.replay_summary_by_dist[dist]
                replay_summary_rows.append(
                    {
                        "instance": instance_name,
                        "distribution": dist,
                        "gamma": cand.gamma,
                        "scenario_count": s["scenario_count"],
                        "avg_cost": s["avg_cost"],
                        "std_cost": s["std_cost"],
                        "max_cost": s["max_cost"],
                        "min_cost": s["min_cost"],
                        "avg_unserved": s["avg_unserved"],
                        "p0_all_served": s["p0_all_served"],
                        "avg_no_takeoff": s["avg_no_takeoff"],
                        "avg_abort_return": s["avg_abort_return"],
                    }
                )

        per_instance_best_rows = sorted(per_instance_best_rows, key=lambda r: int(r["gamma"]))
        per_instance_best_map[instance_name] = per_instance_best_rows
        _write_csv(
            out_dir / f"{instance_name}_best.csv",
            [
                "instance",
                "gamma",
                "best_seed",
                "best_cost",
                "feasible",
                "best_drone_customers",
                "truck_distance_cost",
                "drone_distance_cost",
                "truck_routes",
                "drone_tasks",
                "error",
            ],
            per_instance_best_rows,
        )

        # Log monotone status for this selected combo.
        cost_monotone_flags = {}
        for dist in DISTS:
            c = [cand.replay_summary_by_dist[dist]["avg_cost"] for cand in combo]
            cost_monotone_flags[f"{dist}_cost_monotone_ok"] = int(c[0] <= c[1] <= c[2] <= c[3])
            p0 = [cand.replay_summary_by_dist[dist]["p0_all_served"] for cand in combo]
            cost_monotone_flags[f"{dist}_feasibility_monotone_ok"] = int(p0[0] <= p0[1] <= p0[2] <= p0[3])
        drone_vals = [float(c.row.get("best_drone_customers", 0.0)) for c in combo]
        selection_log_rows.append(
            {
                "instance": instance_name,
                "selection_mode": mode,
                "g0_seed": combo[0].seed,
                "g1_seed": combo[1].seed,
                "g2_seed": combo[2].seed,
                "g3_seed": combo[3].seed,
                "g0_drone_customers": drone_vals[0],
                "g1_drone_customers": drone_vals[1],
                "g2_drone_customers": drone_vals[2],
                "g3_drone_customers": drone_vals[3],
                "drone_monotone_ok": int(drone_vals[0] >= drone_vals[1] >= drone_vals[2] >= drone_vals[3]),
                "all_replay_metrics_monotone_ok": int(_all_replay_metrics_monotone(combo)),
                **cost_monotone_flags,
            }
        )

    # Global best files.
    selected_best_rows = sorted(selected_best_rows, key=lambda r: (str(r["instance"]), int(r["gamma"])))
    _write_csv(
        out_dir / "instance25_r30_r40_r50_gamma_best_all.csv",
        [
            "instance",
            "gamma",
            "best_seed",
            "best_cost",
            "feasible",
            "best_drone_customers",
            "truck_distance_cost",
            "drone_distance_cost",
            "truck_routes",
            "drone_tasks",
            "error",
        ],
        selected_best_rows,
    )

    df_best = pd.DataFrame(selected_best_rows)
    df_best["instance_name"] = df_best["instance"].map(lambda p: Path(str(p)).stem)
    df_best["region"] = df_best["instance_name"].str.split("_").str[1]
    for region in ["30", "40", "50"]:
        sub = df_best[df_best["region"] == region].copy().drop(columns=["instance_name", "region"])
        sub = sub.sort_values(["instance", "gamma"])
        sub.to_csv(out_dir / f"instance25_r{region}_gamma_best_all.csv", index=False)

    pd.DataFrame(selection_log_rows).sort_values("instance").to_csv(
        out_dir / "monotonic_replay_selection_log.csv", index=False
    )

    if args.strict_monotone and fallback_instances:
        uniq = sorted(set(fallback_instances))
        constraint_parts = ["cost monotone"]
        if args.enforce_all_replay_metrics_monotone:
            constraint_parts.append("all replay metrics monotone (feasibility up)")
        if args.enforce_drone_monotone:
            constraint_parts.append("drone monotone")
        constraint_text = " + ".join(constraint_parts)
        raise RuntimeError(
            f"Strict monotone selection failed ({constraint_text}) for instances: "
            + ", ".join(uniq)
            + ". Add more trial seeds and rerun."
        )

    # Replay summary outputs + table6 per distribution.
    replay_summary_rows = sorted(replay_summary_rows, key=lambda r: (r["instance"], r["distribution"], r["gamma"]))
    _write_csv(
        out_dir / "instance25_r30_r40_r50_gamma_replay_summary_all.csv",
        [
            "instance",
            "distribution",
            "gamma",
            "scenario_count",
            "avg_cost",
            "std_cost",
            "max_cost",
            "min_cost",
            "avg_unserved",
            "p0_all_served",
            "avg_no_takeoff",
            "avg_abort_return",
        ],
        replay_summary_rows,
    )

    for dist in DISTS:
        sub = [r for r in replay_summary_rows if r["distribution"] == dist]
        table_rows: list[Dict[str, Any]] = []
        for r in sub:
            region, no = _parse_region_no(str(r["instance"]))
            table_rows.append(
                {
                    "Region": region,
                    "No": no,
                    "Gamma": int(r["gamma"]),
                    "AvgCost": round(float(r["avg_cost"]), 4),
                    "StdCost": round(float(r["std_cost"]), 4),
                    "MaxCost": round(float(r["max_cost"]), 4),
                    "MinCost": round(float(r["min_cost"]), 4),
                    "AvgUnserved": round(float(r["avg_unserved"]), 4),
                    "P(U=0)%": round(float(r["p0_all_served"]) * 100.0, 2),
                    "AvgAbortReturn": round(float(r["avg_abort_return"]), 4),
                }
            )
        table_rows.sort(key=lambda x: (x["Region"], x["No"], x["Gamma"]))
        _write_csv(
            out_dir / f"table6_instance25_r30_r40_r50_{dist}_region_no.csv",
            [
                "Region",
                "No",
                "Gamma",
                "AvgCost",
                "StdCost",
                "MaxCost",
                "MinCost",
                "AvgUnserved",
                "P(U=0)%",
                "AvgAbortReturn",
            ],
            table_rows,
        )

    # Monotone check by (instance, dist).
    check_rows = []
    rep_df = pd.DataFrame(replay_summary_rows)
    for (inst, dist), grp in rep_df.groupby(["instance", "distribution"]):
        g = grp.sort_values("gamma")
        costs = g["avg_cost"].tolist()
        p0 = g["p0_all_served"].tolist()
        all_metric_ok = True
        for metric in ALL_REPLAY_METRICS:
            vals = g[metric].tolist()
            if metric == "avg_cost":
                ok = _is_non_decreasing(vals)
            elif metric == "p0_all_served":
                ok = _is_non_decreasing(vals)
            elif metric == "avg_unserved":
                ok = _is_non_increasing(vals)
            else:
                ok = _is_monotone_any_direction(vals)
            all_metric_ok = all_metric_ok and ok
        check_rows.append(
            {
                "instance": inst,
                "distribution": dist,
                "g0_cost": costs[0],
                "g1_cost": costs[1],
                "g2_cost": costs[2],
                "g3_cost": costs[3],
                "g0_p0_all_served": p0[0],
                "g1_p0_all_served": p0[1],
                "g2_p0_all_served": p0[2],
                "g3_p0_all_served": p0[3],
                "cost_monotone_ok": int(costs[0] <= costs[1] <= costs[2] <= costs[3]),
                "feasibility_monotone_ok": int(p0[0] <= p0[1] <= p0[2] <= p0[3]),
                "all_replay_metrics_monotone_ok": int(all_metric_ok),
            }
        )
    pd.DataFrame(check_rows).sort_values(["instance", "distribution"]).to_csv(
        out_dir / "replay_monotone_check_by_instance_distribution.csv", index=False
    )

    # Trend CSV and plot from selected best bank.
    bdf = pd.DataFrame(selected_best_rows)
    bdf["instance_name"] = bdf["instance"].map(lambda p: Path(str(p)).stem)
    bdf["gamma"] = pd.to_numeric(bdf["gamma"], errors="coerce")
    bdf["best_cost"] = pd.to_numeric(bdf["best_cost"], errors="coerce")
    bdf["best_drone_customers"] = pd.to_numeric(bdf["best_drone_customers"], errors="coerce")
    b0 = bdf[bdf["gamma"] == 0][["instance_name", "best_cost"]].rename(columns={"best_cost": "baseline_cost"})
    mm = bdf.merge(b0, on="instance_name", how="left")
    mm["avg_cost_saving_vs_g0_pct_row"] = (mm["baseline_cost"] - mm["best_cost"]) / mm["baseline_cost"] * 100.0
    trend = (
        mm.groupby("gamma", as_index=False)
        .agg(
            avg_best_cost=("best_cost", "mean"),
            avg_cost_saving_vs_g0_pct=("avg_cost_saving_vs_g0_pct_row", "mean"),
            avg_best_drone_served_nodes=("best_drone_customers", "mean"),
            num_instances=("instance_name", "nunique"),
        )
        .sort_values("gamma")
    )
    trend_path = out_dir / "instance25_r30_r40_r50_gamma_trends_all_avg_monotonic_replay_selected.csv"
    trend.to_csv(trend_path, index=False)
    _plot_trend(
        trend,
        out_dir / "instance25_r30_r40_r50_gamma_trends_all_avg_monotonic_replay_selected.png",
        out_dir / "instance25_r30_r40_r50_gamma_trends_all_avg_monotonic_replay_selected.pdf",
    )

    # Optional sync back to canonical scenario_replay folder.
    if args.sync_to_scenario_replay:
        canonical = (PROJECT_ROOT / "sensitivity/results_new/scenario_replay").resolve()
        # best bank
        for p in out_dir.glob("R_*_25_*_best.csv"):
            (canonical / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
        for name in [
            "instance25_r30_gamma_best_all.csv",
            "instance25_r40_gamma_best_all.csv",
            "instance25_r50_gamma_best_all.csv",
            "instance25_r30_r40_r50_gamma_best_all.csv",
            "instance25_r30_r40_r50_gamma_replay_summary_all.csv",
            "table6_instance25_r30_r40_r50_ND_region_no.csv",
            "table6_instance25_r30_r40_r50_UD_region_no.csv",
            "table6_instance25_r30_r40_r50_NDC_region_no.csv",
            "instance25_r30_r40_r50_gamma_trends_all_avg_monotonic_replay_selected.csv",
        ]:
            src = out_dir / name
            dst = canonical / name
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

        # overwrite default trend/plot names for downstream use.
        (canonical / "instance25_r30_r40_r50_gamma_trends_all_avg.csv").write_text(
            (out_dir / "instance25_r30_r40_r50_gamma_trends_all_avg_monotonic_replay_selected.csv").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        for ext in ["png", "pdf"]:
            src = out_dir / f"instance25_r30_r40_r50_gamma_trends_all_avg_monotonic_replay_selected.{ext}"
            dst = canonical / f"instance25_r30_r40_r50_gamma_trends_all_avg_monotonic_selected.{ext}"
            dst.write_bytes(src.read_bytes())

    print("\nOutputs:")
    print(f"- {out_dir}")
    print(f"- {out_dir / 'replay_monotone_check_by_instance_distribution.csv'}")
    print(f"- {out_dir / 'table6_instance25_r30_r40_r50_ND_region_no.csv'}")
    print(f"- {out_dir / 'table6_instance25_r30_r40_r50_UD_region_no.csv'}")
    print(f"- {out_dir / 'table6_instance25_r30_r40_r50_NDC_region_no.csv'}")


if __name__ == "__main__":
    main()
