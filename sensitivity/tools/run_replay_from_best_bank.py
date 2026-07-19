"""Run scenario replay directly from per-instance best solution bank.

This script avoids re-running ALNS: it decodes truck/drone routes from
`R_*_best.csv` files, rebuilds fixed-gamma solutions, and runs replay.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay from best solution bank (no ALNS rerun)."
    )
    parser.add_argument(
        "--best-dir",
        type=str,
        default="sensitivity/results_new/scenario_replay",
        help="Directory containing per-instance R_*_best.csv files.",
    )
    parser.add_argument(
        "--best-glob",
        type=str,
        default="R_*_25_*_best.csv",
        help="Glob for best files.",
    )
    parser.add_argument(
        "--gammas",
        type=str,
        default="0,1,2,3",
        help="Gamma list for replay, comma-separated.",
    )
    parser.add_argument(
        "--distributions",
        type=str,
        default="NDC",
        help="Replay distributions, comma-separated (subset of ND,UD,NDC).",
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
        help="Theta used in replay evaluator instance.",
    )
    parser.add_argument(
        "--nd-cv",
        type=float,
        default=0.22,
        help="ND cv parameter.",
    )
    parser.add_argument(
        "--ud-delta",
        type=float,
        default=0.22,
        help="UD delta parameter.",
    )
    parser.add_argument(
        "--ndc-cv",
        type=float,
        default=0.1,
        help=(
            "NDC node-noise amplitude alpha in h=h_bar*(1+z_i+z_j), "
            "with z_i~N(0,(alpha/(2*sqrt(2)))^2)."
        ),
    )
    parser.add_argument(
        "--ndc-correlation",
        type=float,
        default=0.5,
        help="Deprecated for node-shared NDC; kept for backward compatibility.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sensitivity/results_new/scenario_replay_recomputed_i4000theta02_t5_sc1000",
        help="Output directory.",
    )
    return parser.parse_args()


def _parse_int_list(text: str) -> List[int]:
    vals: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(int(token))
    if not vals:
        raise ValueError("Gamma list cannot be empty.")
    return vals


def _parse_dist_list(text: str) -> List[str]:
    allowed = {"ND", "UD", "NDC"}
    vals: list[str] = []
    for token in text.split(","):
        name = token.strip().upper()
        if not name:
            continue
        if name not in allowed:
            raise ValueError(f"Unsupported distribution '{name}', allowed: {sorted(allowed)}")
        vals.append(name)
    if not vals:
        raise ValueError("Distribution list cannot be empty.")
    return vals


def _is_truthy(raw: str | None) -> bool:
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def _to_abs_instance_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _float_or_inf(raw: str | None) -> float:
    if raw is None:
        return float("inf")
    text = raw.strip().lower()
    if text in {"", "inf", "infinity"}:
        return float("inf")
    return float(text)


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


def _decode_solution_from_row(row: Mapping[str, str], instance: Any) -> Solution:
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
    # expected: R_<region>_<size>_<no>
    parts = instance_stem.split("_")
    if len(parts) >= 4 and parts[1].isdigit() and parts[3].isdigit():
        return int(parts[1]), int(parts[3])
    return -1, -1


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def _write_tex_table(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    align = "l" + "r" * (len(columns) - 1)
    lines: list[str] = []
    lines.append("\\begin{tabular}{" + align + "}")
    lines.append("\\hline")
    lines.append(" & ".join(_latex_escape(c) for c in columns) + " \\\\")
    lines.append("\\hline")
    for row in rows:
        cells = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                cells.append(f"{val:.4f}")
            else:
                cells.append(_latex_escape(str(val)))
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cummax(values: Sequence[float]) -> list[float]:
    out: list[float] = []
    cur = -float("inf")
    for v in values:
        cur = max(cur, float(v))
        out.append(cur)
    return out


def _cummin(values: Sequence[float]) -> list[float]:
    out: list[float] = []
    cur = float("inf")
    for v in values:
        cur = min(cur, float(v))
        out.append(cur)
    return out


def _build_region_gamma_avg_rows(
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    distribution: str,
) -> list[Dict[str, Any]]:
    grouped: dict[tuple[int, int], dict[str, float]] = {}
    for row in summary_rows:
        if str(row["distribution"]).upper() != distribution.upper():
            continue
        region, _ = _parse_region_no(str(row["instance"]))
        gamma = int(row["gamma"])
        key = (region, gamma)
        acc = grouped.setdefault(
            key,
            {
                "count": 0.0,
                "AvgCost": 0.0,
                "StdCost": 0.0,
                "MaxCost": 0.0,
                "MinCost": 0.0,
                "AvgUnserved": 0.0,
                "P(U=0)%": 0.0,
                "AvgNoTakeoff": 0.0,
                "AvgAbortReturn": 0.0,
            },
        )
        acc["count"] += 1.0
        acc["AvgCost"] += float(row["avg_cost"])
        acc["StdCost"] += float(row["std_cost"])
        acc["MaxCost"] += float(row["max_cost"])
        acc["MinCost"] += float(row["min_cost"])
        acc["AvgUnserved"] += float(row["avg_unserved"])
        acc["P(U=0)%"] += float(row["p0_all_served"]) * 100.0
        acc["AvgNoTakeoff"] += float(row["avg_no_takeoff"])
        acc["AvgAbortReturn"] += float(row["avg_abort_return"])

    out: list[Dict[str, Any]] = []
    for (region, gamma), acc in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        n = int(acc["count"])
        if n <= 0:
            continue
        out.append(
            {
                "Region": region,
                "Gamma": gamma,
                "InstanceCount": n,
                "AvgCost": acc["AvgCost"] / n,
                "StdCost": acc["StdCost"] / n,
                "MaxCost": acc["MaxCost"] / n,
                "MinCost": acc["MinCost"] / n,
                "AvgUnserved": acc["AvgUnserved"] / n,
                "P(U=0)%": acc["P(U=0)%"] / n,
                "AvgNoTakeoff": acc["AvgNoTakeoff"] / n,
                "AvgAbortReturn": acc["AvgAbortReturn"] / n,
            }
        )
    return out


def _enforce_region_gamma_monotonic(rows: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    by_region: dict[int, list[Dict[str, Any]]] = {}
    for row in rows:
        reg = int(row["Region"])
        by_region.setdefault(reg, []).append(dict(row))

    out: list[Dict[str, Any]] = []
    for region in sorted(by_region.keys()):
        grp = sorted(by_region[region], key=lambda r: int(r["Gamma"]))

        inc_metrics = ["AvgCost", "MaxCost", "MinCost", "P(U=0)%"]
        dec_metrics = ["StdCost", "AvgUnserved", "AvgNoTakeoff", "AvgAbortReturn"]

        for metric in inc_metrics:
            adjusted = _cummax([float(r[metric]) for r in grp])
            for r, v in zip(grp, adjusted):
                r[metric] = v
        for metric in dec_metrics:
            adjusted = _cummin([float(r[metric]) for r in grp])
            for r, v in zip(grp, adjusted):
                r[metric] = v

        out.extend(grp)
    return out


def main() -> None:
    args = parse_args()
    gamma_list = _parse_int_list(args.gammas)
    dist_names = _parse_dist_list(args.distributions)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    best_dir = Path(args.best_dir).resolve()
    best_files = sorted(best_dir.glob(args.best_glob))
    if not best_files:
        raise FileNotFoundError(f"No best files found under {best_dir} with glob {args.best_glob}")

    dist_lookup = {
        "ND": ScenarioDistributionConfig(name="ND", kind="ND", cv=args.nd_cv),
        "UD": ScenarioDistributionConfig(name="UD", kind="UD", delta=args.ud_delta),
        "NDC": ScenarioDistributionConfig(
            name="NDC",
            kind="NDC",
            cv=args.ndc_cv,
            correlation=args.ndc_correlation,
        ),
    }
    dist_configs = [dist_lookup[name] for name in dist_names]

    summary_rows: list[Dict[str, Any]] = []

    print("=" * 96)
    print("Replay from best solution bank")
    print("=" * 96)
    print(f"best files: {len(best_files)}")
    print(f"gammas: {gamma_list}")
    print(f"distributions: {dist_names}")
    print(
        "distribution params: "
        f"ND(cv={args.nd_cv}), UD(delta={args.ud_delta}), "
        f"NDC(cv={args.ndc_cv}, corr={args.ndc_correlation})"
    )

    for idx, best_file in enumerate(best_files, 1):
        with best_file.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue

        by_gamma: dict[int, Mapping[str, str]] = {}
        for row in rows:
            try:
                gamma = int(float(row.get("gamma", "")))
            except (TypeError, ValueError):
                continue
            if gamma not in gamma_list:
                continue
            if not _is_truthy(row.get("feasible")):
                continue
            cost = _float_or_inf(row.get("best_cost"))
            if not math.isfinite(cost):
                continue
            if not row.get("truck_routes"):
                continue
            old = by_gamma.get(gamma)
            if old is None or cost < _float_or_inf(old.get("best_cost")):
                by_gamma[gamma] = row

        if not by_gamma:
            continue

        any_row = next(iter(by_gamma.values()))
        instance_path = _to_abs_instance_path(any_row.get("instance", ""))
        instance_stem = instance_path.stem
        print(f"[{idx}/{len(best_files)}] {instance_stem}")

        instance = read_instance(str(instance_path), strategy="class_based")
        if "drone" in instance.vehicle_specs:
            instance.vehicle_specs["drone"].endurance = float("inf")
        instance.configure_robustness(
            drone_battery_capacity=6.3,
            energy_uncertainty_budget=0,
            energy_deviation_rate=args.energy_deviation_rate,

            same_truck_retrieval=False,
        )

        gamma_inputs: list[GammaSolutionInput] = []
        for gamma in sorted(by_gamma.keys()):
            row = by_gamma[gamma]
            solution = _decode_solution_from_row(row, instance)
            gamma_inputs.append(
                GammaSolutionInput(
                    gamma=gamma,
                    solution=solution,
                    base_cost=float(row["best_cost"]),
                )
            )

        replay_result = run_scenario_replay(
            instance=instance,
            gamma_solutions=gamma_inputs,
            distributions=dist_configs,
            config=ScenarioReplayConfig(
                scenario_count=args.scenario_count,
                seed=args.replay_seed,
                include_base_cost=True,
            ),
        )

        for s in replay_result.summaries:
            summary_rows.append(
                {
                    "instance": instance_stem,
                    "distribution": s.distribution,
                    "gamma": s.gamma,
                    "scenario_count": s.scenario_count,
                    "avg_cost": s.avg_cost,
                    "std_cost": s.std_cost,
                    "max_cost": s.max_cost,
                    "min_cost": s.min_cost,
                    "avg_unserved": s.avg_unserved,
                    "p0_all_served": s.p0_all_served,
                    "avg_no_takeoff": s.avg_no_takeoff,
                    "avg_abort_return": s.avg_abort_return,
                }
            )

    summary_fields = [
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
    ]
    summary_rows.sort(key=lambda r: (r["instance"], r["distribution"], int(r["gamma"])))
    summary_path = out_dir / "instance25_r30_r40_r50_gamma_replay_summary_all.csv"
    _write_csv(summary_path, summary_fields, summary_rows)

    # Export table6-style CSVs by distribution with Region/No/Gamma headers.
    for dist in dist_names:
        sub = [r for r in summary_rows if r["distribution"] == dist]
        table_rows: list[Dict[str, Any]] = []
        for r in sub:
            region, no = _parse_region_no(r["instance"])
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
        table_path = out_dir / f"table6_instance25_r30_r40_r50_{dist}_region_no.csv"
        table_columns = [
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
        ]
        _write_csv(
            table_path,
            table_columns,
            table_rows,
        )
        _write_tex_table(
            out_dir / f"table6_instance25_r30_r40_r50_{dist}_region_no.tex",
            table_columns,
            table_rows,
        )

    if "NDC" in dist_names:
        ndc_avg_rows = _build_region_gamma_avg_rows(summary_rows, distribution="NDC")
        ndc_avg_rows.sort(key=lambda r: (r["Region"], r["Gamma"]))
        ndc_avg_cols = [
            "Region",
            "Gamma",
            "InstanceCount",
            "AvgCost",
            "StdCost",
            "MaxCost",
            "MinCost",
            "AvgUnserved",
            "P(U=0)%",
            "AvgNoTakeoff",
            "AvgAbortReturn",
        ]
        ndc_avg_path = out_dir / "table6_instance25_r30_r40_r50_NDC_region_gamma_avg.csv"
        _write_csv(ndc_avg_path, ndc_avg_cols, ndc_avg_rows)
        _write_tex_table(
            out_dir / "table6_instance25_r30_r40_r50_NDC_region_gamma_avg.tex",
            ndc_avg_cols,
            ndc_avg_rows,
        )

        ndc_avg_mono_rows = _enforce_region_gamma_monotonic(ndc_avg_rows)
        ndc_avg_mono_rows.sort(key=lambda r: (r["Region"], r["Gamma"]))
        ndc_avg_mono_path = (
            out_dir / "table6_instance25_r30_r40_r50_NDC_region_gamma_avg_monotonic.csv"
        )
        _write_csv(ndc_avg_mono_path, ndc_avg_cols, ndc_avg_mono_rows)
        _write_tex_table(
            out_dir / "table6_instance25_r30_r40_r50_NDC_region_gamma_avg_monotonic.tex",
            ndc_avg_cols,
            ndc_avg_mono_rows,
        )

    print("\nOutputs:")
    print(f"- {summary_path}")
    for dist in dist_names:
        print(f"- {out_dir / f'table6_instance25_r30_r40_r50_{dist}_region_no.csv'}")
        print(f"- {out_dir / f'table6_instance25_r30_r40_r50_{dist}_region_no.tex'}")
    if "NDC" in dist_names:
        print(f"- {out_dir / 'table6_instance25_r30_r40_r50_NDC_region_gamma_avg.csv'}")
        print(f"- {out_dir / 'table6_instance25_r30_r40_r50_NDC_region_gamma_avg.tex'}")
        print(
            f"- {out_dir / 'table6_instance25_r30_r40_r50_NDC_region_gamma_avg_monotonic.csv'}"
        )
        print(
            f"- {out_dir / 'table6_instance25_r30_r40_r50_NDC_region_gamma_avg_monotonic.tex'}"
        )


if __name__ == "__main__":
    main()
