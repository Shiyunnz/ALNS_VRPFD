"""Select monotonic gamma solutions from trial bank and export best files.

Monotonic target:
- cost(gamma=0) <= cost(1) <= cost(2) <= cost(3)

Secondary target (preferred when feasible):
- drone(0) >= drone(1) >= drone(2) >= drone(3)
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


GAMMAS = [0, 1, 2, 3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pick monotonic gamma best rows from per-instance trials."
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
        help="Glob pattern for trial files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sensitivity/results_new/scenario_replay_monotonic_selected",
        help="Output dir for selected best files.",
    )
    return parser.parse_args()


def _row_cost(row: Dict[str, Any]) -> float:
    return float(row.get("best_cost", float("inf")))


def _row_drone(row: Dict[str, Any]) -> float:
    return float(row.get("best_drone_customers", 0.0))


def _is_finite_feasible(row: Dict[str, Any]) -> bool:
    try:
        feasible = float(row.get("feasible", 0))
        cost = float(row.get("best_cost", float("inf")))
    except (TypeError, ValueError):
        return False
    return feasible > 0 and math.isfinite(cost)


def _combo_score(
    combo: List[Dict[str, Any]],
    min_cost_by_gamma: Dict[int, float],
    *,
    penalize_drone_increase: bool,
) -> float:
    added_cost = 0.0
    for row in combo:
        g = int(float(row["gamma"]))
        added_cost += _row_cost(row) - min_cost_by_gamma[g]
    if not penalize_drone_increase:
        return added_cost
    drones = [_row_drone(r) for r in combo]
    drone_inc_penalty = 0.0
    for i in range(3):
        drone_inc_penalty += max(0.0, drones[i + 1] - drones[i])
    return added_cost + 0.1 * drone_inc_penalty


def _cost_monotone(combo: List[Dict[str, Any]]) -> bool:
    costs = [_row_cost(r) for r in combo]
    return costs[0] <= costs[1] <= costs[2] <= costs[3]


def _drone_nonincreasing(combo: List[Dict[str, Any]]) -> bool:
    drones = [_row_drone(r) for r in combo]
    return drones[0] >= drones[1] >= drones[2] >= drones[3]


def _select_combo(rows_by_gamma: Dict[int, List[Dict[str, Any]]]) -> tuple[List[Dict[str, Any]], str]:
    # Keep candidate list deterministic.
    for g in GAMMAS:
        rows_by_gamma[g] = sorted(
            rows_by_gamma[g],
            key=lambda r: (_row_cost(r), -_row_drone(r), float(r.get("run_time", 0.0))),
        )

    min_cost_by_gamma = {g: _row_cost(rows_by_gamma[g][0]) for g in GAMMAS}
    products = itertools.product(*(rows_by_gamma[g] for g in GAMMAS))

    # Priority 1: cost monotone + drone non-increasing
    best_combo = None
    best_score = float("inf")
    for combo_t in products:
        combo = list(combo_t)
        if not _cost_monotone(combo):
            continue
        if not _drone_nonincreasing(combo):
            continue
        score = _combo_score(combo, min_cost_by_gamma, penalize_drone_increase=False)
        if score < best_score:
            best_score = score
            best_combo = combo
    if best_combo is not None:
        return best_combo, "cost+drone_monotone"

    # Priority 2: cost monotone only
    products = itertools.product(*(rows_by_gamma[g] for g in GAMMAS))
    best_combo = None
    best_score = float("inf")
    for combo_t in products:
        combo = list(combo_t)
        if not _cost_monotone(combo):
            continue
        score = _combo_score(combo, min_cost_by_gamma, penalize_drone_increase=True)
        if score < best_score:
            best_score = score
            best_combo = combo
    if best_combo is not None:
        return best_combo, "cost_monotone_only"

    # Fallback: minimize cost-monotonicity violation.
    products = itertools.product(*(rows_by_gamma[g] for g in GAMMAS))
    best_combo = None
    best_score = float("inf")
    for combo_t in products:
        combo = list(combo_t)
        costs = [_row_cost(r) for r in combo]
        violation = max(0.0, costs[0] - costs[1]) + max(0.0, costs[1] - costs[2]) + max(0.0, costs[2] - costs[3])
        score = 1000.0 * violation + _combo_score(combo, min_cost_by_gamma, penalize_drone_increase=True)
        if score < best_score:
            best_score = score
            best_combo = combo
    assert best_combo is not None
    return best_combo, "fallback_min_violation"


def _write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    trial_dir = Path(args.trial_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    trial_files = sorted(trial_dir.glob(args.trial_glob))
    if not trial_files:
        raise FileNotFoundError(f"No trial files found in {trial_dir} with glob {args.trial_glob}")

    all_best_rows: list[Dict[str, Any]] = []
    selection_logs: list[Dict[str, Any]] = []

    for trial_file in trial_files:
        df = pd.read_csv(trial_file).copy()
        # Normalize numeric fields.
        for c in ["gamma", "seed", "feasible", "best_cost", "best_drone_customers", "run_time"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        rows = df.to_dict(orient="records")

        # Prefer feasible finite candidates. If any gamma has none, fallback to finite rows.
        rows_by_gamma: Dict[int, List[Dict[str, Any]]] = {g: [] for g in GAMMAS}
        for r in rows:
            if r.get("gamma") is None:
                continue
            g = int(float(r["gamma"]))
            if g in rows_by_gamma and _is_finite_feasible(r):
                rows_by_gamma[g].append(r)

        if any(len(rows_by_gamma[g]) == 0 for g in GAMMAS):
            rows_by_gamma = {g: [] for g in GAMMAS}
            for r in rows:
                if r.get("gamma") is None:
                    continue
                g = int(float(r["gamma"]))
                try:
                    c = float(r.get("best_cost", float("inf")))
                except (TypeError, ValueError):
                    c = float("inf")
                if g in rows_by_gamma and math.isfinite(c):
                    rows_by_gamma[g].append(r)

        if any(len(rows_by_gamma[g]) == 0 for g in GAMMAS):
            raise ValueError(f"{trial_file.name}: missing candidates for one or more gamma levels.")

        selected, mode = _select_combo(rows_by_gamma)

        # Build per-instance best CSV rows.
        best_rows_for_file: list[Dict[str, Any]] = []
        selected = sorted(selected, key=lambda r: int(float(r["gamma"])))
        for r in selected:
            best_rows_for_file.append(
                {
                    "instance": r.get("instance", ""),
                    "gamma": int(float(r["gamma"])),
                    "best_seed": int(float(r.get("seed", -1))),
                    "best_cost": float(r.get("best_cost", float("inf"))),
                    "feasible": int(float(r.get("feasible", 0))),
                    "best_drone_customers": float(r.get("best_drone_customers", 0)),
                    "truck_distance_cost": r.get("truck_distance_cost", ""),
                    "drone_distance_cost": r.get("drone_distance_cost", ""),
                    "truck_routes": r.get("truck_routes", ""),
                    "drone_tasks": r.get("drone_tasks", ""),
                    "error": r.get("error", ""),
                }
            )
            all_best_rows.append(best_rows_for_file[-1])

        stem = trial_file.name.replace("_trials.csv", "")
        out_best_path = out_dir / f"{stem}_best.csv"
        _write_csv(
            out_best_path,
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
            best_rows_for_file,
        )

        costs = [float(r["best_cost"]) for r in best_rows_for_file]
        drones = [float(r["best_drone_customers"]) for r in best_rows_for_file]
        selection_logs.append(
            {
                "instance": Path(str(best_rows_for_file[0]["instance"])).stem,
                "selection_mode": mode,
                "g0_cost": costs[0],
                "g1_cost": costs[1],
                "g2_cost": costs[2],
                "g3_cost": costs[3],
                "g0_drone": drones[0],
                "g1_drone": drones[1],
                "g2_drone": drones[2],
                "g3_drone": drones[3],
                "cost_monotone_ok": int(costs[0] <= costs[1] <= costs[2] <= costs[3]),
                "drone_noninc_ok": int(drones[0] >= drones[1] >= drones[2] >= drones[3]),
            }
        )

    # Global all-in-one best file.
    all_best_rows = sorted(all_best_rows, key=lambda r: (str(r["instance"]), int(r["gamma"])))
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
        all_best_rows,
    )

    # Region-level best_all files.
    df_best = pd.DataFrame(all_best_rows)
    df_best["instance_name"] = df_best["instance"].map(lambda p: Path(str(p)).stem)
    df_best["region"] = df_best["instance_name"].str.split("_").str[1]
    for region in ["30", "40", "50"]:
        sub = df_best[df_best["region"] == region].copy()
        sub = sub.drop(columns=["instance_name", "region"])
        sub = sub.sort_values(["instance", "gamma"])
        sub.to_csv(out_dir / f"instance25_r{region}_gamma_best_all.csv", index=False)

    pd.DataFrame(selection_logs).sort_values("instance").to_csv(
        out_dir / "monotonic_selection_log.csv", index=False
    )

    print("=" * 96)
    print("Monotonic best selection completed")
    print("=" * 96)
    print(f"trial files: {len(trial_files)}")
    print(f"output_dir: {out_dir}")
    print(f"log: {out_dir / 'monotonic_selection_log.csv'}")


if __name__ == "__main__":
    main()

