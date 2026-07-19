"""Build per-instance paired same/flexible route CSV from docking comparison results."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _instance_name(instance_path: str) -> str:
    return Path(instance_path).stem


def _instance_region(instance_path: str) -> int | None:
    match = re.match(r"R_(\d+)_", _instance_name(instance_path))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build best-by-instance paired routes CSV for same vs flexible docking."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input results CSV from sensitivity/docking_flexibility_comparison.py",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output paired best-by-instance CSV path.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def build_paired_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: dict[Tuple[str, int], dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        instance = row.get("instance", "")
        seed = _to_int(row.get("seed"))
        mode = str(row.get("mode", ""))
        if not instance or seed is None:
            continue
        if mode not in {"same_truck", "flexible"}:
            continue
        grouped[(instance, seed)][mode] = row

    paired_rows: list[Dict[str, Any]] = []
    for (instance, seed), mode_rows in sorted(grouped.items()):
        same_row = mode_rows.get("same_truck")
        flex_row = mode_rows.get("flexible")
        if same_row is None or flex_row is None:
            continue
        same_cost = _to_float(same_row.get("best_cost"))
        flex_cost = _to_float(flex_row.get("best_cost"))
        saving = math.nan
        if math.isfinite(same_cost) and same_cost > 0 and math.isfinite(flex_cost):
            saving = (same_cost - flex_cost) / same_cost * 100.0

        paired_rows.append(
            {
                "instance": instance,
                "instance_name": _instance_name(instance),
                "region": _instance_region(instance),
                "same_seed": seed,
                "flex_seed": seed,
                "same_cost": same_cost,
                "flexible_cost": flex_cost,
                "flexible_saving_vs_same": saving,
                "same_best_drone_customers": _to_float(same_row.get("best_drone_customers")),
                "flex_best_drone_customers": _to_float(flex_row.get("best_drone_customers")),
                "same_truck_routes": same_row.get("truck_routes", ""),
                "same_drone_tasks": same_row.get("drone_tasks", ""),
                "flexible_truck_routes": flex_row.get("truck_routes", ""),
                "flexible_drone_tasks": flex_row.get("drone_tasks", ""),
            }
        )
    return paired_rows


def select_best_by_instance(paired_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_instance: dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in paired_rows:
        by_instance[str(row["instance"])].append(row)

    selected: list[Dict[str, Any]] = []
    for instance, rows in sorted(by_instance.items()):
        valid = [r for r in rows if math.isfinite(_to_float(r.get("flexible_saving_vs_same")))]
        if valid:
            best = max(
                valid,
                key=lambda r: (
                    _to_float(r.get("flexible_saving_vs_same")),
                    -_to_float(r.get("flexible_cost")),
                ),
            )
        else:
            best = min(rows, key=lambda r: _to_float(r.get("flexible_cost")))
        selected.append(best)
    return selected


def write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "instance",
        "instance_name",
        "region",
        "same_seed",
        "flex_seed",
        "same_cost",
        "flexible_cost",
        "flexible_saving_vs_same",
        "same_best_drone_customers",
        "flex_best_drone_customers",
        "same_truck_routes",
        "same_drone_tasks",
        "flexible_truck_routes",
        "flexible_drone_tasks",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    rows = load_rows(in_path)
    paired_rows = build_paired_rows(rows)
    best_rows = select_best_by_instance(paired_rows)
    write_rows(out_path, best_rows)
    print(f"Input: {in_path}")
    print(f"Paired rows: {len(paired_rows)}")
    print(f"Best-by-instance rows: {len(best_rows)}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
