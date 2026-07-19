"""Summarize travel-time and generated-deadline horizons for benchmark instances."""

from __future__ import annotations

from pathlib import Path
import json
import math
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from alns_vrpfd.utils.io_utils import read_instance  # noqa: E402


def _standard_instance_files() -> list[Path]:
    files: list[Path] = []
    for size in [10, 25, 50, 75, 100]:
        files.extend(sorted((PROJECT_ROOT / "data" / f"Instance{size}").glob("R_*_*.txt")))
    return files


def _matrix_max(matrix: list[list[float]]) -> float:
    finite_values = [value for row in matrix for value in row if math.isfinite(value)]
    return max(finite_values)


def main() -> None:
    records = []
    for path in _standard_instance_files():
        instance = read_instance(str(path), strategy="class_based")
        node_ids = instance.all_node_ids()
        depot_id = instance.customer_manager.depot_start
        depot_idx = node_ids.index(depot_id)
        truck_times = instance.time_matrix("truck")
        drone_times = instance.time_matrix("drone")
        customers = list(instance.customer_manager.customers())
        latest_times = [c.latest_time for c in customers if c.latest_time is not None]
        optimal_times = [c.optimal_time for c in customers if c.optimal_time is not None]
        max_allowed_tardiness = max(
            (c.latest_time or 0.0) - (c.optimal_time or 0.0)
            for c in customers
            if c.latest_time is not None and c.optimal_time is not None
        )
        reachable = [
            min(truck_times[depot_idx][node_ids.index(c.customer_id)], drone_times[depot_idx][node_ids.index(c.customer_id)])
            for c in customers
        ]
        records.append(
            {
                "instance": path.stem,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "max_truck_arc_time_h": _matrix_max(truck_times),
                "max_drone_arc_time_h": _matrix_max(drone_times),
                "max_depot_reachable_time_h": max(reachable),
                "max_optimal_deadline_h": max(optimal_times),
                "max_hard_deadline_h": max(latest_times),
                "max_allowed_tardiness_h": max_allowed_tardiness,
            }
        )

    summary = {
        "num_instances": len(records),
        "max_truck_arc_time_h": max(records, key=lambda r: r["max_truck_arc_time_h"]),
        "max_drone_arc_time_h": max(records, key=lambda r: r["max_drone_arc_time_h"]),
        "max_depot_reachable_time_h": max(records, key=lambda r: r["max_depot_reachable_time_h"]),
        "max_optimal_deadline_h": max(records, key=lambda r: r["max_optimal_deadline_h"]),
        "max_hard_deadline_h": max(records, key=lambda r: r["max_hard_deadline_h"]),
        "max_allowed_tardiness_h": max(records, key=lambda r: r["max_allowed_tardiness_h"]),
    }
    output_path = PROJECT_ROOT / "results" / "instance_time_horizon_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"summary": summary, "records": records}, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
