"""Operational metrics for the flexible-docking comparison.

The script re-evaluates saved same-truck and flexible-docking solutions from the
existing comparison CSV and derives operational metrics requested in revision:
fleet usage, synchronization waiting, delivery tardiness, workload balance, and
instance density/clustering descriptors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alns_vrpfd.evaluation import Evaluator
from alns_vrpfd.model.route import DroneTask, TruckRoute
from alns_vrpfd.model.solution import Solution
from alns_vrpfd.utils.config_loader import ALNSConfig
from alns_vrpfd.utils.io_utils import read_instance


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "sensitivity/results_new/drone_flexibility/"
    / "docking_flexibility_i25_system_t3_best_by_instance.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/revision_experiments/flexible_docking_metrics"


@dataclass(frozen=True)
class MetricRow:
    instance: str
    region: int
    mode: str
    cost: float
    cost_saving_pct: float
    available_trucks: int
    available_drones: int
    active_trucks: int
    used_drones: int
    drone_tasks: int
    drone_customers: int
    cross_truck_tasks: int
    depot_retrieval_tasks: int
    makespan_h: float
    total_truck_duration_h: float
    total_drone_flight_h: float
    truck_utilization: float
    drone_utilization: float
    truck_wait_h: float
    drone_wait_h: float
    sync_delay_h: float
    delay_cost: float
    total_tardiness_h: float
    mean_tardiness_h: float
    max_tardiness_h: float
    delayed_customers: int
    truck_workload_cv: float
    truck_customer_cv: float
    customer_density: float
    mean_nearest_neighbor_km: float
    clustering_index: float


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _mean(values: Iterable[float]) -> float:
    values = [v for v in values if math.isfinite(v)]
    return sum(values) / len(values) if values else 0.0


def _std(values: Iterable[float]) -> float:
    values = [v for v in values if math.isfinite(v)]
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def _cv(values: Iterable[float]) -> float:
    values = [v for v in values if math.isfinite(v)]
    avg = _mean(values)
    if avg <= 1e-12:
        return 0.0
    return _std(values) / avg


def _corr(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) <= 1:
        return math.nan
    x_vals, y_vals = zip(*pairs)
    x_avg = _mean(x_vals)
    y_avg = _mean(y_vals)
    num = sum((x - x_avg) * (y - y_avg) for x, y in pairs)
    x_den = math.sqrt(sum((x - x_avg) ** 2 for x in x_vals))
    y_den = math.sqrt(sum((y - y_avg) ** 2 for y in y_vals))
    if x_den <= 1e-12 or y_den <= 1e-12:
        return math.nan
    return num / (x_den * y_den)


def _rank(values: list[float]) -> list[float]:
    indexed = sorted((v, i) for i, v in enumerate(values))
    ranks = [math.nan] * len(values)
    pos = 0
    while pos < len(indexed):
        end = pos + 1
        while end < len(indexed) and indexed[end][0] == indexed[pos][0]:
            end += 1
        avg_rank = (pos + 1 + end) / 2.0
        for _, original_index in indexed[pos:end]:
            ranks[original_index] = avg_rank
        pos = end
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) <= 1:
        return math.nan
    x_vals = [p[0] for p in pairs]
    y_vals = [p[1] for p in pairs]
    return _corr(_rank(x_vals), _rank(y_vals))


def _solution_from_json(truck_json: str, drone_json: str, instance: Any) -> Solution:
    truck_payload = json.loads(truck_json)
    drone_payload = json.loads(drone_json)
    truck_capacity = float(instance.vehicle_specs["truck"].capacity)
    demands = instance.customer_manager.demands()
    solution = Solution.empty()
    for route_data in truck_payload:
        nodes = [int(node) for node in route_data["nodes"]]
        load = sum(float(demands.get(node, 0.0)) for node in nodes[1:-1])
        solution.add_truck_route(
            TruckRoute(
                route_id=int(route_data["truck_id"]),
                nodes=nodes,
                capacity=truck_capacity,
                current_load=load,
            )
        )
    for task_id, task_data in enumerate(drone_payload):
        solution.add_drone_task(
            DroneTask(
                drone_id=int(task_data.get("drone_id", 0)),
                launch_truck=(
                    None
                    if task_data.get("launch_truck") is None
                    else int(task_data["launch_truck"])
                ),
                launch_node=int(task_data["launch_node"]),
                customers=[int(node) for node in task_data.get("customers", [])],
                land_truck=(
                    None
                    if task_data.get("land_truck") is None
                    else int(task_data["land_truck"])
                ),
                retrieve_node=int(task_data["retrieve_node"]),
                task_id=task_id,
            )
        )
    return solution


def _build_evaluator(instance: Any, cfg: ALNSConfig) -> Evaluator:
    return Evaluator(
        instance,
        rendezvous_tolerance=cfg.drone_rendezvous_tolerance,
        forced_drone_customers=cfg.forced_drone_customers,
        allow_multiple_launch_per_node=cfg.relax_allow_multiple_launch_per_node,
        cost_lambda=cfg.cost_lambda,
        cost_rho=cfg.cost_rho,
        cost_normalized=cfg.cost_normalized,
    )


def _configure_instance(instance: Any, same_truck_retrieval: bool, cfg: ALNSConfig) -> None:
    instance.configure_robustness(
        drone_battery_capacity=cfg.drone_battery_capacity,
        energy_uncertainty_budget=cfg.energy_uncertainty_budget,
        energy_deviation_rate=cfg.energy_deviation_rate,
        same_truck_retrieval=same_truck_retrieval,
    )


def _instance_descriptors(instance: Any, region: int) -> tuple[float, float, float]:
    customers = [
        customer
        for customer in instance.customer_manager.customers()
        if customer.customer_id not in {
            instance.customer_manager.depot_start,
            instance.customer_manager.depot_end,
        }
    ]
    n_customers = len(customers)
    density = n_customers / float(region * region) if region > 0 else math.nan
    coords = [(float(c.location_x), float(c.location_y)) for c in customers]
    nn_distances: list[float] = []
    for i, (x_i, y_i) in enumerate(coords):
        distances = [
            math.hypot(x_i - x_j, y_i - y_j)
            for j, (x_j, y_j) in enumerate(coords)
            if i != j
        ]
        if distances:
            nn_distances.append(min(distances))
    mean_nn = _mean(nn_distances)
    expected_random_nn = 0.5 / math.sqrt(density) if density > 0 else math.nan
    clustering_index = mean_nn / expected_random_nn if expected_random_nn > 0 else math.nan
    return density, mean_nn, clustering_index


def _task_timing_key(task: DroneTask, index: int, tasks: list[DroneTask]) -> int:
    task_id = task.task_id
    if task_id is None:
        return index
    if sum(1 for other in tasks if other.task_id == task_id) == 1:
        return int(task_id)
    return index


def _extract_metrics(
    *,
    instance_path: Path,
    instance_name: str,
    region: int,
    mode: str,
    cost: float,
    saving_pct: float,
    truck_json: str,
    drone_json: str,
    same_truck_retrieval: bool,
    cfg: ALNSConfig,
) -> MetricRow:
    instance = read_instance(str(instance_path), strategy="class_based")
    if "drone" in instance.vehicle_specs:
        instance.vehicle_specs["drone"].endurance = float("inf")
    _configure_instance(instance, same_truck_retrieval, cfg)
    solution = _solution_from_json(truck_json, drone_json, instance)
    evaluator = _build_evaluator(instance, cfg)
    details = evaluator.evaluate_with_details(solution)

    truck_durations = [
        float(details.truck_timings.get(route.id).total_duration)
        for route in solution.truck_routes
        if details.truck_timings.get(route.id) is not None
    ]
    route_customer_counts = [len(route.customers()) for route in solution.truck_routes]
    active_trucks = sum(1 for count in route_customer_counts if count > 0)
    available_trucks = int(instance.vehicle_specs["truck"].number)
    available_drones = int(instance.vehicle_specs["drone"].number)
    used_drones = len({task.drone_id for task in solution.drone_tasks})
    drone_customers = len({node for task in solution.drone_tasks for node in task.customers()})
    cross_truck_tasks = sum(
        1
        for task in solution.drone_tasks
        if task.launch_truck is not None
        and task.land_truck is not None
        and task.launch_truck != task.land_truck
    )
    depot_retrieval_tasks = sum(1 for task in solution.drone_tasks if task.land_truck is None)
    makespan = max(truck_durations) if truck_durations else 0.0
    total_truck_duration = sum(truck_durations)

    total_drone_flight = 0.0
    truck_wait = 0.0
    drone_wait = 0.0
    for index, task in enumerate(solution.drone_tasks):
        key = _task_timing_key(task, index, solution.drone_tasks)
        timing = details.drone_timings.get(key)
        if timing is not None:
            total_drone_flight += max(0.0, float(timing.retrieve_time - timing.launch_time))
        if task.land_truck is None or timing is None:
            continue
        truck_timing = details.truck_timings.get(task.land_truck)
        if truck_timing is None:
            continue
        truck_arrival = truck_timing.arrival_times.get(task.retrieve_node)
        if truck_arrival is None:
            continue
        truck_wait += max(0.0, float(timing.retrieve_time - truck_arrival))
        drone_wait += max(0.0, float(truck_arrival - timing.retrieve_time))

    truck_utilization = (
        total_truck_duration / (available_trucks * makespan)
        if available_trucks > 0 and makespan > 0
        else 0.0
    )
    drone_utilization = (
        total_drone_flight / (available_drones * makespan)
        if available_drones > 0 and makespan > 0
        else 0.0
    )
    tardiness_values = [float(node.delay) for node in details.delay_breakdown.nodes]
    density, mean_nn, clustering_index = _instance_descriptors(instance, region)

    return MetricRow(
        instance=instance_name,
        region=int(region),
        mode=mode,
        cost=cost,
        cost_saving_pct=saving_pct,
        available_trucks=available_trucks,
        available_drones=available_drones,
        active_trucks=active_trucks,
        used_drones=used_drones,
        drone_tasks=len(solution.drone_tasks),
        drone_customers=drone_customers,
        cross_truck_tasks=cross_truck_tasks,
        depot_retrieval_tasks=depot_retrieval_tasks,
        makespan_h=makespan,
        total_truck_duration_h=total_truck_duration,
        total_drone_flight_h=total_drone_flight,
        truck_utilization=truck_utilization,
        drone_utilization=drone_utilization,
        truck_wait_h=truck_wait,
        drone_wait_h=drone_wait,
        sync_delay_h=truck_wait + drone_wait,
        delay_cost=float(details.result.delay_penalty),
        total_tardiness_h=sum(tardiness_values),
        mean_tardiness_h=_mean(tardiness_values),
        max_tardiness_h=max(tardiness_values) if tardiness_values else 0.0,
        delayed_customers=len(tardiness_values),
        truck_workload_cv=_cv(truck_durations),
        truck_customer_cv=_cv(route_customer_counts),
        customer_density=density,
        mean_nearest_neighbor_km=mean_nn,
        clustering_index=clustering_index,
    )


def _read_paired_rows(input_csv: Path) -> list[dict[str, Any]]:
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _round_row(row: dict[str, Any]) -> dict[str, Any]:
    rounded = {}
    for key, value in row.items():
        if isinstance(value, float):
            rounded[key] = round(value, 6)
        else:
            rounded[key] = value
    return rounded


def _aggregate_by_mode(metrics: list[MetricRow]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[MetricRow]] = {}
    for row in metrics:
        groups.setdefault(("ALL", row.mode), []).append(row)
        groups.setdefault((f"R{row.region}", row.mode), []).append(row)

    fields = [
        "cost",
        "cost_saving_pct",
        "active_trucks",
        "used_drones",
        "drone_tasks",
        "drone_customers",
        "cross_truck_tasks",
        "makespan_h",
        "truck_utilization",
        "drone_utilization",
        "truck_wait_h",
        "drone_wait_h",
        "sync_delay_h",
        "delay_cost",
        "total_tardiness_h",
        "delayed_customers",
        "truck_workload_cv",
        "truck_customer_cv",
    ]
    out: list[dict[str, Any]] = []
    for (group, mode), rows in sorted(groups.items()):
        item: dict[str, Any] = {"group": group, "mode": mode, "n": len(rows)}
        for field in fields:
            values = [float(getattr(row, field)) for row in rows]
            item[f"mean_{field}"] = _mean(values)
            item[f"std_{field}"] = _std(values)
        out.append(_round_row(item))
    return out


def _paired_differences(metrics: list[MetricRow]) -> list[dict[str, Any]]:
    by_instance: dict[str, dict[str, MetricRow]] = {}
    for row in metrics:
        by_instance.setdefault(row.instance, {})[row.mode] = row
    fields = [
        "cost",
        "active_trucks",
        "used_drones",
        "drone_tasks",
        "drone_customers",
        "cross_truck_tasks",
        "makespan_h",
        "truck_utilization",
        "drone_utilization",
        "truck_wait_h",
        "drone_wait_h",
        "sync_delay_h",
        "delay_cost",
        "total_tardiness_h",
        "delayed_customers",
        "truck_workload_cv",
        "truck_customer_cv",
    ]
    out: list[dict[str, Any]] = []
    for instance_name, rows in sorted(by_instance.items()):
        if "same_truck" not in rows or "flexible" not in rows:
            continue
        same = rows["same_truck"]
        flex = rows["flexible"]
        item: dict[str, Any] = {
            "instance": instance_name,
            "region": flex.region,
            "saving_pct": flex.cost_saving_pct,
            "customer_density": flex.customer_density,
            "mean_nearest_neighbor_km": flex.mean_nearest_neighbor_km,
            "clustering_index": flex.clustering_index,
        }
        for field in fields:
            item[f"same_{field}"] = getattr(same, field)
            item[f"flex_{field}"] = getattr(flex, field)
            item[f"delta_{field}"] = getattr(flex, field) - getattr(same, field)
        out.append(_round_row(item))
    return out


def _correlation_table(paired_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    y = [_float(row["saving_pct"]) for row in paired_rows]
    variables = [
        ("customer_density", "Customer density"),
        ("mean_nearest_neighbor_km", "Mean nearest-neighbor distance"),
        ("clustering_index", "Clustering index"),
        ("delta_cross_truck_tasks", "Change in cross-truck retrieval tasks"),
        ("delta_drone_customers", "Change in drone-served customers"),
        ("delta_sync_delay_h", "Change in synchronization delay"),
        ("delta_truck_wait_h", "Change in truck waiting time"),
        ("delta_drone_utilization", "Change in drone utilization"),
        ("delta_truck_workload_cv", "Change in truck workload CV"),
    ]
    out = []
    for key, label in variables:
        x = [_float(row[key]) for row in paired_rows]
        out.append(
            _round_row(
                {
                    "variable": key,
                    "description": label,
                    "pearson_r": _corr(x, y),
                    "spearman_rho": _spearman(x, y),
                }
            )
        )
    return out


def _latex_number(value: float, digits: int = 2) -> str:
    if not math.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def _write_latex_tables(
    output_path: Path,
    aggregate_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
) -> None:
    all_rows = {row["mode"]: row for row in aggregate_rows if row["group"] == "ALL"}
    same = all_rows["same_truck"]
    flex = all_rows["flexible"]
    region_flex = [row for row in aggregate_rows if row["mode"] == "flexible" and row["group"] != "ALL"]
    corr_lookup = {row["variable"]: row for row in correlations}

    lines = [
        "% Auto-generated by scripts/analyze_flexible_docking_operational_metrics.py",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Operational metrics of fixed and flexible docking solutions.}",
        "\\label{tab:flexible_docking_operational_metrics}",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Metric & Fixed docking & Flexible docking \\\\",
        "\\midrule",
        f"Total cost & {_latex_number(same['mean_cost'])} & {_latex_number(flex['mean_cost'])} \\\\",
        f"Active trucks & {_latex_number(same['mean_active_trucks'])} & {_latex_number(flex['mean_active_trucks'])} \\\\",
        f"Used drones & {_latex_number(same['mean_used_drones'])} & {_latex_number(flex['mean_used_drones'])} \\\\",
        f"Drone-served customers & {_latex_number(same['mean_drone_customers'])} & {_latex_number(flex['mean_drone_customers'])} \\\\",
        f"Cross-truck retrieval tasks & {_latex_number(same['mean_cross_truck_tasks'])} & {_latex_number(flex['mean_cross_truck_tasks'])} \\\\",
        f"Makespan (h) & {_latex_number(same['mean_makespan_h'])} & {_latex_number(flex['mean_makespan_h'])} \\\\",
        f"Truck utilization & {_latex_number(100 * same['mean_truck_utilization'])}\\% & {_latex_number(100 * flex['mean_truck_utilization'])}\\% \\\\",
        f"Drone utilization & {_latex_number(100 * same['mean_drone_utilization'])}\\% & {_latex_number(100 * flex['mean_drone_utilization'])}\\% \\\\",
        f"Truck waiting time (h) & {_latex_number(same['mean_truck_wait_h'])} & {_latex_number(flex['mean_truck_wait_h'])} \\\\",
        f"Drone waiting time (h) & {_latex_number(same['mean_drone_wait_h'])} & {_latex_number(flex['mean_drone_wait_h'])} \\\\",
        f"Total tardiness (h) & {_latex_number(same['mean_total_tardiness_h'])} & {_latex_number(flex['mean_total_tardiness_h'])} \\\\",
        f"Truck workload CV & {_latex_number(same['mean_truck_workload_cv'])} & {_latex_number(flex['mean_truck_workload_cv'])} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Flexible-docking savings and instance characteristics by region.}",
        "\\label{tab:flexible_docking_instance_characteristics}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Region & Saving (\\%) & Density & NN distance & Clustering index & Cross-truck tasks \\\\",
        "\\midrule",
    ]
    by_region: dict[int, list[dict[str, Any]]] = {}
    for row in paired_rows:
        by_region.setdefault(int(row["region"]), []).append(row)
    for row in sorted(region_flex, key=lambda item: item["group"]):
        region = int(row["group"].replace("R", ""))
        paired = by_region[region]
        lines.append(
            f"$R{region}$ & "
            f"{_latex_number(row['mean_cost_saving_pct'])} & "
            f"{_latex_number(_mean([_float(p['customer_density']) for p in paired]), 4)} & "
            f"{_latex_number(_mean([_float(p['mean_nearest_neighbor_km']) for p in paired]))} & "
            f"{_latex_number(_mean([_float(p['clustering_index']) for p in paired]))} & "
            f"{_latex_number(row['mean_cross_truck_tasks'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
            "% Selected correlations with flexible-docking savings:",
            f"% Density: Pearson {corr_lookup['customer_density']['pearson_r']}, Spearman {corr_lookup['customer_density']['spearman_rho']}",
            f"% NN distance: Pearson {corr_lookup['mean_nearest_neighbor_km']['pearson_r']}, Spearman {corr_lookup['mean_nearest_neighbor_km']['spearman_rho']}",
            f"% Cross-truck tasks: Pearson {corr_lookup['delta_cross_truck_tasks']['pearson_r']}, Spearman {corr_lookup['delta_cross_truck_tasks']['spearman_rho']}",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_appendix_text(
    output_path: Path,
    aggregate_rows: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
) -> None:
    all_rows = {row["mode"]: row for row in aggregate_rows if row["group"] == "ALL"}
    same = all_rows["same_truck"]
    flex = all_rows["flexible"]
    corr = {row["variable"]: row for row in correlations}
    text = f"""\\subsection{{Additional operational analysis of flexible docking}}

Table~\\ref{{tab:flexible_docking_operational_metrics}} reports additional operational metrics for the fixed-docking and flexible-docking solutions used in Section~\\ref{{sec:experiments}}. The metrics are computed from the final routes of the 15 paired $n=25$ instances. The number of active trucks counts routes serving at least one customer, whereas the number of used drones counts drone identifiers appearing in at least one sortie. Truck utilization is defined as the sum of truck route durations divided by the product of the available truck fleet size and the solution makespan. Drone utilization is defined analogously using total sortie flight time and the available drone fleet size. Synchronization delay is decomposed into truck waiting time, when a retrieval truck arrives before the drone, and drone waiting time, when the drone reaches the rendezvous point before the truck.

The additional metrics confirm that the cost advantage of flexible docking is associated with greater exploitation of drone operations and cross-truck recovery opportunities. Compared with fixed docking, flexible docking increases the average number of drone-served customers from {same['mean_drone_customers']:.2f} to {flex['mean_drone_customers']:.2f} and introduces {flex['mean_cross_truck_tasks']:.2f} cross-truck retrieval sorties per instance on average, whereas fixed docking has no such sorties by construction. Average drone utilization also increases from {100 * same['mean_drone_utilization']:.2f}\\% to {100 * flex['mean_drone_utilization']:.2f}\\%. These changes indicate that flexible docking does not merely shorten individual routes; it changes the coordination pattern by allowing drones to be recovered by trucks that are better positioned in space and time.

The synchronization metrics provide a more nuanced interpretation. Flexible docking reduces the average objective value but may increase total synchronization delay in some instances because it creates more rendezvous opportunities and more drone-served customers. Therefore, the benefit of flexible docking should not be interpreted as eliminating all waiting. Rather, it replaces restrictive same-truck waiting with more productive cross-truck coordination, which can still involve local waiting but yields lower total travel and service cost. Workload balance also changes: the truck-duration coefficient of variation is {same['mean_truck_workload_cv']:.2f} under fixed docking and {flex['mean_truck_workload_cv']:.2f} under flexible docking, suggesting that flexible docking can reallocate work between trucks instead of uniformly shortening both routes.

Table~\\ref{{tab:flexible_docking_instance_characteristics}} relates the observed savings to instance characteristics. Because all paired $n=25$ instances have the same nominal customer count and truck fleet size, customer density varies mainly with the service-region size. The correlation analysis shows that savings are positively associated with the mean nearest-neighbor distance (Pearson $r={corr['mean_nearest_neighbor_km']['pearson_r']:.2f}$; Spearman $\\rho={corr['mean_nearest_neighbor_km']['spearman_rho']:.2f}$) and negatively associated with customer density (Pearson $r={corr['customer_density']['pearson_r']:.2f}$; Spearman $\\rho={corr['customer_density']['spearman_rho']:.2f}$). This supports the managerial interpretation that flexible docking is more valuable in sparse or geographically dispersed service regions, where the ability to recover a drone by another truck creates more meaningful shortcuts. Savings are also positively related to the increase in cross-truck retrieval tasks (Pearson $r={corr['delta_cross_truck_tasks']['pearson_r']:.2f}$; Spearman $\\rho={corr['delta_cross_truck_tasks']['spearman_rho']:.2f}$), indicating that the measured cost reductions are directly linked to the operational mechanism introduced by flexible docking.
"""
    output_path.write_text(text, encoding="utf-8")


def run(input_csv: Path, output_dir: Path) -> None:
    cfg = ALNSConfig(PROJECT_ROOT / "config/alns_config.yaml")
    paired_rows = _read_paired_rows(input_csv)
    metrics: list[MetricRow] = []
    for row in paired_rows:
        instance_path = PROJECT_ROOT / row["instance"]
        instance_name = row["instance_name"]
        region = int(row["region"])
        saving = _float(row["flexible_saving_vs_same"])
        metrics.append(
            _extract_metrics(
                instance_path=instance_path,
                instance_name=instance_name,
                region=region,
                mode="same_truck",
                cost=_float(row["same_cost"]),
                saving_pct=0.0,
                truck_json=row["same_truck_routes"],
                drone_json=row["same_drone_tasks"],
                same_truck_retrieval=True,
                cfg=cfg,
            )
        )
        metrics.append(
            _extract_metrics(
                instance_path=instance_path,
                instance_name=instance_name,
                region=region,
                mode="flexible",
                cost=_float(row["flexible_cost"]),
                saving_pct=saving,
                truck_json=row["flexible_truck_routes"],
                drone_json=row["flexible_drone_tasks"],
                same_truck_retrieval=False,
                cfg=cfg,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    detail_rows = [_round_row(row.__dict__.copy()) for row in metrics]
    detail_fields = list(detail_rows[0].keys())
    _write_rows(output_dir / "flexible_docking_operational_metrics.csv", detail_rows, detail_fields)

    aggregate_rows = _aggregate_by_mode(metrics)
    aggregate_fields = list(aggregate_rows[0].keys())
    _write_rows(output_dir / "flexible_docking_operational_metrics_summary.csv", aggregate_rows, aggregate_fields)

    paired_detail = _paired_differences(metrics)
    paired_fields = list(paired_detail[0].keys())
    _write_rows(output_dir / "flexible_docking_operational_metrics_paired.csv", paired_detail, paired_fields)

    correlations = _correlation_table(paired_detail)
    _write_rows(
        output_dir / "flexible_docking_saving_correlations.csv",
        correlations,
        list(correlations[0].keys()),
    )
    _write_latex_tables(output_dir / "flexible_docking_appendix_tables.tex", aggregate_rows, paired_detail, correlations)
    _write_appendix_text(output_dir / "flexible_docking_appendix_text.tex", aggregate_rows, correlations)

    print(f"Wrote {len(detail_rows)} solution-metric rows to {output_dir}")
    print(f"Wrote {len(paired_detail)} paired rows and {len(correlations)} correlations.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.input_csv, args.output_dir)


if __name__ == "__main__":
    main()
