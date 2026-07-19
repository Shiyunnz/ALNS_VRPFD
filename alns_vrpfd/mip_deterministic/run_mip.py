# import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent
for _p in [Path(__file__).resolve().parent] + list(Path(__file__).resolve().parents):
    if (_p / 'run_alns.py').exists():
        _project_root = _p
        break
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
del _p, _project_root
from alns_vrpfd.mip_deterministic.builder import MIPArtifacts, build_mip_model, set_makespan_objective
from alns_vrpfd.utils.io_utils import read_instance
from typing import Any, Dict, List
import time
import json

import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

__doc__ = """Solve the simplified truck-drone MILP formulation with Gurobi."""


RUNS: List[Dict[str, Any]] = [
    {
        "instance": "data/Instance10/R_30_10_5.txt",
        "time_window_strategy": "demand_based",
        "bigm_time": 1000,
        "bigm_load": 1000,
        "drone_endurance": 0.5,
        "time_limit": 3600,
        "notes": "Simplified MILP with makespan objective (extended time)",
    },
    {
        "instance": "data/Instance10/R_30_10_5.txt",
        "time_window_strategy": "demand_based",
        "bigm_time": 1000,
        "bigm_load": 1000,
        "drone_endurance": 0.5,
        "time_limit": 300,
        "notes": "Simplified MILP with distance objective",
        "objective": "distance",
    },
]

OUTPUT_PATH = Path("results") / "mip_simple_runs.json"


class SolverError(RuntimeError):
    """Exception raised when the MILP solve fails."""


def _reconstruct_routes(arcs: list[tuple[int, int]], arrival_times: dict = None) -> list[list[int]]:
    """Reconstruct one or more routes from a set of arcs, ordered by arrival time if provided."""
    if not arcs:
        return []

    successors = {i: j for i, j in arcs}
    predecessors = {j: i for i, j in arcs}

    all_nodes = set()
    for i, j in arcs:
        all_nodes.add(i)
        all_nodes.add(j)

    start_candidates = [node for node in all_nodes if node not in predecessors]

    temp_routes = []
    for start_node in start_candidates:
        route = [start_node]
        current = start_node
        while current in successors:
            nxt = successors[current]
            route.append(nxt)
            current = nxt
            if len(route) > len(arcs) + 2:
                break
        temp_routes.append(route)

    if arrival_times:
        temp_routes.sort(key=lambda r: arrival_times.get(r[-1], 0))

    return temp_routes


def run_single_mip(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single simplified MILP experiment with the given configuration."""
    instance_path = Path(config["instance"])
    print(f"\n=== Running Simplified MILP on {instance_path} ===")

    if not instance_path.exists():
        raise FileNotFoundError(
            f"Instance file {instance_path} does not exist.")

    start_time = time.time()

    instance = read_instance(
        str(instance_path), strategy=config["time_window_strategy"])

    artifacts = build_mip_model(
        instance,
        big_m_time=config.get("bigm_time", 1e5),
        big_m_load=config.get("bigm_load", 1e5),
        drone_endurance=config.get("drone_endurance", 0.5),
        objective=config.get("objective", "makespan"),
    )

    model = artifacts.model
    if config.get("time_limit"):
        model.setParam("TimeLimit", config["time_limit"])

    model.optimize()

    solve_time = time.time() - start_time

    result = {
        "instance": str(instance_path),
        "config": config,
        "solve_time": solve_time,
        "status": model.Status,
    }

    if model.SolCount == 0:
        result["error"] = "No feasible solution found"
        print(f"ERROR: No feasible solution found (status: {model.Status})")
        return result

    objective = model.ObjVal
    result["objective"] = objective

    print(f"Solution found in {solve_time:.2f}s")
    print(f"Objective value: {objective:.3f}")

    data = artifacts.data
    vars = artifacts.variables

    truck_routes = []
    for k in data.trucks:
        arcs = [
            (i, j)
            for (i, j) in data.arcs
            if vars.x_truck[i, j, k].X > 0.5
        ]
        if arcs:
            routes = _reconstruct_routes(arcs, None)
            route = routes[0] if routes else []
            distance = sum(data.truck_distance[(i, j)] for i, j in arcs)
            truck_routes.append({
                "truck": k,
                "route": route,
                "distance": distance
            })
            print(f"Truck {k}: route {route}, distance {distance:.3f}")

    drone_tasks = []
    for d in data.drones:
        arcs = [
            (i, j)
            for (i, j) in data.arcs
            if vars.y_drone[i, j, d].X > 0.5
        ]
        if arcs:
            arrival_times = {
                i: vars.arrival_drone[i, d].X
                for i in data.nodes
            }
            routes = _reconstruct_routes(arcs, arrival_times)
            routes = [r for r in routes if len(r) > 2]

            if not routes:
                print(f"Drone {d}: idle")
                continue

            distance = sum(data.drone_distance[(i, j)] for i, j in arcs)
            total_time = sum(data.drone_time[(i, j)] for i, j in arcs)
            drone_tasks.append({
                "drone": d,
                "routes": routes,
                "distance": distance,
                "total_time": total_time
            })
            routes_with_times = []
            for route in routes:
                start_node = route[0]
                arrival_time = arrival_times.get(start_node, 0)
                routes_with_times.append(f"{route} (start@{arrival_time:.2f})")
            routes_str = ", ".join(routes_with_times)
            print(
                f"Drone {d}: routes {routes_str}, total distance {distance:.3f}, total time {total_time:.3f}h"
            )
        else:
            print(f"Drone {d}: idle")

    result["truck_routes"] = truck_routes
    result["drone_tasks"] = drone_tasks

    return result


def run_experiments() -> None:
    """Run all configured simplified MILP experiments."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_results: List[Dict[str, Any]] = []

    for config in RUNS:
        try:
            result = run_single_mip(config)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR running {config['instance']}: {e}")
            all_results.append({
                "instance": config["instance"],
                "config": config,
                "error": str(e)
            })

    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2)
    print(f"\nResults written to {OUTPUT_PATH}")


def _report_solution(artifacts: MIPArtifacts) -> None:
    data = artifacts.data
    vars = artifacts.variables
    model = artifacts.model

    print(f"Solve status: {model.Status}")
    if model.SolCount == 0:
        raise SolverError("No feasible solution found by Gurobi.")

    objective = model.ObjVal
    print(f"Objective value: {objective:.3f}")

    for k in data.trucks:
        arcs = [
            (i, j)
            for (i, j) in data.arcs
            if vars.x_truck[i, j, k].X > 0.5
        ]
        if arcs:
            routes = _reconstruct_routes(arcs, None)
            route = routes[0] if routes else []
            distance = sum(data.truck_distance[(i, j)] for i, j in arcs)
            print(f"Truck {k}: route {route}, distance {distance:.3f}")

    for d in data.drones:
        arcs = [
            (i, j)
            for (i, j) in data.arcs
            if vars.y_drone[i, j, d].X > 0.5
        ]
        if not arcs:
            print(f"Drone {d}: idle")
            continue
        arrival_times = {
            i: vars.arrival_drone[i, d].X
            for i in data.nodes
        }
        routes = _reconstruct_routes(arcs, arrival_times)
        distance = sum(data.drone_distance[(i, j)] for i, j in arcs)
        routes_str = ", ".join(str(r) for r in routes)
        print(f"Drone {d}: routes {routes_str}, distance {distance:.3f}")


if __name__ == "__main__":
    try:
        run_experiments()
    except SolverError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)