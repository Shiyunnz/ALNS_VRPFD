import sys
from pathlib import Path

# ：VSCode
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from alns_vrpfd.mip.builder import MIPArtifacts, build_mip_model
from alns_vrpfd.utils.io_utils import read_instance
from alns_vrpfd.utils.config_loader import ALNSConfig
from alns_vrpfd.model import Solution, TruckRoute, DroneTask
from alns_vrpfd.core.operators.base import _build_payloads
from typing import Any, Dict, List, Optional
import time
import json

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

__doc__ = """Solve the truck-drone MILP formulation with Gurobi."""
__doc__ = """Solve the truck-drone MILP formulation with Gurobi."""


# ---------------------------------------------------------------------------
# Configuration block: edit this list to choose the instances and parameters

RUNS: List[Dict[str, Any]] = [
    # ── R30 n=10 ──
    {
        "instance": "data/Instance10/R_30_10_1.txt",
        "time_window_strategy": "class_based",
        "epsilon": 1e-3,
        "energy_budget": 3,
        "robust_energy": True,
        "bigm_time": 1000,
        "bigm_load": 1000,
        "bigm_energy": 20.0,
        "tardiness_weight": 1.0,
        "time_limit": 3600,
        "notes": "R_30_10_1 — 3-segment PWL",
    },
    {
        "instance": "data/Instance10/R_30_10_2.txt",
        "time_window_strategy": "class_based",
        "epsilon": 1e-3,
        "energy_budget": 3,
        "robust_energy": True,
        "bigm_time": 1000,
        "bigm_load": 1000,
        "bigm_energy": 20.0,
        "tardiness_weight": 1.0,
        "time_limit": 3600,
        "notes": "R_30_10_2 — 3-segment PWL",
    },
    {
        "instance": "data/Instance10/R_30_10_3.txt",
        "time_window_strategy": "class_based",
        "epsilon": 1e-3,
        "energy_budget": 3,
        "robust_energy": True,
        "bigm_time": 1000,
        "bigm_load": 1000,
        "bigm_energy": 20.0,
        "tardiness_weight": 1.0,
        "time_limit": 3600,
        "notes": "R_30_10_3 — 3-segment PWL",
    },
    {
        "instance": "data/Instance10/R_30_10_4.txt",
        "time_window_strategy": "class_based",
        "epsilon": 1e-3,
        "energy_budget": 3,
        "robust_energy": True,
        "bigm_time": 1000,
        "bigm_load": 1000,
        "bigm_energy": 20.0,
        "tardiness_weight": 1.0,
        "time_limit": 3600,
        "notes": "R_30_10_4 — 3-segment PWL",
    },
    {
        "instance": "data/Instance10/R_30_10_5.txt",
        "time_window_strategy": "class_based",
        "epsilon": 1e-3,
        "energy_budget": 3,
        "robust_energy": True,
        "bigm_time": 1000,
        "bigm_load": 1000,
        "bigm_energy": 20.0,
        "tardiness_weight": 1.0,
        "time_limit": 3600,
        "notes": "R_30_10_5 — 3-segment PWL",
    },
]

OUTPUT_PATH = Path("results") / "MIPresult_new" / "mip_runs.json"


class SolverError(RuntimeError):
    """Exception raised when the MILP solve fails."""
    """Exception raised when the MILP solve fails."""


def _reconstruct_routes(arcs: list[tuple[int, int]], arrival_times: dict = None) -> list[list[int]]:
    """Reconstruct one or more routes from a set of arcs, ordered by arrival time if provided."""
    """Reconstruct one or more routes from a set of arcs, ordered by arrival time if provided."""
    if not arcs:
        return []

    successors = {i: j for i, j in arcs}
    predecessors = {j: i for i, j in arcs}

    # Find all start nodes (nodes that are not destinations in the arc set)
    all_nodes = set()
    for i, j in arcs:
        all_nodes.add(i)
        all_nodes.add(j)

    start_candidates = [node for node in all_nodes if node not in predecessors]

    # Build routes first, then sort by their end node arrival time
    temp_routes = []
    for start_node in start_candidates:
        route = [start_node]
        current = start_node
        # Follow the path
        while current in successors:
            nxt = successors[current]
            route.append(nxt)
            current = nxt
            # Safety break
            if len(route) > len(arcs) + 2:
                break
        temp_routes.append(route)

    # Sort routes by the arrival time at the END node (last node in route)
    if arrival_times:
        temp_routes.sort(key=lambda r: arrival_times.get(r[-1], 0))

    return temp_routes


def run_single_mip(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single MILP experiment with the given configuration."""
    """Run a single MILP experiment with the given configuration."""
    instance_path = Path(config["instance"])
    print(f"\n=== Running MILP on {instance_path} ===")

    if not instance_path.exists():
        raise FileNotFoundError(
            f"Instance file {instance_path} does not exist.")

    start_time = time.time()

    # Read instance
    instance = read_instance(
        str(instance_path), strategy=config["time_window_strategy"])

    # Load delay cost parameters from ALNS config
    try:
        alns_cfg = ALNSConfig()
        cost_lambda = alns_cfg.cost_lambda
        cost_rho = alns_cfg.cost_rho
        cost_normalized = alns_cfg.cost_normalized
    except Exception:
        cost_lambda = 12.0
        cost_rho = 1.0
        cost_normalized = True

    # Build model with piecewise linear energy constraints
    mip_cfg = config.get('mip', {})
    pwl_delay_segments = mip_cfg.get(
        'piecewise', {}).get('delay_num_segments', None)
    artifacts = build_mip_model(
        instance,
        epsilon=config["epsilon"],
        energy_budget=config["energy_budget"],
        num_segments=10,
        use_gurobi_pwl=True,
        robust_energy=config.get("robust_energy", True),
        big_m_time=config.get("bigm_time", 1000.0),
        big_m_load=config.get("bigm_load", 1000.0),
        # Reduced to prevent constraint relaxation
        big_m_energy=config.get("bigm_energy", 20.0),
        tardiness_weight=config.get("tardiness_weight", 1.0),
        pwl_delay_segments=pwl_delay_segments,
        cost_lambda=cost_lambda,
        cost_rho=cost_rho,
        cost_normalized=cost_normalized,
    )

    # Apply solver parameters
    model = artifacts.model
    if config.get("time_limit"):
        model.setParam("TimeLimit", config["time_limit"])

    # Solve
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

    # Extract solution
    objective = model.ObjVal
    result["objective"] = objective
    result["best_bound"] = getattr(model, "ObjBound", None)
    result["mip_gap"] = getattr(model, "MIPGap", None)
    result["runtime"] = getattr(model, "Runtime", solve_time)

    print(f"Solution found in {solve_time:.2f}s")
    print(f"Objective value: {objective:.3f}")

    # Extract routes (simplified)
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
            # Truck usually has one route, but let's handle list
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
            # Extract arrival times for this drone
            arrival_times = {
                i: vars.arrival_drone[i, d].X
                for i in data.nodes
            }
            # Collect u events (launch/retrieve points) for this drone
            u_events: Dict[int, int] = {}
            for (i, k, d2), var in vars.u.items():
                if d2 == d and var.X > 0.5:
                    u_events[i] = k

            routes = _reconstruct_routes(arcs, arrival_times)
            # Filter out dummy depot-to-depot routes (length 2, e.g. [0, 11])
            routes = [r for r in routes if len(r) > 2]

            if not routes:
                print(f"Drone {d}: idle")
                continue

            distance = sum(data.drone_distance[(i, j)] for i, j in arcs)

            # Split each route into task segments using u events
            depot_start = data.depots[0]
            depot_end = data.depots[1]
            tasks_for_this_drone: list = []
            for route in routes:
                segments: list[list[int]] = []
                current = [route[0]]
                for node in route[1:]:
                    current.append(node)
                    if node in u_events or node in (depot_end,):
                        if len(current) >= 3:
                            segments.append(current)
                        current = [node]
                if len(current) >= 3:
                    segments.append(current)

                for seg in segments:
                    launch_n = seg[0]
                    # When segment ends at depot_end, retrieve is the depot itself.
                    # ALL intermediate nodes (including the one just before depot)
                    # are served customers (MILP droneLoad forces unloading at each).
                    if seg[-1] == depot_end:
                        retrieve_n = depot_end
                    else:
                        retrieve_n = seg[-1]
                    customers = [n for n in seg[1:-1]
                                 if n not in (launch_n, depot_start, depot_end)]
                    if not customers:
                        continue
                    lt = u_events.get(launch_n)
                    rt = u_events.get(retrieve_n)
                    if launch_n == depot_start:
                        lt = None
                    if retrieve_n == depot_end:
                        rt = None
                    tasks_for_this_drone.append({
                        "launch_node": launch_n,
                        "retrieve_node": retrieve_n,
                        "customers": customers,
                        "launch_truck": lt,
                        "retrieve_truck": rt,
                    })

            drone_tasks.append({
                "drone": d,
                "tasks": tasks_for_this_drone,
                "distance": distance
            })
            # Print routes with arrival times for debugging
            routes_with_times = []
            for route in routes:
                start_node = route[0]
                arr_time = arrival_times.get(start_node, 0)
                routes_with_times.append(f"{route} (start@{arr_time:.2f})")
            routes_str = ", ".join(routes_with_times)
            print(
                f"Drone {d}: routes {routes_str}, total distance {distance:.3f}")
            _print_energy_breakdown(d, arcs, vars, data)
            _print_energy_decisions(d, arcs, vars, data)
        else:
            print(f"Drone {d}: idle")

    result["truck_routes"] = truck_routes
    result["drone_tasks"] = drone_tasks

    # ── Record decision variable values ──
    decision_vars = _extract_decision_variables(vars, data)
    result["decision_variables"] = decision_vars
    _print_decision_variables(decision_vars)

    # ── Reconstruct ALNS Solution and verify with ALNS evaluator ──
    try:
        alns_verification = _verify_with_alns_evaluator(
            instance, data, vars, truck_routes, drone_tasks, objective,
            cost_lambda=cost_lambda, cost_rho=cost_rho, cost_normalized=cost_normalized)
        result["alns_verification"] = alns_verification
    except Exception as e:
        import traceback
        traceback.print_exc()
        result["alns_verification"] = {
            "alns_feasible": False,
            "alns_cost": None,
            "mip_objective": objective,
            "verification_error": str(e),
        }

    if result.get("alns_verification", {}).get("alns_feasible"):
        print(f"  ALNS verification: FEASIBLE, cost={alns_verification['alns_cost']:.4f}")
        print(f"    MILP objective vs ALNS cost gap: "
              f"{((objective - alns_verification['alns_cost']) / alns_verification['alns_cost'] * 100):.2f}%")
    else:
        print(f"  ALNS verification: INFEASIBLE")
        if alns_verification.get("robustness_violations"):
            for v in alns_verification["robustness_violations"]:
                print(f"    D{v['drone_id']}: energy={v['worst_case_energy']:.4f}, "
                      f"budget={v['capacity']:.4f}, margin={v['margin']:.4f}")

    # Write per-instance output immediately
    instance_name = Path(config["instance"]).stem
    per_instance_path = OUTPUT_PATH.parent / f"{instance_name}.json"
    with per_instance_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=str)
    print(f"  [saved {per_instance_path.name}]")

    return result


def run_experiments() -> None:
    """Run all configured MILP experiments."""
    """Run all configured MILP experiments."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_results: List[Dict[str, Any]] = []

    for config in RUNS:
        try:
            result = run_single_mip(config)
            all_results.append(result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"ERROR running {config['instance']}: {e}")
            all_results.append({
                "instance": config["instance"],
                "config": config,
                "error": str(e)
            })

    # Save results
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2)
    print(f"\nResults written to {OUTPUT_PATH}")


def _extract_decision_variables(vars: Any, data: Any) -> Dict[str, Any]:
    """Extract all decision variable values for post-hoc analysis."""
    """Extract all decision variable values for post-hoc analysis."""
    dv = {}

    # x_truck: truck arc activation
    x_active = {}
    for (i, j, k), var in vars.x_truck.items():
        if var.X > 0.5:
            x_active[f"({i},{j},{k})"] = round(var.X, 6)
    dv["x_truck_active"] = x_active

    # y_drone: drone arc activation
    y_active = {}
    for (i, j, d), var in vars.y_drone.items():
        if var.X > 0.5:
            y_active[f"({i},{j},{d})"] = round(var.X, 6)
    dv["y_drone_active"] = y_active

    # z_coupling: drone-truck coupling
    z_active = {}
    for (i, j, k, d), var in vars.z_coupling.items():
        if var.X > 0.5:
            z_active[f"({i},{j},{k},{d})"] = round(var.X, 6)
    dv["z_coupling_active"] = z_active

    # u: drone launch/retrieval synchronization
    u_active = {}
    for (i, k, d), var in vars.u.items():
        if var.X > 0.5:
            u_active[f"({i},{k},{d})"] = round(var.X, 6)
    dv["u_sync_active"] = u_active

    # Arrival times (truck and drone)
    truck_arrivals = {}
    for (i, k), var in vars.arrival_truck.items():
        if var.X > 1e-6:
            truck_arrivals[f"({i},{k})"] = round(var.X, 6)
    dv["arrival_truck_nonzero"] = truck_arrivals

    drone_arrivals = {}
    for (i, d), var in vars.arrival_drone.items():
        if var.X > 1e-6:
            drone_arrivals[f"({i},{d})"] = round(var.X, 6)
    dv["arrival_drone_nonzero"] = drone_arrivals

    # Tardiness (actual service delay values)
    truck_tardiness = {}
    for (i, k), var in vars.tardiness_truck.items():
        if var.X > 1e-6:
            truck_tardiness[f"({i},{k})"] = round(var.X, 6)
    dv["tardiness_truck_nonzero"] = truck_tardiness

    drone_tardiness = {}
    for (i, d), var in vars.tardiness_drone.items():
        if var.X > 1e-6:
            drone_tardiness[f"({i},{d})"] = round(var.X, 6)
    dv["tardiness_drone_nonzero"] = drone_tardiness

    # Delay cost (PWL mapped values)
    delay_cost_truck = {}
    for (i, k), var in vars.tardiness_cost_truck.items():
        if var.X > 1e-6:
            delay_cost_truck[f"({i},{k})"] = round(var.X, 6)
    dv["delay_cost_truck_nonzero"] = delay_cost_truck

    delay_cost_drone = {}
    for (i, d), var in vars.tardiness_cost_drone.items():
        if var.X > 1e-6:
            delay_cost_drone[f"({i},{d})"] = round(var.X, 6)
    dv["delay_cost_drone_nonzero"] = delay_cost_drone

    # Load variables
    truck_load = {}
    for (i, k), var in vars.load_truck.items():
        if var.X > 1e-6:
            truck_load[f"({i},{k})"] = round(var.X, 6)
    dv["load_truck_nonzero"] = truck_load

    drone_load_plus = {}
    for (i, d), var in vars.load_drone_plus.items():
        if var.X > 1e-6:
            drone_load_plus[f"({i},{d})"] = round(var.X, 6)
    dv["load_drone_plus_nonzero"] = drone_load_plus

    drone_load_minus = {}
    for (i, d), var in vars.load_drone_minus.items():
        if var.X > 1e-6:
            drone_load_minus[f"({i},{d})"] = round(var.X, 6)
    dv["load_drone_minus_nonzero"] = drone_load_minus

    # Energy state (final gamma level = worst case)
    gamma_max = data.gamma_range[-1] if data.gamma_range else 0
    energy_worst = {}
    for (i, d, gamma), var in vars.energy_state_gamma.items():
        if gamma != gamma_max:
            continue
        val = var.X
        if val > 1e-6:
            energy_worst[f"({i},{d})"] = round(val, 6)
    dv["energy_worst_case_nonzero"] = energy_worst

    # Departure times (truck)
    truck_departures = {}
    if hasattr(vars, 'departure_truck'):
        for (i, k), var in vars.departure_truck.items():
            if var.X > 1e-6:
                truck_departures[f"({i},{k})"] = round(var.X, 6)
    dv["departure_truck_nonzero"] = truck_departures

    return dv


def _print_decision_variables(dv: Dict[str, Any]) -> None:
    """Print summary of key decision variables."""
    """Print summary of key decision variables."""
    print("\n  === Decision Variable Summary ===")
    print(f"  x_truck arcs: {len(dv.get('x_truck_active', {}))}")
    print(f"  y_drone arcs: {len(dv.get('y_drone_active', {}))}")
    print(f"  z_coupling arcs: {len(dv.get('z_coupling_active', {}))}")
    print(f"  u_sync events: {len(dv.get('u_sync_active', {}))}")

    if dv.get('u_sync_active'):
        print("  u (launch/retrieve sync):")
        for key, val in sorted(dv['u_sync_active'].items()):
            print(f"    u{key} = {val}")

    if dv.get('z_coupling_active'):
        print("  z (drone-truck coupling):")
        for key, val in sorted(dv['z_coupling_active'].items()):
            print(f"    z{key} = {val}")

    tardiness = dv.get('delay_cost_truck_nonzero', {})
    tardiness.update(dv.get('delay_cost_drone_nonzero', {}))
    if tardiness:
        total_delay = sum(tardiness.values())
        print(f"  Total delay cost: {total_delay:.4f}")
        for key, val in sorted(tardiness.items()):
            print(f"    delay_cost{key} = {val}")


def _verify_with_alns_evaluator(
    instance: Any,
    data: Any,
    vars: Any,
    truck_routes: List[Dict],
    drone_tasks: List[Dict],
    mip_objective: float,
    *,
    cost_lambda: float = 30.0,
    cost_rho: float = 0.20833,
    cost_normalized: bool = True,
) -> Dict[str, Any]:
    """Reconstruct ALNS Solution from MILP decision variables and verify feasibility."""
    """Reconstruct ALNS Solution from MILP decision variables and verify feasibility."""
    from alns_vrpfd.evaluation.evaluator import Evaluator
    from alns_vrpfd.model import Solution, TruckRoute, DroneTask
    from alns_vrpfd.core.operators.base import _build_payloads

    em = instance.customer_manager
    dm = {c.customer_id: c.demand for c in em.customers()}

    result: Dict[str, Any] = {
        "alns_feasible": False,
        "alns_cost": None,
        "mip_objective": mip_objective,
        "reconstruction_method": None,
        "robustness_violations": [],
    }

    # ── Method 1: Reconstruct from y_drone arcs + u events (no v_served) ──
    try:
        sol_v = _reconstruct_from_y_arcs(vars, data, instance, dm)
        if sol_v is not None:
            evaluator = Evaluator(instance, cost_lambda=cost_lambda, cost_rho=cost_rho, cost_normalized=cost_normalized)
            ev = evaluator.evaluate_solution(sol_v)
            details = evaluator.evaluate_with_details(sol_v)
            result["reconstruction_method"] = "y_arcs"
            result["alns_feasible"] = ev.feasible
            if ev.feasible:
                result["alns_cost"] = round(ev.total_cost, 4)
                result["truck_cost"] = round(ev.truck_distance_cost, 4)
                result["drone_cost"] = round(ev.drone_distance_cost, 4)
                result["delay_cost"] = round(ev.delay_penalty, 4)
                result["robustness_feasible"] = details.robustness.feasible
            else:
                result["alns_cost"] = None
                result["violations"] = {
                    "task_violation": evaluator._has_drone_task_violations(sol_v),
                    "coverage": evaluator._has_customer_coverage_violation(sol_v),
                    "robustness": not details.robustness.feasible,
                }
                result["truck_routes"] = [r.nodes for r in sol_v.truck_routes]
                result["drone_tasks"] = [
                    f"D{t.drone_id}: T{t.launch_truck}@{t.launch_node} -> "
                    f"{t.customers()} -> T{t.land_truck}@{t.retrieve_node}"
                    for t in sol_v.drone_tasks
                ]
            for b in details.robustness.task_breakdown:
                if not b.feasible:
                    result["robustness_violations"].append({
                        "drone_id": b.drone_id,
                        "task_id": b.task_id,
                        "nominal_energy": round(b.nominal_energy, 4),
                        "worst_case_energy": round(b.worst_case_energy, 4),
                        "capacity": b.capacity,
                        "margin": round(b.margin, 4),
                    })
    except Exception as e:
        result["y_arcs_reconstruction_error"] = str(e)

    # ── Method 2: Fallback - reconstruct from JSON-like route structure ──
    if not result["alns_feasible"]:
        try:
            sol_json = _reconstruct_from_json_routes(truck_routes, drone_tasks, instance, dm)
            if sol_json is not None:
                evaluator = Evaluator(instance, cost_lambda=cost_lambda, cost_rho=cost_rho, cost_normalized=cost_normalized)
                ev = evaluator.evaluate_solution(sol_json)
                details = evaluator.evaluate_with_details(sol_json)
                result["reconstruction_method"] = "json_routes"
                result["alns_feasible_json"] = ev.feasible
                if ev.feasible:
                    result["alns_cost_json"] = round(ev.total_cost, 4)
                    result["robustness_feasible_json"] = details.robustness.feasible
                else:
                    result["violations_json"] = {
                        "task_violation": evaluator._has_drone_task_violations(sol_json),
                        "coverage": evaluator._has_customer_coverage_violation(sol_json),
                        "robustness": not details.robustness.feasible,
                    }
                for b in details.robustness.task_breakdown:
                    if not b.feasible:
                        result["robustness_violations"].append({
                            "drone_id": b.drone_id,
                            "task_id": b.task_id,
                            "nominal_energy": round(b.nominal_energy, 4),
                            "worst_case_energy": round(b.worst_case_energy, 4),
                            "capacity": b.capacity,
                            "margin": round(b.margin, 4),
                            "reconstruction": "json_routes",
                        })
        except Exception as e:
            result["json_reconstruction_error"] = str(e)

    return result


def _reconstruct_from_y_arcs(
    vars: Any, data: Any, instance: Any, dm: Dict[int, float]
) -> Any:
    """Reconstruct ALNS Solution from MILP y_drone arcs + u events.

    Determines drone-served customers from drone sortie structure
    (y_drone arcs + u launch events) without v_served variables.
    """
    from alns_vrpfd.model import Solution, TruckRoute, DroneTask
    from alns_vrpfd.core.operators.base import _build_payloads

    depot_start = data.depots[0]
    depot_end = data.depots[1]

    # ── Extract truck routes from x_truck ──
    truck_routes_list: Dict[int, list] = {}
    for k in data.trucks:
        arcs = [(i, j) for (i, j) in data.arcs if vars.x_truck[i, j, k].X > 0.5]
        if arcs:
            route = _reconstruct_routes(arcs)
            if route:
                truck_routes_list[k] = route[0] if isinstance(route, list) and route else route

    # Build truck node position maps
    truck_node_positions = {}
    for k, route in truck_routes_list.items():
        truck_node_positions[k] = {node: pos for pos, node in enumerate(route)}

    node_to_truck = {}
    for k, route in truck_routes_list.items():
        for pos, node in enumerate(route):
            node_to_truck[node] = (k, pos)

    # ── Extract drone sorties from y_drone + u ──
    drone_served_customers = set()
    drone_sorties = {}
    next_task_id = 0

    for d in data.drones:
        y_arcs = [(i, j) for (i, j) in data.arcs if vars.y_drone[i, j, d].X > 0.5]
        if not y_arcs:
            continue

        u_events = [(i, k) for (i, k, d2) in vars.u if d2 == d and vars.u[i, k, d].X > 0.5]
        u_by_node = {i: k for i, k in u_events}

        successors = {i: j for i, j in y_arcs}
        predecessors = {j: i for i, j in y_arcs}

        start_nodes = [i for i, _ in y_arcs if i not in predecessors]
        if not start_nodes:
            start_nodes = [i for i, _ in u_events]

        for start in start_nodes:
            path = [start]
            segment_start = start
            current = start
            visited = set()
            closed_segment = False
            while current in successors:
                if current in visited:
                    break
                visited.add(current)
                nxt = successors[current]
                path.append(nxt)
                current = nxt

                if current != segment_start and (current in u_by_node or current == depot_end):
                    launch_node = path[0]
                    retrieve_node = path[-1]

                    customers = [n for n in path[1:-1] if n in data.customers]
                    if customers:
                        lt = u_by_node.get(launch_node)
                        rt = u_by_node.get(retrieve_node)

                        if lt is None:
                            lt_info = node_to_truck.get(launch_node)
                            lt = lt_info[0] if lt_info else None
                        if rt is None:
                            rt_info = node_to_truck.get(retrieve_node)
                            rt = rt_info[0] if rt_info else None

                        if launch_node == depot_start:
                            lt = None
                        if retrieve_node == depot_end:
                            rt = None

                        payloads = _build_payloads(customers, dm)
                        task = DroneTask(
                            task_id=next_task_id,
                            drone_id=d,
                            launch_truck=lt,
                            launch_node=launch_node,
                            customers=customers,
                            land_truck=rt,
                            retrieve_node=retrieve_node,
                            payloads=payloads,
                        )
                        if d not in drone_sorties:
                            drone_sorties[d] = []
                        drone_sorties[d].append(task)
                        next_task_id += 1
                        drone_served_customers.update(customers)
                        closed_segment = True

                    if current == depot_end:
                        break
                    segment_start = current
                    path = [current]
                    closed_segment = False

            if not closed_segment and len(path) > 1:
                launch_node = path[0]
                retrieve_node = path[-1]
                customers = [n for n in path[1:-1] if n in data.customers]
                if customers:
                    lt = u_by_node.get(launch_node)
                    rt = u_by_node.get(retrieve_node)

                    if lt is None:
                        lt_info = node_to_truck.get(launch_node)
                        lt = lt_info[0] if lt_info else None
                    if rt is None:
                        rt_info = node_to_truck.get(retrieve_node)
                        rt = rt_info[0] if rt_info else None

                    if launch_node == depot_start:
                        lt = None
                    if retrieve_node == depot_end:
                        rt = None

                    payloads = _build_payloads(customers, dm)
                    task = DroneTask(
                        task_id=next_task_id,
                        drone_id=d,
                        launch_truck=lt,
                        launch_node=launch_node,
                        customers=customers,
                        land_truck=rt,
                        retrieve_node=retrieve_node,
                        payloads=payloads,
                    )
                    if d not in drone_sorties:
                        drone_sorties[d] = []
                    drone_sorties[d].append(task)
                    next_task_id += 1
                    drone_served_customers.update(customers)

    # ── Build Solution ──
    sol = Solution()
    for k in sorted(truck_routes_list.keys()):
        route_nodes = truck_routes_list[k]
        filtered = [n for n in route_nodes if n not in drone_served_customers or n == depot_start or n == depot_end]
        if filtered and filtered[0] != depot_start:
            filtered = [depot_start] + filtered
        if filtered and filtered[-1] != depot_end:
            filtered = filtered + [depot_end]
        if filtered:
            sol.add_truck_route(TruckRoute(
                route_id=k,
                nodes=filtered,
                capacity=instance.vehicle_specs['truck'].capacity,
            ))

    for d in sorted(drone_sorties.keys()):
        for task in drone_sorties[d]:
            sol.add_drone_task(task)

    return sol


def _reconstruct_from_json_routes(
    truck_routes: List[Dict],
    drone_tasks: List[Dict],
    instance: Any,
    dm: Dict[int, float],
) -> Any:
    """Reconstruct ALNS Solution from JSON-like route structure (fallback method)."""
    """Reconstruct ALNS Solution from JSON-like route structure (fallback method)."""
    from alns_vrpfd.model import Solution, TruckRoute, DroneTask
    from alns_vrpfd.core.operators.base import _build_payloads

    depot_start = instance.customer_manager.depot_start
    depot_end = instance.customer_manager.depot_end

    sol = Solution()

    for tr in truck_routes:
        route = tr["route"]
        sol.add_truck_route(TruckRoute(
            route_id=tr["truck"],
            nodes=route,
            capacity=instance.vehicle_specs['truck'].capacity,
        ))

    node_to_truck = {}
    for tr in truck_routes:
        for pos, n in enumerate(tr["route"]):
            node_to_truck[n] = (tr["truck"], pos)

    for dt in drone_tasks:
        tasks = dt.get("tasks")
        if tasks is not None:
            # New format: pre-segmented tasks with explicit launch/retrieve data
            for t in tasks:
                customers = t["customers"]
                if not customers:
                    continue
                lt = t.get("launch_truck")
                rt = t.get("retrieve_truck")
                launch_node = t["launch_node"]
                retrieve_node = t["retrieve_node"]
                if launch_node == depot_start:
                    lt = None
                if retrieve_node == depot_end:
                    rt = None
                payloads = _build_payloads(customers, dm)
                sol.add_drone_task(DroneTask(
                    task_id=len(sol.drone_tasks),
                    drone_id=dt["drone"],
                    launch_truck=lt,
                    launch_node=launch_node,
                    customers=customers,
                    land_truck=rt,
                    retrieve_node=retrieve_node,
                    payloads=payloads,
                ))
        else:
            # Old format: raw routes (fallback)
            for r in dt.get("routes", []):
                if len(r) < 2:
                    continue
                launch_node = r[0]
                retrieve_node = r[-1]
                customers = [n for n in r[1:-1] if n != depot_start and n != depot_end]
                if not customers:
                    continue
                lt_info = node_to_truck.get(launch_node)
                rt_info = node_to_truck.get(retrieve_node)
                lt = lt_info[0] if lt_info else None
                rt = rt_info[0] if rt_info else None
                if launch_node == depot_start:
                    lt = None
                if retrieve_node == depot_end:
                    rt = None
                payloads = _build_payloads(customers, dm)
                sol.add_drone_task(DroneTask(
                    task_id=len(sol.drone_tasks),
                    drone_id=dt["drone"],
                    launch_truck=lt,
                    launch_node=launch_node,
                    customers=customers,
                    land_truck=rt,
                    retrieve_node=retrieve_node,
                    payloads=payloads,
                ))

    return sol


def _print_energy_breakdown(drone_id: int, arcs: list[tuple[int, int]], vars: Any, data: Any) -> None:
    """Emit energy decision variable values for each selected arc."""
    """Emit energy decision variable values for each selected arc."""

    energy_active = getattr(vars, "energy_active", None)
    omega_active = getattr(vars, "omega_active", None)
    power_approx = getattr(vars, "power_approx", None)
    load_plus = getattr(vars, "load_drone_plus", None)
    load_minus = getattr(vars, "load_drone_minus", None)

    if omega_active is not None:
        print("    Load and energy breakdown per arc (load_minus_at_dest, load_plus_at_origin, omega_active, power_approx):")
        for (i, j) in arcs:
            omega_val = omega_active[i, j, drone_id].X
            power_val = power_approx[i, j,
                                     drone_id].X if power_approx is not None else None
            load_plus_val = load_plus[i,
                                      drone_id].X if load_plus is not None else None
            load_minus_val = load_minus[j,
                                        drone_id].X if load_minus is not None else None
            power_str = f"{power_val:.4f}" if power_val is not None else "NA"
            load_plus_str = f"{load_plus_val:.4f}" if load_plus_val is not None else "NA"
            load_minus_str = f"{load_minus_val:.4f}" if load_minus_val is not None else "NA"
            print(
                f"      arc {(i, j)}: load_minus_dest={load_minus_str}, "
                f"load_plus_origin={load_plus_str}, "
                f"omega_active={omega_val:.4f}, "
                f"power_approx={power_str}"
            )
    elif energy_active is not None:
        print("    Energy breakdown per arc (y, omega_active, power_approx, energy_active, load_plus_at_origin):")
        for (i, j) in arcs:
            y_val = vars.y_drone[i, j, drone_id].X
            omega_val = omega_active[i, j,
                                     drone_id].X if omega_active is not None else None
            power_val = power_approx[i, j,
                                     drone_id].X if power_approx is not None else None
            energy_val = energy_active[i, j, drone_id].X
            load_val = load_plus[i,
                                 drone_id].X if load_plus is not None else None
            omega_str = f"{omega_val:.4f}" if omega_val is not None else "NA"
            power_str = f"{power_val:.4f}" if power_val is not None else "NA"
            load_str = f"{load_val:.4f}" if load_val is not None else "NA"
            print(
                f"      arc {(i, j)}: y={y_val:.3f}, "
                f"omega_active={omega_str}, "
                f"power_approx={power_str}, "
                f"energy_active={energy_val:.4f}, "
                f"load_plus_origin={load_str}"
            )
    else:
        print("    (energy variables unavailable; skip breakdown)")


def _print_energy_decisions(drone_id: int, arcs: list[tuple[int, int]], vars: Any, data: Any) -> None:
    """，。"""
    """打印关键能耗决策变量，帮助定位无人机在各弧上的能量使用情况。"""

    energy_state_gamma = getattr(vars, "energy_state_gamma", None)
    energy_active = getattr(vars, "energy_active", None)
    gamma_range = getattr(data, "gamma_range", None)

    if energy_state_gamma is None or gamma_range is None:
        print("    (energy_state_gamma 缺失，无法输出能耗决策)")
        return

    print("    Energy decisions per arc:")
    header = "      arc (i,j): y  e_nom_delta  e_worst_delta  extra_gamma  energy_active"
    print(header)

    gamma_min = gamma_range[0]
    gamma_max = gamma_range[-1]

    for (i, j) in arcs:
        y_val = vars.y_drone[i, j, drone_id].X
        if y_val <= 1e-6:
            continue

        e_nom_i = energy_state_gamma[i, drone_id, gamma_min].X
        e_nom_j = energy_state_gamma[j, drone_id, gamma_min].X
        e_worst_i = energy_state_gamma[i, drone_id, gamma_max].X
        e_worst_j = energy_state_gamma[j, drone_id, gamma_max].X

        delta_nom = e_nom_j - e_nom_i
        delta_worst = e_worst_j - e_worst_i
        extra = delta_worst - delta_nom
        energy_arc_val = (
            energy_active[i, j,
                          drone_id].X if energy_active is not None else None
        )

        energy_arc_str = f"{energy_arc_val:.4f}" if energy_arc_val is not None else "NA"
        print(
            f"      arc {(i, j)}: y={y_val:.3f}  "
            f"e_nom_delta={delta_nom:.4f}  e_worst_delta={delta_worst:.4f}  "
            f"extra_gamma={extra:.4f}  energy_active={energy_arc_str}"
        )


def _report_solution(artifacts: MIPArtifacts) -> None:
    data = artifacts.data
    vars = artifacts.variables
    model = artifacts.model

    print(f"Solve status: {model.Status}")
    if model.SolCount == 0:
        raise SolverError("No feasible solution found by Gurobi.")

    objective = model.ObjVal
    print(f"Objective value: {objective:.3f}")

    # Truck arcs
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

    # Drone arcs
    for d in data.drones:
        arcs = [
            (i, j)
            for (i, j) in data.arcs
            if vars.y_drone[i, j, d].X > 0.5
        ]
        if not arcs:
            print(f"Drone {d}: idle")
            continue
        # Extract arrival times for this drone to sort routes chronologically
        arrival_times = {
            i: vars.arrival_drone[i, d].X
            for i in data.nodes
        }
        routes = _reconstruct_routes(arcs, arrival_times)
        distance = sum(data.drone_distance[(i, j)] for i, j in arcs)
        routes_str = ", ".join(str(r) for r in routes)
        print(f"Drone {d}: routes {routes_str}, distance {distance:.3f}")


def _apply_warm_start(
    artifacts: MIPArtifacts,
    warm_start_path: Path,
) -> None:
    import json

    with warm_start_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    variables = artifacts.variables
    model = artifacts.model

    # Initialise all binary variables to 0
    for var in variables.x_truck.values():
        var.Start = 0.0
    for var in variables.y_drone.values():
        var.Start = 0.0
    for var in variables.z_coupling.values():
        var.Start = 0.0
    for var in variables.u_launch.values():
        var.Start = 0.0

    trucks = data.get("trucks", {})
    for truck_id_str, route in trucks.items():
        if not route:
            continue
        truck_id = int(truck_id_str)
        nodes = [int(node) for node in route]
        for i, j in zip(nodes, nodes[1:]):
            key = (i, j, truck_id)
            if key in variables.x_truck:
                variables.x_truck[key].Start = 1.0

    drones = data.get("drones", {})
    for drone_id_str, route in drones.items():
        if not route:
            continue
        drone_id = int(drone_id_str)
        nodes = [int(node) for node in route]
        for i, j in zip(nodes, nodes[1:]):
            key = (i, j, drone_id)
            if key in variables.y_drone:
                variables.y_drone[key].Start = 1.0

    model.update()


if __name__ == "__main__":
    try:
        run_experiments()
    except SolverError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
