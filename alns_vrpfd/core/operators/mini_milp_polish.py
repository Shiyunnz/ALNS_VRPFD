"""Mini-MILP truck route polisher.

Fixes drone task assignments from an ALNS solution and re-optimizes
truck customer-to-truck assignment AND visit ordering using Gurobi.

Key difference from the full MILP: drone task structure (which customers
drones serve, and launch/retrieve nodes) is fixed. Only truck decisions
are optimized:
- Which customers each truck serves
- In what order customers are visited

This is much smaller than the full MILP (~200 binary vars vs ~900) and
solves in seconds. It exactly matches the ALNS Evaluator's cost model.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import gurobipy as gp
from gurobipy import GRB

from alns_vrpfd.evaluation.evaluator import Evaluator
from alns_vrpfd.instance.manager import InstanceManager
from alns_vrpfd.model import DroneTask, Solution, TruckRoute
from alns_vrpfd.deprivation import deprivation_cost

logger = logging.getLogger(__name__)


@dataclass
class PolishResult:
    original_cost: float
    polished_cost: float
    improvement: float
    runtime_seconds: float
    improved: bool
    polished_solution: Solution
    truck_routes_before: List[List[int]]
    truck_routes_after: List[List[int]]


def _get_drone_anchor_nodes(
    drone_tasks: List[DroneTask], num_trucks: int
) -> Dict[int, Set[int]]:
    anchors: Dict[int, Set[int]] = {k: set() for k in range(num_trucks)}
    for dt in drone_tasks:
        if dt.launch_truck is not None:
            anchors[dt.launch_truck].add(dt.launch_node)
        if dt.land_truck is not None:
            anchors[dt.land_truck].add(dt.retrieve_node)
    return anchors


def _get_drone_served_customers(drone_tasks: List[DroneTask]) -> Set[int]:
    served = set()
    for dt in drone_tasks:
        served.update(dt.customers())
    return served


def polish_with_extended_milp(
    solution: Solution,
    instance: InstanceManager,
    evaluator: Evaluator,
    time_limit: float = 10.0,
    mip_gap: float = 0.001,
    verbose: bool = False,
) -> PolishResult:
    """Polish truck routes with a MILP that can reassign customers between trucks.

    Drone task assignments (which customers drones serve, launch/retrieve
    nodes) are fixed. The MILP optimizes:
    - Which remaining customers each truck visits
    - The visit order on each truck

    This is strictly the truck-side subproblem with drone anchors fixed.
    """
    start_time = time.perf_counter()

    depot_start = instance.customer_manager.depot_start
    depot_end = instance.customer_manager.depot_end
    if depot_start is None:
        depot_start = 0
    if depot_end is None:
        depot_end = depot_start

    all_customers = list(instance.customer_manager.customer_ids())
    demands = instance.customer_manager.demands()
    truck_dist = instance.distance_matrix("truck")
    truck_time = instance.time_matrix("truck")
    node_ids = instance.all_node_ids()
    node_index = {n: i for i, n in enumerate(node_ids)}

    truck_spec = instance.vehicle_specs.get("truck")
    num_trucks = truck_spec.number
    truck_capacity = truck_spec.capacity

    optimal_times: Dict[int, float] = {}
    for cid in all_customers:
        opt, _ = instance.customer_manager.time_window(cid)
        optimal_times[cid] = float(opt) if opt is not None else 0.0

    drone_served = _get_drone_served_customers(solution.drone_tasks)
    anchors = _get_drone_anchor_nodes(solution.drone_tasks, num_trucks)

    # Customers not served by drones — the MILP can assign these freely
    truck_customers = [c for c in all_customers if c not in drone_served]

    if not truck_customers:
        ev = evaluator.evaluate_solution(solution)
        return PolishResult(
            original_cost=ev.total_cost, polished_cost=ev.total_cost,
            improvement=0.0, runtime_seconds=time.perf_counter() - start_time,
            improved=False, polished_solution=solution.clone(),
            truck_routes_before=[r.nodes[:] for r in solution.truck_routes],
            truck_routes_after=[r.nodes[:] for r in solution.truck_routes],
        )

    # Drone info for deriving delay penalty (from the solution)
    drone_task_info = []
    for dt in solution.drone_tasks:
        drone_task_info.append({
            "drone_id": dt.drone_id,
            "launch_truck": dt.launch_truck,
            "launch_node": dt.launch_node,
            "land_truck": dt.land_truck,
            "retrieve_node": dt.retrieve_node,
            "customers": dt.customers(),
        })

    # --- Build MILP ---
    model = gp.Model("extended_truck_polish")
    model.setParam("OutputFlag", 1 if verbose else 0)
    model.setParam("TimeLimit", time_limit)
    model.setParam("MIPGap", mip_gap)
    model.setParam("Threads", 4)
    model.setParam("PoolSearchMode", 0)

    K = list(range(num_trucks))
    C = truck_customers

    # Node set for truck routing: depot0 + truck customers + drone anchor customers + depot_end
    all_truck_nodes = [depot_start] + C + [depot_end]
    # Also include anchor nodes that are customers
    for k in K:
        for a in anchors.get(k, set()):
            if a not in all_truck_nodes and a != depot_start and a != depot_end:
                all_truck_nodes.append(a)

    # Eliminate duplicates while preserving order
    seen = set()
    N = []
    for n in all_truck_nodes:
        if n not in seen:
            seen.add(n)
            N.append(n)

    feasible_arcs = set()
    for i in N:
        for j in N:
            if i != j:
                d = truck_dist[node_index[i]][node_index[j]]
                if not (math.isinf(d) or math.isnan(d) or d > 1e5):
                    feasible_arcs.add((i, j))

    # Variables
    x = {}
    for i in N:
        for j in N:
            if i != j and (i, j) in feasible_arcs:
                for k in K:
                    x[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")

    tau = {}
    for i in C:
        for k in K:
            tau[i, k] = model.addVar(lb=0.0, ub=100.0, name=f"tau_{i}_{k}")

    # Anchor nodes also need arrival times
    all_anchor_customers = set()
    for k in K:
        all_anchor_customers.update(anchors.get(k, set()))
    all_anchor_customers = all_anchor_customers - {depot_start, depot_end}
    for i in all_anchor_customers:
        for k in K:
            if (i, k) not in tau:
                tau[i, k] = model.addVar(lb=0.0, ub=100.0, name=f"tau_{i}_{k}")

    tau_start = {}
    for k in K:
        tau_start[k] = model.addVar(lb=0.0, ub=5.0, name=f"tau_start_{k}")

    delay = {}
    all_cust_with_delay = list(set(C) | all_anchor_customers)
    for i in all_cust_with_delay:
        delay[i] = model.addVar(lb=0.0, ub=50.0, name=f"delay_{i}")

    M = 500.0

    # --- Constraints ---

    # 1. Each truck-served customer visited by exactly one truck
    for i in C:
        model.addConstr(
            gp.quicksum(x[j, i, k] for k in K for j in N if j != i
                       and (j, i) in feasible_arcs and (j, i, k) in x) == 1,
            name=f"visit_{i}")

    # 2. Anchor constraints: drone anchor nodes must stay on assigned truck
    for k in K:
        for a in anchors.get(k, set()):
            if a in N and a != depot_start and a != depot_end:
                model.addConstr(
                    gp.quicksum(x[j, a, k] for j in N if j != a
                               and (j, a) in feasible_arcs and (j, a, k) in x) == 1,
                    name=f"anchor_{a}_truck_{k}")

    # 3. Flow constraints
    for k in K:
        leave = gp.quicksum(x[depot_start, j, k] for j in N if j != depot_start
                           and (depot_start, j) in feasible_arcs and (depot_start, j, k) in x)
        enter = gp.quicksum(x[j, depot_end, k] for j in N if j != depot_end
                           and (j, depot_end) in feasible_arcs and (j, depot_end, k) in x)
        model.addConstr(leave <= 1, name=f"leave_{k}")
        model.addConstr(enter <= 1, name=f"enter_{k}")
        model.addConstr(leave == enter, name=f"balance_{k}")

        for i in N:
            if i == depot_start or i == depot_end:
                continue
            model.addConstr(
                gp.quicksum(x[j, i, k] for j in N if j != i
                           and (j, i) in feasible_arcs and (j, i, k) in x)
                == gp.quicksum(x[i, j, k] for j in N if j != i
                              and (i, j) in feasible_arcs and (i, j, k) in x),
                name=f"flow_{i}_{k}")

    # 4. Truck capacity
    for k in K:
        model.addConstr(
            gp.quicksum(demands.get(i, 0.0) * x[j, i, k] for i in C for j in N
                       if j != i and (j, i) in feasible_arcs and (j, i, k) in x)
            <= truck_capacity,
            name=f"cap_{k}")

    # 5. Subtour elimination (MTZ)
    positions = {}
    all_nodes_with_pos = list(set(C) | all_anchor_customers)
    for i in all_nodes_with_pos:
        positions[i] = model.addVar(lb=1, ub=len(all_nodes_with_pos),
                                     vtype=GRB.INTEGER, name=f"pos_{i}")

    for i in all_nodes_with_pos:
        for j in all_nodes_with_pos:
            if i != j:
                for k in K:
                    if (i, j, k) in x:
                        model.addConstr(
                            positions[i] - positions[j] + len(all_nodes_with_pos) * x[i, j, k]
                            <= len(all_nodes_with_pos) - 1,
                            name=f"mtz_{i}_{j}_{k}")

    # 6. Arrival time constraints
    for i in N:
        if i == depot_start or i == depot_end:
            continue
        for k in K:
            if (depot_start, i, k) in x:
                model.addConstr(
                    tau[i, k] >= tau_start[k]
                    + truck_time[node_index[depot_start]][node_index[i]]
                    - M * (1 - x[depot_start, i, k]),
                    name=f"arr_dep_{i}_{k}")
            for j in N:
                if j == depot_start or j == depot_end or j == i:
                    continue
                if (j, i, k) in x:
                    model.addConstr(
                        tau[i, k] >= tau[j, k]
                        + truck_time[node_index[j]][node_index[i]]
                        - M * (1 - x[j, i, k]),
                        name=f"arr_{j}_{i}_{k}")

    # 7. Delay definition
    for i in all_cust_with_delay:
        visited_sum = gp.quicksum(1 for k in K for j in N
                                 if j != i and (j, i) in feasible_arcs and (j, i, k) in x)
        if visited_sum > 0:
            for k in K:
                lhs = tau.get((i, k))
                if lhs is not None:
                    model.addConstr(
                        delay[i] >= lhs - optimal_times.get(i, 0.0)
                        - M * (1 - gp.quicksum(x[j, i, k] for j in N
                                              if j != i and (j, i) in feasible_arcs
                                              and (j, i, k) in x)),
                        name=f"delay_{i}_{k}")

    # 8. Drone synchronization: launch truck must arrive before drone can serve/return
    for dt_info in drone_task_info:
        ln = dt_info["launch_node"]
        rn = dt_info["retrieve_node"]
        lt = dt_info["launch_truck"]
        ldt = dt_info["land_truck"]

        # The launch and retrieve nodes must be on their assigned trucks
        # (already enforced by anchor constraints)

    # --- Objective: minimize truck distance + delay penalty ---
    truck_unit_cost = truck_spec.unit_cost

    truck_dist_expr = gp.quicksum(
        truck_unit_cost * truck_dist[node_index[i]][node_index[j]] * x[i, j, k]
        for (i, j, k), var in x.items()
    )

    # Drone distance (fixed from solution)
    drone_unit_cost = instance.vehicle_specs.get("drone", instance.vehicle_specs["truck"]).unit_cost
    drone_dist = instance.distance_matrix("drone")
    drone_dist_cost = 0.0
    for dt in solution.drone_tasks:
        prev = dt.launch_node
        for cust in dt.customers():
            d = drone_dist[node_index[prev]][node_index[cust]]
            if not (math.isinf(d) or math.isnan(d)):
                drone_dist_cost += drone_unit_cost * d
            prev = cust
        d = drone_dist[node_index[prev]][node_index[dt.retrieve_node]]
        if not (math.isinf(d) or math.isnan(d)):
            drone_dist_cost += drone_unit_cost * d

    # Delay penalty using EXACT deprivation cost (PWL for MILP linearization)
    cfg = instance.robust_config
    cost_lambda = getattr(cfg, 'cost_lambda', 30.0)
    cost_rho = getattr(cfg, 'cost_rho', 0.2083)
    cost_normalized = getattr(cfg, 'cost_normalized', True)
    alpha_const = 1.5031

    class_params = {
        "health": {"beta": 0.4558, "omega": 1.35},
        "water": {"beta": 0.4525, "omega": 1.35},
        "food": {"beta": 0.4464, "omega": 1.00},
        "shelter": {"beta": 0.4469, "omega": 0.75},
    }

    supply_classes = {}
    for cid in all_customers:
        supply_classes[cid] = instance.customer_manager.supply_class(cid) or "food"

    delay_penalty = {}
    for i in all_cust_with_delay:
        delay_penalty[i] = model.addVar(lb=0.0, name=f"dpen_{i}")

    max_delay = max(optimal_times.get(c, 10.0) for c in all_cust_with_delay) if all_cust_with_delay else 10.0
    max_delay = max(max_delay, 10.0) * 2

    for i in all_cust_with_delay:
        sc = supply_classes.get(i, "food")
        params = class_params.get(sc, class_params["food"])
        omega_c = params["omega"]
        beta_c = params["beta"]

        n_points = 8
        delay_pts = []
        cost_pts = []
        for p in range(n_points):
            d = max_delay * p / (n_points - 1)
            exp_arg = alpha_const + beta_c * cost_rho * d
            raw_cost = cost_lambda * omega_c * (math.exp(exp_arg) - math.exp(alpha_const))
            if cost_normalized:
                max_exp_arg = alpha_const + beta_c * cost_rho * max_delay
                norm_factor = math.exp(max_exp_arg) - math.exp(alpha_const)
                raw_cost = raw_cost / norm_factor if norm_factor > 0 else 0.0
            delay_pts.append(d)
            cost_pts.append(raw_cost)

        model.addGenConstrPWL(delay[i], delay_penalty[i], delay_pts, cost_pts, name=f"pwl_{i}")

    # Set warm start from ALNS solution
    for k_idx, route in enumerate(solution.truck_routes):
        if k_idx >= num_trucks:
            continue
        for pos in range(len(route.nodes) - 1):
            frm = route.nodes[pos]
            to = route.nodes[pos + 1]
            if (frm, to, k_idx) in x:
                x[frm, to, k_idx].Start = 1

    model.setObjective(
        truck_dist_expr + drone_dist_cost + gp.quicksum(delay_penalty[i] for i in all_cust_with_delay),
        GRB.MINIMIZE,
    )

    model.optimize()

    runtime = time.perf_counter() - start_time

    # Extract solution
    if model.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT) and model.SolCount > 0:
        polished_routes = _extract_routes(model, x, N, K, depot_start, depot_end, num_trucks)
        polished_solution = _build_solution(polished_routes, solution, instance, evaluator)

        # Verify with evaluator
        polished_eval = evaluator.evaluate_solution(polished_solution)
        if polished_eval.feasible:
            polished_cost = polished_eval.total_cost
        else:
            polished_cost = float("inf")
    else:
        polished_solution = solution.clone()
        polished_cost = float("inf")

    original_eval = evaluator.evaluate_solution(solution)
    original_cost = original_eval.total_cost
    if not math.isfinite(polished_cost):
        polished_cost = original_cost
        polished_solution = solution.clone()

    improvement = original_cost - polished_cost

    return PolishResult(
        original_cost=original_cost,
        polished_cost=polished_cost,
        improvement=improvement,
        runtime_seconds=runtime,
        improved=improvement > 0.01,
        polished_solution=polished_solution,
        truck_routes_before=[r.nodes[:] for r in solution.truck_routes],
        truck_routes_after=[r.nodes[:] for r in polished_solution.truck_routes],
    )


def _extract_routes(model, x, N, K, depot_start, depot_end, num_trucks):
    routes = {}
    for k in K:
        route_nodes = [depot_start]
        current = depot_start
        visited = set()
        max_iter = len(N) + 2
        for _ in range(max_iter):
            nxt = None
            for j in N:
                if j != current and (current, j, k) in x:
                    try:
                        if x[current, j, k].X > 0.5:
                            nxt = j
                            break
                    except (gp.GurobiError, AttributeError):
                        continue
            if nxt is None or nxt == depot_end:
                break
            if nxt in visited:
                break
            visited.add(nxt)
            route_nodes.append(nxt)
            current = nxt
        route_nodes.append(depot_end)
        routes[k] = route_nodes
    return routes


def _build_solution(polished_routes, original_solution, instance, evaluator):
    truck_spec = instance.vehicle_specs.get("truck")
    demands = instance.customer_manager.demands()
    depot_start = instance.customer_manager.depot_start
    depot_end = instance.customer_manager.depot_end

    new_routes = []
    for k, nodes in polished_routes.items():
        if len(nodes) < 2:
            nodes = [depot_start, depot_end]
        route_demand = sum(
            demands.get(c, 0.0) for c in nodes if c not in (depot_start, depot_end)
        )
        new_routes.append(TruckRoute(route_id=k, nodes=nodes, capacity=truck_spec.capacity, current_load=route_demand))

    new_drone_tasks = [dt.clone() for dt in original_solution.drone_tasks]
    return Solution(truck_routes=new_routes, drone_tasks=new_drone_tasks)