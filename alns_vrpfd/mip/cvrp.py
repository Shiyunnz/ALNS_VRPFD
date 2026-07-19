"""Solve a capacitated VRP for trucks using Gurobi."""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # pragma: no cover - handled in caller
    gp = None  # type: ignore
    GRB = None  # type: ignore

from ..instance.manager import InstanceManager
from ..utils.constants import DEFAULT_TRUCK_CAPACITY

logger = logging.getLogger(__name__)


class CVRPSolverError(RuntimeError):
    """Raised when the CVRP builder cannot return a feasible route set."""


def solve_cvrp_truck_routes(
    instance: InstanceManager,
    *,
    time_limit: float | None = None,
    mip_gap: float | None = 0.05,
    truck_fixed_cost: float = 1000.0, # Penalty for using a truck
) -> List[List[int]]:
    """Return truck routes covering all customers by solving a CVRP with time windows.

    Parameters
    ----------
    instance:
        Populated instance containing depot, demand, and distance data.
    time_limit:
        Optional solver time limit in seconds.
    mip_gap:
        Optional relative MIP gap tolerance.
    truck_fixed_cost:
        Cost added to objective for each truck used (default 1000.0).
        Encourages minimizing fleet size.

    Returns
    -------
    List[List[int]]
        Sequence of node lists, each representing a truck route from the start
        depot to the end depot (or back to the start when both depots coincide).

    Raises
    ------
    CVRPSolverError
        If Gurobi is unavailable, no feasible routes exist, or solver output
        cannot be interpreted.
    """

    if gp is None or GRB is None:
        raise CVRPSolverError("Gurobi is not available to build CVRP routes.")

    start = instance.customer_manager.depot_start
    if start is None:
        raise CVRPSolverError("Instance is missing a configured start depot.")

    end = instance.customer_manager.depot_end or start
    customers = list(instance.customer_manager.customer_ids())

    truck_spec = instance.vehicle_specs.get("truck")
    num_trucks = truck_spec.number if truck_spec is not None else 1
    capacity = truck_spec.capacity if truck_spec is not None else DEFAULT_TRUCK_CAPACITY
    if capacity <= 0:
        raise CVRPSolverError(
            "Truck capacity must be positive to build routes.")

    if not customers:
        # No customers to visit. Return placeholder routes for each truck.
        route_nodes = [start, end] if start != end else [start, start]
        return [list(route_nodes) for _ in range(num_trucks)]

    demands = instance.customer_manager.demands()
    node_ids = instance.all_node_ids()
    node_index = {node: idx for idx, node in enumerate(node_ids)}
    distance_matrix = instance.distance_matrix("truck")

    def _distance(i: int, j: int) -> float:
        ii = node_index[i]
        jj = node_index[j]
        return distance_matrix[ii][jj]

    arcs = _build_arc_set(start, end, customers, _distance)
    if not arcs:
        raise CVRPSolverError(
            "No finite-distance arcs available for CVRP model.")

    model = gp.Model("initial_truck_cvrp")
    model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap

    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
    u = model.addVars(customers, lb=0.0, ub=capacity, name="u")

    # Time variables
    time_matrix = instance.time_matrix("truck")

    def _travel_time(i: int, j: int) -> float:
        ii = node_index[i]
        jj = node_index[j]
        return time_matrix[ii][jj]

    start_equals_end = start == end

    # Arrival time variables for all nodes
    all_nodes = [start] + customers + ([end] if not start_equals_end else [])
    t = model.addVars(all_nodes, lb=0.0, name="t")

    # Delay variables for soft time windows (only for customers)
    d = model.addVars(customers, lb=0.0, name="d")
    outgoing: Dict[int, List[int]] = {}
    incoming: Dict[int, List[int]] = {}
    arc_costs: Dict[Tuple[int, int], float] = {}
    for i, j in arcs:
        outgoing.setdefault(i, []).append(j)
        incoming.setdefault(j, []).append(i)
        arc_costs[(i, j)] = _distance(i, j)

    # Each customer has exactly one predecessor and one successor.
    for customer in customers:
        preds = [i for i in incoming.get(customer, []) if i != customer]
        succs = [j for j in outgoing.get(customer, []) if j != customer]
        if not preds or not succs:
            raise CVRPSolverError(
                f"Customer {customer} lacks feasible predecessor or successor arcs."
            )
        model.addConstr(gp.quicksum(x[i, customer]
                        for i in preds) == 1, name=f"in_{customer}")
        model.addConstr(gp.quicksum(x[customer, j]
                        for j in succs) == 1, name=f"out_{customer}")

    # Depot degree bounds.
    start_successors = outgoing.get(start, [])
    if not start_successors:
        raise CVRPSolverError(
            "No outgoing arcs from start depot in CVRP model.")
    model.addConstr(
        gp.quicksum(x[start, j] for j in start_successors) <= num_trucks,
        name="start_degree",
    )

    if start_equals_end:
        start_predecessors = incoming.get(start, [])
        model.addConstr(
            gp.quicksum(x[i, start] for i in start_predecessors) <= num_trucks,
            name="end_degree",
        )
        model.addConstr(
            gp.quicksum(x[start, j] for j in start_successors)
            == gp.quicksum(x[i, start] for i in start_predecessors),
            name="depot_balance",
        )
    else:
        end_predecessors = incoming.get(end, [])
        if not end_predecessors:
            raise CVRPSolverError(
                "No incoming arcs to end depot in CVRP model.")
        model.addConstr(
            gp.quicksum(x[i, end] for i in end_predecessors) <= num_trucks,
            name="end_degree",
        )
        model.addConstr(
            gp.quicksum(x[start, j] for j in start_successors)
            == gp.quicksum(x[i, end] for i in end_predecessors),
            name="depot_balance",
        )

    # MTZ load constraints to remove subtours.
    for customer in customers:
        demand = max(demands.get(customer, 0.0), 0.0)
        model.addConstr(u[customer] >= demand, name=f"load_lb_{customer}")
        model.addConstr(u[customer] <= capacity, name=f"load_ub_{customer}")

    for i in customers:
        for j in customers:
            if i == j or (i, j) not in arcs:
                continue
            demand_j = max(demands.get(j, 0.0), 0.0)
            model.addConstr(
                u[i] - u[j] + capacity * x[i, j] <= capacity - demand_j,
                name=f"mtz_{i}_{j}",
            )

    # Time window constraints
    # Set depot start time to 0
    model.addConstr(t[start] == 0, name="depot_start_time")

    # Time propagation constraints
    M = 1e6  # Big-M constant for time window constraints
    for i, j in arcs:
        travel_time = _travel_time(i, j)
        model.addConstr(
            t[j] >= t[i] + travel_time - M * (1 - x[i, j]),
            name=f"time_prop_{i}_{j}"
        )

    # Hard time window constraints (latest time)
    for customer in customers:
        optimal_time, latest_time = instance.customer_manager.time_window(
            customer)
        if latest_time is not None:
            model.addConstr(t[customer] <= latest_time,
                            name=f"hard_tw_{customer}")

    # Soft time window constraints (optimal time with delay penalty)
    for customer in customers:
        optimal_time, latest_time = instance.customer_manager.time_window(
            customer)
        if optimal_time is not None:
            model.addConstr(
                t[customer] >= optimal_time - d[customer],
                name=f"soft_tw_lb_{customer}"
            )
            model.addConstr(
                t[customer] <= optimal_time + d[customer],
                name=f"soft_tw_ub_{customer}"
            )

    # Objective function: distance cost + delay penalty cost + fixed truck cost
    distance_cost = gp.quicksum(arc_costs[i, j] * x[i, j] for (i, j) in arcs)
    delay_cost = gp.quicksum(d[customer] for customer in customers)
    
    # Count used trucks: sum of outgoing arcs from start depot
    # We already have `start_successors`
    trucks_used = gp.quicksum(x[start, j] for j in start_successors)
    fixed_cost = truck_fixed_cost * trucks_used

    total_cost = distance_cost + delay_cost + fixed_cost

    model.setObjective(total_cost, GRB.MINIMIZE)

    model.optimize()
    if model.Status not in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL}:
        raise CVRPSolverError(
            f"Gurobi failed to produce a CVRP solution (status={model.Status}).")
    if model.SolCount == 0:
        raise CVRPSolverError("Gurobi reported no feasible CVRP routes.")

    successor: Dict[int, List[int]] = {}
    for i, j in arcs:
        if x[i, j].X > 0.5:
            successor.setdefault(i, []).append(j)

    routes = _extract_routes(successor, start, end)
    if not routes:
        raise CVRPSolverError(
            "No truck routes could be reconstructed from CVRP solution.")

    total_visits = sum(max(len(route) - 2, 0) for route in routes)
    if total_visits != len(customers):
        raise CVRPSolverError(
            "CVRP solution assigned some customers multiple times.")

    visited_customers = {node for route in routes for node in route[1:-1]}
    if set(customers) != visited_customers:
        missing = sorted(set(customers) - visited_customers)
        raise CVRPSolverError(f"CVRP solution omitted customers: {missing}")

    return routes


def _build_arc_set(
    start: int,
    end: int,
    customers: Sequence[int],
    distance_fn: Callable[[int, int], float],
) -> List[Tuple[int, int]]:
    arcs: Dict[Tuple[int, int], None] = {}
    start_equals_end = start == end

    for customer in customers:
        dist_start = distance_fn(start, customer)
        if dist_start != float("inf"):
            arcs[(start, customer)] = None

    for i in customers:
        for j in customers:
            if i == j:
                continue
            dist = distance_fn(i, j)
            if dist != float("inf"):
                arcs[(i, j)] = None

    if start_equals_end:
        for customer in customers:
            dist_back = distance_fn(customer, start)
            if dist_back != float("inf"):
                arcs[(customer, start)] = None
    else:
        for customer in customers:
            dist_end = distance_fn(customer, end)
            if dist_end != float("inf"):
                arcs[(customer, end)] = None

    return list(arcs.keys())


def _extract_routes(
    successor: Dict[int, List[int]],
    start: int,
    end: int,
) -> List[List[int]]:
    """Reconstruct ordered routes using the selected successor arcs."""
    routes: List[List[int]] = []
    terminal = start if start == end else end
    start_successors = list(successor.get(start, []))

    while start_successors:
        current = start_successors.pop()
        route = [start, current]
        hops = 0
        max_hops = sum(len(nodes) for nodes in successor.values()) + 1

        while current != terminal and hops < max_hops:
            next_candidates = successor.get(current)
            if not next_candidates:
                return []
            next_node = next_candidates.pop()
            route.append(next_node)
            current = next_node
            hops += 1

        if current != terminal:
            return []
        routes.append(route)

    return routes
