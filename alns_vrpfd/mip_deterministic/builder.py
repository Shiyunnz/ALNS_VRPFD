"""Simplified Mixed-integer linear programming (MILP) model builder using Gurobi.

最简单版本：
- 无延误成本
- 无时间窗约束
- 无能耗模型，改用固定飞行时长（endurance）约束
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    gp = None
    GRB = None

from alns_vrpfd.instance.manager import InstanceManager


@dataclass
class ProblemData:
    """Static sets and parameters extracted from the instance description."""

    depots: Tuple[int, int]
    customers: Tuple[int, ...]
    nodes: Tuple[int, ...]
    v0: Tuple[int, ...]
    v_plus: Tuple[int, ...]
    arcs: Tuple[Tuple[int, int], ...]
    trucks: Tuple[int, ...]
    drones: Tuple[int, ...]
    demand: Dict[int, float]
    truck_distance: Dict[Tuple[int, int], float]
    drone_distance: Dict[Tuple[int, int], float]
    truck_time: Dict[Tuple[int, int], float]
    drone_time: Dict[Tuple[int, int], float]
    truck_unit_cost: float
    drone_unit_cost: float
    truck_capacity: float
    drone_capacity: float
    drone_endurance: float


@dataclass
class VariableContainer:
    """Hold references to Gurobi variables created for the MILP model."""

    x_truck: gp.tupledict
    y_drone: gp.tupledict
    z_coupling: gp.tupledict
    u: gp.tupledict
    arrival_truck: gp.tupledict
    arrival_drone: gp.tupledict
    load_truck: gp.tupledict
    load_drone_minus: gp.tupledict
    load_drone_plus: gp.tupledict
    v_served: gp.tupledict


@dataclass
class MIPArtifacts:
    """Return bundle containing the model, data, and variable handles."""

    model: gp.Model
    data: ProblemData
    variables: VariableContainer


def build_mip_model(
    instance: InstanceManager,
    *,
    solver_parameters: Optional[Mapping[str, float | int | str]] = None,
    big_m_time: float = 1e5,
    big_m_load: float = 1e5,
    tardiness_weight: float = 0.0,
    drone_endurance: float = 0.5,
    objective: str = "makespan",
) -> MIPArtifacts:
    """Build the simplified MILP model.

    Parameters
    ----------
    objective : str
        Objective function, either "makespan" (minimize completion time) or "distance" (minimize total cost).
    """

    if gp is None or GRB is None:
        raise RuntimeError(
            "Gurobi is not available. Please install gurobipy before building the MILP model."
        )

    data = _extract_problem_data(instance, drone_endurance=drone_endurance)

    model = gp.Model("truck_drone_vrp_simple")
    if solver_parameters:
        for name, value in solver_parameters.items():
            model.setParam(name, value)

    variables = _create_decision_variables(model, data)

    artifacts = MIPArtifacts(model=model, data=data, variables=variables)

    add_core_constraints(
        artifacts,
        big_m_time=big_m_time,
        big_m_load=big_m_load,
    )

    if objective == "makespan":
        set_makespan_objective(artifacts)
    else:
        set_distance_objective(artifacts)

    artifacts.model.update()

    return artifacts


def set_makespan_objective(artifacts: MIPArtifacts) -> None:
    """Minimize makespan (completion time of the latest event)."""

    model = artifacts.model
    data = artifacts.data
    vars = artifacts.variables

    makespan = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="makespan")

    end_depot = data.depots[1]

    for k in data.trucks:
        model.addConstr(makespan >= vars.arrival_truck[end_depot, k])

    for d in data.drones:
        model.addConstr(makespan >= vars.arrival_drone[end_depot, d])

    model.setObjective(makespan, GRB.MINIMIZE)


def set_distance_objective(artifacts: MIPArtifacts) -> None:
    """Minimize total distance cost."""

    model = artifacts.model
    data = artifacts.data
    vars = artifacts.variables

    truck_cost_coeff = data.truck_unit_cost
    drone_cost_coeff = data.drone_unit_cost

    truck_distance_expr = gp.quicksum(
        truck_cost_coeff * data.truck_distance[(i, j)] * vars.x_truck[i, j, k]
        for (i, j) in data.arcs
        for k in data.trucks
    )

    drone_distance_expr = gp.quicksum(
        drone_cost_coeff * data.drone_distance[(i, j)] * vars.y_drone[i, j, d]
        for (i, j) in data.arcs
        for d in data.drones
    )

    objective = truck_distance_expr + drone_distance_expr

    model.setObjective(objective, GRB.MINIMIZE)


def _extract_problem_data(
    instance: InstanceManager,
    drone_endurance: float = 0.5,
) -> ProblemData:
    depot_start = instance.customer_manager.depot_start
    depot_end = instance.customer_manager.depot_end
    if depot_start is None:
        raise ValueError(
            "Instance must define a start depot before building the MILP model.")
    if depot_end is None:
        depot_end = depot_start

    customers = instance.customer_manager.customer_ids()
    nodes = instance.all_node_ids()
    v0 = tuple(sorted(set(customers) | {depot_start}))
    v_plus = tuple(sorted(set(customers) | {depot_end}))
    arcs = tuple((i, j) for i in v0 for j in v_plus if i != j)

    truck_spec = instance.vehicle_specs.get("truck")
    if truck_spec is None:
        raise ValueError("Instance is missing truck vehicle specifications.")
    drone_spec = instance.vehicle_specs.get("drone")
    if drone_spec is None:
        raise ValueError("Instance is missing drone vehicle specifications.")

    trucks = tuple(range(truck_spec.number))
    drones = tuple(range(drone_spec.number))

    demand = instance.customer_manager.demands()

    truck_matrix = instance.distance_matrix("truck")
    drone_matrix = instance.distance_matrix("drone")
    truck_time_matrix = instance.time_matrix("truck")
    drone_time_matrix = instance.time_matrix("drone")
    index = {node: idx for idx, node in enumerate(nodes)}

    truck_distance = {
        (i, j): truck_matrix[index[i]][index[j]] for i in nodes for j in nodes}
    drone_distance = {
        (i, j): drone_matrix[index[i]][index[j]] for i in nodes for j in nodes}
    truck_time = {
        (i, j): truck_time_matrix[index[i]][index[j]] for i in nodes for j in nodes}
    drone_time = {
        (i, j): drone_time_matrix[index[i]][index[j]] for i in nodes for j in nodes}

    return ProblemData(
        depots=(depot_start, depot_end),
        customers=customers,
        nodes=nodes,
        v0=v0,
        v_plus=v_plus,
        arcs=arcs,
        trucks=trucks,
        drones=drones,
        demand=demand,
        truck_distance=truck_distance,
        drone_distance=drone_distance,
        truck_time=truck_time,
        drone_time=drone_time,
        truck_unit_cost=truck_spec.unit_cost,
        drone_unit_cost=drone_spec.unit_cost,
        truck_capacity=truck_spec.capacity,
        drone_capacity=drone_spec.capacity,
        drone_endurance=drone_endurance,
    )


def _create_decision_variables(model: gp.Model, data: ProblemData) -> VariableContainer:
    x_truck = model.addVars(data.arcs, data.trucks, vtype=GRB.BINARY, name="x")
    y_drone = model.addVars(data.arcs, data.drones, vtype=GRB.BINARY, name="y")
    z_coupling = model.addVars(
        data.arcs, data.trucks, data.drones, vtype=GRB.BINARY, name="z")
    u = model.addVars(data.nodes, data.trucks, data.drones,
                      vtype=GRB.BINARY, name="u")

    arrival_truck = model.addVars(
        data.nodes, data.trucks, vtype=GRB.CONTINUOUS, lb=0.0, name="a_truck")
    arrival_drone = model.addVars(
        data.nodes, data.drones, vtype=GRB.CONTINUOUS, lb=0.0, name="a_drone")

    load_truck = model.addVars(
        data.nodes, data.trucks, vtype=GRB.CONTINUOUS, lb=0.0, name="omega_truck")
    load_drone_minus = model.addVars(
        data.nodes, data.drones, vtype=GRB.CONTINUOUS, lb=0.0, name="omega_drone_minus")
    load_drone_plus = model.addVars(
        data.nodes, data.drones, vtype=GRB.CONTINUOUS, lb=0.0, name="omega_drone_plus")

    v_served = model.addVars(data.customers, data.drones,
                             vtype=GRB.BINARY, name="v_served")

    return VariableContainer(
        x_truck=x_truck,
        y_drone=y_drone,
        z_coupling=z_coupling,
        u=u,
        arrival_truck=arrival_truck,
        arrival_drone=arrival_drone,
        load_truck=load_truck,
        load_drone_minus=load_drone_minus,
        load_drone_plus=load_drone_plus,
        v_served=v_served,
    )


def add_core_constraints(
    artifacts: MIPArtifacts,
    *,
    big_m_time: float = 1e5,
    big_m_load: float = 1e5,
) -> None:
    """添加核心约束（无时间窗、无能耗、仅使用endurance约束）。"""

    model = artifacts.model
    data = artifacts.data
    vars = artifacts.variables

    arc_set = set(data.arcs)
    start_depot, end_depot = data.depots

    # ========== 6.1 Truck Flow Constraints ==========

    for k in data.trucks:
        outgoing = gp.quicksum(
            vars.x_truck[start_depot, j, k] for j in data.v_plus if (start_depot, j) in arc_set
        )
        incoming = gp.quicksum(
            vars.x_truck[i, end_depot, k] for i in data.v0 if (i, end_depot) in arc_set
        )
        model.addConstr(outgoing == incoming, name=f"truck_depot_flow[{k}]")

    for k in data.trucks:
        outgoing = gp.quicksum(
            vars.x_truck[start_depot, j, k] for j in data.v_plus if (start_depot, j) in arc_set
        )
        model.addConstr(outgoing <= 1, name=f"truck_single_departure[{k}]")

    for j in data.customers:
        for k in data.trucks:
            inbound = gp.quicksum(vars.x_truck[i, j, k]
                                  for i in data.v0 if (i, j) in arc_set)
            outbound = gp.quicksum(vars.x_truck[j, h, k]
                                   for h in data.v_plus if (j, h) in arc_set)
            model.addConstr(inbound == outbound,
                            name=f"truck_flow_balance[{j},{k}]")

    for j in data.customers:
        visit = gp.quicksum(
            vars.x_truck[i, j, k] for k in data.trucks for i in data.v0 if (i, j) in arc_set
        )
        model.addConstr(visit <= 1, name=f"truck_single_visit[{j}]")

    # ========== 6.2 Drone Flow Constraints ==========

    for d in data.drones:
        start_out = gp.quicksum(
            vars.y_drone[start_depot, j, d] for j in data.v_plus if (start_depot, j) in arc_set
        ) + gp.quicksum(
            vars.z_coupling[start_depot, j, k, d]
            for k in data.trucks for j in data.v_plus if (start_depot, j) in arc_set
        )
        model.addConstr(start_out <= 1, name=f"drone_single_start[{d}]")

    for d in data.drones:
        for j in data.customers:
            inbound = gp.quicksum(
                vars.y_drone[i, j, d] for i in data.v0 if (i, j) in arc_set
            ) + gp.quicksum(
                vars.z_coupling[i, j, k, d] for k in data.trucks for i in data.v0 if (i, j) in arc_set
            )
            outbound = gp.quicksum(
                vars.y_drone[j, h, d] for h in data.v_plus if (j, h) in arc_set
            ) + gp.quicksum(
                vars.z_coupling[j, h, k, d] for k in data.trucks for h in data.v_plus if (j, h) in arc_set
            )
            model.addConstr(inbound == outbound, name=f"drone_flow[{j},{d}]")

    # ========== 6.3 Truck-Drone Coupling Constraints ==========

    for (i, j) in arc_set:
        for k in data.trucks:
            for d in data.drones:
                model.addConstr(
                    vars.z_coupling[i, j, k, d] <= vars.x_truck[i, j, k],
                    name=f"couple_leq_x[{i},{j},{k},{d}]"
                )

    for j in data.customers:
        truck_visit = gp.quicksum(
            vars.x_truck[i, j, k] for k in data.trucks for i in data.v0 if (i, j) in arc_set
        )
        drone_visit = gp.quicksum(
            vars.y_drone[i, j, d] for d in data.drones for i in data.v0 if (i, j) in arc_set
        )
        sync_sum = gp.quicksum(
            vars.u[j, k, d] for k in data.trucks for d in data.drones
        )
        z_out_sum = gp.quicksum(
            vars.z_coupling[j, h, k, d]
            for k in data.trucks for d in data.drones for h in data.v_plus if (j, h) in arc_set
        )

        model.addConstr(
            truck_visit + drone_visit >= 1 - sync_sum - z_out_sum,
            name=f"visit_lower[{j}]"
        )

        model.addConstr(
            truck_visit + drone_visit <= 1 + sync_sum + z_out_sum,
            name=f"visit_upper[{j}]"
        )

    # ========== 6.4 Synchronization Constraints ==========

    for j in data.customers:
        for k in data.trucks:
            for d in data.drones:
                truck_io = (
                    gp.quicksum(vars.x_truck[i, j, k] for i in data.v0 if (i, j) in arc_set) +
                    gp.quicksum(vars.x_truck[j, h, k]
                                for h in data.v_plus if (j, h) in arc_set)
                )
                drone_io = (
                    gp.quicksum(vars.y_drone[i, j, d] for i in data.v0 if (i, j) in arc_set) +
                    gp.quicksum(vars.y_drone[j, h, d]
                                for h in data.v_plus if (j, h) in arc_set)
                )
                z_out_local = gp.quicksum(
                    vars.z_coupling[j, h, k, d] for h in data.v_plus if (j, h) in arc_set
                )

                model.addConstr(
                    truck_io + drone_io >= 3 * vars.u[j, k, d],
                    name=f"sync_lower[{j},{k},{d}]"
                )
                model.addConstr(
                    truck_io + drone_io <= 2 + 2 *
                    vars.u[j, k, d] + z_out_local,
                    name=f"sync_upper[{j},{k},{d}]"
                )

    for i in data.customers:
        for k in data.trucks:
            for d in data.drones:
                truck_in = gp.quicksum(
                    vars.x_truck[p, i, k] for p in data.v0 if (p, i) in arc_set)
                model.addConstr(
                    truck_in >= vars.u[i, k, d], name=f"launch_requires_truck[{i},{k},{d}]")

    # ========== 6.5 Time Continuity Constraints (No Time Windows) ==========

    for (i, j) in arc_set:
        travel_t = data.truck_time[(i, j)]
        for k in data.trucks:
            model.addConstr(
                vars.arrival_truck[j, k] >= vars.arrival_truck[i, k] +
                travel_t - big_m_time * (1 - vars.x_truck[i, j, k]),
                name=f"truck_time_continuity[{i},{j},{k}]"
            )

    for (i, j) in arc_set:
        travel_d = data.drone_time[(i, j)]
        for d in data.drones:
            model.addConstr(
                vars.arrival_drone[j, d] >= vars.arrival_drone[i, d] +
                travel_d - big_m_time * (1 - vars.y_drone[i, j, d]),
                name=f"drone_time_continuity[{i},{j},{d}]"
            )

    for (i, j) in arc_set:
        travel_d = data.drone_time[(i, j)]
        for k in data.trucks:
            for d in data.drones:
                model.addConstr(
                    vars.arrival_drone[j, d] >= vars.arrival_truck[i, k] +
                    travel_d - big_m_time * (1 - vars.y_drone[i, j, d]),
                    name=f"drone_time_from_truck[{i},{j},{k},{d}]"
                )

    for i in data.customers:
        for k in data.trucks:
            for d in data.drones:
                model.addConstr(
                    vars.arrival_drone[i, d] >= vars.arrival_truck[i,
                                                                   k] - big_m_time * (1 - vars.u[i, k, d]),
                    name=f"launch_sync[{i},{k},{d}]"
                )

    for i in [start_depot]:
        if i in data.nodes:
            for k in data.trucks:
                for d in data.drones:
                    model.addConstr(
                        vars.arrival_drone[i, d] >= vars.arrival_truck[i,
                                                                       k] - big_m_time * (1 - vars.u[i, k, d]),
                        name=f"launch_sync_depot[{i},{k},{d}]"
                    )

    for (i, j) in arc_set:
        for k in data.trucks:
            for d in data.drones:
                model.addConstr(
                    vars.arrival_truck[j, k] >= vars.arrival_drone[i, d] + data.truck_time[(
                        i, j)] - big_m_time * (1 - vars.z_coupling[i, j, k, d]),
                    name=f"retrieve_sync[{i},{j},{k},{d}]"
                )

    for (i, j) in arc_set:
        for k in data.trucks:
            for d in data.drones:
                model.addConstr(
                    vars.arrival_drone[j, d] >= vars.arrival_truck[j, k] -
                    big_m_time * (1 - vars.z_coupling[i, j, k, d]),
                    name=f"carry_sync_lb[{i},{j},{k},{d}]"
                )
                model.addConstr(
                    vars.arrival_drone[j, d] <= vars.arrival_truck[j, k] +
                    big_m_time * (1 - vars.z_coupling[i, j, k, d]),
                    name=f"carry_sync_ub[{i},{j},{k},{d}]"
                )

    # ========== 6.6 Truck Load Constraints ==========

    for (i, j) in arc_set:
        if i in data.customers and j in data.v_plus:
            demand_i = data.demand.get(i, 0.0)
            for k in data.trucks:
                drone_load_sum = gp.quicksum(
                    vars.load_drone_plus[i, d] for d in data.drones)
                model.addConstr(
                    vars.load_truck[j, k] >= vars.load_truck[i, k] - demand_i -
                    drone_load_sum - big_m_load * (1 - vars.x_truck[i, j, k]),
                    name=f"truck_load_lower[{i},{j},{k}]"
                )
                model.addConstr(
                    vars.load_truck[j, k] <= vars.load_truck[i, k] - demand_i -
                    drone_load_sum + big_m_load * (1 - vars.x_truck[i, j, k]),
                    name=f"truck_load_upper[{i},{j},{k}]"
                )

    for j in data.v_plus:
        for k in data.trucks:
            if (start_depot, j) in arc_set:
                model.addConstr(
                    vars.load_truck[j, k] <= vars.load_truck[start_depot, k] +
                    big_m_load * (1 - vars.x_truck[start_depot, j, k]),
                    name=f"initial_truck_load[{j},{k}]"
                )

    for i in data.nodes:
        for k in data.trucks:
            model.addConstr(
                vars.load_truck[i, k] <= data.truck_capacity, name=f"truck_capacity[{i},{k}]")

    # ========== 6.7 Drone Load Constraints ==========

    for (i, j) in arc_set:
        for d in data.drones:
            model.addConstr(
                vars.load_drone_minus[j, d] >= vars.load_drone_plus[i,
                                                                    d] - big_m_load * (1 - vars.y_drone[i, j, d]),
                name=f"drone_load_lower[{i},{j},{d}]"
            )
            model.addConstr(
                vars.load_drone_minus[j, d] <= vars.load_drone_plus[i,
                                                                    d] + big_m_load * (1 - vars.y_drone[i, j, d]),
                name=f"drone_load_upper[{i},{j},{d}]"
            )

    for j in data.customers:
        for d in data.drones:
            y_in = gp.quicksum(vars.y_drone[i, j, d]
                               for i in data.v0 if (i, j) in arc_set)
            z_out = gp.quicksum(
                vars.z_coupling[j, h, k, d] for k in data.trucks for h in data.v_plus if (j, h) in arc_set
            )
            model.addConstr(vars.v_served[j, d] <=
                            y_in, name=f"served_ub_y[{j},{d}]")
            model.addConstr(vars.v_served[j, d] <=
                            1 - z_out, name=f"served_ub_z[{j},{d}]")
            model.addConstr(
                vars.v_served[j, d] >= y_in - z_out, name=f"served_lb[{j},{d}]")

    for j in data.customers:
        demand_j = data.demand.get(j, 0.0)
        for d in data.drones:
            u_sum = gp.quicksum(vars.u[j, k, d] for k in data.trucks)
            model.addConstr(
                vars.load_drone_plus[j, d] >= vars.load_drone_minus[j, d] -
                demand_j * vars.v_served[j, d] - big_m_load * u_sum,
                name=f"drone_load_cont_lower[{j},{d}]"
            )
            model.addConstr(
                vars.load_drone_plus[j, d] <= vars.load_drone_minus[j, d] -
                demand_j * vars.v_served[j, d] + big_m_load * u_sum,
                name=f"drone_load_cont_upper[{j},{d}]"
            )

    for i in data.nodes:
        for d in data.drones:
            model.addConstr(
                vars.load_drone_minus[i, d] <= data.drone_capacity, name=f"drone_capacity[{i},{d}]")
            model.addConstr(
                vars.load_drone_minus[i, d] >= 0.0, name=f"drone_load_nonneg[{i},{d}]")

    for d in data.drones:
        model.addConstr(
            vars.load_drone_plus[start_depot, d] == 0.0,
            name=f"drone_initial_load[{d}]"
        )

    # ========== 7. Endurance Constraints (Simple Flight Time Limit) ==========

    for d in data.drones:
        total_flight_time = gp.quicksum(
            data.drone_time[(i, j)] * vars.y_drone[i, j, d]
            for (i, j) in data.arcs
        )
        model.addConstr(
            total_flight_time <= data.drone_endurance,
            name=f"drone_endurance[{d}]"
        )


__all__ = [
    "ProblemData",
    "VariableContainer",
    "MIPArtifacts",
    "build_mip_model",
    "set_distance_objective",
    "add_core_constraints",
]