"""Mixed-integer linear programming (MILP) model builder using Gurobi."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

try:  # Optional import guard to provide a clear error message if gurobipy is absent.
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # pragma: no cover - handled at runtime when Gurobi is missing.
    gp = None  # type: ignore
    GRB = None  # type: ignore

from alns_vrpfd.evaluation.energy import DroneEnergyModel
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
    optimal_times: Dict[int, float]
    latest_times: Dict[int, float]
    truck_capacity: float
    drone_capacity: float
    battery_capacity: float
    truck_distance: Dict[Tuple[int, int], float]
    drone_distance: Dict[Tuple[int, int], float]
    truck_time: Dict[Tuple[int, int], float]
    drone_time: Dict[Tuple[int, int], float]
    truck_unit_cost: float
    drone_unit_cost: float
    energy_nominal: Dict[Tuple[int, int], float]
    energy_deviation: Dict[Tuple[int, int], float]
    energy_deviation_rate: float
    gamma_range: Tuple[int, ...]
    epsilon: float
    energy_budget: int


@dataclass
class VariableContainer:
    """Hold references to Gurobi variables created for the MILP model."""

    x_truck: gp.tupledict
    y_drone: gp.tupledict
    z_coupling: gp.tupledict
    u: gp.tupledict  # u_{ikd} = 1 if truck k launches drone d at node i
    arrival_truck: gp.tupledict
    arrival_drone: gp.tupledict
    load_truck: gp.tupledict
    load_drone_minus: gp.tupledict
    load_drone_plus: gp.tupledict
    tardiness_truck: gp.tupledict
    tardiness_drone: gp.tupledict
    deviation: gp.tupledict
    energy_state_gamma: gp.tupledict  # e_{idγ} 不同偏差使用次数下的能耗


@dataclass
class MIPArtifacts:
    """Return bundle containing the model, data, and variable handles."""

    model: gp.Model
    data: ProblemData
    variables: VariableContainer


def build_mip_model(
    instance: InstanceManager,
    *,
    epsilon: float = 1e-3,
    energy_budget: Optional[int] = None,
    solver_parameters: Optional[Mapping[str, float | int | str]] = None,
    use_piecewise_energy: bool = True,  # Default to True for correct energy modeling
    num_segments: int = 10,
    use_gurobi_pwl: bool = True,
    robust_energy: bool = False,
    big_m_time: float = 1e5,
    big_m_load: float = 1e5,
    # Reduced from 1e5 to prevent constraint relaxation due to binary variable precision
    big_m_energy: float = 20.0,
    tardiness_weight: float = 2.0,
) -> MIPArtifacts:
    """Build the complete MILP model with piecewise linear energy constraints.

    This function builds the full MILP model including all constraints and objective.
    It uses piecewise linear approximation for the nonlinear energy function to ensure
    correct coupling between drone load and energy consumption.

    Parameters
    ----------
    instance:
        Problem data container configured via the existing ALNS loaders.
    epsilon:
        Small positive constant used in time-related constraints.
    energy_budget:
        Global energy uncertainty budget. If ``None`` the instance's robust
        configuration value is used.
    solver_parameters:
        Optional mapping of Gurobi parameter names to values applied after the
        model is created.
    use_piecewise_energy:
        Whether to use piecewise linear approximation for energy constraints.
        Defaults to True for correct energy modeling.
    num_segments:
        Number of segments for piecewise linear approximation (only used if
        use_piecewise_energy=True).
    use_gurobi_pwl:
        Whether to use Gurobi's addGenConstrPWL (recommended). If False,
        uses manual SOS2 implementation.
    robust_energy:
        If True, use Bertsimas budgeted uncertainty set for robust energy constraints.
        This ensures drone routes remain feasible when up to gamma arcs reach maximum energy.
    big_m_time:
        Big-M constant for time constraints.
    big_m_load:
        Big-M constant for load constraints.
    big_m_energy:
        Big-M constant for energy constraints.
    tardiness_weight:
        Weight for tardiness penalty in objective function.

    Returns
    -------
    MIPArtifacts
        Bundle containing the complete Gurobi model, extracted data, and
        created variable handles.
    """

    if gp is None or GRB is None:
        raise RuntimeError(
            "Gurobi is not available. Please install gurobipy before building the MILP model."
        )

    # Build basic model skeleton
    data = _extract_problem_data(
        instance,
        epsilon=epsilon,
        energy_budget=energy_budget,
    )

    model = gp.Model("truck_drone_vrp")
    if solver_parameters:
        for name, value in solver_parameters.items():
            model.setParam(name, value)

    variables = _create_decision_variables(model, data)

    # Create artifacts bundle
    artifacts = MIPArtifacts(model=model, data=data, variables=variables)

    # Add core constraints
    add_core_constraints(
        artifacts,
        big_m_time=big_m_time,
        big_m_load=big_m_load,
        big_m_energy=big_m_energy,
        use_piecewise_energy=use_piecewise_energy,
    )

    # Add piecewise linear energy constraints if requested
    if use_piecewise_energy:
        try:
            from .piecewise_energy import add_piecewise_linear_energy_constraints

            print(
                f"Adding piecewise linear energy constraints (K={num_segments})...")
            new_vars = add_piecewise_linear_energy_constraints(
                artifacts.model,
                artifacts.data,
                artifacts.variables,
                big_m_energy,
                num_segments=num_segments,
                use_gurobi_pwl=use_gurobi_pwl,
                robust_energy=robust_energy,
            )
            # Add piecewise variables to container for inspection
            artifacts.variables.omega_active = new_vars.get('omega_active')
            artifacts.variables.power_approx = new_vars.get('power_approx')
            artifacts.variables.energy_active = new_vars.get('energy_active')
            print(f"Piecewise energy constraints added successfully")
        except ImportError:
            raise RuntimeError(
                "Piecewise energy module not found. Make sure piecewise_energy.py is available."
            )

    # Set objective function
    set_distance_tardiness_objective(
        artifacts,
        tardiness_weight=1.0,
    )

    # Update model to register all constraints
    artifacts.model.update()

    return artifacts


def set_distance_tardiness_objective(
    artifacts: MIPArtifacts,
    *,
    truck_unit_cost: Optional[float] = None,
    drone_unit_cost: Optional[float] = None,
    # Match ALNS delay_penalty for consistent evaluation
    tardiness_weight: float | None = 2.0,
) -> None:
    """Apply the distance-plus-tardiness objective defined in Eq. (1).

    Parameters
    ----------
    artifacts:
        The bundle returned by :func:`build_mip_model`.
    truck_unit_cost, drone_unit_cost:
        Override distance coefficients. Defaults fall back to the instance
        values stored in ``ProblemData``.
    tardiness_weight:
        Scaling coefficient for the tardiness penalty term ``f(τ)``. When set
        to ``None`` or ``0`` the tardiness contribution is omitted (useful for
        pure distance comparisons). The default assumes a linear penalty where
        ``f(τ) = τ``.
    """

    if gp is None or GRB is None:
        raise RuntimeError(
            "Gurobi is not available. Please install gurobipy before setting the objective."
        )

    model = artifacts.model
    data = artifacts.data
    vars = artifacts.variables

    truck_cost_coeff = data.truck_unit_cost if truck_unit_cost is None else truck_unit_cost
    drone_cost_coeff = data.drone_unit_cost if drone_unit_cost is None else drone_unit_cost

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

    if tardiness_weight:
        tardiness_expr = gp.quicksum(
            vars.tardiness_truck[i, k] for i in data.customers for k in data.trucks
        ) + gp.quicksum(
            vars.tardiness_drone[i, d] for i in data.customers for d in data.drones
        )
        objective += float(tardiness_weight) * tardiness_expr

    model.setObjective(objective, GRB.MINIMIZE)


def _extract_problem_data(
    instance: InstanceManager,
    *,
    epsilon: float,
    energy_budget: Optional[int],
) -> ProblemData:
    depot_start = instance.customer_manager.depot_start
    depot_end = instance.customer_manager.depot_end
    if depot_start is None:
        raise ValueError(
            "Instance must define a start depot before building the MILP model.")
    if depot_end is None:
        depot_end = depot_start

    deviation_rate = instance.robust_config.energy_deviation_rate

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
    optimal_times: Dict[int, float] = {}
    latest_times: Dict[int, float] = {}
    for cid in customers:
        opt, latest = instance.customer_manager.time_window(cid)
        if opt is None:
            opt = 0.0
        if latest is None:
            latest = opt
        optimal_times[cid] = float(opt)
        latest_times[cid] = float(latest)

    truck_matrix = instance.distance_matrix("truck")
    drone_matrix = instance.distance_matrix("drone")
    truck_time_matrix = instance.time_matrix("truck")
    drone_time_matrix = instance.time_matrix("drone")
    index = {node: idx for idx, node in enumerate(nodes)}

    truck_distance = {
        (i, j): truck_matrix[index[i]][index[j]] for i in nodes for j in nodes
    }
    drone_distance = {
        (i, j): drone_matrix[index[i]][index[j]] for i in nodes for j in nodes
    }
    truck_time = {
        (i, j): truck_time_matrix[index[i]][index[j]] for i in nodes for j in nodes
    }
    drone_time = {
        (i, j): drone_time_matrix[index[i]][index[j]] for i in nodes for j in nodes
    }

    battery_capacity = instance.robust_config.drone_battery_capacity
    if energy_budget is None:
        energy_budget = instance.robust_config.energy_uncertainty_budget

    gamma_range = tuple(range(max(0, energy_budget) + 1))

    energy_model = DroneEnergyModel()
    energy_nominal: Dict[Tuple[int, int], float] = {}
    energy_deviation: Dict[Tuple[int, int], float] = {}
    for (i, j) in truck_distance.keys():
        travel_time = drone_time[(i, j)]
        # WARNING: Static payload assumption for non-PWL mode.
        # If use_piecewise_energy=True (default), this is ignored and dynamic payload is used.
        # If False, we assume payload = demand[j] (delivery) as approximation.
        payload = demand.get(j, 0.0)
        energy_value = 0.0
        if not math.isinf(travel_time):
            energy_value = energy_model.energy_kwh(payload, travel_time)

        energy_nominal[(i, j)] = energy_value
        energy_deviation[(i, j)] = deviation_rate * energy_value

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
        optimal_times=optimal_times,
        latest_times=latest_times,
        truck_capacity=truck_spec.capacity,
        drone_capacity=drone_spec.capacity,
        battery_capacity=battery_capacity,
        truck_distance=truck_distance,
        drone_distance=drone_distance,
        truck_time=truck_time,
        drone_time=drone_time,
        truck_unit_cost=truck_spec.unit_cost,
        drone_unit_cost=drone_spec.unit_cost,
        energy_nominal=energy_nominal,
        energy_deviation=energy_deviation,
        energy_deviation_rate=deviation_rate,
        gamma_range=gamma_range,
        epsilon=epsilon,
        energy_budget=energy_budget,
    )


def _create_decision_variables(model: gp.Model, data: ProblemData) -> VariableContainer:
    x_truck = model.addVars(
        data.arcs,
        data.trucks,
        vtype=GRB.BINARY,
        name="x",
    )

    y_drone = model.addVars(
        data.arcs,
        data.drones,
        vtype=GRB.BINARY,
        name="y",
    )

    z_coupling = model.addVars(
        data.arcs,
        data.trucks,
        data.drones,
        vtype=GRB.BINARY,
        name="z",
    )

    u = model.addVars(
        data.nodes,
        data.trucks,
        data.drones,
        vtype=GRB.BINARY,
        name="u",
    )

    arrival_truck = model.addVars(
        data.nodes,
        data.trucks,
        vtype=GRB.CONTINUOUS,
        lb=0.0,
        name="a_truck",
    )

    arrival_drone = model.addVars(
        data.nodes,
        data.drones,
        vtype=GRB.CONTINUOUS,
        lb=0.0,
        name="a_drone",
    )

    load_truck = model.addVars(
        data.nodes,
        data.trucks,
        vtype=GRB.CONTINUOUS,
        lb=0.0,
        name="omega_truck",
    )

    load_drone_minus = model.addVars(
        data.nodes,
        data.drones,
        vtype=GRB.CONTINUOUS,
        lb=0.0,
        name="omega_drone_minus",
    )

    load_drone_plus = model.addVars(
        data.nodes,
        data.drones,
        vtype=GRB.CONTINUOUS,
        lb=0.0,
        name="omega_drone_plus",
    )

    tardiness_truck = model.addVars(
        data.customers,
        data.trucks,
        vtype=GRB.CONTINUOUS,
        lb=0.0,
        name="tau_truck",
    )

    tardiness_drone = model.addVars(
        data.customers,
        data.drones,
        vtype=GRB.CONTINUOUS,
        lb=0.0,
        name="tau_drone",
    )

    deviation = model.addVars(
        data.arcs,
        vtype=GRB.CONTINUOUS,
        lb=0.0,
        name="xi",
    )

    energy_state_gamma = model.addVars(
        data.nodes,
        data.drones,
        data.gamma_range,
        vtype=GRB.CONTINUOUS,
        lb=0.0,
        name="e_gamma",
    )

    return VariableContainer(
        x_truck=x_truck,
        y_drone=y_drone,
        z_coupling=z_coupling,
        u=u,
        arrival_truck=arrival_truck,
        arrival_drone=arrival_drone,
        load_truck=load_truck,
        load_drone_minus=load_drone_minus,  # ω_{id}^{D-} 无人机访问节点前的累计载荷
        load_drone_plus=load_drone_plus,    # ω_{id}^{D+} 无人机访问节点后的累计载荷
        tardiness_truck=tardiness_truck,
        tardiness_drone=tardiness_drone,
        deviation=deviation,
        energy_state_gamma=energy_state_gamma,
    )


def add_core_constraints(
    artifacts: MIPArtifacts,
    *,
    big_m_time: float = 1e5,
    big_m_load: float = 1e5,
    big_m_energy: float = 20.0,  # Reduced to prevent constraint relaxation
    use_piecewise_energy: bool = True,
) -> None:
    """Add flow, coupling, timing, load, and energy constraints to the model."""

    model = artifacts.model
    data = artifacts.data
    vars = artifacts.variables



    arc_set = set(data.arcs)
    start_depot, end_depot = data.depots
    depot_deadline = max(data.latest_times.values(), default=0.0)
    depot_latest = data.latest_times.get(start_depot, depot_deadline)

    # --- Truck flow constraints ---
    for k in data.trucks:
        outgoing = gp.quicksum(
            vars.x_truck[start_depot, j, k] for j in data.v_plus if (start_depot, j) in arc_set
        )
        incoming = gp.quicksum(
            vars.x_truck[i, end_depot, k] for i in data.v0 if (i, end_depot) in arc_set
        )
        model.addConstr(outgoing == incoming, name=f"truck_depot_flow[{k}]")

        model.addConstr(outgoing <= 1, name=f"truck_single_departure[{k}]")

        if (start_depot, end_depot) in arc_set and start_depot != end_depot:
            model.addConstr(
                vars.x_truck[start_depot, end_depot, k] == 0,
                name=f"truck_no_direct_return[{k}]",
            )

        for j in data.customers:
            inbound = gp.quicksum(
                vars.x_truck[i, j, k] for i in data.v0 if (i, j) in arc_set
            )
            outbound = gp.quicksum(
                vars.x_truck[j, h, k] for h in data.v_plus if (j, h) in arc_set
            )
            model.addConstr(inbound == outbound,
                            name=f"truck_flow_balance[{j},{k}]")

    for j in data.customers:
        visit = gp.quicksum(
            vars.x_truck[i, j, k]
            for k in data.trucks
            for i in data.v0
            if (i, j) in arc_set
        )
        model.addConstr(visit <= 1, name=f"truck_single_visit[{j}]")

    # Constraint: Each drone leaves the start depot at most once.
    # This prevents a single drone ID from performing "parallel" disjoint routes
    # (e.g. 0->A->11 and 0->B->11) which would define two physical drones.
    # For multi-trip, the flow must be continuous: 0 -> Task1 -> Truck -> Task2 -> 11.
    for d in data.drones:
        start_out = gp.quicksum(
            vars.y_drone[start_depot, j, d] for j in data.v_plus if (start_depot, j) in arc_set
        ) + gp.quicksum(
            vars.z_coupling[start_depot, j, k, d]
            for k in data.trucks
            for j in data.v_plus
            if (start_depot, j) in arc_set
        )
        model.addConstr(start_out <= 1, name=f"drone_single_start[{d}]")

        for j in data.customers:
            inbound_j = gp.quicksum(
                vars.y_drone[i, j, d] for i in data.v0 if (i, j) in arc_set
            ) + gp.quicksum(
                vars.z_coupling[i, j, k, d]
                for k in data.trucks
                for i in data.v0
                if (i, j) in arc_set
            )
            outbound_j = gp.quicksum(
                vars.y_drone[j, h, d] for h in data.v_plus if (j, h) in arc_set
            ) + gp.quicksum(
                vars.z_coupling[j, h, k, d]
                for k in data.trucks
                for h in data.v_plus
                if (j, h) in arc_set
            )
            model.addConstr(inbound_j == outbound_j,
                            name=f"drone_flow[{j},{d}]")

    # 约束: 每个节点最多被所有无人机访问一次(对所有无人机求和)
    # for j in data.v_plus:
    #     visits = gp.quicksum(
    #         vars.y_drone[i, j, d]
    #         for d in data.drones
    #         for i in data.v0
    #         if (i, j) in arc_set
    #     ) + gp.quicksum(
    #         vars.z_coupling[i, j, k, d]
    #         for d in data.drones
    #         for k in data.trucks
    #         for i in data.v0
    #         if (i, j) in arc_set
    #     )
    #     model.addConstr(visits <= 1, name=f"drone_single_visit[{j}]")

    # Constraint `drone_single_visit` removed:
    # It prevented a drone from arriving at a node twice (e.g. once by truck, once by air).
    # This blocked "Launch and Retrieve at same node" maneuvers (Arrive by Truck -> Launch -> Serve -> Return by Air).
    # Flow conservation and Capacity/Time constraints are sufficient to prevent infinite loops.

    for (i, j) in arc_set:
        for k in data.trucks:
            for d in data.drones:
                model.addConstr(
                    vars.z_coupling[i, j, k, d] <= vars.x_truck[i, j, k],
                    name=f"couple_leq_x[{i},{j},{k},{d}]",
                )

    # --- Load Constraints ---




    # --- Visit coupling constraints ---
    for j in data.customers:
        truck_v = gp.quicksum(
            vars.x_truck[i, j, k]
            for k in data.trucks
            for i in data.v0
            if (i, j) in arc_set
        )
        
        # Use drone_served_var for coverage check (MUST SERVE)
        drone_s = gp.quicksum(drone_served_var[j, d] for d in data.drones)
        
        # Visit Lower: At least one vehicle must SERVE the customer
        # Truck visit implies service (assumption). Drone service is explicit.
        model.addConstr(
            truck_v + drone_s >= 1,
            name=f"visit_lower[{j}]",
        )
        
        # Visit Upper: Physical visits (congestion/consistency)
        drone_v_physical = gp.quicksum(
            vars.y_drone[i, j, d]
            for d in data.drones
            for i in data.v0
            if (i, j) in arc_set
        )
        
        sync = gp.quicksum(
            vars.u[j, k, d]
            for k in data.trucks
            for d in data.drones
        )
        
        # Sum of Z_out (Retrieved/Carried out)
        z_out_sum = gp.quicksum(
            vars.z_coupling[j, h, k, d]
            for k in data.trucks
            for d in data.drones
            for h in data.v_plus
            if (j, h) in arc_set
        )
        
        model.addConstr(
            truck_v + drone_v_physical <= 1 + sync + z_out_sum,
            name=f"visit_upper[{j}]",
        )

    # Sync constraints (Same truck launch/retrieve logic)
    for j in data.customers:
        for k in data.trucks:
            for d in data.drones:
                truck_io = gp.quicksum(vars.x_truck[i, j, k] for i in data.v0 if (i, j) in arc_set) + \
                           gp.quicksum(vars.x_truck[j, h, k] for h in data.v_plus if (j, h) in arc_set)
                drone_io = gp.quicksum(vars.y_drone[i, j, d] for i in data.v0 if (i, j) in arc_set) + \
                           gp.quicksum(vars.y_drone[j, h, d] for h in data.v_plus if (j, h) in arc_set)
                
                # Sum of Z_out at this node for this truck/drone
                z_out_local = gp.quicksum(
                    vars.z_coupling[j, h, k, d] for h in data.v_plus if (j, h) in arc_set
                )

                model.addConstr(truck_io + drone_io >= 3 * vars.u[j, k, d], name=f"sync_lower[{j},{k},{d}]")
                # Relax upper bound if retrieved (z_out=1) or launched (u=1)
                model.addConstr(truck_io + drone_io <= 2 + 2 * vars.u[j, k, d] + z_out_local, name=f"sync_upper[{j},{k},{d}]")

    # --- Time windows and sequencing ---
    for (i, j) in arc_set:
        travel_t = data.truck_time[(i, j)]
        travel_d = data.drone_time[(i, j)]
        for k in data.trucks:
            # Truck time continuity
            model.addConstr(
                vars.arrival_truck[j, k]
                >= vars.arrival_truck[i, k] + travel_t - big_m_time * (1 - vars.x_truck[i, j, k]),
                name=f"truck_time_continuity[{i},{j},{k}]",
            )
        for d in data.drones:
            # Drone flight time continuity
            model.addConstr(
                vars.arrival_drone[j, d]
                >= vars.arrival_drone[i, d] + travel_d - big_m_time * (1 - vars.y_drone[i, j, d]),
                name=f"drone_time_continuity[{i},{j},{d}]",
            )

    # 1. Launch Synchronization: Drone start time >= Truck arrival time
    for i in data.customers:
        for k in data.trucks:
            for d in data.drones:
                model.addConstr(
                    vars.arrival_drone[i, d]
                    >= vars.arrival_truck[i, k] - big_m_time * (1 - vars.u[i, k, d]),
                    name=f"launch_sync[{i},{k},{d}]"
                )

    # Also handle Launch at Depot (u variables exist for all nodes in builder)
    # Check VariableContainer def: u is defined for data.nodes
    for i in data.depots:
        for k in data.trucks:
            for d in data.drones:
                if i in data.nodes:  # Safety check
                    model.addConstr(
                        vars.arrival_drone[i, d]
                        >= vars.arrival_truck[i, k] - big_m_time * (1 - vars.u[i, k, d]),
                        name=f"launch_sync_depot[{i},{k},{d}]"
                    )

    # 2. Retrieval & Carry Synchronization (via Z variables)
    for (i, j) in arc_set:
        for k in data.trucks:
            for d in data.drones:
                # a. Wait for Retrieval: Truck departure (Arrival at j) >= Drone Arrival at i + Travel
                # This ensures Truck waits at i if Drone arrives late
                model.addConstr(
                    vars.arrival_truck[j, k]
                    >= vars.arrival_drone[i, d] + data.truck_time[(i, j)]
                    - big_m_time * (1 - vars.z_coupling[i, j, k, d]),
                    name=f"retrieve_sync[{i},{j},{k},{d}]"
                )

                # b. Carry Continuity: Drone arrival at j == Truck arrival at j
                model.addConstr(
                    vars.arrival_drone[j, d]
                    >= vars.arrival_truck[j, k] - big_m_time * (1 - vars.z_coupling[i, j, k, d]),
                    name=f"carry_sync_lb[{i},{j},{k},{d}]"
                )
                model.addConstr(
                    vars.arrival_drone[j, d]
                    <= vars.arrival_truck[j, k] + big_m_time * (1 - vars.z_coupling[i, j, k, d]),
                    name=f"carry_sync_ub[{i},{j},{k},{d}]"
                )

    for i in data.customers:
        latest = data.latest_times.get(i, depot_deadline)
        optimal = data.optimal_times.get(i, 0.0)
        for k in data.trucks:
            model.addConstr(
                vars.arrival_truck[i, k] <= latest, name=f"truck_deadline[{i},{k}]")
            model.addConstr(
                vars.arrival_truck[i, k] -
                optimal <= vars.tardiness_truck[i, k],
                name=f"truck_tardiness[{i},{k}]",
            )
        for d in data.drones:
            model.addConstr(
                vars.arrival_drone[i, d] <= latest, name=f"drone_deadline[{i},{d}]")
            model.addConstr(
                vars.arrival_drone[i, d] -
                optimal <= vars.tardiness_drone[i, d],
                name=f"drone_tardiness[{i},{d}]",
            )

    for i in data.customers:
        for k in data.trucks:
            model.addConstr(
                vars.arrival_truck[i, k] <= depot_latest, name=f"truck_max[{i},{k}]")
        for d in data.drones:
            model.addConstr(
                vars.arrival_drone[i, d] <= depot_latest, name=f"drone_max[{i},{d}]")

    # --- Energy constraints following paper's gamma-indexed formulation ---

    gamma_min = data.gamma_range[0]
    gamma_max = data.gamma_range[-1]

    # Constraint \eqref{cons:energyBound} removed:
    # It incorrectly forced energy=0 at ANY sync point (Launch OR Retrieval).
    # For retrieval, energy > 0 (consumed during flight) is required.
    # The `energy_capacity` constraint below handles the reset logic correctly:
    # If arriving by truck (z=1, y=0), capacity constraint forces e <= 0 -> e=0.
    # If arriving by air (y=1), capacity constraint forces e <= Cap.

    # 约束 \eqref{cons:energyFlow}: 按 gamma 层递推能耗 (若未启用分段能耗)
    if not use_piecewise_energy:
        for (i, j) in arc_set:
            energy_nom = data.energy_nominal[(i, j)]
            energy_dev = data.energy_deviation[(i, j)]
            for d in data.drones:
                for gamma in data.gamma_range:
                    # 不消耗偏差额度
                    model.addConstr(
                        vars.energy_state_gamma[j, d, gamma]
                        >= vars.energy_state_gamma[i, d, gamma]
                        + energy_nom
                        - big_m_energy * (1 - vars.y_drone[i, j, d]),
                        name=f"energy_flow_nominal[{i},{j},{d},{gamma}]",
                    )

                    if gamma > gamma_min:
                        # 消耗 1 个偏差额度 (从 gamma-1 层转移)
                        model.addConstr(
                            vars.energy_state_gamma[j, d, gamma]
                            >= vars.energy_state_gamma[i, d, gamma - 1]
                            + energy_nom
                            + energy_dev
                            - big_m_energy * (1 - vars.y_drone[i, j, d]),
                            name=f"energy_flow_deviation[{i},{j},{d},{gamma}]",
                        )

    # 约束 \eqref{cons:energyCapacity}: 使用最坏层 (gamma = Gamma_d) 限制电池容量
    for j in data.v_plus:
        for d in data.drones:
            model.addConstr(
                vars.energy_state_gamma[j, d, gamma_max]
                <= data.battery_capacity
                * gp.quicksum(
                    vars.y_drone[i, j, d] for i in data.v0 if (i, j) in arc_set
                ),
                name=f"energy_capacity[{j},{d}]",
            )

    # --- Load constraints ---

    # 卡车载荷约束
    # 约束 \eqref{cons:loadLower} 和 \eqref{cons:loadUpper}: 卡车载荷守恒与更新
    # ω_{jk}^T >= ω_{ik}^T - q_i - Σ_{d∈K^D} ω_{id}^{D+} - M^T(1-x_{ijk}), ∀i∈N, j∈V_+, k∈K^T
    # ω_{jk}^T <= ω_{ik}^T - q_i - Σ_{d∈K^D} ω_{id}^{D+} + M^T(1-x_{ijk}), ∀i∈N, j∈V_+, k∈K^T
    for (i, j) in arc_set:
        if i in data.customers and j in data.v_plus:  # i∈N, j∈V_+
            demand_i = data.demand.get(i, 0.0)
            for k in data.trucks:
                # 无人机在节点 i 离开后的载荷总和
                drone_load_sum = gp.quicksum(
                    vars.load_drone_plus[i, d] for d in data.drones)

                # 约束 \eqref{cons:loadLower}
                model.addConstr(
                    vars.load_truck[j, k]
                    >= vars.load_truck[i, k]
                    - demand_i
                    - drone_load_sum
                    - big_m_load * (1 - vars.x_truck[i, j, k]),
                    name=f"truck_load_lower[{i},{j},{k}]",
                )

                # 约束 \eqref{cons:loadUpper}
                model.addConstr(
                    vars.load_truck[j, k]
                    <= vars.load_truck[i, k]
                    - demand_i
                    - drone_load_sum
                    + big_m_load * (1 - vars.x_truck[i, j, k]),
                    name=f"truck_load_upper[{i},{j},{k}]",
                )

    # 约束 \eqref{cons:initialLoad}: 初始载荷约束
    # ω_{jk}^T <= ω_{0k}^T + M^T(1-x_{0jk}), ∀j∈V_+, k∈K^T
    for j in data.v_plus:
        for k in data.trucks:
            if (start_depot, j) in arc_set:
                model.addConstr(
                    vars.load_truck[j, k]
                    <= vars.load_truck[start_depot, k]
                    + big_m_load * (1 - vars.x_truck[start_depot, j, k]),
                    name=f"initial_truck_load[{j},{k}]",
                )

    # 约束 \eqref{cons:truckCapacity}: 卡车容量上限
    # ω_{ik}^T <= Q^T, ∀i∈V, k∈K^T
    for i in data.nodes:  # V = N (所有节点包括depot)
        for k in data.trucks:
            model.addConstr(
                vars.load_truck[i, k] <= data.truck_capacity,
                name=f"truck_capacity[{i},{k}]",
            )

    # 无人机载荷约束
    # 约束: 无人机在弧上的载荷守恒
    # load_drone_minus[j,d] = load_drone_plus[i,d] 当无人机沿弧(i,j)飞行时
    # 即到达j时的累积负载 = 离开i时的累积负载(已扣除i的需求)
    for (i, j) in arc_set:
        if i in data.v0 and j in data.v_plus:  # i∈V_0, j∈V_+ (Include depot)
            for d in data.drones:
                # 弧上载荷守恒: minus[j] = plus[i] (当弧被使用时)
                model.addConstr(
                    vars.load_drone_minus[j, d]
                    >= vars.load_drone_plus[i, d]
                    - big_m_load * (1 - vars.y_drone[i, j, d]),
                    name=f"drone_load_lower[{i},{j},{d}]",
                )
                model.addConstr(
                    vars.load_drone_minus[j, d]
                    <= vars.load_drone_plus[i, d]
                    + big_m_load * (1 - vars.y_drone[i, j, d]),
                    name=f"drone_load_upper[{i},{j},{d}]",
                )

    # 约束: 在客户节点服务后更新离开时的累积负载
    # load_drone_plus[j,d] = load_drone_minus[j,d] - demand[j] (如果服务)
    # load_drone_plus[j,d] = load_drone_minus[j,d] (如果不服务)
    # 如果在节点j发射/同步(u=1)，则允许负载重置/增加 (plus > minus)，即松弛上述约束
    for j in data.customers:  # j∈N
        demand_j = data.demand.get(j, 0.0)
        for d in data.drones:
            # 无人机访问该节点：检测是否有入边 (服务指示)
            drone_served = gp.quicksum(
                vars.y_drone[i, j, d] for i in data.v0 if (i, j) in arc_set
            ) + gp.quicksum(
                vars.z_coupling[i, j, k, d]
                for k in data.trucks
                for i in data.v0
                if (i, j) in arc_set
            )

            # 同步指示 (发射或回收)
            u_sum = gp.quicksum(vars.u[j, k, d] for k in data.trucks)

            # Unified Load Continuity Constraint:
            # plus = minus - demand * served
            # Relaxed by u_sum (if u=1, plus can be anything up to capacity)

            model.addConstr(
                vars.load_drone_plus[j, d]
                >= vars.load_drone_minus[j, d]
                - demand_j * drone_served
                - big_m_load * u_sum,
                name=f"drone_load_continuity_lower[{j},{d}]",
            )
            model.addConstr(
                vars.load_drone_plus[j, d]
                <= vars.load_drone_minus[j, d]
                - demand_j * drone_served
                + big_m_load * u_sum,
                name=f"drone_load_continuity_upper[{j},{d}]",
            )

    # 约束 \eqref{cons:droneCapacity}: 无人机容量限制
    # ω_{id}^{D-} <= Q^D, ∀i∈V, d∈K^D
    for i in data.nodes:
        for d in data.drones:
            model.addConstr(
                vars.load_drone_minus[i, d] <= data.drone_capacity,
                name=f"drone_capacity[{i},{d}]",
            )

    # 约束: depot 处无人机的初始载荷等于将要服务的客户总需求
    # load_drone_plus[0, d] = Σ_j (demand_j × drone_serves_j[d])
    # 其中 drone_serves_j[d] = Σ_i y[i,j,d] 表示无人机 d 是否服务客户 j
    start_depot = data.nodes[0]
    for d in data.drones:
        total_demand_served = gp.quicksum(
            data.demand.get(j, 0.0) * gp.quicksum(
                vars.y_drone[i, j, d] for i in data.v0 if (i, j) in arc_set
            )
            for j in data.customers
        )
        model.addConstr(
            vars.load_drone_plus[start_depot, d] == total_demand_served,
            name=f"drone_initial_load[{d}]",
        )


def build_mip_model_with_piecewise_energy(
    instance: InstanceManager,
    *,
    epsilon: float = 1e-3,
    energy_budget: Optional[int] = None,
    solver_parameters: Optional[Mapping[str, float | int | str]] = None,
    use_piecewise_energy: bool = False,
    num_segments: int = 10,
    use_gurobi_pwl: bool = True,
) -> MIPArtifacts:
    """Build the complete MILP model with optional piecewise linear energy constraints.

    This function builds the full MILP model including all constraints and objective.
    When use_piecewise_energy=True, it uses piecewise linear approximation for
    the nonlinear energy function instead of pre-computed constants.

    Parameters
    ----------
    instance:
        Problem data container configured via the existing ALNS loaders.
    epsilon:
        Small positive constant used in time-related constraints.
    energy_budget:
        Global energy uncertainty budget. If ``None`` the instance's robust
        configuration value is used.
    solver_parameters:
        Optional mapping of Gurobi parameter names to values applied after the
        model is created.
    use_piecewise_energy:
        Whether to use piecewise linear approximation for energy constraints.
        If False, uses the original pre-computed energy constants.
    num_segments:
        Number of segments for piecewise linear approximation (only used if
        use_piecewise_energy=True).
    use_gurobi_pwl:
        Whether to use Gurobi's addGenConstrPWL (recommended). If False,
        uses manual SOS2 implementation.

    Returns
    -------
    MIPArtifacts
        Bundle containing the complete Gurobi model, extracted data, and
        created variable handles.
    """
    if gp is None or GRB is None:
        raise RuntimeError(
            "Gurobi is not available. Please install gurobipy before building the MILP model."
        )

    # Build basic model skeleton
    artifacts = build_mip_model(
        instance,
        epsilon=epsilon,
        energy_budget=energy_budget,
        solver_parameters=solver_parameters,
    )

    # Add core constraints
    add_core_constraints(
        artifacts,
        big_m_time=1e5,
        big_m_load=1e5,
        big_m_energy=20.0,  # Reduced to prevent constraint relaxation
        use_piecewise_energy=use_piecewise_energy,
    )

    # Add piecewise linear energy constraints if requested
    if use_piecewise_energy:
        try:
            from .piecewise_energy import add_piecewise_linear_energy_constraints

            print(
                f"Adding piecewise linear energy constraints (K={num_segments})...")
            new_vars = add_piecewise_linear_energy_constraints(
                artifacts.model,
                artifacts.data,
                artifacts.variables,
                1e5,  # big_m_energy
                num_segments=num_segments,
                use_gurobi_pwl=use_gurobi_pwl,
            )
            print(f"Piecewise energy constraints added successfully")
        except ImportError:
            raise RuntimeError(
                "Piecewise energy module not found. Make sure piecewise_energy.py is available."
            )

    # Set objective function
    set_distance_tardiness_objective(
        artifacts,
        tardiness_weight=1.0,
    )

    # Update model to register all constraints
    artifacts.model.update()

    return artifacts


__all__ = [
    "ProblemData",
    "VariableContainer",
    "MIPArtifacts",
    "build_mip_model",
    "set_distance_tardiness_objective",
    "add_core_constraints",
]
