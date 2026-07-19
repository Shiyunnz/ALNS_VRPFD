"""Mixed-integer linear programming (MILP) model builder using Gurobi.

根据 milp_formulation.tex 完全重构的版本。
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    gp = None
    GRB = None

from alns_vrpfd.deprivation import deprivation_cost
from alns_vrpfd.evaluation.energy import DroneEnergyModel
from alns_vrpfd.instance.manager import InstanceManager
from typing import List


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
    supply_classes: Dict[int, str]
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
    u: gp.tupledict
    arrival_truck: gp.tupledict
    departure_truck: gp.tupledict
    arrival_drone: gp.tupledict
    load_truck: gp.tupledict
    load_drone_minus: gp.tupledict
    load_drone_plus: gp.tupledict
    tardiness_truck: gp.tupledict
    tardiness_drone: gp.tupledict
    deviation: gp.tupledict
    energy_state_gamma: gp.tupledict
    tardiness_cost_truck: gp.tupledict = None
    tardiness_cost_drone: gp.tupledict = None


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
    use_piecewise_energy: bool = True,
    use_piecewise_delay: bool = True,
    num_segments: int = 10,
    pwl_delay_segments: int | None = 10,
    pwl_delay_spacing_power: float = 1.0,
    use_three_piece_delay: bool = False,
    three_piece_bounds: tuple | None = None,
    three_piece_middle_scale: float | None = None,
    use_gurobi_pwl: bool = True,
    robust_energy: bool = False,
    big_m_time: float = 1000.0,
    big_m_load: float = 1000.0,
    big_m_energy: float = 20.0,
    tardiness_weight: float = 1.0,
    cost_lambda: float = 12.0,
    cost_rho: float = 1.0,
    cost_normalized: bool = True,
) -> MIPArtifacts:
    """Build the complete MILP model following milp_formulation.tex."""

    if gp is None or GRB is None:
        raise RuntimeError(
            "Gurobi is not available. Please install gurobipy before building the MILP model."
        )

    data = _extract_problem_data(
        instance, epsilon=epsilon, energy_budget=energy_budget)

    model = gp.Model("truck_drone_vrp")
    if solver_parameters:
        for name, value in solver_parameters.items():
            model.setParam(name, value)

    variables = _create_decision_variables(model, data)

    artifacts = MIPArtifacts(model=model, data=data, variables=variables)

    add_core_constraints(
        artifacts,
        big_m_time=big_m_time,
        big_m_load=big_m_load,
        big_m_energy=big_m_energy,
        use_piecewise_energy=use_piecewise_energy,
    )

    if use_piecewise_energy:
        try:
            # 如果调用方使用了默认分段数(10)，则尝试从 YAML 配置覆盖它，方便集中管理
            if num_segments == 10:
                try:
                    from alns_vrpfd.utils.config_loader import ALNSConfig

                    cfg = ALNSConfig()
                    num_segments = int(cfg.piecewise_energy_segments)
                except Exception:
                    # 忽略配置加载错误，继续使用传入的 num_segments
                    pass

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
            artifacts.variables.omega_active = new_vars.get('omega_active')
            artifacts.variables.power_approx = new_vars.get('power_approx')
            artifacts.variables.energy_active = new_vars.get('energy_active')
            print(f"Piecewise energy constraints added successfully")
        except ImportError:
            raise RuntimeError("Piecewise energy module not found.")

    set_distance_tardiness_objective(
        artifacts,
        tardiness_weight=tardiness_weight,
        use_piecewise_delay=use_piecewise_delay,
        pwl_delay_segments=pwl_delay_segments,
        pwl_delay_spacing_power=pwl_delay_spacing_power,
        use_three_piece_delay=use_three_piece_delay,
        three_piece_bounds=three_piece_bounds,
        three_piece_middle_scale=three_piece_middle_scale,
        use_gurobi_pwl=use_gurobi_pwl,
        cost_lambda=cost_lambda,
        cost_rho=cost_rho,
        cost_normalized=cost_normalized,
    )
    artifacts.model.update()

    return artifacts


def set_distance_tardiness_objective(
    artifacts: MIPArtifacts,
    *,
    truck_unit_cost: Optional[float] = None,
    drone_unit_cost: Optional[float] = None,
    tardiness_weight: float | None = 1.0,
    use_piecewise_delay: bool = True,
    pwl_delay_segments: int | None = 10,
    pwl_delay_spacing_power: float = 1.0,
    use_gurobi_pwl: bool = True,
    use_three_piece_delay: bool = False,
    three_piece_bounds: tuple | None = None,
    three_piece_middle_scale: float | None = None,
    cost_lambda: float = 12.0,
    cost_rho: float = 1.0,
    cost_normalized: bool = True,
) -> None:
    """Apply the distance-plus-tardiness objective defined in Eq. (1)."""

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
        if use_piecewise_delay:
            # Determine segments override from YAML config if default used
            segments = pwl_delay_segments
            if segments is None:
                try:
                    from alns_vrpfd.utils.config_loader import ALNSConfig

                    cfg = ALNSConfig()
                    # If the caller passed None, fall back to YAML config; else use caller value
                    if segments is None:
                        segments = int(cfg.piecewise_delay_segments)
                    else:
                        segments = int(segments)
                except Exception:
                    pass

            # Attempt to derive a reasonable upper bound for delay (hours) from instance times
            try:
                max_delay = 0.0
                for cid in data.customers:
                    opt = data.optimal_times.get(cid, 0.0)
                    latest = data.latest_times.get(cid, opt)
                    diff = max(0.0, latest - opt)
                    if diff > max_delay:
                        max_delay = diff
                # Add a modest margin
                # Ensure a sensible upper bound for PWL coverage (in hours)
                if max_delay < 1.0:
                    max_delay = 1.0
                max_delay += 2.0
                # Cap the PWL domain to a conservative upper bound to avoid huge
                # exponential values that exceed solver numerical limits.
                # 6 hours should be more than enough for most routing instances.
                max_delay = min(max_delay, 6.0)
            except Exception:
                max_delay = 3.0

            pwl_points_by_class: Dict[str, tuple[list[float], list[float]]] = {}

            def points_for(customer_id: int) -> tuple[list[float], list[float]]:
                supply_class = data.supply_classes.get(customer_id, "water")
                if supply_class not in pwl_points_by_class:
                    xs, ys = _compute_delay_pwl_points(
                        max_delay,
                        segments,
                        power=pwl_delay_spacing_power,
                        use_three_piece_delay=use_three_piece_delay,
                        three_piece_bounds=three_piece_bounds,
                        three_piece_middle_scale=three_piece_middle_scale,
                        supply_class=supply_class,
                        cost_lambda=cost_lambda,
                        cost_rho=cost_rho,
                        cost_normalized=cost_normalized,
                    )
                    pwl_points_by_class[supply_class] = (xs, ys)
                    _warn_if_delay_pwl_is_coarse(xs, max_delay, segments, use_three_piece_delay)
                return pwl_points_by_class[supply_class]

            # Create cost vars and PWL constraints
            delay_cost_expr = gp.quicksum(
                vars.tardiness_cost_truck[i, k] for i in data.customers for k in data.trucks
            ) + gp.quicksum(
                vars.tardiness_cost_drone[i, d] for i in data.customers for d in data.drones
            )

            # Add PWL constraints for each tardiness var
            if use_gurobi_pwl:
                for i in data.customers:
                    xs, ys = points_for(i)
                    for k in data.trucks:
                        model.addGenConstrPWL(vars.tardiness_truck[i, k], vars.tardiness_cost_truck[i, k], xs, ys,
                                              name=f"pwl_delay_truck[{i},{k}]")
                    for d in data.drones:
                        model.addGenConstrPWL(vars.tardiness_drone[i, d], vars.tardiness_cost_drone[i, d], xs, ys,
                                              name=f"pwl_delay_drone[{i},{d}]")
            else:
                # Fallback: linear interpolation with SOS2 or lambda variables - not implemented
                raise RuntimeError(
                    "Non-Gurobi PWL linearization not yet supported for delay cost")

            objective += float(tardiness_weight) * delay_cost_expr
        else:
            tardiness_expr = gp.quicksum(
                vars.tardiness_truck[i, k] for i in data.customers for k in data.trucks
            ) + gp.quicksum(
                vars.tardiness_drone[i, d] for i in data.customers for d in data.drones
            )
            objective += float(tardiness_weight) * tardiness_expr

    model.setObjective(objective, GRB.MINIMIZE)


def _compute_delay_pwl_points(
    max_delay: float,
    segments: int | None,
    *,
    power: float = 1.0,
    use_three_piece_delay: bool = False,
    three_piece_bounds: tuple | None = None,
    three_piece_middle_scale: float | None = None,
    supply_class: str | None = "water",
    cost_lambda: float = 12.0,
    cost_rho: float = 1.0,
    cost_normalized: bool = True,
) -> tuple[list[float], list[float]]:
    """Return (xs, ys) for delay PWL mapping.

    - If use_three_piece_delay=True, return 4-point xs (0,x1,x2,max_delay) and the
      corresponding ys = f(xs). Optionally scale the middle slope using
      three_piece_middle_scale.
    - Otherwise, return quadratic spacing with 'segments' parts.
    """
    def delay_f(x: float) -> float:
        return deprivation_cost(x, supply_class,
                                cost_lambda=cost_lambda,
                                rho=cost_rho,
                                normalized=cost_normalized)

    if use_three_piece_delay:
        if three_piece_bounds is None:
            # Default bounds are 85 and 105 minutes (convert to hours)
            three_piece_bounds = (85.0 / 60.0, 105.0 / 60.0)
        b1, b2 = three_piece_bounds
        x0 = 0.0
        x1 = float(b1)
        x2 = float(b2)
        x3 = float(max_delay)
        # Ensure ordering
        if x1 <= x0:
            x1 = x0 + 1e-6
        if x2 <= x1:
            x2 = x1 + 1e-6
        if x3 <= x2:
            x3 = x2 + 1e-6
        xs = [x0, x1, x2, x3]
        ys = [delay_f(x) for x in xs]
        if three_piece_middle_scale is not None and three_piece_middle_scale != 1.0:
            if xs[2] != xs[1]:
                orig_slope = (ys[2] - ys[1]) / (xs[2] - xs[1])
                new_slope = orig_slope * float(three_piece_middle_scale)
                ys[2] = ys[1] + new_slope * (xs[2] - xs[1])
    else:
        if segments is None:
            segments = 10
        if segments <= 0:
            xs = [0.0]
        else:
            # Use power-spacing: (i/segments)**power; default power=2.0 (quadratic)
            xs = [max_delay * (float(i) / float(segments)) **
                  float(power) for i in range(segments + 1)]
        ys = [delay_f(x) for x in xs]

    return xs, ys


def _warn_if_delay_pwl_is_coarse(
    xs: Sequence[float],
    max_delay: float,
    segments: int | None,
    use_three_piece_delay: bool,
) -> None:
    """Warn when delay PWL spacing is too coarse near zero."""
    if use_three_piece_delay:
        x1 = xs[1] if len(xs) > 1 else None
        if x1 is not None and x1 > max(1e-6, 0.01 * max_delay):
            logging.getLogger(__name__).warning(
                "Three-piece PWL delay first positive breakpoint is %.6fh (>1%% of max_delay). "
                "Consider reducing your lower bound or choosing a denser spacing to improve small-delay accuracy.",
                x1,
            )
        return

    first_pos_x = next((x for x in xs if x > 0.0), None)
    if first_pos_x is not None and (segments or 0) < 50:
        if first_pos_x > max(1e-6, 0.01 * max_delay):
            logging.getLogger(__name__).warning(
                "PWL delay first positive breakpoint is %.6fh (>1%% of max_delay). "
                "Consider increasing `pwl_delay_segments` or using a denser spacing to improve accuracy.",
                first_pos_x,
            )


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
    supply_classes = {
        cid: instance.customer_manager.supply_class(cid) or "water"
        for cid in customers
    }
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
        (i, j): truck_matrix[index[i]][index[j]] for i in nodes for j in nodes}
    drone_distance = {
        (i, j): drone_matrix[index[i]][index[j]] for i in nodes for j in nodes}
    truck_time = {
        (i, j): truck_time_matrix[index[i]][index[j]] for i in nodes for j in nodes}
    drone_time = {
        (i, j): drone_time_matrix[index[i]][index[j]] for i in nodes for j in nodes}

    battery_capacity = instance.robust_config.drone_battery_capacity
    if energy_budget is None:
        energy_budget = instance.robust_config.energy_uncertainty_budget

    gamma_range = tuple(range(max(0, energy_budget) + 1))

    energy_model = DroneEnergyModel()
    energy_nominal: Dict[Tuple[int, int], float] = {}
    energy_deviation: Dict[Tuple[int, int], float] = {}
    for (i, j) in truck_distance.keys():
        travel_time = drone_time[(i, j)]
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
        supply_classes=supply_classes,
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
    x_truck = model.addVars(data.arcs, data.trucks, vtype=GRB.BINARY, name="x")
    y_drone = model.addVars(data.arcs, data.drones, vtype=GRB.BINARY, name="y")
    z_coupling = model.addVars(
        data.arcs, data.trucks, data.drones, vtype=GRB.BINARY, name="z")
    u = model.addVars(data.nodes, data.trucks, data.drones,
                      vtype=GRB.BINARY, name="u")

    arrival_truck = model.addVars(
        data.nodes, data.trucks, vtype=GRB.CONTINUOUS, lb=0.0, name="a_truck")
    departure_truck = model.addVars(
        data.nodes, data.trucks, vtype=GRB.CONTINUOUS, lb=0.0, name="dep_truck")
    arrival_drone = model.addVars(
        data.nodes, data.drones, vtype=GRB.CONTINUOUS, lb=0.0, name="a_drone")

    load_truck = model.addVars(
        data.nodes, data.trucks, vtype=GRB.CONTINUOUS, lb=0.0, name="omega_truck")
    load_drone_minus = model.addVars(
        data.nodes, data.drones, vtype=GRB.CONTINUOUS, lb=0.0, name="omega_drone_minus")
    load_drone_plus = model.addVars(
        data.nodes, data.drones, vtype=GRB.CONTINUOUS, lb=0.0, name="omega_drone_plus")

    tardiness_truck = model.addVars(
        data.customers, data.trucks, vtype=GRB.CONTINUOUS, lb=0.0, name="tau_truck")
    tardiness_drone = model.addVars(
        data.customers, data.drones, vtype=GRB.CONTINUOUS, lb=0.0, name="tau_drone")
    # 延误成本变量（用于 PWL 映射）
    tardiness_cost_truck = model.addVars(
        data.customers, data.trucks, vtype=GRB.CONTINUOUS, lb=0.0, name="delay_cost_truck")
    tardiness_cost_drone = model.addVars(
        data.customers, data.drones, vtype=GRB.CONTINUOUS, lb=0.0, name="delay_cost_drone")

    deviation = model.addVars(
        data.arcs, vtype=GRB.CONTINUOUS, lb=0.0, name="xi")
    energy_state_gamma = model.addVars(
        data.nodes, data.drones, data.gamma_range, vtype=GRB.CONTINUOUS, lb=0.0, name="e_gamma")

    return VariableContainer(
        x_truck=x_truck,
        y_drone=y_drone,
        z_coupling=z_coupling,
        u=u,
        arrival_truck=arrival_truck,
        departure_truck=departure_truck,
        arrival_drone=arrival_drone,
        load_truck=load_truck,
        load_drone_minus=load_drone_minus,
        load_drone_plus=load_drone_plus,
        tardiness_truck=tardiness_truck,
        tardiness_drone=tardiness_drone,
        deviation=deviation,
        energy_state_gamma=energy_state_gamma,
        tardiness_cost_truck=tardiness_cost_truck,
        tardiness_cost_drone=tardiness_cost_drone,
    )


def _latest_or_depot_deadline(data: ProblemData, node: int, depot_deadline: float) -> float:
    return float(data.latest_times.get(node, depot_deadline))


def _finite_time(value: float, fallback: float) -> float:
    return float(value) if math.isfinite(value) else float(fallback)


def _paper_truck_time_m(data: ProblemData, i: int, j: int, depot_deadline: float) -> float:
    """Paper bound M^T_ij = l_i + t^T_ij."""
    latest_i = _latest_or_depot_deadline(data, i, depot_deadline)
    travel = _finite_time(data.truck_time[(i, j)], depot_deadline)
    return latest_i + travel


def _paper_drone_time_m(data: ProblemData, i: int, j: int, depot_deadline: float) -> float:
    """Paper bound M^D_ij = l_i + t^D_ij."""
    latest_i = _latest_or_depot_deadline(data, i, depot_deadline)
    travel = _finite_time(data.drone_time[(i, j)], depot_deadline)
    return latest_i + travel


def _paper_sync_m(data: ProblemData, i: int, depot_deadline: float) -> float:
    """Paper synchronization bound M^S_i = l_i."""
    return _latest_or_depot_deadline(data, i, depot_deadline)


def _paper_truck_load_lower_m(data: ProblemData, i: int) -> float:
    """Paper bound M^T_i = Q^T - q_i."""
    return max(0.0, data.truck_capacity - data.demand.get(i, 0.0))


def _paper_truck_load_upper_m(data: ProblemData, i: int) -> float:
    """Paper bound \\bar{M}^T_i = Q^T + Q^D + q_i."""
    return data.truck_capacity + data.drone_capacity + data.demand.get(i, 0.0)


def add_core_constraints(
    artifacts: MIPArtifacts,
    *,
    big_m_time: float = 1000.0,
    big_m_load: float = 1000.0,
    big_m_energy: float = 20.0,
    use_piecewise_energy: bool = True,
) -> None:
    """根据 milp_formulation.tex 添加所有核心约束。"""

    model = artifacts.model
    data = artifacts.data
    vars = artifacts.variables

    arc_set = set(data.arcs)
    start_depot, end_depot = data.depots
    depot_deadline = max(data.latest_times.values(), default=0.0)

    # ========== 6.1 Truck Flow Constraints ==========

    # Eq (2): Depot flow conservation
    for k in data.trucks:
        outgoing = gp.quicksum(
            vars.x_truck[start_depot, j, k] for j in data.v_plus if (start_depot, j) in arc_set
        )
        incoming = gp.quicksum(
            vars.x_truck[i, end_depot, k] for i in data.v0 if (i, end_depot) in arc_set
        )
        model.addConstr(outgoing == incoming, name=f"truck_depot_flow[{k}]")

    # Eq (3): Each truck departs at most once
    for k in data.trucks:
        outgoing = gp.quicksum(
            vars.x_truck[start_depot, j, k] for j in data.v_plus if (start_depot, j) in arc_set
        )
        model.addConstr(outgoing <= 1, name=f"truck_single_departure[{k}]")

    # Eq (4): Flow conservation at customers
    for j in data.customers:
        for k in data.trucks:
            inbound = gp.quicksum(vars.x_truck[i, j, k]
                                  for i in data.v0 if (i, j) in arc_set)
            outbound = gp.quicksum(vars.x_truck[j, h, k]
                                   for h in data.v_plus if (j, h) in arc_set)
            model.addConstr(inbound == outbound,
                            name=f"truck_flow_balance[{j},{k}]")

    # Eq (5): Each customer visited by at most one truck
    for j in data.customers:
        visit = gp.quicksum(
            vars.x_truck[i, j, k] for k in data.trucks for i in data.v0 if (i, j) in arc_set
        )
        model.addConstr(visit <= 1, name=f"truck_single_visit[{j}]")

    # ========== 6.2 Drone Flow Constraints ==========

    # Eq (6): Each drone starts at most once from depot
    for d in data.drones:
        start_out = gp.quicksum(
            vars.y_drone[start_depot, j, d] for j in data.v_plus if (start_depot, j) in arc_set
        ) + gp.quicksum(
            vars.z_coupling[start_depot, j, k, d]
            for k in data.trucks for j in data.v_plus if (start_depot, j) in arc_set
        )
        model.addConstr(start_out <= 1, name=f"drone_single_start[{d}]")

    # Eq (7): Drone flow conservation at customers
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

    # Tighten physical launch semantics.
    #
    # The original model used u[i,k,d] mainly as a synchronization marker. That
    # allowed a drone flight path to start at a customer whenever the aggregate
    # y/z flow could be balanced, even if no explicit truck launch was selected
    # at that node. ALNS interprets u as a physical launch/reload event, so we
    # bind each launch to one outgoing flight and one truck visit.
    for d in data.drones:
        for i in data.v0:
            y_in = gp.quicksum(
                vars.y_drone[p, i, d] for p in data.v0 if (p, i) in arc_set
            )
            y_out = gp.quicksum(
                vars.y_drone[i, h, d] for h in data.v_plus if (i, h) in arc_set
            )
            u_sum = gp.quicksum(vars.u[i, k, d] for k in data.trucks)

            model.addConstr(
                u_sum <= 1,
                name=f"drone_single_launch_event[{i},{d}]",
            )
            model.addConstr(
                u_sum <= y_out,
                name=f"launch_requires_outgoing_flight[{i},{d}]",
            )
            model.addConstr(
                y_out <= y_in + u_sum,
                name=f"flight_start_requires_launch_or_continuity[{i},{d}]",
            )
            model.addConstr(
                y_in <= 1,
                name=f"drone_single_air_in[{i},{d}]",
            )
            model.addConstr(
                y_out <= 1,
                name=f"drone_single_air_out[{i},{d}]",
            )

    # ========== 6.3 Truck-Drone Coupling Constraints ==========

    # Eq (8): Coupling requires truck traversal
    for (i, j) in arc_set:
        for k in data.trucks:
            for d in data.drones:
                model.addConstr(
                    vars.z_coupling[i, j, k, d] <= vars.x_truck[i, j, k],
                    name=f"couple_leq_x[{i},{j},{k},{d}]"
                )

    # Eq (9) & (10): Visit bounds with launch and retrieval relaxation
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

        # Eq (9): 访问下界 - 至少有一辆车访问（除非被 sync 或 z_out 松弛）
        # 根据 tex: truck_v + drone_v >= 1 - sync - z_out
        model.addConstr(
            truck_visit + drone_visit >= 1 - sync_sum - z_out_sum,
            name=f"visit_lower[{j}]"
        )

        # Eq (10): 访问上界 - 最多一辆车访问（被 sync 和 z_out 松弛）
        model.addConstr(
            truck_visit + drone_visit <= 1 + sync_sum + z_out_sum,
            name=f"visit_upper[{j}]"
        )

    # ========== 6.4 Synchronization Constraints ==========

    # Eq (11) & (12): Launch/retrieval synchronization logic
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

                # Eq (11): sync_lower
                model.addConstr(
                    truck_io + drone_io >= 3 * vars.u[j, k, d],
                    name=f"sync_lower[{j},{k},{d}]"
                )
                # Eq (12): sync_upper (relaxed by z_out)
                model.addConstr(
                    truck_io + drone_io <= 2 + 2 *
                    vars.u[j, k, d] + z_out_local,
                    name=f"sync_upper[{j},{k},{d}]"
                )

    # Eq (13): Launch requires truck visit
    for k in data.trucks:
        truck_departure = gp.quicksum(
            vars.x_truck[start_depot, j, k]
            for j in data.v_plus
            if (start_depot, j) in arc_set
        )
        for d in data.drones:
            model.addConstr(
                truck_departure >= vars.u[start_depot, k, d],
                name=f"depot_launch_requires_truck[{k},{d}]",
            )

    for i in data.customers:
        for k in data.trucks:
            for d in data.drones:
                truck_in = gp.quicksum(
                    vars.x_truck[p, i, k] for p in data.v0 if (p, i) in arc_set)
                model.addConstr(
                    truck_in >= vars.u[i, k, d], name=f"launch_requires_truck[{i},{k},{d}]")

    # ========== 6.5 Time Window Constraints ==========

    # Eq (14): Truck time continuity
    # Truck departure may be delayed by drone retrieval/synchronization.
    for i in data.nodes:
        for k in data.trucks:
            model.addConstr(
                vars.departure_truck[i, k] >= vars.arrival_truck[i, k],
                name=f"truck_depart_after_arrival[{i},{k}]"
            )

    for (i, j) in arc_set:
        travel_t = data.truck_time[(i, j)]
        m_t = _paper_truck_time_m(data, i, j, depot_deadline)
        for k in data.trucks:
            model.addConstr(
                vars.arrival_truck[j, k] >= vars.departure_truck[i, k] +
                travel_t - m_t * (1 - vars.x_truck[i, j, k]),
                name=f"truck_time_continuity[{i},{j},{k}]"
            )

    # Eq (15): Drone flight time continuity (self-flight)
    for (i, j) in arc_set:
        travel_d = data.drone_time[(i, j)]
        m_d = _paper_drone_time_m(data, i, j, depot_deadline)
        for d in data.drones:
            model.addConstr(
                vars.arrival_drone[j, d] >= vars.arrival_drone[i, d] +
                travel_d - m_d * (1 - vars.y_drone[i, j, d]),
                name=f"drone_time_continuity[{i},{j},{d}]"
            )

    # Eq (16): Drone flight time from truck launch
    for (i, j) in arc_set:
        travel_d = data.drone_time[(i, j)]
        m_d = _paper_drone_time_m(data, i, j, depot_deadline)
        for k in data.trucks:
            for d in data.drones:
                model.addConstr(
                    vars.arrival_drone[j, d] >= vars.arrival_truck[i, k] +
                    travel_d - m_d * (1 - vars.y_drone[i, j, d]),
                    name=f"drone_time_from_truck[{i},{j},{k},{d}]"
                )

    # Eq (17): Launch synchronization
    for i in data.nodes:
        m_s = _paper_sync_m(data, i, depot_deadline)
        for k in data.trucks:
            for d in data.drones:
                model.addConstr(
                    vars.arrival_drone[i, d] >= vars.arrival_truck[i,
                                                                   k] - m_s * (1 - vars.u[i, k, d]),
                    name=f"launch_sync[{i},{k},{d}]"
                )
                model.addConstr(
                    vars.departure_truck[i, k] >= vars.arrival_drone[i, d]
                    - m_s * (1 - vars.u[i, k, d]),
                    name=f"truck_waits_launch_sync[{i},{k},{d}]"
                )

    # Eq (18): Retrieval synchronization
    for (i, j) in arc_set:
        m_t = _paper_truck_time_m(data, i, j, depot_deadline)
        m_s = _paper_sync_m(data, i, depot_deadline)
        for k in data.trucks:
            for d in data.drones:
                model.addConstr(
                    vars.arrival_truck[j, k] >= vars.arrival_drone[i, d] + data.truck_time[(
                        i, j)] - m_t * (1 - vars.z_coupling[i, j, k, d]),
                    name=f"retrieve_sync[{i},{j},{k},{d}]"
                )
                model.addConstr(
                    vars.departure_truck[i, k] >= vars.arrival_drone[i, d]
                    - m_s * (1 - vars.z_coupling[i, j, k, d]),
                    name=f"truck_waits_retrieve[{i},{j},{k},{d}]"
                )

    # Eq (19) & (20): Carry synchronization
    for (i, j) in arc_set:
        m_t = _paper_truck_time_m(data, i, j, depot_deadline)
        for k in data.trucks:
            for d in data.drones:
                model.addConstr(
                    vars.arrival_drone[j, d] >= vars.arrival_truck[j, k] -
                    m_t * (1 - vars.z_coupling[i, j, k, d]),
                    name=f"carry_sync_lb[{i},{j},{k},{d}]"
                )
                model.addConstr(
                    vars.arrival_drone[j, d] <= vars.arrival_truck[j, k] +
                    m_t * (1 - vars.z_coupling[i, j, k, d]),
                    name=f"carry_sync_ub[{i},{j},{k},{d}]"
                )

    # Eq (21) & (22): Time window deadline
    for i in data.customers:
        latest = data.latest_times.get(i, depot_deadline)
        optimal = data.optimal_times.get(i, 0.0)
        for k in data.trucks:
            model.addConstr(
                vars.arrival_truck[i, k] <= latest, name=f"truck_deadline[{i},{k}]")
        for d in data.drones:
            model.addConstr(
                vars.arrival_drone[i, d] <= latest, name=f"drone_deadline[{i},{d}]")

    # Eq (23) & (24): Tardiness definition
    for i in data.customers:
        optimal = data.optimal_times.get(i, 0.0)
        for k in data.trucks:
            model.addConstr(
                vars.tardiness_truck[i,
                                     k] >= vars.arrival_truck[i, k] - optimal,
                name=f"truck_tardiness[{i},{k}]"
            )
        for d in data.drones:
            y_in = gp.quicksum(vars.y_drone[p, i, d] for p in data.v0 if (p, i) in arc_set)
            u_sum = gp.quicksum(vars.u[i, k, d] for k in data.trucks)
            m_s = _paper_sync_m(data, i, depot_deadline)
            model.addConstr(
                vars.tardiness_drone[i, d] >= vars.arrival_drone[i, d] -
                optimal - m_s * (1 - y_in + u_sum),
                name=f"drone_tardiness[{i},{d}]"
            )

    # ========== 6.6 Truck Load Constraints ==========

    # Eq (25) & (26): Truck load update
    for (i, j) in arc_set:
        if i in data.customers and j in data.v_plus:
            demand_i = data.demand.get(i, 0.0)
            m_lower = _paper_truck_load_lower_m(data, i)
            m_upper = _paper_truck_load_upper_m(data, i)
            for k in data.trucks:
                drone_load_sum = gp.quicksum(
                    vars.load_drone_plus[i, d] for d in data.drones)
                model.addConstr(
                    vars.load_truck[j, k] >= vars.load_truck[i, k] - demand_i -
                    drone_load_sum - m_lower * (1 - vars.x_truck[i, j, k]),
                    name=f"truck_load_lower[{i},{j},{k}]"
                )
                model.addConstr(
                    vars.load_truck[j, k] <= vars.load_truck[i, k] - demand_i -
                    drone_load_sum + m_upper * (1 - vars.x_truck[i, j, k]),
                    name=f"truck_load_upper[{i},{j},{k}]"
                )

    # Eq (27): Initial load constraint
    for j in data.v_plus:
        for k in data.trucks:
            if (start_depot, j) in arc_set:
                model.addConstr(
                    vars.load_truck[j, k] <= vars.load_truck[start_depot, k] +
                    data.truck_capacity * (1 - vars.x_truck[start_depot, j, k]),
                    name=f"initial_truck_load[{j},{k}]"
                )

    # Eq (28): Truck capacity
    for i in data.nodes:
        for k in data.trucks:
            model.addConstr(
                vars.load_truck[i, k] <= data.truck_capacity, name=f"truck_capacity[{i},{k}]")

    # ========== 6.7 Drone Load Constraints ==========

    # Eq (29) & (30): Drone load arc conservation
    m_drone = data.drone_capacity
    for (i, j) in arc_set:
        for d in data.drones:
            model.addConstr(
                vars.load_drone_minus[j, d] >= vars.load_drone_plus[i,
                                                                    d] - m_drone * (1 - vars.y_drone[i, j, d]),
                name=f"drone_load_lower[{i},{j},{d}]"
            )
            model.addConstr(
                vars.load_drone_minus[j, d] <= vars.load_drone_plus[i,
                                                                    d] + m_drone * (1 - vars.y_drone[i, j, d]),
                name=f"drone_load_upper[{i},{j},{d}]"
            )

    # Tight droneLoad: |ω⁺_i - ω⁻_i + q_i| ≤ M^D (1 - Y_in_i + U_i)
    for i in data.nodes:
        demand_i = data.demand.get(i, 0.0)
        for d in data.drones:
            y_in = gp.quicksum(vars.y_drone[p, i, d] for p in data.v0 if (p, i) in arc_set)
            u_sum = gp.quicksum(vars.u[i, k, d] for k in data.trucks)
            rhs = 1 - y_in + u_sum
            model.addConstr(
                vars.load_drone_plus[i, d] - vars.load_drone_minus[i, d] + demand_i <= m_drone * rhs,
                name=f"drone_load_node_ub[{i},{d}]"
            )
            model.addConstr(
                -(vars.load_drone_plus[i, d] - vars.load_drone_minus[i, d] + demand_i) <= m_drone * rhs,
                name=f"drone_load_node_lb[{i},{d}]"
            )

    # droneEmpty: ω⁺_j ≤ Q^D * Y_out_j (forces zero post-service load at retrieval nodes)
    for j in data.v_plus:
        for d in data.drones:
            y_out = gp.quicksum(vars.y_drone[j, h, d] for h in data.v_plus if (j, h) in arc_set)
            model.addConstr(
                vars.load_drone_plus[j, d] <= data.drone_capacity * y_out,
                name=f"drone_empty[{j},{d}]"
            )

    # Eq (36): Drone capacity - 每个节点的负载不能超过无人机容量
    # 这确保每个独立任务的负载都在容量范围内
    for i in data.nodes:
        for d in data.drones:
            model.addConstr(
                vars.load_drone_minus[i, d] <= data.drone_capacity, name=f"drone_capacity[{i},{d}]")
            # 非负约束
            model.addConstr(
                vars.load_drone_minus[i, d] >= 0.0, name=f"drone_load_nonneg[{i},{d}]")

    # Eq (37): 仓库初始负载设为0
    # 无人机从仓库发射时，负载将在发射约束中设置
    for d in data.drones:
        model.addConstr(
            vars.load_drone_plus[start_depot, d] == 0.0,
            name=f"drone_initial_load[{d}]"
        )

    # 注意：tight droneLoad + droneEmpty 约束已替代旧的 v_served 方法。
    # droneEmpty 从回收节点向后传播，通过 arc 约束和 tight 客户节点约束，
    # 迫使发射点负载等于该 sortie 所有客户需求之和 —— 不再需要显式的 v_served。

    # ========== 7. Energy Constraints ==========

    gamma_max = data.gamma_range[-1]
    gamma_min = data.gamma_range[0]

    if not use_piecewise_energy:
        # Eq (39) & (40): Energy flow (nominal and deviation)
        # 修改版本：当从发射点出发时，能耗从0开始（每个任务独立计算）
        # 这与 ALNS 的实现一致：每个 DroneTask 的能耗是独立检查的
        for (i, j) in arc_set:
            energy_nom = data.energy_nominal[(i, j)]
            energy_dev = data.energy_deviation[(i, j)]
            for d in data.drones:
                u_sum = gp.quicksum(vars.u[i, k, d] for k in data.trucks)
                for gamma in data.gamma_range:
                    # 标准能耗流约束（非发射点）
                    # 当 u_sum=1 时，这个约束被松弛
                    model.addConstr(
                        vars.energy_state_gamma[j, d, gamma] >= vars.energy_state_gamma[i, d,
                                                                                        gamma] + energy_nom - big_m_energy * (1 - vars.y_drone[i, j, d]) - big_m_energy * u_sum,
                        name=f"energy_flow_nominal[{i},{j},{d},{gamma}]"
                    )
                    # 发射点能耗流约束（能耗从0开始）
                    # 只有当 u_sum=1 且 y[i,j]=1 时约束才有效
                    model.addConstr(
                        vars.energy_state_gamma[j, d, gamma] >= energy_nom - big_m_energy * (
                            1 - vars.y_drone[i, j, d]) - big_m_energy * (1 - u_sum),
                        name=f"energy_flow_launch[{i},{j},{d},{gamma}]"
                    )
                    if gamma > gamma_min:
                        model.addConstr(
                            vars.energy_state_gamma[j, d, gamma] >= vars.energy_state_gamma[i, d, gamma -
                                                                                            1] + energy_nom + energy_dev - big_m_energy * (1 - vars.y_drone[i, j, d]) - big_m_energy * u_sum,
                            name=f"energy_flow_deviation[{i},{j},{d},{gamma}]"
                        )
                        model.addConstr(
                            vars.energy_state_gamma[j, d, gamma] >= energy_nom + energy_dev - big_m_energy * (
                                1 - vars.y_drone[i, j, d]) - big_m_energy * (1 - u_sum),
                            name=f"energy_flow_launch_dev[{i},{j},{d},{gamma}]"
                        )
        # # 原始版本（已注释）：不处理发射点重置，能耗跨任务累积
        # for (i, j) in arc_set:
        #     energy_nom = data.energy_nominal[(i, j)]
        #     energy_dev = data.energy_deviation[(i, j)]
        #     for d in data.drones:
        #         for gamma in data.gamma_range:
        #             model.addConstr(
        #                 vars.energy_state_gamma[j, d, gamma] >= vars.energy_state_gamma[i, d,
        #                                                                                 gamma] + energy_nom - big_m_energy * (1 - vars.y_drone[i, j, d]),
        #                 name=f"energy_flow_nominal[{i},{j},{d},{gamma}]"
        #             )
        #             if gamma > gamma_min:
        #                 model.addConstr(
        #                     vars.energy_state_gamma[j, d, gamma] >= vars.energy_state_gamma[i, d, gamma -
        #                                                                                     1] + energy_nom + energy_dev - big_m_energy * (1 - vars.y_drone[i, j, d]),
        #                     name=f"energy_flow_deviation[{i},{j},{d},{gamma}]"
        #                 )

    # Eq (41): Battery capacity constraint
    # 检查每个节点的能耗是否超过电池容量
    for j in data.v_plus:
        for d in data.drones:
            model.addConstr(
                vars.energy_state_gamma[j, d, gamma_max] <= data.battery_capacity * gp.quicksum(
                    vars.y_drone[i, j, d] for i in data.v0 if (i, j) in arc_set
                ),
                name=f"energy_capacity[{j},{d}]"
            )

    # Eq (42): 发射点能耗初始化约束
    # 当无人机在节点 i 发射时 (u[i,k,d]=1)，该节点发出的飞行弧的能耗
    # 应该只包含该弧本身的能耗，而不包含之前任务的累积能耗。
    # 这通过在发射节点设置 energy_state_gamma[i,d,gamma] = 0 实现，
    # 但需要用 Big-M 方法处理发射和非发射的情况。
    #
    # 注意：由于同一个节点可能同时是上一任务的回收点和下一任务的发射点，
    # 我们不能简单地把 energy_state_gamma[i] 设为 0。
    #
    # 更好的方法是：修改能耗流约束，当从发射点出发时，能耗从 0 开始累加。
    # 即：e[j] >= e[i] + energy_use - M*(1-y[i,j]) 变为
    #     e[j] >= 0 + energy_use - M*(1-y[i,j]) 当 u[i,k,d]=1 时
    #
    # 这个逻辑在 piecewise_energy.py 中的能耗流约束里实现。
    # 这里我们只需要设置仓库（depot）的初始能耗为 0。
    for d in data.drones:
        for gamma in data.gamma_range:
            model.addConstr(
                vars.energy_state_gamma[start_depot, d, gamma] == 0.0,
                name=f"energy_initial_depot[{d},{gamma}]"
            )


__all__ = [
    "ProblemData",
    "VariableContainer",
    "MIPArtifacts",
    "build_mip_model",
    "set_distance_tardiness_objective",
    "add_core_constraints",
]
