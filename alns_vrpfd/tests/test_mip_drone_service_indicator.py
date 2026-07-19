"""Regression tests for MILP drone service/load semantics."""

from __future__ import annotations

import pytest
from types import SimpleNamespace

gp = pytest.importorskip("gurobipy")
GRB = pytest.importorskip("gurobipy").GRB

from alns_vrpfd.instance import InstanceManager
from alns_vrpfd.evaluation.run_record import reconstruct_solution_from_mip
from alns_vrpfd.mip.builder import build_mip_model


def _build_sync_path_instance() -> InstanceManager:
    instance = InstanceManager()
    instance.configure_depots(start=0, end=5)
    for customer_id, demand in ((1, 6.0), (2, 0.0), (4, 0.0)):
        instance.register_customer(customer_id=customer_id, demand=demand)
        instance.customer_manager.assign_time_window(customer_id, optimal=0.0, latest=100.0)

    instance.register_vehicle_type(
        "truck",
        number=1,
        capacity=50.0,
        endurance=100.0,
        speed=10.0,
        unit_cost=1.0,
    )
    instance.register_vehicle_type(
        "drone",
        number=1,
        capacity=30.0,
        endurance=100.0,
        speed=10.0,
        unit_cost=1.0,
    )
    for mode in ("truck", "drone"):
        for origin in instance.all_node_ids():
            for destination in instance.all_node_ids():
                if origin != destination:
                    instance.add_distance(mode, origin, destination, 1.0)

    instance.configure_robustness(
        drone_battery_capacity=100.0,
        energy_uncertainty_budget=0,
        energy_deviation_rate=0.0,
        same_truck_retrieval=False,
    )
    return instance


def _build_waiting_instance() -> InstanceManager:
    instance = InstanceManager()
    instance.configure_depots(start=0, end=3)
    instance.register_customer(customer_id=1, demand=0.0)
    instance.customer_manager.assign_time_window(1, optimal=0.0, latest=100.0)

    instance.register_vehicle_type(
        "truck",
        number=1,
        capacity=50.0,
        endurance=100.0,
        speed=1.0,
        unit_cost=1.0,
    )
    instance.register_vehicle_type(
        "drone",
        number=1,
        capacity=30.0,
        endurance=100.0,
        speed=1.0,
        unit_cost=1.0,
    )
    nodes = (0, 1, 3)
    for origin in nodes:
        for destination in nodes:
            if origin == destination:
                continue
            truck_distance = 1.0
            drone_distance = 5.0 if (origin, destination) == (0, 1) else 1.0
            instance.add_distance("truck", origin, destination, truck_distance)
            instance.add_distance("drone", origin, destination, drone_distance)

    instance.configure_robustness(
        drone_battery_capacity=100.0,
        energy_uncertainty_budget=0,
        energy_deviation_rate=0.0,
        same_truck_retrieval=False,
    )
    return instance


def _fix_binary_tupledict(model, variables, active_keys: set[tuple]) -> None:
    for key, var in variables.items():
        model.addConstr(var == (1.0 if key in active_keys else 0.0))


class _FakeVar:
    def __init__(self, value: float) -> None:
        self.X = value


def _fake_tupledict(keys: list[tuple], active_keys: set[tuple]):
    return {key: _FakeVar(1.0 if key in active_keys else 0.0) for key in keys}


def _row_coeff(model, constr_name: str, var) -> float:
    constr = model.getConstrByName(constr_name)
    assert constr is not None
    row = model.getRow(constr)
    for idx in range(row.size()):
        if row.getVar(idx).sameAs(var):
            return row.getCoeff(idx)
    return 0.0


def test_core_big_m_coefficients_follow_paper_bounds():
    instance = _build_waiting_instance()
    artifacts = build_mip_model(
        instance,
        solver_parameters={"OutputFlag": 0},
        use_piecewise_energy=False,
        use_piecewise_delay=False,
        energy_budget=0,
    )
    model = artifacts.model
    vars = artifacts.variables
    model.update()

    assert _row_coeff(model, "truck_time_continuity[0,1,0]", vars.x_truck[0, 1, 0]) == pytest.approx(-101.0)
    assert _row_coeff(model, "drone_time_continuity[0,1,0]", vars.y_drone[0, 1, 0]) == pytest.approx(-105.0)
    assert _row_coeff(model, "launch_sync[1,0,0]", vars.u[1, 0, 0]) == pytest.approx(-100.0)
    assert _row_coeff(model, "truck_load_lower[1,3,0]", vars.x_truck[1, 3, 0]) == pytest.approx(-50.0)
    assert _row_coeff(model, "truck_load_upper[1,3,0]", vars.x_truck[1, 3, 0]) == pytest.approx(80.0)
    assert _row_coeff(model, "drone_load_lower[0,1,0]", vars.y_drone[0, 1, 0]) == pytest.approx(-30.0)
    assert _row_coeff(model, "drone_load_upper[0,1,0]", vars.y_drone[0, 1, 0]) == pytest.approx(30.0)
    assert _row_coeff(model, "drone_load_node_ub[1,0]", vars.y_drone[0, 1, 0]) == pytest.approx(30.0)
    assert _row_coeff(model, "drone_load_node_ub[1,0]", vars.u[1, 0, 0]) == pytest.approx(-30.0)


@pytest.mark.parametrize("use_piecewise_energy", [False, True])
def test_drone_customer_with_y_in_and_z_out_is_served_when_not_launch_node(use_piecewise_energy):
    instance = _build_sync_path_instance()
    artifacts = build_mip_model(
        instance,
        solver_parameters={"OutputFlag": 0},
        use_piecewise_energy=use_piecewise_energy,
        use_piecewise_delay=False,
        energy_budget=0,
        num_segments=3,
    )
    model = artifacts.model
    vars = artifacts.variables

    _fix_binary_tupledict(
        model,
        vars.x_truck,
        {(0, 4, 0), (4, 1, 0), (1, 2, 0), (2, 5, 0)},
    )
    _fix_binary_tupledict(model, vars.y_drone, {(4, 1, 0)})
    _fix_binary_tupledict(
        model,
        vars.z_coupling,
        {(0, 4, 0, 0), (1, 2, 0, 0), (2, 5, 0, 0)},
    )
    _fix_binary_tupledict(model, vars.u, {(4, 0, 0)})

    model.setObjective(0.0, GRB.MINIMIZE)
    model.optimize()

    assert model.Status == GRB.OPTIMAL
    assert vars.load_drone_plus[1, 0].X == pytest.approx(
        vars.load_drone_minus[1, 0].X - instance.customer_manager.demands()[1],
        abs=1e-6,
    )


def test_truck_departure_waits_for_drone_sync_before_next_arc():
    instance = _build_waiting_instance()
    artifacts = build_mip_model(
        instance,
        solver_parameters={"OutputFlag": 0},
        use_piecewise_energy=False,
        use_piecewise_delay=False,
        energy_budget=0,
        big_m_time=1000.0,
    )
    model = artifacts.model
    vars = artifacts.variables

    _fix_binary_tupledict(model, vars.x_truck, {(0, 1, 0), (1, 3, 0)})
    _fix_binary_tupledict(model, vars.y_drone, {(0, 1, 0), (1, 3, 0)})
    _fix_binary_tupledict(model, vars.z_coupling, set())
    _fix_binary_tupledict(model, vars.u, {(0, 0, 0), (1, 0, 0)})

    model.setObjective(vars.arrival_truck[3, 0], GRB.MINIMIZE)
    model.optimize()

    assert model.Status == GRB.OPTIMAL
    assert vars.arrival_drone[1, 0].X == pytest.approx(5.0)
    assert vars.arrival_truck[3, 0].X >= 6.0 - 1e-6


def test_carried_drone_cannot_start_flight_without_launch_event():
    instance = _build_waiting_instance()
    artifacts = build_mip_model(
        instance,
        solver_parameters={"OutputFlag": 0},
        use_piecewise_energy=False,
        use_piecewise_delay=False,
        energy_budget=0,
        big_m_time=1000.0,
    )
    model = artifacts.model
    vars = artifacts.variables

    _fix_binary_tupledict(model, vars.x_truck, {(0, 1, 0), (1, 3, 0)})
    _fix_binary_tupledict(model, vars.y_drone, {(1, 3, 0)})
    _fix_binary_tupledict(model, vars.z_coupling, {(0, 1, 0, 0)})
    _fix_binary_tupledict(model, vars.u, set())

    model.setObjective(0.0, GRB.MINIMIZE)
    model.optimize()

    assert model.Status == GRB.INFEASIBLE


def test_drone_cannot_take_direct_empty_depot_to_depot_flight():
    instance = _build_waiting_instance()
    artifacts = build_mip_model(
        instance,
        solver_parameters={"OutputFlag": 0},
        use_piecewise_energy=False,
        use_piecewise_delay=False,
        energy_budget=0,
    )
    model = artifacts.model
    vars = artifacts.variables

    _fix_binary_tupledict(model, vars.x_truck, {(0, 1, 0), (1, 3, 0)})
    _fix_binary_tupledict(model, vars.y_drone, {(0, 3, 0)})
    _fix_binary_tupledict(model, vars.z_coupling, set())
    _fix_binary_tupledict(model, vars.u, {(0, 0, 0)})

    model.setObjective(0.0, GRB.MINIMIZE)
    model.optimize()

    assert model.Status == GRB.INFEASIBLE


def test_reconstruct_solution_skips_empty_drone_segments():
    arcs = [(0, 1), (1, 2), (0, 2)]
    data = SimpleNamespace(
        arcs=arcs,
        nodes=(0, 1, 2),
        trucks=(0,),
        drones=(0,),
        truck_capacity=50.0,
    )
    variables = SimpleNamespace(
        x_truck=_fake_tupledict(
            [(i, j, 0) for i, j in arcs],
            {(0, 1, 0), (1, 2, 0)},
        ),
        y_drone=_fake_tupledict(
            [(i, j, 0) for i, j in arcs],
            {(0, 2, 0)},
        ),
        u=_fake_tupledict(
            [(0, 0, 0), (1, 0, 0), (2, 0, 0)],
            {(0, 0, 0)},
        ),
        v_served=_fake_tupledict([(1, 0)], set()),
        arrival_drone={
            (0, 0): _FakeVar(0.0),
            (1, 0): _FakeVar(0.0),
            (2, 0): _FakeVar(1.0),
        },
        load_drone_minus={
            (0, 0): _FakeVar(0.0),
            (1, 0): _FakeVar(0.0),
            (2, 0): _FakeVar(0.0),
        },
    )
    artifacts = SimpleNamespace(
        data=data,
        variables=variables,
        model=SimpleNamespace(SolCount=1),
    )

    solution = reconstruct_solution_from_mip(artifacts)

    assert solution is not None
    assert solution.drone_tasks == []
