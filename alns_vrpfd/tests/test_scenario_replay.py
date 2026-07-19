"""Tests for scenario replay robustness evaluation."""

from __future__ import annotations

import math

from alns_vrpfd.evaluation import (
    GammaSolutionInput,
    ScenarioDistributionConfig,
    ScenarioReplayConfig,
    run_scenario_replay,
)
from alns_vrpfd.instance import InstanceManager
from alns_vrpfd.model.route import DroneTask, TruckRoute
from alns_vrpfd.model.solution import Solution


class _UnitEnergyModel:
    """Simple deterministic energy model for replay tests."""

    def energy_kwh(
        self,
        payload_weight_kg: float,
        travel_time_hours: float,
    ) -> float:
        _ = payload_weight_kg
        _ = travel_time_hours
        return 1.0


def _build_instance(single_task: bool = True) -> InstanceManager:
    instance = InstanceManager()
    instance.configure_depots(start=0, end=0)

    instance.register_customer(customer_id=1, demand=1.0)
    instance.register_customer(customer_id=2, demand=1.0)
    if not single_task:
        instance.register_customer(customer_id=3, demand=1.0)

    instance.register_vehicle_type(
        "truck",
        number=1,
        capacity=50.0,
        endurance=480.0,
        speed=1.0,
        unit_cost=1.0,
    )
    instance.register_vehicle_type(
        "drone",
        number=1,
        capacity=10.0,
        endurance=120.0,
        speed=1.0,
        unit_cost=1.0,
    )

    # truck route arcs
    instance.add_distance("truck", 0, 1, 1.0)
    instance.add_distance("truck", 1, 0, 1.0)

    # drone arcs used by test tasks
    instance.add_distance("drone", 1, 2, 1.0)
    instance.add_distance("drone", 2, 0, 1.0)
    if not single_task:
        instance.add_distance("drone", 2, 3, 1.0)
        instance.add_distance("drone", 3, 0, 1.0)

    return instance


def _build_solution(single_task: bool = True) -> Solution:
    truck_route = TruckRoute(route_id=0, nodes=[0, 1, 0], capacity=50.0)
    if single_task:
        drone_task = DroneTask(
            task_id=11,
            drone_id=0,
            launch_truck=0,
            launch_node=1,
            customers=[2],
            land_truck=0,
            retrieve_node=0,
            payloads=[1.0, 0.0],
        )
    else:
        drone_task = DroneTask(
            task_id=12,
            drone_id=0,
            launch_truck=0,
            launch_node=1,
            customers=[2, 3],
            land_truck=0,
            retrieve_node=0,
            payloads=[1.0, 1.0, 0.0],
        )
    return Solution(truck_routes=[truck_route], drone_tasks=[drone_task])


def test_scenario_replay_crn_same_solution_same_scenario_outputs():
    instance = _build_instance(single_task=True)
    solution = _build_solution(single_task=True)

    result = run_scenario_replay(
        instance=instance,
        gamma_solutions=[
            GammaSolutionInput(gamma=0, solution=solution),
            GammaSolutionInput(gamma=6, solution=solution.clone()),
        ],
        distributions=[
            ScenarioDistributionConfig(
                name="det",
                kind="DETERMINISTIC",
                deterministic_multiplier=1.1,
            )
        ],
        config=ScenarioReplayConfig(
            scenario_count=5,
            include_base_cost=False,
            safety_margin_kwh=0.0,
            energy_unit_cost=1.0,
        ),
        energy_model=_UnitEnergyModel(),  # type: ignore[arg-type]
    )

    rec_g0 = {(r.scenario_id): r for r in result.records if r.gamma == "0"}
    rec_g6 = {(r.scenario_id): r for r in result.records if r.gamma == "6"}
    assert set(rec_g0.keys()) == set(rec_g6.keys())

    for sid in rec_g0:
        left = rec_g0[sid]
        right = rec_g6[sid]
        assert math.isclose(left.cost, right.cost, rel_tol=1e-9)
        assert left.unserved == right.unserved
        assert left.no_takeoff == right.no_takeoff
        assert left.abort_return == right.abort_return


def test_scenario_replay_counts_no_takeoff_and_unserved():
    instance = _build_instance(single_task=True)
    instance.configure_robustness(drone_battery_capacity=1.5)
    solution = _build_solution(single_task=True)

    result = run_scenario_replay(
        instance=instance,
        gamma_solutions=[GammaSolutionInput(gamma=0, solution=solution)],
        distributions=[
            ScenarioDistributionConfig(name="det", kind="DETERMINISTIC")
        ],
        config=ScenarioReplayConfig(
            scenario_count=4,
            include_base_cost=False,
            safety_margin_kwh=0.1,
        ),
        energy_model=_UnitEnergyModel(),  # type: ignore[arg-type]
    )

    summary = result.summaries[0]
    assert summary.scenario_count == 4
    assert math.isclose(summary.avg_no_takeoff, 1.0, rel_tol=1e-9)
    assert math.isclose(summary.avg_abort_return, 0.0, rel_tol=1e-9)
    assert math.isclose(summary.avg_unserved, 1.0, rel_tol=1e-9)
    assert math.isclose(summary.p0_all_served, 0.0, rel_tol=1e-9)


def test_scenario_replay_counts_abort_return():
    instance = _build_instance(single_task=False)
    instance.configure_robustness(drone_battery_capacity=3.1)
    solution = _build_solution(single_task=False)

    result = run_scenario_replay(
        instance=instance,
        gamma_solutions=[GammaSolutionInput(gamma=12, solution=solution)],
        distributions=[
            ScenarioDistributionConfig(
                name="high",
                kind="DETERMINISTIC",
                deterministic_multiplier=1.5,
            )
        ],
        config=ScenarioReplayConfig(
            scenario_count=3,
            include_base_cost=False,
            safety_margin_kwh=0.0,
        ),
        energy_model=_UnitEnergyModel(),  # type: ignore[arg-type]
    )

    summary = result.summaries[0]
    assert summary.scenario_count == 3
    assert math.isclose(summary.avg_no_takeoff, 0.0, rel_tol=1e-9)
    assert math.isclose(summary.avg_abort_return, 1.0, rel_tol=1e-9)
    assert math.isclose(summary.avg_unserved, 1.0, rel_tol=1e-9)
    assert math.isclose(summary.p0_all_served, 0.0, rel_tol=1e-9)
