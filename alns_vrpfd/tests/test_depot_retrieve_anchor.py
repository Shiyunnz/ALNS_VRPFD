"""Regression tests for start/end depot anchor semantics."""

from __future__ import annotations

from alns_vrpfd.core.operators.matheuristic_lns import MatheuristicLNSRepair
from alns_vrpfd.core.operators.truck_drone_rechain import TruckDroneRechainRepair
from alns_vrpfd.evaluation import Evaluator
from alns_vrpfd.instance import InstanceManager
from alns_vrpfd.model import DroneTask, Solution, TruckRoute


def _build_two_depot_instance(*, omit_drone_return_to_start: bool = False) -> InstanceManager:
    instance = InstanceManager()
    instance.configure_depots(start=0, end=3)
    for customer_id in (1, 2):
        instance.register_customer(customer_id=customer_id, demand=1.0)
        instance.customer_manager.assign_time_window(
            customer_id, optimal=0.0, latest=100.0)

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
                    if (omit_drone_return_to_start
                            and mode == "drone"
                            and origin == 2
                            and destination == 0):
                        continue
                    instance.add_distance(mode, origin, destination, 1.0)

    instance.configure_robustness(
        drone_battery_capacity=100.0,
        energy_uncertainty_budget=0,
        energy_deviation_rate=0.0,
        same_truck_retrieval=False,
    )
    return instance


def test_two_depot_solution_cannot_retrieve_at_start_depot():
    instance = _build_two_depot_instance(omit_drone_return_to_start=True)
    solution = Solution(
        truck_routes=[
            TruckRoute(route_id=0, nodes=[0, 1, 3], capacity=50.0),
        ],
        drone_tasks=[
            DroneTask(
                task_id=0,
                drone_id=0,
                launch_truck=0,
                launch_node=1,
                customers=[2],
                land_truck=None,
                retrieve_node=0,
                payloads=[1.0, 0.0],
            ),
        ],
    )

    result = Evaluator(instance).evaluate_solution(solution)

    assert not result.feasible


def test_reconstruction_operators_exclude_start_depot_from_retrieve_anchors():
    instance = _build_two_depot_instance()
    evaluator = Evaluator(instance)
    anchors = [0, 1, 3]

    mlns = MatheuristicLNSRepair(instance, evaluator)
    launch_anchors, retrieve_anchors = mlns._filter_anchors_by_distance(
        [2], anchors)

    assert 0 in launch_anchors
    assert 0 not in retrieve_anchors
    assert 3 in retrieve_anchors

    rechain = TruckDroneRechainRepair(instance, evaluator)
    launch_anchors, retrieve_anchors = rechain._filter_anchors_by_distance(
        [2], anchors)

    assert 0 in launch_anchors
    assert 0 not in retrieve_anchors
    assert 3 in retrieve_anchors
