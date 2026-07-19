"""Regression tests for route identity and canonical feasibility checks."""

from __future__ import annotations

from alns_vrpfd.evaluation import Evaluator
from alns_vrpfd.instance import InstanceManager
from alns_vrpfd.model import DroneTask, Solution, TruckRoute
from heuristics.ga.ga import FeasibilityRepair, GAConfig, GeneticAlgorithm, Individual


def _build_instance() -> InstanceManager:
    instance = InstanceManager()
    instance.configure_depots(start=0, end=4)
    for customer in (1, 2, 3):
        instance.register_customer(customer_id=customer, demand=4.0)
    instance.register_vehicle_type(
        "truck", number=2, capacity=10.0, endurance=100.0, speed=40.0, unit_cost=1.0
    )
    instance.register_vehicle_type(
        "drone", number=1, capacity=10.0, endurance=100.0, speed=20.0, unit_cost=1.0
    )
    for mode in ("truck", "drone"):
        for start in range(5):
            for end in range(5):
                if start != end:
                    instance.add_distance(mode, start, end, 1.0)
    instance.configure_robustness(
        drone_battery_capacity=100.0,
        energy_uncertainty_budget=0,
        energy_deviation_rate=0.1,
        same_truck_retrieval=False,
    )
    return instance


def test_evaluator_rejects_duplicate_truck_route_ids():
    instance = _build_instance()
    evaluator = Evaluator(instance, rendezvous_tolerance=float("inf"))
    solution = Solution(
        truck_routes=[
            TruckRoute(route_id=1, nodes=[0, 1, 4], capacity=10.0),
            TruckRoute(route_id=1, nodes=[0, 2, 3, 4], capacity=10.0),
        ]
    )

    result = evaluator.evaluate_solution(solution)

    assert result.feasible is False


def test_evaluator_rejects_truck_capacity_violation():
    instance = _build_instance()
    evaluator = Evaluator(instance, rendezvous_tolerance=float("inf"))
    solution = Solution(
        truck_routes=[
            TruckRoute(route_id=0, nodes=[0, 1, 2, 3, 4], capacity=10.0),
        ]
    )

    result = evaluator.evaluate_solution(solution)

    assert result.feasible is False


def test_evaluator_rejects_duplicate_drone_task_ids():
    instance = _build_instance()
    evaluator = Evaluator(instance, rendezvous_tolerance=float("inf"))
    solution = Solution(
        truck_routes=[
            TruckRoute(route_id=0, nodes=[0, 1, 2, 3, 4], capacity=20.0),
        ],
        drone_tasks=[
            DroneTask(
                task_id=0,
                drone_id=0,
                launch_truck=0,
                launch_node=1,
                customers=[],
                land_truck=0,
                retrieve_node=2,
                payloads=[0.0],
            ),
            DroneTask(
                task_id=0,
                drone_id=0,
                launch_truck=0,
                launch_node=2,
                customers=[],
                land_truck=0,
                retrieve_node=3,
                payloads=[0.0],
            ),
        ],
    )

    result = evaluator.evaluate_solution(solution)

    assert result.feasible is False


def test_ga_normalization_remaps_duplicate_route_ids_and_task_ids():
    instance = _build_instance()
    evaluator = Evaluator(instance, rendezvous_tolerance=float("inf"))
    repair = FeasibilityRepair(instance, evaluator)
    solution = Solution(
        truck_routes=[
            TruckRoute(route_id=1, nodes=[0, 1, 4], capacity=10.0),
            TruckRoute(route_id=1, nodes=[0, 2, 3, 4], capacity=10.0),
        ],
        drone_tasks=[
            DroneTask(
                task_id=7,
                drone_id=0,
                launch_truck=1,
                launch_node=2,
                customers=[1],
                land_truck=1,
                retrieve_node=3,
                payloads=[4.0, 0.0],
            )
        ],
    )

    normalized = repair.normalize_solution_ids(solution)

    assert [route.id for route in normalized.truck_routes] == [0, 1]
    assert [task.task_id for task in normalized.drone_tasks] == [0]
    assert normalized.drone_tasks[0].launch_truck == 1
    assert normalized.drone_tasks[0].land_truck == 1


def test_ga_route_crossover_tracks_all_customers_in_multi_customer_task():
    instance = _build_instance()
    ga = GeneticAlgorithm.__new__(GeneticAlgorithm)
    ga.instance = instance
    ga._depot_start = 0
    ga._depot_end = 4
    ga._depots = {0, 4}

    class FixedRng:
        @staticmethod
        def sample(_population, _count):
            return [1]

        @staticmethod
        def shuffle(_values):
            return None

    ga.rng = FixedRng()
    parent1 = Solution(
        truck_routes=[
            TruckRoute(route_id=0, nodes=[0, 4], capacity=10.0),
            TruckRoute(route_id=1, nodes=[0, 3, 4], capacity=10.0),
        ],
        drone_tasks=[
            DroneTask(
                task_id=0,
                drone_id=0,
                launch_truck=1,
                launch_node=3,
                customers=[1, 2],
                land_truck=1,
                retrieve_node=4,
                payloads=[8.0, 4.0, 0.0],
            ),
            DroneTask(
                task_id=1,
                drone_id=0,
                launch_truck=1,
                launch_node=3,
                customers=[2],
                land_truck=1,
                retrieve_node=4,
                payloads=[4.0, 0.0],
            ),
        ],
    )

    child = ga._route_crossover(parent1, parent1)

    assert len(child.drone_tasks) == 1
    assert child.drone_tasks[0].customers() == [1, 2]


def test_ga_strict_time_budget_ignores_generation_and_stagnation_caps():
    ga = GeneticAlgorithm.__new__(GeneticAlgorithm)
    ga.config = GAConfig(
        generations=1,
        max_stagnation=0,
        time_limit=600.0,
        strict_time_budget=True,
        adaptive_enabled=False,
    )
    ga.generation = 0
    ga.stagnation_counter = 0
    ga.start_time = 0.0
    ga.stats = {"generations": []}
    individual = Individual(solution=Solution(), fitness=1.0, feasible=True)

    def initialize_population(_initial):
        ga.population = [individual]
        ga.best_individual = individual
        ga.best_feasible_individual = individual

    checks = iter([False, False, False, True])
    ga.initialize_population = initialize_population
    ga._check_time_limits = lambda: next(checks)
    ga._create_new_population = lambda: [individual]
    ga._update_best_feasible_from_population = lambda: None
    ga._apply_local_search = lambda: None
    ga._apply_aggressive_drone_optimization = lambda: None
    ga._record_statistics = lambda: ga.stats["generations"].append(ga.generation)

    best = ga.run(Solution())

    assert best is individual
    assert ga.stats["generations"] == [0, 1, 2]
