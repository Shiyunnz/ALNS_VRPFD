"""Deadline-aware initial solution construction with feasibility diagnostics."""

from __future__ import annotations

from itertools import combinations
from dataclasses import dataclass
from math import isfinite
from typing import Any, Sequence

from alns_vrpfd.model.initializer import build_two_phase_initial_solution
from alns_vrpfd.model.route import DroneTask, TruckRoute
from alns_vrpfd.model.solution import Solution


@dataclass(frozen=True)
class FeasibleInitialDiagnostics:
    """Summary of the initial construction and repair pass."""

    constructor: str
    feasible: bool
    initial_violations: int
    final_violations: int
    initial_lateness: float
    final_lateness: float
    initial_delay_cost: float
    final_delay_cost: float
    drone_tasks_added: int
    iterations: int
    reason: str = ""


def build_feasible_initial_solution(
    instance: Any,
    evaluator: Any,
    *,
    max_drone_repair_iterations: int = 30,
) -> tuple[Solution, FeasibleInitialDiagnostics]:
    """Build a deadline-aware initial solution and repair hard TW violations.

    The constructor first evaluates the legacy two-phase solution so callers get
    diagnostics against the old entry point. It then builds a parallel
    earliest-deadline truck backbone and greedily moves selected route-prefix
    customers to robustly feasible same-route drone sorties when that improves
    hard time-window feasibility.
    """

    legacy = build_two_phase_initial_solution(instance)
    legacy_score, legacy_details = _score(evaluator, legacy)

    current = _build_deadline_truck_backbone(instance, evaluator)
    current_score, current_details = _score(evaluator, current)
    if legacy_score < current_score:
        current = legacy
        current_score = legacy_score
        current_details = legacy_details

    iterations = 0
    while (
        current_details is not None
        and not current_details.result.feasible
        and iterations < max_drone_repair_iterations
    ):
        candidate = _best_same_route_drone_repair(
            instance,
            evaluator,
            current,
            current_score,
            current_details,
        )
        if candidate is None:
            break
        current, current_score, current_details = candidate
        iterations += 1

    final_score, final_details = _score(evaluator, current)
    if final_details is None:
        final_feasible = False
        final_violations = 10**9
        final_lateness = float("inf")
        final_delay = float("inf")
    else:
        final_feasible = bool(final_details.result.feasible)
        final_violations = len(final_details.delay_breakdown.violations)
        final_lateness = _total_lateness(final_details)
        final_delay = float(final_details.result.delay_penalty)

    initial_violations = (
        len(legacy_details.delay_breakdown.violations)
        if legacy_details is not None
        else 10**9
    )
    initial_lateness = (
        _total_lateness(legacy_details) if legacy_details is not None else float("inf")
    )
    initial_delay = (
        float(legacy_details.result.delay_penalty)
        if legacy_details is not None
        else float("inf")
    )

    reason = "feasible" if final_feasible else "bounded_repair_stopped"
    diagnostics = FeasibleInitialDiagnostics(
        constructor="deadline_backbone_drone_repair",
        feasible=final_feasible,
        initial_violations=initial_violations,
        final_violations=final_violations,
        initial_lateness=initial_lateness,
        final_lateness=final_lateness,
        initial_delay_cost=initial_delay,
        final_delay_cost=final_delay,
        drone_tasks_added=len(current.drone_tasks) - len(legacy.drone_tasks),
        iterations=iterations,
        reason=reason,
    )
    return current, diagnostics


def _build_deadline_truck_backbone(instance: Any, evaluator: Any) -> Solution:
    start = instance.customer_manager.depot_start
    end = instance.customer_manager.depot_end or start
    truck_spec = instance.vehicle_specs["truck"]
    truck_count = max(1, int(getattr(truck_spec, "number", 1) or 1))
    capacity = float(getattr(truck_spec, "capacity", float("inf")))

    solution = Solution(
        truck_routes=[
            TruckRoute(route_id=route_id, nodes=[start, end], capacity=capacity)
            for route_id in range(truck_count)
        ],
        drone_tasks=[],
    )

    customers = sorted(
        instance.customer_manager.customers(),
        key=lambda customer: (
            _finite_time(getattr(customer, "latest_time", None)),
            _finite_time(getattr(customer, "optimal_time", None)),
            customer.customer_id,
        ),
    )
    demands = instance.customer_manager.demands()

    for customer in customers:
        best: tuple[tuple[float, ...], Solution] | None = None
        for route_index, route in enumerate(solution.truck_routes):
            for pos in range(1, len(route.nodes)):
                candidate = solution.clone()
                candidate.truck_routes[route_index].nodes.insert(pos, customer.customer_id)
                _refresh_truck_loads(candidate, demands)
                candidate_score, _ = _score(evaluator, candidate)
                if best is None or candidate_score < best[0]:
                    best = (candidate_score, candidate)
        if best is not None:
            solution = best[1]

    _refresh_truck_loads(solution, demands)
    return solution


def _best_same_route_drone_repair(
    instance: Any,
    evaluator: Any,
    base: Solution,
    base_score: tuple[float, ...],
    base_details: Any,
) -> tuple[Solution, tuple[float, ...], Any] | None:
    demands = instance.customer_manager.demands()
    drone_spec = instance.vehicle_specs.get("drone")
    if drone_spec is None or getattr(drone_spec, "number", 0) <= 0:
        return None
    drone_capacity = float(getattr(drone_spec, "capacity", 0.0))

    violation_routes = {
        violation.route_id
        for violation in base_details.delay_breakdown.violations
        if violation.served_by == "truck"
    }
    if not violation_routes:
        return None

    protected_nodes = {
        node
        for task in base.drone_tasks
        for node in (task.launch_node, task.retrieve_node)
    }

    best: tuple[tuple[float, ...], Solution, Any] | None = None
    for route_index, route in enumerate(base.truck_routes):
        if route.id not in violation_routes:
            continue
        last_violation_pos = _last_violation_position(route, base_details)
        if last_violation_pos <= 0:
            continue

        route_len = len(route.nodes)
        for launch_pos in range(0, max(0, last_violation_pos)):
            launch_node = route.nodes[launch_pos]
            for retrieve_pos in range(launch_pos + 2, route_len):
                retrieve_node = route.nodes[retrieve_pos]
                movable_stop = min(retrieve_pos, last_violation_pos + 1)
                segment = [
                    node
                    for node in route.nodes[launch_pos + 1 : movable_stop]
                    if node not in protected_nodes
                ]
                if not segment:
                    continue
                max_customers = min(3, len(segment))
                for count in range(1, max_customers + 1):
                    for customers in combinations(segment, count):
                        if sum(demands.get(customer, 0.0) for customer in customers) > drone_capacity:
                            continue
                        ordered_customers = [
                            node for node in segment if node in customers
                        ]
                        candidate = _with_same_route_drone_task(
                            base,
                            route_index,
                            ordered_customers,
                            launch_node,
                            retrieve_node,
                            demands,
                        )
                        candidate_score, candidate_details = _score(evaluator, candidate)
                        if candidate_details is None:
                            continue
                        if not _non_delay_feasible(evaluator, candidate, candidate_details):
                            continue
                        if candidate_score >= base_score:
                            continue
                        if best is None or candidate_score < best[0]:
                            best = (candidate_score, candidate, candidate_details)

    if best is None:
        return None
    return best[1], best[0], best[2]


def _with_single_customer_drone_task(
    solution: Solution,
    route_index: int,
    customer: int,
    launch_node: int,
    retrieve_node: int,
    demands: dict[int, float],
) -> Solution:
    candidate = solution.clone()
    route = candidate.truck_routes[route_index]
    route.remove_customer(customer)
    task = DroneTask(
        drone_id=0,
        launch_truck=route.id,
        launch_node=launch_node,
        customers=[customer],
        land_truck=route.id,
        retrieve_node=retrieve_node,
        payloads=_payloads([customer], demands),
        task_id=len(candidate.drone_tasks),
    )
    candidate.drone_tasks.append(task)
    _sort_and_reindex_drone_tasks(candidate)
    _refresh_truck_loads(candidate, demands)
    return candidate


def _with_same_route_drone_task(
    solution: Solution,
    route_index: int,
    customers: Sequence[int],
    launch_node: int,
    retrieve_node: int,
    demands: dict[int, float],
) -> Solution:
    candidate = solution.clone()
    route = candidate.truck_routes[route_index]
    for customer in customers:
        route.remove_customer(customer)
    task = DroneTask(
        drone_id=0,
        launch_truck=route.id,
        launch_node=launch_node,
        customers=list(customers),
        land_truck=route.id,
        retrieve_node=retrieve_node,
        payloads=_payloads(customers, demands),
        task_id=len(candidate.drone_tasks),
    )
    candidate.drone_tasks.append(task)
    _sort_and_reindex_drone_tasks(candidate)
    _refresh_truck_loads(candidate, demands)
    return candidate


def _sort_and_reindex_drone_tasks(solution: Solution) -> None:
    positions = {
        route.id: {node: pos for pos, node in enumerate(route.nodes)}
        for route in solution.truck_routes
    }

    def key(task: DroneTask) -> tuple[int, int, int, int, int]:
        launch_truck = -1 if task.launch_truck is None else int(task.launch_truck)
        land_truck = -1 if task.land_truck is None else int(task.land_truck)
        launch_pos = positions.get(task.launch_truck, {}).get(task.launch_node, 0)
        retrieve_pos = positions.get(task.land_truck, {}).get(task.retrieve_node, 10**9)
        return (int(task.drone_id), launch_truck, launch_pos, land_truck, retrieve_pos)

    solution.drone_tasks.sort(key=key)
    for task_id, task in enumerate(solution.drone_tasks):
        task.task_id = task_id


def _last_violation_position(route: TruckRoute, details: Any) -> int:
    positions = []
    for violation in details.delay_breakdown.violations:
        if violation.route_id != route.id:
            continue
        if violation.node_id not in route.nodes:
            continue
        positions.append(route.nodes.index(violation.node_id))
    return max(positions, default=-1)


def _score(evaluator: Any, solution: Solution) -> tuple[tuple[float, ...], Any | None]:
    try:
        details = evaluator.evaluate_with_details(solution)
    except Exception:
        return (9.0, 10**9, float("inf"), float("inf"), float("inf"), 9.0), None

    result = details.result
    violations = len(details.delay_breakdown.violations)
    lateness = _total_lateness(details)
    distance = _finite_or_large(result.truck_distance_cost) + _finite_or_large(
        result.drone_distance_cost
    )
    robust_flag = 0.0 if details.robustness.feasible else 1.0
    return (
        0.0 if result.feasible else 1.0,
        0.0 if _non_delay_feasible(evaluator, solution, details) else 1.0,
        float(violations),
        lateness,
        _finite_or_large(result.delay_penalty),
        distance,
        robust_flag,
    ), details


def _non_delay_feasible(evaluator: Any, solution: Solution, details: Any) -> bool:
    if not getattr(details.robustness, "feasible", False):
        return False
    checks = (
        "_has_drone_anchor_conflicts",
        "_has_drone_limit_violations",
        "_has_drone_task_violations",
        "_has_customer_coverage_violation",
        "_has_forced_drone_violation",
    )
    for name in checks:
        check = getattr(evaluator, name, None)
        if check is None:
            continue
        if name == "_has_drone_anchor_conflicts":
            if check(solution.drone_tasks):
                return False
        elif check(solution):
            return False
    return True


def _total_lateness(details: Any) -> float:
    return float(
        sum(
            max(0.0, violation.arrival_time - violation.latest_time)
            for violation in details.delay_breakdown.violations
        )
    )


def _payloads(customers: Sequence[int], demands: dict[int, float]) -> list[float]:
    remaining = float(sum(demands.get(customer, 0.0) for customer in customers))
    payloads = [remaining]
    for customer in customers:
        remaining -= float(demands.get(customer, 0.0))
        payloads.append(remaining)
    return payloads


def _refresh_truck_loads(solution: Solution, demands: dict[int, float]) -> None:
    for route in solution.truck_routes:
        route.current_load = float(sum(demands.get(customer, 0.0) for customer in route.customers()))


def _finite_time(value: float | None) -> float:
    if value is None:
        return float("inf")
    return float(value)


def _finite_or_large(value: float) -> float:
    value = float(value)
    if isfinite(value):
        return value
    return 1e9
