"""Evaluation logic for computing costs, timings, and robustness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple
from collections import Counter
import logging

from alns_vrpfd.deprivation import deprivation_cost
from alns_vrpfd.evaluation.energy import DroneEnergyModel
from alns_vrpfd.evaluation.robustness import (
    RobustnessChecker,
    RobustnessResult,
)
from alns_vrpfd.evaluation.timing import (
    RendezvousResult,
    TimingCalculator,
    TruckRouteTiming,
)
from alns_vrpfd.instance.manager import InstanceManager, VehicleSpec
from alns_vrpfd.model.route import DroneTask, DroneTaskContext, DroneTaskTiming, TruckRoute
from alns_vrpfd.model.solution import Solution

__all__ = [
    "EvaluationResult",
    "EvaluationDetails",
    "DelayBreakdown",
    "NodeDelay",
    "Evaluator",
]


@dataclass(frozen=True)
class EvaluationResult:
    """Container for aggregated evaluation metrics."""
    """Container for aggregated evaluation metrics."""

    feasible: bool
    total_cost: float
    truck_distance_cost: float
    drone_distance_cost: float
    delay_penalty: float
    energy_penalty: float


@dataclass(frozen=True)
class NodeDelay:
    """Delay information for an individual serviced node."""
    """Delay information for an individual serviced node."""

    node_id: int
    arrival_time: float
    reference_time: float
    delay: float
    served_by: str  # "truck" or "drone"
    route_id: int


@dataclass(frozen=True)
class DelayBreakdown:
    """Aggregated delay penalty details."""
    """Aggregated delay penalty details."""

    total_delay: float
    nodes: Tuple[NodeDelay, ...] = field(default_factory=tuple)
    violations: Tuple["TimeWindowViolation", ...] = field(
        default_factory=tuple)


@dataclass(frozen=True)
class TimeWindowViolation:
    """Record of a hard time-window violation."""
    """Record of a hard time-window violation."""

    node_id: int
    arrival_time: float
    latest_time: float
    served_by: str
    route_id: int


@dataclass(frozen=True)
class EvaluationDetails:
    """Full evaluation outcome including timings and robustness."""
    """Full evaluation outcome including timings and robustness."""

    result: EvaluationResult
    truck_timings: Mapping[int, TruckRouteTiming]
    drone_timings: Mapping[int, DroneTaskTiming]
    rendezvous_results: Mapping[int, RendezvousResult]
    robustness: RobustnessResult
    delay_breakdown: DelayBreakdown


class Evaluator:
    """Compute costs, timing tables, and robustness checks for a solution."""
    """Compute costs, timing tables, and robustness checks for a solution."""

    def __init__(
        self,
        instance: InstanceManager,
        *,
        truck_cost_per_km: float | None = None,
        drone_cost_per_km: float | None = None,
        # Default to infinite to match MIP
        rendezvous_tolerance: float = float('inf'),
        energy_model: DroneEnergyModel | None = None,
        robustness_checker: RobustnessChecker | None = None,
        service_times: Mapping[int, float] | None = None,
        drone_service_times: Mapping[int, float] | None = None,
        time_tolerance: float = 1e-6,
        forced_drone_customers: Sequence[int] | None = None,
        allow_multiple_launch_per_node: bool = True,
        cost_lambda: float = 12.0,
        cost_rho: float = 1.0,
        cost_normalized: bool = True,
    ) -> None:
        self._instance = instance
        self._truck_cost_per_km = self._resolve_unit_cost(
            instance, "truck", truck_cost_per_km)
        self._drone_cost_per_km = self._resolve_unit_cost(
            instance, "drone", drone_cost_per_km)
        self._rendezvous_tolerance = rendezvous_tolerance
        self._time_tolerance = time_tolerance
        self._service_times = dict(service_times or {})
        self._drone_service_times = dict(drone_service_times or {})
        self._forced_drone_customers = set(forced_drone_customers or [])
        self._allow_multiple_launch_per_node = allow_multiple_launch_per_node
        self._cost_lambda = cost_lambda
        self._cost_rho = cost_rho
        self._cost_normalized = cost_normalized
        self._logger = logging.getLogger(__name__)
        self._duplicate_task_warning_emitted = False

        config = instance.robust_config
        self._energy_model = energy_model or DroneEnergyModel()
        self._robustness_checker = robustness_checker or RobustnessChecker(
            energy_model=self._energy_model,
            battery_capacity=config.drone_battery_capacity,
            energy_uncertainty_budget=config.energy_uncertainty_budget,
            energy_deviation_rate=config.energy_deviation_rate,
        )

        # Precompute lookup for customer data
        self._customer_lookup = {
            customer.customer_id: customer
            for customer in instance.customer_manager.customers()
        }
        self._depot_start = instance.customer_manager.depot_start
        self._depot_end = instance.customer_manager.depot_end

    # ------------------------------------------------------------------
    def evaluate_solution(self, solution: Solution) -> EvaluationResult:
        """Return cost-only summary using the detailed evaluation pipeline."""
        """Return cost-only summary using the detailed evaluation pipeline."""
        return self.evaluate_with_details(solution).result

    # ------------------------------------------------------------------
    def evaluate_with_details(self, solution: Solution) -> EvaluationDetails:
        """Compute timings, costs, and robustness for the given solution."""
        """Compute timings, costs, and robustness for the given solution."""
        if self._has_duplicate_truck_route_ids(solution):
            return self._infeasible_details()
        if self._has_duplicate_drone_task_ids(solution):
            return self._infeasible_details()

        if self._has_depot_start_retrieve_violation(solution):
            return self._infeasible_details()

        node_ids = self._instance.all_node_ids()
        node_index = {node: idx for idx, node in enumerate(node_ids)}
        truck_time_matrix = self._instance.time_matrix("truck")
        drone_time_matrix = self._instance.time_matrix("drone")
        timing_calculator = TimingCalculator(
            node_index=node_index,
            truck_time_matrix=truck_time_matrix,
            drone_time_matrix=drone_time_matrix,
            service_times=self._service_times,
            drone_service_times=self._drone_service_times,
        )

        truck_timings = self._compute_truck_timings(
            solution.truck_routes, timing_calculator)

        # Precompute task_id counts for disambiguation of keys
        task_id_counts = Counter(
            t.task_id for t in solution.drone_tasks if t.task_id is not None)

        # Iterative synchronization (2 passes)
        for _ in range(2):
            truck_arrival_map = {
                route_id: timing.arrival_times for route_id, timing in truck_timings.items()
            }

            drone_timings, rendezvous_results = self._compute_drone_timings(
                solution.drone_tasks,
                timing_calculator,
                truck_timings,
            )

            # Calculate required delays for synchronization
            required_departures: Dict[int, Dict[int, float]] = {}
            needs_update = False

            for index, task in enumerate(solution.drone_tasks):
                if task.land_truck is None:
                    continue

                # Determine a unique internal key for this task to avoid
                # overwriting timings if task.task_id is duplicated across tasks.
                # If task_id is duplicated, fall back to enumerated index.
                # This keeps existing behaviour when task_id is unique.
                if getattr(task, "task_id", None) is not None and task_id_counts.get(task.task_id, 0) == 1:
                    key = task.task_id
                else:
                    key = index
                d_timing = drone_timings.get(key)
                if not d_timing:
                    continue

                retrieve_node = task.retrieve_node
                truck_id = task.land_truck
                # Truck departure must be >= Drone Arrival (Retrieve Time)
                min_dep = d_timing.retrieve_time

                if truck_id not in required_departures:
                    required_departures[truck_id] = {}

                current_req = required_departures[truck_id].get(
                    retrieve_node, 0.0)
                if min_dep > current_req:
                    required_departures[truck_id][retrieve_node] = min_dep

            # Check for violations and update
            for route in solution.truck_routes:
                tid = route.id
                if tid in required_departures:
                    current_timing = truck_timings[tid]
                    for node, min_time in required_departures[tid].items():
                        # Check departure time
                        if current_timing.departure_times.get(node, 0.0) < min_time - 1e-5:
                            needs_update = True
                            break
                if needs_update:
                    break

            if not needs_update:
                break

            # Recompute with constraints
            truck_timings = self._compute_truck_timings(
                solution.truck_routes,
                timing_calculator,
                min_departures=required_departures
            )

        # FINAL UPDATE: Ensure drone timings reflect the FINAL truck schedule (with delays)
        # This fixes the bug where drones launched "in the past" relative to delayed trucks.
        drone_timings, rendezvous_results = self._compute_drone_timings(
            solution.drone_tasks,
            timing_calculator,
            truck_timings,
        )

        # Update arrival map for checks
        truck_arrival_map = {
            route_id: timing.arrival_times for route_id, timing in truck_timings.items()
        }

        delay_breakdown = self._compute_delay_penalty(
            truck_timings, drone_timings)
        truck_distance_cost, drone_distance_cost = self._compute_distance_costs(
            solution)

        anchor_conflict = self._has_drone_anchor_conflicts(
            solution.drone_tasks)

        drone_limit_violation = self._has_drone_limit_violations(
            solution)

        drone_task_violation = self._has_drone_task_violations(
            solution)

        # Check customer coverage
        customer_coverage_violation = self._has_customer_coverage_violation(
            solution)

        # Check truck capacities in the canonical evaluator as well as in the
        # search-phase evaluator. Final result validation must be self-contained.
        truck_capacity_violation = self._has_truck_capacity_violation(solution)

        # Check forced drone customers
        forced_drone_violation = self._has_forced_drone_violation(solution)

        robustness = self._run_robustness_checks(
            solution=solution,
            truck_arrivals=truck_arrival_map,
            drone_timings=drone_timings,
        )

        base_cost = (
            truck_distance_cost
            + drone_distance_cost
            + delay_breakdown.total_delay
        )
        # (robustness.feasible)
        feasible = (not delay_breakdown.violations and not anchor_conflict and
                    not drone_limit_violation and not drone_task_violation and
                    not customer_coverage_violation and not truck_capacity_violation and
                    not forced_drone_violation and
                    robustness.feasible)
        total_cost = base_cost if feasible else float("inf")

        result = EvaluationResult(
            feasible=feasible,
            total_cost=total_cost,
            truck_distance_cost=truck_distance_cost,
            drone_distance_cost=drone_distance_cost,
            delay_penalty=delay_breakdown.total_delay,
            energy_penalty=0.0,
        )

        return EvaluationDetails(
            result=result,
            truck_timings=truck_timings,
            drone_timings=drone_timings,
            rendezvous_results=rendezvous_results,
            robustness=robustness,
            delay_breakdown=delay_breakdown,
        )

    # ------------------------------------------------------------------
    def _has_duplicate_truck_route_ids(self, solution: Solution) -> bool:
        """Return True when multiple truck routes share the same identity."""
        """Return True when multiple truck routes share the same identity."""
        route_ids = [route.id for route in solution.truck_routes]
        return len(route_ids) != len(set(route_ids))

    # ------------------------------------------------------------------
    def _has_duplicate_drone_task_ids(self, solution: Solution) -> bool:
        """Return True when multiple drone tasks share a non-null identity."""
        """Return True when multiple drone tasks share a non-null identity."""
        task_ids = [
            task.task_id
            for task in solution.drone_tasks
            if task.task_id is not None
        ]
        return len(task_ids) != len(set(task_ids))

    # ------------------------------------------------------------------
    def _has_truck_capacity_violation(self, solution: Solution) -> bool:
        """Return True when any truck route exceeds the instance capacity."""
        """Return True when any truck route exceeds the instance capacity."""
        truck_spec = self._instance.vehicle_specs.get("truck")
        if truck_spec is None:
            return False
        capacity = float(truck_spec.capacity)
        demands = self._instance.customer_manager.demands()
        for route in solution.truck_routes:
            load = sum(float(demands.get(customer, 0.0)) for customer in route.customers())
            if load > capacity + 1e-9:
                return True
        return False

    # ------------------------------------------------------------------
    def _has_depot_start_retrieve_violation(self, solution: Solution) -> bool:
        """Return True when a two-depot solution retrieves a drone at the start depot."""
        """Return True when a two-depot solution retrieves a drone at the start depot."""
        return any(
            self._is_start_depot_retrieve(task)
            for task in solution.drone_tasks
        )

    def _is_start_depot_retrieve(self, task: DroneTask) -> bool:
        if (self._depot_start is None
                or self._depot_end is None
                or self._depot_start == self._depot_end):
            return False
        return task.retrieve_node == self._depot_start

    def _infeasible_details(self) -> EvaluationDetails:
        result = EvaluationResult(
            feasible=False,
            total_cost=float("inf"),
            truck_distance_cost=float("inf"),
            drone_distance_cost=float("inf"),
            delay_penalty=0.0,
            energy_penalty=0.0,
        )
        return EvaluationDetails(
            result=result,
            truck_timings={},
            drone_timings={},
            rendezvous_results={},
            robustness=RobustnessResult(feasible=False, margin=float("-inf")),
            delay_breakdown=DelayBreakdown(total_delay=0.0),
        )

    # ------------------------------------------------------------------
    def _compute_truck_timings(
        self,
        truck_routes: Iterable[TruckRoute],
        timing_calculator: TimingCalculator,
        min_departures: Mapping[int, Mapping[int, float]] | None = None,
    ) -> Dict[int, TruckRouteTiming]:
        timings: Dict[int, TruckRouteTiming] = {}
        min_deps = min_departures or {}
        for route in truck_routes:
            constraints = min_deps.get(route.id)
            timing = timing_calculator.truck_timing(
                route, min_departure_times=constraints)
            timings[route.id] = timing
        return timings

    # ------------------------------------------------------------------
    def _compute_drone_timings(
        self,
        drone_tasks: Sequence[DroneTask],
        timing_calculator: TimingCalculator,
        truck_timings: Mapping[int, TruckRouteTiming],
    ) -> Tuple[Dict[int, DroneTaskTiming], Dict[int, RendezvousResult]]:
        timings: Dict[int, DroneTaskTiming] = {}
        rendezvous_results: Dict[int, RendezvousResult] = {}

        # Precompute task_id counts to detect duplicates and avoid key collisions
        task_id_counts = Counter(
            t.task_id for t in drone_tasks if t.task_id is not None)
        # Warn when duplicate task_id values exist; use enumerated index to guarantee unique keys
        duplicate_ids = [tid for tid, cnt in task_id_counts.items() if cnt > 1]
        if duplicate_ids and not self._duplicate_task_warning_emitted:
            self._logger.warning(
                f"Duplicate DroneTask.task_id values detected: {duplicate_ids}. "
                "Evaluator will disambiguate by using task indices for these tasks."
            )
            self._duplicate_task_warning_emitted = True
        for index, task in enumerate(drone_tasks):
            if task.launch_truck is None:
                # Depot launch
                launch_departure = 0.0
            else:
                launch_route_timing = truck_timings.get(task.launch_truck)
                if launch_route_timing is None:
                    raise KeyError(
                        f"Missing truck timing for launch truck {task.launch_truck}"
                    )
                launch_departure = launch_route_timing.departure_times.get(
                    task.launch_node)
                if launch_departure is None:
                    raise KeyError(
                        f"Launch node {task.launch_node} not visited by truck {task.launch_truck}."
                    )

            drone_timing = timing_calculator.drone_timing(
                task,
                launch_time=launch_departure,
            )
            if getattr(task, "task_id", None) is not None and task_id_counts.get(task.task_id, 0) == 1:
                key = task.task_id
            else:
                key = index
            timings[key] = drone_timing

            if task.land_truck is not None:
                land_route_timing = truck_timings.get(task.land_truck)
                if land_route_timing is None:
                    raise KeyError(
                        f"Missing truck timing for retrieval truck {task.land_truck}"
                    )
                truck_retrieve_arrival = land_route_timing.arrival_times.get(
                    task.retrieve_node)
                if truck_retrieve_arrival is None:
                    raise KeyError(
                        f"Retrieve node {task.retrieve_node} not visited by truck {task.land_truck}."
                    )

                rendezvous = timing_calculator.rendezvous(
                    drone_timing=drone_timing,
                    truck_retrieve_arrival=truck_retrieve_arrival,
                    tolerance=self._rendezvous_tolerance,
                )
                rendezvous_results[key] = rendezvous

        return timings, rendezvous_results

    # ------------------------------------------------------------------
    def _compute_delay_penalty(
        self,
        truck_timings: Mapping[int, TruckRouteTiming],
        drone_timings: Mapping[int, DroneTaskTiming],
    ) -> DelayBreakdown:
        delays: List[NodeDelay] = []
        violations: List[TimeWindowViolation] = []

        # Truck customer delays and violations
        for route_id, timing in truck_timings.items():
            for node_id, arrival in timing.arrival_times.items():
                if not self._is_customer(node_id):
                    continue
                customer = self._customer_lookup.get(node_id)
                optimal = customer.optimal_time if customer else None
                latest = customer.latest_time if customer else None

                if latest is not None and arrival - latest > self._time_tolerance:
                    violations.append(
                        TimeWindowViolation(
                            node_id=node_id,
                            arrival_time=arrival,
                            latest_time=latest,
                            served_by="truck",
                            route_id=route_id,
                        )
                    )

                delay_value = 0.0
                if optimal is not None and arrival - optimal > self._time_tolerance:
                    delay_value = arrival - optimal

                if delay_value > 0.0:
                    delays.append(
                        NodeDelay(
                            node_id=node_id,
                            arrival_time=arrival,
                            reference_time=optimal or 0.0,
                            delay=delay_value,
                            served_by="truck",
                            route_id=route_id,
                        )
                    )

        # Drone customer delays and violations
        for task_key, timing in drone_timings.items():
            for node_id, arrival in timing.customer_arrival_times.items():
                customer = self._customer_lookup.get(node_id)
                optimal = customer.optimal_time if customer else None
                latest = customer.latest_time if customer else None

                if latest is not None and arrival - latest > self._time_tolerance:
                    violations.append(
                        TimeWindowViolation(
                            node_id=node_id,
                            arrival_time=arrival,
                            latest_time=latest,
                            served_by="drone",
                            route_id=int(task_key),
                        )
                    )

                delay_value = 0.0
                if optimal is not None and arrival - optimal > self._time_tolerance:
                    delay_value = arrival - optimal

                if delay_value > 0.0:
                    delays.append(
                        NodeDelay(
                            node_id=node_id,
                            arrival_time=arrival,
                            reference_time=optimal or 0.0,
                            delay=delay_value,
                            served_by="drone",
                            route_id=int(task_key),
                        )
                    )

        total_delay = 0.0
        for delay in delays:
            tau = delay.delay
            # Ensure non-negative
            if tau <= 0.0:
                continue
            customer = self._customer_lookup.get(delay.node_id)
            supply_class = customer.supply_class if customer else None
            cost = deprivation_cost(
                tau, supply_class,
                cost_lambda=self._cost_lambda,
                rho=self._cost_rho,
                normalized=self._cost_normalized,
            )
            total_delay += cost
        return DelayBreakdown(
            total_delay=total_delay,
            nodes=tuple(delays),
            violations=tuple(violations),
        )

    # ------------------------------------------------------------------
    def _has_drone_anchor_conflicts(self, drone_tasks: Sequence[DroneTask]) -> bool:
        """Detect whether multiple drone tasks reuse the same truck-node anchor.

        UPDATE: Disabled to allow multiple launches/retrievals at the same node.
        """
        if self._allow_multiple_launch_per_node:
            return False

        used_anchors: set[tuple[int, int]] = set()
        for task in drone_tasks:
            if task.launch_truck is not None:
                anchor = (task.launch_truck, task.launch_node)
                if anchor in used_anchors:
                    return True
                used_anchors.add(anchor)
            if task.land_truck is not None:
                anchor = (task.land_truck, task.retrieve_node)
                if anchor in used_anchors:
                    return True
                used_anchors.add(anchor)
        return False

    def _has_drone_limit_violations(self, solution: Solution) -> bool:
        """Check for drone consistency: Truck binding, Single start, and Sequential Tasks.

        Note: Truck binding check respects the same_truck_retrieval configuration.
        When same_truck_retrieval=False (flexible docking), cross-truck retrieval is allowed.
        """
        drone_tasks = solution.drone_tasks
        drone_trucks: Dict[int, set[int]] = {}
        drone_depot_starts: Dict[int, int] = {}
        drone_depot_ends: Dict[int, int] = {}
        drone_intervals: Dict[int, List[Tuple[int, int]]] = {}

        # Check if cross-truck retrieval is allowed
        same_truck_only = self._instance.robust_config.same_truck_retrieval

        # Map truck route nodes to indices for sequencing
        truck_node_indices: Dict[int, Dict[int, int]] = {}
        for route in solution.truck_routes:
            truck_node_indices[route.id] = {
                node: i for i, node in enumerate(route.nodes)}

        for task in drone_tasks:
            d_id = task.drone_id
            if d_id not in drone_trucks:
                drone_trucks[d_id] = set()
                drone_depot_starts[d_id] = 0
                drone_depot_ends[d_id] = 0
                drone_intervals[d_id] = []

            if self._is_start_depot_retrieve(task):
                return True

            # 1. Collect Truck Bindings
            if task.launch_truck is not None:
                drone_trucks[d_id].add(task.launch_truck)
            if task.land_truck is not None:
                drone_trucks[d_id].add(task.land_truck)

            # 2. Track Depot usage
            if task.launch_node == self._depot_start:
                drone_depot_starts[d_id] += 1
            if task.retrieve_node == self._depot_end or (
                self._depot_start == self._depot_end
                and task.retrieve_node == self._depot_start
            ):
                drone_depot_ends[d_id] += 1

            # 3. Build Intervals
            # We need the truck ID to look up indices.
            # Assume drone bound to 1 truck (checked later).
            # If task uses Depot, treat index as 0 (Start) or Len-1 (End).

            # Determining active truck for this task
            t_id = task.launch_truck if task.launch_truck is not None else task.land_truck
            if t_id is not None and t_id in truck_node_indices:
                indices = truck_node_indices[t_id]

                # Get Start Index
                l_idx = -1
                if task.launch_node == self._depot_start:
                    l_idx = 0
                elif task.launch_truck is not None and task.launch_node in indices:
                    l_idx = indices[task.launch_node]

                # Get End Index
                r_idx = -1
                if task.retrieve_node == self._depot_end or (
                    self._depot_start == self._depot_end
                    and task.retrieve_node == self._depot_start
                ):
                    r_idx = len(indices) + 999  # Large number for end
                    # Better: use actual length if depot is in route?
                    # Usually Truck Route ends with Depot.
                    # If route is [0, A, B, 11], index of 11 is 3.
                    if self._depot_end in indices:
                        r_idx = indices[self._depot_end]
                    elif self._depot_start in indices:  # If 0==11
                        r_idx = indices[self._depot_start]
                        if r_idx == 0:
                            r_idx = 9999  # Force end
                elif task.land_truck is not None and task.retrieve_node in indices:
                    r_idx = indices[task.retrieve_node]

                if l_idx != -1 and r_idx != -1:
                    drone_intervals[d_id].append((l_idx, r_idx))

        for d_id, trucks in drone_trucks.items():
            # Check 1: Truck Consistency (only when same_truck_retrieval is enabled)
            if same_truck_only and len(trucks) > 1:
                return True
            # Check 2: Single Depot Start/End
            if drone_depot_starts[d_id] > 1:
                return True
            if drone_depot_ends[d_id] > 1:
                return True

            # Check 3: Sequential Continuity
            if len(trucks) > 1:
                continue
            # Sort intervals by launch index
            intervals = sorted(drone_intervals[d_id])
            for i in range(len(intervals) - 1):
                curr_launch, curr_retrieve = intervals[i]
                next_launch, next_retrieve = intervals[i+1]

                # Overlap check: Next task must start AFTER or AT current task retrieval
                if next_launch < curr_retrieve:
                    # Overlap!
                    # Special case: If next_launch == curr_launch, it's a fork (Definite Violation)
                    # If next_launch < curr_retrieve, drone is still busy/in-air/on-truck for Task A.
                    return True

        return False

    def _has_drone_task_violations(self, solution: Solution) -> bool:
        """Check if any drone task has feasibility violations."""
        """Check if any drone task has feasibility violations."""
        drone_tasks = solution.drone_tasks

        # Build context with capacity
        drone_spec = self._instance.vehicle_specs.get("drone")
        drone_capacity = drone_spec.capacity if drone_spec is not None else None
        context = DroneTaskContext(drone_capacity=drone_capacity)

        # Build set of customers served by trucks
        truck_served_customers = set()
        for task in drone_tasks:
            # Check individual task feasibility first (duplicates, launch/retrieve issues, capacity)
            if task.feasibility_errors(context):
                return True

        # Check for customer conflicts between drone tasks
        served_customers = set()
        for task in drone_tasks:
            task_customers = set(task.customers())
            conflicts = task_customers.intersection(served_customers)
            if conflicts:
                return True
            served_customers.update(task_customers)

        # Check drone physical location consistency (critical for flexible docking)
        if self._has_drone_location_violation(drone_tasks, solution):
            return True

        # Check for node synchronization conflicts (MIP constraint)
        # Multiple drones cannot launch/land at the same node simultaneously
        if self._has_node_sync_conflict(drone_tasks, solution):
            return True

        return False

    def _has_node_sync_conflict(self, drone_tasks: Sequence[DroneTask], solution: Solution) -> bool:
        """Check if multiple drones launch/land at the same node on the same truck route position.

        UPDATE (2024-12): MIP allows different drones to share the same launch/land node.
        The constraint that matters is that the SAME drone cannot have overlapping tasks
        (which is checked in _has_drone_limit_violations via interval overlap).

        Different drones CAN launch/land at the same node because they are independent
        vehicles. This matches MIP behavior where:
        - Drone 0: Launch 1 -> Serves [9] -> Land 5
        - Drone 1: Launch 1 -> Serves [10] -> Land 5
        is a valid solution.

        We only need to ensure that if multiple tasks use the same node, the truck
        actually visits that node (which is ensured by the truck route structure).
        """
        # Disabled: MIP allows multiple drones to use same launch/land nodes
        return False

    def _has_drone_location_violation(self, drone_tasks: Sequence[DroneTask], solution: Solution = None) -> bool:
        """Check if drone tasks violate physical location constraints.

        A drone that lands on truck B cannot subsequently launch from truck A.
        Additionally, if a drone lands at node X on truck T, its next task
        from truck T must launch from a node at or after X in T's route.

        This validates that drone task sequences are physically feasible.

        IMPORTANT: Tasks must be sorted by their launch position in the truck route
        to determine the correct execution order.
        """
        # Build truck route node position maps
        truck_node_positions: dict[int, dict[int, int]] = {}
        if solution is not None:
            for route in solution.truck_routes:
                truck_node_positions[route.id] = {
                    node: pos for pos, node in enumerate(route.nodes)
                }

        # Group tasks by drone
        from collections import defaultdict
        drone_task_map: dict[int, list[DroneTask]] = defaultdict(list)
        for task in drone_tasks:
            drone_task_map[task.drone_id].append(task)

        for drone_id, tasks in drone_task_map.items():
            # Sort tasks by launch position (needed even for single-task
            # to correctly run Check 4: same-truck launch/retrieve ordering)
            if not tasks:
                continue

            # Sort tasks by launch position in truck route (critical for correct ordering)
            def get_launch_position(task: DroneTask) -> int:
                """Get launch position in truck route for sorting."""
                """Get launch position in truck route for sorting."""
                launch_truck = task.launch_truck
                if launch_truck is None:
                    # Depot launch - position 0
                    return 0
                if launch_truck in truck_node_positions:
                    return truck_node_positions[launch_truck].get(task.launch_node, 0)
                return 0

            involved_trucks = {
                truck
                for task in tasks
                for truck in (task.launch_truck, task.land_truck)
                if truck is not None
            }
            if len(involved_trucks) <= 1:
                sorted_tasks = sorted(tasks, key=get_launch_position)
            else:
                # Flexible docking has no single route coordinate system across
                # trucks. Preserve the task sequence produced by the route
                # reconstruction/ALNS solution.
                sorted_tasks = tasks

            current_truck = None  # None means drone is free to start anywhere
            current_node = None   # The node where drone is currently located
            current_retrieve_pos = -1  # Position of current retrieve node in truck route
            at_depot_end = False  # Drone returned to depot_end, cannot fly again

            for task in sorted_tasks:
                launch_truck = task.launch_truck
                launch_node = task.launch_node
                land_truck = task.land_truck
                retrieve_node = task.retrieve_node

                # Check 0: drone at depot_end cannot start new sorties
                if at_depot_end:
                    return True

                # Check 1: flexible docking allows drone to switch between trucks.
                # Reset tracking when drone switches to a different truck.
                if current_truck is not None and launch_truck != current_truck:
                    current_truck = None
                    current_node = None
                    current_retrieve_pos = -1

                # Check 2: if same truck, is launch_node reachable from current_node?
                # (launch_node must be at or after current_node in the truck's route)
                if (current_truck is not None and current_truck == launch_truck
                        and current_node is not None and launch_truck in truck_node_positions):
                    node_positions = truck_node_positions[launch_truck]
                    current_pos = node_positions.get(current_node)
                    launch_pos = node_positions.get(launch_node)

                    if current_pos is not None and launch_pos is not None:
                        if launch_pos < current_pos:
                            # launch_node is BEFORE current_node - impossible to go back!
                            return True

                # Check 3: If this task's launch position is before the previous task's retrieve position,
                # it's a time conflict (drone cannot be in two places at once)
                if current_retrieve_pos >= 0:
                    launch_pos = get_launch_position(task)
                    if launch_pos < current_retrieve_pos:
                        # This task launches before the previous task completes - CONFLICT!
                        return True

                # Check 4: For same-truck sorties (launch_truck == land_truck),
                # the drone departs from launch_node and must reach retrieve_node
                # before the truck has moved past it. Physically, launch_node must
                # be at or before retrieve_node in the truck's route, otherwise
                # the drone would need to travel back in time.
                if (launch_truck is not None and land_truck is not None
                        and launch_truck == land_truck
                        and launch_truck in truck_node_positions):
                    positions = truck_node_positions[launch_truck]
                    l_pos = positions.get(launch_node)
                    r_pos = positions.get(retrieve_node)
                    if l_pos is not None and r_pos is not None:
                        if l_pos > r_pos:
                            # Same-truck sortie: launch after retrieve — physically impossible
                            return True

                # Update drone location after this task
                current_truck = land_truck
                current_node = retrieve_node

                if land_truck is None and retrieve_node == self._instance.customer_manager.depot_end:
                    at_depot_end = True

                # Update retrieve position for next iteration
                if land_truck is not None and land_truck in truck_node_positions:
                    current_retrieve_pos = truck_node_positions[land_truck].get(
                        retrieve_node, -1)
                elif land_truck is None:
                    current_retrieve_pos = 999999

        return False

    # ------------------------------------------------------------------
    def _has_customer_coverage_violation(self, solution: Solution) -> bool:
        """Check if all customers are served exactly once."""
        """Check if all customers are served exactly once."""
        # Get all required customers (excluding depots)
        required_customers = set(self._customer_lookup.keys())

        # Collect served customers from truck routes (with duplicate detection)
        truck_served = set()
        for route in solution.truck_routes:
            for node in route.nodes:
                if node in required_customers:
                    if node in truck_served:
                        # Duplicate customer in truck routes!
                        return True
                    truck_served.add(node)

        # Collect served customers from drone tasks (with duplicate detection)
        drone_served = set()
        for task in solution.drone_tasks:
            for cust in task.customers():
                if cust in required_customers:
                    if cust in drone_served:
                        # Duplicate customer in drone tasks!
                        return True
                    drone_served.add(cust)

        all_served = truck_served | drone_served

        # Check for missing customers
        missing = required_customers - all_served
        if missing:
            return True

        # Check for duplicates (served by both truck and drone)
        duplicates = truck_served & drone_served
        if duplicates:
            return True

        return False

    # ------------------------------------------------------------------
    def _has_forced_drone_violation(self, solution: Solution) -> bool:
        """Check if forced drone customers appear in truck routes (forbidden)."""
        """Check if forced drone customers appear in truck routes (forbidden)."""
        if not self._forced_drone_customers:
            return False

        # （depot）
        for route in solution.truck_routes:
            for cust in self._forced_drone_customers:
                if cust in route.nodes[1:-1]:
                    return True

        return False

    # ------------------------------------------------------------------
    def _compute_distance_costs(self, solution: Solution) -> Tuple[float, float]:
        truck_matrix = self._instance.distance_matrix("truck")
        drone_matrix = self._instance.distance_matrix("drone")
        node_index = {node: idx for idx, node in enumerate(
            self._instance.all_node_ids())}

        truck_distance = 0.0
        for route in solution.truck_routes:
            truck_distance += self._route_distance(
                route.nodes, truck_matrix, node_index)

        drone_distance = 0.0
        for task in solution.drone_tasks:
            nodes = [task.launch_node, *task.customers(), task.retrieve_node]
            drone_distance += self._route_distance(
                nodes, drone_matrix, node_index)

        return (
            truck_distance * self._truck_cost_per_km,
            drone_distance * self._drone_cost_per_km,
        )

    # ------------------------------------------------------------------
    def _run_robustness_checks(
        self,
        *,
        solution: Solution,
        truck_arrivals: Mapping[int, Mapping[int, float]],
        drone_timings: Mapping[int, DroneTaskTiming],
    ) -> RobustnessResult:
        demands = self._instance.customer_manager.demands()
        latest_times = {
            cid: customer.latest_time
            for cid, customer in self._customer_lookup.items()
        }

        contexts: MutableMapping[int, DroneTaskContext] = {}
        drone_spec = self._instance.vehicle_specs.get("drone")
        drone_capacity = drone_spec.capacity if drone_spec is not None else None
        # Ensure we use the same disambiguation logic as _compute_drone_timings.
        task_id_counts = Counter(
            t.task_id for t in solution.drone_tasks if t.task_id is not None)
        for index, task in enumerate(solution.drone_tasks):
            if getattr(task, "task_id", None) is not None and task_id_counts.get(task.task_id, 0) == 1:
                key = task.task_id
            else:
                key = index
            timing = drone_timings[key]
            context = DroneTaskContext(
                truck_arrival_times=truck_arrivals,
                wait_max=self._rendezvous_tolerance,
                drone_capacity=drone_capacity,
                customer_demands=demands,
                customer_latest_times=latest_times,
                timing=timing,
                energy_model=self._energy_model,
                energy_uncertainty_budget=self._instance.robust_config.energy_uncertainty_budget,
                drone_energy_capacity=self._instance.robust_config.drone_battery_capacity,
                energy_deviation_rate=self._instance.robust_config.energy_deviation_rate,
            )
            contexts[key] = context

        return self._robustness_checker.check(solution, contexts=contexts)

    # ------------------------------------------------------------------
    def _route_distance(
        self,
        nodes: Iterable[int],
        matrix: Sequence[Sequence[float]],
        node_index: Mapping[int, int],
    ) -> float:
        nodes = list(nodes)
        distance = 0.0
        for origin, destination in zip(nodes, nodes[1:]):
            try:
                i = node_index[origin]
                j = node_index[destination]
            except KeyError as exc:
                raise KeyError(
                    f"Missing node {origin}->{destination} in distance matrix."
                ) from exc
            segment = matrix[i][j]
            if segment == float("inf"):
                raise ValueError(
                    f"Infinite distance between nodes {origin} and {destination}."
                )
            distance += float(segment)
        return distance

    # ------------------------------------------------------------------
    def _is_customer(self, node_id: int) -> bool:
        if node_id == self._depot_start:
            return False
        if self._depot_end is not None and node_id == self._depot_end:
            return False
        return node_id in self._customer_lookup

    @staticmethod
    def _task_key(task: DroneTask, index: int) -> int:
        return task.task_id if task.task_id is not None else index

    @staticmethod
    def _resolve_unit_cost(
        instance: InstanceManager,
        vehicle_type: str,
        override: float | None,
    ) -> float:
        if override is not None:
            return override
        spec = Evaluator._get_vehicle_spec(instance, vehicle_type)
        return spec.unit_cost if spec is not None else 1.0

    @staticmethod
    def _get_vehicle_spec(
        instance: InstanceManager, vehicle_type: str
    ) -> VehicleSpec | None:
        return instance.vehicle_specs.get(vehicle_type.lower())
