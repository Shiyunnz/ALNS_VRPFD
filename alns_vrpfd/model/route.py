"""Route abstractions: base class + TruckRoute + DroneTask."""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
import math
from dataclasses import dataclass
from typing import (
    AbstractSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
)

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    from alns_vrpfd.evaluation.energy import DroneEnergyModel

__all__ = [
    "Route",
    "TruckRoute",
    "DroneTask",
    "DroneTaskTiming",
    "DroneTaskContext",
]


@dataclass(frozen=True)
class DroneTaskTiming:
    """Timing data for a drone task under lazy evaluation assumptions."""
    """Timing data for a drone task under lazy evaluation assumptions."""

    launch_time: float
    customer_arrival_times: Mapping[int, float]
    retrieve_time: float


@dataclass(frozen=True)
class DroneTaskContext:
    """Contextual information required to validate a drone task."""
    """Contextual information required to validate a drone task."""

    valid_nodes: Optional[AbstractSet[int]] = None
    valid_trucks: Optional[AbstractSet[int]] = None
    served_customers: Optional[AbstractSet[int]] = None
    truck_routes: Optional[Mapping[int, Sequence[int]]] = None
    truck_arrival_times: Optional[Mapping[int, Mapping[int, float]]] = None
    wait_max: float = math.inf
    drone_capacity: Optional[float] = None
    customer_demands: Optional[Mapping[int, float]] = None
    customer_latest_times: Optional[Mapping[int, float]] = None
    drone_schedule: Optional[Mapping[int,
                                     Sequence[Tuple[float, float]]]] = None
    timing: Optional[DroneTaskTiming] = None
    payload_tolerance: float = 1e-6
    time_tolerance: float = 1e-6
    drone_energy_capacity: float | Mapping[int, float] | None = None
    energy_uncertainty_budget: float | Mapping[int, float] | None = None
    energy_deviation_rate: float = 0.1
    energy_model: Optional["DroneEnergyModel"] = None
    energy_tolerance: float = 1e-6


def _intervals_overlap(
    first: Tuple[float, float], second: Tuple[float, float], tolerance: float
) -> bool:
    """Return True when the open intervals overlap beyond a tolerance."""
    """Return True when the open intervals overlap beyond a tolerance."""

    start_a, end_a = first
    start_b, end_b = second
    return (start_a < end_b - tolerance) and (start_b < end_a - tolerance)


def _value_for_drone(
    value: float | Mapping[int, float] | None, drone_id: int
) -> float | None:
    """Return scalar value resolved for a specific drone identifier."""
    """Return scalar value resolved for a specific drone identifier."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        return value.get(drone_id)
    return value


class Route(ABC):
    """Base representation of an ordered sequence of nodes."""
    """Base representation of an ordered sequence of nodes."""

    def __init__(self, route_id: int, nodes: Sequence[int]) -> None:
        self.id = route_id
        self.nodes: List[int] = list(nodes)
        if len(self.nodes) < 2:
            raise ValueError(
                "A route must contain at least a start and end node.")

    def customers(self) -> List[int]:
        """Return the customer nodes between the start and end points."""
        """Return the customer nodes between the start and end points."""
        return self.nodes[1:-1]

    def insert_customer(self, index: int, customer: int) -> None:
        """Insert a customer at the given customer index (0-based)."""
        """Insert a customer at the given customer index (0-based)."""
        # ：
        if customer in self.nodes:
            return
        count = len(self.customers())
        if index < 0 or index > count:
            raise IndexError("Customer index out of range for insertion.")
        self.nodes.insert(index + 1, customer)
        self._on_customers_changed()

    def remove_customer(self, customer: int) -> None:
        """Remove the specified customer from the route."""
        """Remove the specified customer from the route."""
        try:
            idx = self.nodes.index(customer, 1, len(self.nodes) - 1)
        except ValueError as exc:
            raise ValueError("Customer not present on route.") from exc
        self.nodes.pop(idx)
        self._on_customers_changed()

    def swap_customers(self, first_index: int, second_index: int) -> None:
        """Swap the order of two customers identified by their indices."""
        """Swap the order of two customers identified by their indices."""
        count = len(self.customers())
        if not (0 <= first_index < count) or not (0 <= second_index < count):
            raise IndexError("Customer index out of range for swap.")
        if first_index == second_index:
            return
        a = first_index + 1
        b = second_index + 1
        self.nodes[a], self.nodes[b] = self.nodes[b], self.nodes[a]
        self._on_customers_changed()

    def total_distance(self, dist_matrix: Sequence[Sequence[float]]) -> float:
        """Compute the total distance using a distance matrix."""
        """Compute the total distance using a distance matrix."""
        distance = 0.0
        for start, end in zip(self.nodes, self.nodes[1:]):
            distance += dist_matrix[start][end]
        return distance

    def clone(self) -> "Route":
        """Create a deep copy of the route."""
        """Create a deep copy of the route."""
        return copy.deepcopy(self)

    @abstractmethod
    def is_feasible(self, context: Optional[object] = None) -> bool:
        """Return whether the current route satisfies its constraints."""
        """Return whether the current route satisfies its constraints."""

    def _on_customers_changed(self) -> None:
        """Hook called after customer-related structure updates."""
        """Hook called after customer-related structure updates."""
        return


class TruckRoute(Route):
    """Represents a truck tour with capacity tracking."""
    """Represents a truck tour with capacity tracking."""

    def __init__(
        self,
        route_id: int,
        nodes: Sequence[int],
        capacity: float,
        current_load: float = 0.0,
    ) -> None:
        super().__init__(route_id=route_id, nodes=nodes)
        self.capacity = capacity
        self.current_load = current_load

    def is_feasible(self, context: Optional[object] = None) -> bool:
        """Return True when current load does not exceed capacity."""
        """Return True when current load does not exceed capacity."""
        return self.current_load <= self.capacity

    def check_capacity(self, demands: Mapping[int, float]) -> bool:
        """Verify that customer demands do not exceed truck capacity."""
        """Verify that customer demands do not exceed truck capacity."""
        route_demand = sum(demands.get(c, 0.0) for c in self.customers())
        return (route_demand + self.current_load) <= self.capacity


class DroneTask(Route):
    """Describe a full drone task from launch to landing through customers."""
    """Describe a full drone task from launch to landing through customers."""

    def __init__(
        self,
        drone_id: int,
        launch_truck: Optional[int],
        launch_node: int,
        customers: Sequence[int],
        land_truck: Optional[int],
        retrieve_node: int,
        payloads: Optional[Sequence[float]] = None,
        task_id: Optional[int] = None,
    ) -> None:
        nodes = [launch_node, *customers, retrieve_node]
        super().__init__(route_id=task_id if task_id is not None else -1, nodes=nodes)
        self.task_id = task_id
        self.drone_id = drone_id
        self.launch_truck = launch_truck
        self.land_truck = land_truck
        self.payloads: List[float] = self._initialise_payloads(payloads)

    def __repr__(self) -> str:
        """Return a human readable representation of the task."""
        """Return a human readable representation of the task."""
        launch_str = f"depot@{self.launch_node}" if self.launch_truck is None else f"T{self.launch_truck}@{self.launch_node}"
        retrieve_str = f"depot@{self.retrieve_node}" if self.land_truck is None else f"T{self.land_truck}@{self.retrieve_node}"
        return (
            f"Drone {self.drone_id} Task {self.task_id}: "
            f"launch {launch_str} -> {self.customers()} "
            f"-> retrieve {retrieve_str}"
        )

    @property
    def launch_node(self) -> int:
        """Return the launch node for the mission."""
        """Return the launch node for the mission."""
        return self.nodes[0]

    @property
    def retrieve_node(self) -> int:
        """Return the retrieval node for the mission."""
        """Return the retrieval node for the mission."""
        return self.nodes[-1]

    def is_feasible(self, context: Optional[DroneTaskContext] = None) -> bool:
        """Return True when no feasibility violations are detected."""
        """Return True when no feasibility violations are detected."""

        return not self.feasibility_errors(context)

    def feasibility_errors(
        self, context: Optional[DroneTaskContext] = None
    ) -> List[str]:
        """Return a list of feasibility violations for the task."""
        """Return a list of feasibility violations for the task."""

        ctx = context or DroneTaskContext()
        errors: List[str] = []
        customers = self.customers()

        if not customers:
            errors.append("DroneTask must include at least one customer.")

        seen_customers: set[int] = set()
        duplicate_customers: set[int] = set()
        for customer in customers:
            if customer in seen_customers:
                duplicate_customers.add(customer)
            seen_customers.add(customer)
        if duplicate_customers:
            errors.append(
                "Duplicate customers within task: "
                f"{sorted(duplicate_customers)}."
            )

        if self.launch_node == self.retrieve_node:
            errors.append(
                "Launch and retrieval nodes must differ for a DroneTask.")

        node_sequence = [self.launch_node, *customers, self.retrieve_node]
        if ctx.valid_nodes is not None:
            invalid_nodes = [
                node for node in node_sequence if node not in ctx.valid_nodes]
            if invalid_nodes:
                errors.append(
                    "Node(s) outside allowed set: "
                    f"{sorted(set(invalid_nodes))}."
                )

        if ctx.served_customers is not None:
            global_conflicts = set(customers).intersection(
                ctx.served_customers)
            if global_conflicts:
                errors.append(
                    "Customers already served elsewhere: "
                    f"{sorted(global_conflicts)}."
                )

        if ctx.valid_trucks is not None:
            if self.launch_truck not in ctx.valid_trucks:
                errors.append(
                    f"Launch truck {self.launch_truck} is not in the valid truck set."
                )
            if self.land_truck not in ctx.valid_trucks:
                errors.append(
                    f"Retrieval truck {self.land_truck} is not in the valid truck set."
                )

        if ctx.truck_routes is not None:
            launch_route = ctx.truck_routes.get(self.launch_truck)
            if launch_route is None or self.launch_node not in launch_route:
                errors.append(
                    f"Launch node {self.launch_node} is not visited by truck "
                    f"{self.launch_truck}."
                )
            retrieve_route = ctx.truck_routes.get(self.land_truck)
            if retrieve_route is None or self.retrieve_node not in retrieve_route:
                errors.append(
                    f"Retrieve node {self.retrieve_node} is not visited by truck "
                    f"{self.land_truck}."
                )

        payloads = self.payloads
        payload_tol = ctx.payload_tolerance
        for idx, payload in enumerate(payloads):
            if payload < -payload_tol:
                errors.append(
                    f"Payload at segment {idx} is negative ({payload})."
                )
            if (
                ctx.drone_capacity is not None
                and payload - ctx.drone_capacity > payload_tol
            ):
                errors.append(
                    f"Payload at segment {idx} ({payload}) exceeds drone capacity "
                    f"{ctx.drone_capacity}."
                )

        for index, (current_payload, next_payload) in enumerate(
            zip(payloads, payloads[1:]), start=0
        ):
            segment_customer = customers[index] if index < len(
                customers) else None
            if ctx.customer_demands is not None and segment_customer is not None:
                demand = ctx.customer_demands.get(segment_customer)
                if demand is None:
                    errors.append(
                        f"Missing demand data for customer {segment_customer}."
                    )
                    continue
                expected_next = current_payload - demand
                if not math.isclose(
                    next_payload,
                    expected_next,
                    rel_tol=payload_tol,
                    abs_tol=payload_tol,
                ):
                    errors.append(
                        "Payload drop does not match demand for customer "
                        f"{segment_customer}: expected {expected_next}, "
                        f"found {next_payload}."
                    )
            elif next_payload - current_payload > payload_tol:
                errors.append(
                    "Payload increases between segments "
                    f"{index} and {index + 1}."
                )

        timing = ctx.timing
        time_tol = ctx.time_tolerance
        if timing is not None:
            launch_time = timing.launch_time
            retrieve_time = timing.retrieve_time
            if retrieve_time + time_tol < launch_time:
                errors.append(
                    "Retrieve time must not precede launch time for the task."
                )

            if ctx.truck_arrival_times is not None:
                launch_times = ctx.truck_arrival_times.get(self.launch_truck)
                if launch_times is None or self.launch_node not in launch_times:
                    errors.append(
                        "Missing truck arrival information for launch node "
                        f"{self.launch_node} on truck {self.launch_truck}."
                    )
                else:
                    truck_arrival = launch_times[self.launch_node]
                    if launch_time + time_tol < truck_arrival:
                        errors.append(
                            "Drone launches before truck arrival at node "
                            f"{self.launch_node}: drone {launch_time}, truck {truck_arrival}."
                        )

            current_time = launch_time
            for customer in customers:
                customer_time = timing.customer_arrival_times.get(customer)
                if customer_time is None:
                    errors.append(
                        f"Missing arrival time for customer {customer}."
                    )
                    continue
                if customer_time + time_tol < current_time:
                    errors.append(
                        f"Arrival at customer {customer} ({customer_time}) "
                        f"precedes previous leg completion ({current_time})."
                    )
                if (
                    ctx.customer_latest_times is not None
                    and customer in ctx.customer_latest_times
                    and customer_time - ctx.customer_latest_times[customer] > time_tol
                ):
                    latest = ctx.customer_latest_times[customer]
                    errors.append(
                        f"Customer {customer} served after latest time "
                        f"{latest}: arrival {customer_time}."
                    )
                current_time = max(current_time, customer_time)

            if current_time - retrieve_time > time_tol:
                errors.append(
                    "Retrieve time occurs before the last customer delivery completes."
                )

            if ctx.truck_arrival_times is not None:
                retrieve_times = ctx.truck_arrival_times.get(self.land_truck)
                if retrieve_times is None or self.retrieve_node not in retrieve_times:
                    errors.append(
                        "Missing truck arrival information for retrieve node "
                        f"{self.retrieve_node} on truck {self.land_truck}."
                    )
                else:
                    truck_arrival = retrieve_times[self.retrieve_node]
                    if abs(retrieve_time - truck_arrival) - ctx.wait_max > time_tol:
                        errors.append(
                            "Truck-drone rendezvous exceeds wait limit at node "
                            f"{self.retrieve_node}: |{retrieve_time} - {truck_arrival}| > {ctx.wait_max}."
                        )

            if ctx.drone_schedule is not None:
                interval = (timing.launch_time, timing.retrieve_time)
                schedule = ctx.drone_schedule.get(self.drone_id, ())
                for window in schedule:
                    if _intervals_overlap(interval, window, time_tol):
                        errors.append(
                            f"Drone {self.drone_id} task window {interval} overlaps "
                            f"with existing window {window}."
                        )

        energy_model = ctx.energy_model
        if customers and energy_model is not None and timing is not None:
            budget_value = _value_for_drone(
                ctx.energy_uncertainty_budget, self.drone_id)
            capacity_value = _value_for_drone(
                ctx.drone_energy_capacity, self.drone_id)
            deviation_rate = ctx.energy_deviation_rate
            if deviation_rate < 0:
                errors.append("Energy deviation rate must be non-negative.")
            else:
                try:
                    from alns_vrpfd.evaluation.robustness import assess_drone_task_energy

                    assessment = assess_drone_task_energy(
                        task=self,
                        timing=timing,
                        energy_model=energy_model,
                        deviation_rate=deviation_rate,
                        uncertainty_budget=budget_value or 0.0,
                        capacity=capacity_value,
                        tolerance=ctx.energy_tolerance,
                        time_tolerance=ctx.time_tolerance,
                    )
                except ValueError as exc:
                    errors.append(f"Energy assessment failed: {exc}")
                else:
                    if capacity_value is not None and not assessment.feasible:
                        errors.append(
                            "Energy budget violation under uncertainty: "
                            f"worst-case consumption {assessment.worst_case_energy:.3f} "
                            f"exceeds capacity {capacity_value:.3f}."
                        )

        return errors

    def _initialise_payloads(self, payloads: Optional[Sequence[float]]) -> List[float]:
        """Normalise payload data and align with customer count."""
        """Normalise payload data and align with customer count."""
        expected = len(self.customers()) + 1
        if payloads is None:
            return [0.0] * expected
        payload_list = list(payloads)
        if len(payload_list) != expected:
            raise ValueError(
                "Payloads must contain len(customers) + 1 entries, "
                f"got {len(payload_list)} while expecting {expected}."
            )
        return payload_list

    def _on_customers_changed(self) -> None:
        """Resize payloads when customer sequence length changes."""
        """Resize payloads when customer sequence length changes."""
        expected = len(self.customers()) + 1
        if len(self.payloads) < expected:
            self.payloads.extend([0.0] * (expected - len(self.payloads)))
        elif len(self.payloads) > expected:
            del self.payloads[expected:]
