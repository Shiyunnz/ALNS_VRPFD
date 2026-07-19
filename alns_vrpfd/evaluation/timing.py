"""Lazy timing computations for truck routes and drone tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping, Sequence, Tuple

from alns_vrpfd.model.route import DroneTask, DroneTaskTiming, TruckRoute

__all__ = [
    "TruckRouteTiming",
    "RendezvousResult",
    "TimingCalculator",
]


@dataclass(frozen=True)
class TruckRouteTiming:
    """Arrival/departure timestamps for a truck route."""

    arrival_times: Mapping[int, float]
    departure_times: Mapping[int, float]
    total_duration: float


@dataclass(frozen=True)
class RendezvousResult:
    """Outcome of a truck-drone meeting feasibility check."""

    feasible: bool
    deviation: float
    tolerance: float
    truck_arrival: float
    drone_retrieve: float


class TimingCalculator:
    """Provide on-demand timing data with per-evaluation caching."""

    def __init__(
        self,
        *,
        node_index: Mapping[int, int],
        truck_time_matrix: Sequence[Sequence[float]],
        drone_time_matrix: Sequence[Sequence[float]],
        service_times: Mapping[int, float] | None = None,
        drone_service_times: Mapping[int, float] | None = None,
    ) -> None:
        self._node_index = node_index
        self._truck_time_matrix = truck_time_matrix
        self._drone_time_matrix = drone_time_matrix
        self._service_times = service_times or {}
        self._drone_service_times = drone_service_times or {}
        self._truck_cache: MutableMapping[tuple[int, tuple[int, ...]], TruckRouteTiming] = {}
        self._drone_cache: MutableMapping[tuple[int, tuple[int, ...], float], DroneTaskTiming] = {}

    # ------------------------------------------------------------------
    def truck_timing(
        self,
        route: TruckRoute,
        min_departure_times: Mapping[int, float] | None = None,
    ) -> TruckRouteTiming:
        """Return timing for the truck route, optionally enforcing minimum departures."""
        key = (route.id, tuple(route.nodes))
        
        # Only use cache if no custom constraints
        if min_departure_times is None:
            cached = self._truck_cache.get(key)
            if cached is not None:
                return cached
        
        min_deps = min_departure_times or {}

        arrival: Dict[int, float] = {}
        departure: Dict[int, float] = {}

        nodes = route.nodes
        if not nodes:
            timing = TruckRouteTiming(arrival, departure, 0.0)
            if min_departure_times is None:
                self._truck_cache[key] = timing
            return timing

        current_time = 0.0
        first_node = nodes[0]
        arrival[first_node] = current_time
        
        # Service then Wait
        current_time += self._service_times.get(first_node, 0.0)
        current_time = max(current_time, min_deps.get(first_node, 0.0))
        departure[first_node] = current_time

        for prev_node, next_node in zip(nodes, nodes[1:]):
            travel_time = self._travel_time(self._truck_time_matrix, prev_node, next_node)
            current_time += travel_time
            arrival[next_node] = current_time
            
            current_time += self._service_times.get(next_node, 0.0)
            current_time = max(current_time, min_deps.get(next_node, 0.0))
            departure[next_node] = current_time

        timing = TruckRouteTiming(arrival_times=arrival, departure_times=departure, total_duration=current_time)
        
        if min_departure_times is None:
            self._truck_cache[key] = timing
            
        return timing

    def drone_timing(
        self,
        task: DroneTask,
        *,
        launch_time: float,
    ) -> DroneTaskTiming:
        """Return cached timing for the drone task using the provided launch time."""
        task_key = task.task_id if task.task_id is not None else id(task)
        nodes_signature = tuple(task.nodes)
        launch_signature = float(round(launch_time, 6))
        key = (task_key, nodes_signature, launch_signature)
        cached = self._drone_cache.get(key)
        if cached is not None:
            return cached

        current_time = launch_time
        arrival_times: Dict[int, float] = {}

        nodes = [task.launch_node, *task.customers(), task.retrieve_node]
        for prev_node, next_node in zip(nodes, nodes[1:]):
            travel_time = self._travel_time(self._drone_time_matrix, prev_node, next_node)
            current_time += travel_time
            if next_node != task.retrieve_node:
                arrival_times[next_node] = current_time
                current_time += self._drone_service_times.get(next_node, 0.0)

        retrieve_time = current_time
        timing = DroneTaskTiming(
            launch_time=launch_time,
            customer_arrival_times=arrival_times,
            retrieve_time=retrieve_time,
        )
        self._drone_cache[key] = timing
        return timing

    # ------------------------------------------------------------------
    def rendezvous(
        self,
        *,
        drone_timing: DroneTaskTiming,
        truck_retrieve_arrival: float,
        tolerance: float,
    ) -> RendezvousResult:
        """Compare drone retrieval against truck arrival with tolerance."""
        deviation = abs(drone_timing.retrieve_time - truck_retrieve_arrival)
        feasible = deviation <= tolerance
        return RendezvousResult(
            feasible=feasible,
            deviation=deviation,
            tolerance=tolerance,
            truck_arrival=truck_retrieve_arrival,
            drone_retrieve=drone_timing.retrieve_time,
        )

    # ------------------------------------------------------------------
    def _travel_time(
        self,
        matrix: Sequence[Sequence[float]],
        origin: int,
        destination: int,
    ) -> float:
        try:
            i = self._node_index[origin]
            j = self._node_index[destination]
        except KeyError as exc:
            raise KeyError(f"Missing node index for timing computation: {origin}->{destination}.") from exc

        travel_time = matrix[i][j]
        if travel_time == float("inf"):
            raise ValueError(f"Infinite travel time between nodes {origin} and {destination}.")
        return float(travel_time)
