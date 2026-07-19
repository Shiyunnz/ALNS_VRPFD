"""Solution representation for the refactored ALNS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .route import DroneTask, TruckRoute


@dataclass
class Solution:
    """Container for truck routes and assigned drone tasks."""

    truck_routes: List[TruckRoute] = field(default_factory=list)
    drone_tasks: List[DroneTask] = field(default_factory=list)

    @classmethod
    def empty(cls) -> "Solution":
        """Construct an empty solution with no routes or drone tasks."""
        return cls()

    def add_truck_route(self, route: TruckRoute) -> None:
        """Append a truck route to the current solution."""
        self.truck_routes.append(route)

    def add_drone_task(self, task: DroneTask) -> None:
        """Append a drone task to the current solution."""
        self.drone_tasks.append(task)

    def clone(self) -> "Solution":
        """Create a deep copy of the current solution."""
        return Solution(
            truck_routes=[route.clone() for route in self.truck_routes],
            drone_tasks=[task.clone() for task in self.drone_tasks],
        )

    def __str__(self) -> str:
        """Return a concise string showing truck routes and drone tasks."""
        truck_part = ", ".join(str(route.nodes) for route in self.truck_routes)
        drone_part = ", ".join(str(task) for task in self.drone_tasks)
        return f"Solution(truck_routes=[{truck_part}], drone_tasks=[{drone_part}])"

