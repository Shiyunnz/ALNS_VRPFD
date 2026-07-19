"""Truck-focused Tabu Search for VRPFD that preserves drone tasks."""

from __future__ import annotations

import random
import time
from collections import deque
from typing import Optional, List, Tuple, Set

from alns_vrpfd.evaluation import Evaluator
from alns_vrpfd.model.solution import Solution


class TruckFocusedTabuSearch:
    """Tabu Search that optimizes truck routes while preserving drone tasks."""

    def __init__(
        self,
        *,
        evaluator: Evaluator,
        tabu_tenure: int = 10,
        max_iterations: int = 1000,
        max_neighbors: int = 50,
        rng: Optional[random.Random] = None,
    ):
        self.evaluator = evaluator
        self.tabu_tenure = tabu_tenure
        self.max_iterations = max_iterations
        self.max_neighbors = max_neighbors
        self.rng = rng or random.Random()

    def run(self, initial: Solution, time_limit: float | None = None) -> Solution:
        """Run truck-focused Tabu Search."""
        current = initial.clone()
        current_cost = self.evaluator.evaluate_solution(current).total_cost

        best = current.clone()
        best_cost = current_cost

        tabu_list = deque(maxlen=self.tabu_tenure)
        start_time = time.perf_counter()

        for iteration in range(self.max_iterations):
            if time_limit and time.perf_counter() - start_time >= time_limit:
                break

            # Generate neighbors
            neighbors = self._generate_truck_neighbors(current)

            if not neighbors:
                break

            best_neighbor = None
            best_neighbor_cost = float('inf')

            for neighbor, move in neighbors:
                if move not in tabu_list:
                    try:
                        cost = self.evaluator.evaluate_solution(
                            neighbor).total_cost
                        if cost < best_neighbor_cost:
                            best_neighbor = neighbor
                            best_neighbor_cost = cost
                            best_move = move
                    except Exception:
                        # Skip invalid neighbors
                        continue

            if best_neighbor is not None:
                current = best_neighbor
                current_cost = best_neighbor_cost
                tabu_list.append(best_move)

                if current_cost < best_cost:
                    best = current.clone()
                    best_cost = current_cost

        return best

    def _generate_truck_neighbors(self, solution: Solution) -> List[Tuple[Solution, tuple]]:
        """Generate truck route neighbors that preserve drone task validity."""
        neighbors = []
        truck_routes = solution.truck_routes

        if len(truck_routes) < 2:
            return neighbors

        # Get nodes used by drone tasks to avoid moving them
        drone_nodes: Set[int] = set()
        for task in solution.drone_tasks:
            drone_nodes.add(task.launch_node)
            drone_nodes.add(task.retrieve_node)
            drone_nodes.update(task.customers())

        # Generate relocate moves between routes
        for i, route in enumerate(truck_routes):
            customers = [node for node in route.nodes
                         if node not in [0, 11] and node not in drone_nodes]
            for cust in customers:
                for j, other_route in enumerate(truck_routes):
                    if i != j:
                        # Try inserting at different positions
                        for pos in range(1, len(other_route.nodes)):
                            neighbor = solution.clone()
                            neighbor.truck_routes[i].nodes.remove(cust)
                            neighbor.truck_routes[j].nodes.insert(pos, cust)

                            move = (cust, i, j, pos)
                            neighbors.append((neighbor, move))

                            if len(neighbors) >= self.max_neighbors:
                                return neighbors

        # Generate swap moves within routes
        for route in truck_routes:
            customers = [node for node in route.nodes
                         if node not in [0, 11] and node not in drone_nodes]
            for i in range(len(customers)):
                for j in range(i + 1, len(customers)):
                    if customers[i] != customers[j]:
                        neighbor = solution.clone()
                        # Find positions in route
                        route_nodes = neighbor.truck_routes[truck_routes.index(
                            route)].nodes
                        pos1 = route_nodes.index(customers[i])
                        pos2 = route_nodes.index(customers[j])
                        # Swap
                        route_nodes[pos1], route_nodes[pos2] = route_nodes[pos2], route_nodes[pos1]

                        move = ('swap', customers[i], customers[j])
                        neighbors.append((neighbor, move))

                        if len(neighbors) >= self.max_neighbors:
                            return neighbors

        return neighbors
