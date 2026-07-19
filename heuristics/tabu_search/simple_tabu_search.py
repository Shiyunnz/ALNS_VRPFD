"""Simple truck-only Tabu Search for quick testing."""

from __future__ import annotations

import random
import time
from collections import deque
from typing import Optional, List, Tuple

from alns_vrpfd.evaluation import Evaluator
from alns_vrpfd.model.solution import Solution


class SimpleTabuSearch:
    """Simple Tabu Search focusing only on truck route optimization."""

    def __init__(
        self,
        *,
        evaluator: Evaluator,
        tabu_tenure: int = 10,
        max_iterations: int = 1000,
        max_neighbors: int = 20,
        rng: Optional[random.Random] = None,
    ):
        self.evaluator = evaluator
        self.tabu_tenure = tabu_tenure
        self.max_iterations = max_iterations
        self.max_neighbors = max_neighbors
        self.rng = rng or random.Random()

    def run(self, initial: Solution, time_limit: float | None = None) -> Solution:
        """Run simple Tabu Search."""
        current = initial.clone()
        current_cost = self._evaluate_truck_only(current)

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
                    cost = self._evaluate_truck_only(neighbor)
                    if cost < best_neighbor_cost:
                        best_neighbor = neighbor
                        best_neighbor_cost = cost
                        best_move = move

            if best_neighbor is None:
                # Aspiration
                for neighbor, move in neighbors:
                    cost = self._evaluate_truck_only(neighbor)
                    if cost < best_neighbor_cost:
                        best_neighbor = neighbor
                        best_neighbor_cost = cost
                        best_move = move

            if best_neighbor is not None:
                current = best_neighbor
                current_cost = best_neighbor_cost
                tabu_list.append(best_move)

                if current_cost < best_cost:
                    best = current.clone()
                    best_cost = current_cost

        return best

    def _evaluate_truck_only(self, solution: Solution) -> float:
        """Evaluate only truck distance cost."""
        total_distance = 0
        for route in solution.truck_routes:
            nodes = route.nodes
            for i in range(len(nodes) - 1):
                # Simple distance calculation (placeholder)
                total_distance += abs(nodes[i+1] - nodes[i])
        return total_distance

    def _generate_truck_neighbors(self, solution: Solution) -> List[Tuple[Solution, tuple]]:
        """Generate simple truck route neighbors."""
        neighbors = []
        truck_routes = solution.truck_routes

        if len(truck_routes) < 2:
            return neighbors

        # Simple relocate between routes
        for i, route in enumerate(truck_routes):
            customers = [node for node in route.nodes if node not in [0, 11]]
            for cust in customers[:2]:  # Limit customers
                for j, other_route in enumerate(truck_routes):
                    if i != j:
                        for pos in [1, len(other_route.nodes)-1]:  # Simple positions
                            neighbor = solution.clone()
                            # Remove drone tasks to avoid validation issues
                            neighbor.drone_tasks.clear()

                            neighbor.truck_routes[i].nodes.remove(cust)
                            neighbor.truck_routes[j].nodes.insert(pos, cust)

                            move = (cust, i, j)
                            neighbors.append((neighbor, move))

                            if len(neighbors) >= self.max_neighbors:
                                return neighbors

        return neighbors
