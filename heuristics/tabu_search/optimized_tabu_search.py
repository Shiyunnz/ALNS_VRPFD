"""Optimized Tabu Search implementation for VRPFD with performance improvements."""

from __future__ import annotations

import random
import time
from collections import deque
from typing import Optional, List, Tuple, Set
from dataclasses import dataclass

from alns_vrpfd.evaluation import Evaluator
from alns_vrpfd.model.solution import Solution


@dataclass
class TabuMove:
    """Efficient representation of a tabu move."""
    move_type: str  # 'relocate' or 'swap'
    customer: int
    from_route: int
    to_route: int
    position: int = -1  # For relocate moves

    def __hash__(self):
        return hash((self.move_type, self.customer, self.from_route, self.to_route, self.position))

    def __eq__(self, other):
        return (self.move_type, self.customer, self.from_route, self.to_route, self.position) == \
               (other.move_type, other.customer,
                other.from_route, other.to_route, other.position)


class OptimizedTabuSearch:
    """Optimized Tabu Search with performance improvements."""

    def __init__(
        self,
        *,
        evaluator: Evaluator,
        tabu_tenure: int = 15,
        max_iterations: int = 2000,
        max_neighbors: int = 50,  # Limit neighbors per iteration
        rng: Optional[random.Random] = None,
    ):
        self.evaluator = evaluator
        self.tabu_tenure = tabu_tenure
        self.max_iterations = max_iterations
        self.max_neighbors = max_neighbors
        self.rng = rng or random.Random()

    def run(self, initial: Solution, time_limit: float | None = None) -> Solution:
        """Run optimized Tabu Search."""
        current = initial.clone()
        current_eval = self.evaluator.evaluate_solution(current)
        current_cost = current_eval.total_cost

        best = current.clone()
        best_cost = current_cost

        tabu_set: Set[TabuMove] = set()
        start_time = time.perf_counter()

        for iteration in range(self.max_iterations):
            if time_limit and time.perf_counter() - start_time >= time_limit:
                break

            # Generate limited number of neighbors with pre-filtering
            neighbors_and_moves = self._generate_neighbors_efficient(current)

            if not neighbors_and_moves:
                break  # No valid neighbors

            best_neighbor = None
            best_neighbor_cost = float('inf')
            best_move = None

            # Evaluate neighbors
            for neighbor, move in neighbors_and_moves:
                # Check tabu status
                if move in tabu_set:
                    # Aspiration criterion: allow if better than best found
                    neighbor_eval = self.evaluator.evaluate_solution(neighbor)
                    if neighbor_eval.total_cost >= best_cost:
                        continue
                else:
                    neighbor_eval = self.evaluator.evaluate_solution(neighbor)

                if neighbor_eval.total_cost < best_neighbor_cost:
                    best_neighbor = neighbor
                    best_neighbor_cost = neighbor_eval.total_cost
                    best_move = move

            if best_neighbor is None:
                break  # No improvement possible

            # Update current solution
            current = best_neighbor
            current_cost = best_neighbor_cost

            # Update best solution
            if current_cost < best_cost:
                best = current.clone()
                best_cost = current_cost

            # Update tabu list
            tabu_set.add(best_move)
            if len(tabu_set) > self.tabu_tenure:
                # Remove random old move (simple FIFO alternative)
                tabu_set.pop()

        return best

    def _generate_neighbors_efficient(self, solution: Solution) -> List[Tuple[Solution, TabuMove]]:
        """Generate neighbors efficiently with early termination - truck routes only."""
        neighbors = []
        truck_routes = solution.truck_routes

        if len(truck_routes) < 2:
            return neighbors

        # Only generate relocate moves between truck routes
        for i, route in enumerate(truck_routes):
            # Skip depots
            customers = [node for node in route.nodes if node not in [0, 11]]
            if not customers:
                continue

            # Sample customers to reduce neighbors
            sampled_customers = self.rng.sample(
                customers, min(len(customers), 2))

            for cust in sampled_customers:
                for j, other_route in enumerate(truck_routes):
                    if i == j:
                        continue

                    # Sample insertion positions
                    positions = list(range(1, len(other_route.nodes)))
                    if len(positions) > 2:
                        positions = self.rng.sample(positions, 2)

                    for pos in positions:
                        # Create neighbor efficiently
                        neighbor = self._create_relocate_neighbor(
                            solution, i, cust, j, pos)
                        if neighbor:
                            move = TabuMove('relocate', cust, i, j, pos)
                            neighbors.append((neighbor, move))

                            if len(neighbors) >= self.max_neighbors:
                                return neighbors

        return neighbors

    def _create_relocate_neighbor(self, solution: Solution, from_route: int, customer: int,
                                  to_route: int, position: int) -> Optional[Solution]:
        """Create a neighbor by relocating a customer between routes."""
        try:
            neighbor = solution.clone()

            # Remove from source route
            source_route = neighbor.truck_routes[from_route]
            if customer not in source_route.nodes:
                return None
            source_route.nodes.remove(customer)

            # Insert into target route
            target_route = neighbor.truck_routes[to_route]
            if position >= len(target_route.nodes):
                return None
            target_route.nodes.insert(position, customer)

            # Quick feasibility check - ensure routes are not empty (except for depots)
            for route in neighbor.truck_routes:
                if len([n for n in route.nodes if n not in [0, 11]]) == 0:
                    return None  # Empty route

            return neighbor
        except:
            return None

    def _create_swap_neighbor(self, solution: Solution, route_idx: int, i: int, j: int) -> Optional[Solution]:
        """Create a neighbor by swapping two customers within a route."""
        try:
            neighbor = solution.clone()
            route = neighbor.truck_routes[route_idx]

            # Get customer positions (skip depot)
            customers = route.nodes[1:-1]
            if i >= len(customers) or j >= len(customers):
                return None

            # Swap customers
            customers[i], customers[j] = customers[j], customers[i]
            route.nodes = [route.nodes[0]] + customers + [route.nodes[-1]]

            return neighbor
        except:
            return None
