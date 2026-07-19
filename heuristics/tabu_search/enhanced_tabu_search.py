"""Enhanced Tabu Search implementation for VRPFD with multi-level neighborhoods."""

from __future__ import annotations

import random
import time
from collections import deque, defaultdict
from typing import Optional, List, Tuple, Dict, Set
from dataclasses import dataclass
from enum import Enum

from alns_vrpfd.evaluation import Evaluator
from alns_vrpfd.model.solution import Solution
from alns_vrpfd.model.route import TruckRoute, DroneTask
from alns_vrpfd.instance.manager import InstanceManager


class MoveType(Enum):
    """Types of neighborhood moves."""
    RELOCATE_CUSTOMER = "relocate_customer"
    SWAP_CUSTOMERS = "swap_customers"
    TWO_OPT_TRUCK = "two_opt_truck"
    DRONE_INSERTION = "drone_insertion"
    DRONE_EXTRACTION = "drone_extraction"
    DRONE_TRUCK_REBALANCE = "drone_truck_rebalance"
    SORTIE_LENGTH_ADJUST = "sortie_length_adjust"


@dataclass
class TabuMove:
    """Represents a tabu move with attributes for advanced tabu mechanisms."""
    move_type: MoveType
    attributes: Tuple  # Hashable attributes for tabu list
    timestamp: int
    frequency: int = 1

    def __hash__(self):
        return hash(self.attributes)

    def __eq__(self, other):
        return self.attributes == other.attributes


class FeasibilityChecker:
    """Advanced feasibility checking for VRPFD constraints."""

    def __init__(self, instance: InstanceManager):
        self.instance = instance
        self.demands = instance.customer_manager.demands()
        self.time_windows = {}
        self.truck_distances = instance.distance_matrix("truck")
        self.drone_distances = instance.distance_matrix("drone")

        # Pre-compute time windows for efficiency
        customer_ids = instance.customer_manager.customer_ids()
        for customer in customer_ids:
            try:
                self.time_windows[customer] = instance.customer_manager.time_window(
                    customer)
            except KeyError:
                continue

    def check_truck_capacity(self, route: TruckRoute) -> bool:
        """Check if truck route respects capacity constraints."""
        capacity = self.instance.vehicle_specs['truck'].capacity
        depot_start = self.instance.customer_manager.depot_start
        depot_end = self.instance.customer_manager.depot_end
        total_demand = sum(self.demands.get(node, 0) for node in route.nodes
                           if node not in (depot_start, depot_end))
        return total_demand <= capacity

    def check_drone_endurance(self, drone_task: DroneTask) -> bool:
        """Check if drone sortie respects endurance constraints."""
        endurance = self.instance.vehicle_specs['drone'].endurance
        # Calculate total distance for the sortie
        total_distance = 0
        nodes = [drone_task.launch_node] + \
            list(drone_task.customers()) + [drone_task.retrieve_node]
        for i in range(len(nodes) - 1):
            total_distance += self.drone_distances[nodes[i]][nodes[i + 1]]
        return total_distance <= endurance

    def check_time_windows(self, route: TruckRoute) -> bool:
        """Check if route respects time window constraints."""
        # Simplified time window check - could be enhanced
        return True  # Placeholder

    def is_feasible_solution(self, solution: Solution) -> bool:
        """Check if entire solution is feasible."""
        # Check truck routes
        for route in solution.truck_routes:
            if not (self.check_truck_capacity(route) and self.check_time_windows(route)):
                return False

        # Check drone tasks
        for task in solution.drone_tasks:
            if not self.check_drone_endurance(task):
                return False

        return True


class EnhancedTabuSearch:
    """Enhanced Tabu Search with multi-level neighborhoods and advanced mechanisms."""

    def __init__(
        self,
        *,
        instance: InstanceManager,
        evaluator: Evaluator,
        tabu_tenure: int = 15,
        max_iterations: int = 2000,
        min_tabu_tenure: int = 5,
        max_tabu_tenure: int = 30,
        adaptive_tabu: bool = True,
        rng: Optional[random.Random] = None,
    ):
        self.instance = instance
        self.evaluator = evaluator
        self.base_tabu_tenure = tabu_tenure
        self.tabu_tenure = tabu_tenure
        self.max_iterations = max_iterations
        self.min_tabu_tenure = min_tabu_tenure
        self.max_tabu_tenure = max_tabu_tenure
        self.adaptive_tabu = adaptive_tabu
        self.rng = rng or random.Random()

        # Advanced components
        self.feasibility_checker = FeasibilityChecker(instance)
        self.move_frequencies = defaultdict(int)
        self.solution_history = deque(maxlen=50)
        self.diversification_counter = 0

        # Neighborhood weights for adaptive selection
        self.neighborhood_weights = {
            MoveType.RELOCATE_CUSTOMER: 1.0,
            MoveType.SWAP_CUSTOMERS: 1.0,
            MoveType.TWO_OPT_TRUCK: 0.8,
            MoveType.DRONE_INSERTION: 0.6,
            MoveType.DRONE_EXTRACTION: 0.6,
            MoveType.DRONE_TRUCK_REBALANCE: 0.4,
            MoveType.SORTIE_LENGTH_ADJUST: 0.3,
        }

    def run(self, initial: Solution, time_limit: float | None = None) -> Solution:
        """Run enhanced Tabu Search with advanced mechanisms."""
        current = initial.clone()
        current_cost = self.evaluator.evaluate_solution(current).total_cost
        best = current.clone()
        best_cost = current_cost

        tabu_list = deque(maxlen=self.max_tabu_tenure)
        tabu_attributes = set()  # For attribute-based tabu
        start_time = time.perf_counter()
        iteration = 0

        while iteration < self.max_iterations:
            if time_limit and time.perf_counter() - start_time >= time_limit:
                break

            # Adaptive tabu tenure adjustment
            if self.adaptive_tabu and iteration % 50 == 0:
                self._adapt_tabu_tenure()

            # Generate and filter neighbors
            neighbors = self._generate_filtered_neighbors(current)

            if not neighbors:
                # Diversification mechanism
                current = self._diversify_solution(current)
                continue

            best_neighbor = None
            best_neighbor_cost = float('inf')
            best_move = None

            # Evaluate neighbors with aspiration criteria
            for neighbor, move in neighbors:
                move_hash = hash(move.attributes)

                # Check tabu status (both move-based and attribute-based)
                is_tabu = (move in tabu_list or
                           any(attr in tabu_attributes for attr in move.attributes))

                # Aspiration criteria
                if is_tabu and not self._check_aspiration_criteria(move, neighbor, best_cost):
                    continue

                try:
                    cost = self.evaluator.evaluate_solution(
                        neighbor).total_cost
                    if cost < best_neighbor_cost:
                        best_neighbor = neighbor
                        best_neighbor_cost = cost
                        best_move = move
                except:
                    continue

            if best_neighbor is None:
                # If no good neighbor found, allow tabu moves with aspiration
                for neighbor, move in neighbors:
                    try:
                        cost = self.evaluator.evaluate_solution(
                            neighbor).total_cost
                        if cost < best_neighbor_cost:
                            best_neighbor = neighbor
                            best_neighbor_cost = cost
                            best_move = move
                    except:
                        continue

            if best_neighbor is not None:
                current = best_neighbor
                current_cost = best_neighbor_cost

                # Update tabu lists
                tabu_list.append(best_move)
                for attr in best_move.attributes:
                    tabu_attributes.add(attr)

                # Maintain tabu list size
                while len(tabu_attributes) > self.tabu_tenure * 2:
                    oldest_move = tabu_list.popleft()
                    for attr in oldest_move.attributes:
                        tabu_attributes.discard(attr)

                # Update best solution
                if current_cost < best_cost:
                    best = current.clone()
                    best_cost = current_cost
                    self.diversification_counter = 0  # Reset diversification
                else:
                    self.diversification_counter += 1

                # Frequency-based diversification
                self.move_frequencies[best_move.move_type] += 1

                # Store solution for restart mechanism
                self.solution_history.append(current.clone())

            # Periodic diversification check
            if self.diversification_counter > 100:
                current = self._diversify_solution(current)
                self.diversification_counter = 0

            iteration += 1

        return best

    def _generate_filtered_neighbors(self, solution: Solution) -> List[Tuple[Solution, TabuMove]]:
        """Generate neighbors with feasibility filtering."""
        neighbors = []

        # Adaptive neighborhood selection based on weights
        neighborhood_types = list(self.neighborhood_weights.keys())
        weights = [self.neighborhood_weights[nt] for nt in neighborhood_types]

        selected_types = self.rng.choices(
            neighborhood_types,
            weights=weights,
            # Select 3 neighborhood types per iteration
            k=min(3, len(neighborhood_types))
        )

        for move_type in selected_types:
            type_neighbors = self._generate_neighbors_by_type(
                solution, move_type)

            # Filter feasible neighbors only
            for neighbor, move in type_neighbors:
                if self.feasibility_checker.is_feasible_solution(neighbor):
                    neighbors.append((neighbor, move))

        # Limit total neighbors for efficiency
        if len(neighbors) > 200:
            neighbors = self.rng.sample(neighbors, 200)

        return neighbors

    def _generate_neighbors_by_type(self, solution: Solution, move_type: MoveType) -> List[Tuple[Solution, TabuMove]]:
        """Generate neighbors of specific type."""
        neighbors = []

        if move_type == MoveType.RELOCATE_CUSTOMER:
            neighbors.extend(self._generate_relocate_moves(solution))
        elif move_type == MoveType.SWAP_CUSTOMERS:
            neighbors.extend(self._generate_swap_moves(solution))
        elif move_type == MoveType.TWO_OPT_TRUCK:
            neighbors.extend(self._generate_two_opt_moves(solution))
        elif move_type == MoveType.DRONE_INSERTION:
            neighbors.extend(self._generate_drone_insertion_moves(solution))
        elif move_type == MoveType.DRONE_EXTRACTION:
            neighbors.extend(self._generate_drone_extraction_moves(solution))
        elif move_type == MoveType.DRONE_TRUCK_REBALANCE:
            neighbors.extend(self._generate_rebalance_moves(solution))
        elif move_type == MoveType.SORTIE_LENGTH_ADJUST:
            neighbors.extend(self._generate_sortie_adjust_moves(solution))

        return neighbors

    def _generate_relocate_moves(self, solution: Solution) -> List[Tuple[Solution, TabuMove]]:
        """Generate customer relocation moves between routes."""
        neighbors = []

        for i, route in enumerate(solution.truck_routes):
            customers = [node for node in route.nodes if node not in [0, 11]]
            for cust in customers:
                for j, other_route in enumerate(solution.truck_routes):
                    if i != j:
                        for pos in range(1, len(other_route.nodes)):
                            neighbor = solution.clone()
                            neighbor.truck_routes[i].nodes.remove(cust)
                            neighbor.truck_routes[j].nodes.insert(pos, cust)

                            move = TabuMove(
                                move_type=MoveType.RELOCATE_CUSTOMER,
                                attributes=(cust, i, j, pos),
                                timestamp=0
                            )
                            neighbors.append((neighbor, move))

        return neighbors

    def _generate_swap_moves(self, solution: Solution) -> List[Tuple[Solution, TabuMove]]:
        """Generate customer swap moves within and between routes."""
        neighbors = []

        # Intra-route swaps
        for route_idx, route in enumerate(solution.truck_routes):
            customers = route.nodes[1:-1]
            if len(customers) >= 2:
                for i in range(len(customers)):
                    for j in range(i + 1, len(customers)):
                        neighbor = solution.clone()
                        new_customers = customers[:]
                        new_customers[i], new_customers[j] = new_customers[j], new_customers[i]
                        neighbor.truck_routes[route_idx].nodes = [
                            route.nodes[0]] + new_customers + [route.nodes[-1]]

                        move = TabuMove(
                            move_type=MoveType.SWAP_CUSTOMERS,
                            attributes=(route_idx, customers[i], customers[j]),
                            timestamp=0
                        )
                        neighbors.append((neighbor, move))

        return neighbors

    def _generate_two_opt_moves(self, solution: Solution) -> List[Tuple[Solution, TabuMove]]:
        """Generate 2-opt moves for truck routes."""
        neighbors = []

        for route_idx, route in enumerate(solution.truck_routes):
            nodes = route.nodes
            if len(nodes) > 4:  # Need at least 2 customers + depots
                for i in range(1, len(nodes) - 2):
                    for j in range(i + 1, len(nodes) - 1):
                        neighbor = solution.clone()
                        # 2-opt: reverse segment between i and j
                        new_nodes = nodes[:i] + \
                            list(reversed(nodes[i:j+1])) + nodes[j+1:]
                        neighbor.truck_routes[route_idx].nodes = new_nodes

                        move = TabuMove(
                            move_type=MoveType.TWO_OPT_TRUCK,
                            attributes=(route_idx, i, j),
                            timestamp=0
                        )
                        neighbors.append((neighbor, move))

        return neighbors

    def _generate_drone_insertion_moves(self, solution: Solution) -> List[Tuple[Solution, TabuMove]]:
        """Generate moves to insert customers into drone sorties."""
        neighbors = []
        # Placeholder - would need drone task structure
        return neighbors

    def _generate_drone_extraction_moves(self, solution: Solution) -> List[Tuple[Solution, TabuMove]]:
        """Generate moves to extract customers from drone sorties."""
        neighbors = []
        # Placeholder - would need drone task structure
        return neighbors

    def _generate_rebalance_moves(self, solution: Solution) -> List[Tuple[Solution, TabuMove]]:
        """Generate moves to rebalance work between truck and drone."""
        neighbors = []
        # Placeholder - complex rebalancing logic
        return neighbors

    def _generate_sortie_adjust_moves(self, solution: Solution) -> List[Tuple[Solution, TabuMove]]:
        """Generate moves to adjust drone sortie lengths."""
        neighbors = []
        # Placeholder - sortie length optimization
        return neighbors

    def _check_aspiration_criteria(self, move: TabuMove, solution: Solution, best_cost: float) -> bool:
        """Check if move should be allowed despite being tabu."""
        try:
            cost = self.evaluator.evaluate_solution(solution).total_cost
            improvement = best_cost - cost

            # Aspiration based on contribution
            truck_distance_reduction = self._estimate_truck_distance_reduction(
                move, solution)
            energy_reduction = self._estimate_energy_reduction(move, solution)

            return (improvement > 5.0 or  # Significant total improvement
                    truck_distance_reduction > 2.0 or  # Truck distance reduction
                    energy_reduction > 1.0)  # Energy reduction

        except:
            return False

    def _estimate_truck_distance_reduction(self, move: TabuMove, solution: Solution) -> float:
        """Estimate truck distance reduction from move."""
        # Placeholder - would need distance calculations
        return 0.0

    def _estimate_energy_reduction(self, move: TabuMove, solution: Solution) -> float:
        """Estimate total energy reduction from move."""
        # Placeholder - would need energy calculations
        return 0.0

    def _adapt_tabu_tenure(self):
        """Adapt tabu tenure based on search progress."""
        if len(self.solution_history) < 10:
            return

        # Calculate diversity metric
        recent_solutions = list(self.solution_history)[-10:]
        diversity = self._calculate_solution_diversity(recent_solutions)

        if diversity < 0.1:  # Low diversity - increase tabu tenure
            self.tabu_tenure = min(self.tabu_tenure + 2, self.max_tabu_tenure)
        elif diversity > 0.5:  # High diversity - decrease tabu tenure
            self.tabu_tenure = max(self.tabu_tenure - 1, self.min_tabu_tenure)

    def _calculate_solution_diversity(self, solutions: List[Solution]) -> float:
        """Calculate diversity among solutions."""
        if len(solutions) < 2:
            return 0.0

        total_distance = 0
        count = 0

        for i in range(len(solutions)):
            for j in range(i + 1, len(solutions)):
                dist = self._solution_distance(solutions[i], solutions[j])
                total_distance += dist
                count += 1

        return total_distance / count if count > 0 else 0.0

    def _solution_distance(self, sol1: Solution, sol2: Solution) -> float:
        """Calculate distance between two solutions."""
        # Simple distance based on route structures
        distance = 0
        for r1, r2 in zip(sol1.truck_routes, sol2.truck_routes):
            if r1.nodes != r2.nodes:
                distance += 1
        return distance

    def _diversify_solution(self, solution: Solution) -> Solution:
        """Apply diversification to escape local optima."""
        # Frequency-based diversification
        rare_moves = [mt for mt, freq in self.move_frequencies.items()
                      if freq < self.move_frequencies[mt] * 0.1]  # Rare moves

        if rare_moves:
            # Perturb solution using rare move types
            move_type = self.rng.choice(rare_moves)
            neighbors = self._generate_neighbors_by_type(solution, move_type)
            if neighbors:
                return self.rng.choice(neighbors)[0]

        # Partial restart - keep some structure, randomize rest
        restart_solution = solution.clone()
        # Randomly shuffle some routes
        for i in range(len(restart_solution.truck_routes)):
            if self.rng.random() < 0.3:  # 30% chance to perturb each route
                route = restart_solution.truck_routes[i]
                customers = route.nodes[1:-1]
                self.rng.shuffle(customers)
                restart_solution.truck_routes[i].nodes = [
                    route.nodes[0]] + customers + [route.nodes[-1]]

        return restart_solution
