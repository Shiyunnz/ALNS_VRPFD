"""Genetic Algorithm implementation for VRPFD.

Optimized version that incorporates ALNS-inspired drone task scheduling:
1. Energy utilization scoring for drone tasks
2. Drone chain building for consecutive tasks
3. Multi-customer drone task optimization
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Set
import json
from pathlib import Path

import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent
for _p in [Path(__file__).resolve().parent] + list(Path(__file__).resolve().parents):
    if (_p / 'run_alns.py').exists():
        _project_root = _p
        break
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
del _p, _project_root

from alns_vrpfd.evaluation import Evaluator
from alns_vrpfd.evaluation.energy import DroneEnergyModel
from alns_vrpfd.model.solution import Solution
from alns_vrpfd.model.route import TruckRoute, DroneTask
from alns_vrpfd.model.initializer import build_two_phase_initial_solution
from alns_vrpfd.instance.manager import InstanceManager
from alns_vrpfd.core.operators.base import _build_payloads


class FeasibilityRepair:
    """Repair infeasible solutions to make them feasible."""

    def __init__(self, instance: InstanceManager, evaluator: Evaluator):
        self.instance = instance
        self.evaluator = evaluator
        self._depot_start = instance.customer_manager.depot_start
        self._depot_end = instance.customer_manager.depot_end
        self._depots = {self._depot_start, self._depot_end}
        self._demands = instance.customer_manager.demands()
        self._node_index = {n: i for i, n in enumerate(instance.all_node_ids())}
        self._energy_model = DroneEnergyModel()
        self._battery = instance.robust_config.drone_battery_capacity
        self._deviation_rate = instance.robust_config.energy_deviation_rate
        self._uncertainty_budget = instance.robust_config.energy_uncertainty_budget
        self._drone_time = instance.time_matrix("drone")

    def _is_depot(self, node: int) -> bool:
        """Check if a node is a depot."""
        return node in self._depots

    def repair_solution(self, solution: Solution) -> Solution:
        """Repair a solution to make it feasible."""
        repaired = solution.clone()

        # Route crossover can retain a route whose old ID collides with the ID
        # assigned to a newly created route. Canonicalize identities before any
        # dictionary-keyed timing or drone-anchor logic is used.
        repaired = self.normalize_solution_ids(repaired)

        # 0. Remove duplicates first
        repaired = self._remove_duplicates(repaired)

        # Apply repairs in sequence
        repaired = self._repair_capacity_violations(repaired)
        repaired = self._repair_time_window_violations(repaired)
        repaired = self._repair_drone_assignment_conflicts(repaired)
        repaired = self._repair_drone_endurance_violations(repaired)

        # Sort drone tasks by launch position to ensure correct execution order
        repaired = self._sort_drone_tasks_by_launch_position(repaired)

        # Final check: ensure all customers are served
        repaired = self._repair_missing_customers(repaired)
        repaired = self.normalize_solution_ids(repaired)

        # Final validation: use evaluator to catch any remaining violations
        try:
            details = self.evaluator.evaluate_with_details(repaired)
            if not details.result.feasible:
                # Only remove energy-violating drone tasks, keep feasible ones
                if not details.robustness.feasible:
                    violating_ids = {b.task_id for b in details.robustness.task_breakdown if not b.feasible}
                    if violating_ids:
                        customers_to_truck = set()
                        surviving = []
                        for t in repaired.drone_tasks:
                            if t.task_id in violating_ids:
                                customers_to_truck.update(t.customers())
                            else:
                                surviving.append(t)
                        repaired.drone_tasks = surviving
                        for c in customers_to_truck:
                            self._insert_customer_cheapest(repaired, c)
        except Exception:
            pass

        return repaired

    def normalize_solution_ids(self, solution: Solution) -> Solution:
        """Assign unique route/task IDs and remap drone anchors by route membership."""
        routes = list(solution.truck_routes)

        def resolve_route_index(
            old_id: int | None,
            anchor_node: int,
            other_node: int,
        ) -> int | None:
            if old_id is None:
                return None
            candidates = [
                index
                for index, route in enumerate(routes)
                if route.id == old_id and anchor_node in route.nodes
            ]
            if len(candidates) == 1:
                return candidates[0]
            node_candidates = [
                index for index, route in enumerate(routes) if anchor_node in route.nodes
            ]
            if len(node_candidates) == 1:
                return node_candidates[0]
            paired = [
                index
                for index in candidates or node_candidates
                if other_node in routes[index].nodes
            ]
            if len(paired) == 1:
                return paired[0]
            return None

        valid_tasks = []
        removed_customers = []
        for task in solution.drone_tasks:
            launch_index = resolve_route_index(
                task.launch_truck, task.launch_node, task.retrieve_node
            )
            land_index = resolve_route_index(
                task.land_truck, task.retrieve_node, task.launch_node
            )
            if task.launch_truck is not None and launch_index is None:
                removed_customers.extend(task.customers())
                continue
            if task.land_truck is not None and land_index is None:
                removed_customers.extend(task.customers())
                continue
            task.launch_truck = launch_index
            task.land_truck = land_index
            valid_tasks.append(task)

        for route_index, route in enumerate(routes):
            route.id = route_index
        solution.truck_routes = routes
        solution.drone_tasks = valid_tasks
        for task_index, task in enumerate(solution.drone_tasks):
            task.task_id = task_index
            task.id = task_index

        for customer in removed_customers:
            self._insert_customer_cheapest(solution, customer)

        # New routes may have been created while reinserting ambiguous tasks.
        for route_index, route in enumerate(solution.truck_routes):
            route.id = route_index
        return solution

    def _remove_duplicates(self, solution: Solution) -> Solution:
        """Remove duplicate customer visits, prioritizing drone tasks."""
        served = set()

        # 1. Check Drone Tasks first
        valid_tasks = []
        for task in solution.drone_tasks:
            # Check if task customers are already served
            is_duplicate = False
            for cust in task.customers():
                if cust in served:
                    is_duplicate = True
                    break

            if not is_duplicate:
                for cust in task.customers():
                    served.add(cust)
                valid_tasks.append(task)
            else:
                # Task has duplicates, discard it (customers might be re-inserted later if needed)
                # or just rely on truck to serve them if they are in 'served'
                pass

        solution.drone_tasks = valid_tasks

        # 2. Check Truck Routes
        for route in solution.truck_routes:
            nodes_to_remove = []
            # Iterate copy to avoid modification issues, but we collect to remove
            for node in route.nodes:
                if self._is_depot(node):
                    continue

                if node in served:
                    nodes_to_remove.append(node)
                else:
                    served.add(node)

            for node in nodes_to_remove:
                route.nodes.remove(node)

        return solution

    def _insert_customer_cheapest(self, solution: Solution, customer: int) -> None:
        """Insert customer into the best position in any truck route."""
        # Check if already served
        for route in solution.truck_routes:
            if customer in route.nodes:
                return
        for task in solution.drone_tasks:
            if customer in task.customers():
                return

        demands = self.instance.customer_manager.demands()
        truck_capacity = self.instance.vehicle_specs['truck'].capacity

        best_route_idx = -1
        best_pos = -1
        min_cost_increase = float('inf')

        # Try to find best insertion in existing routes
        for r_idx, route in enumerate(solution.truck_routes):
            route_customers = [n for n in route.nodes if not self._is_depot(n)]
            route_demand = sum(demands.get(c, 0) for c in route_customers)

            if route_demand + demands.get(customer, 0) <= truck_capacity:
                # Try all positions
                # Nodes: [0, c1, c2, ..., 11]
                # Insert positions: after 0 (idx 1) to before 11 (idx len-1)
                # Range must cover insertion BEFORE the last node (11)
                # so range(1, len) is correct for insert index
                for pos in range(1, len(route.nodes)):
                    prev_node = route.nodes[pos-1]
                    next_node = route.nodes[pos]

                    # Calculate cost increase (distance)
                    # d(prev, cust) + d(cust, next) - d(prev, next)
                    d1 = self.instance.distances.get(
                        'truck', prev_node, customer)
                    d2 = self.instance.distances.get(
                        'truck', customer, next_node)
                    d3 = self.instance.distances.get(
                        'truck', prev_node, next_node)

                    increase = d1 + d2 - d3

                    if increase < min_cost_increase:
                        min_cost_increase = increase
                        best_route_idx = r_idx
                        best_pos = pos

        if best_route_idx != -1:
            solution.truck_routes[best_route_idx].nodes.insert(
                best_pos, customer)
        else:
            # Create new route
            from alns_vrpfd.model.route import TruckRoute
            new_route = TruckRoute(
                route_id=len(solution.truck_routes),
                nodes=[self._depot_start, customer, self._depot_end],
                capacity=truck_capacity
            )
            solution.add_truck_route(new_route)

    def strip_all_drone_tasks_to_trucks(self, solution: Solution) -> Solution:
        """Conservative final fallback: serve every drone customer by truck."""
        repaired = solution.clone()
        drone_customers = []
        for task in repaired.drone_tasks:
            drone_customers.extend(task.customers())
        repaired.drone_tasks = []
        for customer in drone_customers:
            self._insert_customer_cheapest(repaired, customer)
        repaired = self._repair_capacity_violations(repaired)
        repaired = self._repair_missing_customers(repaired)
        return repaired

    def _repair_capacity_violations(self, solution: Solution) -> Solution:
        """Repair truck capacity violations by reassigning customers to other routes."""
        demands = self.instance.customer_manager.demands()
        truck_capacity = self.instance.vehicle_specs['truck'].capacity

        # Collect customers that need to be reassigned
        customers_to_reassign = []

        # Check each route for capacity violations
        for route in solution.truck_routes:
            total_demand = sum(demands.get(node, 0)
                               for node in route.nodes if not self._is_depot(node))  # Exclude depots

            if total_demand > truck_capacity:
                # Remove customers from the end until capacity is satisfied
                nodes_to_remove = []
                current_demand = total_demand
                for i in range(len(route.nodes) - 2, 0, -1):  # Skip depots
                    node = route.nodes[i]
                    if not self._is_depot(node):
                        current_demand -= demands.get(node, 0)
                        nodes_to_remove.append(node)
                        if current_demand <= truck_capacity:
                            break

                # Remove the nodes and collect customers for reassignment
                for node in nodes_to_remove:
                    route.nodes.remove(node)
                    customers_to_reassign.append(node)

        # Reassign collected customers using cheapest insertion
        for customer in customers_to_reassign:
            self._insert_customer_cheapest(solution, customer)

        return solution

    def _repair_time_window_violations(self, solution: Solution) -> Solution:
        """Repair time window violations by reordering customers using latest time.

        Tries multiple orderings and keeps the best feasible one.
        """
        # First repair drone tasks to ensure valid truck references
        solution = self._repair_drone_launch_retrieve(solution)

        # Try sorting by latest time
        sol_latest = solution.clone()
        for route in sol_latest.truck_routes:
            valid_nodes = [n for n in route.nodes if n is not None and not self._is_depot(n)]
            if len(valid_nodes) <= 1:
                continue
            def get_latest(node):
                _, latest = self.instance.customer_manager.time_window(node)
                return latest if latest is not None else float('inf')
            valid_nodes.sort(key=get_latest)
            route.nodes = [self._depot_start] + valid_nodes + [self._depot_end]
        sol_latest = self._repair_drone_launch_retrieve(sol_latest)

        # Try sorting by optimal time
        sol_optimal = solution.clone()
        for route in sol_optimal.truck_routes:
            valid_nodes = [n for n in route.nodes if n is not None and not self._is_depot(n)]
            if len(valid_nodes) <= 1:
                continue
            def get_optimal(node):
                optimal, _ = self.instance.customer_manager.time_window(node)
                return optimal if optimal is not None else float('inf')
            valid_nodes.sort(key=get_optimal)
            route.nodes = [self._depot_start] + valid_nodes + [self._depot_end]
        sol_optimal = self._repair_drone_launch_retrieve(sol_optimal)

        # Evaluate all candidates
        candidates = [solution, sol_latest, sol_optimal]
        best_sol = solution
        best_violations = float('inf')

        for sol in candidates:
            try:
                details = self.evaluator.evaluate_with_details(sol)
                n_violations = len(details.delay_breakdown.violations)
                if details.result.feasible:
                    # Feasible is best
                    if best_violations > 0 or (details.result.total_cost < self.evaluator.evaluate_solution(best_sol).total_cost):
                        best_sol = sol
                        best_violations = 0
                elif n_violations < best_violations:
                    best_sol = sol
                    best_violations = n_violations
            except Exception:
                continue

        return best_sol

    def _repair_drone_launch_retrieve(self, solution: Solution) -> Solution:
        """Repair drone tasks to have valid launch/retrieve nodes after truck route changes."""
        if not solution.drone_tasks:
            return solution

        # Build a set of truck route nodes for quick lookup
        truck_nodes = set()
        for route in solution.truck_routes:
            truck_nodes.update(route.nodes)

        valid_tasks = []
        removed_customers = []

        for task in solution.drone_tasks:
            # Check if launch and retrieve nodes are still in truck route
            if task.launch_node in truck_nodes and task.retrieve_node in truck_nodes:
                # Check if launch comes before retrieve in any route
                is_valid = False
                for route in solution.truck_routes:
                    try:
                        launch_idx = route.nodes.index(task.launch_node)
                        retrieve_idx = route.nodes.index(task.retrieve_node)
                        if launch_idx < retrieve_idx:
                            is_valid = True
                            break
                    except ValueError:
                        continue

                if is_valid and self._robust_energy_feasible(task.launch_node, list(task.customers()), task.retrieve_node):
                    valid_tasks.append(task)
                else:
                    # Try to reassign to nearest valid positions
                    new_task = self._try_reassign_drone_task(solution, task)
                    if new_task:
                        valid_tasks.append(new_task)
                    else:
                        removed_customers.extend(task.customers())
            else:
                # Try to reassign to nearest valid positions
                new_task = self._try_reassign_drone_task(solution, task)
                if new_task:
                    valid_tasks.append(new_task)
                else:
                    removed_customers.extend(task.customers())

        solution.drone_tasks = valid_tasks

        # Insert removed customers into truck route
        for customer in removed_customers:
            self._insert_customer_cheapest(solution, customer)

        return solution

    def _try_reassign_drone_task(self, solution: Solution, task) -> Optional['DroneTask']:
        """Try to reassign a drone task to valid launch/retrieve nodes."""
        from alns_vrpfd.model.route import DroneTask

        customers = task.customers()
        if not customers:
            return None

        drone_speed = self.instance.vehicle_specs['drone'].speed
        drone_endurance = self.instance.vehicle_specs['drone'].endurance

        best_task = None
        best_score = float('inf')

        for route_idx, route in enumerate(solution.truck_routes):
            nodes = route.nodes
            # Try all launch/retrieve combinations
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    launch = nodes[i]
                    retrieve = nodes[j]

                    # Calculate drone distance
                    total_dist = self.instance.distances.get(
                        'drone', launch, customers[0])
                    for k in range(len(customers) - 1):
                        total_dist += self.instance.distances.get(
                            'drone', customers[k], customers[k+1])
                    total_dist += self.instance.distances.get(
                        'drone', customers[-1], retrieve)

                    # Check endurance
                    flight_time = total_dist / \
                        drone_speed + len(customers) * 0.1
                    if flight_time > drone_endurance:
                        continue

                    # Score: prefer shorter distances
                    if total_dist < best_score:
                        best_score = total_dist
                        best_task = DroneTask(
                            task_id=task.task_id,
                            drone_id=task.drone_id,
                            launch_truck=route_idx,
                            launch_node=launch,
                            customers=customers.copy(),
                            land_truck=route_idx,
                            retrieve_node=retrieve
                        )

        return best_task

    def _repair_drone_assignment_conflicts(self, solution: Solution) -> Solution:
        """Repair drone assignment conflicts - validate task sequences.

        Checks for:
        1. Customer duplicates across drone tasks
        2. Interval overlaps for same drone (tasks must be sequential)
        """
        if not solution.drone_tasks:
            return solution

        # Build truck node indices for interval checking
        truck_node_indices = {}
        for route in solution.truck_routes:
            truck_node_indices[route.id] = {
                node: i for i, node in enumerate(route.nodes)}

        depot_start = self._depot_start
        depot_end = self._depot_end

        # Check for customer duplicates and interval overlaps
        served_customers = set()
        # drone_id -> [(start_idx, end_idx, task)]
        drone_intervals: Dict[int, List[Tuple[int, int, Any]]] = {}
        valid_tasks = []

        for task in solution.drone_tasks:
            is_valid = True
            task_customers = task.customers()

            # Check 1: Customer duplicates
            for cust in task_customers:
                if cust in served_customers:
                    is_valid = False
                    break

            if not is_valid:
                # Re-insert conflicting customers to truck
                for cust in task_customers:
                    if cust not in served_customers:
                        self._insert_customer_cheapest(solution, cust)
                continue

            # Check 2: Interval overlap for same drone
            d_id = task.drone_id
            if d_id not in drone_intervals:
                drone_intervals[d_id] = []

            # Calculate interval
            t_id = task.launch_truck if task.launch_truck is not None else task.land_truck
            l_idx, r_idx = -1, -1

            if t_id is not None and t_id in truck_node_indices:
                indices = truck_node_indices[t_id]

                if task.launch_node == depot_start:
                    l_idx = 0
                elif task.launch_truck is not None and task.launch_node in indices:
                    l_idx = indices[task.launch_node]

                if task.retrieve_node == depot_end:
                    if depot_end in indices:
                        r_idx = indices[depot_end]
                    else:
                        r_idx = len(indices)  # End of route
                elif task.land_truck is not None and task.retrieve_node in indices:
                    r_idx = indices[task.retrieve_node]

            # Check for overlap with existing intervals
            has_overlap = False
            if l_idx != -1 and r_idx != -1:
                for existing_l, existing_r, _ in drone_intervals[d_id]:
                    # Overlap if new task's launch is before existing task's retrieve
                    # and new task's retrieve is after existing task's launch
                    if l_idx < existing_r and r_idx > existing_l:
                        has_overlap = True
                        break

            if has_overlap:
                # Conflict: insert customers to truck
                for cust in task_customers:
                    self._insert_customer_cheapest(solution, cust)
            else:
                # Valid task
                for cust in task_customers:
                    served_customers.add(cust)
                if l_idx != -1 and r_idx != -1:
                    drone_intervals[d_id].append((l_idx, r_idx, task))
                valid_tasks.append(task)

        solution.drone_tasks = valid_tasks
        return solution

    def _robust_energy_feasible(self, launch: int, customers: List[int], retrieve: int) -> bool:
        """Check if a drone sortie is robust-energy-feasible."""
        if not customers:
            return False
        payloads = _build_payloads(customers, self._demands)
        nodes = [launch] + customers + [retrieve]
        ni = self._node_index
        dt = self._drone_time
        nom, devs = 0.0, []
        for p, a, b in zip(payloads, nodes, nodes[1:]):
            try:
                t = dt[ni[a]][ni[b]]
            except (KeyError, IndexError):
                return False
            e = self._energy_model.energy_kwh(p, t)
            nom += e
            devs.append(e * self._deviation_rate)
        worst = nom + self._budgeted_sum(devs, self._uncertainty_budget)
        return worst <= self._battery + 1e-6

    @staticmethod
    def _budgeted_sum(values, budget):
        if not values or budget <= 0:
            return 0.0
        sorted_vals = sorted(values, reverse=True)
        integer = int(min(budget, len(sorted_vals)))
        fractional = max(0.0, budget - integer)
        total = sum(sorted_vals[:integer])
        if fractional > 0 and integer < len(sorted_vals):
            total += fractional * sorted_vals[integer]
        return total

    def _repair_drone_endurance_violations(self, solution: Solution) -> Solution:
        """Repair drone energy violations by removing infeasible tasks."""
        valid_tasks = []
        removed_tasks = []
        for task in solution.drone_tasks:
            customers = list(task.customers())
            if not customers:
                removed_tasks.append(task)
                continue
            if self._robust_energy_feasible(task.launch_node, customers, task.retrieve_node):
                valid_tasks.append(task)
            else:
                removed_tasks.append(task)

        solution.drone_tasks = valid_tasks
        for task in removed_tasks:
            for customer in task.customers():
                self._insert_customer_cheapest(solution, customer)
        return solution

    def _sort_drone_tasks_by_launch_position(self, solution: Solution) -> Solution:
        """Sort drone tasks by their launch position in the truck route to ensure correct execution order."""
        if not solution.drone_tasks or not solution.truck_routes:
            return solution

        # Build truck route node positions
        truck_positions = {}
        for route in solution.truck_routes:
            truck_positions[route.id] = {node: pos for pos, node in enumerate(route.nodes)}

        def get_launch_position(task):
            if task.launch_truck is None:
                return -1  # Depot launch - first
            if task.launch_truck in truck_positions:
                return truck_positions[task.launch_truck].get(task.launch_node, 0)
            return 0

        # Sort tasks by launch position
        solution.drone_tasks.sort(key=get_launch_position)
        return solution

    def _repair_missing_customers(self, solution: Solution) -> Solution:
        """Ensure all customers are served by inserting missing ones into truck routes."""
        # Get all required customers from instance
        all_customers = set(self.instance.customer_manager.demands().keys())

        # Collect served customers
        served_by_truck = set()
        for route in solution.truck_routes:
            for node in route.nodes:
                if not self._is_depot(node) and node in all_customers:
                    served_by_truck.add(node)

        served_by_drone = set()
        for task in solution.drone_tasks:
            for cust in task.customers():
                if cust in all_customers:
                    served_by_drone.add(cust)

        all_served = served_by_truck | served_by_drone
        missing = all_customers - all_served

        # Insert missing customers into truck routes
        for customer in missing:
            self._insert_customer_cheapest(solution, customer)

        return solution


class DroneChainBuilder:
    """ALNS-inspired drone chain builder for GA.

    This class builds chained drone tasks to create more aggressive drone schedules.
    """

    def __init__(self, instance: InstanceManager, evaluator: Evaluator, rng: random.Random):
        self.instance = instance
        self.evaluator = evaluator
        self.rng = rng

        # Cache instance data
        self._demands = instance.customer_manager.demands()
        self._drone_cap = instance.vehicle_specs['drone'].capacity
        self._drone_endurance = instance.vehicle_specs['drone'].endurance
        self._drone_speed = instance.vehicle_specs['drone'].speed
        self._truck_capacity = instance.vehicle_specs['truck'].capacity
        # Get actual drone count
        self._drone_count = instance.vehicle_specs['drone'].number

        # Build distance matrices
        self._drone_dist = instance.distance_matrix('drone')
        self._truck_dist = instance.distance_matrix('truck')
        self._node_index = self._build_node_index()

        # Get depot nodes from instance
        self._depot_start = instance.customer_manager.depot_start
        self._depot_end = instance.customer_manager.depot_end
        self._depots = {self._depot_start, self._depot_end}

        # Robust energy model
        self._energy_model = DroneEnergyModel()
        self._battery = instance.robust_config.drone_battery_capacity
        self._deviation_rate = instance.robust_config.energy_deviation_rate
        self._uncertainty_budget = instance.robust_config.energy_uncertainty_budget
        self._drone_time = instance.time_matrix("drone")

    def _build_node_index(self) -> Dict[int, int]:
        """Build mapping from node ID to matrix index."""
        return {n: i for i, n in enumerate(self.instance.all_node_ids())}

    def get_drone_eligible_customers(self, solution: Solution) -> Set[int]:
        """Get customers eligible for drone delivery that aren't already served by drones."""
        # Get all served customers
        served_by_drone = set()
        for task in solution.drone_tasks:
            for cust in task.customers():
                served_by_drone.add(cust)

        # Find truck customers eligible for drone
        eligible = set()
        for route in solution.truck_routes:
            for node in route.nodes:
                if node in self._depots:
                    continue
                demand = self._demands.get(node, float('inf'))
                if demand <= self._drone_cap and node not in served_by_drone:
                    eligible.add(node)

        return eligible

    def _calculate_drone_distance(self, launch: int, customers: List[int], retrieve: int) -> float:
        """Calculate total drone distance for a task."""
        if not customers:
            return float('inf')

        total = 0.0
        nodes = [launch] + customers + [retrieve]
        for i in range(len(nodes) - 1):
            i_idx = self._node_index.get(nodes[i])
            j_idx = self._node_index.get(nodes[i + 1])
            if i_idx is None or j_idx is None:
                return float('inf')
            total += self._drone_dist[i_idx][j_idx]
        return total

    def _estimate_drone_time(self, launch: int, customers: List[int], retrieve: int) -> float:
        """Estimate drone task completion time."""
        distance = self._calculate_drone_distance(launch, customers, retrieve)
        if distance == float('inf'):
            return float('inf')
        # Add service time per customer
        return distance / self._drone_speed + len(customers) * 0.1

    def _robust_energy_feasible(self, launch: int, customers: List[int], retrieve: int) -> bool:
        """Check if a drone sortie is robust-energy-feasible."""
        if not customers:
            return False
        payloads = _build_payloads(customers, self._demands)
        nodes = [launch] + customers + [retrieve]
        ni = self._node_index
        dt = self._drone_time
        nom, devs = 0.0, []
        for p, a, b in zip(payloads, nodes, nodes[1:]):
            try:
                t = dt[ni[a]][ni[b]]
            except (KeyError, IndexError):
                return False
            e = self._energy_model.energy_kwh(p, t)
            nom += e
            devs.append(e * self._deviation_rate)
        worst = nom + self._budgeted_sum(devs, self._uncertainty_budget)
        return worst <= self._battery + 1e-6

    @staticmethod
    def _budgeted_sum(values: List[float], budget: float) -> float:
        if not values or budget <= 0:
            return 0.0
        sorted_vals = sorted(values, reverse=True)
        integer = int(min(budget, len(sorted_vals)))
        fractional = max(0.0, budget - integer)
        total = sum(sorted_vals[:integer])
        if fractional > 0 and integer < len(sorted_vals):
            total += fractional * sorted_vals[integer]
        return total

    def _score_drone_task(self, customers: List[int], launch: int, retrieve: int) -> float:
        """Score a potential drone task. Returns -inf if energy-infeasible."""
        if not self._robust_energy_feasible(launch, customers, retrieve):
            return -float('inf')
        distance = self._calculate_drone_distance(launch, customers, retrieve)
        if distance == float('inf'):
            return -float('inf')
        score = len(customers) * 2.0
        if len(customers) >= 2:
            score += 0.8 * len(customers)
        return score

    def build_drone_task(
        self,
        solution: Solution,
        launch_node: int,
        retrieve_node: int,
        candidates: List[int],
        max_customers: int = 3
    ) -> Optional[DroneTask]:
        """Build a drone task from launch to retrieve serving candidates."""
        # Filter candidates by drone capacity
        valid_candidates = [c for c in candidates
                            if self._demands.get(c, float('inf')) <= self._drone_cap]

        if not valid_candidates:
            return None

        # Greedily build multi-customer task
        best_customers = []
        best_score = -float('inf')

        # Try single customer first
        for cust in valid_candidates[:10]:  # Limit search
            score = self._score_drone_task([cust], launch_node, retrieve_node)
            if score > best_score:
                best_score = score
                best_customers = [cust]

        # Try adding more customers
        if best_customers and max_customers > 1:
            current_customers = list(best_customers)
            for _ in range(max_customers - 1):
                best_addition = None
                best_addition_score = best_score

                remaining = [
                    c for c in valid_candidates if c not in current_customers]
                for cust in remaining[:5]:  # Limit search
                    test_customers = current_customers + [cust]
                    score = self._score_drone_task(
                        test_customers, launch_node, retrieve_node)
                    if score > best_addition_score:
                        best_addition_score = score
                        best_addition = cust

                if best_addition:
                    current_customers.append(best_addition)
                    best_score = best_addition_score
                else:
                    break

            if len(current_customers) > len(best_customers):
                best_customers = current_customers

        if not best_customers:
            return None

        # Allow ALL drones (even those with existing tasks) - evaluator checks overlap
        # Find first drone ID (evaluator will validate time overlap)
        drone_id = 0  # Use drone 0; evaluator checks if tasks overlap

        # Create task
        task_id = max((t.task_id or 0)
                      for t in solution.drone_tasks) + 1 if solution.drone_tasks else 1

        # Calculate payloads - cumulative demand remaining at each segment
        # Payload at segment i = sum of demands of customers from i to end
        payloads = []
        remaining_demand = sum(self._demands.get(c, 0.0) for c in best_customers)
        payloads.append(remaining_demand)  # After launch
        for cust in best_customers:
            remaining_demand -= self._demands.get(cust, 0.0)
            payloads.append(remaining_demand)  # After serving each customer

        return DroneTask(
            task_id=task_id,
            drone_id=drone_id,
            launch_truck=None,
            launch_node=launch_node,
            customers=best_customers,
            land_truck=None,
            retrieve_node=retrieve_node,
            payloads=payloads,
        )

    def optimize_drone_tasks(self, solution: Solution) -> Solution:
        """Optimize drone task scheduling for a solution (ALNS-inspired).
        
        This version loops multiple times to create as many drone tasks as possible,
        similar to ALNS's aggressive drone utilization strategy.
        """
        optimized = solution.clone()
        
        # Loop multiple times to create more drone tasks
        max_rounds = 5  # Try up to 5 rounds of drone task creation
        for round_num in range(max_rounds):
            # Get drone-eligible customers currently served by trucks
            eligible = self.get_drone_eligible_customers(optimized)
            
            # DEBUG output disabled
            # if round_num == 0:
            #     print(f\"[DEBUG DroneOpt] Round {round_num}: eligible={len(eligible)} customers, existing_drones={len(optimized.drone_tasks)}\")
            
            if not eligible:
                break  # No more eligible customers
            
            tasks_created_this_round = 0
            
            # Try to create drone tasks from truck routes
            for route_idx, route in enumerate(optimized.truck_routes):
                if len(route.nodes) < 4:  # Need launch and retrieve points
                    continue
                
                # Find candidates from this route that are still eligible
                route_eligible = [n for n in route.nodes if n in eligible]
                if not route_eligible:
                    continue
                
                # Try different launch/retrieve pairs
                best_task = None
                best_score = -float('inf')
                best_launch_pos = -1
                best_retrieve_pos = -1
                
                for launch_pos in range(len(route.nodes) - 2):
                    launch_node = route.nodes[launch_pos]
                    if launch_node in route_eligible:
                        continue  # Don't launch from a customer we want to serve
                    
                    for retrieve_pos in range(launch_pos + 2, len(route.nodes)):
                        retrieve_node = route.nodes[retrieve_pos]
                        if retrieve_node in route_eligible:
                            continue
                        
                        # Get candidates between launch and retrieve
                        between_candidates = [route.nodes[i] for i in range(launch_pos + 1, retrieve_pos)
                                              if route.nodes[i] in eligible]
                        
                        if not between_candidates:
                            continue
                        
                        task = self.build_drone_task(
                            optimized, launch_node, retrieve_node, between_candidates, max_customers=3
                        )
                        
                        if task:
                            # Score based on number of customers served
                            score = len(task.customers())
                            if score > best_score:
                                best_score = score
                                best_task = task
                                best_launch_pos = launch_pos
                                best_retrieve_pos = retrieve_pos
                
                # Apply best task found for this route
                if best_task:
                    # Remove customers from truck route
                    customers_removed = []
                    for cust in best_task.customers():
                        if cust in optimized.truck_routes[route_idx].nodes:
                            optimized.truck_routes[route_idx].nodes.remove(cust)
                            customers_removed.append(cust)
                        eligible.discard(cust)
                    
                    optimized.drone_tasks.append(best_task)
                    tasks_created_this_round += 1
                    
                    # DEBUG output disabled
                    # if round_num == 0 and route_idx == 0:
                    #     print(f\"[DEBUG DroneTask] Created task with {len(best_task.customers())} customers: {best_task.customers()}, removed from route: {customers_removed}\")
            
            if tasks_created_this_round == 0:
                break  # No more tasks can be created
        
        return optimized


@dataclass
class GAConfig:
    """Configuration parameters for the Genetic Algorithm."""

    population_size: int = 100
    generations: int = 100
    tournament_size: int = 5
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elite_size: int = 5
    max_stagnation: int = 20

    # Operator-specific parameters
    truck_route_crossover_rate: float = 0.7
    drone_task_mutation_rate: float = 0.3
    route_segment_swap_rate: float = 0.4

    # Time limits
    time_limit: Optional[float] = None
    generation_time_limit: Optional[float] = None
    strict_time_budget: bool = False

    # Adaptive parameters
    adaptive_enabled: bool = True
    adaptation_interval: int = 5  # Adapt every N generations
    diversity_threshold: float = 2.0  # Diversity threshold for adaptation


@dataclass
class Individual:
    """Represents a candidate solution in the GA population."""

    solution: Solution
    fitness: float = float('inf')
    feasible: bool = False
    evaluation_time: float = 0.0
    truck_distance: float = 0.0
    drone_distance: float = 0.0
    delay_penalty: float = 0.0

    def __lt__(self, other: Individual) -> bool:
        """Compare individuals by fitness (lower is better)."""
        return self.fitness < other.fitness

    def __le__(self, other: Individual) -> bool:
        """Compare individuals by fitness (lower is better)."""
        return self.fitness <= other.fitness


class GeneticAlgorithm:
    """Genetic Algorithm for solving VRPFD problems.

    Enhanced with ALNS-inspired drone task optimization.
    """

    def __init__(
        self,
        instance: InstanceManager,
        config: GAConfig,
        evaluator: Evaluator,
        rng: Optional[random.Random] = None,
        search_evaluator=None,
    ):
        self.instance = instance
        self.config = config
        self.evaluator = evaluator
        self.search_evaluator = search_evaluator
        self.rng = rng or random.Random()

        # Get depot nodes from instance
        self._depot_start = instance.customer_manager.depot_start
        self._depot_end = instance.customer_manager.depot_end
        self._depots = {self._depot_start, self._depot_end}
        self._demands = instance.customer_manager.demands()
        self._truck_dist = instance.distance_matrix('truck')
        self._drone_dist = instance.distance_matrix('drone')
        self._node_index = {n: i for i, n in enumerate(instance.all_node_ids())}
        if self.search_evaluator is not None:
            self._drone_mutation_candidate_eval_limit = 32
            self._aggressive_drone_top_n = 5
        else:
            self._drone_mutation_candidate_eval_limit = 48
            self._aggressive_drone_top_n = 10

        # Initialize feasibility repair
        self.repair = FeasibilityRepair(instance, evaluator)

        # Initialize drone chain builder (ALNS-inspired)
        self.drone_builder = DroneChainBuilder(instance, evaluator, self.rng)

        # Population and statistics
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.best_feasible_individual: Optional[Individual] = None
        self.generation = 0
        self.stagnation_counter = 0
        self.start_time = 0.0

        # Statistics tracking
        self.stats = {
            'generations': [],
            'best_fitness': [],
            'avg_fitness': [],
            'feasible_count': [],
            'diversity': [],
            'unique_solutions': [],  # Track number of unique solutions
            'elapsed_time': [],
            'best_feasible_fitness': [],
        }

    def initialize_population(self, initial_solution: Optional[Solution] = None) -> None:
        """Initialize the population using the provided initial solution as base.

        All population members are variations (mutations) of the same initial solution,
        ensuring fair comparison with ALNS/TS which also start from the same initial.
        """
        self.population = []
        init_time_limit = self.config.time_limit * \
            0.3 if self.config.time_limit else None
        init_start = time.time()

        # First individual is always the provided initial solution (repaired)
        if initial_solution is not None:
            base = initial_solution.clone()
            base = self.repair.repair_solution(base)
            individual = Individual(solution=base)
            self._evaluate_individual(individual)
            self.population.append(individual)

        # Create remaining population as mutations of the base
        remaining = self.config.population_size - len(self.population)
        for i in range(remaining):
            if init_time_limit and (time.time() - init_start) > init_time_limit:
                break

            if initial_solution is not None:
                # Mutate from the provided initial solution
                solution = initial_solution.clone()
                # Apply varying levels of mutation for diversity
                n_mutations = self.rng.randint(1, 3)
                for _ in range(n_mutations):
                    solution = self._mutate(solution)
                    solution = self.repair.repair_solution(solution)
            else:
                # Fallback: create from scratch
                solution = build_two_phase_initial_solution(self.instance)
                solution = self.repair.repair_solution(solution)

            individual = Individual(solution=solution)
            self._evaluate_individual(individual)
            self.population.append(individual)

        # Ensure we have at least a few individuals
        if len(self.population) < 5:
            while len(self.population) < 5:
                if initial_solution is not None:
                    solution = initial_solution.clone()
                    solution = self._mutate(solution)
                    solution = self.repair.repair_solution(solution)
                else:
                    solution = build_two_phase_initial_solution(self.instance)
                    solution = self.repair.repair_solution(solution)
                individual = Individual(solution=solution)
                self._evaluate_individual(individual)
                self.population.append(individual)

        # Sort population by fitness
        self.population.sort()
        self.best_individual = self.population[0]
        self._update_best_feasible_from_population()

    def _copy_individual(self, individual: Individual) -> Individual:
        return Individual(
            solution=individual.solution.clone(),
            fitness=individual.fitness,
            feasible=individual.feasible,
            evaluation_time=individual.evaluation_time,
            truck_distance=individual.truck_distance,
            drone_distance=individual.drone_distance,
            delay_penalty=individual.delay_penalty,
        )

    def _remember_feasible_individual(self, individual: Individual) -> None:
        if not individual.feasible or not math.isfinite(individual.fitness):
            return
        if (
            getattr(self, "best_feasible_individual", None) is None
            or individual.fitness < self.best_feasible_individual.fitness
        ):
            self.best_feasible_individual = self._copy_individual(individual)

    def _update_best_feasible_from_population(self) -> None:
        for individual in self.population:
            self._remember_feasible_individual(individual)

    def _repair_best_individual_for_return(self) -> Optional[Individual]:
        if self.best_individual is None:
            return None
        repair_attempts = (
            self.repair.repair_solution,
            self.repair.strip_all_drone_tasks_to_trucks,
        )
        for repair_attempt in repair_attempts:
            try:
                repaired = repair_attempt(self.best_individual.solution)
            except Exception:
                continue
            candidate = Individual(solution=repaired)
            try:
                self._evaluate_individual(candidate)
                if candidate.feasible:
                    return candidate
            except Exception:
                continue
        return None

    def _evaluate_individual(self, individual: Individual) -> None:
        """Evaluate an individual's fitness with penalized objective for infeasible solutions."""
        eval_start = time.time()

        try:
            repair = getattr(self, "repair", None)
            if repair is not None:
                individual.solution = repair.normalize_solution_ids(
                    individual.solution
                )
            search_evaluator = getattr(self, "search_evaluator", None)
            if search_evaluator is not None:
                fitness, feasible, _ = search_evaluator.penalized_cost(individual.solution)
                result = search_evaluator.evaluate_solution(individual.solution)
                individual.truck_distance = result.truck_distance_cost
                individual.drone_distance = result.drone_distance_cost
                individual.delay_penalty = result.delay_penalty
                individual.fitness = fitness
                individual.feasible = feasible
                individual.evaluation_time = time.time() - eval_start
                return

            details = self.evaluator.evaluate_with_details(individual.solution)
            individual.truck_distance = details.result.truck_distance_cost
            individual.drone_distance = details.result.drone_distance_cost
            individual.delay_penalty = details.result.delay_penalty

            if details.result.feasible:
                individual.fitness = details.result.total_cost
                individual.feasible = True
            else:
                # Penalized fitness for infeasible solutions
                penalty = 0.0
                if details.result.delay_penalty > 0:
                    penalty += details.result.delay_penalty
                # Energy violations
                for b in details.robustness.task_breakdown:
                    if not b.feasible:
                        penalty += 100.0 * abs(b.margin)
                # Time window violations
                if details.delay_breakdown.violations:
                    penalty += 50.0 * len(details.delay_breakdown.violations)
                # Coverage violation
                all_customers = set(self.instance.customer_manager.demands().keys())
                truck_served = set()
                for r in individual.solution.truck_routes:
                    for n in r.customers():
                        truck_served.add(n)
                drone_served = set()
                for t in individual.solution.drone_tasks:
                    for c in t.customers():
                        drone_served.add(c)
                missing = all_customers - truck_served - drone_served
                if missing:
                    penalty += 200.0 * len(missing)
                # Capacity violation
                truck_cap = self.instance.vehicle_specs['truck'].capacity
                for r in individual.solution.truck_routes:
                    load = sum(self.instance.customer_manager.demands().get(c, 0) for c in r.customers())
                    excess = max(0, load - truck_cap)
                    if excess > 0:
                        penalty += 100.0 * excess

                base_cost = details.result.total_cost if math.isfinite(details.result.total_cost) else 1e6
                individual.fitness = base_cost + penalty
                individual.feasible = False
        except Exception:
            individual.fitness = float('inf')
            individual.feasible = False

        individual.evaluation_time = time.time() - eval_start

    def _evaluate_solution(self, solution: Solution):
        search_evaluator = getattr(self, "search_evaluator", None)
        if search_evaluator is not None:
            return search_evaluator.evaluate_solution(solution)
        return self.evaluator.evaluate_solution(solution)

    def _drone_mutation_eval_limit(self) -> int:
        limit = getattr(self, "_drone_mutation_candidate_eval_limit", 24)
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 24
        return max(1, limit)

    def _rank_drone_anchor_candidates(self, candidates: List[Tuple]) -> List[Tuple]:
        limit = min(len(candidates), self._drone_mutation_eval_limit())

        scored = []
        for pos, candidate in enumerate(candidates):
            try:
                score = self._score_drone_anchor_candidate(candidate)
            except Exception:
                score = float("inf")
            if score is None or not math.isfinite(score):
                score = float("inf")
            scored.append((score, pos, candidate))

        scored.sort(key=lambda item: (item[0], item[1]))
        return [candidate for _, _, candidate in scored[:limit]]

    def _score_drone_anchor_candidate(self, candidate: Tuple) -> float:
        source_route, cust, launch_route, launch_node, retrieve_route, retrieve_node = candidate
        drone_distance = (
            self._matrix_distance(self._drone_dist, launch_node, cust)
            + self._matrix_distance(self._drone_dist, cust, retrieve_node)
        )
        truck_saving = self._truck_removal_saving(source_route, cust)
        return drone_distance - truck_saving

    def _truck_removal_saving(self, route, customer: int) -> float:
        nodes = getattr(route, "nodes", []) or []
        try:
            idx = nodes.index(customer)
        except ValueError:
            return 0.0
        if idx <= 0 or idx >= len(nodes) - 1:
            return 0.0
        prev_node = nodes[idx - 1]
        next_node = nodes[idx + 1]
        return (
            self._matrix_distance(self._truck_dist, prev_node, customer)
            + self._matrix_distance(self._truck_dist, customer, next_node)
            - self._matrix_distance(self._truck_dist, prev_node, next_node)
        )

    def _matrix_distance(self, matrix, a: int, b: int) -> float:
        try:
            return matrix[self._node_index[a]][self._node_index[b]]
        except (KeyError, IndexError, TypeError, AttributeError):
            return float("inf")

    def run(self, initial_solution: Optional[Solution] = None) -> Individual:
        """Run the genetic algorithm."""
        self.start_time = time.time()

        # Initialize population - use provided initial solution as base
        self.initialize_population(initial_solution)

        # Main GA loop. Fixed-time comparison experiments explicitly opt into
        # a strict budget so generation/stagnation caps cannot end the run early.
        strict_time_budget = bool(
            getattr(self.config, "strict_time_budget", False)
            and getattr(self.config, "time_limit", None)
        )
        generation_limit = (
            10**12 if strict_time_budget else self.config.generations
        )
        for generation in range(generation_limit):
            self.generation = generation

            # Check time limits
            if self._check_time_limits():
                break

            # Create new population
            new_population = self._create_new_population()

            # Update population
            self.population = new_population
            self.population.sort()
            self._update_best_feasible_from_population()

            # Update best individual
            if self.population[0].fitness < self.best_individual.fitness:
                self.best_individual = self.population[0]
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1

            # Apply local search to elite individuals (every few generations)
            # Adaptive frequency: more often if stagnating
            ls_freq = 2 if self.stagnation_counter < 10 else 1  # More frequent

            if generation % ls_freq == 0:
                self._apply_local_search()
                self._update_best_feasible_from_population()
            
            # Drone optimization on top individuals every 3 generations
            if generation % 3 == 0:
                self._apply_aggressive_drone_optimization()
                self._update_best_feasible_from_population()

            # Adaptive parameter adjustment
            if self.config.adaptive_enabled and generation % self.config.adaptation_interval == 0:
                self._adapt_parameters()

            # Record statistics
            self._record_statistics()

            # Check for early stopping
            if (
                not strict_time_budget
                and self.stagnation_counter >= self.config.max_stagnation
            ):
                break

        if self.best_feasible_individual is not None:
            return self.best_feasible_individual

        repaired = self._repair_best_individual_for_return()
        if repaired is not None:
            return repaired

        return self.best_individual

    def _create_new_population(self) -> List[Individual]:
        """Create a new population through selection, crossover, and mutation."""
        new_population = []

        # Elitism: keep the best individuals
        for i in range(self.config.elite_size):
            new_population.append(self.population[i])

        if self._check_time_limits():
            return new_population

        # Fill the rest of the population
        while len(new_population) < self.config.population_size:
            if self._check_time_limits():
                break

            # Selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()

            # Crossover
            if self.rng.random() < self.config.crossover_rate:
                # Choose between crossover types
                if self.rng.random() < 0.5:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1 = self._route_crossover(parent1, parent2)
                    child2 = self._route_crossover(parent2, parent1)

                # Repair after crossover
                child1 = self.repair.repair_solution(child1)
                child2 = self.repair.repair_solution(child2)
            else:
                child1, child2 = parent1.clone(), parent2.clone()

            # Mutation (aggressive exploration for competitive results)
            effective_mutation_rate = self.config.mutation_rate * 1.5  # Aggressive boost
            if self.rng.random() < effective_mutation_rate:
                child1 = self._mutate(child1)
                child1 = self.repair.repair_solution(child1)
                if self._check_time_limits():
                    break
            if self.rng.random() < effective_mutation_rate:
                child2 = self._mutate(child2)
                child2 = self.repair.repair_solution(child2)
                if self._check_time_limits():
                    break

            # Apply 2-opt local search on most offspring (80% - competitive quality)
            if self.rng.random() < 0.80:
                child1 = self._apply_simple_2opt(child1)
                if self._check_time_limits():
                    break
            if self.rng.random() < 0.80:
                child2 = self._apply_simple_2opt(child2)
                if self._check_time_limits():
                    break

            # Create individuals and evaluate
            ind1 = Individual(solution=child1)
            ind2 = Individual(solution=child2)
            self._evaluate_individual(ind1)
            self._evaluate_individual(ind2)

            new_population.extend([ind1, ind2])

        # Trim to population size and ensure diversity
        new_population = new_population[:self.config.population_size]
        if not self._check_time_limits():
            new_population = self._ensure_diversity(new_population)
        return new_population

    def _route_crossover(self, parent1: Solution, parent2: Solution) -> Solution:
        """Inherit complete routes from parent1 and fill rest from parent2."""
        child = parent1.clone()

        # Keep 50% of routes from parent1
        num_routes = len(parent1.truck_routes)
        if num_routes < 2:
            return child  # Fallback

        keep_indices = self.rng.sample(
            range(num_routes), max(1, num_routes // 2))
        kept_routes = [parent1.truck_routes[i] for i in keep_indices]

        # Identify customers served by kept routes
        served = set()
        for r in kept_routes:
            for node in r.nodes:
                if node not in self._depots:
                    served.add(node)

        # Rebuild child routes
        child.truck_routes = [r.clone() for r in kept_routes]

        # Get remaining customers from parent2 (or all unserved)
        remaining = []
        # Check parent2 for order preference?
        # Or just get all unserved customers from instance
        all_customers = set(self.instance.customer_manager.demands().keys())
        unserved = list(all_customers - served)

        # Shuffle unserved for randomness
        self.rng.shuffle(unserved)

        # Insert remaining customers (Greedy Insertion)
        # Simplified: just append to last route or new route for now,
        # rely on repair/mutation to fix.
        # Better: Try to insert into existing routes

        for cust in unserved:
            inserted = False
            for route in child.truck_routes:
                # Try inserting at end (before depot)
                route.nodes.insert(-1, cust)
                inserted = True  # Assume feasible for now, repair will fix
                break

            if not inserted:
                # Create new route
                from alns_vrpfd.model.route import TruckRoute
                new_route = TruckRoute(
                    route_id=len(child.truck_routes),
                    nodes=[self._depot_start, cust, self._depot_end],
                    capacity=self.instance.vehicle_specs['truck'].capacity
                )
                child.add_truck_route(new_route)

        # Drone tasks: inherit if possible
        child.drone_tasks = []

        # Collect all nodes served by kept truck routes for fast lookup
        kept_nodes = set()
        for r in child.truck_routes:
            for node in r.nodes:
                kept_nodes.add(node)

        for task in parent1.drone_tasks:
            # Check if task is valid with kept routes
            # 1. Launch/Retrieve nodes must exist in kept routes
            if task.launch_node not in kept_nodes or task.retrieve_node not in kept_nodes:
                continue

            # 2. None of the task's customers may already be served.
            # Drone sorties may contain multiple customers.
            task_customers = task.customers()
            if not task_customers or any(
                customer in served for customer in task_customers
            ):
                continue

            # Keep task
            child.drone_tasks.append(task.clone())
            served.update(task_customers)

        return child

    def _ensure_diversity(self, population: List[Individual]) -> List[Individual]:
        """Ensure population diversity by removing duplicate solutions."""
        seen_solutions = set()
        diverse_population = []

        for individual in population:
            # Create a hashable representation of the solution
            solution_hash = self._solution_hash(individual.solution)
            if solution_hash not in seen_solutions:
                seen_solutions.add(solution_hash)
                diverse_population.append(individual)
            else:
                served_customers = set(n for r in individual.solution.truck_routes for n in r.nodes if n not in
                                       self._depots and n is not None)
                if 4 in served_customers:
                    pass

        # If we have too few individuals after deduplication, keep the best ones
        # and allow some duplicates rather than creating new random solutions during runtime
        while len(diverse_population) < len(population):
            # Duplicate the best individual with slight mutation
            best_individual = diverse_population[0]
            mutated_solution = self._mutate(best_individual.solution.clone())
            mutated_individual = Individual(solution=mutated_solution)
            self._evaluate_individual(mutated_individual)
            diverse_population.append(mutated_individual)

        return diverse_population[:len(population)]

    def _solution_hash(self, solution: Solution) -> str:
        """Create a hashable representation of a solution for diversity checking."""
        # Simple hash based on route structures
        truck_part = tuple(tuple(route.nodes)
                           for route in solution.truck_routes)
        drone_part = tuple(tuple(task.nodes) for task in solution.drone_tasks)
        return str((truck_part, drone_part))

    def _tournament_selection(self) -> Solution:
        """Perform tournament selection."""
        candidates = self.rng.sample(
            self.population, self.config.tournament_size)
        return min(candidates, key=lambda x: x.fitness).solution

    def _crossover(self, parent1: Solution, parent2: Solution) -> Tuple[Solution, Solution]:
        """Perform crossover between two parent solutions."""
        child1 = parent1.clone()
        child2 = parent2.clone()

        # Truck route crossover using Order Crossover (OX)
        if (self.rng.random() < self.config.truck_route_crossover_rate and
                len(parent1.truck_routes) > 0 and len(parent2.truck_routes) > 0):

            for i in range(min(len(parent1.truck_routes), len(parent2.truck_routes))):
                route1 = parent1.truck_routes[i]
                route2 = parent2.truck_routes[i]

                if len(route1.nodes) > 4 and len(route2.nodes) > 4:  # Need enough customers
                    # Apply Order Crossover to customer sequence
                    child1.truck_routes[i], child2.truck_routes[i] = self._order_crossover(
                        route1, route2)

        # Drone task crossover - ensure no drone conflicts
        if (self.rng.random() < 0.5 and  # Lower probability for drone crossover
                len(parent1.drone_tasks) > 0 and len(parent2.drone_tasks) > 0):

            # Simple drone crossover: randomly select tasks from each parent
            # but ensure no drone ID conflicts
            child1_tasks = []
            child2_tasks = []

            # Collect all tasks from both parents
            all_tasks_p1 = parent1.drone_tasks[:]
            all_tasks_p2 = parent2.drone_tasks[:]

            # Randomly assign tasks to children while avoiding drone conflicts
            used_drones_c1 = set()
            used_drones_c2 = set()

            # Shuffle tasks to add randomness
            self.rng.shuffle(all_tasks_p1)
            self.rng.shuffle(all_tasks_p2)

            # Assign tasks to child1
            drone_count = self.instance.vehicle_specs['drone'].number
            for task in all_tasks_p1:
                if task.drone_id not in used_drones_c1:
                    child1_tasks.append(task)
                    used_drones_c1.add(task.drone_id)
                    if len(child1_tasks) >= min(drone_count, len(parent1.drone_tasks)):
                        break

            # Assign tasks to child2
            for task in all_tasks_p2:
                if task.drone_id not in used_drones_c2:
                    child2_tasks.append(task)
                    used_drones_c2.add(task.drone_id)
                    if len(child2_tasks) >= min(drone_count, len(parent2.drone_tasks)):
                        break

            child1.drone_tasks = child1_tasks
            child2.drone_tasks = child2_tasks

        return child1, child2

    def _order_crossover(self, route1: TruckRoute, route2: TruckRoute) -> Tuple[TruckRoute, TruckRoute]:
        """Apply Order Crossover (OX) to two truck routes."""
        # Extract customer nodes (exclude depots)
        customers1 = [
            node for node in route1.nodes if node not in self._depots]
        customers2 = [
            node for node in route2.nodes if node not in self._depots]

        if len(customers1) < 2 or len(customers2) < 2:
            return route1.clone(), route2.clone()

        # Select crossover points
        size = min(len(customers1), len(customers2))
        cx_point1 = self.rng.randint(0, size - 2)
        cx_point2 = self.rng.randint(cx_point1 + 1, size - 1)

        # Create offspring
        def create_offspring(parent1_customers, parent2_customers):
            # Copy segment from parent1
            offspring = [None] * len(parent1_customers)
            for i in range(cx_point1, cx_point2 + 1):
                offspring[i] = parent1_customers[i]

            # Fill remaining positions with parent2 customers in order
            pos = (cx_point2 + 1) % len(offspring)
            for customer in parent2_customers:
                if customer not in offspring:
                    offspring[pos] = customer
                    pos = (pos + 1) % len(offspring)
                    if pos == cx_point1:
                        pos = (cx_point2 + 1) % len(offspring)

            return offspring

        offspring1_customers = create_offspring(customers1, customers2)
        offspring2_customers = create_offspring(customers2, customers1)

        # Create new routes with dynamic depot IDs
        child1 = route1.clone()
        child2 = route2.clone()
        child1.nodes = [self._depot_start] + offspring1_customers + [self._depot_end]
        child2.nodes = [self._depot_start] + offspring2_customers + [self._depot_end]

        return child1, child2


    def _mutate(self, solution: Solution) -> Solution:
        """Perform mutation on a solution."""
        mutated = solution.clone()

        r = self.rng.random()

        # Route segment swap mutation for truck routes
        if r < 0.2 and len(mutated.truck_routes) > 0:
            route = self.rng.choice(mutated.truck_routes)
            if len(route.nodes) > 4:  # Need at least depot + 2 customers + depot
                # Swap two random customer positions
                customer_positions = [i for i, node in enumerate(
                    route.nodes) if node not in self._depots]
                if len(customer_positions) >= 2:
                    pos1, pos2 = self.rng.sample(customer_positions, 2)
                    route.nodes[pos1], route.nodes[pos2] = route.nodes[pos2], route.nodes[pos1]

        # Ruin and Recreate (ALNS-style - larger scale)
        elif r < 0.4 and len(mutated.truck_routes) > 0:
            # Collect all truck customers
            all_customers = []
            for route in mutated.truck_routes:
                for node in route.nodes:
                    if node not in self._depots:
                        all_customers.append(node)
            
            if all_customers:
                # Remove 10-20% of customers (ALNS-style)
                num_to_remove = max(2, int(len(all_customers) * 0.15))
                num_to_remove = min(num_to_remove, len(all_customers))
                removed = self.rng.sample(all_customers, num_to_remove)

                # Remove from all routes
                for route in mutated.truck_routes:
                    route.nodes = [n for n in route.nodes if n not in removed]

                # Re-insert using cheapest insertion
                for cust in removed:
                    self.repair._insert_customer_cheapest(mutated, cust)

        # Truck to Drone Mutation (40% chance - more aggressive drone optimization)
        elif r < 0.8 and len(mutated.truck_routes) > 0:
            # Pick one random eligible customer from truck routes
            drone_cap = self.instance.vehicle_specs['drone'].capacity
            demands = self.instance.customer_manager.demands()
            eligible_customers = []
            for route in mutated.truck_routes:
                for node in route.nodes:
                    if node not in self._depots and demands.get(node, 0) <= drone_cap:
                        eligible_customers.append((route, node))

            if eligible_customers:
                route, cust = self.rng.choice(eligible_customers)

                # Try all launch-retrieve combinations, then fully evaluate only the
                # best lightweight candidates.
                anchor_candidates = []
                for launch_route in mutated.truck_routes:
                    if self._check_time_limits():
                        break
                    for launch_pos in range(len(launch_route.nodes)):
                        if self._check_time_limits():
                            break
                        launch_node = launch_route.nodes[launch_pos]

                        for retrieve_route in mutated.truck_routes:
                            if self._check_time_limits():
                                break
                            for retrieve_pos in range(len(retrieve_route.nodes)):
                                if self._check_time_limits():
                                    break
                                retrieve_node = retrieve_route.nodes[retrieve_pos]

                                if launch_node == retrieve_node and launch_route.id == retrieve_route.id:
                                    continue

                                if not self._energy_model_ok(launch_node, [cust], retrieve_node):
                                    continue

                                anchor_candidates.append((
                                    route,
                                    cust,
                                    launch_route,
                                    launch_node,
                                    retrieve_route,
                                    retrieve_node,
                                ))

                best_task = None
                best_cost = float('inf')

                for _, _, launch_route, launch_node, retrieve_route, retrieve_node in (
                    self._rank_drone_anchor_candidates(anchor_candidates)
                ):
                    if self._check_time_limits():
                        break

                    # Create candidate solution and evaluate
                    test_sol = mutated.clone()
                    # Remove customer from truck
                    for r in test_sol.truck_routes:
                        if cust in r.nodes:
                            r.nodes.remove(cust)
                            break

                    from alns_vrpfd.model.route import DroneTask
                    from alns_vrpfd.core.operators.base import _build_payloads
                    task_id = max((t.task_id or 0) for t in test_sol.drone_tasks) + 1 if test_sol.drone_tasks else 1
                    payloads = _build_payloads([cust], self._demands)
                    new_task = DroneTask(
                        task_id=task_id,
                        drone_id=0,
                        launch_truck=launch_route.id if launch_node != self._depot_start else None,
                        launch_node=launch_node,
                        customers=[cust],
                        land_truck=retrieve_route.id if retrieve_node != self._depot_end else None,
                        retrieve_node=retrieve_node,
                        payloads=payloads,
                    )
                    test_sol.drone_tasks.append(new_task)

                    # Evaluate
                    try:
                        result = self._evaluate_solution(test_sol)
                        if result.feasible and result.total_cost < best_cost:
                            best_cost = result.total_cost
                            best_task = new_task
                    except Exception:
                        pass

                if best_task is not None:
                    for truck_route in mutated.truck_routes:
                        if cust in truck_route.nodes:
                            truck_route.nodes.remove(cust)
                            break
                    mutated.drone_tasks.append(best_task)

        # Improved drone task mutation - ensure no conflicts
        else:
            if len(mutated.drone_tasks) > 0:
                # Option 1: Remove a random drone task (70% probability)
                if self.rng.random() < 0.7:
                    if len(mutated.drone_tasks) > 0:
                        idx = self.rng.randint(0, len(mutated.drone_tasks) - 1)
                        removed_task = mutated.drone_tasks.pop(idx)

                        # Reassign the customers from the removed drone task to truck routes
                        for customer in removed_task.customers():
                            self.repair._insert_customer_cheapest(
                                mutated, customer)

                # Option 2: Change drone assignment for a task (30% probability)
                else:
                    if len(mutated.drone_tasks) > 0:
                        # Select a random task
                        task_idx = self.rng.randint(
                            0, len(mutated.drone_tasks) - 1)
                        task = mutated.drone_tasks[task_idx]

                        # Find available drone IDs (not used by other tasks)
                        used_drones = set(
                            t.drone_id for t in mutated.drone_tasks)
                        drone_count = self.instance.vehicle_specs['drone'].number
                        # Allow all drones (evaluator checks overlap)
                        all_drones = list(range(drone_count))

                        if all_drones:
                            # Assign to a different drone
                            new_drone_id = self.rng.choice(all_drones)
                            task.drone_id = new_drone_id

        return mutated

    def _energy_model_ok(self, launch: int, customers: List[int], retrieve: int) -> bool:
        """Check if a drone sortie is robust-energy-feasible."""
        if not customers:
            return False
        from alns_vrpfd.core.operators.base import _build_payloads
        from alns_vrpfd.evaluation.energy import DroneEnergyModel
        if not hasattr(self, '_energy_model'):
            self._energy_model = DroneEnergyModel()
            self._battery = self.instance.robust_config.drone_battery_capacity
            self._deviation_rate = self.instance.robust_config.energy_deviation_rate
            self._uncertainty_budget = self.instance.robust_config.energy_uncertainty_budget
            self._drone_time = self.instance.time_matrix("drone")
            self._node_index = {n: i for i, n in enumerate(self.instance.all_node_ids())}
            self._demands = self.instance.customer_manager.demands()
        payloads = _build_payloads(customers, self._demands)
        nodes = [launch] + customers + [retrieve]
        ni = self._node_index
        dt = self._drone_time
        nom, devs = 0.0, []
        for p, a, b in zip(payloads, nodes, nodes[1:]):
            try:
                t = dt[ni[a]][ni[b]]
            except (KeyError, IndexError):
                return False
            e = self._energy_model.energy_kwh(p, t)
            nom += e
            devs.append(e * self._deviation_rate)
        worst = nom + self._budgeted_sum(devs, self._uncertainty_budget)
        return worst <= self._battery + 1e-6

    @staticmethod
    def _budgeted_sum(values, budget):
        if not values or budget <= 0:
            return 0.0
        sorted_vals = sorted(values, reverse=True)
        integer = int(min(budget, len(sorted_vals)))
        fractional = max(0.0, budget - integer)
        total = sum(sorted_vals[:integer])
        if fractional > 0 and integer < len(sorted_vals):
            total += fractional * sorted_vals[integer]
        return total

    def _apply_simple_2opt(self, solution: Solution) -> Solution:
        """Apply simple 2-opt improvement to truck routes."""
        for route in solution.truck_routes:
            if self._check_time_limits():
                break
            nodes = route.nodes
            if len(nodes) <= 4:  # depot + 2 customers + depot min
                continue
            
            improved = True
            while improved:
                improved = False
                for i in range(1, len(nodes) - 2):
                    if self._check_time_limits():
                        return solution
                    for j in range(i + 2, len(nodes) - 1):
                        if self._check_time_limits():
                            return solution
                        # Check if reversing segment improves distance
                        try:
                            d1 = self.instance.distances.get('truck', nodes[i-1], nodes[i])
                            d2 = self.instance.distances.get('truck', nodes[j], nodes[j+1])
                            d3 = self.instance.distances.get('truck', nodes[i-1], nodes[j])
                            d4 = self.instance.distances.get('truck', nodes[i], nodes[j+1])
                            
                            if d3 + d4 < d1 + d2:
                                # Reverse segment [i, j]
                                route.nodes = nodes[:i] + nodes[i:j+1][::-1] + nodes[j+1:]
                                nodes = route.nodes
                                improved = True
                                break
                        except:
                            continue
                    if improved:
                        break
        
        return solution

    def _check_time_limits(self) -> bool:
        """Check if time limits have been exceeded."""
        elapsed = time.time() - self.start_time

        if self.config.time_limit and elapsed >= self.config.time_limit:
            return True

        if self.config.generation_time_limit:
            # Could implement per-generation time limit if needed
            pass

        return False

    def _record_statistics(self) -> None:
        """Record population statistics for this generation."""
        fitness_values = [ind.fitness for ind in self.population]
        feasible_count = sum(1 for ind in self.population if ind.feasible)

        # Count unique solutions
        seen_hashes = set()
        for ind in self.population:
            seen_hashes.add(self._solution_hash(ind.solution))
        unique_count = len(seen_hashes)

        self.stats['generations'].append(self.generation)
        self.stats['best_fitness'].append(self.population[0].fitness)
        self.stats['avg_fitness'].append(
            sum(fitness_values) / len(fitness_values))
        self.stats['feasible_count'].append(feasible_count)
        self.stats['unique_solutions'].append(unique_count)
        self.stats['elapsed_time'].append(time.time() - self.start_time)
        best_feasible = getattr(self, "best_feasible_individual", None)
        self.stats['best_feasible_fitness'].append(
            best_feasible.fitness if best_feasible is not None else float('inf')
        )

        # Simple diversity measure (standard deviation of fitness)
        if len(fitness_values) > 1:
            mean = sum(fitness_values) / len(fitness_values)
            variance = sum(
                (x - mean) ** 2 for x in fitness_values) / len(fitness_values)
            diversity = variance ** 0.5
        else:
            diversity = 0.0

        self.stats['diversity'].append(diversity)

    def _apply_local_search(self) -> None:
        """Apply local search to elite individuals."""
        elite_count = min(5, len(self.population)
                          )  # Apply to top 5 individuals

        for i in range(elite_count):
            if self._check_time_limits():
                break
            # Apply multiple local search operators
            improved = False
            improved |= self._local_search_2opt(self.population[i])
            improved |= self._local_search_relocate(self.population[i])
            improved |= self._local_search_relocate_inter_route(
                self.population[i])
            improved |= self._local_search_drone_optimization(
                self.population[i])

            # NEW: Apply drone chain optimization (ALNS-inspired)
            improved |= self._local_search_drone_chain(self.population[i])

            if improved:
                # Re-evaluate the improved individual
                self._evaluate_individual(self.population[i])

        # Re-sort population after local search
        self.population.sort()

    def _apply_aggressive_drone_optimization(self) -> None:
        """Try to add drone tasks to top individuals using ALNS operators."""
        top_n = min(getattr(self, "_aggressive_drone_top_n", 10), len(self.population))
        for i in range(top_n):
            if self._check_time_limits():
                break
            individual = self.population[i]
            solution = individual.solution
            
            original_drone_count = len(solution.drone_tasks)
            
            # First try ALNS DroneTaskSplitMergeLocalSearch for cross-truck sorties
            try:
                from alns_vrpfd.core.operators.drone_reanchor import DroneTaskSplitMergeLocalSearch, MultiCustomerSortieConstructor
                ls = DroneTaskSplitMergeLocalSearch(
                    instance=self.instance,
                    evaluator=self.evaluator,
                    max_moves=3,
                )
                improved = ls.apply(solution)
                result = self._evaluate_solution(improved)
                if result.feasible and result.total_cost < individual.fitness:
                    individual.solution = improved
                    individual.fitness = result.total_cost
                    individual.feasible = result.feasible
                    individual.truck_distance = result.truck_distance_cost
                    individual.drone_distance = result.drone_distance_cost
                    individual.delay_penalty = result.delay_penalty
                    continue
                
                # Also try MultiCustomerSortieConstructor
                sortie = MultiCustomerSortieConstructor(
                    instance=self.instance,
                    evaluator=self.evaluator,
                    max_customers=3,
                    top_k=5,
                    max_sorties=20,
                    rng=self.rng,
                )
                sortie_sol = sortie.apply(solution)
                sortie_result = self._evaluate_solution(sortie_sol)
                if sortie_result.feasible and sortie_result.total_cost < individual.fitness:
                    individual.solution = sortie_sol
                    individual.fitness = sortie_result.total_cost
                    individual.feasible = sortie_result.feasible
                    individual.truck_distance = sortie_result.truck_distance_cost
                    individual.drone_distance = sortie_result.drone_distance_cost
                    individual.delay_penalty = sortie_result.delay_penalty
                    continue
            except Exception:
                pass
            
            # Fallback to standard drone optimization
            if original_drone_count >= 5:
                continue
            
            optimized = self.drone_builder.optimize_drone_tasks(solution)
            optimized = self.repair.repair_solution(optimized)
            
            try:
                result = self._evaluate_solution(optimized)
                if result.feasible:
                    added_drones = len(optimized.drone_tasks) > original_drone_count
                    cost_ok = result.total_cost < individual.fitness * 1.10
                    if result.total_cost < individual.fitness or (added_drones and cost_ok):
                        individual.solution = optimized
                        individual.fitness = result.total_cost
                        individual.feasible = result.feasible
                        individual.truck_distance = result.truck_distance_cost
                        individual.drone_distance = result.drone_distance_cost
                        individual.delay_penalty = result.delay_penalty
            except Exception:
                pass
        
        self.population.sort()

    def _local_search_drone_chain(self, individual: Individual) -> bool:
        """Apply ALNS drone operators for cross-truck sorties."""
        solution = individual.solution
        original_cost = individual.fitness

        # Apply ALNS DroneTaskSplitMergeLocalSearch for cross-truck sorties
        try:
            from alns_vrpfd.core.operators.drone_reanchor import DroneTaskSplitMergeLocalSearch
            ls = DroneTaskSplitMergeLocalSearch(
                instance=self.instance,
                evaluator=self.evaluator,
                max_moves=5,
            )
            improved = ls.apply(solution)
            result = self._evaluate_solution(improved)
            if result.feasible and result.total_cost < original_cost:
                individual.solution = improved
                individual.fitness = result.total_cost
                individual.feasible = result.feasible
                individual.truck_distance = result.truck_distance_cost
                individual.drone_distance = result.drone_distance_cost
                individual.delay_penalty = result.delay_penalty
                return True
        except Exception:
            pass

        # Fallback: try to create more drone tasks from truck customers
        optimized = self.drone_builder.optimize_drone_tasks(solution)
        optimized = self.repair.repair_solution(optimized)

        try:
            result = self._evaluate_solution(optimized)
            if result.feasible and result.total_cost < original_cost:
                individual.solution = optimized
                individual.fitness = result.total_cost
                individual.feasible = result.feasible
                individual.truck_distance = result.truck_distance_cost
                individual.drone_distance = result.drone_distance_cost
                individual.delay_penalty = result.delay_penalty
                return True
        except:
            pass

        return False

    def _local_search_relocate_inter_route(self, individual: Individual) -> bool:
        """Apply inter-route relocate local search (move customer to different route)."""
        improved = False
        solution = individual.solution
        best_cost = individual.fitness

        demands = self.instance.customer_manager.demands()
        truck_capacity = self.instance.vehicle_specs['truck'].capacity

        # Iterate all pairs of routes
        for r1_idx, route1 in enumerate(solution.truck_routes):
            if len(route1.nodes) <= 2:
                continue

            customers1 = [n for n in route1.nodes if n not in self._depots]

            for cust in customers1:
                # Try to move cust to other routes
                for r2_idx, route2 in enumerate(solution.truck_routes):
                    if r1_idx == r2_idx:
                        continue

                    # Check capacity first
                    route2_demand = sum(demands.get(n, 0)
                                        for n in route2.nodes if n not in self._depots)
                    if route2_demand + demands.get(cust, 0) > truck_capacity:
                        continue

                    # Try insertion at all positions
                    # Optimization: Just try best position or random?
                    # Let's try all for elite search
                    temp_route2_nodes = list(route2.nodes)

                    best_insert_pos = -1
                    min_increase = float('inf')

                    # Find best insertion spot in route2 based on distance only (proxy)
                    for pos in range(1, len(temp_route2_nodes)):
                        prev = temp_route2_nodes[pos-1]
                        curr = temp_route2_nodes[pos]
                        d1 = self.instance.distances.get('truck', prev, cust)
                        d2 = self.instance.distances.get('truck', cust, curr)
                        d3 = self.instance.distances.get('truck', prev, curr)
                        increase = d1 + d2 - d3
                        if increase < min_increase:
                            min_increase = increase
                            best_insert_pos = pos

                    if best_insert_pos != -1:
                        # Create temp solution
                        temp_solution = solution.clone()

                        # Remove from r1
                        try:
                            temp_solution.truck_routes[r1_idx].nodes.remove(
                                cust)
                        except:
                            continue  # Should not happen

                        # Insert into r2
                        temp_solution.truck_routes[r2_idx].nodes.insert(
                            best_insert_pos, cust)

                        # Evaluate
                        try:
                            res = self._evaluate_solution(temp_solution)
                            if res.feasible and res.total_cost < best_cost:
                                # Accept
                                solution.truck_routes[r1_idx].nodes = temp_solution.truck_routes[r1_idx].nodes
                                solution.truck_routes[r2_idx].nodes = temp_solution.truck_routes[r2_idx].nodes
                                best_cost = res.total_cost
                                improved = True
                                break  # Next customer
                        except:
                            pass

                if improved:
                    break  # Restart/Next pair?

            if improved:
                break

        return improved

    def _local_search_2opt(self, individual: Individual) -> bool:
        """Apply 2-opt local search to truck routes."""
        improved = False
        solution = individual.solution
        best_cost = individual.fitness

        for route_idx, route in enumerate(solution.truck_routes):
            if len(route.nodes) < 6:  # Need at least 2 customers for 2-opt
                continue

            customer_nodes = [
                node for node in route.nodes if node not in self._depots]
            current_best = best_cost

            # Try 2-opt moves
            for i in range(len(customer_nodes) - 1):
                for j in range(i + 1, len(customer_nodes)):
                    # Create new route with 2-opt swap
                    new_customers = customer_nodes[:i] + \
                        customer_nodes[i:j+1][::-1] + customer_nodes[j+1:]
                    new_route_nodes = [self._depot_start] + \
                        new_customers + [self._depot_end]

                    # Create temporary solution for evaluation
                    temp_solution = solution.clone()
                    temp_solution.truck_routes[route_idx] = route.clone()
                    temp_solution.truck_routes[route_idx].nodes = new_route_nodes

                    # Evaluate the move
                    temp_individual = Individual(solution=temp_solution)
                    self._evaluate_individual(temp_individual)

                    if temp_individual.fitness < current_best and temp_individual.feasible:
                        # Accept the improvement
                        solution.truck_routes[route_idx].nodes = new_route_nodes
                        current_best = temp_individual.fitness
                        improved = True

            if improved:
                best_cost = current_best

        return improved

    def _local_search_relocate(self, individual: Individual) -> bool:
        """Apply relocate local search to truck routes (move customer to different position)."""
        improved = False
        solution = individual.solution
        best_cost = individual.fitness

        for route_idx, route in enumerate(solution.truck_routes):
            if len(route.nodes) < 4:  # Need at least depot + 1 customer + depot
                continue

            customer_nodes = [
                node for node in route.nodes if node not in self._depots]

            # Try relocating each customer to different positions
            for i, customer in enumerate(customer_nodes):
                # Remove customer from current position
                temp_route_nodes = [self._depot_start] + customer_nodes[:i] + \
                    customer_nodes[i+1:] + [self._depot_end]

                # Try inserting at different positions
                for insert_pos in range(1, len(temp_route_nodes)):  # Skip depot
                    new_route_nodes = temp_route_nodes[:insert_pos] + [
                        customer] + temp_route_nodes[insert_pos:]

                    # Create temporary solution for evaluation
                    temp_solution = solution.clone()
                    temp_solution.truck_routes[route_idx] = route.clone()
                    temp_solution.truck_routes[route_idx].nodes = new_route_nodes

                    # Evaluate the move
                    temp_individual = Individual(solution=temp_solution)
                    self._evaluate_individual(temp_individual)

                    if temp_individual.fitness < best_cost and temp_individual.feasible:
                        # Accept the improvement
                        solution.truck_routes[route_idx].nodes = new_route_nodes
                        best_cost = temp_individual.fitness
                        improved = True
                        break  # Move to next customer

                if improved:
                    break  # Restart with improved solution

        return improved

    def _local_search_drone_optimization(self, individual: Individual) -> bool:
        """Apply drone-specific local search operations."""
        improved = False
        solution = individual.solution
        best_cost = individual.fitness

        # Try removing unnecessary drone tasks
        for i in range(len(solution.drone_tasks) - 1, -1, -1):  # Reverse order
            task = solution.drone_tasks[i]

            # Create solution without this task
            temp_solution = solution.clone()
            removed_task = temp_solution.drone_tasks.pop(i)

            # Reassign customers to truck routes
            for customer in removed_task.customers():
                assigned = False
                demands = self.instance.customer_manager.demands()
                truck_capacity = self.instance.vehicle_specs['truck'].capacity

                for route in temp_solution.truck_routes:
                    route_customers = [
                        n for n in route.nodes if n not in self._depots]
                    route_demand = sum(demands.get(c, 0)
                                       for c in route_customers)

                    if route_demand + demands.get(customer, 0) <= truck_capacity:
                        # Find insertion position: before depot_end if exists, otherwise at end
                        if self._depot_end in route.nodes:
                            depot_idx = route.nodes.index(self._depot_end)
                            route.nodes.insert(depot_idx, customer)
                        else:
                            # Insert before the last node (assumed to be depot)
                            route.nodes.insert(len(route.nodes) - 1, customer)
                        assigned = True
                        break

                if not assigned:
                    # Create new route if necessary
                    new_route = TruckRoute(
                        route_id=len(temp_solution.truck_routes),
                        nodes=[self._depot_start, customer, self._depot_end],
                        capacity=truck_capacity
                    )
                    temp_solution.add_truck_route(new_route)

            # Evaluate the solution without this drone task
            temp_individual = Individual(solution=temp_solution)
            self._evaluate_individual(temp_individual)

            if temp_individual.fitness < best_cost and temp_individual.feasible:
                # Accept the improvement (remove the drone task)
                solution = temp_solution
                best_cost = temp_individual.fitness
                improved = True
                # Continue checking other tasks with the improved solution

        return improved

    def _adapt_parameters(self) -> None:
        """Adaptively adjust algorithm parameters based on population statistics."""
        if len(self.stats['diversity']) < 2:
            return

        current_diversity = self.stats['diversity'][-1]
        recent_diversities = self.stats['diversity'][-min(
            5, len(self.stats['diversity'])):]

        avg_recent_diversity = sum(
            recent_diversities) / len(recent_diversities)

        # If diversity is too low, increase mutation and decrease crossover
        if avg_recent_diversity < self.config.diversity_threshold:
            self.config.mutation_rate = min(
                0.3, self.config.mutation_rate * 1.1)
            self.config.crossover_rate = max(
                0.5, self.config.crossover_rate * 0.95)
        # If diversity is too high, decrease mutation and increase crossover
        elif avg_recent_diversity > self.config.diversity_threshold * 2:
            self.config.mutation_rate = max(
                0.01, self.config.mutation_rate * 0.9)
            self.config.crossover_rate = min(
                0.95, self.config.crossover_rate * 1.05)

    def get_statistics(self) -> Dict[str, Any]:
        """Get algorithm statistics."""
        return {
            'final_generation': self.generation,
            'total_time': time.time() - self.start_time,
            'best_fitness': self.best_individual.fitness if self.best_individual else float('inf'),
            'best_feasible': self.best_individual.feasible if self.best_individual else False,
            'stagnation_counter': self.stagnation_counter,
            'population_size': len(self.population),
            'stats_history': self.stats,
        }
