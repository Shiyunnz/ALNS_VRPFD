"""Synchronized Truck Route Polish — Step 9 local search.

After a multi-customer drone sortie is constructed (Step 8), the affected
truck routes may have sub-optimal customer ordering.  This operator:

1. Collects the set of "affected" truck routes (those whose nodes appear
   as launch/retrieve anchors in any drone task, plus routes that lost
   customers to a new sortie).
2. Applies intra-route moves (2-opt reverse, Or-opt relocate) and
   inter-route moves (relocate one customer between routes, swap two
   customers between routes).
3. After each move, validates that all drone-task launch/retrieve anchors
   still exist on their respective truck routes.
4. Evaluates the full solution via the Evaluator — only accepts if
   total cost improves AND feasibility holds.

The goal is to discover reorderings like:
    [0,5,4,7,11] → [0,7,4,5,11]
that enable the drone sortie `T1@5 → [8,10] → T0@1` to actually improve
cost when evaluated with properly sequenced truck routes.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

from alns_vrpfd.evaluation import Evaluator
from alns_vrpfd.instance.manager import InstanceManager
from alns_vrpfd.model import DroneTask, Solution


class SynchronizedTruckRoutePolish:
    """Synchronized truck-route polish that respects drone anchor constraints.

    Moves:
    - 2-opt reverse within one route
    - Or-opt (relocate single customer within route)
    - Cross-route relocate (move one customer from route A to route B)
    - Cross-route swap (exchange one customer between two routes)

    All moves are validated against drone anchor integrity.
    """

    name = "TruckRouteSyncPolish"

    def __init__(
        self,
        instance: InstanceManager,
        evaluator: Evaluator,
        max_iterations: int = 30,
    ) -> None:
        self._instance = instance
        self._evaluator = evaluator
        self._max_iterations = max_iterations
        self._truck_dist = instance.distance_matrix("truck")
        self._demands = instance.customer_manager.demands()
        self._depot_start = instance.customer_manager.depot_start
        self._depot_end = instance.customer_manager.depot_end
        node_ids = instance.all_node_ids()
        self._node_index = {n: idx for idx, n in enumerate(node_ids)}
        truck_spec = instance.vehicle_specs.get("truck")
        self._truck_capacity = truck_spec.capacity if truck_spec else 500.0
        self.improvements: int = 0

    def apply(self, solution: Solution) -> Solution:
        best = solution.clone()
        best_eval = self._evaluator.evaluate_solution(best)
        if not best_eval.feasible or not math.isfinite(best_eval.total_cost):
            return solution
        best_cost = best_eval.total_cost

        for _ in range(self._max_iterations):
            improved = False

            improved |= self._try_2opt(best, best_cost)
            cand_eval = self._evaluator.evaluate_solution(best)
            if cand_eval.feasible and math.isfinite(cand_eval.total_cost):
                best_cost = cand_eval.total_cost
            else:
                continue

            improved |= self._try_or_opt(best, best_cost)
            cand_eval = self._evaluator.evaluate_solution(best)
            if cand_eval.feasible and math.isfinite(cand_eval.total_cost):
                best_cost = cand_eval.total_cost
            else:
                continue

            improved |= self._try_cross_relocate(best, best_cost)
            cand_eval = self._evaluator.evaluate_solution(best)
            if cand_eval.feasible and math.isfinite(cand_eval.total_cost):
                best_cost = cand_eval.total_cost
            else:
                continue

            improved |= self._try_cross_swap(best, best_cost)
            cand_eval = self._evaluator.evaluate_solution(best)
            if cand_eval.feasible and math.isfinite(cand_eval.total_cost):
                best_cost = cand_eval.total_cost

            if not improved:
                break

        return best

    def _anchor_nodes(self, solution: Solution) -> Dict[int, Set[int]]:
        """Return mapping truck_id -> set of nodes that are drone anchors."""
        anchors: Dict[int, Set[int]] = {}
        for task in solution.drone_tasks:
            if task.launch_truck is not None:
                anchors.setdefault(task.launch_truck, set()).add(task.launch_node)
            if task.land_truck is not None:
                anchors.setdefault(task.land_truck, set()).add(task.retrieve_node)
        return anchors

    def _anchors_valid(self, solution: Solution, anchors: Dict[int, Set[int]]) -> bool:
        """Check that all anchor nodes still exist on their truck routes."""
        for truck_id, nodes in anchors.items():
            route = None
            for r in solution.truck_routes:
                if r.id == truck_id:
                    route = r
                    break
            if route is None:
                return False
            for n in nodes:
                if n not in route.nodes:
                    return False
        return True

    # ------------------------------------------------------------------
    # 2-opt reverse
    # ------------------------------------------------------------------
    def _try_2opt(self, solution: Solution, current_best: float) -> bool:
        improved = False
        anchors = self._anchor_nodes(solution)
        for route in solution.truck_routes:
            nodes = route.nodes
            for i in range(1, len(nodes) - 2):
                for j in range(i + 2, len(nodes) - 1):
                    reversed_segment = nodes[i:j + 1][::-1]
                    new_nodes = nodes[:i] + reversed_segment + nodes[j + 1:]

                    if self._anchors_on_route(anchors, route.id, new_nodes):
                        old_nodes = route.nodes
                        route.nodes = new_nodes
                        if self._eval_accepts(solution, current_best):
                            improved = True
                            current_best = self._evaluator.evaluate_solution(solution).total_cost
                            anchors = self._anchor_nodes(solution)
                        else:
                            route.nodes = old_nodes
        return improved

    # ------------------------------------------------------------------
    # Or-opt relocate within route
    # ------------------------------------------------------------------
    def _try_or_opt(self, solution: Solution, current_best: float) -> bool:
        improved = False
        anchors = self._anchor_nodes(solution)
        for route in solution.truck_routes:
            nodes = route.nodes
            for i in range(1, len(nodes) - 1):
                cust = nodes[i]
                if cust in anchors.get(route.id, set()):
                    continue
                remaining = nodes[:i] + nodes[i + 1:]
                for j in range(1, len(remaining)):
                    if j == i:
                        continue
                    new_nodes = remaining[:j] + [cust] + remaining[j:]
                    if self._anchors_on_route(anchors, route.id, new_nodes):
                        old_nodes = route.nodes
                        route.nodes = new_nodes
                        if self._eval_accepts(solution, current_best):
                            improved = True
                            current_best = self._evaluator.evaluate_solution(solution).total_cost
                            anchors = self._anchor_nodes(solution)
                        else:
                            route.nodes = old_nodes
        return improved

    # ------------------------------------------------------------------
    # Cross-route relocate: move one customer from route A to route B
    # ------------------------------------------------------------------
    def _try_cross_relocate(self, solution: Solution, current_best: float) -> bool:
        if len(solution.truck_routes) < 2:
            return False
        improved = False
        anchors = self._anchor_nodes(solution)
        routes = solution.truck_routes

        for ri, route_a in enumerate(routes):
            for ci in range(1, len(route_a.nodes) - 1):
                cust = route_a.nodes[ci]
                if cust in anchors.get(route_a.id, set()):
                    continue
                demand = self._demands.get(cust, 0.0)

                for rj, route_b in enumerate(routes):
                    if ri == rj:
                        continue
                    current_load_b = sum(self._demands.get(n, 0.0) for n in route_b.customers())
                    if current_load_b + demand > self._truck_capacity:
                        continue

                    remaining_a = route_a.nodes[:ci] + route_a.nodes[ci + 1:]
                    if len(remaining_a) < 2:
                        continue

                    for pos in range(1, len(route_b.nodes)):
                        new_b = route_b.nodes[:pos] + [cust] + route_b.nodes[pos:]

                        if not self._anchors_on_route(anchors, route_a.id, remaining_a):
                            continue
                        if not self._anchors_on_route(anchors, route_b.id, new_b):
                            continue

                        old_a = route_a.nodes
                        old_b = route_b.nodes
                        route_a.nodes = remaining_a
                        route_b.nodes = new_b

                        if self._eval_accepts(solution, current_best):
                            improved = True
                            current_best = self._evaluator.evaluate_solution(solution).total_cost
                            anchors = self._anchor_nodes(solution)
                        else:
                            route_a.nodes = old_a
                            route_b.nodes = old_b
        return improved

    # ------------------------------------------------------------------
    # Cross-route swap: exchange one customer between two routes
    # ------------------------------------------------------------------
    def _try_cross_swap(self, solution: Solution, current_best: float) -> bool:
        if len(solution.truck_routes) < 2:
            return False
        improved = False
        anchors = self._anchor_nodes(solution)
        routes = solution.truck_routes

        for ri, route_a in enumerate(routes):
            for ci in range(1, len(route_a.nodes) - 1):
                cust_a = route_a.nodes[ci]
                if cust_a in anchors.get(route_a.id, set()):
                    continue
                demand_a = self._demands.get(cust_a, 0.0)

                for rj, route_b in enumerate(routes):
                    if rj <= ri:
                        continue
                    load_b = sum(self._demands.get(n, 0.0) for n in route_b.customers())
                    load_a = sum(self._demands.get(n, 0.0) for n in route_a.customers())
                    load_a_without = load_a - demand_a

                    for cj in range(1, len(route_b.nodes) - 1):
                        cust_b = route_b.nodes[cj]
                        if cust_b in anchors.get(route_b.id, set()):
                            continue
                        demand_b = self._demands.get(cust_b, 0.0)

                        if load_a_without + demand_b > self._truck_capacity:
                            continue
                        if load_b - demand_b + demand_a > self._truck_capacity:
                            continue

                        new_a = route_a.nodes[:ci] + [cust_b] + route_a.nodes[ci + 1:]
                        new_b = route_b.nodes[:cj] + [cust_a] + route_b.nodes[cj + 1:]

                        if not self._anchors_on_route(anchors, route_a.id, new_a):
                            continue
                        if not self._anchors_on_route(anchors, route_b.id, new_b):
                            continue

                        old_a = route_a.nodes
                        old_b = route_b.nodes
                        route_a.nodes = new_a
                        route_b.nodes = new_b

                        if self._eval_accepts(solution, current_best):
                            improved = True
                            current_best = self._evaluator.evaluate_solution(solution).total_cost
                            anchors = self._anchor_nodes(solution)
                        else:
                            route_a.nodes = old_a
                            route_b.nodes = old_b
        return improved

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _anchors_on_route(self, anchors: Dict[int, Set[int]], route_id: int, new_nodes: List[int]) -> bool:
        """Check that all anchor nodes for this route are still in new_nodes."""
        if route_id not in anchors:
            return True
        for n in anchors[route_id]:
            if n not in new_nodes:
                return False
        return True

    def _eval_accepts(self, solution: Solution, current_best: float) -> bool:
        """Evaluate solution and return True if it improves."""
        ev = self._evaluator.evaluate_solution(solution)
        if ev.feasible and math.isfinite(ev.total_cost) and ev.total_cost < current_best - 1e-6:
            self.improvements += 1
            return True
        return False