"""Drone task split/merge/reanchor composite operator.

Addresses the "neighborhood deficiency" where single-customer insertion
cannot reach solutions requiring simultaneous multi-customer drone sortie
restructuring.  This operator provides three move types:

1. **Re-anchor**: Change launch or retrieve anchor of a drone task to a
   different truck node or depot.
2. **Split**: Split a multi-customer drone task into two tasks, possibly
   with a different launch anchor for the back half.
3. **Merge**: Merge two tasks of the same drone into one task, trying
   several anchor combinations.

Also includes a composite destroy-repair variant (
DroneTaskReanchorRepair) that dissolves entire drone tasks and
re-inserts customers using the full repair framework.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

from alns_vrpfd.evaluation import Evaluator
from alns_vrpfd.evaluation.energy import DroneEnergyModel
from alns_vrpfd.evaluation.timing import TimingCalculator
from alns_vrpfd.instance.manager import InstanceManager
from alns_vrpfd.model import DroneTask, Solution
from alns_vrpfd.core.operators.truck_route_polish import SynchronizedTruckRoutePolish


# ------------------------------------------------------------------
# Statistics tracker
# ------------------------------------------------------------------
class OperatorStats:
    """Tracks attempts / feasible / accepted / best_delta for one move type."""

    __slots__ = ("attempts", "feasible", "accepted", "best_delta")

    def __init__(self) -> None:
        self.attempts: int = 0
        self.feasible: int = 0
        self.accepted: int = 0
        self.best_delta: float = 0.0

    def record(self, delta: float, feasible: bool, accepted: bool) -> None:
        self.attempts += 1
        if feasible:
            self.feasible += 1
        if accepted:
            self.accepted += 1
            if delta < self.best_delta:
                self.best_delta = delta

    def __repr__(self) -> str:
        return (
            f"Stats(attempts={self.attempts}, feasible={self.feasible}, "
            f"accepted={self.accepted}, best_delta={self.best_delta:.4f})"
        )


# ------------------------------------------------------------------
# Composite destroy-repair operator
# ------------------------------------------------------------------
class DroneTaskReanchorRepair:
    """Composite destroy-repair: dissolve drone tasks and re-insert freely.

    Removes 1-3 drone tasks at random, puts their customers back into
    the unassigned pool, and uses a repair operator to re-insert them.
    The repair operator can create multi-customer drone sorties,
    cross-truck retrieval, and depot-launched tasks — neighborhoods that
    single-customer ALNS destroy-repair cycles cannot reach.

    Parameters
    ----------
    instance : InstanceManager
    repair_operators : list of RepairOperator
        All available repair operators; the best result across all is kept.
    evaluator : Evaluator
    max_tasks : int
    rng : random.Random, optional
    """

    name = "DroneTaskReanchor"

    def __init__(
        self,
        instance: InstanceManager,
        repair_operators: list,
        evaluator: Evaluator,
        max_tasks: int = 3,
        rng: Optional[random.Random] = None,
    ) -> None:
        self._instance = instance
        self._repair_ops = repair_operators
        self._evaluator = evaluator
        self._max_tasks = max(1, max_tasks)
        self._rng = rng or random.Random(random.getrandbits(32))
        self.stats = OperatorStats()

    def apply(self, solution: Solution) -> Solution:
        if not solution.drone_tasks:
            return solution

        current_eval = self._evaluator.evaluate_solution(solution)
        if not current_eval.feasible or not math.isfinite(current_eval.total_cost):
            return solution

        target_cost = current_eval.total_cost
        n_remove = min(self._max_tasks, len(solution.drone_tasks))
        if n_remove == 0:
            return solution

        count = self._rng.randint(1, n_remove)
        task_indices = list(range(len(solution.drone_tasks)))
        self._rng.shuffle(task_indices)
        selected = sorted(task_indices[:count])

        removed_customers: List[int] = []
        for idx in selected:
            removed_customers.extend(solution.drone_tasks[idx].customers())

        mutated = solution.clone()
        removed_task_ids = set()
        for idx in reversed(selected):
            if idx < len(mutated.drone_tasks):
                removed_task_ids.add(id(mutated.drone_tasks[idx]))
                mutated.drone_tasks.pop(idx)

        orphan_ids: Set[int] = set()
        for task in list(mutated.drone_tasks):
            launch_ok = (
                task.launch_truck is None
                or any(
                    task.launch_node in r.nodes
                    for r in mutated.truck_routes
                    if r.id == task.launch_truck
                )
            )
            retrieve_ok = (
                task.land_truck is None
                or any(
                    task.retrieve_node in r.nodes
                    for r in mutated.truck_routes
                    if r.id == task.land_truck
                )
            )
            if not launch_ok or not retrieve_ok:
                removed_customers.extend(task.customers())
                orphan_ids.add(id(task))

        mutated.drone_tasks = [
            t for t in mutated.drone_tasks if id(t) not in orphan_ids
        ]

        unique_customers = list(dict.fromkeys(removed_customers))
        best = solution
        best_cost = target_cost

        for repair in self._repair_ops:
            self.stats.attempts += 1
            repaired = repair.apply(mutated, unique_customers)
            repaired_eval = self._evaluator.evaluate_solution(repaired)
            if repaired_eval.feasible and math.isfinite(repaired_eval.total_cost):
                self.stats.feasible += 1
                if repaired_eval.total_cost < best_cost - 1e-6:
                    best = repaired
                    best_cost = repaired_eval.total_cost
                    self.stats.accepted += 1

        delta = best_cost - target_cost
        self.stats.record(delta, feasible=True, accepted=best is not solution)
        return best


# ------------------------------------------------------------------
# Local search: re-anchor / split / merge
# ------------------------------------------------------------------
class DroneTaskSplitMergeLocalSearch:
    """Local search that tries split, merge, and re-anchor moves on drone tasks.

    Three move types:
    1. **Re-anchor**: Change launch or retrieve anchor of a drone task.
    2. **Split**: Split a multi-customer drone task into two tasks.
    3. **Merge**: Merge two tasks of the same drone into one task.

    Each candidate move is evaluated by the full evaluator; only accepted
    if it yields a lower total cost and is feasible.

    Statistics are tracked per move type in ``self.stats``.
    """

    name = "DroneTaskSplitMergeLS"

    def __init__(
        self,
        instance: InstanceManager,
        evaluator: Evaluator,
        max_moves: int = 20,
    ) -> None:
        self._instance = instance
        self._evaluator = evaluator
        self._max_moves = max_moves
        self._demands = instance.customer_manager.demands()
        self._drone_dist = instance.distance_matrix("drone")
        self._truck_dist = instance.distance_matrix("truck")
        self._drone_time = instance.time_matrix("drone")
        self._truck_time = instance.time_matrix("truck")
        node_ids = instance.all_node_ids()
        self._node_index = {n: idx for idx, n in enumerate(node_ids)}
        self._depot_start = instance.customer_manager.depot_start
        self._depot_end = instance.customer_manager.depot_end
        drone_spec = instance.vehicle_specs.get("drone")
        self._drone_capacity = drone_spec.capacity if drone_spec else 30.0
        self._drone_count = drone_spec.number if drone_spec else 2
        self._truck_capacity = (
            instance.vehicle_specs.get("truck").capacity
            if "truck" in instance.vehicle_specs
            else 500.0
        )
        self._battery = instance.robust_config.drone_battery_capacity
        self._deviation_rate = instance.robust_config.energy_deviation_rate
        self._uncertainty_budget = instance.robust_config.energy_uncertainty_budget
        self._energy_model = DroneEnergyModel()
        self.stats: Dict[str, OperatorStats] = {
            "reanchor": OperatorStats(),
            "split": OperatorStats(),
            "merge": OperatorStats(),
        }

    def apply(self, solution: Solution, unassigned: list[int] = None) -> Solution:
        unassigned = unassigned or []
        best = solution.clone()
        best_eval = self._evaluator.evaluate_solution(best)
        if not best_eval.feasible or not math.isfinite(best_eval.total_cost):
            return solution
        best_cost = best_eval.total_cost

        for _ in range(self._max_moves):
            improved = False

            new_cost = self._try_reanchor_moves(best, best_cost)
            if new_cost < best_cost - 1e-6:
                best_cost = new_cost
                improved = True

            new_cost = self._try_split_moves(best, best_cost)
            if new_cost < best_cost - 1e-6:
                best_cost = new_cost
                improved = True

            new_cost = self._try_merge_moves(best, best_cost)
            if new_cost < best_cost - 1e-6:
                best_cost = new_cost
                improved = True

            if not improved:
                break

        return best

    # ------------------------------------------------------------------
    # Helper: evaluate a candidate and return cost (inf if infeasible)
    # ------------------------------------------------------------------
    def _eval_cost(self, solution: Solution) -> float:
        ev = self._evaluator.evaluate_solution(solution)
        if ev.feasible and math.isfinite(ev.total_cost):
            return ev.total_cost
        return float("inf")

    # ------------------------------------------------------------------
    # Re-anchor: change launch or retrieve anchor of an existing task
    # ------------------------------------------------------------------
    def _try_reanchor_moves(self, solution: Solution, current_best_cost: float) -> float:
        best_cost = current_best_cost
        if not solution.drone_tasks or not solution.truck_routes:
            return best_cost

        for task in list(solution.drone_tasks):
            if len(task.customers()) == 0:
                continue

            anchor_candidates = []
            for route in solution.truck_routes:
                for node in route.customers():
                    anchor_candidates.append((route.id, node))
            anchor_candidates.append((None, self._depot_start))
            anchor_candidates.append((None, self._depot_end))

            for (truck_id, node) in anchor_candidates:
                is_depot = truck_id is None

                if not is_depot:
                    if node == task.launch_node and truck_id == task.launch_truck:
                        pass
                    else:
                        self.stats["reanchor"].attempts += 1
                        cand = self._clone_and_reanchor(
                            solution, task,
                            new_launch_truck=truck_id,
                            new_launch_node=node,
                        )
                        if cand is not None:
                            cost = self._eval_cost(cand)
                            if math.isfinite(cost) and cost < best_cost - 1e-6:
                                self.stats["reanchor"].feasible += 1
                                solution.truck_routes = cand.truck_routes
                                solution.drone_tasks = cand.drone_tasks
                                best_cost = cost
                                self.stats["reanchor"].accepted += 1

                    if node == task.retrieve_node and truck_id == task.land_truck:
                        pass
                    else:
                        self.stats["reanchor"].attempts += 1
                        cand = self._clone_and_reanchor(
                            solution, task,
                            new_retrieve_truck=truck_id,
                            new_retrieve_node=node,
                        )
                        if cand is not None:
                            cost = self._eval_cost(cand)
                            if math.isfinite(cost) and cost < best_cost - 1e-6:
                                self.stats["reanchor"].feasible += 1
                                solution.truck_routes = cand.truck_routes
                                solution.drone_tasks = cand.drone_tasks
                                best_cost = cost
                                self.stats["reanchor"].accepted += 1
                else:
                    if node == self._depot_start:
                        self.stats["reanchor"].attempts += 1
                        cand = self._clone_and_reanchor(
                            solution, task,
                            new_launch_truck=None,
                            new_launch_node=self._depot_start,
                        )
                        if cand is not None:
                            cost = self._eval_cost(cand)
                            if math.isfinite(cost) and cost < best_cost - 1e-6:
                                self.stats["reanchor"].feasible += 1
                                solution.truck_routes = cand.truck_routes
                                solution.drone_tasks = cand.drone_tasks
                                best_cost = cost
                                self.stats["reanchor"].accepted += 1

                    if node == self._depot_end:
                        self.stats["reanchor"].attempts += 1
                        cand = self._clone_and_reanchor(
                            solution, task,
                            new_retrieve_truck=None,
                            new_retrieve_node=self._depot_end,
                        )
                        if cand is not None:
                            cost = self._eval_cost(cand)
                            if math.isfinite(cost) and cost < best_cost - 1e-6:
                                self.stats["reanchor"].feasible += 1
                                solution.truck_routes = cand.truck_routes
                                solution.drone_tasks = cand.drone_tasks
                                best_cost = cost
                                self.stats["reanchor"].accepted += 1

        return best_cost

    # ------------------------------------------------------------------
    # Split: split a multi-customer drone task into two tasks
    # ------------------------------------------------------------------
    def _try_split_moves(self, solution: Solution, current_best_cost: float) -> float:
        best_cost = current_best_cost
        if not solution.drone_tasks:
            return best_cost

        for task in list(solution.drone_tasks):
            customers = task.customers()
            if len(customers) < 2:
                continue

            for split_pos in range(1, len(customers)):
                front = customers[:split_pos]
                back = customers[split_pos:]

                total_demand_front = sum(self._demands.get(c, 0) for c in front)
                total_demand_back = sum(self._demands.get(c, 0) for c in back)
                if total_demand_front > self._drone_capacity or total_demand_back > self._drone_capacity:
                    continue

                split_anchor_candidates = []

                split_anchor_candidates.append(
                    (task.land_truck, task.retrieve_node)
                )

                for route in solution.truck_routes:
                    for node in route.customers():
                        if node == task.retrieve_node and route.id == task.land_truck:
                            continue
                        split_anchor_candidates.append((route.id, node))

                split_anchor_candidates.append((None, self._depot_end))

                for (back_launch_truck, back_launch_node) in split_anchor_candidates:
                    back_land_truck = task.land_truck
                    back_retrieve_node = task.retrieve_node

                    if back_launch_truck is None and back_launch_node == self._depot_end:
                        back_land_truck = None

                    self.stats["split"].attempts += 1
                    cand = self._clone_and_split(
                        solution, task,
                        front=list(front),
                        back=list(back),
                        back_launch_truck=back_launch_truck,
                        back_launch_node=back_launch_node,
                        back_land_truck=back_land_truck,
                        back_retrieve_node=back_retrieve_node,
                    )
                    if cand is not None:
                        cost = self._eval_cost(cand)
                        if math.isfinite(cost) and cost < best_cost - 1e-6:
                            self.stats["split"].feasible += 1
                            solution.truck_routes = cand.truck_routes
                            solution.drone_tasks = cand.drone_tasks
                            best_cost = cost
                            self.stats["split"].accepted += 1

        return best_cost

    # ------------------------------------------------------------------
    # Merge: merge two tasks of the same drone into one task
    # ------------------------------------------------------------------
    def _try_merge_moves(self, solution: Solution, current_best_cost: float) -> float:
        best_cost = current_best_cost
        if len(solution.drone_tasks) < 2:
            return best_cost

        for i in range(len(solution.drone_tasks)):
            for j in range(i + 1, len(solution.drone_tasks)):
                task_a = solution.drone_tasks[i]
                task_b = solution.drone_tasks[j]

                if task_a.drone_id != task_b.drone_id:
                    continue

                merged_customers = task_a.customers() + task_b.customers()
                total_demand = sum(self._demands.get(c, 0) for c in merged_customers)
                if total_demand > self._drone_capacity:
                    continue

                anchor_combos = [
                    (task_a.launch_truck, task_a.launch_node, task_b.land_truck, task_b.retrieve_node),
                    (task_a.launch_truck, task_a.launch_node, task_a.land_truck, task_a.retrieve_node),
                    (task_b.launch_truck, task_b.launch_node, task_b.land_truck, task_b.retrieve_node),
                    (None, self._depot_start, task_b.land_truck, task_b.retrieve_node),
                    (None, self._depot_start, None, self._depot_end),
                ]

                for (lt, ln, rlt, rn) in anchor_combos:
                    self.stats["merge"].attempts += 1
                    cand = self._clone_and_merge(
                        solution, task_a, task_b,
                        launch_truck=lt, launch_node=ln,
                        land_truck=rlt, retrieve_node=rn,
                    )
                    if cand is not None:
                        cost = self._eval_cost(cand)
                        if math.isfinite(cost) and cost < best_cost - 1e-6:
                            self.stats["merge"].feasible += 1
                            solution.truck_routes = cand.truck_routes
                            solution.drone_tasks = cand.drone_tasks
                            best_cost = cost
                            self.stats["merge"].accepted += 1

        return best_cost

    # ------------------------------------------------------------------
    # Cloning helpers
    # ------------------------------------------------------------------
    def _clone_and_reanchor(
        self,
        solution: Solution,
        task: DroneTask,
        new_launch_truck=None,
        new_launch_node=None,
        new_retrieve_truck=None,
        new_retrieve_node=None,
    ) -> Optional[Solution]:
        cand = solution.clone()

        target_idx = None
        for idx, t in enumerate(cand.drone_tasks):
            if t.task_id == task.task_id:
                target_idx = idx
                break
            if t.launch_node == task.launch_node and t.retrieve_node == task.retrieve_node and t.drone_id == task.drone_id and list(t.customers()) == list(task.customers()):
                target_idx = idx
                break

        if target_idx is None:
            return None

        old_t = cand.drone_tasks[target_idx]
        lt = new_launch_truck if new_launch_truck is not None else old_t.launch_truck
        ln = new_launch_node if new_launch_node is not None else old_t.launch_node
        rlt = new_retrieve_truck if new_retrieve_truck is not None else old_t.land_truck
        rn = new_retrieve_node if new_retrieve_node is not None else old_t.retrieve_node

        if ln == rn and lt == rlt:
            return None

        total_demand = sum(self._demands.get(c, 0) for c in old_t.customers())
        if total_demand > self._drone_capacity:
            return None

        cand.drone_tasks[target_idx] = DroneTask(
            task_id=old_t.task_id,
            drone_id=old_t.drone_id,
            launch_truck=lt,
            launch_node=ln,
            customers=list(old_t.customers()),
            land_truck=rlt,
            retrieve_node=rn,
            payloads=self._build_payloads(list(old_t.customers())),
        )
        return cand

    def _clone_and_split(
        self,
        solution: Solution,
        task: DroneTask,
        front: List[int],
        back: List[int],
        back_launch_truck,
        back_launch_node: int,
        back_land_truck,
        back_retrieve_node: int,
    ) -> Optional[Solution]:
        cand = solution.clone()

        target_idx = None
        for idx, t in enumerate(cand.drone_tasks):
            if t.task_id == task.task_id:
                target_idx = idx
                break
        if target_idx is None:
            return None

        del cand.drone_tasks[target_idx]

        max_id = max((t.task_id or 0) for t in cand.drone_tasks) if cand.drone_tasks else 0

        front_task = DroneTask(
            task_id=max_id + 1,
            drone_id=task.drone_id,
            launch_truck=task.launch_truck,
            launch_node=task.launch_node,
            customers=list(front),
            land_truck=task.land_truck,
            retrieve_node=task.retrieve_node,
            payloads=self._build_payloads(list(front)),
        )

        back_task = DroneTask(
            task_id=max_id + 2,
            drone_id=task.drone_id,
            launch_truck=back_launch_truck,
            launch_node=back_launch_node,
            customers=list(back),
            land_truck=back_land_truck,
            retrieve_node=back_retrieve_node,
            payloads=self._build_payloads(list(back)),
        )

        cand.drone_tasks.append(front_task)
        cand.drone_tasks.append(back_task)
        return cand

    def _clone_and_merge(
        self,
        solution: Solution,
        task_a: DroneTask,
        task_b: DroneTask,
        launch_truck,
        launch_node: int,
        land_truck,
        retrieve_node: int,
    ) -> Optional[Solution]:
        merged_customers = task_a.customers() + task_b.customers()
        total_demand = sum(self._demands.get(c, 0) for c in merged_customers)
        if total_demand > self._drone_capacity:
            return None

        cand = solution.clone()

        remove_ids = {task_a.task_id, task_b.task_id}
        cand.drone_tasks = [t for t in cand.drone_tasks if t.task_id not in remove_ids]

        remove_objs = {id(task_a), id(task_b)}
        cand.drone_tasks = [t for t in cand.drone_tasks if id(t) not in remove_objs]

        max_id = max((t.task_id or 0) for t in cand.drone_tasks) if cand.drone_tasks else 0

        merged = DroneTask(
            task_id=max_id + 1,
            drone_id=task_a.drone_id,
            launch_truck=launch_truck,
            launch_node=launch_node,
            customers=merged_customers,
            land_truck=land_truck,
            retrieve_node=retrieve_node,
            payloads=self._build_payloads(merged_customers),
        )
        cand.drone_tasks.append(merged)
        return cand

    def _build_payloads(self, customers: List[int]) -> List[float]:
        payloads: List[float] = []
        remaining = sum(self._demands.get(c, 0.0) for c in customers)
        payloads.append(remaining)
        for c in customers:
            remaining -= self._demands.get(c, 0.0)
            payloads.append(max(remaining, 0.0))
        return payloads


class MultiCustomerSortieConstructor:
    """Construct multi-customer drone sorties by enumerating 2-3 customer
    combinations from truck routes, allowing cross-truck launch/retrieve.

    This addresses the ``neighborhood deficiency'' where ALNS single-customer
    insertion cannot reach solutions requiring simultaneous multi-customer
    drone sortie restructuring (e.g. MILP's ``5 -> 8 -> 10 -> 1'' sortie
    that serves two customers on one cross-truck sortie).

    The constructor:
    1. Collects customers currently on truck routes.
    2. Enumerates size-2 and size-3 subsets whose total demand <= drone capacity.
    3. For each subset, tries launch/retrieve anchor pairs drawn from the
       remaining truck-route nodes (and depot), with cross-truck allowed.
    4. Checks drone endurance (robust energy) and timing feasibility.
    5. Evaluates the full solution via the Evaluator; accepts only improvements.
    6. Caps candidates per route-pair (``top_k``) to prevent explosion.

    Parameters
    ----------
    instance : InstanceManager
    evaluator : Evaluator
    max_customers : int
        Maximum customers per sortie (2 or 3).
    top_k : int
        Keep only the top-k best candidates per (launch_route, retrieve_route) pair.
    max_sorties : int
        Maximum number of sorties to attempt per call.
    rng : random.Random, optional
    """

    name = "MultiCustSortie"

    def __init__(
        self,
        instance: InstanceManager,
        evaluator: Evaluator,
        max_customers: int = 3,
        top_k: int = 5,
        max_sorties: int = 20,
        rng: Optional[random.Random] = None,
        polish_after: bool = True,
        reanchor_after: bool = True,
    ) -> None:
        self._instance = instance
        self._evaluator = evaluator
        self._max_customers = max(2, min(max_customers, 3))
        self._top_k = top_k
        self._max_sorties = max_sorties
        self._rng = rng or random.Random(random.getrandbits(32))
        self._polish_after = polish_after
        self._reanchor_after = reanchor_after
        self._demands = instance.customer_manager.demands()
        self._drone_dist = instance.distance_matrix("drone")
        self._truck_dist = instance.distance_matrix("truck")
        self._drone_time = instance.time_matrix("drone")
        self._truck_time = instance.time_matrix("truck")
        node_ids = instance.all_node_ids()
        self._node_index = {n: idx for idx, n in enumerate(node_ids)}
        self._depot_start = instance.customer_manager.depot_start
        self._depot_end = instance.customer_manager.depot_end
        drone_spec = instance.vehicle_specs.get("drone")
        self._drone_capacity = drone_spec.capacity if drone_spec else 30.0
        self._drone_count = drone_spec.number if drone_spec else 2
        self._battery = instance.robust_config.drone_battery_capacity
        self._deviation_rate = instance.robust_config.energy_deviation_rate
        self._uncertainty_budget = instance.robust_config.energy_uncertainty_budget
        self._energy_model = DroneEnergyModel()
        self.stats: Dict[str, OperatorStats] = {
            "size2": OperatorStats(),
            "size3": OperatorStats(),
        }

    def apply(self, solution: Solution) -> Solution:
        """Try to add multi-customer drone sorties and evaluate improvement."""
        best = solution.clone()
        best_eval = self._evaluator.evaluate_solution(best)
        if not best_eval.feasible or not math.isfinite(best_eval.total_cost):
            return solution
        best_cost = best_eval.total_cost

        drone_ids_available: List[int] = list(range(self._drone_count))
        candidates_tried = 0
        improved = True

        while improved and candidates_tried < self._max_sorties:
            improved = False

            drone_customers_now: Set[int] = set()
            for task in best.drone_tasks:
                drone_customers_now.update(task.customers())

            truck_custs_now: List[int] = []
            for route in best.truck_routes:
                for n in route.customers():
                    if n not in drone_customers_now:
                        truck_custs_now.append(n)
            drone_custs_now: List[int] = []
            for task in best.drone_tasks:
                drone_custs_now.extend(task.customers())

            all_custs = list(dict.fromkeys(truck_custs_now + drone_custs_now))
            if len(all_custs) < 2:
                break

            anchor_nodes: List[tuple] = []
            for route in best.truck_routes:
                for node in route.nodes:
                    anchor_nodes.append((route.id, node))
            anchor_nodes.append((None, self._depot_start))
            anchor_nodes.append((None, self._depot_end))

            for size in (2, 3):
                if size > self._max_customers:
                    continue
                stat_key = f"size{size}"
                if stat_key not in self.stats:
                    self.stats[stat_key] = OperatorStats()

                combos = self._enumerate_with_permutations(all_custs, size)
                if not combos:
                    continue

                buckets: Dict[tuple, List[tuple]] = {}
                for combo in combos:
                    total_demand = sum(self._demands.get(c, 0.0) for c in combo)
                    if total_demand > self._drone_capacity:
                        continue

                    for launch_truck, launch_node in anchor_nodes:
                        idx_l = self._node_index.get(launch_node)
                        idx_c0 = self._node_index.get(combo[0])
                        if idx_l is None or idx_c0 is None:
                            continue
                        dist_to_first = self._drone_dist[idx_l][idx_c0]
                        if not math.isfinite(dist_to_first):
                            continue
                        time_to_first = self._drone_time[idx_l][idx_c0]

                        for retrieve_truck, retrieve_node in anchor_nodes:
                            if launch_node == retrieve_node and launch_truck == retrieve_truck:
                                continue
                            idx_r = self._node_index.get(retrieve_node)
                            idx_cn = self._node_index.get(combo[-1])
                            if idx_r is None or idx_cn is None:
                                continue
                            dist_from_last = self._drone_dist[idx_cn][idx_r]
                            time_from_last = self._drone_time[idx_cn][idx_r]
                            if not math.isfinite(dist_from_last):
                                continue

                            total_drone_time = time_to_first
                            total_drone_dist = dist_to_first
                            infeasible = False
                            for i in range(len(combo) - 1):
                                idx_ci = self._node_index.get(combo[i])
                                idx_cj = self._node_index.get(combo[i + 1])
                                if idx_ci is None or idx_cj is None:
                                    infeasible = True
                                    break
                                seg_dist = self._drone_dist[idx_ci][idx_cj]
                                seg_time = self._drone_time[idx_ci][idx_cj]
                                if not math.isfinite(seg_dist):
                                    infeasible = True
                                    break
                                total_drone_dist += seg_dist
                                total_drone_time += seg_time
                            if infeasible:
                                continue
                            total_drone_dist += dist_from_last
                            total_drone_time += time_from_last

                            if total_drone_time > self._battery:
                                continue

                            truck_saving = 0.0
                            for c in combo:
                                on_truck = False
                                for route in best.truck_routes:
                                    if c in route.nodes:
                                        ci = route.nodes.index(c)
                                        if ci > 0:
                                            truck_saving += self._truck_dist[self._node_index.get(route.nodes[ci - 1], 0)][self._node_index.get(c, 0)]
                                        if ci < len(route.nodes) - 1:
                                            truck_saving += self._truck_dist[self._node_index.get(c, 0)][self._node_index.get(route.nodes[ci + 1], 0)]
                                        if ci > 0 and ci < len(route.nodes) - 1:
                                            truck_saving -= self._truck_dist[self._node_index.get(route.nodes[ci - 1], 0)][self._node_index.get(route.nodes[ci + 1], 0)]
                                        on_truck = True
                                        break
                                if not on_truck:
                                    pass

                            is_cross_truck = (launch_truck is not None and retrieve_truck is not None
                                              and launch_truck != retrieve_truck)
                            cross_bonus = 2.0 if is_cross_truck else 0.0
                            estimated_benefit = truck_saving - total_drone_dist + cross_bonus

                            key = (launch_truck, retrieve_truck)
                            bucket = buckets.setdefault(key, [])
                            bucket.append((estimated_benefit, combo, launch_truck, launch_node, retrieve_truck, retrieve_node))

                for key, bucket in buckets.items():
                    bucket.sort(key=lambda x: -x[0])
                    buckets[key] = bucket[:self._top_k]

                all_sorted: List[tuple] = []
                for bucket in buckets.values():
                    all_sorted.extend(bucket)
                all_sorted.sort(key=lambda x: -x[0])

                if not all_sorted:
                    continue

                for benefit, combo, launch_truck, launch_node, retrieve_truck, retrieve_node in all_sorted:
                    if candidates_tried >= self._max_sorties:
                        break
                    self.stats[stat_key].attempts += 1
                    candidates_tried += 1

                    for chosen_drone in drone_ids_available:
                        cand = self._try_sortie(
                            best, combo, chosen_drone,
                            launch_truck, launch_node,
                            retrieve_truck, retrieve_node,
                        )
                        if cand is None:
                            self.stats[stat_key].record(0, feasible=False, accepted=False)
                            continue

                        # Composite acceptance: polish + reanchor before evaluating
                        if self._polish_after:
                            polisher = SynchronizedTruckRoutePolish(
                                self._instance, self._evaluator, max_iterations=15)
                            cand = polisher.apply(cand)
                        if self._reanchor_after:
                            reanchor = DroneTaskSplitMergeLocalSearch(
                                self._instance, self._evaluator, max_moves=5)
                            cand = reanchor.apply(cand)

                        cand_eval = self._evaluator.evaluate_solution(cand)
                        if cand_eval.feasible and math.isfinite(cand_eval.total_cost):
                            if cand_eval.total_cost < best_cost - 1e-6:
                                self.stats[stat_key].record(
                                    cand_eval.total_cost - best_cost, feasible=True, accepted=True)
                                best = cand
                                best_cost = cand_eval.total_cost
                                improved = True
                                break
                            else:
                                self.stats[stat_key].record(
                                    cand_eval.total_cost - best_cost, feasible=True, accepted=False)
                        else:
                            self.stats[stat_key].record(0, feasible=False, accepted=False)

                    if improved:
                        break

                if improved:
                    break

        return best

    def _enumerate_with_permutations(self, customers: List[int], size: int) -> List[tuple]:
        """Return all permutations of customer combinations of given size.

        For size=2, generates both (a,b) and (b,a) orders since drone
        visit sequence affects timing, energy, and cost.
        For size=3 and above, enumerates full permutations.
        Limits total permutations to avoid explosion.
        """
        from itertools import permutations
        if len(customers) < size:
            return []
        from itertools import combinations
        combos = list(combinations(customers, size))
        result: List[tuple] = []
        max_total = 200
        for combo in combos:
            for perm in permutations(combo):
                result.append(perm)
                if len(result) >= max_total:
                    return result
        return result

    def _try_sortie(
        self,
        solution: Solution,
        customers: tuple,
        drone_id: int,
        launch_truck: Optional[int],
        launch_node: int,
        retrieve_truck: Optional[int],
        retrieve_node: int,
    ) -> Optional[Solution]:
        """Attempt to create a multi-customer drone sortie.

        Customers may come from truck routes OR from existing drone tasks.
        Removes them from their current assignment and creates a new drone
        task serving all customers in the sortie.  Returns None if infeasible.
        """
        cust_list = list(customers)

        # Validate: all customers must exist somewhere in the solution
        for c in cust_list:
            found = False
            for route in solution.truck_routes:
                if c in route.customers():
                    found = True
                    break
            if not found:
                for task in solution.drone_tasks:
                    if c in task.customers():
                        found = True
                        break
            if not found:
                return None

        if launch_truck is not None:
            if launch_node in (self._depot_start, self._depot_end):
                return None
            if launch_node in cust_list:
                return None

        if retrieve_truck is not None:
            if retrieve_node in (self._depot_start, self._depot_end):
                return None
            if retrieve_node in cust_list:
                return None

        cand = solution.clone()

        # Remove customers from truck routes
        for c in cust_list:
            for route in cand.truck_routes:
                if c in route.customers():
                    new_nodes = [n for n in route.nodes if n != c]
                    if len(new_nodes) >= 2:
                        route.nodes = new_nodes
                    break

        # Remove customers from existing drone tasks (split/shorten tasks)
        tasks_to_remove: Set[int] = set()
        tasks_to_modify: List[Tuple[int, List[int]]] = []
        for idx, task in enumerate(cand.drone_tasks):
            overlap = [c for c in cust_list if c in task.customers()]
            if overlap:
                remaining = [c for c in task.customers() if c not in cust_list]
                if len(remaining) == 0:
                    tasks_to_remove.add(idx)
                elif len(remaining) > 0:
                    tasks_to_modify.append((idx, remaining))

        # Remove wholly-consumed drone tasks
        cand.drone_tasks = [t for i, t in enumerate(cand.drone_tasks) if i not in tasks_to_remove]

        # Shorten partially-consumed drone tasks
        for idx, remaining in tasks_to_modify:
            if idx in tasks_to_remove:
                continue
            if idx < len(cand.drone_tasks):
                task = cand.drone_tasks[idx]
                payloads = _build_payloads_static(remaining, self._demands)
                cand.drone_tasks[idx] = DroneTask(
                    task_id=task.task_id,
                    drone_id=task.drone_id,
                    launch_truck=task.launch_truck,
                    launch_node=task.launch_node,
                    customers=remaining,
                    land_truck=task.land_truck,
                    retrieve_node=task.retrieve_node,
                    payloads=payloads,
                )

        # Validate: launch node must still be on its truck route
        if launch_truck is not None:
            launch_route = None
            for r in cand.truck_routes:
                if r.id == launch_truck:
                    launch_route = r
                    break
            if launch_route is None:
                return None
            if launch_node not in launch_route.nodes:
                return None

        # Validate: retrieve node must still be on its truck route
        if retrieve_truck is not None:
            retrieve_route = None
            for r in cand.truck_routes:
                if r.id == retrieve_truck:
                    retrieve_route = r
                    break
            if retrieve_route is None:
                return None
            if retrieve_node not in retrieve_route.nodes:
                return None

        max_id = max((t.task_id or 0) for t in cand.drone_tasks) if cand.drone_tasks else 0

        payloads = _build_payloads_static(cust_list, self._demands)

        new_task = DroneTask(
            task_id=max_id + 1,
            drone_id=drone_id,
            launch_truck=launch_truck,
            launch_node=launch_node,
            customers=cust_list,
            land_truck=retrieve_truck,
            retrieve_node=retrieve_node,
            payloads=payloads,
        )
        cand.drone_tasks.append(new_task)
        return cand


def _build_payloads_static(customers: List[int], demands: dict) -> List[float]:
    payloads: List[float] = []
    remaining = sum(demands.get(c, 0.0) for c in customers)
    payloads.append(remaining)
    for c in customers:
        remaining -= demands.get(c, 0.0)
        payloads.append(max(remaining, 0.0))
    return payloads