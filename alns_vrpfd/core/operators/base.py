"""Base classes and helpers for destroy operators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from alns_vrpfd.instance.manager import InstanceManager
from alns_vrpfd.model import DroneTask, Solution, TruckRoute
from alns_vrpfd.evaluation.energy import DroneEnergyModel

ANCHOR_DROP = "drop_tasks"
ANCHOR_REBASE = "rebase_to_neighbor"
_ANCHOR_POLICIES = {ANCHOR_DROP, ANCHOR_REBASE}

__all__ = [
    "DestroyOperator",
    "UnassignedPool",
    "CustomerAssignment",
    "budgeted_sum",
    "_build_payloads",
    "_segment_energy",
]


@dataclass
class UnassignedPool:
    """Container for customers removed by a destroy operator."""
    """Container for customers removed by a destroy operator."""

    customers: List[int] = field(default_factory=list)

    def add(self, customer_id: int) -> None:
        self.customers.append(customer_id)

    def extend(self, customer_ids: Iterable[int]) -> None:
        self.customers.extend(customer_ids)


@dataclass
class CustomerAssignment:
    """Record describing where a customer is currently served."""
    """Record describing where a customer is currently served."""

    customer_id: int
    kind: str  # "truck" or "drone"
    route: Optional[TruckRoute]
    task: Optional[DroneTask]
    prev_node: int
    next_node: int
    demand: float
    payload_before: Optional[float] = None
    payload_after: Optional[float] = None


class DestroyOperator(ABC):
    """Common scaffolding for destroy operators."""
    """Common scaffolding for destroy operators."""

    def __init__(
        self,
        instance: InstanceManager,
        *,
        anchor_strategy: str = ANCHOR_DROP,
        rng: Optional[random.Random] = None,
    ) -> None:
        if anchor_strategy not in _ANCHOR_POLICIES:
            raise ValueError(
                f"anchor_strategy must be one of {_ANCHOR_POLICIES}, got {anchor_strategy}."
            )
        self._instance = instance
        self._anchor_strategy = anchor_strategy
        # If no RNG provided, derive one deterministically from the global RNG state
        self._rng = rng or random.Random(random.getrandbits(32))
        self._demands = instance.customer_manager.demands()
        self._truck_distance = instance.distance_matrix("truck")
        self._drone_distance = instance.distance_matrix("drone")
        self._drone_time = instance.time_matrix("drone")
        self._node_ids = instance.all_node_ids()
        self._node_index = {node: idx for idx,
                            node in enumerate(self._node_ids)}
        self._energy_model = DroneEnergyModel()

    def apply(self, solution: Solution, count: int) -> Tuple[Solution, UnassignedPool]:
        """Remove up to ``count`` customers from the solution."""
        """Remove up to ``count`` customers from the solution."""

        assignments = self._collect_assignments(solution)
        if not assignments:
            return solution.clone(), UnassignedPool()

        limit = min(count, len(assignments))
        targets = self._select_customers(assignments, limit)
        if len(set(targets)) != len(targets):
            raise ValueError("Destroy operators must select unique customers.")

        mutated = solution.clone()
        pool = UnassignedPool()

        for customer_id in targets:
            self._remove_customer(mutated, customer_id, pool)

        for route in mutated.truck_routes:
            self._recalculate_truck_load(route)

        return mutated, pool

    @abstractmethod
    def _select_customers(
        self,
        assignments: Mapping[int, CustomerAssignment],
        count: int,
    ) -> List[int]:
        """Return the customers to be removed."""
        """Return the customers to be removed."""

    # ------------------------------------------------------------------
    def _collect_assignments(self, solution: Solution) -> Dict[int, CustomerAssignment]:
        assignments: Dict[int, CustomerAssignment] = {}
        # Truck routes
        for route in solution.truck_routes:
            nodes = route.nodes
            customers = route.customers()
            for idx, customer in enumerate(customers):
                node_index = idx + 1
                prev_node = nodes[node_index - 1]
                next_node = nodes[node_index + 1]
                assignments[customer] = CustomerAssignment(
                    customer_id=customer,
                    kind="truck",
                    route=route,
                    task=None,
                    prev_node=prev_node,
                    next_node=next_node,
                    demand=self._demands.get(customer, 0.0),
                )
        # Drone tasks
        for task in solution.drone_tasks:
            customers = task.customers()
            if not customers:
                continue
            payloads = _build_payloads(customers, self._demands)
            nodes = [task.launch_node, *customers, task.retrieve_node]
            for idx, customer in enumerate(customers):
                prev_node = nodes[idx]
                next_node = nodes[idx + 2]
                assignments[customer] = CustomerAssignment(
                    customer_id=customer,
                    kind="drone",
                    route=None,
                    task=task,
                    prev_node=prev_node,
                    next_node=next_node,
                    demand=self._demands.get(customer, 0.0),
                    payload_before=payloads[idx],
                    payload_after=payloads[idx + 1],
                )
        return assignments

    def _remove_customer(
        self,
        solution: Solution,
        customer_id: int,
        pool: UnassignedPool,
    ) -> None:
        # Try truck routes first
        for route in solution.truck_routes:
            if customer_id in route.customers():
                self._remove_from_truck_route(
                    solution, route, customer_id, pool)
                return
        # Fall back to drone tasks
        for task in list(solution.drone_tasks):
            if customer_id in task.customers():
                self._remove_from_drone_task(solution, task, customer_id, pool)
                return
        raise ValueError(f"Customer {customer_id} not present in solution.")

    def _remove_from_truck_route(
        self,
        solution: Solution,
        route: TruckRoute,
        customer_id: int,
        pool: UnassignedPool,
    ) -> None:
        """
        ：
        1.  (V(j)=2): ，/
        2.  (V(j)>2): /，
        3. : （）
        """
        nodes = route.nodes
        node_index = nodes.index(customer_id)
        prev_node = nodes[node_index - 1]
        next_node = nodes[node_index + 1]

        # V(j): j
        # = 2（ + ）
        # j +1（）
        # j +1（）
        vehicle_actions = self._count_vehicle_actions(solution, customer_id)

        if vehicle_actions == 2:
            # ：
            route.remove_customer(customer_id)
            pool.add(customer_id)
        else:
            # ： +
            affected_tasks = self._get_anchor_tasks(solution, customer_id)


            for task in affected_tasks:
                pool.extend(task.customers())
                if task in solution.drone_tasks:
                    solution.drone_tasks.remove(task)


            route.remove_customer(customer_id)
            pool.add(customer_id)

            # ：
            self._ensure_docking_consistency(solution, route, pool)

    def _count_vehicle_actions(self, solution: Solution, node: int) -> int:
        """
         V(j)
        -  = 2（）
        -  +1
        -  +1
        """
        actions = 2  # +
        for task in solution.drone_tasks:
            if task.launch_node == node:
                actions += 1
            if task.retrieve_node == node:
                actions += 1
        return actions

    def _get_anchor_tasks(self, solution: Solution, node: int) -> List[DroneTask]:
        """"""
        """获取以指定节点为发射或回收点的所有无人机任务"""
        return [
            task for task in solution.drone_tasks
            if task.launch_node == node or task.retrieve_node == node
        ]

    def _ensure_docking_consistency(
        self,
        solution: Solution,
        route: TruckRoute,
        pool: UnassignedPool,
    ) -> None:
        """
        ：/。

        （A，B），：
        1.  launch_truck 
        2.  land_truck 
        3. ，

        ，。
        """

        truck_route_nodes: Dict[int, set] = {}
        truck_route_position: Dict[int, Dict[int, int]] = {}
        for tr in solution.truck_routes:
            truck_route_nodes[tr.id] = set(tr.nodes)
            truck_route_position[tr.id] = {
                node: idx for idx, node in enumerate(tr.nodes)}

        tasks_to_remove: List[DroneTask] = []

        # ：/
        for task in solution.drone_tasks:
            launch_valid = True
            retrieve_valid = True


            if task.launch_truck is not None:
                launch_nodes = truck_route_nodes.get(task.launch_truck, set())
                if task.launch_node not in launch_nodes:
                    launch_valid = False
            # launch_truck=None ，0


            if task.land_truck is not None:
                land_nodes = truck_route_nodes.get(task.land_truck, set())
                if task.retrieve_node not in land_nodes:
                    retrieve_valid = False
            # land_truck=None ，0

            if not launch_valid or not retrieve_valid:
                tasks_to_remove.append(task)

        # ：

        drone_tasks_by_id: Dict[int, List[DroneTask]] = {}
        for task in solution.drone_tasks:
            if task not in tasks_to_remove:
                if task.drone_id not in drone_tasks_by_id:
                    drone_tasks_by_id[task.drone_id] = []
                drone_tasks_by_id[task.drone_id].append(task)

        for drone_id, tasks in drone_tasks_by_id.items():
            if len(tasks) < 2:
                continue


            # ，
            def get_task_order_key(t: DroneTask) -> Tuple[int, int]:
                """ (land_truck_id, retrieve_position) """
                """返回 (land_truck_id, retrieve_position) 用于排序"""
                if t.land_truck is None:
                    return (float('inf'), 0)
                positions = truck_route_position.get(t.land_truck, {})
                pos = positions.get(t.retrieve_node, 0)
                return (t.land_truck, pos)

            tasks.sort(key=get_task_order_key)

            # ：
            for i in range(len(tasks) - 1):
                curr_task = tasks[i]
                next_task = tasks[i + 1]

                # ，
                if curr_task.land_truck == next_task.launch_truck and curr_task.land_truck is not None:
                    positions = truck_route_position.get(
                        curr_task.land_truck, {})
                    curr_retrieve_pos = positions.get(
                        curr_task.retrieve_node, -1)
                    next_launch_pos = positions.get(next_task.launch_node, -1)

                    # <=
                    if curr_retrieve_pos > next_launch_pos:
                        tasks_to_remove.append(next_task)


        for task in set(tasks_to_remove):
            if task in solution.drone_tasks:
                pool.extend(task.customers())
                solution.drone_tasks.remove(task)

    def _remove_from_drone_task(
        self,
        solution: Solution,
        task: DroneTask,
        customer_id: int,
        pool: UnassignedPool,
    ) -> None:
        task.remove_customer(customer_id)
        remaining = task.customers()
        if not remaining:
            solution.drone_tasks.remove(task)
        else:
            task.payloads = _build_payloads(remaining, self._demands)
        pool.add(customer_id)

    def _handle_anchor(
        self,
        solution: Solution,
        anchor_node: int,
        prev_node: int,
        next_node: int,
        pool: UnassignedPool,
    ) -> None:
        """，。 _remove_from_truck_route"""
        """遗留方法，保持向后兼容。实际逻辑已移至 _remove_from_truck_route"""
        affected: List[DroneTask] = [
            task
            for task in solution.drone_tasks
            if task.launch_node == anchor_node or task.retrieve_node == anchor_node
        ]
        if not affected:
            return

        if self._anchor_strategy == ANCHOR_DROP:
            for task in affected:
                pool.extend(task.customers())
                solution.drone_tasks.remove(task)
            return

        # rebase
        for task in affected:
            launch = task.launch_node
            retrieve = task.retrieve_node
            if launch == anchor_node:
                new_launch = prev_node if prev_node != anchor_node else next_node
                if new_launch is None or new_launch == anchor_node:
                    pool.extend(task.customers())
                    solution.drone_tasks.remove(task)
                    continue
                task.nodes[0] = new_launch
            if task in solution.drone_tasks and retrieve == anchor_node:
                new_retrieve = next_node if next_node != anchor_node else prev_node
                if new_retrieve is None or new_retrieve == anchor_node:
                    pool.extend(task.customers())
                    solution.drone_tasks.remove(task)
                    continue
                task.nodes[-1] = new_retrieve

    def _recalculate_truck_load(self, route: TruckRoute) -> None:
        load = sum(self._demands.get(customer, 0.0)
                   for customer in route.customers())
        route.current_load = load


def _build_payloads(customers: Sequence[int], demands: Mapping[int, float]) -> List[float]:
    payloads: List[float] = []
    remaining = sum(demands.get(customer, 0.0) for customer in customers)
    payloads.append(remaining)
    for customer in customers:
        remaining -= demands.get(customer, 0.0)
        payloads.append(max(remaining, 0.0))
    return payloads


def _segment_distance(
    matrix: Sequence[Sequence[float]],
    node_index: Mapping[int, int],
    a: int,
    b: int,
) -> float:
    i = node_index.get(a)
    j = node_index.get(b)
    if i is None or j is None:
        return float("inf")
    return matrix[i][j]


def _segment_energy(
    energy_model: DroneEnergyModel,
    time_matrix: Sequence[Sequence[float]],
    node_index: Mapping[int, int],
    origin: int,
    destination: int,
    payload: float,
) -> float:
    i = node_index.get(origin)
    j = node_index.get(destination)
    if i is None or j is None:
        return float("inf")
    travel_time = time_matrix[i][j]
    if travel_time == float("inf"):
        return float("inf")
    return energy_model.energy_kwh(payload, travel_time)


def budgeted_sum(values: Sequence[float], budget: float) -> float:
    if not values or budget <= 0:
        return 0.0
    sorted_vals = sorted(values, reverse=True)
    integer = int(min(budget, len(sorted_vals)))
    fractional = max(0.0, budget - integer)
    total = sum(sorted_vals[:integer])
    if fractional > 0 and integer < len(sorted_vals):
        total += fractional * sorted_vals[integer]
    return total
