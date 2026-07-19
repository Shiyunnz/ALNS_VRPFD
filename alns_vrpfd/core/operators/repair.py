"""Repair operators with unified truck/drone insertion framework."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from alns_vrpfd.evaluation.energy import DroneEnergyModel
from alns_vrpfd.evaluation.timing import TimingCalculator, TruckRouteTiming
from alns_vrpfd.instance.manager import InstanceManager
from alns_vrpfd.model import DroneTask, Solution, TruckRoute

from .base import _build_payloads, _segment_energy


@dataclass
class Candidate:
    customer_id: int
    kind: str  # "truck" or "drone"
    route_index: int
    position: int
    payloads: Optional[List[float]] = None
    drone_task_index: Optional[int] = None
    drone_launch: Optional[int] = None
    drone_retrieve: Optional[int] = None
    drone_id: Optional[int] = None
    launch_route_index: Optional[int] = None
    retrieve_route_index: Optional[int] = None
    delta_distance: float = 0.0
    delta_energy: float = 0.0
    delta_delay: float = 0.0
    score: float = 0.0
    time_slack: float = float("inf")


class RepairOperator:
    """Base class for traditional repair operators."""
    """Base class for traditional repair operators."""

    def __init__(
        self,
        instance: InstanceManager,
        *,
        k_insert: int = 3,
        wait_max: float | None = None,
        weights: Tuple[float, float, float] = (1.0, 1.0, 0.0),
        rng: Optional[random.Random] = None,
        drone_priority: float = 0.0,
        depot_bonus: float = 0.0,
        multi_customer_bonus: float = 0.0,
        multi_customer_threshold: int = 2,
        forced_drone_customers: Sequence[int] | None = None,
        allow_multiple_launch_per_node: bool = True,
        max_launch_slots_per_customer: int = 0,
        max_same_route_retrieves_per_launch: int = 0,
        max_cross_route_retrieves_per_launch: int = 0,
        max_new_task_candidates_per_customer: int = 0,
        robust_energy_mode: str = "embedded",
        energy_model: DroneEnergyModel | None = None,
    ) -> None:
        self._instance = instance
        self._k_insert = max(1, k_insert)
        self._wait_max = wait_max if wait_max is not None else 0.5
        self._weights = weights
        self._rng = rng or random.Random(random.getrandbits(32))
        self._drone_priority = drone_priority
        self._depot_start = instance.customer_manager.depot_start
        self._depot_bonus = max(0.0, depot_bonus)
        self._multi_customer_bonus = max(0.0, multi_customer_bonus)
        self._multi_customer_threshold = max(2, multi_customer_threshold)
        self._forced_drone_customers = set(forced_drone_customers or [])
        # MIP allows different drones to launch/land at the same node
        self._allow_multiple_launch_per_node = allow_multiple_launch_per_node
        self._demands = instance.customer_manager.demands()
        self._truck_dist = instance.distance_matrix("truck")
        self._drone_dist = instance.distance_matrix("drone")
        self._drone_time = instance.time_matrix("drone")
        self._truck_time = instance.time_matrix("truck")
        self._node_index = {node: idx for idx,
                            node in enumerate(instance.all_node_ids())}
        self._battery = instance.robust_config.drone_battery_capacity
        self._deviation_rate = instance.robust_config.energy_deviation_rate
        self._uncertainty_budget = instance.robust_config.energy_uncertainty_budget
        drone_spec = instance.vehicle_specs.get("drone")
        truck_spec = instance.vehicle_specs.get("truck")
        self._drone_count = drone_spec.number if drone_spec is not None else 1
        self._truck_capacity = truck_spec.capacity if truck_spec is not None else float(
            "inf")
        self._drone_capacity = drone_spec.capacity if drone_spec is not None else float(
            "inf")
        self._truck_unit_cost = truck_spec.unit_cost if truck_spec is not None else 1.0
        self._drone_unit_cost = drone_spec.unit_cost if drone_spec is not None else 0.5
        self._energy_model = energy_model or DroneEnergyModel()
        self._time_window_weight = 2.0
        self._max_launch_slots_per_customer = max(
            0, int(max_launch_slots_per_customer))
        self._max_same_route_retrieves_per_launch = max(
            0, int(max_same_route_retrieves_per_launch))
        self._max_cross_route_retrieves_per_launch = max(
            0, int(max_cross_route_retrieves_per_launch))
        self._max_new_task_candidates_per_customer = max(
            0, int(max_new_task_candidates_per_customer))
        mode = str(robust_energy_mode).strip().lower()
        if mode not in {"embedded", "verification"}:
            raise ValueError(
                "robust_energy_mode must be 'embedded' or 'verification'."
            )
        self._robust_energy_mode = mode

    def apply(self, solution: Solution, unassigned: Iterable[int]) -> Solution:
        """Apply repair operator following Algorithm B1 from the paper.

        Steps:
        1. For each customer to insert, generate all candidate positions
           (truck route, existing drone task, or new drone task)
        2. Sort candidates by cost increment in ascending order
        3. Check each candidate for feasibility (load + energy constraints)
        4. Execute insertion at first feasible position
        """
        mutated = solution.clone()
        pool = list(unassigned)
        while pool:
            customer = self._select_customer(mutated, pool)
            pool.remove(customer)
            candidate = self._choose_candidate(mutated, customer)
            if candidate is None:
                continue
            self._apply_candidate(mutated, candidate)
        return mutated

    # ------------------------------------------------------------------
    def _select_customer(self, solution: Solution, pool: List[int]) -> int:
        manager = self._instance.customer_manager

        def priority(customer: int) -> tuple[float, float, int]:
            window = manager.time_window(customer)
            if not window:
                # No hard window: treat as very relaxed.
                return (float("inf"), float("inf"), customer)
            start, end = window
            if start is None or end is None:
                # No hard window: treat as very relaxed.
                return (float("inf"), float("inf"), customer)
            slack = max(0.0, end - start)
            # Smaller slack first, then earlier deadline; final tie-breaker keeps determinism.
            return (slack, end, customer)

        return min(pool, key=priority)

    def _choose_candidate(self, solution: Solution, customer: int) -> Optional[Candidate]:
        candidates = self._generate_candidates(solution, customer)
        if not candidates:
            return None

        # ：
        if customer in self._forced_drone_customers:
            drone_candidates = [c for c in candidates if c.kind == "drone"]
            if drone_candidates:
                self._normalise(drone_candidates)
                return min(drone_candidates, key=lambda cand: cand.score)
            # ， None
            return None

        self._normalise(candidates)
        return min(candidates, key=lambda cand: cand.score)

    # ------------------------------------------------------------------
    def _generate_candidates(self, solution: Solution, customer: int) -> List[Candidate]:
        calculator = self._build_timing_calculator()
        truck_timing = {
            route.id: calculator.truck_timing(route) for route in solution.truck_routes
        }
        drone_launch_times: Dict[int, float] = {}
        for task in solution.drone_tasks:
            launch_route, _ = self._resolve_route_ref(
                solution, task.launch_truck)
            if launch_route is None:
                launch_time = 0.0
            else:
                timing = truck_timing.get(launch_route.id)
                launch_time = timing.departure_times.get(
                    task.launch_node, 0.0) if timing else 0.0
            drone_launch_times[task.task_id or id(task)] = launch_time

        candidates: List[Candidate] = []
        candidates.extend(self._truck_candidates(
            solution, customer, truck_timing))
        candidates.extend(self._drone_task_candidates(
            solution, customer, calculator, truck_timing, drone_launch_times))
        candidates.extend(self._drone_new_task_candidates(
            solution, customer, calculator, truck_timing))
        # NEW: Add depot launch candidates for customers far from truck routes
        candidates.extend(self._drone_depot_launch_candidates(
            solution, customer, calculator, truck_timing))
        return self._filter_valid_candidates(solution, candidates)

    def _build_timing_calculator(self) -> TimingCalculator:
        return TimingCalculator(
            node_index=self._node_index,
            truck_time_matrix=self._instance.time_matrix("truck"),
            drone_time_matrix=self._drone_time,
        )

    def _truck_candidates(
        self,
        solution: Solution,
        customer: int,
        truck_timing: Mapping[int, "TruckRouteTiming"],
    ) -> List[Candidate]:
        candidates: List[Candidate] = []
        for route_index, route in enumerate(solution.truck_routes):
            nodes = route.nodes
            for pos in range(len(nodes) - 1):
                a = nodes[pos]
                b = nodes[pos + 1]
                arrival_a = truck_timing[route.id].arrival_times.get(a)
                if arrival_a is None:
                    continue
                travel_ac = self._transition_time(a, customer)
                travel_cb = self._transition_time(customer, b)
                if travel_ac == float("inf") or travel_cb == float("inf"):
                    continue
                delta = (
                    self._segment_distance(a, customer)
                    + self._segment_distance(customer, b)
                    - self._segment_distance(a, b)
                )
                arrival_customer = arrival_a + travel_ac
                window = self._instance.customer_manager.time_window(customer)
                if window and window[0] is not None and window[1] is not None:
                    start, end = window
                    service_start = max(arrival_customer, start)
                    slack = end - service_start
                    if slack < 0.0:
                        continue
                else:
                    slack = float("inf")
                cand = Candidate(
                    customer_id=customer,
                    kind="truck",
                    route_index=route_index,
                    position=pos + 1,
                    delta_distance=delta,
                    time_slack=slack,
                )
                candidates.append(cand)
        return candidates

    def _drone_task_candidates(
        self,
        solution: Solution,
        customer: int,
        calculator: TimingCalculator,
        truck_timing: Mapping[int, "TruckRouteTiming"],
        launch_times: Mapping[int, float],
    ) -> List[Candidate]:
        candidates: List[Candidate] = []
        # ALLOW zero demand customers
        # demand = self._demands.get(customer, 0.0)
        # if demand <= 0.0:
        #     return candidates
        for task_index, task in enumerate(solution.drone_tasks):
            task_id = task.task_id or id(task)
            base_launch_time = launch_times.get(task_id, 0.0)
            base_customers = task.customers()
            base_energy = self._worst_case_energy(
                task.launch_node, task.retrieve_node, base_customers)
            base_nodes = [task.launch_node, *
                          base_customers, task.retrieve_node]
            base_distance = self._drone_route_distance(base_nodes)
            base_timing = calculator.drone_timing(
                task, launch_time=base_launch_time)
            positions = len(base_customers) + 1
            step = max(1, math.ceil(positions / self._k_insert))

            # Handle depot tasks (land_truck=None)
            is_depot_task = task.land_truck is None

            if is_depot_task:
                # For depot tasks, no truck timing constraint
                truck_arrival = None
            else:
                _, retrieve_route_idx = self._resolve_route_ref(
                    solution, task.land_truck)
                retrieve_route = None
                truck_info = None
                if retrieve_route_idx is not None:
                    retrieve_route = solution.truck_routes[retrieve_route_idx]
                    truck_info = truck_timing.get(retrieve_route.id)
                if retrieve_route is None or truck_info is None:
                    continue
                truck_arrival = truck_info.arrival_times.get(
                    task.retrieve_node)
                if truck_arrival is None:
                    continue

            for pos in range(0, positions, step):
                new_customers = base_customers[:]
                new_customers.insert(pos, customer)
                payloads = _build_payloads(new_customers, self._demands)
                energy = self._worst_case_energy(
                    task.launch_node, task.retrieve_node, new_customers)
                if energy > self._battery:
                    continue
                retrieve_time = self._retrieve_time(
                    task.launch_node, new_customers, task.retrieve_node, base_launch_time)

                # For non-depot tasks, check timing constraint
                if not is_depot_task:
                    if abs(retrieve_time - truck_arrival) > self._wait_max:
                        continue

                new_nodes = [task.launch_node, *
                             new_customers, task.retrieve_node]
                new_distance = self._drone_route_distance(new_nodes)
                if new_distance == float("inf") or base_distance == float("inf"):
                    continue
                delta_energy = energy - base_energy
                delta_delay = max(0.0, retrieve_time -
                                  base_timing.retrieve_time) if not is_depot_task else 0.0
                cand = Candidate(
                    customer_id=customer,
                    kind="drone",
                    route_index=-1,
                    position=pos,
                    payloads=payloads,
                    drone_task_index=task_index,
                    drone_launch=task.launch_node,
                    drone_retrieve=task.retrieve_node,
                    drone_id=task.drone_id,
                    launch_route_index=task.launch_truck,
                    retrieve_route_index=task.land_truck,
                    delta_distance=new_distance - base_distance,
                    delta_energy=delta_energy,
                    delta_delay=delta_delay,
                )
                candidates.append(cand)
        return candidates

    def _get_drone_current_truck(self, solution: Solution, drone_id: int) -> int | None:
        """Get the truck where the drone is currently located after all its tasks.

        A drone's physical location is determined by its LAST task's land_truck.
        If the drone has no tasks, it's assumed to be at its initial position (depot/truck 0).

        This is critical for flexible docking: if a drone lands on truck B,
        its next task MUST launch from truck B.
        """
        drone_tasks = [
            t for t in solution.drone_tasks if t.drone_id == drone_id]
        if not drone_tasks:
            return None  # Free to start anywhere

        # The last task determines current location
        # Note: We need to determine task order, which depends on timing
        # For simplicity, assume tasks are in execution order
        last_task = drone_tasks[-1]
        return last_task.land_truck

    def _get_drone_last_retrieve_node(self, solution: Solution, drone_id: int) -> int | None:
        """Get the retrieve_node of the drone's last task.

        This determines where the drone is physically located and where it can
        launch from next.
        """
        drone_tasks = [
            t for t in solution.drone_tasks if t.drone_id == drone_id]
        if not drone_tasks:
            return None  # Free to start anywhere

        last_task = drone_tasks[-1]
        return last_task.retrieve_node

    def _get_compatible_drones(self, solution: Solution, truck_id: int, launch_node: int = None) -> List[int]:
        """Get list of drones that can launch from the given truck at the given node.

        A drone can launch from truck_id at launch_node only if:
        1. It has no tasks yet (free drone), OR
        2. Its last task landed on truck_id AND at a node that is BEFORE launch_node
           on the truck's route (physically present and available)

        This ensures physical consistency in flexible docking scenarios.
        """
        compatible = []

        # Get truck route to check node ordering
        truck_route = None
        for route in solution.truck_routes:
            if route.id == truck_id:
                truck_route = route
                break

        # Build node position map for this truck
        node_positions = {}
        if truck_route:
            for pos, node in enumerate(truck_route.nodes):
                node_positions[node] = pos

        for d in range(self._drone_count):
            current_truck = self._get_drone_current_truck(solution, d)
            last_retrieve_node = self._get_drone_last_retrieve_node(
                solution, d)

            if current_truck is None:
                # Free drone - can start from any truck at any position
                compatible.append(d)
            elif current_truck == truck_id:
                # Drone is physically on this truck
                # Check if launch_node is AFTER the drone's last retrieve_node
                if launch_node is None:
                    compatible.append(d)
                elif last_retrieve_node is None:
                    compatible.append(d)
                elif last_retrieve_node in node_positions and launch_node in node_positions:
                    # launch_node must be at or after retrieve_node
                    if node_positions[launch_node] >= node_positions[last_retrieve_node]:
                        compatible.append(d)
                else:
                    # Can't verify ordering, be conservative
                    compatible.append(d)
            # else: Drone is on a different truck, cannot launch from here

        return compatible

    def _get_occupied_nodes(self, solution: Solution) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], int]]:
        """Get all nodes occupied by existing drone tasks (launch and land).

        MIP constraint: Each node can only have ONE drone launch/land operation.
        This means we need to track which (truck_id, node) pairs are already used.

        Returns:
            Tuple of (launch_occupied, land_occupied) where each is a dict
            mapping (truck_id, node) -> drone_id that occupies it.
        """
        launch_occupied: Dict[Tuple[int, int],
                              int] = {}  # (truck_id, node) -> drone_id
        # (truck_id, node) -> drone_id
        land_occupied: Dict[Tuple[int, int], int] = {}

        # Iterate through all drone tasks in the solution
        for task in solution.drone_tasks:
            drone_id = task.drone_id
            launch_node = task.launch_node
            land_node = task.retrieve_node
            launch_truck = task.launch_truck
            land_truck = task.land_truck

            # Record launch node occupation
            if launch_truck is not None:
                launch_occupied[(launch_truck, launch_node)] = drone_id

            # Record land node occupation
            if land_truck is not None:
                land_occupied[(land_truck, land_node)] = drone_id

        return launch_occupied, land_occupied

    def _is_node_available_for_launch(
        self, launch_occupied: Dict[Tuple[int, int], int],
        truck_id: int, node: int, drone_id: int
    ) -> bool:
        """Check if a node is available for drone launch.

        When allow_multiple_launch_per_node is True (MIP-compatible mode):
        - Different drones CAN launch from the same node
        - This matches the MIP formulation where drones are independent

        When allow_multiple_launch_per_node is False (restrictive mode):
        - A node is available only if:
          1. No other drone launches from (truck_id, node), OR
          2. The same drone_id is already using this launch node (extending task)
        """
        # MIP allows multiple drones to launch from the same node
        if self._allow_multiple_launch_per_node:
            return True

        key = (truck_id, node)
        if key not in launch_occupied:
            return True
        return launch_occupied[key] == drone_id

    def _is_node_available_for_land(
        self, land_occupied: Dict[Tuple[int, int], int],
        truck_id: int, node: int, drone_id: int
    ) -> bool:
        """Check if a node is available for drone landing.

        When allow_multiple_launch_per_node is True (MIP-compatible mode):
        - Different drones CAN land at the same node
        - This matches the MIP formulation where drones are independent

        When allow_multiple_launch_per_node is False (restrictive mode):
        - A node is available only if:
          1. No other drone lands at (truck_id, node), OR
          2. The same drone_id is already using this land node
        """
        # MIP allows multiple drones to land at the same node
        if self._allow_multiple_launch_per_node:
            return True

        key = (truck_id, node)
        if key not in land_occupied:
            return True
        return land_occupied[key] == drone_id

    def _drone_new_task_candidates(
        self,
        solution: Solution,
        customer: int,
        calculator: TimingCalculator,
        truck_timing: Mapping[int, "TruckRouteTiming"],
    ) -> List[Candidate]:
        candidates: List[Candidate] = []
        # ALLOW zero demand
        # demand = self._demands.get(customer, 0.0)
        # if demand <= 0.0:
        #     return candidates

        same_truck_only = self._instance.robust_config.same_truck_retrieval
        launch_occupied, land_occupied = self._get_occupied_nodes(solution)
        depot_end = self._instance.customer_manager.depot_end or self._depot_start
        payloads = _build_payloads([customer], self._demands)

        # Build launch slots once, then optionally prune by an inexpensive geometric proxy.
        launch_slots: List[tuple[int, TruckRouteTiming, int, int, float, List[int], float]] = []
        for launch_route_index, launch_route in enumerate(solution.truck_routes):
            launch_info = truck_timing[launch_route.id]
            launch_nodes = launch_route.nodes
            for launch_pos in range(len(launch_nodes) - 1):
                launch_node = launch_nodes[launch_pos]
                launch_time = launch_info.departure_times.get(launch_node, 0.0)
                launch_to_customer = self._drone_segment_distance(
                    launch_node, customer)
                if not math.isfinite(launch_to_customer):
                    continue

                compatible_drones = self._get_compatible_drones(
                    solution, launch_route.id, launch_node)
                if not compatible_drones:
                    continue

                available_drones = [
                    drone_id for drone_id in compatible_drones
                    if self._is_node_available_for_launch(
                        launch_occupied,
                        launch_route.id,
                        launch_node,
                        drone_id,
                    )
                ]
                if not available_drones:
                    continue

                launch_slots.append(
                    (
                        launch_route_index,
                        launch_info,
                        launch_pos,
                        launch_node,
                        launch_time,
                        available_drones,
                        launch_to_customer,
                    )
                )

        if self._max_launch_slots_per_customer > 0 and len(launch_slots) > self._max_launch_slots_per_customer:
            launch_slots.sort(key=lambda item: item[6])
            launch_slots = launch_slots[:self._max_launch_slots_per_customer]

        for launch_slot in launch_slots:
            (
                launch_route_index,
                _launch_info,
                launch_pos,
                launch_node,
                launch_time,
                available_drones,
                launch_to_customer,
            ) = launch_slot

            retrieve_same: List[tuple[int, int, float, bool]] = []
            retrieve_cross: List[tuple[int, int, float, bool]] = []
            retrieve_depot: tuple[int, int, float, bool] | None = None

            for retrieve_route_index, retrieve_route in enumerate(solution.truck_routes):
                if same_truck_only and retrieve_route_index != launch_route_index:
                    continue

                retrieve_nodes = retrieve_route.nodes
                start_pos = 1
                if retrieve_route_index == launch_route_index:
                    start_pos = max(start_pos, launch_pos + 1)
                for retrieve_pos in range(start_pos, len(retrieve_nodes)):
                    retrieve_node = retrieve_nodes[retrieve_pos]
                    if retrieve_node == launch_node:
                        continue

                    customer_to_retrieve = self._drone_segment_distance(
                        customer, retrieve_node)
                    if not math.isfinite(customer_to_retrieve):
                        continue

                    is_depot_retrieve = retrieve_node in {
                        self._depot_start, depot_end}
                    proxy_score = launch_to_customer + customer_to_retrieve
                    slot = (retrieve_route_index, retrieve_node,
                            proxy_score, is_depot_retrieve)
                    if is_depot_retrieve:
                        if retrieve_depot is None or slot[2] < retrieve_depot[2]:
                            retrieve_depot = slot
                        continue
                    if retrieve_route_index == launch_route_index:
                        retrieve_same.append(slot)
                    else:
                        retrieve_cross.append(slot)

            if (
                self._max_same_route_retrieves_per_launch > 0
                and len(retrieve_same) > self._max_same_route_retrieves_per_launch
            ):
                retrieve_same.sort(key=lambda item: item[2])
                retrieve_same = retrieve_same[:self._max_same_route_retrieves_per_launch]

            if (
                self._max_cross_route_retrieves_per_launch > 0
                and len(retrieve_cross) > self._max_cross_route_retrieves_per_launch
            ):
                retrieve_cross.sort(key=lambda item: item[2])
                retrieve_cross = retrieve_cross[:self._max_cross_route_retrieves_per_launch]

            retrieve_slots = retrieve_same + retrieve_cross
            if retrieve_depot is not None:
                retrieve_slots.append(retrieve_depot)
            retrieve_slots.sort(key=lambda item: item[2])

            for retrieve_route_index, retrieve_node, _proxy_score, is_depot_retrieve in retrieve_slots:
                if is_depot_retrieve:
                    truck_arrival = 0.0
                else:
                    retrieve_route = solution.truck_routes[retrieve_route_index]
                    retrieve_info = truck_timing[retrieve_route.id]
                    truck_arrival = retrieve_info.arrival_times.get(retrieve_node)
                    if truck_arrival is None:
                        continue

                energy = self._worst_case_energy(
                    launch_node, retrieve_node, [customer])
                if energy > self._battery:
                    continue

                retrieve_time = self._retrieve_time(
                    launch_node, [customer], retrieve_node, launch_time)
                if retrieve_time == float("inf"):
                    continue

                if (not is_depot_retrieve) and abs(retrieve_time - truck_arrival) > self._wait_max:
                    continue

                selected_drone = None
                for drone_id in available_drones:
                    if is_depot_retrieve:
                        selected_drone = drone_id
                        break
                    if self._is_node_available_for_land(
                        land_occupied,
                        solution.truck_routes[retrieve_route_index].id,
                        retrieve_node,
                        drone_id,
                    ):
                        selected_drone = drone_id
                        break

                if selected_drone is None:
                    continue

                new_nodes = [launch_node, customer, retrieve_node]
                new_distance = self._drone_route_distance(new_nodes)
                if new_distance == float("inf"):
                    continue

                actual_retrieve_route_index = None if is_depot_retrieve else retrieve_route_index
                candidates.append(
                    Candidate(
                        customer_id=customer,
                        kind="drone",
                        route_index=launch_route_index,
                        position=0,
                        payloads=payloads,
                        drone_task_index=None,
                        drone_launch=launch_node,
                        drone_retrieve=retrieve_node,
                        drone_id=selected_drone,
                        launch_route_index=launch_route_index,
                        retrieve_route_index=actual_retrieve_route_index,
                        delta_distance=new_distance,
                        delta_energy=energy,
                        delta_delay=max(
                            0.0, retrieve_time - truck_arrival) if not is_depot_retrieve else 0.0,
                    )
                )

                if (
                    self._max_new_task_candidates_per_customer > 0
                    and len(candidates) >= self._max_new_task_candidates_per_customer
                ):
                    return candidates

        return candidates

    def _drone_depot_launch_candidates(
        self,
        solution: Solution,
        customer: int,
        calculator: TimingCalculator,
        truck_timing: Mapping[int, "TruckRouteTiming"],
    ) -> List[Candidate]:
        """Generate candidates for depot-based drone tasks.

        This enables MILP-like solutions where drones launch directly from depot
        to serve customers that are far from truck routes.
        """
        candidates: List[Candidate] = []
        # ALLOW zero demand
        # demand = self._demands.get(customer, 0.0)
        # if demand <= 0.0:
        #     return candidates

        depot_start = self._depot_start
        depot_end = self._instance.customer_manager.depot_end or depot_start
        if depot_start is None:
            return candidates

        # Find available drones (not assigned to any truck)
        drone_assignments = {d: set() for d in range(self._drone_count)}
        for task in solution.drone_tasks:
            if task.launch_truck is not None:
                drone_assignments[task.drone_id].add(task.launch_truck)
            if task.land_truck is not None:
                drone_assignments[task.drone_id].add(task.land_truck)

        free_drones = [d for d in range(self._drone_count)
                       if not drone_assignments[d]]

        # Also consider drones already doing depot tasks (launch_truck=None)
        depot_drones = set()
        for task in solution.drone_tasks:
            if task.launch_truck is None:
                depot_drones.add(task.drone_id)

        # Keep deterministic ordering across runs/processes.
        available_drones = sorted(set(free_drones) | depot_drones)
        if not available_drones:
            return candidates

        # For depot launch, drone starts at time 0
        launch_time = 0.0
        launch_node = depot_start

        # Try different retrieve points
        # Option 1: Return to depot_end (pure depot-to-depot task)
        retrieve_candidates = [depot_end]

        # Option 2: Return to truck route endpoint (depot_end in truck route)
        for route in solution.truck_routes:
            if depot_end in route.nodes:
                retrieve_candidates.append(depot_end)
                break

        for retrieve_node in sorted(set(retrieve_candidates)):
            payloads = _build_payloads([customer], self._demands)
            energy = self._worst_case_energy(
                launch_node, retrieve_node, [customer])
            if energy > self._battery:
                continue

            retrieve_time = self._retrieve_time(
                launch_node, [customer], retrieve_node, launch_time)
            if retrieve_time == float("inf"):
                continue

            # For depot-based tasks, no truck arrival timing constraint
            # (drone returns to depot independently)
            new_nodes = [launch_node, customer, retrieve_node]
            new_distance = self._drone_route_distance(new_nodes)
            if new_distance == float("inf"):
                continue

            drone_id = available_drones[0]

            cand = Candidate(
                customer_id=customer,
                kind="drone",
                route_index=-1,  # Not associated with a truck route
                position=0,
                payloads=payloads,
                drone_task_index=None,
                drone_launch=launch_node,
                drone_retrieve=retrieve_node,
                drone_id=drone_id,
                launch_route_index=None,  # Depot launch
                retrieve_route_index=None,  # Depot retrieve
                delta_distance=new_distance,
                delta_energy=energy,
                delta_delay=0.0,  # No delay for depot tasks
            )
            candidates.append(cand)

        return candidates

    # ------------------------------------------------------------------
    def _normalise(self, candidates: List[Candidate]) -> None:
        # Convert delta_distance to delta_cost
        cost_vals = []
        for cand in candidates:
            unit_cost = self._truck_unit_cost if cand.kind == "truck" else self._drone_unit_cost
            cost_vals.append(cand.delta_distance * unit_cost)

        e_vals = [cand.delta_energy for cand in candidates]
        l_vals = [cand.delta_delay for cand in candidates]

        c_min, c_max = min(cost_vals), max(cost_vals)
        e_min, e_max = min(e_vals), max(e_vals)
        l_min, l_max = min(l_vals), max(l_vals)

        for i, cand in enumerate(candidates):
            cost_val = cost_vals[i]
            c_norm = 0.0 if c_max == c_min else (
                cost_val - c_min) / (c_max - c_min)
            e_norm = 0.0 if e_max == e_min else (
                cand.delta_energy - e_min) / (e_max - e_min)
            l_norm = 0.0 if l_max == l_min else (
                cand.delta_delay - l_min) / (l_max - l_min)

            cand.score = (
                self._weights[0] * c_norm
                + self._weights[1] * e_norm
                + self._weights[2] * l_norm
            )
            if cand.kind == "drone":
                bonus = self._drone_priority
                if (
                    self._depot_start is not None
                    and cand.drone_launch is not None
                    and cand.drone_launch == self._depot_start
                ):
                    bonus += self._depot_bonus
                payload_points = len(cand.payloads) if cand.payloads else 0
                customer_count = max(0, payload_points - 1)
                if customer_count >= self._multi_customer_threshold:
                    bonus += self._multi_customer_bonus

                # NEW: Energy utilization bonus for drone candidates
                # Encourage high battery utilization (closer to MILP behavior)
                if self._battery > 0 and cand.delta_energy > 0:
                    utilization = cand.delta_energy / self._battery
                    if utilization >= 0.80:
                        bonus += 0.3  # Bonus for high utilization
                    elif utilization >= 0.65:
                        bonus += 0.1  # Small bonus for moderate utilization

                cand.score = max(0.0, cand.score - bonus)
            if math.isfinite(cand.time_slack):
                if cand.time_slack < 0.0:
                    penalty = self._time_window_weight * \
                        (1.0 + abs(cand.time_slack))
                else:
                    penalty = self._time_window_weight / \
                        (1.0 + cand.time_slack)
                cand.score += penalty

    def _apply_candidate(self, solution: Solution, cand: Candidate) -> None:
        if cand.kind == "truck":
            route = solution.truck_routes[cand.route_index]
            # ：
            if cand.customer_id in route.nodes:
                return
            route.nodes.insert(cand.position, cand.customer_id)
            route.current_load += self._demands.get(cand.customer_id, 0.0)
        else:
            if cand.drone_task_index is not None and cand.drone_task_index >= 0:
                task = solution.drone_tasks[cand.drone_task_index]
                # ：
                if cand.customer_id in task.nodes:
                    return
                task.nodes.insert(cand.position + 1, cand.customer_id)
                task.payloads = list(cand.payloads or [])
            else:
                new_id = (max((t.task_id or 0)
                          for t in solution.drone_tasks) + 1) if solution.drone_tasks else 1
                drone_id = cand.drone_id if cand.drone_id is not None else 0

                # Handle depot launch (launch_route_index=None means depot launch)
                if cand.launch_route_index is None:
                    launch_truck = None  # Depot launch
                elif cand.launch_route_index >= 0:
                    launch_truck = cand.launch_route_index
                elif cand.route_index >= 0:
                    launch_truck = cand.route_index
                else:
                    launch_truck = 0

                # Handle depot retrieve (retrieve_route_index=None means depot retrieve)
                if cand.retrieve_route_index is None:
                    land_truck = None  # Depot retrieve
                elif cand.retrieve_route_index >= 0:
                    land_truck = cand.retrieve_route_index
                else:
                    land_truck = launch_truck

                new_task = DroneTask(
                    task_id=new_id,
                    drone_id=drone_id,
                    launch_truck=launch_truck,
                    launch_node=cand.drone_launch or 0,
                    customers=[cand.customer_id],
                    land_truck=land_truck,
                    retrieve_node=cand.drone_retrieve or (
                        cand.drone_launch or 0),
                    payloads=list(cand.payloads or []),
                )
                solution.drone_tasks.append(new_task)

    # ------------------------------------------------------------------
    def _filter_valid_candidates(
        self,
        solution: Solution,
        candidates: List[Candidate],
    ) -> List[Candidate]:
        """Filter candidates based on feasibility constraints.

        Feasibility checks (per paper Section 3.3):
        1. Loading aspect: Check vehicle capacity constraints
        2. Energy aspect: Check robust energy constraints for drones
        3. Route validity: Verify launch/retrieve points exist in routes
        """
        if not candidates:
            return candidates
        filtered: List[Candidate] = []
        for cand in candidates:
            # === Loading Feasibility Check ===
            demand = self._demands.get(cand.customer_id, 0.0)

            if cand.kind == "truck":
                # Check truck capacity
                route = solution.truck_routes[cand.route_index]
                if route.current_load + demand > self._truck_capacity:
                    continue
                filtered.append(cand)
                continue

            # === Drone Candidate Validation ===
            launch = cand.drone_launch
            retrieve = cand.drone_retrieve
            if launch is None or retrieve is None:
                continue
            if launch == retrieve:
                continue

            # Check drone capacity
            if demand > self._drone_capacity:
                continue

            # Check robust energy constraint (already computed in candidate generation)
            # The energy is stored in delta_energy for new tasks
            if cand.drone_task_index is None and cand.delta_energy > self._battery:
                continue

            # === Drone Physical Location Consistency Check ===
            # For new drone tasks, verify the selected drone can launch from the specified node
            if cand.drone_task_index is None and cand.drone_id is not None:
                launch_truck_id = cand.launch_route_index
                launch_node = cand.drone_launch
                if launch_truck_id is not None:
                    compatible = self._get_compatible_drones(
                        solution, launch_truck_id, launch_node)
                    if cand.drone_id not in compatible:
                        continue

            if cand.drone_task_index is not None and cand.drone_task_index >= 0:
                if cand.drone_task_index >= len(solution.drone_tasks):
                    continue
                task = solution.drone_tasks[cand.drone_task_index]
                launch_route_index = task.launch_truck
                retrieve_route_index = task.land_truck
            else:
                launch_route_index = cand.launch_route_index
                retrieve_route_index = cand.retrieve_route_index

            # Handle depot launch/retrieve cases
            is_depot_launch = launch_route_index is None
            is_depot_retrieve = retrieve_route_index is None

            launch_route, _ = self._resolve_route_ref(
                solution, launch_route_index)
            retrieve_route, _ = self._resolve_route_ref(
                solution, retrieve_route_index)

            # Validate launch point
            if is_depot_launch:
                # For depot launch, verify launch node is a depot
                if not self._launch_from_depot(launch):
                    continue
            else:
                # For truck launch, verify launch route exists and contains launch node
                if launch_route is None:
                    continue
                if launch not in launch_route.nodes:
                    continue

            # Validate retrieve point
            if is_depot_retrieve:
                # For depot retrieve, verify retrieve node is a depot
                depot_end = self._instance.customer_manager.depot_end or self._depot_start
                if retrieve not in {self._depot_start, depot_end}:
                    continue
            else:
                # For truck retrieve, verify retrieve route exists and contains retrieve node
                if retrieve_route is None:
                    continue
                if retrieve not in retrieve_route.nodes:
                    continue

            filtered.append(cand)
        return filtered

    # ------------------------------------------------------------------
    def _segment_distance(self, a: int, b: int) -> float:
        i = self._node_index.get(a)
        j = self._node_index.get(b)
        if i is None or j is None:
            return float("inf")
        return self._truck_dist[i][j]

    def _transition_time(self, a: int, b: int) -> float:
        i = self._node_index.get(a)
        j = self._node_index.get(b)
        if i is None or j is None:
            return float("inf")
        return self._truck_time[i][j]

    def _drone_segment_distance(self, a: int, b: int) -> float:
        i = self._node_index.get(a)
        j = self._node_index.get(b)
        if i is None or j is None:
            return float("inf")
        return self._drone_dist[i][j]

    def _drone_route_distance(self, nodes: Sequence[int]) -> float:
        distance = 0.0
        for origin, dest in zip(nodes, nodes[1:]):
            segment = self._drone_segment_distance(origin, dest)
            if segment == float("inf"):
                return float("inf")
            distance += segment
        return distance

    def _worst_case_energy(self, launch: int, retrieve: int, customers: Sequence[int]) -> float:
        if not customers:
            return 0.0
        payloads = _build_payloads(customers, self._demands)
        nodes = [launch, *customers, retrieve]
        nominal = 0.0
        deviations: List[float] = []
        for payload, origin, dest in zip(payloads, nodes, nodes[1:]):
            energy = _segment_energy(
                self._energy_model, self._drone_time, self._node_index, origin, dest, payload)
            if energy == float("inf"):
                return float("inf")
            nominal += energy
            deviations.append(energy * self._deviation_rate)
        if self._robust_energy_mode == "embedded":
            budget = self._uncertainty_budget
        elif self._robust_energy_mode == "verification":
            # Partial robustness: use ~67% of full budget (min 1) to firmly
            # steer repair towards robustly feasible regions, reducing
            # downstream rejections by the subroute verifier.
            budget = min(max(1.0, self._uncertainty_budget * 0.67), self._uncertainty_budget)
        else:
            budget = 0.0
        worst = nominal + _budgeted_sum(deviations, budget)
        return worst

    def _retrieve_time(
        self,
        launch: int,
        customers: Sequence[int],
        retrieve: int,
        launch_time: float,
    ) -> float:
        current = launch_time
        nodes = [launch, *customers, retrieve]
        payloads = _build_payloads(customers, self._demands)
        for payload, a, b in zip(payloads, nodes, nodes[1:]):
            i = self._node_index.get(a)
            j = self._node_index.get(b)
            if i is None or j is None:
                return float("inf")
            travel_time = self._drone_time[i][j]
            if travel_time == float("inf"):
                return float("inf")
            current += travel_time
        return current

    def _available_drone_id(self, solution: Solution) -> Optional[int]:
        if self._drone_count <= 0:
            return 0
        used = {task.drone_id for task in solution.drone_tasks}
        for candidate_id in range(self._drone_count):
            if candidate_id not in used:
                return candidate_id
        return None

    def _resolve_route_ref(
        self, solution: Solution, ref: Optional[int]
    ) -> tuple[Optional[TruckRoute], Optional[int]]:
        if ref is None:
            return None, None
        routes = solution.truck_routes
        if 0 <= ref < len(routes):
            return routes[ref], ref
        for idx, route in enumerate(routes):
            if route.id == ref:
                return route, idx
        return None, None

    def _launch_from_depot(self, node: Optional[int]) -> bool:
        if node is None:
            return False
        depot_start = self._instance.customer_manager.depot_start
        depot_end = self._instance.customer_manager.depot_end or depot_start
        return node in {depot_start, depot_end}


def _budgeted_sum(values: Sequence[float], budget: float) -> float:
    if not values or budget <= 0:
        return 0.0
    sorted_vals = sorted(values, reverse=True)
    integer = int(min(budget, len(sorted_vals)))
    fractional = max(0.0, budget - integer)
    total = sum(sorted_vals[:integer])
    if fractional > 0 and integer < len(sorted_vals):
        total += fractional * sorted_vals[integer]
    return total


class RepairCheapest(RepairOperator):
    pass


class RepairRegret(RepairOperator):
    def __init__(self, *args, k: int = 2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._k = max(2, k)

    def _select_customer(self, solution: Solution, pool: List[int]) -> int:
        best_customer = pool[0]
        best_regret = float("-inf")
        cache: Dict[int, List[Candidate]] = {}
        for customer in pool:
            candidates = self._generate_candidates(solution, customer)
            if not candidates:
                continue
            self._normalise(candidates)
            cache[customer] = candidates
            sorted_scores = sorted(c.score for c in candidates)
            base = sorted_scores[0]
            others = sorted_scores[1:self._k]
            regret = sum(others) - base * (self._k - 1)
            if regret > best_regret:
                best_regret = regret
                best_customer = customer
        self._cache = cache
        return best_customer

    def _choose_candidate(self, solution: Solution, customer: int) -> Optional[Candidate]:
        candidates = getattr(self, "_cache", {}).get(customer)
        if candidates is None:
            candidates = self._generate_candidates(solution, customer)
        if not candidates:
            return None

        # ：
        if customer in self._forced_drone_customers:
            drone_candidates = [c for c in candidates if c.kind == "drone"]
            if drone_candidates:
                self._normalise(drone_candidates)
                return min(drone_candidates, key=lambda c: c.score)
            return None

        self._normalise(candidates)
        return min(candidates, key=lambda c: c.score)


class RepairBiasedRandomized(RepairOperator):
    def __init__(self, *args, beta: float = 3.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._beta = max(0.0, beta)

    def _choose_candidate(self, solution: Solution, customer: int) -> Optional[Candidate]:
        candidates = self._generate_candidates(solution, customer)
        if not candidates:
            return None

        # ：
        if customer in self._forced_drone_customers:
            drone_candidates = [c for c in candidates if c.kind == "drone"]
            if drone_candidates:
                self._normalise(drone_candidates)
                return min(drone_candidates, key=lambda c: c.score)
            return None

        self._normalise(candidates)
        candidates.sort(key=lambda c: c.score)
        if self._beta == 0.0:
            return self._rng.choice(candidates)
        weights = [1.0 / ((idx + 1) ** self._beta)
                   for idx in range(len(candidates))]
        total = sum(weights)
        r = self._rng.random() * total
        cumulative = 0.0
        for cand, weight in zip(candidates, weights):
            cumulative += weight
            if r <= cumulative:
                return cand
        return candidates[-1]


class RepairEqualPriority(RepairOperator):
    """Equal-priority insertion ().

    Strategy: Treats truck and drone insertion options as equally important.

    Operation:
    1. Calculate minimum cost increment for inserting customer i as truck node (Δf_i^T)
       and as drone node (Δf_i^D)
    2. Select the position corresponding to min(Δf_i^T, Δf_i^D)

    Per Algorithm B1: Sort all candidates by cost increment, select first feasible.
    """

    def _choose_candidate(self, solution: Solution, customer: int) -> Optional[Candidate]:
        candidates = self._generate_candidates(solution, customer)
        if not candidates:
            return None

        # ：
        if customer in self._forced_drone_customers:
            drone_candidates = [c for c in candidates if c.kind == "drone"]
            if drone_candidates:
                # Sort by cost and return first (feasibility already checked in generation)
                return min(
                    drone_candidates,
                    key=lambda c: self._compute_cost(c),
                )
            return None

        # Algorithm B1: Sort ALL candidates by cost increment, pick minimum
        # Cost = distance * unit_cost (truck vs drone have different unit costs)
        return min(candidates, key=lambda c: self._compute_cost(c))

    def _compute_cost(self, cand: Candidate) -> float:
        """Compute insertion cost considering vehicle unit costs."""
        """Compute insertion cost considering vehicle unit costs."""
        if cand.kind == "truck":
            return cand.delta_distance * self._truck_unit_cost
        else:
            return cand.delta_distance * self._drone_unit_cost


class RepairDronePriorityRegret(RepairOperator):
    """Drone-first truck-second insertion (，).

    Strategy: Aims to quickly reduce cost by prioritizing drone's low-cost advantage.

    Operation:
    1. First try to insert customer into drone sub-route at lowest cost position
    2. Only when no feasible drone position exists (Δf_i^D = ∞), select lowest cost
       feasible truck position

    Uses regret-based customer selection to prioritize customers that benefit most
    from drone insertion.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cache: Dict[int, List[Candidate]] = {}
        self.initial_weight = 1.25

    def _select_customer(self, solution: Solution, pool: List[int]) -> int:
        """Select customer with highest regret (drone vs truck cost difference)."""
        """Select customer with highest regret (drone vs truck cost difference)."""
        best_customer = pool[0]
        best_regret = float("-inf")
        cache: Dict[int, List[Candidate]] = {}
        for customer in pool:
            candidates = self._generate_candidates(solution, customer)
            if not candidates:
                continue
            cache[customer] = candidates

            # Compute best truck and drone costs
            truck_best = min(
                (c.delta_distance *
                 self._truck_unit_cost for c in candidates if c.kind == "truck"),
                default=float("inf"),
            )
            drone_best = min(
                (c.delta_distance *
                 self._drone_unit_cost for c in candidates if c.kind == "drone"),
                default=float("inf"),
            )

            # Regret = how much we save by using drone vs truck
            if math.isinf(drone_best):
                regret = float("-inf")
            else:
                regret = truck_best - drone_best
            if regret > best_regret:
                best_regret = regret
                best_customer = customer
        self._cache = cache
        return best_customer

    def _choose_candidate(self, solution: Solution, customer: int) -> Optional[Candidate]:
        """Drone-first: Only use truck if no feasible drone position exists."""
        """Drone-first: Only use truck if no feasible drone position exists."""
        candidates = self._cache.pop(customer, None)
        if candidates is None:
            candidates = self._generate_candidates(solution, customer)
        if not candidates:
            return None

        # ：
        if customer in self._forced_drone_customers:
            drone_candidates = [c for c in candidates if c.kind == "drone"]
            if drone_candidates:
                return min(
                    drone_candidates,
                    key=lambda c: c.delta_distance * self._drone_unit_cost,
                )
            return None

        # Drone-first strategy: Try drone first
        drone_candidates = [c for c in candidates if c.kind == "drone"]
        if drone_candidates:
            # Return lowest cost drone position
            return min(
                drone_candidates,
                key=lambda c: c.delta_distance * self._drone_unit_cost,
            )

        # Only when no feasible drone position exists, use truck
        truck_candidates = [c for c in candidates if c.kind == "truck"]
        if truck_candidates:
            return min(
                truck_candidates,
                key=lambda c: c.delta_distance * self._truck_unit_cost
            )

        return None


class RepairTruckFirst(RepairOperator):
    """Truck-first drone-second insertion (，).

    Strategy: Ensure connectivity first, then optimize structure.

    Operation:
    1. First insert ALL removed nodes into truck routes at lowest cost positions
    2. After all nodes are on truck routes, try to "convert" some truck nodes
       to drone tasks while satisfying resource and synchronization constraints

    This two-phase approach ensures feasibility first, then improves solution quality.
    """
    initial_weight = 0.9

    def apply(self, solution: Solution, unassigned: Iterable[int]) -> Solution:
        """Two-phase repair: truck insertion then drone migration."""
        """Two-phase repair: truck insertion then drone migration."""
        mutated = solution.clone()
        pool = list(unassigned)
        staged_truck: List[int] = []

        # Phase 1: Insert all nodes into truck routes first
        while pool:
            customer = self._select_customer(mutated, pool)
            pool.remove(customer)
            candidates = self._generate_candidates(mutated, customer)
            if not candidates:
                continue

            # ：
            if customer in self._forced_drone_customers:
                drone_candidates = [c for c in candidates if c.kind == "drone"]
                if drone_candidates:
                    best = min(drone_candidates,
                               key=lambda c: c.delta_distance * self._drone_unit_cost)
                    self._apply_candidate(mutated, best)
                continue

            # Truck-first: Insert into truck route
            truck_candidates = [c for c in candidates if c.kind == "truck"]
            if truck_candidates:
                best = min(truck_candidates,
                           key=lambda c: c.delta_distance * self._truck_unit_cost)
                self._apply_candidate(mutated, best)
                staged_truck.append(customer)
            else:
                # No truck position available, try drone as fallback
                drone_candidates = [c for c in candidates if c.kind == "drone"]
                if drone_candidates:
                    best = min(drone_candidates,
                               key=lambda c: c.delta_distance * self._drone_unit_cost)
                    self._apply_candidate(mutated, best)

        # Phase 2: Try to migrate truck nodes to drones for optimization
        self._migrate_to_drones(mutated, staged_truck)
        return mutated

    def _migrate_to_drones(self, solution: Solution, customers: List[int]) -> None:
        """Try to convert truck customers to drone tasks if beneficial.

        For each customer currently on truck route:
        1. Calculate removal gain (truck distance saved)
        2. Find best drone insertion position
        3. If drone cost < removal gain, execute migration
        """
        for customer in customers:
            location = self._find_truck_location(solution, customer)
            if location is None:
                continue
            route_index, position = location
            route = solution.truck_routes[route_index]

            # Skip depot nodes
            if position <= 0 or position >= len(route.nodes) - 1:
                continue

            prev_node = route.nodes[position - 1]
            next_node = route.nodes[position + 1]

            # Calculate truck distance saved by removal
            removal_gain = (
                self._segment_distance(prev_node, customer)
                + self._segment_distance(customer, next_node)
                - self._segment_distance(prev_node, next_node)
            ) * self._truck_unit_cost

            # Temporarily remove from truck to generate drone candidates
            demand = self._demands.get(customer, 0.0)
            route.nodes.pop(position)
            route.current_load -= demand

            # Find best drone position
            candidate = self._best_drone_candidate(solution, customer)

            if candidate is not None:
                drone_cost = candidate.delta_distance * self._drone_unit_cost
                # Migrate if drone cost is lower than truck removal gain
                if drone_cost < removal_gain:
                    self._apply_candidate(solution, candidate)
                    continue

            # Migration not beneficial, restore to truck
            fallback = self._best_truck_candidate(solution, customer)
            if fallback is not None:
                self._apply_candidate(solution, fallback)
            else:
                # Restore original position
                route.nodes.insert(position, customer)
                route.current_load += demand

    def _find_truck_location(self, solution: Solution, customer: int) -> Optional[tuple[int, int]]:
        for route_index, route in enumerate(solution.truck_routes):
            for position, node in enumerate(route.nodes):
                if node == customer:
                    return route_index, position
        return None

    def _best_truck_candidate(self, solution: Solution, customer: int) -> Optional[Candidate]:
        candidates = self._generate_candidates(solution, customer)
        truck_candidates = [c for c in candidates if c.kind == "truck"]
        if not truck_candidates:
            return None
        return min(truck_candidates,
                   key=lambda c: c.delta_distance * self._truck_unit_cost)

    def _best_drone_candidate(self, solution: Solution, customer: int) -> Optional[Candidate]:
        candidates = self._generate_candidates(solution, customer)
        drone_candidates = [c for c in candidates if c.kind ==
                            "drone" and math.isfinite(c.delta_distance)]
        if not drone_candidates:
            return None
        return min(
            drone_candidates,
            key=lambda c: c.delta_distance * self._drone_unit_cost,
        )
