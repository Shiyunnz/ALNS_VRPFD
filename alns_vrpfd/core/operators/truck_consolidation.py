"""Truck consolidation operator: eliminate underutilized trucks."""
import math, random
from typing import Optional, List
from alns_vrpfd.model import Solution, TruckRoute, DroneTask
from alns_vrpfd.core.operators.base import _build_payloads


class TruckConsolidationRepair:
    """Move all customers from an underutilized truck to other trucks.
    
    If a truck has ≤ threshold customers, attempt to relocate each customer
    to another truck's route (cheapest insertion by distance heuristic).
    If all succeed, remove the truck entirely and update affected drone tasks.
    """

    def __init__(self, instance, evaluator, *, threshold: int = 2,
                 rng: Optional[random.Random] = None):
        self._instance = instance
        self._evaluator = evaluator
        self._threshold = threshold
        self._rng = rng or random.Random()
        self.attempts = 0
        self.successes = 0

        self._dm = {c.customer_id: c.demand
                    for c in instance.customer_manager.customers()}
        self._nodes = list(instance.all_node_ids())
        self._ix = {n: i for i, n in enumerate(self._nodes)}
        self._td = instance.distance_matrix('truck')

    def apply(self, solution: Solution) -> Solution:
        self.attempts += 1

        truck_routes = list(solution.truck_routes)
        if len(truck_routes) <= 1:
            return solution

        depot_start = self._instance.customer_manager.depot_start
        depot_end = self._instance.customer_manager.depot_end

        # Sort candidates by customer count (ascending)
        candidates = sorted(truck_routes,
                            key=lambda r: len([n for n in r.nodes if n in self._dm]))

        for candidate in candidates:
            custs = [n for n in candidate.nodes if n in self._dm]
            if len(custs) > self._threshold or len(custs) == 0:
                continue

            others = [r for r in truck_routes if r.id != candidate.id]
            if not others:
                continue

            # Greedy: for each customer, pick the cheapest insertion position
            new_route_nodes: dict[int, List[int]] = {r.id: list(r.nodes) for r in others}
            ok = True

            for c in custs:
                best_pos = None
                best_delta = float('inf')
                for rid, nodes in new_route_nodes.items():
                    for pos in range(1, len(nodes)):
                        prev_node = nodes[pos - 1]
                        next_node = nodes[pos]
                        old_dist = self._td[self._ix[prev_node]][self._ix[next_node]]
                        new_dist = (self._td[self._ix[prev_node]][self._ix[c]] +
                                    self._td[self._ix[c]][self._ix[next_node]])
                        delta = new_dist - old_dist
                        if delta < best_delta:
                            best_delta = delta
                            best_pos = (rid, pos)

                if best_pos is None:
                    ok = False
                    break
                rid, pos = best_pos
                new_route_nodes[rid].insert(pos, c)

            if not ok:
                continue

            # Build consolidated solution
            sol = Solution()
            for r in others:
                sol.add_truck_route(TruckRoute(
                    route_id=r.id,
                    nodes=new_route_nodes[r.id],
                    capacity=r.capacity))

            # Update drone task truck references
            new_node_to_truck: dict[int, int] = {}
            for rid, nodes in new_route_nodes.items():
                for n in nodes:
                    new_node_to_truck[n] = rid

            for dt in solution.drone_tasks:
                lt = dt.launch_truck
                ln = dt.launch_node
                rt = dt.land_truck
                rn = dt.retrieve_node

                if lt is not None and lt == candidate.id:
                    lt = new_node_to_truck.get(ln)
                if rt is not None and rt == candidate.id:
                    rt = new_node_to_truck.get(rn)
                if ln == depot_start:
                    lt = None
                if rn == depot_end:
                    rt = None

                sol.add_drone_task(DroneTask(
                    task_id=dt.task_id, drone_id=dt.drone_id,
                    launch_truck=lt, launch_node=dt.launch_node,
                    customers=list(dt.customers()),
                    land_truck=rt, retrieve_node=dt.retrieve_node,
                    payloads=list(dt.payloads),
                ))

            self.successes += 1

            # Evaluate consolidated solution
            ev = self._evaluator.evaluate_solution(sol)
            if not ev.feasible or not math.isfinite(ev.total_cost):
                return sol

            # Try flipping depot-launch sorties to truck-launch
            new_tasks = list(sol.drone_tasks)
            flipped = False
            for i, dt in enumerate(new_tasks):
                if dt.launch_truck is None and dt.land_truck is not None:
                    rev_custs = list(reversed(dt.customers()))
                    new_tasks[i] = DroneTask(
                        task_id=dt.task_id, drone_id=dt.drone_id,
                        launch_truck=dt.land_truck, launch_node=dt.retrieve_node,
                        customers=rev_custs, land_truck=None, retrieve_node=depot_end,
                        payloads=_build_payloads(rev_custs, self._dm))
                    flipped = True

            if flipped:
                flip_sol = Solution()
                for tr in sol.truck_routes:
                    flip_sol.add_truck_route(TruckRoute(
                        route_id=tr.id, nodes=list(tr.nodes), capacity=tr.capacity))
                for t in new_tasks:
                    flip_sol.add_drone_task(t)
                ev2 = self._evaluator.evaluate_solution(flip_sol)
                if ev2.feasible and math.isfinite(ev2.total_cost) and ev2.total_cost < ev.total_cost:
                    sol = flip_sol
                    ev = ev2
                    self.successes += 1

            return sol
