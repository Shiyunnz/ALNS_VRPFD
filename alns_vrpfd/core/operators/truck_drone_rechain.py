"""Truck Backbone Repartition + Sortie Rechain operator (Step 12).

Selects 2-4 customers, removes them from BOTH truck routes and drone sorties,
then exhaustively:
1. Repartitions which customers go on which truck route
2. Re-enumerates all possible drone sortie assignments for removed customers

This closes the gap where ALNS cannot restructure the truck backbone to enable
better drone sortie configurations (e.g., moving customer 7 from truck route
to a drone sortie requires changing the truck backbone from [0,1,3,5,7,1,11]
to [0,1,11] + [0,3,5,11]).

Key insight from R_30_10_2: The 50.96 MILP solution requires truck backbone
[0,1,11] + [0,3,5,11] instead of ALNS's [0,1,3,5,7,1,11] + [0,1,11].
Customer 7 must move from truck to drone, which requires backbone repartition.
"""

from __future__ import annotations

import itertools
import math
import random
from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple

from alns_vrpfd.evaluation import Evaluator
from alns_vrpfd.evaluation.energy import DroneEnergyModel
from alns_vrpfd.evaluation.subroute_robust_verifier import _budgeted_sum
from alns_vrpfd.instance.manager import InstanceManager
from alns_vrpfd.model import DroneTask, Solution, TruckRoute
from alns_vrpfd.core.operators.base import _build_payloads


class TruckDroneRechainRepair:
    """Joint truck backbone repartition + drone sortie enumeration.

    Selects a neighborhood of 2-4 customers, removes them entirely from the
    solution (from both truck routes and drone sorties), then exhaustively
    tries all possible:
    - Truck route reassignments (which truck serves which of the removed customers)
    - Drone sortie configurations (set partitions × orderings × anchors × drones)
    for the removed customers.

    Parameters
    ----------
    instance : InstanceManager
    evaluator : Evaluator
    max_customers : int
        Max neighborhood size. Default 3.
    max_anchor_dist_factor : float
        Max anchor distance factor. Default 2.0.
    energy_tolerance : float
        Battery tolerance multiplier for pre-check. Default 1.0.
    rng : random.Random, optional
    """

    name = "TruckDroneRechain"

    def __init__(
        self,
        instance: InstanceManager,
        evaluator: Evaluator,
        max_customers: int = 3,
        max_anchor_dist_factor: float = 2.0,
        energy_tolerance: float = 1.0,
        max_candidates: int = 5000,
        rng: Optional[random.Random] = None,
    ) -> None:
        self._instance = instance
        self._evaluator = evaluator
        self._max_customers = max(2, min(max_customers, 4))
        self._max_anchor_dist_factor = max_anchor_dist_factor
        self._energy_tolerance = max(0.8, min(energy_tolerance, 2.0))
        self._max_candidates = max_candidates
        self._rng = rng or random.Random(random.getrandbits(32))
        self._demands = {
            c.customer_id: c.demand
            for c in instance.customer_manager._customers.values()
        }
        self._energy_model = DroneEnergyModel()
        self._rob_config = instance.robust_config
        self._node_index = {
            n: i for i, n in enumerate(instance.all_node_ids())
        }
        self._drone_time = instance.time_matrix("drone")
        self._drone_cap = instance.vehicle_specs.get(
            "drone", type("", (), {"capacity": 30})()
        ).capacity
        self._truck_cap = instance.vehicle_specs.get(
            "truck", type("", (), {"capacity": 200})()
        ).capacity
        self._battery = self._rob_config.drone_battery_capacity
        self._dist_matrix = instance.distance_matrix("drone")
        self._truck_dist_matrix = instance.distance_matrix("truck")
        self._depot_start = instance.customer_manager.depot_start
        self._depot_end = instance.customer_manager.depot_end
        self._num_trucks = instance.vehicle_specs.get(
            "truck", type("", (), {"number": 2})()
        ).number
        self._num_drones = instance.vehicle_specs.get(
            "drone", type("", (), {"number": 2})()
        ).number

        self.attempts = 0
        self.created = 0
        self.accepted = 0

    def _set_partitions(self, xs: List[int]):
        """Generate all set partitions preserving order."""
        if not xs:
            yield []
            return
        first = xs[0]
        for rest in self._set_partitions(xs[1:]):
            yield [[first]] + rest
            for i, blk in enumerate(rest):
                new = deepcopy(rest)
                new[i] = [first] + blk
                yield new

    def _quick_energy(self, launch: int, customers: List[int],
                      retrieve: int) -> float:
        """Worst-case robust energy for a drone sortie (no timing)."""
        payloads = _build_payloads(customers, self._demands)
        nodes = [launch] + customers + [retrieve]
        ni, dt = self._node_index, self._drone_time
        nom, devs = 0.0, []
        for p, a, b in zip(payloads, nodes, nodes[1:]):
            try:
                t = dt[ni[a]][ni[b]]
            except (KeyError, IndexError):
                return float("inf")
            e = self._energy_model.energy_kwh(p, t)
            nom += e
            devs.append(e * self._rob_config.energy_deviation_rate)
        return nom + _budgeted_sum(devs, self._rob_config.energy_uncertainty_budget)

    def _truck_route_distance(self, nodes: List[int]) -> float:
        """Compute truck route distance."""
        ni = self._node_index
        dm = self._truck_dist_matrix
        dist = 0.0
        for a, b in zip(nodes, nodes[1:]):
            try:
                dist += dm[ni[a]][ni[b]]
            except (KeyError, IndexError):
                return float("inf")
        return dist

    def _filter_anchors_by_distance(self, customers: List[int],
                                     anchors: List[int]) -> Tuple[List[int], List[int]]:
        """Return (launch, retrieve) anchors within max_anchor_dist × diameter."""
        launch_candidates = list(anchors)
        retrieve_candidates = [
            a for a in anchors
            if a != self._depot_start or self._depot_start == self._depot_end
        ]
        if not customers or self._max_anchor_dist_factor == float("inf"):
            return launch_candidates, retrieve_candidates

        ni = self._node_index
        dm = self._dist_matrix
        max_pair = 0.0
        for i in customers:
            for j in customers:
                try:
                    d = dm[ni[i]][ni[j]]
                    if d > max_pair:
                        max_pair = d
                except (KeyError, IndexError):
                    pass
        max_allowed = max_pair * self._max_anchor_dist_factor
        if max_allowed <= 0:
            return launch_candidates, retrieve_candidates

        launch_ok = []
        retrieve_ok = []
        for a in launch_candidates:
            try:
                for c in customers:
                    d_launch = dm[ni[min(c, a)]][ni[max(c, a)]]
                    if d_launch <= max_allowed:
                        if a not in launch_ok:
                            launch_ok.append(a)
                        break
            except (KeyError, IndexError):
                pass
        for a in retrieve_candidates:
            try:
                for c in customers:
                    d_retrieve = dm[ni[min(c, a)]][ni[max(c, a)]]
                    if d_retrieve <= max_allowed:
                        if a not in retrieve_ok:
                            retrieve_ok.append(a)
                        break
            except (KeyError, IndexError):
                pass

        return launch_ok or launch_candidates, retrieve_ok or retrieve_candidates

    def _score_truck_customers(
        self, solution: Solution, truck_served: List[int],
    ) -> List[Tuple[float, int]]:
        """Score truck-served customers by drone-savings potential.

        A customer c has high savings if removing it from the truck route
        shortens the truck route significantly AND it can be served by drone
        from nearby anchors.
        """
        scored = []
        ni = self._node_index
        dm = self._dist_matrix
        t_dm = self._truck_dist_matrix
        drone_t = self._drone_time

        for c in truck_served:
            truck_savings = 0.0
            for r in solution.truck_routes:
                if c in r.nodes:
                    idx = r.nodes.index(c)
                    if idx > 0 and idx < len(r.nodes) - 1:
                        prev_n, next_n = r.nodes[idx - 1], r.nodes[idx + 1]
                        try:
                            old_dist = (t_dm[ni[prev_n]][ni[c]]
                                        + t_dm[ni[c]][ni[next_n]])
                            new_dist = t_dm[ni[prev_n]][ni[next_n]]
                            truck_savings = old_dist - new_dist
                        except (KeyError, IndexError):
                            pass
                    break

            anchors = self._get_anchors(solution.truck_routes)
            launch_anchors, retrieve_anchors = self._filter_anchors_by_distance(
                [c], anchors)
            best_sortie_energy = float("inf")
            for ln in launch_anchors:
                for rn in retrieve_anchors:
                    if ln == rn:
                        continue
                    try:
                        e = self._quick_energy(ln, [c], rn)
                        if e < best_sortie_energy:
                            best_sortie_energy = e
                    except (KeyError, IndexError):
                        pass

            if best_sortie_energy <= self._battery:
                score = truck_savings
            else:
                score = -1.0

            scored.append((score, c))

        scored.sort(key=lambda x: -x[0])
        return scored

    def _select_neighborhood(
        self, solution: Solution, attempt: int = 0,
    ) -> List[int]:
        """Select 2-4 customers for joint truck-drone restructuring.

        Uses scoring to prioritize truck-served customers with high savings.
        On different attempts, varies which customers are picked.
        """
        drone_served = set()
        multi_customers = []
        single_customers = []

        for t in solution.drone_tasks:
            cs = t.customers()
            drone_served.update(cs)
            if len(cs) >= 2:
                multi_customers.extend(cs)
            elif len(cs) == 1:
                single_customers.append(cs[0])

        customer_ids = set(self._instance.customer_manager.customer_ids())
        truck_served = []
        for r in solution.truck_routes:
            for n in r.nodes:
                if (n not in drone_served
                        and n in customer_ids):
                    truck_served.append(n)

        scored = self._score_truck_customers(solution, truck_served)
        viable = [(s, c) for s, c in scored if s >= 0]

        if not viable:
            return []

        selected = []
        n_truck = min(self._rng.randint(1, 2), len(viable))
        offset = (attempt * n_truck) % len(viable)
        for i in range(n_truck):
            idx = (offset + i) % len(viable)
            selected.append(viable[idx][1])

        remaining = self._max_customers - len(selected)
        if remaining > 0:
            candidates = multi_customers + single_customers
            self._rng.shuffle(candidates)
            selected.extend(candidates[:remaining])

        return list(dict.fromkeys(selected))[:self._max_customers]

    def _truck_of(self, node: int, truck_routes: List[TruckRoute]) -> Optional[int]:
        """Find which truck route contains this node."""
        for route in truck_routes:
            if node in route.nodes:
                return route.id
        return None

    def _get_anchors(self, truck_routes: List[TruckRoute]) -> List[int]:
        """Return all unique nodes from all truck routes."""
        seen: Set[int] = set()
        result: List[int] = []
        for r in truck_routes:
            for n in r.nodes:
                if n not in seen:
                    seen.add(n)
                    result.append(n)
        return result

    def _rebuild_truck_routes(
        self,
        base_routes: List[TruckRoute],
        removed: List[int],
        truck_assignments: Dict[int, int],
    ) -> List[TruckRoute]:
        """Rebuild truck routes with removed customers reassigned to trucks.

        Parameters
        ----------
        base_routes : List[TruckRoute]
            Routes with removed customers already stripped out.
        removed : List[int]
            Customer IDs that were removed and need truck reinsertion.
        truck_assignments : Dict[int, int]
            Maps customer ID -> truck ID for truck-served customers.
        """
        new_routes = []
        for route in base_routes:
            nodes = list(route.nodes)
            # Collect customers to insert on this truck
            to_insert = [(c, truck_assignments[c])
                         for c in removed
                         if truck_assignments.get(c) == route.id]
            if to_insert:
                # Greedy nearest-neighbor insertion
                for c, _ in to_insert:
                    best_pos = len(nodes) - 1  # Before depot_end
                    best_cost = float("inf")
                    for pos in range(1, len(nodes)):
                        # Insert between pos-1 and pos
                        prev, nxt = nodes[pos - 1], nodes[pos]
                        ni = self._node_index
                        dm = self._truck_dist_matrix
                        try:
                            old_dist = dm[ni[prev]][ni[nxt]]
                            new_dist = dm[ni[prev]][ni[c]] + dm[ni[c]][ni[nxt]]
                            delta = new_dist - old_dist
                            if delta < best_cost:
                                best_cost = delta
                                best_pos = pos
                        except (KeyError, IndexError):
                            pass
                    nodes.insert(best_pos, c)
            new_routes.append(TruckRoute(
                route_id=route.id,
                nodes=nodes,
                capacity=route.capacity,
            ))
        return new_routes

    def _enumerate_neighborhood(
        self, solution: Solution, neighborhood: List[int], target_cost: float,
    ) -> Tuple[Solution, float, int]:
        """Enumerate all reconstructions for a given neighborhood.

        Returns (best_solution, best_cost, candidates_evaluated).
        """
        customer_ids = set(self._instance.customer_manager.customer_ids())

        # ── Destroy: remove ALL neighborhood customers ──
        clean = solution.clone()
        for c in neighborhood:
            for t in list(clean.drone_tasks):
                try:
                    t.remove_customer(c)
                except (ValueError, IndexError):
                    pass
            for r in clean.truck_routes:
                try:
                    r.remove_customer(c)
                except ValueError:
                    pass

        clean.drone_tasks = [t for t in clean.drone_tasks if len(t.customers()) > 0]

        # ── Get anchors from cleaned truck routes ──
        all_anchors = self._get_anchors(clean.truck_routes)

        # ── Pre-compute energy cache ──
        energy_ok: Dict[Tuple, bool] = {}
        blocks_seen: Set[Tuple[int, ...]] = set()
        block_permutations: Dict[Tuple[int, ...], List[List[int]]] = {}

        for partition in self._set_partitions(neighborhood):
            for blk in partition:
                bk = tuple(sorted(blk))
                if bk not in blocks_seen:
                    blocks_seen.add(bk)
                    perms = [list(p) for p in set(itertools.permutations(blk))]
                    block_permutations[bk] = perms

                    launch_anchors, retrieve_anchors = self._filter_anchors_by_distance(
                        list(blk), all_anchors)
                    for ln in launch_anchors:
                        for rn in retrieve_anchors:
                            e = self._quick_energy(ln, list(blk), rn)
                            if e < self._battery * self._energy_tolerance:
                                energy_ok[(bk, ln, rn)] = True

        feasible_lr: Dict[Tuple[int, ...], List[Tuple[int, int]]] = {}
        for bk in blocks_seen:
            pairs = [(ln, rn) for (bk_, ln, rn) in energy_ok if bk_ == bk]
            feasible_lr[bk] = pairs

        # ── Enumerate ──
        best = solution
        best_cost = target_cost
        candidates_evaluated = 0

        for partition in self._set_partitions(neighborhood):
            if candidates_evaluated >= self._max_candidates:
                break
            for truck_mask in range(1 << len(partition)):
                if candidates_evaluated >= self._max_candidates:
                    break
                truck_blocks = []
                drone_blocks = []
                for i, blk in enumerate(partition):
                    if truck_mask & (1 << i):
                        truck_blocks.append(blk)
                    else:
                        drone_blocks.append(blk)

                truck_served_customers = []
                for blk in truck_blocks:
                    truck_served_customers.extend(blk)

                truck_ids = list(range(self._num_trucks))
                if not truck_served_customers:
                    truck_assignment_combos = [{}]
                else:
                    assignments = list(itertools.product(truck_ids, repeat=len(truck_served_customers)))
                    truck_assignment_combos = [
                        dict(zip(truck_served_customers, a)) for a in assignments
                    ]

                for truck_assign in truck_assignment_combos:
                    if candidates_evaluated >= self._max_candidates:
                        break
                    rebuilt_routes = self._rebuild_truck_routes(
                        clean.truck_routes, truck_served_customers, truck_assign)

                    valid_routes = True
                    for r in rebuilt_routes:
                        total_demand = sum(
                            self._demands.get(n, 0) for n in r.nodes
                            if n != self._depot_start and n != self._depot_end
                        )
                        if total_demand > r.capacity:
                            valid_routes = False
                            break

                    if not valid_routes:
                        continue

                    rebuilt_anchors = self._get_anchors(rebuilt_routes)

                    node_to_truck: Dict[int, int] = {}
                    for r in rebuilt_routes:
                        for pos, n in enumerate(r.nodes):
                            node_to_truck[n] = r.id

                    if not drone_blocks:
                        sol = Solution()
                        for r in rebuilt_routes:
                            sol.add_truck_route(TruckRoute(
                                route_id=r.id, nodes=list(r.nodes),
                                capacity=r.capacity))
                        for t in clean.drone_tasks:
                            sol.add_drone_task(deepcopy(t))

                        try:
                            ev = self._evaluator.evaluate_solution(sol)
                        except (KeyError, ValueError):
                            candidates_evaluated += 1
                            continue
                        candidates_evaluated += 1
                        if (ev.feasible and math.isfinite(ev.total_cost)
                                and ev.total_cost < best_cost - 1e-6):
                            best = sol
                            best_cost = ev.total_cost
                        continue

                    drone_perms = [block_permutations.get(tuple(sorted(blk)), [])
                                   for blk in drone_blocks]
                    if any(len(p) == 0 for p in drone_perms):
                        continue

                    for perm_combo in itertools.product(*drone_perms):
                        n_drone = len(perm_combo)
                        lr_opts = [
                            feasible_lr.get(tuple(sorted(blk)), [])
                            for blk in perm_combo
                        ]
                        if any(len(opts) == 0 for opts in lr_opts):
                            continue

                        rebuilt_lr = []
                        for bi in range(n_drone):
                            blk = perm_combo[bi]
                            bk = tuple(sorted(blk))
                            pairs = []
                            for ln, rn in feasible_lr.get(bk, []):
                                if ln in rebuilt_anchors or ln == self._depot_start:
                                    if rn in rebuilt_anchors or rn == self._depot_end:
                                        pairs.append((ln, rn))
                            rebuilt_lr.append(pairs if pairs else feasible_lr.get(bk, []))

                        for drone_ids in itertools.product(
                            range(self._num_drones), repeat=n_drone,
                        ):
                            if candidates_evaluated >= self._max_candidates:
                                break
                            for lr_combo in itertools.product(*rebuilt_lr):
                                sol = Solution()
                                for r in rebuilt_routes:
                                    sol.add_truck_route(TruckRoute(
                                        route_id=r.id, nodes=list(r.nodes),
                                        capacity=r.capacity))
                                for t in clean.drone_tasks:
                                    sol.add_drone_task(deepcopy(t))

                                nid = max(
                                    (t.task_id or 0) for t in sol.drone_tasks
                                ) + 1 if sol.drone_tasks else 0

                                for bi in range(n_drone):
                                    blk = perm_combo[bi]
                                    ln, rn = lr_combo[bi]
                                    di = drone_ids[bi]
                                    lt = node_to_truck.get(ln)
                                    rt = node_to_truck.get(rn)
                                    if ln == self._depot_start:
                                        lt = None
                                    if rn == self._depot_end:
                                        rt = None

                                    sol.add_drone_task(DroneTask(
                                        task_id=nid, drone_id=di,
                                        launch_truck=lt, launch_node=ln,
                                        customers=blk,
                                        land_truck=rt, retrieve_node=rn,
                                        payloads=_build_payloads(blk, self._demands),
                                    ))
                                    nid += 1

                                try:
                                    ev = self._evaluator.evaluate_solution(sol)
                                    if (ev.feasible and math.isfinite(ev.total_cost)
                                            and ev.total_cost < best_cost - 1e-6):
                                        best = sol
                                        best_cost = ev.total_cost
                                except (KeyError, ValueError):
                                    pass
                                candidates_evaluated += 1

        return best, best_cost, candidates_evaluated

    def apply(self, solution: Solution) -> Solution:
        """Apply joint truck-drone rechaining.

        Tries multiple neighborhood selections. Returns the best improved
        solution, or the original if no improvement.
        """
        self.attempts += 1
        self._current_truck_routes = solution.truck_routes

        current_eval = self._evaluator.evaluate_solution(solution)
        if not current_eval.feasible or not math.isfinite(current_eval.total_cost):
            return solution

        target_cost = current_eval.total_cost

        best_overall = solution
        best_overall_cost = target_cost
        total_candidates = 0
        max_attempts = min(3, len(set(
            n for r in solution.truck_routes
            for n in r.nodes
            if n in set(self._instance.customer_manager.customer_ids())
        )))

        for attempt in range(max_attempts):
            neighborhood = self._select_neighborhood(solution, attempt=attempt)
            if len(neighborhood) < 2:
                continue

            best, best_cost, n_eval = self._enumerate_neighborhood(
                solution, neighborhood, best_overall_cost)

            total_candidates += n_eval
            if best_cost < best_overall_cost - 1e-6:
                best_overall = best
                best_overall_cost = best_cost

            if total_candidates >= self._max_candidates:
                break

        if best_overall_cost < target_cost:
            self.accepted += 1

        self.created += total_candidates
        return best_overall
