"""Matheuristic LNS repair: exhaustive neighborhood reconstruction (Step 11).

Selects 2-4 drone-served customers, removes them, and exhaustively
enumerates all possible drone sortie assignments (set partitions ×
permutations × anchor choices × drone IDs).  This operator directly
targets the "3D assignment problem" (customer pairing × drone
assignment × launch/retrieve truck) that ALNS local search cannot solve.

The oracle on R_40_10_1 confirmed this approach closes the 0.43% gap
(97.60 → 97.18) by reconstructing customers {6, 8, 10} into the
cross-truck sortie D0: [5->[8,10]->1] + D0: [4->[6]->5].
"""

from __future__ import annotations

import itertools
import math
import random
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Set, Tuple

from alns_vrpfd.evaluation import Evaluator
from alns_vrpfd.evaluation.energy import DroneEnergyModel
from alns_vrpfd.evaluation.subroute_robust_verifier import _budgeted_sum
from alns_vrpfd.instance.manager import InstanceManager
from alns_vrpfd.model import DroneTask, Solution
from alns_vrpfd.core.operators.base import _build_payloads


class MatheuristicLNSRepair:
    """Exhaustive drone sortie reconstruction via set-partition enumeration.

    Removes a neighborhood of drone-served customers, then exhaustively
    tries all possible assignments of those customers to drone sorties
    (partitioning, ordering, anchor selection, and drone allocation).

    Parameters
    ----------
    instance : InstanceManager
    evaluator : Evaluator
    max_customers : int
        Max neighborhood size. Default 3.
    max_anchor_dist_factor : float
        Max anchor distance = factor × (customer max pairwise distance).
        Default 2.0 (anchors within 2× the neighborhood's diameter).
    energy_tolerance : float
        Multiplier on battery for energy pre-check. Default 1.0.
    rng : random.Random, optional
    """

    name = "MatheuristicLNS"

    def __init__(
        self,
        instance: InstanceManager,
        evaluator: Evaluator,
        max_customers: int = 3,
        max_anchor_dist_factor: float = 2.0,
        energy_tolerance: float = 1.0,
        rng: Optional[random.Random] = None,
    ) -> None:
        self._instance = instance
        self._evaluator = evaluator
        self._max_customers = max(2, min(max_customers, 4))
        self._max_anchor_dist_factor = max_anchor_dist_factor
        self._energy_tolerance = max(0.8, min(energy_tolerance, 2.0))
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
        self._battery = self._rob_config.drone_battery_capacity
        self._dist_matrix = instance.distance_matrix("drone")

        # Statistics
        self.attempts = 0
        self.created = 0
        self.accepted = 0

    # ── Helpers ──────────────────────────────────────────────

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

    def _truck_of(self, node: int) -> Optional[int]:
        for route in self._current_truck_routes:
            if node in route.nodes:
                return route.id
        return None

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

    def _all_anchors(self) -> List[int]:
        """Return all unique nodes from all truck routes."""
        seen: Set[int] = set()
        result: List[int] = []
        for r in self._current_truck_routes:
            for n in r.nodes:
                if n not in seen:
                    seen.add(n)
                    result.append(n)
        return result

    def _filter_anchors_by_distance(self, customers: List[int],
                                    anchors: List[int]) -> Tuple[List[int], List[int]]:
        """Return (launch, retrieve) anchors within max_anchor_dist × diameter."""
        depot_start = self._instance.customer_manager.depot_start
        depot_end = self._instance.customer_manager.depot_end
        launch_candidates = list(anchors)
        retrieve_candidates = [
            a for a in anchors
            if a != depot_start or depot_start == depot_end
        ]
        if not customers or self._max_anchor_dist_factor == float("inf"):
            return launch_candidates, retrieve_candidates

        # Compute max pairwise drone distance among neighborhood customers
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

        first_c = customers[0]
        last_c = customers[-1]
        launch_ok = []
        retrieve_ok = []
        for a in launch_candidates:
            try:
                d_launch = dm[ni[first_c]][ni[a]]
                if d_launch <= max_allowed:
                    launch_ok.append(a)
            except (KeyError, IndexError):
                pass
        for a in retrieve_candidates:
            try:
                d_retrieve = dm[ni[last_c]][ni[a]]
                if d_retrieve <= max_allowed:
                    retrieve_ok.append(a)
            except (KeyError, IndexError):
                pass

        return launch_ok or launch_candidates, retrieve_ok or retrieve_candidates

    def _select_neighborhood(self, solution: Solution) -> List[int]:
        """Select 2-4 customers for the neighborhood.

        Prioritizes multi-customer tasks (most restructuring potential),
        fills remaining slots with single-customer task customers,
        and also includes truck-served customers that could benefit from
        being reassigned to drone sorties.
        """
        multi_customers: List[int] = []
        single_customers: List[int] = []

        for t in solution.drone_tasks:
            cs = t.customers()
            if len(cs) >= 2:
                multi_customers.extend(cs)
            elif len(cs) == 1:
                single_customers.append(cs[0])

        # Also collect truck-served customers that could be drone candidates
        drone_served = set()
        for t in solution.drone_tasks:
            drone_served.update(t.customers())

        customer_ids = set(self._instance.customer_manager.customer_ids())
        truck_served = []
        for r in solution.truck_routes:
            for n in r.nodes:
                if (n not in drone_served
                        and n in customer_ids):
                    truck_served.append(n)

        # Start with multi-customer task customers (they need restructuring most)
        self._rng.shuffle(multi_customers)
        selected = multi_customers[:self._max_customers]

        # Fill remaining from single-customer tasks
        if len(selected) < self._max_customers:
            self._rng.shuffle(single_customers)
            remaining = self._max_customers - len(selected)
            selected.extend(single_customers[:remaining])

        # Fill remaining from truck-served customers (enables truck->drone transfer)
        if len(selected) < self._max_customers:
            self._rng.shuffle(truck_served)
            remaining = self._max_customers - len(selected)
            selected.extend(truck_served[:remaining])

        return selected[:self._max_customers]

    # ── Main entry ──────────────────────────────────────────

    def apply(self, solution: Solution) -> Solution:
        """Apply matheuristic LNS: select neighborhood, destroy, enumerate, rebuild.

        Returns the best improved solution, or the original if no improvement.
        """
        self.attempts += 1
        self._current_truck_routes = solution.truck_routes

        current_eval = self._evaluator.evaluate_solution(solution)
        if not current_eval.feasible or not math.isfinite(current_eval.total_cost):
            return solution

        target_cost = current_eval.total_cost
        neighborhood = self._select_neighborhood(solution)
        if len(neighborhood) < 2:
            return solution

        # ── Destroy: remove neighborhood customers from drone tasks ──
        clean = solution.clone()
        for c in neighborhood:
            for t in list(clean.drone_tasks):
                try:
                    t.remove_customer(c)
                except (ValueError, IndexError):
                    pass
        clean.drone_tasks = [t for t in clean.drone_tasks if len(t.customers()) > 0]

        # Also remove from truck routes
        for c in neighborhood:
            for r in clean.truck_routes:
                try:
                    r.remove_customer(c)
                except ValueError:
                    pass

        # ── Pre-compute feasible (launch, retrieve) anchors ──
        all_anchors = self._all_anchors()

        # Energy cache: (sorted_tuple, launch, retrieve) -> feasible
        energy_ok: Dict[Tuple, bool] = {}

        # Generate all distinct blocks and their permutations
        blocks_seen: Set[Tuple[int, ...]] = set()
        block_permutations: Dict[Tuple[int, ...], List[List[int]]] = {}

        for partition in self._set_partitions(neighborhood):
            for blk in partition:
                bk = tuple(sorted(blk))
                if bk not in blocks_seen:
                    blocks_seen.add(bk)
                    perms: List[List[int]] = []
                    for p in set(itertools.permutations(blk)):
                        perms.append(list(p))
                    block_permutations[bk] = perms

                    # Distance-filtered anchors for this block
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
            block_perms = [block_permutations[tuple(sorted(blk))] for blk in partition]
            for perm_combo in itertools.product(*block_perms):
                ordered_blocks = perm_combo
                n = len(ordered_blocks)

                # Gather viable anchor pairs for each block
                lr_opts = [feasible_lr.get(tuple(sorted(blk)), []) for blk in ordered_blocks]
                if any(len(opts) == 0 for opts in lr_opts):
                    continue

                for drone_ids in itertools.product(
                    range(self._instance.vehicle_specs["drone"].number),
                    repeat=n,
                ):
                    for lr_combo in itertools.product(*lr_opts):
                        # Build candidate
                        sol = clean.clone()
                        nid = max(
                            (t.task_id or 0) for t in sol.drone_tasks
                        ) + 1 if sol.drone_tasks else 0

                        for bi in range(n):
                            blk = ordered_blocks[bi]
                            ln, rn = lr_combo[bi]
                            di = drone_ids[bi]
                            lt = self._truck_of(ln)
                            rt = self._truck_of(rn)
                            if ln == self._instance.customer_manager.depot_start:
                                lt = None
                            if rn == self._instance.customer_manager.depot_end:
                                rt = None

                            sol.add_drone_task(DroneTask(
                                task_id=nid, drone_id=di,
                                launch_truck=lt, launch_node=ln,
                                customers=blk,
                                land_truck=rt, retrieve_node=rn,
                                payloads=_build_payloads(blk, self._demands),
                            ))
                            nid += 1

                        # Evaluate
                        candidates_evaluated += 1
                        try:
                            ev = self._evaluator.evaluate_solution(sol)
                            if ev.feasible and math.isfinite(ev.total_cost) and ev.total_cost < best_cost - 1e-6:
                                best = sol
                                best_cost = ev.total_cost
                        except (KeyError, ValueError):
                            pass

        if best_cost < target_cost:
            self.accepted += 1

        self.created += candidates_evaluated
        return best
