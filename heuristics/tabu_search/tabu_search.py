"""Constraint-aware Tabu Search for VRPFD.

Key improvements over the original:
1. Energy-aware repair: only removes energy-violating drone tasks
2. Penalized objective: allows controlled infeasibility during search
3. Violation-directed neighborhood: prioritizes fixing worst violations
4. Robust energy checks via evaluator (reuses ALNS components)
5. Adaptive penalty factors based on feasibility rate
"""

from __future__ import annotations

import math
import random
import time
from collections import deque
from typing import Any, Optional, List, Tuple, Set, Dict

from alns_vrpfd.evaluation import Evaluator
from alns_vrpfd.evaluation.energy import DroneEnergyModel
from alns_vrpfd.model.solution import Solution
from alns_vrpfd.model.route import TruckRoute, DroneTask
from alns_vrpfd.core.operators.base import _build_payloads


class DroneTaskOptimizer:
    """Drone task optimizer with robust energy checks."""

    def __init__(self, evaluator: Evaluator, rng: random.Random):
        self._evaluator = evaluator
        self._rng = rng
        self._instance = evaluator._instance

        self._vehicle_specs = self._instance.vehicle_specs
        self._demands = self._instance.customer_manager.demands()
        self._drone_cap = self._instance.vehicle_specs['drone'].capacity
        self._drone_speed = self._instance.vehicle_specs['drone'].speed
        self._drone_count = self._instance.vehicle_specs['drone'].number

        self._depot_start = self._instance.customer_manager.depot_start
        self._depot_end = self._instance.customer_manager.depot_end
        self._depots = {self._depot_start, self._depot_end}

        self._drone_dist = self._instance.distance_matrix('drone')
        self._truck_dist = self._instance.distance_matrix('truck')
        self._node_index = {n: i for i, n in enumerate(self._instance.all_node_ids())}

        self._drone_eligible = self._get_drone_eligible_customers()

        self._energy_model = DroneEnergyModel()
        self._battery = self._instance.robust_config.drone_battery_capacity
        self._deviation_rate = self._instance.robust_config.energy_deviation_rate
        self._uncertainty_budget = self._instance.robust_config.energy_uncertainty_budget

    def _get_drone_eligible_customers(self) -> Set[int]:
        eligible = set()
        for cust_id, demand in self._demands.items():
            if demand <= self._drone_cap:
                eligible.add(cust_id)
        return eligible

    def _calculate_drone_distance(self, launch: int, customers: List[int], retrieve: int) -> float:
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

    def _robust_energy_feasible(self, launch: int, customers: List[int], retrieve: int) -> bool:
        """Check if a drone sortie is robust-energy-feasible using the ALNS energy model."""
        if not customers:
            return False
        payloads = _build_payloads(customers, self._demands)
        nodes = [launch] + customers + [retrieve]
        ni = self._node_index
        dt = self._instance.time_matrix("drone")
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
        """Score a drone task. Returns -inf if energy-infeasible."""
        if not self._robust_energy_feasible(launch, customers, retrieve):
            return -float('inf')
        distance = self._calculate_drone_distance(launch, customers, retrieve)
        if distance == float('inf'):
            return -float('inf')
        score = len(customers) * 2.0
        if len(customers) >= 2:
            score += 0.8 * len(customers)
        return score

    def create_multi_customer_task(
        self,
        solution: Solution,
        launch_node: int,
        retrieve_node: int,
        candidates: List[int],
        max_customers: int = 2,
    ) -> Optional[DroneTask]:
        """Create a multi-customer drone task with robust energy check."""
        valid = [c for c in candidates if c in self._drone_eligible]
        if not valid:
            return None

        best_customers = []
        best_score = -float('inf')

        for cust in valid[:8]:
            score = self._score_drone_task([cust], launch_node, retrieve_node)
            if score > best_score:
                best_score = score
                best_customers = [cust]

        if best_customers and max_customers > 1:
            current = list(best_customers)
            for _ in range(max_customers - 1):
                best_addition = None
                best_add_score = best_score
                remaining = [c for c in valid if c not in current]
                for cust in remaining[:5]:
                    test = current + [cust]
                    score = self._score_drone_task(test, launch_node, retrieve_node)
                    if score > best_add_score:
                        best_add_score = score
                        best_addition = cust
                if best_addition:
                    current.append(best_addition)
                    best_score = best_add_score
                else:
                    break
            if len(current) > len(best_customers):
                best_customers = current

        if not best_customers or best_score < 0:
            return None

        used_drones = {task.drone_id for task in solution.drone_tasks}
        available_drone = None
        for d in range(self._drone_count):
            if d not in used_drones:
                available_drone = d
                break
        if available_drone is None:
            return None

        task_id = max((t.task_id or 0) for t in solution.drone_tasks) + 1 if solution.drone_tasks else 1
        payloads = _build_payloads(best_customers, self._demands)

        return DroneTask(
            task_id=task_id,
            drone_id=available_drone,
            launch_truck=None,
            launch_node=launch_node,
            customers=best_customers,
            land_truck=None,
            retrieve_node=retrieve_node,
            payloads=payloads,
        )

    def enumerate_drone_candidates(
        self,
        solution: Solution,
        customer: int,
    ) -> List[Tuple[int, int, List[int], int]]:
        """Enumerate ALL feasible drone positions for a customer (ALNS-style).

        Returns list of (launch_truck, launch_node, customers, retrieve_truck, retrieve_node, drone_id)
        considering same-truck and cross-truck sorties.

        IMPORTANT: Allows multiple tasks per drone (sequential, non-overlapping).
        The evaluator will validate time overlap feasibility.
        """
        candidates = []
        if customer not in self._drone_eligible:
            return candidates

        drone_count = self._vehicle_specs['drone'].number
        # Allow ALL drones (even those with existing tasks) - evaluator checks overlap
        all_drones = list(range(drone_count))

        # Enumerate all launch-retrieve combinations across all truck routes
        for launch_r_idx, launch_route in enumerate(solution.truck_routes):
            for launch_pos in range(len(launch_route.nodes)):
                launch_node = launch_route.nodes[launch_pos]

                for retrieve_r_idx, retrieve_route in enumerate(solution.truck_routes):
                    for retrieve_pos in range(len(retrieve_route.nodes)):
                        retrieve_node = retrieve_route.nodes[retrieve_pos]

                        # Skip same node
                        if launch_node == retrieve_node and launch_r_idx == retrieve_r_idx:
                            continue

                        # Check capacity
                        total_demand = self._demands.get(customer, 0.0)
                        if total_demand > self._vehicle_specs['drone'].capacity:
                            continue

                        # Check energy feasibility
                        if not self._robust_energy_feasible(launch_node, [customer], retrieve_node):
                            continue

                        # Try each drone (evaluator checks time overlap)
                        for drone_id in all_drones:
                            candidates.append((
                                launch_r_idx if launch_node != self._depot_start else None,
                                launch_node,
                                [customer],
                                retrieve_r_idx if retrieve_node != self._depot_end else None,
                                retrieve_node,
                                drone_id,
                            ))

        # Also try depot launch
        for retrieve_r_idx, retrieve_route in enumerate(solution.truck_routes):
            for retrieve_pos in range(len(retrieve_route.nodes)):
                retrieve_node = retrieve_route.nodes[retrieve_pos]
                if retrieve_node == self._depot_start:
                    continue
                if not self._robust_energy_feasible(self._depot_start, [customer], retrieve_node):
                    continue
                for drone_id in all_drones:
                    candidates.append((
                        None,
                        self._depot_start,
                        [customer],
                        retrieve_r_idx if retrieve_node != self._depot_end else None,
                        retrieve_node,
                        drone_id,
                    ))

        return candidates


class TabuSearch:
    """Constraint-aware Tabu Search for VRPFD.

    Uses penalized objective to allow controlled infeasibility during search,
    with adaptive penalty factors and violation-directed neighborhoods.
    """

    def __init__(
        self,
        *,
        evaluator: Evaluator,
        tabu_tenure: int = None,
        max_iterations: int = 1000,
        max_stagnation: int = None,
        rng: Optional[random.Random] = None,
        search_evaluator=None,
    ):
        self._evaluator = evaluator
        self._search_evaluator = search_evaluator
        self._max_iterations = max_iterations
        self._rng = rng or random.Random(random.getrandbits(32))

        self._vehicle_specs = self._evaluator._instance.vehicle_specs
        self._demands = self._evaluator._instance.customer_manager.demands()
        self._n_customers = len(self._demands)
        self._drone_eligible = self._get_drone_eligible_customers()

        self._depot_start = self._evaluator._instance.customer_manager.depot_start
        self._depot_end = self._evaluator._instance.customer_manager.depot_end
        self._depots = {self._depot_start, self._depot_end}

        if tabu_tenure is None:
            self._base_tabu_tenure = max(15, min(40, self._n_customers // 3))
        else:
            self._base_tabu_tenure = tabu_tenure

        if max_stagnation is None:
            self._max_stagnation = max(50, min(150, self._n_customers * 2))
        else:
            self._max_stagnation = max_stagnation

        if self._n_customers >= 100:
            self._max_neighbors = 200
            self._moves_per_type = 40
        elif self._n_customers >= 75:
            self._max_neighbors = 300
            self._moves_per_type = 60
        elif self._n_customers >= 50:
            self._max_neighbors = 400
            self._moves_per_type = 80
        else:
            self._max_neighbors = 250
            self._moves_per_type = 60

        if self._search_evaluator is not None:
            self._candidate_eval_limit = max(50, self._max_neighbors // 2)
        else:
            self._candidate_eval_limit = self._max_neighbors
        self._candidate_prefilter_enabled = True
        self._candidate_bucket_enabled = True
        self._candidate_bucket_shares = {
            "distance_proxy": 0.35,
            "drone_saving": 0.30,
            "violation_fix": 0.20,
            "random_diversity": 0.15,
        }
        self._last_candidate_pool_size = 0
        self._last_selected_candidate_count = 0
        self._last_selected_by_bucket = {
            "distance_proxy": 0,
            "drone_saving": 0,
            "violation_fix": 0,
            "random_diversity": 0,
        }

        self._drone_optimizer = DroneTaskOptimizer(evaluator, self._rng)
        self._truck_dist = self._evaluator._instance.distance_matrix('truck')
        self._drone_dist = self._evaluator._instance.distance_matrix('drone')
        self._all_nodes = self._evaluator._instance.all_node_ids()
        self._node_index = {n: i for i, n in enumerate(self._all_nodes)}

        # Penalized objective parameters
        self._alpha_energy = 100.0
        self._alpha_tw = 50.0
        self._alpha_coverage = 200.0
        self._alpha_capacity = 100.0
        self._alpha_hard = 100_000.0
        self._penalty_adapt_interval = 50
        self.stats = {
            "iterations": [],
            "elapsed_time": [],
            "current_cost": [],
            "best_cost": [],
            "best_feasible_cost": [],
            "neighbors_checked": [],
            "candidate_pool_size": [],
            "selected_candidate_count": [],
            "selected_by_bucket": [],
        }

    def _get_drone_eligible_customers(self) -> Set[int]:
        eligible = set()
        drone_cap = self._vehicle_specs['drone'].capacity
        for cust_id, demand in self._demands.items():
            if demand <= drone_cap:
                eligible.add(cust_id)
        return eligible

    def _penalized_cost(self, solution: Solution) -> Tuple[float, bool, Dict[str, float]]:
        """Compute penalized cost with violation penalties.

        Returns (penalized_cost, is_feasible, violation_details).
        """
        search_evaluator = getattr(self, "_search_evaluator", None)
        if search_evaluator is not None:
            return search_evaluator.penalized_cost(solution)

        try:
            details = self._evaluator.evaluate_with_details(solution)
        except Exception:
            return float('inf'), False, {"error": 1.0}

        if details.result.feasible:
            return details.result.total_cost, True, {}

        penalty = 0.0
        violations = {}

        # Energy violations (from robustness check)
        for b in details.robustness.task_breakdown:
            if not b.feasible:
                excess = abs(b.margin)
                penalty += self._alpha_energy * excess
                violations["energy"] = violations.get("energy", 0) + excess

        # Time window violations
        if details.delay_breakdown.violations:
            tw_count = len(details.delay_breakdown.violations)
            penalty += self._alpha_tw * tw_count
            violations["tw"] = tw_count

        # Delay penalty (already in total_cost for feasible, but add for infeasible)
        if details.result.delay_penalty > 0:
            penalty += details.result.delay_penalty

        hard_violations = self._hard_violations(solution, details)
        for name, amount in hard_violations.items():
            if name == "capacity":
                penalty += self._alpha_capacity * amount
            elif name == "coverage":
                penalty += self._alpha_coverage * amount
            else:
                penalty += self._alpha_hard * amount
            violations[name] = violations.get(name, 0.0) + float(amount)

        # Coverage violation
        truck_served = set()
        for r in solution.truck_routes:
            for n in r.customers():
                truck_served.add(n)
        drone_served = set()
        for t in solution.drone_tasks:
            for c in t.customers():
                drone_served.add(c)
        all_customers = set(self._demands.keys())
        missing = all_customers - truck_served - drone_served
        if missing:
            if "coverage" not in hard_violations:
                penalty += self._alpha_coverage * len(missing)
            violations["coverage"] = max(
                violations.get("coverage", 0.0),
                float(len(missing)),
            )

        # Capacity violation
        capacity_excess = self._capacity_excess(solution)
        if capacity_excess > 0:
            if "capacity" not in hard_violations:
                penalty += self._alpha_capacity * capacity_excess
            violations["capacity"] = max(
                violations.get("capacity", 0.0),
                capacity_excess,
            )

        base_cost = details.result.total_cost if math.isfinite(details.result.total_cost) else 1e6
        return base_cost + penalty, False, violations

    def _non_delay_feasible(self, solution: Solution, details=None) -> bool:
        """Check all hard constraints while allowing time-window violations."""
        if details is None:
            try:
                details = self._evaluate_with_details(solution)
            except Exception:
                return False
        return not self._hard_violations(solution, details)

    def _evaluate_with_details(self, solution: Solution):
        search_evaluator = getattr(self, "_search_evaluator", None)
        if search_evaluator is not None and hasattr(search_evaluator, "evaluate_with_details"):
            return search_evaluator.evaluate_with_details(solution)
        return self._evaluator.evaluate_with_details(solution)

    def _hard_violations(self, solution: Solution, details=None) -> Dict[str, float]:
        violations: Dict[str, float] = {}
        robustness = getattr(details, "robustness", None) if details is not None else None
        if robustness is not None and hasattr(robustness, "feasible") and not robustness.feasible:
            violations["robust"] = 1.0

        evaluator = getattr(self, "_evaluator", None)
        checks = (
            ("anchor", "_has_drone_anchor_conflicts"),
            ("drone_limit", "_has_drone_limit_violations"),
            ("drone_task", "_has_drone_task_violations"),
            ("coverage", "_has_customer_coverage_violation"),
            ("forced_drone", "_has_forced_drone_violation"),
        )
        for key, name in checks:
            check = getattr(evaluator, name, None)
            if check is None:
                continue
            try:
                violated = check(solution.drone_tasks) if key == "anchor" else check(solution)
            except Exception:
                violated = True
            if violated:
                violations[key] = 1.0

        capacity_excess = self._capacity_excess(solution)
        if capacity_excess > 0:
            violations["capacity"] = capacity_excess
        return violations

    def _capacity_excess(self, solution: Solution) -> float:
        truck_spec = (getattr(self, "_vehicle_specs", {}) or {}).get("truck")
        truck_cap = getattr(truck_spec, "capacity", float("inf"))
        excess = 0.0
        for route in getattr(solution, "truck_routes", []) or []:
            load = sum(
                self._demands.get(customer, 0.0)
                for customer in route.customers()
            )
            excess += max(0.0, load - truck_cap)
        return excess

    def _candidate_passes_non_delay_gate(self, candidate: Solution) -> bool:
        evaluator = getattr(self, "_evaluator", None)
        if evaluator is None or not hasattr(evaluator, "evaluate_with_details"):
            return True
        return self._non_delay_feasible(candidate)

    def _non_delay_infeasible_flag(self, solution: Solution, details=None) -> float:
        return 0.0 if self._non_delay_feasible(solution, details) else 1.0

    def _final_feasibility_profile(self, solution: Solution, details=None) -> Dict[str, Any]:
        """Build a compact final diagnostic without writing files."""
        try:
            if details is None:
                details = self._evaluate_with_details(solution)
        except Exception as exc:
            return {
                "error": repr(exc),
                "hard_ok": False,
                "hard_flags": {},
                "tw_count": None,
                "total_lateness": None,
                "late_nodes": [],
            }

        hard_flags = self._hard_flags(solution, details)
        hard_ok = bool(hard_flags.get("robust", False)) and not any(
            value for key, value in hard_flags.items() if key != "robust"
        )
        violations = getattr(getattr(details, "delay_breakdown", None), "violations", []) or []
        return {
            "feasible": bool(getattr(getattr(details, "result", None), "feasible", False)),
            "hard_ok": hard_ok,
            "hard_flags": hard_flags,
            "tw_count": len(violations),
            "total_lateness": self._total_lateness(details),
            "delay_cost": float(getattr(getattr(details, "result", None), "delay_penalty", 0.0) or 0.0),
            "late_nodes": self._late_node_profiles(solution, details, limit=10),
        }

    def _hard_flags(self, solution: Solution, details=None) -> Dict[str, bool]:
        robustness = getattr(details, "robustness", None) if details is not None else None
        robust_ok = bool(getattr(robustness, "feasible", False)) if robustness is not None else False
        return {
            "robust": robust_ok,
            "anchor": bool(self._safe_evaluator_check("_has_drone_anchor_conflicts", solution)),
            "drone_limit": bool(self._safe_evaluator_check("_has_drone_limit_violations", solution)),
            "drone_task": bool(self._safe_evaluator_check("_has_drone_task_violations", solution)),
            "coverage": bool(self._safe_evaluator_check("_has_customer_coverage_violation", solution)),
            "forced_drone": bool(self._safe_evaluator_check("_has_forced_drone_violation", solution)),
            "capacity": self._capacity_excess(solution) > 1e-9,
        }

    def _safe_evaluator_check(self, name: str, solution: Solution) -> bool:
        evaluator = getattr(self, "_evaluator", None)
        check = getattr(evaluator, name, None)
        if check is None:
            return False
        try:
            if name == "_has_drone_anchor_conflicts":
                return bool(check(solution.drone_tasks))
            return bool(check(solution))
        except Exception:
            return True

    def _total_lateness(self, details) -> float:
        lateness = 0.0
        violations = getattr(getattr(details, "delay_breakdown", None), "violations", []) or []
        for violation in violations:
            arrival = getattr(violation, "arrival_time", 0.0) or 0.0
            latest = getattr(violation, "latest_time", 0.0) or 0.0
            lateness += max(0.0, arrival - latest)
        return lateness

    def _late_node_profiles(self, solution: Solution, details, limit: int = 10) -> List[Dict[str, Any]]:
        anchor_map = self._drone_anchor_nodes_by_route(solution)
        profiles = []
        violations = getattr(getattr(details, "delay_breakdown", None), "violations", []) or []
        for violation in violations:
            node = getattr(violation, "node_id", None)
            served_by = getattr(violation, "served_by", None)
            route_id = getattr(violation, "route_id", None)
            route_idx, pos, route_nodes = self._find_route_position(solution, node, route_id)
            if pos is None and served_by == "truck":
                continue
            start = max(0, pos - 5) if pos is not None else 0
            end = min(len(route_nodes), pos + 6) if pos is not None else 0
            arrival = getattr(violation, "arrival_time", 0.0) or 0.0
            latest = getattr(violation, "latest_time", 0.0) or 0.0
            profiles.append({
                "node": node,
                "served_by": served_by,
                "route_id": route_id,
                "route_index": route_idx,
                "position": pos,
                "arrival": arrival,
                "latest": latest,
                "lateness": max(0.0, arrival - latest),
                "window": list(route_nodes[start:end]) if route_nodes else [],
                "is_drone_anchor": node in anchor_map.get(route_id, set()),
            })
        profiles.sort(key=lambda item: item["lateness"], reverse=True)
        return profiles[:limit]

    def _find_route_position(self, solution: Solution, node: int, route_id: int | None = None):
        for idx, route in enumerate(getattr(solution, "truck_routes", []) or []):
            if route_id is not None and getattr(route, "id", idx) != route_id:
                continue
            nodes = getattr(route, "nodes", []) or []
            if node in nodes:
                return idx, nodes.index(node), nodes
        if route_id is not None:
            for idx, route in enumerate(getattr(solution, "truck_routes", []) or []):
                nodes = getattr(route, "nodes", []) or []
                if node in nodes:
                    return idx, nodes.index(node), nodes
        return None, None, []

    def _adapt_penalties(
        self,
        *,
        feasible_count: int,
        total_checked: int,
        has_feasible_solution: bool,
    ) -> None:
        """Adapt penalties to push TS toward feasibility before cost polishing."""
        if total_checked <= 0:
            return

        feasible_rate = feasible_count / total_checked
        if not has_feasible_solution:
            if feasible_rate < 0.10:
                self._scale_constraint_penalties(1.25)
            elif feasible_rate < 0.25:
                self._scale_constraint_penalties(1.10)
            return

        if feasible_rate < 0.05:
            self._scale_constraint_penalties(1.10)
        elif feasible_rate > 0.50:
            self._scale_constraint_penalties(0.95)

    def _scale_constraint_penalties(self, factor: float) -> None:
        min_values = {
            "_alpha_energy": 25.0,
            "_alpha_tw": 25.0,
            "_alpha_coverage": 100.0,
            "_alpha_capacity": 50.0,
            "_alpha_hard": 10_000.0,
        }
        max_values = {
            "_alpha_energy": 10_000.0,
            "_alpha_tw": 10_000.0,
            "_alpha_coverage": 20_000.0,
            "_alpha_capacity": 20_000.0,
            "_alpha_hard": 1_000_000.0,
        }
        for attr, min_value in min_values.items():
            current = float(getattr(self, attr, min_value))
            scaled = current * factor
            setattr(self, attr, min(max(scaled, min_value), max_values[attr]))

    def _max_drone_tasks_allowed(self) -> int:
        """Dynamic sortie cap for larger instances with many drone-eligible customers."""
        n_customers = max(1, int(getattr(self, "_n_customers", 1)))
        eligible_count = len(getattr(self, "_drone_eligible", set()) or set())
        drone_spec = (getattr(self, "_vehicle_specs", {}) or {}).get("drone")
        drone_count = max(1, int(getattr(drone_spec, "number", 1) or 1))

        customer_scaled = int(math.ceil(n_customers * 0.16))
        eligible_scaled = int(math.ceil(eligible_count * 0.12))
        drone_scaled = drone_count * 8
        cap = max(4, customer_scaled, eligible_scaled, drone_scaled)
        return min(max(4, n_customers // 2), cap)

    def _candidate_scoring_limit(self) -> int:
        limit = getattr(self, "_candidate_eval_limit", None)
        if limit is None:
            limit = getattr(self, "_max_neighbors", 0)
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = getattr(self, "_max_neighbors", 0)
        return max(1, limit)

    def _rank_potential_moves(
        self,
        current: Solution,
        potential_moves: List[Tuple[Solution, Tuple]],
    ) -> List[Tuple[Solution, Tuple]]:
        """Select candidates for expensive scoring using mixed neighborhood buckets."""
        if not getattr(self, "_candidate_prefilter_enabled", True):
            self._set_last_candidate_selection_stats(
                pool_size=len(potential_moves),
                selected_count=len(potential_moves),
                selected_by_bucket={"distance_proxy": len(potential_moves)},
            )
            return potential_moves

        deduped = self._deduplicate_potential_moves(potential_moves)
        limit = min(len(deduped), self._candidate_scoring_limit())
        if not getattr(self, "_candidate_bucket_enabled", True):
            selected = self._select_by_distance_proxy(current, deduped, limit)
            self._set_last_candidate_selection_stats(
                pool_size=len(potential_moves),
                selected_count=len(selected),
                selected_by_bucket={"distance_proxy": len(selected)},
            )
            return selected

        selected, selected_by_bucket = self._select_by_candidate_buckets(
            current, deduped, limit
        )
        self._set_last_candidate_selection_stats(
            pool_size=len(potential_moves),
            selected_count=len(selected),
            selected_by_bucket=selected_by_bucket,
        )
        return selected

    def _deduplicate_potential_moves(
        self, potential_moves: List[Tuple[Solution, Tuple]]
    ) -> List[Tuple[Solution, Tuple]]:
        deduped = []
        seen = set()
        for neighbor, move_sig in potential_moves:
            if move_sig in seen:
                continue
            seen.add(move_sig)
            deduped.append((neighbor, move_sig))
        return deduped

    def _select_by_distance_proxy(
        self,
        current: Solution,
        potential_moves: List[Tuple[Solution, Tuple]],
        limit: int,
    ) -> List[Tuple[Solution, Tuple]]:
        ranked = []
        for pos, (neighbor, move_sig) in enumerate(potential_moves):
            try:
                score = self._cheap_neighbor_score(current, neighbor, move_sig)
            except Exception:
                score = float("inf")
            if score is None or not math.isfinite(score):
                score = float("inf")
            ranked.append((score, pos, neighbor, move_sig))

        ranked.sort(key=lambda item: (item[0], item[1]))
        return [(neighbor, move_sig) for _, _, neighbor, move_sig in ranked[:limit]]

    def _select_by_candidate_buckets(
        self,
        current: Solution,
        potential_moves: List[Tuple[Solution, Tuple]],
        limit: int,
    ) -> Tuple[List[Tuple[Solution, Tuple]], Dict[str, int]]:
        bucket_order = [
            "distance_proxy",
            "drone_saving",
            "violation_fix",
            "random_diversity",
        ]
        selected: List[Tuple[Solution, Tuple]] = []
        selected_sigs = set()
        selected_by_bucket = {name: 0 for name in bucket_order}
        quotas = self._candidate_bucket_quotas(limit, bucket_order)

        ranked_distance = self._rank_candidates_by_distance(current, potential_moves)
        ranked_drone = self._rank_candidates_by_drone_saving(current, potential_moves)
        ranked_violation = [
            item for item in potential_moves if self._is_violation_fix_move(item[1])
        ]
        random_candidates = list(potential_moves)
        self._rng.shuffle(random_candidates)

        bucket_candidates = {
            "distance_proxy": ranked_distance,
            "drone_saving": ranked_drone,
            "violation_fix": ranked_violation,
            "random_diversity": random_candidates,
        }

        for bucket in bucket_order:
            self._take_bucket_candidates(
                bucket_candidates[bucket],
                quotas.get(bucket, 0),
                selected,
                selected_sigs,
                selected_by_bucket,
                bucket,
            )

        if len(selected) < limit:
            self._take_bucket_candidates(
                ranked_distance,
                limit - len(selected),
                selected,
                selected_sigs,
                selected_by_bucket,
                "distance_proxy",
            )

        return selected, selected_by_bucket

    def _candidate_bucket_quotas(
        self, limit: int, bucket_order: List[str]
    ) -> Dict[str, int]:
        shares = getattr(self, "_candidate_bucket_shares", {}) or {}
        raw = {
            bucket: max(0.0, float(shares.get(bucket, 0.0))) * limit
            for bucket in bucket_order
        }
        quotas = {bucket: int(math.floor(raw[bucket])) for bucket in bucket_order}
        assigned = sum(quotas.values())
        remainders = sorted(
            ((raw[bucket] - quotas[bucket], bucket) for bucket in bucket_order),
            key=lambda item: (-item[0], bucket_order.index(item[1])),
        )
        for _, bucket in remainders:
            if assigned >= limit:
                break
            quotas[bucket] += 1
            assigned += 1
        if assigned < limit:
            quotas["distance_proxy"] += limit - assigned
        return quotas

    def _take_bucket_candidates(
        self,
        candidates: List[Tuple[Solution, Tuple]],
        quota: int,
        selected: List[Tuple[Solution, Tuple]],
        selected_sigs: Set[Tuple],
        selected_by_bucket: Dict[str, int],
        bucket: str,
    ) -> None:
        if quota <= 0:
            return
        for neighbor, move_sig in candidates:
            if len(selected) >= self._candidate_scoring_limit():
                break
            if selected_by_bucket[bucket] >= quota:
                break
            if move_sig in selected_sigs:
                continue
            selected.append((neighbor, move_sig))
            selected_sigs.add(move_sig)
            selected_by_bucket[bucket] += 1

    def _rank_candidates_by_distance(
        self,
        current: Solution,
        candidates: List[Tuple[Solution, Tuple]],
    ) -> List[Tuple[Solution, Tuple]]:
        ranked = []
        for pos, (neighbor, move_sig) in enumerate(candidates):
            try:
                score = self._cheap_neighbor_score(current, neighbor, move_sig)
            except Exception:
                score = float("inf")
            if score is None or not math.isfinite(score):
                score = float("inf")
            ranked.append((score, pos, neighbor, move_sig))
        ranked.sort(key=lambda item: (item[0], item[1]))
        return [(neighbor, move_sig) for _, _, neighbor, move_sig in ranked]

    def _rank_candidates_by_drone_saving(
        self,
        current: Solution,
        candidates: List[Tuple[Solution, Tuple]],
    ) -> List[Tuple[Solution, Tuple]]:
        ranked = []
        for pos, (neighbor, move_sig) in enumerate(candidates):
            if not self._is_drone_saving_move(move_sig, current, neighbor):
                continue
            try:
                score = self._drone_saving_score(current, neighbor, move_sig)
            except Exception:
                score = float("inf")
            if score is None or not math.isfinite(score):
                score = float("inf")
            ranked.append((score, pos, neighbor, move_sig))
        ranked.sort(key=lambda item: (item[0], item[1]))
        return [(neighbor, move_sig) for _, _, neighbor, move_sig in ranked]

    def _is_violation_fix_move(self, move_sig: Tuple) -> bool:
        move_type = move_sig[0] if move_sig else ""
        return isinstance(move_type, str) and move_type.startswith("fix_")

    def _is_drone_saving_move(
        self, move_sig: Tuple, current: Solution, neighbor: Solution
    ) -> bool:
        move_type = move_sig[0] if move_sig else ""
        if move_type in {"drone_insert", "cross_truck_drone"}:
            return True
        return self._drone_customer_count(neighbor) > self._drone_customer_count(current)

    def _drone_saving_score(
        self, current: Solution, neighbor: Solution, move_sig: Tuple
    ) -> float:
        current_truck = self._truck_customer_count(current)
        neighbor_truck = self._truck_customer_count(neighbor)
        truck_removed = max(0, current_truck - neighbor_truck)
        neighbor_distance = self._cheap_solution_distance_proxy(neighbor)
        current_distance = self._cheap_solution_distance_proxy(current)
        distance_delta = neighbor_distance - current_distance
        return distance_delta - 10.0 * truck_removed

    @staticmethod
    def _truck_customer_count(solution: Solution) -> int:
        total = 0
        for route in getattr(solution, "truck_routes", []) or []:
            customers_attr = getattr(route, "customers", None)
            if callable(customers_attr):
                total += len(customers_attr())
            else:
                nodes = getattr(route, "nodes", []) or []
                total += max(0, len(nodes) - 2)
        return total

    @staticmethod
    def _drone_customer_count(solution: Solution) -> int:
        total = 0
        for task in getattr(solution, "drone_tasks", []) or []:
            customers_attr = getattr(task, "customers", [])
            customers = customers_attr() if callable(customers_attr) else customers_attr
            total += len(customers or [])
        return total

    def _set_last_candidate_selection_stats(
        self,
        *,
        pool_size: int,
        selected_count: int,
        selected_by_bucket: Dict[str, int],
    ) -> None:
        buckets = {
            "distance_proxy": 0,
            "drone_saving": 0,
            "violation_fix": 0,
            "random_diversity": 0,
        }
        buckets.update(selected_by_bucket)
        self._last_candidate_pool_size = pool_size
        self._last_selected_candidate_count = selected_count
        self._last_selected_by_bucket = buckets

    def _cheap_neighbor_score(self, current: Solution, neighbor: Solution, move_sig: Tuple) -> float:
        return self._cheap_solution_distance_proxy(neighbor)

    def _cheap_solution_distance_proxy(self, solution: Solution) -> float:
        total = 0.0
        truck_dist = getattr(self, "_truck_dist", None)
        drone_dist = getattr(self, "_drone_dist", truck_dist)
        node_index = getattr(self, "_node_index", {})

        for route in getattr(solution, "truck_routes", []) or []:
            nodes = getattr(route, "nodes", []) or []
            total += self._path_distance_proxy(nodes, truck_dist, node_index)

        for task in getattr(solution, "drone_tasks", []) or []:
            customers_attr = getattr(task, "customers", [])
            customers = customers_attr() if callable(customers_attr) else customers_attr
            nodes = [getattr(task, "launch_node", None)] + list(customers or []) + [
                getattr(task, "retrieve_node", None)
            ]
            total += self._path_distance_proxy(nodes, drone_dist, node_index)

        return total

    @staticmethod
    def _path_distance_proxy(nodes: List[int], matrix, node_index: Dict[int, int]) -> float:
        total = 0.0
        for a, b in zip(nodes, nodes[1:]):
            if a is None or b is None:
                return float("inf")
            try:
                total += matrix[node_index[a]][node_index[b]]
            except (KeyError, IndexError, TypeError):
                return float("inf")
        return total

    def _reset_stats(self) -> None:
        self.stats = {
            "iterations": [],
            "elapsed_time": [],
            "current_cost": [],
            "best_cost": [],
            "best_feasible_cost": [],
            "neighbors_checked": [],
            "candidate_pool_size": [],
            "selected_candidate_count": [],
            "selected_by_bucket": [],
        }

    def _record_stats(
        self,
        *,
        iteration: int,
        start_time: float,
        current_cost: float,
        best_cost: float,
        best_feasible_cost: float,
        neighbors_checked: int,
    ) -> None:
        if not hasattr(self, "stats"):
            self._reset_stats()
        self.stats["iterations"].append(iteration)
        self.stats["elapsed_time"].append(time.perf_counter() - start_time)
        self.stats["current_cost"].append(current_cost)
        self.stats["best_cost"].append(best_cost)
        self.stats["best_feasible_cost"].append(best_feasible_cost)
        self.stats["neighbors_checked"].append(neighbors_checked)
        self.stats["candidate_pool_size"].append(
            getattr(self, "_last_candidate_pool_size", 0)
        )
        self.stats["selected_candidate_count"].append(
            getattr(self, "_last_selected_candidate_count", 0)
        )
        self.stats["selected_by_bucket"].append(
            dict(getattr(self, "_last_selected_by_bucket", {}))
        )

    def _repair_energy_violations(self, solution: Solution) -> Solution:
        """Remove only energy-violating drone tasks, keep feasible ones.

        For removed tasks, enumerate ALL feasible drone positions (ALNS-style)
        before falling back to truck insertion.
        """
        repaired = solution.clone()
        details = self._evaluator.evaluate_with_details(repaired)

        if details.robustness.feasible:
            return repaired

        # Identify violating tasks
        violating_task_ids = set()
        for b in details.robustness.task_breakdown:
            if not b.feasible:
                violating_task_ids.add(b.task_id)

        if not violating_task_ids:
            return repaired

        # Collect customers from violating tasks
        customers_to_reassign = []
        surviving_tasks = []
        for task in repaired.drone_tasks:
            if task.task_id in violating_task_ids:
                customers_to_reassign.extend(task.customers())
            else:
                surviving_tasks.append(task)

        repaired.drone_tasks = surviving_tasks

        # Try ALNS-style drone reinsert for each customer
        still_unassigned = []
        for cust in customers_to_reassign:
            inserted = self._try_drone_reinsert_alns(repaired, cust)
            if not inserted:
                still_unassigned.append(cust)

        # Fall back to truck insertion for remaining customers
        truck_cap = self._vehicle_specs['truck'].capacity
        for cust in still_unassigned:
            self._insert_customer_with_capacity(repaired, cust, truck_cap)

        return repaired

    def _try_drone_reinsert_alns(self, solution: Solution, customer: int) -> bool:
        """Try to insert a customer into a drone task using ALNS-style enumeration."""
        if customer not in self._drone_eligible:
            return False

        candidates = self._drone_optimizer.enumerate_drone_candidates(solution, customer)
        if not candidates:
            return False

        # Try each candidate, keeping the best feasible one
        best_sol = None
        best_cost = float('inf')

        for launch_truck, launch_node, custs, retrieve_truck, retrieve_node, drone_id in candidates:
            candidate_sol = solution.clone()
            task_id = max((t.task_id or 0) for t in candidate_sol.drone_tasks) + 1 if candidate_sol.drone_tasks else 1
            payloads = _build_payloads(custs, self._demands)
            new_task = DroneTask(
                task_id=task_id,
                drone_id=drone_id,
                launch_truck=launch_truck,
                launch_node=launch_node,
                customers=custs,
                land_truck=retrieve_truck,
                retrieve_node=retrieve_node,
                payloads=payloads,
            )
            candidate_sol.drone_tasks.append(new_task)

            # Remove customer from truck routes
            for r in candidate_sol.truck_routes:
                if customer in r.nodes:
                    r.nodes.remove(customer)
                    break

            # Evaluate
            try:
                cost, feasible, _ = self._penalized_cost(candidate_sol)
                if feasible and cost < best_cost:
                    best_cost = cost
                    best_sol = candidate_sol
            except Exception:
                continue

        if best_sol is not None:
            solution.truck_routes = best_sol.truck_routes
            solution.drone_tasks = best_sol.drone_tasks
            return True

        return False

    def _insert_customer_with_capacity(
        self, solution: Solution, customer: int, truck_capacity: float
    ) -> bool:
        """Insert customer into best feasible truck position using full evaluator check."""
        if not solution.truck_routes:
            return False

        customer_demand = self._demands.get(customer, 0.0)
        best_cost = float('inf')
        best_route_idx = -1
        best_pos = -1

        for r_idx, route in enumerate(solution.truck_routes):
            current_load = sum(
                self._demands.get(node, 0.0)
                for node in route.nodes
                if node not in self._depots
            )
            if current_load + customer_demand > truck_capacity:
                continue

            for pos in range(1, len(route.nodes)):
                prev_node = route.nodes[pos - 1]
                next_node = route.nodes[pos]
                prev_idx = self._node_index.get(prev_node)
                cust_idx = self._node_index.get(customer)
                next_idx = self._node_index.get(next_node)
                if prev_idx is None or cust_idx is None or next_idx is None:
                    continue
                cost = (
                    self._truck_dist[prev_idx][cust_idx]
                    + self._truck_dist[cust_idx][next_idx]
                    - self._truck_dist[prev_idx][next_idx]
                )
                if cost < best_cost:
                    best_cost = cost
                    best_route_idx = r_idx
                    best_pos = pos

        if best_route_idx >= 0 and best_pos >= 0:
            if customer not in solution.truck_routes[best_route_idx].nodes:
                solution.truck_routes[best_route_idx].nodes.insert(best_pos, customer)
                return True

        if customer_demand <= truck_capacity:
            solution.truck_routes.append(
                TruckRoute(
                    route_id=len(solution.truck_routes),
                    nodes=[self._depot_start, customer, self._depot_end],
                    capacity=truck_capacity,
                )
            )
            return True
        return False

    def _ensure_all_customers_served(self, solution: Solution) -> Solution:
        all_customers = set(self._demands.keys())
        served = set()
        for route in solution.truck_routes:
            for node in route.nodes:
                if node not in self._depots:
                    served.add(node)
        for task in solution.drone_tasks:
            for cust in task.customers():
                served.add(cust)
        missing = all_customers - served
        if missing and solution.truck_routes:
            truck_cap = self._vehicle_specs['truck'].capacity
            for cust in missing:
                self._insert_customer_with_capacity(solution, cust, truck_cap)
        return solution

    def _apply_2opt(self, solution: Solution) -> Solution:
        improved = solution.clone()
        dist = self._truck_dist
        idx = self._node_index
        for route in improved.truck_routes:
            nodes = route.nodes
            if len(nodes) <= 4:
                continue
            best_improvement = 0
            best_i, best_j = -1, -1
            for i in range(1, len(nodes) - 2):
                for j in range(i + 2, len(nodes) - 1):
                    n_i, n_i1 = nodes[i], nodes[i - 1]
                    n_j, n_j1 = nodes[j], nodes[j + 1]
                    try:
                        old_cost = dist[idx[n_i1]][idx[n_i]] + dist[idx[n_j]][idx[n_j1]]
                        new_cost = dist[idx[n_i1]][idx[n_j]] + dist[idx[n_i]][idx[n_j1]]
                        improvement = old_cost - new_cost
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_i, best_j = i, j
                    except (KeyError, IndexError):
                        continue
            if best_i > 0:
                route.nodes = nodes[:best_i] + nodes[best_i:best_j+1][::-1] + nodes[best_j+1:]
        return improved

    def _apply_drone_optimization(self, solution: Solution) -> Solution:
        optimized = solution.clone()
        for _ in range(3):
            eligible = set()
            for cid, demand in self._demands.items():
                if demand <= self._vehicle_specs['drone'].capacity:
                    eligible.add(cid)
            drone_served = set()
            for t in optimized.drone_tasks:
                drone_served.update(t.customers())
            eligible -= drone_served

            if not eligible:
                break

            tasks_created = 0
            for route_idx, route in enumerate(optimized.truck_routes):
                if len(route.nodes) < 4:
                    continue
                route_eligible = [n for n in route.nodes if n in eligible]
                if not route_eligible:
                    continue

                best_task = None
                best_customers = []

                for launch_pos in range(len(route.nodes) - 2):
                    launch_node = route.nodes[launch_pos]
                    if launch_node in eligible:
                        continue
                    for retrieve_pos in range(launch_pos + 2, min(launch_pos + 5, len(route.nodes))):
                        retrieve_node = route.nodes[retrieve_pos]
                        if retrieve_node in eligible:
                            continue
                        candidates = [route.nodes[i] for i in range(launch_pos + 1, retrieve_pos)
                                      if route.nodes[i] in eligible]
                        if not candidates:
                            continue
                        task = self._drone_optimizer.create_multi_customer_task(
                            optimized, launch_node, retrieve_node, candidates[:3], max_customers=2
                        )
                        if task and len(task.customers()) > len(best_customers):
                            best_task = task
                            best_customers = list(task.customers())

                if best_task:
                    for cust in best_customers:
                        if cust in optimized.truck_routes[route_idx].nodes:
                            optimized.truck_routes[route_idx].nodes.remove(cust)
                        eligible.discard(cust)
                    optimized.drone_tasks.append(best_task)
                    tasks_created += 1

            if tasks_created == 0:
                break
        return optimized

    def _gen_ruin_recreate(self, solution: Solution):
        routes = solution.truck_routes
        if not routes:
            return
        dist = self._truck_dist
        idx = self._node_index
        all_customers = []
        for route in routes:
            for node in route.nodes:
                if node not in self._depots:
                    all_customers.append(node)
        if len(all_customers) < 3:
            return
        remove_count = max(3, int(len(all_customers) * 0.20))

        for _ in range(5):
            neighbor = solution.clone()
            removed = self._rng.sample(all_customers, min(remove_count, len(all_customers)))
            for route in neighbor.truck_routes:
                route.nodes = [n for n in route.nodes if n not in removed]
            for cust in removed:
                best_cost = float('inf')
                best_pos = None
                best_route_idx = None
                cust_idx = idx.get(cust)
                if cust_idx is None:
                    continue
                for r_idx, route in enumerate(neighbor.truck_routes):
                    for pos in range(1, len(route.nodes)):
                        prev = route.nodes[pos - 1]
                        curr = route.nodes[pos]
                        prev_idx = idx.get(prev)
                        curr_idx = idx.get(curr)
                        if prev_idx is None or curr_idx is None:
                            continue
                        ins_cost = dist[prev_idx][cust_idx] + dist[cust_idx][curr_idx] - dist[prev_idx][curr_idx]
                        if ins_cost < best_cost:
                            best_cost = ins_cost
                            best_pos = pos
                            best_route_idx = r_idx
                if best_route_idx is not None:
                    neighbor.truck_routes[best_route_idx].nodes.insert(best_pos, cust)
            yield neighbor, ("ruin_recreate", "random", tuple(removed[:3]))

    def _gen_or_opt(self, solution: Solution):
        routes = solution.truck_routes
        if len(routes) < 1:
            return
        for chain_len in [2, 3]:
            for r_idx, route in enumerate(routes):
                nodes = route.nodes
                if len(nodes) < chain_len + 3:
                    continue
                for start_pos in range(1, len(nodes) - chain_len - 1):
                    chain = nodes[start_pos:start_pos + chain_len]
                    for target_r_idx, target_route in enumerate(routes):
                        target_nodes = target_route.nodes
                        for ins_pos in range(1, len(target_nodes)):
                            if r_idx == target_r_idx and abs(ins_pos - start_pos) <= chain_len:
                                continue
                            neighbor = solution.clone()
                            src_route = neighbor.truck_routes[r_idx]
                            for _ in range(chain_len):
                                src_route.nodes.pop(start_pos)
                            actual_ins = ins_pos
                            if r_idx == target_r_idx and ins_pos > start_pos:
                                actual_ins = ins_pos - chain_len
                            tgt_route = neighbor.truck_routes[target_r_idx]
                            for i, cust in enumerate(chain):
                                tgt_route.nodes.insert(actual_ins + i, cust)
                            yield neighbor, ("or_opt", chain_len, tuple(chain))

    def _gen_truck_relocate(self, solution: Solution):
        routes = solution.truck_routes
        if len(routes) < 2:
            return
        idxs = list(range(len(routes)))
        self._rng.shuffle(idxs)
        for i in idxs:
            route_src = routes[i]
            if len(route_src.nodes) <= 2:
                continue
            cust_idxs = list(range(1, len(route_src.nodes) - 1))
            self._rng.shuffle(cust_idxs)
            for c_idx in cust_idxs[:3]:
                customer = route_src.nodes[c_idx]
                for j in idxs:
                    if i == j:
                        continue
                    route_dst = routes[j]
                    for pos in range(1, len(route_dst.nodes)):
                        neighbor = solution.clone()
                        neighbor.truck_routes[i].nodes.pop(c_idx)
                        neighbor.truck_routes[j].nodes.insert(pos, customer)
                        yield neighbor, ("relocate", customer, i, j)

    def _gen_drone_moves(self, solution: Solution):
        """Generate drone moves for ONE random eligible customer (ALNS-style).

        Instead of generating all possible moves, pick one random eligible
        customer and try all drone positions for it. This matches ALNS's
        destroy-repair approach where one customer is destroyed and reinserted.
        """
        routes = solution.truck_routes
        if not routes:
            return

        # Collect all eligible truck customers
        eligible_customers = []
        for r_idx, route in enumerate(routes):
            for k, node in enumerate(route.nodes):
                if node in self._drone_eligible:
                    eligible_customers.append((r_idx, k, node))

        if not eligible_customers:
            return

        # Pick ONE random customer
        r_idx, k, customer = self._rng.choice(eligible_customers)

        # Try all launch-retrieve combinations for this customer
        for launch_r_idx, launch_route in enumerate(routes):
            for launch_pos in range(len(launch_route.nodes)):
                launch_node = launch_route.nodes[launch_pos]

                for retrieve_r_idx, retrieve_route in enumerate(routes):
                    for retrieve_pos in range(len(retrieve_route.nodes)):
                        retrieve_node = retrieve_route.nodes[retrieve_pos]

                        # Skip same node on same route
                        if launch_node == retrieve_node and launch_r_idx == retrieve_r_idx:
                            continue

                        # Check energy feasibility
                        if not self._drone_optimizer._robust_energy_feasible(
                            launch_node, [customer], retrieve_node
                        ):
                            continue

                        # Create candidate solution
                        neighbor = solution.clone()
                        task_id = max((t.task_id or 0) for t in neighbor.drone_tasks) + 1 if neighbor.drone_tasks else 1
                        drone_id = 0  # Use any drone (evaluator checks overlap)

                        payloads = _build_payloads([customer], self._demands)
                        new_task = DroneTask(
                            task_id=task_id,
                            drone_id=drone_id,
                            launch_truck=launch_r_idx if launch_node != self._depot_start else None,
                            launch_node=launch_node,
                            customers=[customer],
                            land_truck=retrieve_r_idx if retrieve_node != self._depot_end else None,
                            retrieve_node=retrieve_node,
                            payloads=payloads,
                        )
                        neighbor.drone_tasks.append(new_task)

                        # Remove customer from truck route
                        for route in neighbor.truck_routes:
                            if customer in route.nodes:
                                route.nodes.remove(customer)
                                break

                        if self._non_delay_feasible(neighbor):
                            yield neighbor, ("drone_insert", customer, launch_node, retrieve_node)

        # Also try drone-to-truck moves
        if solution.drone_tasks:
            task_idx = self._rng.randint(0, len(solution.drone_tasks) - 1)
            task = solution.drone_tasks[task_idx]
            customers = task.customers()
            if customers:
                neighbor = solution.clone()
                neighbor.drone_tasks.pop(task_idx)
                target_r_idx = self._rng.randint(0, len(routes) - 1)
                if len(routes[target_r_idx].nodes) >= 2:
                    for customer in customers:
                        pos = self._rng.randint(1, len(neighbor.truck_routes[target_r_idx].nodes) - 1)
                        neighbor.truck_routes[target_r_idx].nodes.insert(pos, customer)
                    if self._non_delay_feasible(neighbor):
                        yield neighbor, ("drone_to_truck", tuple(customers))

    def _gen_violation_directed_moves(self, solution: Solution):
        """Generate moves targeted at fixing worst violations."""
        try:
            details = self._evaluator.evaluate_with_details(solution)
        except Exception:
            return

        yield from self._gen_critical_delay_drone_moves(solution, details)
        yield from self._gen_truck_backbone_rechain_moves(solution, details)

        if details.robustness.feasible:
            return

        violating_tasks = []
        for b in details.robustness.task_breakdown:
            if not b.feasible:
                violating_tasks.append(b)

        for b in violating_tasks[:2]:
            task = None
            for t in solution.drone_tasks:
                if t.task_id == b.task_id:
                    task = t
                    break
            if task is None:
                continue

            customers = list(task.customers())

            if len(customers) > 1:
                for split_pos in range(1, len(customers)):
                    front = customers[:split_pos]
                    back = customers[split_pos:]
                    for custs in [front, back]:
                        if not custs:
                            continue
                        if self._drone_optimizer._robust_energy_feasible(
                            task.launch_node, custs, task.retrieve_node
                        ):
                            neighbor = solution.clone()
                            for t2 in neighbor.drone_tasks:
                                if t2.task_id == b.task_id:
                                    t2.nodes = [t2.launch_node] + custs + [t2.retrieve_node]
                                    t2.payloads = _build_payloads(custs, self._demands)
                                    break
                            other_custs = [c for c in customers if c not in custs]
                            if other_custs:
                                truck_cap = self._vehicle_specs['truck'].capacity
                                for c in other_custs:
                                    self._insert_customer_with_capacity(neighbor, c, truck_cap)
                            yield neighbor, ("fix_energy_split", b.task_id, tuple(custs))

            for cust in customers:
                neighbor = solution.clone()
                neighbor.drone_tasks = [t for t in neighbor.drone_tasks if t.task_id != b.task_id]
                truck_cap = self._vehicle_specs['truck'].capacity
                self._insert_customer_with_capacity(neighbor, cust, truck_cap)
                yield neighbor, ("fix_energy_remove", b.task_id, cust)

    def _gen_critical_delay_drone_moves(self, solution: Solution, details):
        """Move the most delayed drone-eligible truck customers to drone sorties."""
        critical_customers = self._critical_delay_customers(details, limit=3)
        if not critical_customers or not getattr(solution, "truck_routes", None):
            return

        drone_cap = self._vehicle_specs["drone"].capacity
        for customer in critical_customers:
            if self._demands.get(customer, float("inf")) > drone_cap:
                continue
            if not self._customer_on_truck(solution, customer):
                continue

            for launch_r_idx, launch_route in enumerate(solution.truck_routes):
                for launch_pos, launch_node in enumerate(launch_route.nodes):
                    if launch_node == customer:
                        continue

                    for retrieve_r_idx, retrieve_route in enumerate(solution.truck_routes):
                        for retrieve_pos, retrieve_node in enumerate(retrieve_route.nodes):
                            if retrieve_node == customer:
                                continue
                            if launch_node == retrieve_node and launch_r_idx == retrieve_r_idx:
                                continue
                            if launch_r_idx == retrieve_r_idx and launch_pos > retrieve_pos:
                                continue
                            if not self._drone_optimizer._robust_energy_feasible(
                                launch_node, [customer], retrieve_node
                            ):
                                continue

                            neighbor = solution.clone()
                            task_id = (
                                max((t.task_id or 0) for t in neighbor.drone_tasks) + 1
                                if neighbor.drone_tasks else 1
                            )
                            payloads = _build_payloads([customer], self._demands)
                            new_task = DroneTask(
                                task_id=task_id,
                                drone_id=0,
                                launch_truck=launch_r_idx if launch_node != self._depot_start else None,
                                launch_node=launch_node,
                                customers=[customer],
                                land_truck=retrieve_r_idx if retrieve_node != self._depot_end else None,
                                retrieve_node=retrieve_node,
                                payloads=payloads,
                            )
                            neighbor.drone_tasks.append(new_task)
                            for route in neighbor.truck_routes:
                                if customer in route.nodes:
                                    route.nodes.remove(customer)
                                    break
                            if self._non_delay_feasible(neighbor):
                                yield neighbor, ("fix_tw_drone_insert", customer, launch_node, retrieve_node)

    def _gen_truck_backbone_rechain_moves(self, solution: Solution, details=None):
        """Generate local truck backbone repairs around late truck-served nodes."""
        if details is None:
            try:
                details = self._evaluate_with_details(solution)
            except Exception:
                return

        late_nodes = self._late_truck_nodes(details, limit=5)
        if not late_nodes:
            return

        base_score = self._delay_repair_score(solution, details)
        emitted = 0
        move_limit = int(getattr(self, "_moves_per_type", 60))

        for route_idx, route in enumerate(getattr(solution, "truck_routes", []) or []):
            route_id = getattr(route, "id", route_idx)
            route_late_nodes = [node for node in late_nodes if node in route.nodes]
            if route_late_nodes:
                route_customers = [n for n in route.nodes if n not in self._depots]
                sorted_route = sorted(route_customers, key=self._latest_time_key)
                if len(sorted_route) > 1 and sorted_route != route_customers:
                    neighbor = solution.clone()
                    neighbor.truck_routes[route_idx].nodes = [
                        route.nodes[0],
                        *sorted_route,
                        route.nodes[-1],
                    ]
                    touched = self._touched_map((route_id, set(route_customers)))
                    self._repair_touched_drone_anchors(neighbor, touched)
                    if self._is_improving_non_delay_candidate(neighbor, base_score):
                        emitted += 1
                        yield neighbor, ("fix_tw_truck_rechain_route_latest_sort", route_id, tuple(route_late_nodes))
                    if emitted >= move_limit:
                        return

                # Pull the late cluster to the earliest feasible prefix, preserving
                # the relative urgency order inside the cluster.
                urgent = sorted(route_late_nodes, key=self._latest_time_key)
                if urgent:
                    remaining = [n for n in route_customers if n not in set(urgent)]
                    truck_spec = (getattr(self, "_vehicle_specs", {}) or {}).get("truck")
                    if remaining and truck_spec is not None:
                        neighbor = solution.clone()
                        neighbor.truck_routes[route_idx].nodes = [
                            route.nodes[0],
                            *remaining,
                            route.nodes[-1],
                        ]
                        new_route_id = max(
                            (getattr(r, "id", i) for i, r in enumerate(neighbor.truck_routes)),
                            default=-1,
                        ) + 1
                        truck_cap = truck_spec.capacity
                        neighbor.truck_routes.append(
                            TruckRoute(
                                route_id=new_route_id,
                                nodes=[self._depot_start, *urgent, self._depot_end],
                                capacity=truck_cap,
                            )
                        )
                        touched = self._touched_map((route_id, set(urgent)))
                        self._repair_touched_drone_anchors(neighbor, touched)
                        if self._is_improving_non_delay_candidate(neighbor, base_score):
                            emitted += 1
                            yield neighbor, ("fix_tw_late_ejection_route_split", route_id, new_route_id, tuple(urgent))
                        if emitted >= move_limit:
                            return

                    for prefix_len in range(0, min(6, len(remaining)) + 1):
                        candidate_order = remaining[:prefix_len] + urgent + remaining[prefix_len:]
                        if candidate_order == route_customers:
                            continue
                        neighbor = solution.clone()
                        neighbor.truck_routes[route_idx].nodes = [
                            route.nodes[0],
                            *candidate_order,
                            route.nodes[-1],
                        ]
                        touched = self._touched_map((route_id, set(route_customers)))
                        self._repair_touched_drone_anchors(neighbor, touched)
                        if self._is_improving_non_delay_candidate(neighbor, base_score):
                            emitted += 1
                            yield neighbor, ("fix_tw_late_ejection_route_prefix", route_id, tuple(urgent), prefix_len)
                        if emitted >= move_limit:
                            return

                    # Move the contiguous segment spanning all late nodes as a
                    # block. This targets R_50_50_1-like cases where several
                    # late nodes sit in one local truck backbone cluster.
                    late_positions = [route.nodes.index(node) for node in route_late_nodes]
                    seg_start = max(1, min(late_positions) - 1)
                    seg_end = min(len(route.nodes) - 1, max(late_positions) + 2)
                    segment = [n for n in route.nodes[seg_start:seg_end] if n not in self._depots]
                    if len(segment) >= 2:
                        sorted_segment = sorted(segment, key=self._latest_time_key)
                        base_without_segment = [n for n in route_customers if n not in set(segment)]
                        for insert_pos in range(0, len(base_without_segment) + 1):
                            for candidate_segment in (segment, sorted_segment):
                                candidate_order = (
                                    base_without_segment[:insert_pos]
                                    + list(candidate_segment)
                                    + base_without_segment[insert_pos:]
                                )
                                if candidate_order == route_customers:
                                    continue
                                neighbor = solution.clone()
                                neighbor.truck_routes[route_idx].nodes = [
                                    route.nodes[0],
                                    *candidate_order,
                                    route.nodes[-1],
                                ]
                                touched = self._touched_map((route_id, set(route_customers)))
                                self._repair_touched_drone_anchors(neighbor, touched)
                                if self._is_improving_non_delay_candidate(neighbor, base_score):
                                    emitted += 1
                                    yield neighbor, (
                                        "fix_tw_late_ejection_segment",
                                        route_id,
                                        tuple(route_late_nodes),
                                        insert_pos,
                                    )
                                if emitted >= move_limit:
                                    return

            for late_node in late_nodes:
                if late_node not in route.nodes:
                    continue
                pos = route.nodes.index(late_node)
                if pos <= 0 or pos >= len(route.nodes) - 1:
                    continue
                start = max(1, pos - 4)
                end = min(len(route.nodes) - 1, pos + 5)
                window_nodes = set(route.nodes[start:end])

                # Local window sorted by latest time: a focused TW repair move.
                window_customers = [n for n in route.nodes[start:end] if n not in self._depots]
                if len(window_customers) > 1:
                    sorted_window = sorted(window_customers, key=self._latest_time_key)
                    if sorted_window != window_customers:
                        neighbor = solution.clone()
                        neighbor.truck_routes[route_idx].nodes[start:end] = sorted_window
                        touched = self._touched_map((route_id, set(window_customers)))
                        self._repair_touched_drone_anchors(neighbor, touched)
                        if self._is_improving_non_delay_candidate(neighbor, base_score):
                            emitted += 1
                            yield neighbor, ("fix_tw_truck_rechain_latest_sort", route_id, start, tuple(sorted_window))
                        if emitted >= move_limit:
                            return

                # Eject the late node and try every route/position reinsertion.
                for target_r_idx, target_route in enumerate(solution.truck_routes):
                    target_id = getattr(target_route, "id", target_r_idx)
                    for target_pos in range(1, len(target_route.nodes)):
                        if target_r_idx == route_idx and target_pos in {pos, pos + 1}:
                            continue
                        neighbor = solution.clone()
                        source_nodes = neighbor.truck_routes[route_idx].nodes
                        if late_node not in source_nodes:
                            continue
                        source_pos = source_nodes.index(late_node)
                        moved = source_nodes.pop(source_pos)
                        insert_pos = target_pos
                        if target_r_idx == route_idx and target_pos > source_pos:
                            insert_pos -= 1
                        insert_pos = max(1, min(insert_pos, len(neighbor.truck_routes[target_r_idx].nodes) - 1))
                        neighbor.truck_routes[target_r_idx].nodes.insert(insert_pos, moved)
                        touched = self._touched_map(
                            (route_id, window_nodes | {late_node}),
                            (target_id, {late_node}),
                        )
                        self._repair_touched_drone_anchors(neighbor, touched)
                        if self._is_improving_non_delay_candidate(neighbor, base_score):
                            emitted += 1
                            yield neighbor, ("fix_tw_late_ejection", late_node, route_id, target_id, target_pos)
                        if emitted >= move_limit:
                            return

                for target_pos in range(start, end + 1):
                    if target_pos == pos or target_pos == pos + 1:
                        continue
                    neighbor = solution.clone()
                    moved = neighbor.truck_routes[route_idx].nodes.pop(pos)
                    insert_pos = target_pos
                    if target_pos > pos:
                        insert_pos -= 1
                    insert_pos = max(1, min(insert_pos, len(neighbor.truck_routes[route_idx].nodes) - 1))
                    neighbor.truck_routes[route_idx].nodes.insert(insert_pos, moved)
                    self._repair_touched_drone_anchors(neighbor, {route_id: window_nodes | {late_node}})
                    if self._is_improving_non_delay_candidate(neighbor, base_score):
                        emitted += 1
                        yield neighbor, ("fix_tw_truck_rechain_relocate", late_node, route_id, target_pos)
                    if emitted >= move_limit:
                        return

                for seg_len in (2, 3):
                    seg_start_min = max(1, pos - seg_len + 1)
                    seg_start_max = min(pos, len(route.nodes) - seg_len - 1)
                    for seg_start in range(seg_start_min, seg_start_max + 1):
                        seg_end = seg_start + seg_len
                        segment = route.nodes[seg_start:seg_end]
                        neighbor = solution.clone()
                        neighbor.truck_routes[route_idx].nodes[seg_start:seg_end] = list(reversed(segment))
                        self._repair_touched_drone_anchors(neighbor, {route_id: set(segment)})
                        if self._is_improving_non_delay_candidate(neighbor, base_score):
                            emitted += 1
                            yield neighbor, ("fix_tw_truck_rechain_reverse", route_id, seg_start, seg_len)
                        if emitted >= move_limit:
                            return

                if len(solution.truck_routes) < 2:
                    continue
                for other_idx, other_route in enumerate(solution.truck_routes):
                    if other_idx == route_idx or len(other_route.nodes) <= 2:
                        continue
                    other_id = getattr(other_route, "id", other_idx)
                    # Cross-route relocate of the late node.
                    for target_pos in range(1, len(other_route.nodes)):
                        neighbor = solution.clone()
                        src_nodes = neighbor.truck_routes[route_idx].nodes
                        if late_node not in src_nodes:
                            continue
                        src_nodes.pop(src_nodes.index(late_node))
                        neighbor.truck_routes[other_idx].nodes.insert(target_pos, late_node)
                        touched = self._touched_map(
                            (route_id, window_nodes | {late_node}),
                            (other_id, {late_node}),
                        )
                        self._repair_touched_drone_anchors(neighbor, touched)
                        if self._is_improving_non_delay_candidate(neighbor, base_score):
                            emitted += 1
                            yield neighbor, ("fix_tw_truck_rechain_cross_relocate", late_node, route_id, other_id, target_pos)
                        if emitted >= move_limit:
                            return

                    for other_pos in range(1, len(other_route.nodes) - 1):
                        other_node = other_route.nodes[other_pos]
                        neighbor = solution.clone()
                        src_nodes = neighbor.truck_routes[route_idx].nodes
                        dst_nodes = neighbor.truck_routes[other_idx].nodes
                        src_nodes[pos], dst_nodes[other_pos] = dst_nodes[other_pos], src_nodes[pos]
                        touched = self._touched_map(
                            (route_id, window_nodes | {late_node, other_node}),
                            (other_id, {late_node, other_node}),
                        )
                        self._repair_touched_drone_anchors(neighbor, touched)
                        if self._is_improving_non_delay_candidate(neighbor, base_score):
                            emitted += 1
                            yield neighbor, ("fix_tw_truck_rechain_exchange", late_node, other_node, route_id, other_id)
                        if emitted >= move_limit:
                            return

    def _repair_touched_drone_anchors(self, solution: Solution, touched_by_route: Dict[int, Set[int]]) -> List[int]:
        """Remove drone tasks anchored in touched truck windows and truck-insert their customers."""
        if not touched_by_route:
            return []

        removed_customers: List[int] = []
        surviving_tasks: List[DroneTask] = []
        for task in getattr(solution, "drone_tasks", []) or []:
            touches_launch = (
                task.launch_truck is not None
                and task.launch_node in touched_by_route.get(task.launch_truck, set())
            )
            touches_land = (
                task.land_truck is not None
                and task.retrieve_node in touched_by_route.get(task.land_truck, set())
            )
            if touches_launch or touches_land:
                removed_customers.extend(task.customers())
            else:
                surviving_tasks.append(task)
        if len(surviving_tasks) == len(getattr(solution, "drone_tasks", []) or []):
            return []

        solution.drone_tasks = surviving_tasks
        truck_cap = self._vehicle_specs["truck"].capacity
        for customer in removed_customers:
            if not self._customer_served_by_truck(solution, customer):
                self._insert_customer_with_capacity(solution, customer, truck_cap)
        return removed_customers

    def _customer_served_by_truck(self, solution: Solution, customer: int) -> bool:
        return any(customer in getattr(route, "nodes", []) for route in getattr(solution, "truck_routes", []) or [])

    def _latest_time_key(self, node: int) -> Tuple[float, int]:
        if node in self._depots:
            return (float("inf"), node)
        try:
            _, latest = self._evaluator._instance.customer_manager.time_window(node)
        except Exception:
            latest = None
        if latest is None:
            latest = float("inf")
        return (float(latest), node)

    def _touched_map(self, *items: Tuple[int, Set[int]]) -> Dict[int, Set[int]]:
        touched: Dict[int, Set[int]] = {}
        for route_id, nodes in items:
            touched.setdefault(route_id, set()).update(nodes)
        return touched

    def _late_truck_nodes(self, details, limit: int) -> List[int]:
        violations = getattr(getattr(details, "delay_breakdown", None), "violations", []) or []
        scored = []
        for violation in violations:
            if getattr(violation, "served_by", "truck") != "truck":
                continue
            customer = getattr(violation, "node_id", None)
            if customer is None:
                continue
            arrival = getattr(violation, "arrival_time", 0.0) or 0.0
            latest = getattr(violation, "latest_time", 0.0) or 0.0
            scored.append((arrival - latest, customer))
        scored.sort(reverse=True)
        return [customer for _, customer in scored[:limit]]

    def _drone_anchor_nodes_by_route(self, solution: Solution) -> Dict[int, Set[int]]:
        protected: Dict[int, Set[int]] = {}
        for task in getattr(solution, "drone_tasks", []) or []:
            if task.launch_truck is not None:
                protected.setdefault(task.launch_truck, set()).add(task.launch_node)
            if task.land_truck is not None:
                protected.setdefault(task.land_truck, set()).add(task.retrieve_node)
        return protected

    def _is_improving_non_delay_candidate(self, candidate: Solution, base_score: Tuple) -> bool:
        try:
            details = self._evaluate_with_details(candidate)
        except Exception:
            return False
        if not self._non_delay_feasible(candidate, details):
            return False
        return self._delay_repair_score(candidate, details) < base_score

    def _delay_repair_score(self, solution: Solution, details) -> Tuple[float, float, float, float, float]:
        violations = getattr(getattr(details, "delay_breakdown", None), "violations", []) or []
        lateness = self._total_lateness(details)
        result = getattr(details, "result", None)
        delay_penalty = float(getattr(result, "delay_penalty", 0.0) or 0.0)
        distance = self._cheap_solution_distance_proxy(solution)
        if not math.isfinite(distance):
            distance = 1e9
        return (
            self._non_delay_infeasible_flag(solution, details),
            float(len(violations)),
            lateness,
            delay_penalty,
            distance,
        )

    def _critical_delay_customers(self, details, limit: int) -> List[int]:
        violations = getattr(getattr(details, "delay_breakdown", None), "violations", []) or []
        scored = []
        for violation in violations:
            customer = getattr(violation, "node_id", None)
            if customer not in self._drone_eligible:
                continue
            arrival = getattr(violation, "arrival_time", 0.0) or 0.0
            latest = getattr(violation, "latest_time", 0.0) or 0.0
            scored.append((arrival - latest, customer))
        scored.sort(reverse=True)
        return [customer for _, customer in scored[:limit]]

    @staticmethod
    def _customer_on_truck(solution: Solution, customer: int) -> bool:
        for route in getattr(solution, "truck_routes", []) or []:
            if customer in getattr(route, "nodes", []):
                return True
        return False

    def _perturb(self, solution: Solution) -> Solution:
        new_sol = solution.clone()
        base_moves = max(3, self._n_customers // 10)
        n_moves = self._rng.randint(base_moves, base_moves * 2)
        for _ in range(n_moves):
            move_type = self._rng.choice(['swap', 'relocate', 'drone_shuffle'])
            routes = new_sol.truck_routes
            if len(routes) < 2:
                continue
            if move_type == 'drone_shuffle' and new_sol.drone_tasks:
                task_idx = self._rng.randint(0, len(new_sol.drone_tasks) - 1)
                task = new_sol.drone_tasks.pop(task_idx)
                for cust in task.customers():
                    target_r = self._rng.randint(0, len(routes) - 1)
                    if len(routes[target_r].nodes) > 1:
                        pos = self._rng.randint(1, len(routes[target_r].nodes) - 1)
                        routes[target_r].nodes.insert(pos, cust)
                continue
            r1_idx = self._rng.randint(0, len(routes) - 1)
            r2_idx = self._rng.randint(0, len(routes) - 1)
            r1 = routes[r1_idx]
            r2 = routes[r2_idx]
            if len(r1.nodes) > 2 and len(r2.nodes) > 2:
                c1_idx = self._rng.randint(1, len(r1.nodes) - 2)
                if move_type == 'swap':
                    c2_idx = self._rng.randint(1, len(r2.nodes) - 2)
                    r1.nodes[c1_idx], r2.nodes[c2_idx] = r2.nodes[c2_idx], r1.nodes[c1_idx]
                else:
                    cust = r1.nodes.pop(c1_idx)
                    ins_idx = self._rng.randint(1, len(r2.nodes) - 1)
                    r2.nodes.insert(ins_idx, cust)
        return new_sol

    def _apply_final_tw_polish(self, solution: Solution, deadline: float | None) -> Solution:
        """Final truck-backbone-only repair for hard-feasible TW-infeasible states."""
        try:
            details = self._evaluate_with_details(solution)
        except Exception:
            return solution
        if not self._non_delay_feasible(solution, details):
            return solution
        if not getattr(getattr(details, "delay_breakdown", None), "violations", None):
            return solution

        best = solution.clone()
        best_details = details
        best_score = self._delay_repair_score(best, best_details)
        iterations = 0
        max_iterations = max(20, min(250, int(getattr(self, "_n_customers", 50)) * 4))

        def expired() -> bool:
            return deadline is not None and time.perf_counter() >= deadline

        while not expired() and iterations < max_iterations:
            iterations += 1
            best_neighbor = None
            best_neighbor_details = None
            best_neighbor_score = best_score
            try:
                candidates = list(self._gen_truck_backbone_rechain_moves(best, best_details))
            except Exception:
                candidates = []
            if not candidates:
                break

            for neighbor, _ in candidates:
                if expired():
                    break
                try:
                    neighbor_details = self._evaluate_with_details(neighbor)
                except Exception:
                    continue
                if not self._non_delay_feasible(neighbor, neighbor_details):
                    continue
                score = self._delay_repair_score(neighbor, neighbor_details)
                if score < best_neighbor_score:
                    best_neighbor = neighbor
                    best_neighbor_details = neighbor_details
                    best_neighbor_score = score

            if best_neighbor is None:
                break

            best = best_neighbor
            best_details = best_neighbor_details
            best_score = best_neighbor_score
            if getattr(best_details.result, "feasible", False):
                break

        if not hasattr(self, "stats"):
            self._reset_stats()
        self.stats["tw_polish_iterations"] = iterations
        self.stats["tw_polish_best_score"] = tuple(float(v) for v in best_score)
        return best

    def _remember_final_feasibility_profile(self, solution: Solution) -> None:
        if not hasattr(self, "stats"):
            self._reset_stats()
        self.stats["final_feasibility_profile"] = self._final_feasibility_profile(solution)

    def run(self, initial: Solution, time_limit: float | None = None) -> Solution:
        """Run constraint-aware Tabu Search with penalized objective."""
        self._reset_stats()
        current = initial.clone()

        # Energy-aware initial repair
        try:
            current = self._repair_energy_violations(current)
            current = self._ensure_all_customers_served(current)
        except Exception:
            pass

        current_cost, current_feasible, _ = self._penalized_cost(current)
        best = current.clone()
        best_cost = current_cost
        best_feasible = current_feasible
        best_feasible_sol = current.clone() if current_feasible else None
        best_feasible_cost = current_cost if current_feasible else float('inf')

        tabu_list = deque(maxlen=self._base_tabu_tenure)
        stagnation_counter = 0
        start_time = time.perf_counter()
        polish_budget = 0.0
        if time_limit is not None and time_limit > 30.0:
            polish_budget = min(90.0, time_limit * 0.20)
        final_deadline = start_time + time_limit if time_limit is not None else None
        deadline = (
            start_time + max(0.0, time_limit - polish_budget)
            if time_limit is not None else None
        )
        feasible_count = 0
        total_checked = 0

        def time_expired() -> bool:
            return deadline is not None and time.perf_counter() >= deadline

        for iteration in range(self._max_iterations):
            if time_expired():
                break
            if best_feasible and stagnation_counter >= self._max_stagnation:
                break

            # Adaptive penalty adjustment
            if iteration > 0 and iteration % self._penalty_adapt_interval == 0 and total_checked > 0:
                self._adapt_penalties(
                    feasible_count=feasible_count,
                    total_checked=total_checked,
                    has_feasible_solution=best_feasible_sol is not None,
                )
                feasible_count = 0
                total_checked = 0

            best_neighbor = None
            best_neighbor_cost = float('inf')
            best_neighbor_feasible = False
            best_move = None

            generators = [
                self._gen_violation_directed_moves(current),
                self._gen_truck_backbone_rechain_moves(current),
                self._gen_violation_directed_moves(current),
                self._gen_drone_moves(current),
                self._gen_drone_moves(current),
                self._gen_cross_truck_drone_moves(current),
                self._gen_truck_relocate(current),
                self._gen_ruin_recreate(current),
                self._gen_or_opt(current),
            ]

            potential_moves = []
            for gen in generators:
                if time_expired():
                    break
                count = 0
                for neighbor, move_sig in gen:
                    if time_expired():
                        break
                    if not self._candidate_passes_non_delay_gate(neighbor):
                        continue
                    potential_moves.append((neighbor, move_sig))
                    count += 1
                    if count > self._moves_per_type:
                        break

            self._rng.shuffle(potential_moves)
            potential_moves = self._rank_potential_moves(current, potential_moves)

            neighbors_checked = 0
            for neighbor, move_sig in potential_moves:
                if time_expired():
                    break
                is_tabu = move_sig in tabu_list
                total_checked += 1

                max_drone_tasks = self._max_drone_tasks_allowed()
                if len(neighbor.drone_tasks) > max_drone_tasks:
                    neighbors_checked += 1
                    if neighbors_checked >= self._max_neighbors:
                        break
                    continue

                try:
                    if (
                        self._search_evaluator is not None
                        and not self._search_evaluator.verify_candidate(
                            base=current,
                            candidate=neighbor,
                        )
                    ):
                        neighbors_checked += 1
                        if neighbors_checked >= self._max_neighbors:
                            break
                        continue

                    cost, feasible, _ = self._penalized_cost(neighbor)
                    if feasible:
                        feasible_count += 1

                    if is_tabu and (not feasible or cost >= best_cost):
                        neighbors_checked += 1
                        if neighbors_checked >= self._max_neighbors:
                            break
                        continue

                    if cost < best_neighbor_cost:
                        best_neighbor = neighbor
                        best_neighbor_cost = cost
                        best_neighbor_feasible = feasible
                        best_move = move_sig
                except Exception:
                    continue

                neighbors_checked += 1
                if neighbors_checked >= self._max_neighbors:
                    break

            if time_expired():
                break

            if best_neighbor is not None:
                current = best_neighbor
                current_cost = best_neighbor_cost
                current_feasible = best_neighbor_feasible
                tabu_list.append(best_move)

                if current_cost < best_cost:
                    best = current.clone()
                    best_cost = current_cost
                    best_feasible = current_feasible
                    stagnation_counter = 0

                    if current_feasible and current_cost < best_feasible_cost:
                        best_feasible_sol = current.clone()
                        best_feasible_cost = current_cost

                        improved = self._apply_2opt(current)
                        try:
                            imp_cost, imp_feas, _ = self._penalized_cost(improved)
                            if imp_feas and imp_cost < current_cost:
                                current = improved
                                current_cost = imp_cost
                                best = current.clone()
                                best_cost = current_cost
                                best_feasible = True
                                best_feasible_sol = current.clone()
                                best_feasible_cost = current_cost
                        except Exception:
                            pass

                        try:
                            drone_opt = self._apply_drone_optimization(current)
                            drone_cost, drone_feas, _ = self._penalized_cost(drone_opt)
                            if drone_feas and drone_cost < current_cost:
                                current = drone_opt
                                current_cost = drone_cost
                                best = current.clone()
                                best_cost = current_cost
                                best_feasible = True
                                best_feasible_sol = current.clone()
                                best_feasible_cost = current_cost
                        except Exception:
                            pass

                else:
                    stagnation_counter += 1
            else:
                stagnation_counter += 1

            # Perturbation restart
            if time_expired():
                break
            restart_threshold = max(30, self._max_stagnation // 2)
            if stagnation_counter >= restart_threshold:
                current = self._perturb(best)
                current = self._apply_2opt(current)
                try:
                    current = self._repair_energy_violations(current)
                    current = self._ensure_all_customers_served(current)
                except Exception:
                    pass

                current_cost, current_feasible, _ = self._penalized_cost(current)
                if current_cost < best_cost:
                    best = current.clone()
                    best_cost = current_cost
                    best_feasible = current_feasible
                stagnation_counter = 0
                tabu_list.clear()

            self._record_stats(
                iteration=iteration,
                start_time=start_time,
                current_cost=current_cost,
                best_cost=best_cost,
                best_feasible_cost=best_feasible_cost,
                neighbors_checked=neighbors_checked,
            )

        # Return best feasible solution if found
        if best_feasible_sol is not None:
            self._remember_final_feasibility_profile(best_feasible_sol)
            return best_feasible_sol

        # Final time-window polish: keep hard constraints valid and repair truck backbone only.
        try:
            polished = self._apply_final_tw_polish(best, final_deadline)
            polished_details = self._evaluate_with_details(polished)
            if polished_details.result.feasible:
                self._remember_final_feasibility_profile(polished)
                return polished
            if (
                self._non_delay_feasible(polished, polished_details)
                and self._delay_repair_score(polished, polished_details)
                < self._delay_repair_score(best, self._evaluate_with_details(best))
            ):
                best = polished.clone()
                best_cost, best_feasible, _ = self._penalized_cost(best)
        except Exception:
            pass

        # Fallback: try to repair best
        try:
            repaired = self._repair_energy_violations(best)
            repaired = self._ensure_all_customers_served(repaired)
            rep_cost, rep_feas, _ = self._penalized_cost(repaired)
            if rep_feas:
                self._remember_final_feasibility_profile(repaired)
                return repaired
        except Exception:
            pass

        # Final fallback: try to fix infeasible solution
        best_eval = self._evaluator.evaluate_solution(best)
        if not best_eval.feasible:
            # Step 1: Remove only energy-violating drone tasks
            repaired = best.clone()
            details = self._evaluator.evaluate_with_details(repaired)
            if not details.robustness.feasible:
                violating_ids = {b.task_id for b in details.robustness.task_breakdown if not b.feasible}
                if violating_ids:
                    removed_customers = []
                    for t in repaired.drone_tasks:
                        if t.task_id in violating_ids:
                            removed_customers.extend(t.customers())
                    repaired.drone_tasks = [t for t in repaired.drone_tasks if t.task_id not in violating_ids]
                    truck_dist = self._evaluator._instance.distance_matrix("truck")
                    node_index = {n: i for i, n in enumerate(self._evaluator._instance.all_node_ids())}
                    for cust in removed_customers:
                        best_pos, best_route, best_delta = None, None, float('inf')
                        for route in repaired.truck_routes:
                            for pos in range(1, len(route.nodes)):
                                prev, nxt = route.nodes[pos-1], route.nodes[pos]
                                i_prev, i_cust, i_nxt = node_index.get(prev), node_index.get(cust), node_index.get(nxt)
                                if i_prev is not None and i_cust is not None and i_nxt is not None:
                                    delta = truck_dist[i_prev][i_cust] + truck_dist[i_cust][i_nxt] - truck_dist[i_prev][i_nxt]
                                    if delta < best_delta:
                                        best_delta, best_pos, best_route = delta, pos, route
                        if best_route and best_pos:
                            best_route.nodes.insert(best_pos, cust)
                repaired_eval = self._evaluator.evaluate_solution(repaired)
                if repaired_eval.feasible:
                    self._remember_final_feasibility_profile(repaired)
                    return repaired
                try:
                    repaired_details = self._evaluate_with_details(repaired)
                    best_details = self._evaluate_with_details(best)
                    if (
                        self._non_delay_feasible(repaired, repaired_details)
                        and self._delay_repair_score(repaired, repaired_details)
                        < self._delay_repair_score(best, best_details)
                    ):
                        best = repaired.clone()
                except Exception:
                    pass

            # Step 2: If still infeasible, strip ALL drone tasks
            if not self._evaluator.evaluate_solution(best).feasible:
                repaired2 = best.clone()
                drone_customers = []
                for t in repaired2.drone_tasks:
                    drone_customers.extend(t.customers())
                repaired2.drone_tasks = []
                truck_dist = self._evaluator._instance.distance_matrix("truck")
                node_index = {n: i for i, n in enumerate(self._evaluator._instance.all_node_ids())}
                for cust in drone_customers:
                    best_pos, best_route, best_delta = None, None, float('inf')
                    for route in repaired2.truck_routes:
                        for pos in range(1, len(route.nodes)):
                            prev, nxt = route.nodes[pos-1], route.nodes[pos]
                            i_prev, i_cust, i_nxt = node_index.get(prev), node_index.get(cust), node_index.get(nxt)
                            if i_prev is not None and i_cust is not None and i_nxt is not None:
                                delta = truck_dist[i_prev][i_cust] + truck_dist[i_cust][i_nxt] - truck_dist[i_prev][i_nxt]
                                if delta < best_delta:
                                    best_delta, best_pos, best_route = delta, pos, route
                    if best_route and best_pos:
                        best_route.nodes.insert(best_pos, cust)
                # Sort by latest time window
                depot_start = self._evaluator._instance.customer_manager.depot_start
                depot_end = self._evaluator._instance.customer_manager.depot_end
                for route in repaired2.truck_routes:
                    customers = [n for n in route.nodes if n != depot_start and n != depot_end]
                    if len(customers) > 1:
                        def get_latest(node):
                            _, latest = self._evaluator._instance.customer_manager.time_window(node)
                            return latest if latest is not None else float('inf')
                        customers.sort(key=get_latest)
                        route.nodes = [depot_start] + customers + [depot_end]
                repaired2_eval = self._evaluator.evaluate_solution(repaired2)
                if repaired2_eval.feasible:
                    self._remember_final_feasibility_profile(repaired2)
                    return repaired2
                try:
                    repaired2_details = self._evaluate_with_details(repaired2)
                    best_details = self._evaluate_with_details(best)
                    if (
                        self._non_delay_feasible(repaired2, repaired2_details)
                        and self._delay_repair_score(repaired2, repaired2_details)
                        < self._delay_repair_score(best, best_details)
                    ):
                        best = repaired2.clone()
                except Exception:
                    pass

        self._remember_final_feasibility_profile(best)
        return best

    def _gen_cross_truck_drone_moves(self, solution: Solution):
        """Generate cross-truck drone sortie moves (ALNS-inspired)."""
        routes = solution.truck_routes
        if len(routes) < 2:
            return

        # Try to create cross-truck sorties by moving customers between trucks
        for r1_idx, route1 in enumerate(routes):
            if len(route1.nodes) < 4:
                continue
            for r2_idx, route2 in enumerate(routes):
                if r1_idx == r2_idx or len(route2.nodes) < 4:
                    continue

                # Try launching from route1 and retrieving on route2
                for launch_pos in range(1, len(route1.nodes) - 1):
                    launch_node = route1.nodes[launch_pos]
                    if launch_node in self._drone_eligible:
                        continue

                    for retrieve_pos in range(1, len(route2.nodes) - 1):
                        retrieve_node = route2.nodes[retrieve_pos]
                        if retrieve_node in self._drone_eligible:
                            continue

                        # Find customers between launch and retrieve that could be droneserved
                        candidates = []
                        for k in range(launch_pos + 1, len(route1.nodes) - 1):
                            c = route1.nodes[k]
                            if c in self._drone_eligible:
                                candidates.append(c)
                        for k in range(1, retrieve_pos):
                            c = route2.nodes[k]
                            if c in self._drone_eligible and c not in candidates:
                                candidates.append(c)

                        if not candidates:
                            continue

                        # Try to create a drone task
                        for custs in [candidates[:1], candidates[:2], candidates[:3]]:
                            if not custs:
                                continue
                            total_demand = sum(self._demands.get(c, 0) for c in custs)
                            if total_demand > self._vehicle_specs['drone'].capacity:
                                continue

                            # Check energy feasibility
                            if not self._drone_optimizer._robust_energy_feasible(launch_node, custs, retrieve_node):
                                continue

                            # Create candidate solution
                            neighbor = solution.clone()
                            task_id = max((t.task_id or 0) for t in neighbor.drone_tasks) + 1 if neighbor.drone_tasks else 1
                            used_drones = {t.drone_id for t in neighbor.drone_tasks}
                            drone_count = self._vehicle_specs['drone'].number
                            available = None
                            for d in range(drone_count):
                                if d not in used_drones:
                                    available = d
                                    break
                            if available is None:
                                continue

                            payloads = _build_payloads(custs, self._demands)
                            new_task = DroneTask(
                                task_id=task_id,
                                drone_id=available,
                                launch_truck=r1_idx,
                                launch_node=launch_node,
                                customers=custs,
                                land_truck=r2_idx,
                                retrieve_node=retrieve_node,
                                payloads=payloads,
                            )
                            neighbor.drone_tasks.append(new_task)

                            # Remove customers from truck routes
                            for c in custs:
                                for r in neighbor.truck_routes:
                                    if c in r.nodes:
                                        r.nodes.remove(c)
                                        break

                            if self._non_delay_feasible(neighbor):
                                yield neighbor, ("cross_truck_drone", launch_node, tuple(custs), retrieve_node)
