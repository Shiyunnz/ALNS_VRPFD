"""Search-phase evaluator helpers shared by TS and GA.

This wrapper keeps the canonical :class:`Evaluator` as the source of truth,
while adding the cache and candidate-gate behavior that metaheuristics need
inside tight loops.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple
import math

from alns_vrpfd.evaluation.evaluator import Evaluator, EvaluationResult
from alns_vrpfd.evaluation.subroute_robust_verifier import SubrouteRobustVerifier
from alns_vrpfd.model.solution import Solution


def _finite_or_zero(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if math.isfinite(numeric) else 0.0


class SearchEvaluator:
    """Shared evaluator facade for heuristic search loops.

    The canonical evaluator remains responsible for exact feasibility and cost.
    This class adds ALNS-style robust feasibility caching and optional
    sub-route candidate verification so TS/GA can avoid repeated full checks.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        *,
        candidate_subroute_verifier: Optional[SubrouteRobustVerifier] = None,
        robust_cache_size: int = 100_000,
        alpha_energy: float = 100.0,
        alpha_tw: float = 50.0,
        alpha_coverage: float = 200.0,
        alpha_capacity: float = 100.0,
        alpha_hard: float = 100_000.0,
    ) -> None:
        self._evaluator = evaluator
        self._cached_instance = _CachedInstanceProxy(evaluator._instance)
        self._candidate_subroute_verifier = candidate_subroute_verifier
        self._robust_cache_size = max(0, int(robust_cache_size))
        self._robust_feasible_cache: Dict[tuple, bool] = {}
        self._robust_cache_order: Deque[tuple] = deque()

        self.alpha_energy = float(alpha_energy)
        self.alpha_tw = float(alpha_tw)
        self.alpha_coverage = float(alpha_coverage)
        self.alpha_capacity = float(alpha_capacity)
        self.alpha_hard = float(alpha_hard)

        self.full_eval_calls = 0
        self.detail_eval_calls = 0
        self.robust_eval_calls = 0
        self.robust_cache_hits = 0
        self.candidate_checks = 0
        self.candidate_rejections = 0

        instance = evaluator._instance
        self._demands = instance.customer_manager.demands()
        self._truck_capacity = instance.vehicle_specs["truck"].capacity

    def __getattr__(self, name: str) -> Any:
        return getattr(self._evaluator, name)

    @property
    def matrix_cache_hits(self) -> int:
        return self._cached_instance.cache_hits

    @property
    def matrix_cache_misses(self) -> int:
        return self._cached_instance.cache_misses

    def evaluate_solution(self, solution: Solution) -> EvaluationResult:
        self.full_eval_calls += 1
        return self._with_cached_instance(lambda: self._evaluator.evaluate_solution(solution))

    def evaluate_with_details(self, solution: Solution):
        self.detail_eval_calls += 1
        return self._with_cached_instance(lambda: self._evaluator.evaluate_with_details(solution))

    def verify_candidate(self, *, base: Solution, candidate: Solution) -> bool:
        """Return whether a candidate passes the configured robust gate."""
        self.candidate_checks += 1
        if self._candidate_subroute_verifier is not None:
            ok = self._candidate_subroute_verifier.verify_candidate(
                base=base,
                candidate=candidate,
            )
        else:
            ok = self.robust_feasible_cached(candidate)
        if not ok:
            self.candidate_rejections += 1
        return ok

    def robust_feasible_cached(self, solution: Solution) -> bool:
        """Check robust feasibility with a drone-task signature cache."""
        signature = self._drone_only_signature(solution)
        if self._robust_cache_size > 0:
            cached = self._robust_feasible_cache.get(signature)
            if cached is not None:
                self.robust_cache_hits += 1
                return cached

        try:
            details = self.evaluate_with_details(solution)
            feasible = bool(details.robustness.feasible)
        except Exception:
            feasible = False
        self.robust_eval_calls += 1

        if self._robust_cache_size > 0:
            self._remember_robust_feasible(signature, feasible)
        return feasible

    def penalized_cost(self, solution: Solution) -> Tuple[float, bool, Dict[str, float]]:
        """Compute a finite search cost for infeasible solutions."""
        try:
            details = self.evaluate_with_details(solution)
        except Exception:
            return float("inf"), False, {"error": 1.0}

        result = details.result
        if result.feasible and math.isfinite(result.total_cost):
            return result.total_cost, True, {}

        penalty = 0.0
        violations: Dict[str, float] = {}

        for breakdown in details.robustness.task_breakdown:
            if not breakdown.feasible:
                excess = abs(_finite_or_zero(getattr(breakdown, "margin", 0.0)))
                penalty += self.alpha_energy * excess
                violations["energy"] = violations.get("energy", 0.0) + excess

        delay_violations = getattr(details.delay_breakdown, "violations", ()) or ()
        if delay_violations:
            count = float(len(delay_violations))
            penalty += self.alpha_tw * count
            violations["tw"] = count

        delay_penalty = _finite_or_zero(getattr(result, "delay_penalty", 0.0))
        if delay_penalty > 0:
            penalty += delay_penalty

        hard_violations = self._hard_violations(solution, details)
        for name, amount in hard_violations.items():
            if name == "capacity":
                penalty += self.alpha_capacity * amount
            elif name == "coverage":
                penalty += self.alpha_coverage * amount
            else:
                penalty += self.alpha_hard * amount
            violations[name] = violations.get(name, 0.0) + float(amount)

        missing = self._missing_customers(solution)
        if missing:
            if "coverage" not in hard_violations:
                penalty += self.alpha_coverage * len(missing)
            violations["coverage"] = max(
                violations.get("coverage", 0.0),
                float(len(missing)),
            )

        capacity_excess = self._capacity_excess(solution)
        if capacity_excess > 0:
            if "capacity" not in hard_violations:
                penalty += self.alpha_capacity * capacity_excess
            violations["capacity"] = max(
                violations.get("capacity", 0.0),
                capacity_excess,
            )

        base_cost = result.total_cost if math.isfinite(result.total_cost) else 1e6
        return base_cost + penalty, False, violations

    def _hard_violations(self, solution: Solution, details) -> Dict[str, float]:
        violations: Dict[str, float] = {}

        robustness = getattr(details, "robustness", None)
        if robustness is not None and hasattr(robustness, "feasible") and not robustness.feasible:
            violations["robust"] = 1.0

        checks = (
            ("duplicate_route_id", "_has_duplicate_truck_route_ids"),
            ("duplicate_task_id", "_has_duplicate_drone_task_ids"),
            ("anchor", "_has_drone_anchor_conflicts"),
            ("drone_limit", "_has_drone_limit_violations"),
            ("drone_task", "_has_drone_task_violations"),
            ("coverage", "_has_customer_coverage_violation"),
            ("forced_drone", "_has_forced_drone_violation"),
        )
        for key, name in checks:
            check = getattr(self._evaluator, name, None)
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

    def _remember_robust_feasible(self, signature: tuple, feasible: bool) -> None:
        if signature in self._robust_feasible_cache:
            self._robust_feasible_cache[signature] = feasible
            return
        if len(self._robust_feasible_cache) >= self._robust_cache_size:
            oldest = self._robust_cache_order.popleft()
            self._robust_feasible_cache.pop(oldest, None)
        self._robust_feasible_cache[signature] = feasible
        self._robust_cache_order.append(signature)

    def _drone_only_signature(self, solution: Solution) -> tuple:
        return tuple(
            sorted(
                (
                    int(task.drone_id),
                    task.launch_truck,
                    int(task.launch_node),
                    tuple(int(c) for c in task.customers()),
                    task.land_truck,
                    int(task.retrieve_node),
                    tuple(float(p) for p in task.payloads),
                )
                for task in solution.drone_tasks
            )
        )

    def _missing_customers(self, solution: Solution) -> set[int]:
        served = set()
        for route in solution.truck_routes:
            served.update(route.customers())
        for task in solution.drone_tasks:
            served.update(task.customers())
        return set(self._demands) - served

    def _capacity_excess(self, solution: Solution) -> float:
        excess = 0.0
        for route in solution.truck_routes:
            load = sum(self._demands.get(customer, 0.0) for customer in route.customers())
            excess += max(0.0, load - self._truck_capacity)
        return excess

    def _with_cached_instance(self, callback):
        original = self._evaluator._instance
        self._evaluator._instance = self._cached_instance
        try:
            return callback()
        finally:
            self._evaluator._instance = original


class _CachedInstanceProxy:
    """Proxy immutable instance data and cache dense matrices for search loops."""

    def __init__(self, instance) -> None:
        self._instance = instance
        self._node_ids = instance.all_node_ids()
        self._distance_matrices: Dict[str, Any] = {}
        self._time_matrices: Dict[str, Any] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._instance, name)

    def all_node_ids(self):
        return self._node_ids

    def distance_matrix(self, vehicle_type: str):
        key = vehicle_type.lower()
        if key in self._distance_matrices:
            self.cache_hits += 1
            return self._distance_matrices[key]
        self.cache_misses += 1
        matrix = self._instance.distance_matrix(vehicle_type)
        self._distance_matrices[key] = matrix
        return matrix

    def time_matrix(self, vehicle_type: str):
        key = vehicle_type.lower()
        if key in self._time_matrices:
            self.cache_hits += 1
            return self._time_matrices[key]
        self.cache_misses += 1
        matrix = self._instance.time_matrix(vehicle_type)
        self._time_matrices[key] = matrix
        return matrix
