"""ALNS main loop and operator base class (scaffold)."""

from __future__ import annotations

from abc import ABC, abstractmethod
import random
import time
from typing import Iterable, Optional

from alns_vrpfd.evaluation import Evaluator
from alns_vrpfd.model.solution import Solution
from .adaptive import AdaptiveOperatorManager
from alns_vrpfd.core.operators import DestroyOperator, RepairOperator

__all__ = ["Operator", "LocalSearchOperator", "alns_search"]


class Operator(ABC):
    """Abstract base class for operators that modify a solution."""

    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def apply(self, solution: Solution) -> Solution:
        """Apply the operator to a solution and return the modified result."""


class LocalSearchOperator(ABC):
    """Abstract base class for local search operators."""

    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def apply(self, solution: Solution, unassigned: list[int]) -> Solution:
        """Apply local search to improve the solution."""


def alns_search(
    initial: Solution,
    destroy_ops: Iterable[DestroyOperator],
    repair_ops: Iterable[RepairOperator],
    evaluator: Evaluator,
    iterations: int = 1,
    *,
    manager: Optional["AdaptiveOperatorManager"] = None,
    local_search_ops: Iterable[LocalSearchOperator] = (),
    eta: float = 0.2,
    alpha: float = 0.6,
    rng: Optional[random.Random] = None,
) -> Solution:
    """Run a minimal ALNS scaffold returning the best found solution.

    This placeholder performs no sophisticated acceptance or selection; it
    exists to wire together responsibilities and enable incremental build-out.
    """
    local_search_ops = list(local_search_ops)
    _rng = rng or random.Random(random.getrandbits(32))

    if manager is None:
        from .adaptive import AdaptiveOperatorManager
        manager = AdaptiveOperatorManager(
            destroy_ops, repair_ops, eta=eta, alpha=alpha, rng=_rng,
        )

    current = initial.clone()
    current_cost = evaluator.evaluate_solution(current).total_cost
    best = current.clone()
    best_cost = current_cost

    for _ in range(iterations):
        destroy = manager.select_destroy()
        repair = manager.select_repair()

        start = time.perf_counter()
        destroyed, pool = destroy.apply(current)
        destroy_time = time.perf_counter() - start

        start = time.perf_counter()
        candidate = repair.apply(destroyed, pool.customers)
        repair_time = time.perf_counter() - start

        # Apply local search if available
        if local_search_ops:
            local_search = _rng.choice(local_search_ops)
            start = time.perf_counter()
            candidate = local_search.apply(candidate, [])
            repair_time += time.perf_counter() - start

        candidate_cost = evaluator.evaluate_solution(candidate).total_cost
        reward = "rejected"
        delta_improvement = 0.0

        accepted = False
        if candidate_cost <= current_cost + 1e-9:
            accepted = True
            delta_improvement = max(
                0.0, (current_cost - candidate_cost) / (abs(current_cost) + 1e-9))
            current = candidate
            current_cost = candidate_cost
            if candidate_cost < best_cost - 1e-9:
                best = candidate.clone()
                best_cost = candidate_cost
                reward = "global"
            elif delta_improvement > 0:
                reward = "better"
        else:
            reward = "rejected"

        manager.update(destroy, repair, reward,
                       delta_improvement, destroy_time, repair_time)

    return best
