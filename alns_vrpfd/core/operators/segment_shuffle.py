"""Destroy operator that removes contiguous truck segments to enable route reshuffling."""

from __future__ import annotations

from typing import List

from .base import DestroyOperator, UnassignedPool


class DestroySegmentShuffle(DestroyOperator):
    """Remove contiguous segments from truck routes for stronger diversification."""

    def __init__(
        self,
        instance,
        *,
        min_segment: int = 2,
        max_segment: int = 6,
        route_fraction: float = 0.4,
        anchor_strategy: str = "drop_tasks",
        rng=None,
    ) -> None:
        super().__init__(instance, anchor_strategy=anchor_strategy, rng=rng)
        self._min_segment = max(1, min_segment)
        self._max_segment = max(self._min_segment, max_segment)
        self._route_fraction = max(0.1, min(route_fraction, 1.0))

    def apply(self, solution, count):
        assignments = self._collect_assignments(solution)
        if not assignments:
            return solution.clone(), UnassignedPool()

        mutated = solution.clone()
        pool = UnassignedPool()

        total_customers = sum(len(route.customers())
                              for route in mutated.truck_routes)
        if total_customers == 0:
            return mutated, pool

        target = self._target_count(len(assignments), total_customers, count)
        targets = self._select_segments(mutated, target)
        if not targets:
            # Fallback to base behaviour if segmentation failed
            return super().apply(solution, count)

        for customer_id in targets:
            self._remove_customer(mutated, customer_id, pool)

        for route in mutated.truck_routes:
            self._recalculate_truck_load(route)

        return mutated, pool

    def _target_count(self, n_assignments: int, n_customers: int, count: int) -> int:
        desired = max(count, int(round(self._route_fraction * n_customers)))
        desired = max(desired, self._min_segment)
        return min(n_assignments, desired)

    def _select_segments(self, solution, target: int) -> List[int]:
        remaining = target
        selected: List[int] = []
        routes = [route for route in solution.truck_routes
                  if len(route.customers()) >= self._min_segment]
        if not routes:
            return []

        guard = 0
        max_attempts = len(routes) * 4
        while remaining > 0 and routes and guard < max_attempts:
            guard += 1
            route = self._rng.choice(routes)
            customers = route.customers()
            if len(customers) < self._min_segment:
                routes.remove(route)
                continue

            seg_len = min(self._max_segment, len(customers))
            if seg_len <= 0:
                break
            seg_len = self._rng.randint(
                self._min_segment, max(self._min_segment, seg_len))
            if seg_len > len(customers):
                seg_len = len(customers)
            start_limit = len(customers) - seg_len
            if start_limit < 0:
                continue
            start_idx = self._rng.randint(0, start_limit)
            segment = customers[start_idx:start_idx + seg_len]

            for cid in segment:
                if cid not in selected:
                    selected.append(cid)
                    remaining -= 1
                    if remaining <= 0:
                        break

        return selected

    def _select_customers(self, assignments, count):
        # Unused because apply() is overridden; method provided to satisfy the ABC.
        return []
