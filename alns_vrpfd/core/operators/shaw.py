"""Shaw removal operator focusing on related customers."""

from __future__ import annotations

from typing import List, Mapping, Tuple

from .base import CustomerAssignment, DestroyOperator, _segment_distance


class DestroyShaw(DestroyOperator):
    """Remove clusters of related customers using Shaw's heuristic."""

    def __init__(
        self,
        instance,
        *,
        anchor_strategy: str = "drop_tasks",
        rng=None,
        weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        super().__init__(instance, anchor_strategy=anchor_strategy, rng=rng)
        self._weights = weights

    def _select_customers(
        self,
        assignments: Mapping[int, CustomerAssignment],
        count: int,
    ) -> List[int]:
        if count <= 0:
            return []
        pool = list(assignments.keys())
        if count >= len(pool):
            return pool

        seed = self._rng.choice(pool)
        selected = [seed]
        remaining = set(pool)
        remaining.remove(seed)

        while remaining and len(selected) < count:
            best_candidate = None
            best_score = None
            for candidate in remaining:
                similarity = min(
                    self._relatedness(assignments[candidate], assignments[chosen])
                    for chosen in selected
                )
                if best_score is None or similarity < best_score:
                    best_score = similarity
                    best_candidate = candidate
            selected.append(best_candidate)
            remaining.remove(best_candidate)

        return selected

    def _relatedness(
        self,
        a: CustomerAssignment,
        b: CustomerAssignment,
    ) -> float:
        w_distance, w_demand, w_group = self._weights
        dist = _segment_distance(self._truck_distance, self._node_index, a.customer_id, b.customer_id)
        if dist == float("inf"):
            dist = 0.0
        demand_diff = abs(a.demand - b.demand)
        same_group = 0.0
        if a.kind == b.kind:
            if a.kind == "truck" and a.route is b.route:
                same_group = 0.0
            elif a.kind == "drone" and a.task is b.task:
                same_group = 0.0
            else:
                same_group = 1.0
        else:
            same_group = 1.0
        return w_distance * dist + w_demand * demand_diff + w_group * same_group
