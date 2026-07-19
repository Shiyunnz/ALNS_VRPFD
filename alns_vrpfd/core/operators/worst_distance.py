"""Destroy operator removing customers with the largest distance penalty."""

from __future__ import annotations

from typing import List, Mapping

from .base import (
    CustomerAssignment,
    DestroyOperator,
    _segment_distance,
    _segment_energy,
)


class DestroyWorstDistance(DestroyOperator):
    """Remove customers with the highest incremental travel/energy cost."""

    def _select_customers(
        self,
        assignments: Mapping[int, CustomerAssignment],
        count: int,
    ) -> List[int]:
        if count <= 0:
            return []
        scores = []
        for customer_id, assignment in assignments.items():
            if assignment.kind == "truck":
                score = self._truck_metric(assignment)
            else:
                score = self._drone_metric(assignment)
            scores.append((score, customer_id))

        scores.sort(reverse=True)
        chosen = [customer_id for _, customer_id in scores[:count]]
        return chosen

    def _truck_metric(self, assignment: CustomerAssignment) -> float:
        prev_node = assignment.prev_node
        customer = assignment.customer_id
        next_node = assignment.next_node
        dist_prev = _segment_distance(self._truck_distance, self._node_index, prev_node, customer)
        dist_next = _segment_distance(self._truck_distance, self._node_index, customer, next_node)
        bypass = _segment_distance(self._truck_distance, self._node_index, prev_node, next_node)
        if any(value == float("inf") for value in (dist_prev, dist_next, bypass)):
            return float("-inf")
        return (dist_prev + dist_next) - bypass

    def _drone_metric(self, assignment: CustomerAssignment) -> float:
        prev_node = assignment.prev_node
        customer = assignment.customer_id
        next_node = assignment.next_node
        before = assignment.payload_before or 0.0
        after = assignment.payload_after or 0.0
        energy_prev = _segment_energy(
            self._energy_model,
            self._drone_time,
            self._node_index,
            prev_node,
            customer,
            before,
        )
        energy_next = _segment_energy(
            self._energy_model,
            self._drone_time,
            self._node_index,
            customer,
            next_node,
            after,
        )
        if energy_prev == float("inf") or energy_next == float("inf"):
            return float("-inf")
        return energy_prev + energy_next
