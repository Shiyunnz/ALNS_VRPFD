from __future__ import annotations

from typing import List, Mapping

from .base import CustomerAssignment, DestroyOperator


class DestroyRouteRemoval(DestroyOperator):
    def _select_customers(
        self,
        assignments: Mapping[int, CustomerAssignment],
        count: int,
    ) -> List[int]:
        if count <= 0:
            return []
        route_customers: List[List[int]] = []
        for a in assignments.values():
            if a.kind == "truck" and a.route is not None:
                custs = a.route.customers()
                if custs:
                    route_customers.append(custs)
        if not route_customers:
            return []
        target = self._rng.choice(route_customers)
        return target
