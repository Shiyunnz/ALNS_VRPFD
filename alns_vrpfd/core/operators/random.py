"""Random destroy operator."""

from __future__ import annotations

from typing import List, Mapping

from .base import CustomerAssignment, DestroyOperator


class DestroyRandom(DestroyOperator):
    """Uniformly remove customers at random."""

    def _select_customers(
        self,
        assignments: Mapping[int, CustomerAssignment],
        count: int,
    ) -> List[int]:
        if count <= 0:
            return []
        population = list(assignments.keys())
        if count >= len(population):
            return population
        return self._rng.sample(population, count)


class DestroyLargeRandom(DestroyRandom):
    """Compatibility variant for historical operator sets.

    The legacy code referenced a separate "large random" destroy operator.
    The current implementation keeps equivalent behavior through the removal
    count requested by the ALNS controller, so this subclass intentionally
    reuses `DestroyRandom`.
    """
