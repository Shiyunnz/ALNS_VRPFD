"""Dense distance matrices for truck and drone with unified access."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

__all__ = ["DistanceStore"]


class DistanceStore:
    """Maintain dense distance matrices for truck and drone.

    The store keeps an ordered list of node ids and 2 NxN matrices (truck/drone).
    It grows dynamically when new nodes are introduced via `add_distance` or
    when `set_nodes` is called. Unspecified entries are initialized to +inf,
    diagonal entries to 0.0.
    """

    def __init__(self) -> None:
        self._node_ids: List[int] = []
        self._index_of: Dict[int, int] = {}
        self._truck: List[List[float]] = []
        self._drone: List[List[float]] = []

    # ----- Node handling -------------------------------------------------
    def nodes(self) -> Tuple[int, ...]:
        """Return the current ordered node ids."""
        return tuple(self._node_ids)

    def set_nodes(self, node_ids: Sequence[int]) -> None:
        """Reset matrices to cover exactly the provided nodes.

        Existing distances for overlapping nodes are preserved; new nodes are
        initialized; removed nodes are dropped.
        """
        new_order = list(dict.fromkeys(node_ids))  # keep unique, preserve order
        old_index = {nid: idx for idx, nid in enumerate(self._node_ids)}

        n = len(new_order)
        inf = float("inf")

        def blank() -> List[List[float]]:
            m = [[inf] * n for _ in range(n)]
            for i in range(n):
                m[i][i] = 0.0
            return m

        new_truck = blank()
        new_drone = blank()

        # Copy overlapping entries
        for to_new_idx, nid_i in enumerate(new_order):
            i_old = old_index.get(nid_i)
            for to_new_jdx, nid_j in enumerate(new_order):
                j_old = old_index.get(nid_j)
                if i_old is not None and j_old is not None:
                    if self._truck:
                        new_truck[to_new_idx][to_new_jdx] = self._truck[i_old][j_old]
                    if self._drone:
                        new_drone[to_new_idx][to_new_jdx] = self._drone[i_old][j_old]

        self._node_ids = new_order
        self._index_of = {nid: idx for idx, nid in enumerate(self._node_ids)}
        self._truck = new_truck
        self._drone = new_drone

    def _ensure_nodes_for(self, origins: Iterable[int], destinations: Iterable[int]) -> None:
        """Ensure matrices cover provided origins/destinations."""
        needed = set(self._node_ids)
        needed.update(origins)
        needed.update(destinations)
        if list(needed) != self._node_ids:
            # Keep stable ascending order by default
            self.set_nodes(sorted(needed))

    # ----- Updates -------------------------------------------------------
    def add_distance(self, vehicle_type: str, origin: int, destination: int, distance: float) -> None:
        """Add or update a single distance entry for the vehicle type."""
        self._ensure_nodes_for([origin], [destination])
        i = self._index_of[origin]
        j = self._index_of[destination]
        matrix = self._select(vehicle_type)
        matrix[i][j] = float(distance)

    def bulk_update(self, vehicle_type: str, entries: Mapping[Tuple[int, int], float]) -> None:
        """Bulk set distances from a mapping of (origin, destination) -> distance."""
        if not entries:
            return
        origins = (i for i, _ in entries.keys())
        dests = (j for _, j in entries.keys())
        self._ensure_nodes_for(origins, dests)
        matrix = self._select(vehicle_type)
        for (origin, dest), val in entries.items():
            i = self._index_of[origin]
            j = self._index_of[dest]
            matrix[i][j] = float(val)

    # ----- Queries -------------------------------------------------------
    def get(self, vehicle_type: str, origin: int, destination: int) -> float:
        """Return a single distance value; +inf if unknown."""
        if origin not in self._index_of or destination not in self._index_of:
            return float("inf")
        i = self._index_of[origin]
        j = self._index_of[destination]
        return self._select(vehicle_type)[i][j]

    def matrix(self, vehicle_type: str) -> List[List[float]]:
        """Return the dense matrix (aligned with `nodes()`)."""
        return self._select(vehicle_type)

    def time_matrix(self, vehicle_type: str, speed: float) -> List[List[float]]:
        """Return a dense time matrix computed from distances and speed."""
        dist = self._select(vehicle_type)
        inf = float("inf")
        n = len(dist)
        return [[(d / speed) if d != inf else inf for d in row] for row in dist]

    # ----- Internals -----------------------------------------------------
    def _select(self, vehicle_type: str) -> List[List[float]]:
        key = vehicle_type.lower()
        if key == "truck":
            return self._truck
        if key == "drone":
            return self._drone
        raise ValueError(f"Unknown vehicle type: {vehicle_type}")

