"""Parse legacy instance files into the refactored data structures."""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Sequence, Tuple

from alns_vrpfd.instance import InstanceManager, TimeWindowConfig

__all__ = ["InstanceDataReader"]


class InstanceDataReader:
    """Read RVRPFD text instances into an InstanceManager."""

    def __init__(
        self,
        *,
        time_window_strategy: str = "class_based",
        time_window_config: TimeWindowConfig | None = None,
        apply_time_windows: bool = True,
    ) -> None:
        self._strategy = time_window_strategy
        self._config = time_window_config
        self._apply_time_windows_default = apply_time_windows
        self._instance: InstanceManager | None = None

    def read_instance(
        self,
        file_path: str,
        *,
        apply_time_windows: bool | None = None,
    ) -> InstanceManager:
        """Parse the provided instance file and return an InstanceManager."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Instance file not found: {file_path}")

        instance = InstanceManager()

        with open(file_path, "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]

        idx = 0
        while idx < len(lines):
            token = lines[idx]
            if token == "VEHICLE INFORMATION":
                idx = self._read_vehicle_info(lines, idx + 1, instance)
            elif token == "CUSTOMER INFORMATION":
                idx = self._read_customer_info(lines, idx + 1, instance)
            elif token == "Distance For Drone":
                idx = self._read_distances(lines, idx + 1, instance, vehicle_type="drone")
            elif token == "Distance For Truck":
                idx = self._read_distances(lines, idx + 1, instance, vehicle_type="truck")
            else:
                idx += 1

        do_apply = self._apply_time_windows_default if apply_time_windows is None else apply_time_windows
        if do_apply and instance.customer_manager.customer_ids():
            instance.generate_time_windows(strategy=self._strategy, config=self._config)

        self._instance = instance
        return instance

    def distance_matrix(self, vehicle_type: str) -> List[List[float]]:
        """Return the dense distance matrix from the last read instance."""
        if self._instance is None:
            raise RuntimeError("No instance has been read yet.")
        return self._instance.distance_matrix(vehicle_type)

    def time_matrix(self, vehicle_type: str) -> List[List[float]]:
        """Return the dense travel-time matrix for the last read instance."""
        if self._instance is None:
            raise RuntimeError("No instance has been read yet.")
        return self._instance.time_matrix(vehicle_type)

    # ------------------------------------------------------------------
    def _read_vehicle_info(
        self,
        lines: Sequence[str],
        start_idx: int,
        instance: InstanceManager,
    ) -> int:
        """Parse vehicle specifications starting from the given index."""
        idx = start_idx
        if idx < len(lines) and "Type" in lines[idx]:
            idx += 1

        while idx < len(lines):
            line = lines[idx]
            if self._is_section_header(line):
                break
            if line.startswith("//"):
                idx += 1
                continue

            parts = self._split_fields(line)
            if len(parts) >= 6:
                vehicle_type = parts[0]
                instance.register_vehicle_type(
                    vehicle_type,
                    number=int(parts[1]),
                    capacity=float(parts[2]),
                    endurance=float(parts[3]),
                    speed=float(parts[4]),
                    unit_cost=float(parts[5]),
                )
            idx += 1

        return idx

    def _read_customer_info(
        self,
        lines: Sequence[str],
        start_idx: int,
        instance: InstanceManager,
    ) -> int:
        """Parse customer, depot, and demand information."""
        idx = start_idx
        if idx < len(lines) and lines[idx].lower().startswith("id"):
            idx += 1

        records: List[Tuple[int, float, float, float]] = []
        while idx < len(lines):
            line = lines[idx]
            if self._is_section_header(line):
                break
            if line.startswith("//"):
                idx += 1
                continue
            parts = self._split_fields(line)
            if len(parts) >= 4:
                customer_id = int(parts[0])
                x_coord = float(parts[1])
                y_coord = float(parts[2])
                demand = float(parts[3])
                records.append((customer_id, x_coord, y_coord, demand))
            idx += 1

        if not records:
            return idx

        customer_ids = [rec[0] for rec in records]
        depot_start = min(customer_ids)
        depot_end = max(customer_ids) if len(customer_ids) > 1 else None
        if depot_end == depot_start:
            depot_end = None

        instance.configure_depots(start=depot_start, end=depot_end)

        for customer_id, x_coord, y_coord, demand in records:
            if depot_end is not None and customer_id in (depot_start, depot_end):
                continue
            if depot_end is None and customer_id == depot_start:
                continue
            instance.register_customer(
                customer_id=customer_id,
                demand=demand,
                location_x=x_coord,
                location_y=y_coord,
            )

        instance.distances.set_nodes(instance.all_node_ids())
        return idx

    def _read_distances(
        self,
        lines: Sequence[str],
        start_idx: int,
        instance: InstanceManager,
        *,
        vehicle_type: str,
    ) -> int:
        """Parse distance entries for the specified vehicle type."""
        idx = start_idx
        while idx < len(lines):
            line = lines[idx]
            if self._is_section_header(line):
                break
            if line.startswith("//"):
                idx += 1
                continue

            parts = self._split_fields(line)
            if len(parts) >= 3:
                origin = int(parts[0])
                destination = int(parts[1])
                distance = float(parts[2])
                instance.add_distance(vehicle_type, origin, destination, distance)
            idx += 1

        return idx

    @staticmethod
    def _split_fields(line: str) -> List[str]:
        """Split a line on whitespace and tabs into meaningful fields."""
        return line.replace("\t", " ").split()

    @staticmethod
    def _is_section_header(line: str) -> bool:
        """Return True when the line marks the beginning of a new section."""
        headers = {
            "VEHICLE INFORMATION",
            "CUSTOMER INFORMATION",
            "Distance For Drone",
            "Distance For Truck",
        }
        return line in headers
