"""Instance manager coordinating customers, fleet, and distances."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, Iterable, Mapping, Tuple, List

from alns_vrpfd.deprivation import DEFAULT_SUPPLY_CLASS_SEQUENCE, WANG_SUPPLY_CLASSES

from .customers import CustomerManager
from .time_windows import TimeWindowConfig
from .distances import DistanceStore

__all__ = ["VehicleSpec", "DistanceStore", "InstanceManager", "RobustConfig"]


@dataclass
class VehicleSpec:
    """Describe a single vehicle type participating in the instance."""

    number: int
    capacity: float
    endurance: float
    speed: float
    unit_cost: float


@dataclass
class RobustConfig:
    """Configuration parameters for robust drone evaluations."""

    drone_battery_capacity: float = 6.3
    energy_uncertainty_budget: int = 3
    energy_deviation_rate: float = 0.1
    drone_energy_rate: float = 1.2
    enable_logging: bool = True
    # If True, drones must return to the same truck that launched them
    same_truck_retrieval: bool = False


class InstanceManager:
    """Aggregate all data needed to run the ALNS on a specific instance."""

    def __init__(self) -> None:
        self.customer_manager = CustomerManager()
        self.vehicle_specs: Dict[str, VehicleSpec] = {}
        self.distances = DistanceStore()
        self.robust_config = RobustConfig()

    def configure_depots(self, start: int, end: int | None = None) -> None:
        """Set depot identifiers for the managed instance."""
        self.customer_manager.set_depots(start=start, end=end)

    def register_customer(
        self,
        customer_id: int,
        demand: float = 0.0,
        location_x: float = 0.0,
        location_y: float = 0.0,
    ) -> None:
        """Add a customer record to the instance."""
        self.customer_manager.register_customer(
            customer_id=customer_id,
            demand=demand,
            location_x=location_x,
            location_y=location_y,
        )

    def register_vehicle_type(
        self,
        vehicle_type: str,
        *,
        number: int,
        capacity: float,
        endurance: float,
        speed: float,
        unit_cost: float,
    ) -> None:
        """Store specifications for a vehicle type."""
        self.vehicle_specs[vehicle_type.lower()] = VehicleSpec(
            number=number,
            capacity=capacity,
            endurance=endurance,
            speed=speed,
            unit_cost=unit_cost,
        )

    def add_distance(
        self,
        vehicle_type: str,
        origin: int,
        destination: int,
        distance: float,
    ) -> None:
        """Add distance data for the selected vehicle type."""
        # Ensure the matrices cover current nodes; grow as needed
        self.distances.add_distance(
            vehicle_type, origin, destination, distance)

    def distance_matrix(self, vehicle_type: str) -> List[List[float]]:
        """Return dense distance matrix aligned with `node_ids()` order."""
        # Keep nodes in sync with the instance's nodes
        self.distances.set_nodes(self.all_node_ids())
        return self.distances.matrix(vehicle_type)

    def time_matrix(self, vehicle_type: str) -> List[List[float]]:
        """Return dense time matrix using the registered speed for the vehicle type."""
        # Fix vehicle key case-insensitively
        key = None
        for k in self.vehicle_specs.keys():
            if k.lower() == vehicle_type.lower():
                key = k
                break
        speed = self.vehicle_specs.get(
            key, VehicleSpec(0, 0.0, 0.0, 40.0, 0.0)).speed
        self.distances.set_nodes(self.all_node_ids())
        return self.distances.time_matrix(vehicle_type, speed)

    def all_node_ids(self) -> Tuple[int, ...]:
        """Return all nodes including depots and customers."""
        depot_start = self.customer_manager.depot_start
        depot_end = self.customer_manager.depot_end
        customers = self.customer_manager.customer_ids()
        node_ids = []
        if depot_start is not None:
            node_ids.append(depot_start)
        node_ids.extend(customers)
        if depot_end is not None and depot_end != depot_start:
            node_ids.append(depot_end)
        return tuple(node_ids)

    def generate_time_windows(
        self,
        strategy: str,
        config: TimeWindowConfig | None = None,
    ) -> None:
        """Delegate time window generation to the customer manager."""
        if strategy.lower() == "class_based":
            self._generate_class_based_time_windows(config or TimeWindowConfig())
            return
        self.customer_manager.generate_time_windows(
            strategy=strategy, config=config)

    def _generate_class_based_time_windows(self, config: TimeWindowConfig) -> None:
        """Generate Wang/Holguin supply-class deadlines for all customers."""
        depot_id = self.customer_manager.depot_start
        if depot_id is None:
            raise ValueError("Depot start must be configured before generating time windows.")
        customer_ids = self.customer_manager.customer_ids()
        if not customer_ids:
            return

        node_ids = self.all_node_ids()
        node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
        if depot_id not in node_index:
            raise ValueError(f"Depot {depot_id} is missing from node list.")

        truck_times = self.time_matrix("truck")
        drone_times = self.time_matrix("drone")
        depot_idx = node_index[depot_id]
        rng = random.Random(config.class_seed)
        classes = list(DEFAULT_SUPPLY_CLASS_SEQUENCE)

        for offset, customer_id in enumerate(customer_ids):
            supply_class = classes[offset % len(classes)]
            if offset >= len(classes):
                supply_class = rng.choice(classes)
            spec = WANG_SUPPLY_CLASSES[supply_class]
            customer_idx = node_index[customer_id]
            reachable_time = min(
                truck_times[depot_idx][customer_idx],
                drone_times[depot_idx][customer_idx],
            )
            delta_o = rng.uniform(*spec.deadline_optimal_delta_hours)
            delta_l = rng.uniform(*spec.deadline_latest_delta_hours)
            optimal = reachable_time + delta_o
            latest = optimal + delta_l
            self.customer_manager.assign_supply_class(customer_id, supply_class)
            self.customer_manager.assign_time_window(customer_id, optimal, latest)

    def configure_robustness(
        self,
        *,
        drone_battery_capacity: float | None = None,
        energy_uncertainty_budget: int | None = None,
        energy_deviation_rate: float | None = None,
        drone_energy_rate: float | None = None,
        enable_logging: bool | None = None,
        same_truck_retrieval: bool | None = None,
    ) -> None:
        """Update robust evaluation parameters selectively."""
        if drone_battery_capacity is not None:
            self.robust_config.drone_battery_capacity = drone_battery_capacity
        if energy_uncertainty_budget is not None:
            self.robust_config.energy_uncertainty_budget = energy_uncertainty_budget
        if energy_deviation_rate is not None:
            self.robust_config.energy_deviation_rate = energy_deviation_rate
        if drone_energy_rate is not None:
            self.robust_config.drone_energy_rate = drone_energy_rate
        if enable_logging is not None:
            self.robust_config.enable_logging = enable_logging
        if same_truck_retrieval is not None:
            self.robust_config.same_truck_retrieval = same_truck_retrieval

    def vehicle_types(self) -> Mapping[str, VehicleSpec]:
        """Expose registered vehicle specifications."""
        return dict(self.vehicle_specs)
