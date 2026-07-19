"""Customer management utilities and time-window application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

from .time_windows import TimeWindowConfig, TimeWindowGenerator

__all__ = ["Customer", "CustomerManager"]


@dataclass
class Customer:
    """Basic customer attributes used during instance configuration."""

    customer_id: int
    demand: float = 0.0
    location_x: float = 0.0
    location_y: float = 0.0
    optimal_time: float | None = None
    latest_time: float | None = None
    supply_class: str | None = None


class CustomerManager:
    """Maintain customer data and derive time windows."""

    def __init__(self) -> None:
        self._depot_start: int | None = None
        self._depot_end: int | None = None
        self._customers: Dict[int, Customer] = {}

    @property
    def depot_start(self) -> int | None:
        """Return the id of the start depot."""
        return self._depot_start

    @property
    def depot_end(self) -> int | None:
        """Return the id of the end depot."""
        return self._depot_end

    def set_depots(self, start: int, end: int | None = None) -> None:
        """Assign depot identifiers for the instance."""
        self._depot_start = start
        self._depot_end = end

    def register_customer(
        self,
        customer_id: int,
        demand: float = 0.0,
        location_x: float = 0.0,
        location_y: float = 0.0,
    ) -> None:
        """Register a customer with basic demand and position data."""
        if customer_id in (self._depot_start, self._depot_end):
            raise ValueError("Use set_depots for depot configuration.")
        self._customers[customer_id] = Customer(
            customer_id=customer_id,
            demand=demand,
            location_x=location_x,
            location_y=location_y,
        )

    def customer_ids(self) -> Tuple[int, ...]:
        """Return registered customer ids in ascending order."""
        return tuple(sorted(self._customers.keys()))

    def demands(self) -> Dict[int, float]:
        """Return a mapping of customer demand values."""
        return {cid: customer.demand for cid, customer in self._customers.items()}

    def assign_time_window(self, customer_id: int, optimal: float, latest: float) -> None:
        """Assign a precomputed time window to the customer."""
        customer = self._customers.get(customer_id)
        if customer is None:
            raise KeyError(f"Customer {customer_id} is not registered.")
        customer.optimal_time = optimal
        customer.latest_time = latest

    def assign_supply_class(self, customer_id: int, supply_class: str) -> None:
        """Assign a relief supply class to the customer."""
        customer = self._customers.get(customer_id)
        if customer is None:
            raise KeyError(f"Customer {customer_id} is not registered.")
        customer.supply_class = supply_class

    def generate_time_windows(
        self,
        strategy: str,
        config: TimeWindowConfig | None = None,
    ) -> None:
        """Generate time windows and apply them to all registered customers."""
        if self._depot_start is None:
            raise ValueError("Depot start must be configured before generating time windows.")

        optimal_times, latest_times = TimeWindowGenerator.generate(
            strategy=strategy,
            customer_demands=self.demands(),
            depot_start=self._depot_start,
            depot_end=self._depot_end,
            config=config,
        )

        for customer_id, optimal in optimal_times.items():
            if customer_id in (self._depot_start, self._depot_end):
                continue
            latest = latest_times[customer_id]
            if customer_id not in self._customers:
                continue
            self.assign_time_window(customer_id, optimal, latest)

    def time_window(self, customer_id: int) -> Tuple[float | None, float | None]:
        """Return the assigned time window for the requested customer."""
        customer = self._customers.get(customer_id)
        if customer is None:
            raise KeyError(f"Customer {customer_id} is not registered.")
        return customer.optimal_time, customer.latest_time

    def supply_class(self, customer_id: int) -> str | None:
        """Return the relief supply class for the requested customer."""
        customer = self._customers.get(customer_id)
        if customer is None:
            raise KeyError(f"Customer {customer_id} is not registered.")
        return customer.supply_class

    def customers(self) -> Iterable[Customer]:
        """Yield customer records in ascending id order."""
        for customer_id in self.customer_ids():
            yield self._customers[customer_id]
