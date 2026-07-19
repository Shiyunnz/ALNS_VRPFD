"""Time window generation utilities for various operating regimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

__all__ = [
    "TimeWindowGenerator",
    "TimeWindowConfig",
]


@dataclass(frozen=True)
class TimeWindowConfig:
    """Configuration parameters controlling time window generation."""

    operation_horizon: float = 8.0
    min_window_width: float = 0.33
    max_window_width: float = 2.0
    service_time: float = 0.17
    shift_factor: float = 0.5
    road_condition_factor: float = 1.3
    priority_levels: int = 3
    # Extra slack added to latest time to avoid overly tight windows
    latest_time_slack: float = 1.5
    # Seed used by class_based Wang/Holguin supply-class deadline generation.
    class_seed: int = 42


class TimeWindowGenerator:
    """Generate time windows according to the requested strategy."""

    @classmethod
    def generate(
        cls,
        strategy: str,
        customer_demands: Mapping[int, float],
        depot_start: int,
        depot_end: int | None,
        config: TimeWindowConfig | None = None,
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Dispatch the requested time window strategy."""
        config = config or TimeWindowConfig()
        strategy = strategy.lower()
        if strategy == "demand_based":
            return cls._generate_demand_based(customer_demands, depot_start, depot_end, config)
        if strategy == "emergency":
            return cls._generate_emergency(customer_demands, depot_start, depot_end, config)
        if strategy == "early_shift":
            return cls._generate_early_shift(customer_demands, depot_start, depot_end, config)
        raise ValueError(f"Unsupported time window strategy: {strategy}")

    @staticmethod
    def _seed_depots(
        depot_start: int,
        depot_end: int | None,
        horizon: float,
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        optimal = {depot_start: 0.0}
        latest = {depot_start: 0.0}
        if depot_end is not None:
            optimal[depot_end] = horizon
            latest[depot_end] = horizon
        return optimal, latest

    @classmethod
    def _generate_demand_based(
        cls,
        customer_demands: Mapping[int, float],
        depot_start: int,
        depot_end: int | None,
        config: TimeWindowConfig,
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        optimal, latest = cls._seed_depots(
            depot_start, depot_end, config.operation_horizon)
        if not customer_demands:
            return optimal, latest

        min_demand = min(customer_demands.values())
        max_demand = max(customer_demands.values())
        demand_range = max(max_demand - min_demand, 1e-6)

        sorted_customers = sorted(
            customer_demands, key=customer_demands.get, reverse=True)
        current_time_slot = 1.0

        for index, customer_id in enumerate(sorted_customers):
            demand = customer_demands[customer_id]
            demand_ratio = (demand - min_demand) / demand_range
            window_width = config.max_window_width - demand_ratio * (
                config.max_window_width - config.min_window_width
            )
            window_width = max(config.min_window_width, min(
                config.max_window_width, window_width))

            base_time = max(0.5, current_time_slot)
            optimal_start = base_time + index * 0.25

            latest_possible_start = (
                config.operation_horizon
                - window_width
                - config.service_time
                - 0.5
            )
            optimal_start = min(optimal_start, latest_possible_start)
            optimal_start = max(0.5, optimal_start)

            optimal[customer_id] = optimal_start
            # Add slack to latest time to avoid overly tight windows
            latest[customer_id] = optimal_start + \
                window_width + config.latest_time_slack
            current_time_slot = optimal_start + 0.33

        return optimal, latest

    @classmethod
    def _generate_early_shift(
        cls,
        customer_demands: Mapping[int, float],
        depot_start: int,
        depot_end: int | None,
        config: TimeWindowConfig,
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        base_optimal, base_latest = cls._generate_demand_based(
            customer_demands,
            depot_start,
            depot_end,
            config,
        )
        optimal, latest = cls._seed_depots(
            depot_start, depot_end, config.operation_horizon)

        for customer_id, base_start in base_optimal.items():
            if customer_id in optimal:
                continue
            base_end = base_latest[customer_id]
            window_width = base_end - base_start
            new_start = max(0.0, base_start - config.shift_factor)
            new_end = new_start + window_width
            if new_end > config.operation_horizon:
                new_end = config.operation_horizon
                new_start = max(0.0, new_end - window_width)
            optimal[customer_id] = new_start
            # Add slack to latest time
            latest[customer_id] = new_end + config.latest_time_slack

        return optimal, latest

    @classmethod
    def _generate_emergency(
        cls,
        customer_demands: Mapping[int, float],
        depot_start: int,
        depot_end: int | None,
        config: TimeWindowConfig,
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        optimal, latest = cls._seed_depots(
            depot_start, depot_end, config.operation_horizon)
        if not customer_demands:
            return optimal, latest

        min_demand = min(customer_demands.values())
        max_demand = max(customer_demands.values())
        demand_range = max(max_demand - min_demand, 1e-6)

        sorted_customers = sorted(
            customer_demands, key=customer_demands.get, reverse=True)
        current_time_slot = 0.5

        for index, customer_id in enumerate(sorted_customers):
            demand = customer_demands[customer_id]
            demand_ratio = (demand - min_demand) / demand_range
            priority_level = max(
                1,
                min(config.priority_levels, int(
                    config.priority_levels * (1 - demand_ratio)) + 1),
            )

            window_width = config.max_window_width - (priority_level - 1) * 0.3
            window_width = max(config.min_window_width, min(
                config.max_window_width, window_width))
            adjusted_width = min(
                window_width * config.road_condition_factor,
                config.operation_horizon * 0.8,
            )

            base_time = max(0.25, current_time_slot)
            optimal_start = base_time + index * 0.2
            latest_possible_start = (
                config.operation_horizon
                - adjusted_width
                - config.service_time
                - 0.5
            )
            optimal_start = min(optimal_start, latest_possible_start)
            optimal_start = max(0.25, optimal_start)

            optimal[customer_id] = optimal_start
            # Add slack to latest time
            latest[customer_id] = optimal_start + \
                adjusted_width + config.latest_time_slack
            current_time_slot = optimal_start + 0.25

        return optimal, latest
