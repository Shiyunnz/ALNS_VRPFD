"""Robustness checks for drone energy under budgeted uncertainty."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence, Tuple, TYPE_CHECKING

from alns_vrpfd.evaluation.energy import DroneEnergyModel
from alns_vrpfd.model.solution import Solution

if TYPE_CHECKING:  # pragma: no cover - only for static type checking
    from alns_vrpfd.model.route import DroneTask, DroneTaskContext, DroneTaskTiming

__all__ = [
    "DroneEnergyAssessment",
    "RobustnessChecker",
    "RobustnessResult",
    "assess_drone_task_energy",
]


@dataclass(frozen=True)
class DroneEnergyAssessment:
    """Per-drone-task energy summary under budgeted uncertainty."""

    drone_id: int
    task_id: int | None
    nominal_energy: float
    worst_case_energy: float
    deviation_rate: float
    uncertainty_budget: float
    capacity: float | None
    margin: float
    feasible: bool
    segment_energies: Tuple[float, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RobustnessResult:
    """Container for robustness evaluation outcomes."""

    feasible: bool
    margin: float
    task_breakdown: Tuple[DroneEnergyAssessment, ...] = field(
        default_factory=tuple)


class RobustnessChecker:
    """Evaluate drone tasks against budgeted energy uncertainty."""

    def __init__(
        self,
        *,
        energy_model: DroneEnergyModel | None = None,
        battery_capacity: float | Mapping[int, float] | None = None,
        energy_uncertainty_budget: float | Mapping[int, float] = 0.0,
        energy_deviation_rate: float = 0.1,
        tolerance: float = 1e-6,
    ) -> None:
        self._energy_model = energy_model
        self._battery_capacity = battery_capacity
        self._uncertainty_budget = energy_uncertainty_budget
        self._deviation_rate = energy_deviation_rate
        self._tolerance = tolerance

    def check(
        self,
        solution: Solution,
        *,
        contexts: Mapping[int, "DroneTaskContext"]
        | Sequence["DroneTaskContext"]
        | None = None,
    ) -> RobustnessResult:
        """Return robustness summary across all drone tasks."""

        energy_details: list[DroneEnergyAssessment] = []
        feasible = True
        min_margin = float("inf")

        for index, task in enumerate(solution.drone_tasks):
            context = _select_context(contexts, index, task)
            timing = getattr(context, "timing", None) if context else None
            if timing is None:
                # Without timing data we cannot evaluate energy for this task.
                continue

            energy_model = getattr(
                context, "energy_model", None) if context else None
            if energy_model is None:
                energy_model = self._energy_model
            if energy_model is None:
                continue

            deviation_rate = getattr(
                context, "energy_deviation_rate", None) if context else None
            if deviation_rate is None or deviation_rate < 0:
                deviation_rate = self._deviation_rate

            raw_budget = _resolve_per_drone(
                getattr(context, "energy_uncertainty_budget",
                        None) if context else None,
                task.drone_id,
            )
            if raw_budget is None:
                raw_budget = _resolve_per_drone(
                    self._uncertainty_budget, task.drone_id)
            budget = _normalise_gamma_budget(raw_budget)

            capacity = _resolve_per_drone(
                getattr(context, "drone_energy_capacity",
                        None) if context else None,
                task.drone_id,
            )
            if capacity is None:
                capacity = _resolve_per_drone(
                    self._battery_capacity, task.drone_id)

            time_tolerance = getattr(
                context, "time_tolerance", None) if context else None
            energy_tolerance = getattr(
                context, "energy_tolerance", None) if context else None

            assessment = assess_drone_task_energy(
                task=task,
                timing=timing,
                energy_model=energy_model,
                deviation_rate=deviation_rate,
                uncertainty_budget=budget,
                capacity=capacity,
                tolerance=energy_tolerance if energy_tolerance is not None else self._tolerance,
                time_tolerance=time_tolerance,
            )

            energy_details.append(assessment)
            feasible = feasible and assessment.feasible
            min_margin = min(min_margin, assessment.margin)

        if not energy_details:
            # No energy-based assessment was possible.
            return RobustnessResult(feasible=True, margin=0.0, task_breakdown=tuple())

        return RobustnessResult(
            feasible=feasible,
            margin=min_margin,
            task_breakdown=tuple(energy_details),
        )


def assess_drone_task_energy(
    *,
    task: "DroneTask",
    timing: "DroneTaskTiming",
    energy_model: DroneEnergyModel,
    deviation_rate: float,
    uncertainty_budget: float,
    capacity: float | None,
    tolerance: float = 1e-6,
    time_tolerance: float | None = None,
) -> DroneEnergyAssessment | None:
    """Return worst-case energy assessment for a single drone task.

    The uncertainty model follows the budgeted uncertainty set of Bertsimas & Sim,
    where each flight segment energy can deviate by ``deviation_rate`` of its
    nominal value and at most ``uncertainty_budget`` segments realise the full
    deviation. Fractional budgets scale the next deviation accordingly.

    Returns None if timing and task are inconsistent.
    """

    if deviation_rate < 0:
        raise ValueError("Deviation rate must be non-negative.")
    if uncertainty_budget < 0:
        raise ValueError("Uncertainty budget must be non-negative.")

    segment_energies = _compute_segment_energies(
        task=task,
        timing=timing,
        energy_model=energy_model,
        time_tolerance=time_tolerance,
    )

    # If segment energies couldn't be computed due to inconsistent timing/task
    if segment_energies is None:
        return DroneEnergyAssessment(
            drone_id=task.drone_id,
            task_id=getattr(task, "task_id", None),
            nominal_energy=float("inf"),
            worst_case_energy=float("inf"),
            deviation_rate=deviation_rate,
            uncertainty_budget=uncertainty_budget,
            capacity=capacity,
            margin=float("-inf"),
            feasible=False,
            segment_energies=tuple(),
        )

    nominal_total, worst_case_energy = _gamma_layer_energy_total(
        segment_energies,
        deviation_rate=deviation_rate,
        uncertainty_budget=uncertainty_budget,
    )

    if capacity is None:
        margin = float("inf")
        feasible = True
    else:
        margin = capacity - worst_case_energy
        feasible = margin >= -tolerance

    return DroneEnergyAssessment(
        drone_id=task.drone_id,
        task_id=getattr(task, "task_id", None),
        nominal_energy=nominal_total,
        worst_case_energy=worst_case_energy,
        deviation_rate=deviation_rate,
        uncertainty_budget=uncertainty_budget,
        capacity=capacity,
        margin=margin,
        feasible=feasible,
        segment_energies=segment_energies,
    )


def _compute_segment_energies(
    *,
    task: "DroneTask",
    timing: "DroneTaskTiming",
    energy_model: DroneEnergyModel,
    time_tolerance: float | None,
) -> Tuple[float, ...] | None:
    """Compute energy for each segment of a drone task.

    Returns None if timing and task are inconsistent.
    """
    durations = _segment_durations(task, timing, time_tolerance)
    if durations is None:
        # Timing and task are inconsistent
        return None
    if len(durations) != len(task.payloads):
        # Payload sequence length mismatch
        return None

    energies = [
        energy_model.energy_kwh(payload, duration)
        for payload, duration in zip(task.payloads, durations)
    ]
    return tuple(energies)


def _segment_durations(
    task: "DroneTask",
    timing: "DroneTaskTiming",
    time_tolerance: float | None,
) -> Tuple[float, ...] | None:
    """Calculate segment durations for a drone task.

    Returns None if timing and task are inconsistent (e.g., after solution modification).
    """
    tolerance = time_tolerance if time_tolerance is not None else 0.0
    durations: list[float] = []

    previous_time = timing.launch_time
    for customer in task.customers():
        if customer not in timing.customer_arrival_times:
            # Timing and task are inconsistent - return None to indicate infeasibility
            return None
        arrival_time = timing.customer_arrival_times[customer]
        delta = arrival_time - previous_time
        if delta < -tolerance:
            # Negative duration indicates inconsistent timing
            return None
        durations.append(max(delta, 0.0))
        previous_time = arrival_time

    final_delta = timing.retrieve_time - previous_time
    if final_delta < -tolerance:
        # Negative duration indicates inconsistent timing
        return None
    durations.append(max(final_delta, 0.0))

    return tuple(durations)


def _gamma_layer_energy_total(
    segment_energies: Sequence[float],
    *,
    deviation_rate: float,
    uncertainty_budget: float,
) -> Tuple[float, float]:
    """Replicate MILP-style gamma layering to accumulate worst-case energy."""

    nominal_total = float(sum(segment_energies))
    if not segment_energies or deviation_rate <= 0 or uncertainty_budget <= 0:
        return nominal_total, nominal_total
    if uncertainty_budget < 0:
        raise ValueError("Uncertainty budget must be non-negative.")

    deviations = [energy * deviation_rate for energy in segment_energies]
    gamma_budget = int(math.floor(uncertainty_budget))
    fractional_part = float(uncertainty_budget - gamma_budget)
    if fractional_part > 1e-9:
        raise ValueError("Energy uncertainty budget must be an integer.")
    gamma_budget = max(0, gamma_budget)

    states = [0.0] * (gamma_budget + 1)
    for energy, deviation in zip(segment_energies, deviations):
        next_states = [0.0] * (gamma_budget + 1)
        for gamma in range(gamma_budget + 1):
            worst_value = states[gamma] + energy
            if gamma > 0:
                worst_value = max(
                    worst_value,
                    states[gamma - 1] + energy + deviation,
                )
            next_states[gamma] = worst_value
        states = next_states

    worst_case = states[gamma_budget]
    # fractional budgets disallowed; no extra term

    return nominal_total, worst_case


def _resolve_per_drone(
    value: float | Mapping[int, float] | None,
    drone_id: int,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value.get(drone_id)
    return value


def _select_context(
    contexts: Mapping[int, "DroneTaskContext"] | Sequence["DroneTaskContext"] | None,
    index: int,
    task: "DroneTask",
) -> "DroneTaskContext" | None:
    if contexts is None:
        return None

    if isinstance(contexts, Mapping):
        if getattr(task, "task_id", None) is not None and task.task_id in contexts:
            return contexts[task.task_id]
        if index in contexts:
            return contexts[index]
        return contexts.get(task.drone_id)

    if index < len(contexts):
        return contexts[index]

    return None


def _normalise_gamma_budget(value: float | None) -> float:
    if value is None:
        return 0.0
    if value < 0:
        raise ValueError("Energy uncertainty budget must be non-negative.")
    integer_part = math.floor(value)
    if abs(value - integer_part) > 1e-9:
        raise ValueError(
            "Energy uncertainty budget must be an integer representing arc count."
        )
    return float(integer_part)
