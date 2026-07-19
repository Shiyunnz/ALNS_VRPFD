"""Sub-route robust feasibility checks for deterministic pre-check strategies."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence
import math

from alns_vrpfd.evaluation.energy import DroneEnergyModel
from alns_vrpfd.instance.manager import InstanceManager
from alns_vrpfd.model.route import DroneTask
from alns_vrpfd.model.solution import Solution

__all__ = [
    "SubrouteVerificationSummary",
    "SubrouteRobustVerifier",
]


@dataclass(frozen=True)
class SubrouteVerificationSummary:
    """Summary of one incremental sub-route robustness check."""

    changed_truck_routes: int
    changed_drone_tasks: int
    checked_drone_tasks: int
    failed_drone_tasks: int
    feasible: bool


class SubrouteRobustVerifier:
    """Verify robust drone-energy feasibility on changed sub-routes only.

    This checker is lightweight and does not require full evaluator details.
    It uses per-segment drone flight time + payload to compute worst-case energy
    under the Bertsimas-Sim budget.
    """

    def __init__(
        self,
        *,
        instance: InstanceManager,
        drone_energy_capacity: float | Mapping[int, float] | None,
        energy_uncertainty_budget: float | Mapping[int, float],
        energy_deviation_rate: float,
        energy_model: DroneEnergyModel | None = None,
        tolerance: float = 1e-6,
    ) -> None:
        self._instance = instance
        self._drone_energy_capacity = drone_energy_capacity
        self._energy_uncertainty_budget = energy_uncertainty_budget
        self._energy_deviation_rate = float(energy_deviation_rate)
        self._energy_model = energy_model or DroneEnergyModel()
        self._tolerance = tolerance
        self._demands = instance.customer_manager.demands()
        self._drone_time = instance.time_matrix("drone")
        self._node_index = {
            node: idx for idx, node in enumerate(instance.all_node_ids())
        }
        self.last_summary = SubrouteVerificationSummary(
            changed_truck_routes=0,
            changed_drone_tasks=0,
            checked_drone_tasks=0,
            failed_drone_tasks=0,
            feasible=True,
        )

    def verify_candidate(
        self,
        *,
        base: Solution,
        candidate: Solution,
    ) -> bool:
        """Return whether changed sub-routes are robustly feasible."""

        changed_routes = _changed_truck_route_ids(base, candidate)
        changed_task_indices = _changed_drone_task_indices(
            base=base,
            candidate=candidate,
            changed_truck_route_ids=changed_routes,
        )

        if not changed_task_indices:
            self.last_summary = SubrouteVerificationSummary(
                changed_truck_routes=len(changed_routes),
                changed_drone_tasks=0,
                checked_drone_tasks=0,
                failed_drone_tasks=0,
                feasible=True,
            )
            return True

        checked = 0
        failed = 0
        for task_index in sorted(changed_task_indices):
            task = candidate.drone_tasks[task_index]
            checked += 1
            if not self._task_is_robust_feasible(task):
                failed += 1

        feasible = failed == 0
        self.last_summary = SubrouteVerificationSummary(
            changed_truck_routes=len(changed_routes),
            changed_drone_tasks=len(changed_task_indices),
            checked_drone_tasks=checked,
            failed_drone_tasks=failed,
            feasible=feasible,
        )
        return feasible

    def verify_all_tasks(self, solution: Solution) -> bool:
        """Check ALL drone tasks for robust energy feasibility.

        Unlike ``verify_candidate`` this does not compare against a base
        solution; it simply walks every drone task and applies the same
        per-task energy check.  This is much cheaper than a full
        ``Evaluator.evaluate_solution()`` call because it skips timing
        synchronisation, distance costing, delay computation, and all
        structural feasibility checks (which the search-phase evaluator
        already validated with gamma=0).

        Uses early exit: returns ``False`` as soon as the first infeasible
        task is found.
        """
        tasks = solution.drone_tasks
        if not tasks:
            return True

        checked = 0
        failed = 0
        for task in tasks:
            checked += 1
            if not self._task_is_robust_feasible(task):
                failed += 1
                # Early exit: no need to check remaining tasks.
                self.last_summary = SubrouteVerificationSummary(
                    changed_truck_routes=0,
                    changed_drone_tasks=len(tasks),
                    checked_drone_tasks=checked,
                    failed_drone_tasks=failed,
                    feasible=False,
                )
                return False

        self.last_summary = SubrouteVerificationSummary(
            changed_truck_routes=0,
            changed_drone_tasks=len(tasks),
            checked_drone_tasks=checked,
            failed_drone_tasks=0,
            feasible=True,
        )
        return True

    def _task_is_robust_feasible(self, task: DroneTask) -> bool:
        customers = list(task.customers())
        if not customers:
            return False

        payloads = _build_payloads(customers, self._demands)
        nodes = [task.launch_node, *customers, task.retrieve_node]
        nominal = 0.0
        deviations: list[float] = []
        for payload, a, b in zip(payloads, nodes, nodes[1:]):
            energy = _segment_energy(
                energy_model=self._energy_model,
                drone_time=self._drone_time,
                node_index=self._node_index,
                origin=a,
                destination=b,
                payload=payload,
            )
            if not math.isfinite(energy):
                return False
            nominal += energy
            deviations.append(energy * self._energy_deviation_rate)

        budget = _resolve_per_drone(
            self._energy_uncertainty_budget,
            task.drone_id,
            fallback=0.0,
        )
        capacity = _resolve_per_drone(
            self._drone_energy_capacity,
            task.drone_id,
            fallback=None,
        )
        if capacity is None:
            return True
        worst = nominal + _budgeted_sum(deviations, budget)
        return worst <= float(capacity) + self._tolerance


def _task_signature(task: DroneTask) -> tuple:
    return (
        int(task.drone_id),
        task.launch_truck,
        int(task.launch_node),
        tuple(int(x) for x in task.customers()),
        task.land_truck,
        int(task.retrieve_node),
        tuple(float(x) for x in task.payloads),
    )


def _changed_truck_route_ids(base: Solution, candidate: Solution) -> set[int]:
    base_routes = {route.id: tuple(route.nodes) for route in base.truck_routes}
    candidate_routes = {
        route.id: tuple(route.nodes) for route in candidate.truck_routes}
    all_ids = set(base_routes).union(candidate_routes)
    changed: set[int] = set()
    for route_id in all_ids:
        if base_routes.get(route_id) != candidate_routes.get(route_id):
            changed.add(route_id)
    return changed


def _changed_drone_task_indices(
    *,
    base: Solution,
    candidate: Solution,
    changed_truck_route_ids: set[int],
) -> set[int]:
    remaining = Counter(_task_signature(task) for task in base.drone_tasks)
    changed: set[int] = set()
    for index, task in enumerate(candidate.drone_tasks):
        signature = _task_signature(task)
        unchanged_by_signature = remaining.get(signature, 0) > 0
        if unchanged_by_signature:
            remaining[signature] -= 1
        else:
            changed.add(index)
            continue

        if (
            task.launch_truck in changed_truck_route_ids
            or task.land_truck in changed_truck_route_ids
        ):
            changed.add(index)
    return changed


def _resolve_per_drone(
    value: float | Mapping[int, float] | None,
    drone_id: int,
    *,
    fallback: float | None,
) -> float | None:
    if value is None:
        return fallback
    if isinstance(value, Mapping):
        return value.get(drone_id, fallback)
    return value


def _build_payloads(customers: Sequence[int], demands: Mapping[int, float]) -> list[float]:
    loads = [float(demands.get(node, 0.0)) for node in customers]
    payloads: list[float] = []
    remaining = sum(loads)
    for load in loads:
        payloads.append(remaining)
        remaining -= load
    payloads.append(0.0)
    return payloads


def _segment_energy(
    *,
    energy_model: DroneEnergyModel,
    drone_time: Sequence[Sequence[float]],
    node_index: Mapping[int, int],
    origin: int,
    destination: int,
    payload: float,
) -> float:
    i = node_index.get(origin)
    j = node_index.get(destination)
    if i is None or j is None:
        return float("inf")
    duration = drone_time[i][j]
    if not math.isfinite(duration):
        return float("inf")
    return energy_model.energy_kwh(payload, duration)


def _budgeted_sum(deviations: Sequence[float], budget: float | None) -> float:
    if budget is None or budget <= 0:
        return 0.0
    sorted_dev = sorted((max(0.0, float(d)) for d in deviations), reverse=True)
    if not sorted_dev:
        return 0.0
    capped_budget = max(0.0, min(float(budget), float(len(sorted_dev))))
    full = int(math.floor(capped_budget))
    frac = capped_budget - full
    total = sum(sorted_dev[:full])
    if full < len(sorted_dev) and frac > 0:
        total += frac * sorted_dev[full]
    return total
