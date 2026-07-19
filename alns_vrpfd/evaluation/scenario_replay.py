"""Scenario replay for out-of-sample robustness evaluation.

This module implements a Monte Carlo replay pipeline for fixed solutions under
stochastic energy realizations. It is designed to evaluate the stability of
robust solutions across different uncertainty budgets (gamma values) without
re-optimization.

Design notes:
- Distribution set follows Jeong et al. (2024, TRC, Section 5.3): ND/UD/NDC.
- Budgeted-uncertainty model in the solver follows Bertsimas & Sim (2004).
- Infeasible scenarios are not converted into pseudo-cost penalties here.
  Costs and service failures are reported as separate channels.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import csv
import math
from pathlib import Path
import random
import statistics
from typing import Dict, Mapping, Sequence, Tuple

from alns_vrpfd.evaluation.energy import DroneEnergyModel
from alns_vrpfd.evaluation.evaluator import Evaluator
from alns_vrpfd.evaluation.robustness import assess_drone_task_energy
from alns_vrpfd.instance.manager import InstanceManager
from alns_vrpfd.model.route import DroneTask
from alns_vrpfd.model.solution import Solution

__all__ = [
    "GammaSolutionInput",
    "ScenarioDistributionConfig",
    "ScenarioReplayConfig",
    "ScenarioReplayRecord",
    "ScenarioReplayResult",
    "ScenarioReplaySummary",
    "run_scenario_replay",
    "write_scenario_records_csv",
    "write_scenario_summary_csv",
]


_SUPPORTED_DISTRIBUTIONS = {
    "ND",
    "UD",
    "NDC",
    "LOGNORMAL",
    "STUDENT_T",
    "MIXTURE",
    "DETERMINISTIC",
}


@dataclass(frozen=True)
class ScenarioDistributionConfig:
    """Distribution configuration used to sample scenario multipliers."""

    name: str
    kind: str = "ND"
    cv: float = 0.1
    delta: float = 0.1
    correlation: float = 0.3
    degrees_of_freedom: int = 6
    mixture_probability: float = 0.2
    stress_mean: float = 1.2
    stress_cv: float = 0.1
    deterministic_multiplier: float = 1.0

    def normalized_kind(self) -> str:
        kind = self.kind.upper()
        if kind not in _SUPPORTED_DISTRIBUTIONS:
            raise ValueError(f"Unsupported scenario distribution kind: {self.kind}")
        return kind


@dataclass(frozen=True)
class ScenarioReplayConfig:
    """Global replay settings shared by all distributions and gamma values."""

    scenario_count: int = 1000
    seed: int = 2024
    safety_margin_kwh: float = 0.0
    clip_min_ratio: float = 0.7
    clip_max_ratio: float = 1.5
    energy_unit_cost: float = 1.0
    include_base_cost: bool = True
    time_tolerance: float = 1e-6


@dataclass(frozen=True)
class GammaSolutionInput:
    """A fixed solution produced under one gamma value."""

    gamma: int | float | str
    solution: Solution
    base_cost: float | None = None


@dataclass(frozen=True)
class ScenarioReplayRecord:
    """Scenario-level replay output for one gamma value."""

    distribution: str
    gamma: str
    scenario_id: int
    cost: float
    unserved: int
    no_takeoff: int
    abort_return: int
    served_customers: int
    total_customers: int
    all_served: bool


@dataclass(frozen=True)
class ScenarioReplaySummary:
    """Aggregated replay metrics (Table-6 style)."""

    distribution: str
    gamma: str
    scenario_count: int
    avg_cost: float
    std_cost: float
    max_cost: float
    min_cost: float
    avg_unserved: float
    p0_all_served: float
    avg_no_takeoff: float
    avg_abort_return: float


@dataclass(frozen=True)
class ScenarioReplayResult:
    """Container for scenario-level and aggregated replay outputs."""

    records: Tuple[ScenarioReplayRecord, ...]
    summaries: Tuple[ScenarioReplaySummary, ...]


@dataclass(frozen=True)
class _PreparedTask:
    gamma: str
    drone_id: int
    customer_count: int
    launch_time: float
    nominal_total_energy: float
    segment_nominal_energies: Tuple[float, ...]
    segment_arcs: Tuple[Tuple[int, int], ...]
    segment_arc_indices: Tuple[int, ...]


@dataclass(frozen=True)
class _PreparedSolution:
    gamma: str
    base_cost: float
    total_customers: int
    tasks: Tuple[_PreparedTask, ...] = field(default_factory=tuple)


def run_scenario_replay(
    *,
    instance: InstanceManager,
    gamma_solutions: Sequence[GammaSolutionInput],
    distributions: Sequence[ScenarioDistributionConfig],
    config: ScenarioReplayConfig | None = None,
    energy_model: DroneEnergyModel | None = None,
) -> ScenarioReplayResult:
    """Replay fixed gamma solutions on common uncertainty scenarios."""

    if not gamma_solutions:
        raise ValueError("gamma_solutions must contain at least one solution.")
    if not distributions:
        raise ValueError("distributions must contain at least one entry.")

    replay_cfg = config or ScenarioReplayConfig()
    _validate_replay_config(replay_cfg)

    evaluator = Evaluator(instance, energy_model=energy_model)
    model = energy_model or DroneEnergyModel()

    prepared_raw, all_arc_keys = _prepare_solutions(
        evaluator=evaluator,
        energy_model=model,
        gamma_solutions=gamma_solutions,
        include_base_cost=replay_cfg.include_base_cost,
        time_tolerance=replay_cfg.time_tolerance,
    )
    arc_keys_by_index = tuple(sorted(all_arc_keys))
    arc_to_index = {arc: idx for idx, arc in enumerate(arc_keys_by_index)}
    prepared_solutions = tuple(
        _with_arc_indices(sol, arc_to_index) for sol in prepared_raw
    )

    rng = random.Random(replay_cfg.seed)
    all_records: list[ScenarioReplayRecord] = []
    for dist_cfg in distributions:
        multipliers = _generate_scenario_multipliers(
            rng=rng,
            distribution=dist_cfg,
            scenario_count=replay_cfg.scenario_count,
            unit_count=len(arc_to_index),
            arc_keys=arc_keys_by_index,
            clip_min_ratio=replay_cfg.clip_min_ratio,
            clip_max_ratio=replay_cfg.clip_max_ratio,
        )
        dist_records = _replay_distribution(
            prepared_solutions=prepared_solutions,
            distribution_name=dist_cfg.name,
            multipliers=multipliers,
            safety_margin_kwh=replay_cfg.safety_margin_kwh,
            battery_capacity=instance.robust_config.drone_battery_capacity,
            energy_unit_cost=replay_cfg.energy_unit_cost,
            tolerance=replay_cfg.time_tolerance,
        )
        all_records.extend(dist_records)

    summaries = _summarize_records(all_records)
    return ScenarioReplayResult(
        records=tuple(all_records),
        summaries=tuple(summaries),
    )


def write_scenario_records_csv(
    path: str | Path,
    records: Sequence[ScenarioReplayRecord],
) -> None:
    """Persist scenario-level replay records to CSV."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "distribution",
                "gamma",
                "scenario_id",
                "cost",
                "unserved",
                "no_takeoff",
                "abort_return",
                "served_customers",
                "total_customers",
                "all_served",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "distribution": record.distribution,
                    "gamma": record.gamma,
                    "scenario_id": record.scenario_id,
                    "cost": record.cost,
                    "unserved": record.unserved,
                    "no_takeoff": record.no_takeoff,
                    "abort_return": record.abort_return,
                    "served_customers": record.served_customers,
                    "total_customers": record.total_customers,
                    "all_served": int(record.all_served),
                }
            )


def write_scenario_summary_csv(
    path: str | Path,
    summaries: Sequence[ScenarioReplaySummary],
) -> None:
    """Persist aggregated replay metrics to CSV."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "distribution",
                "gamma",
                "scenario_count",
                "avg_cost",
                "std_cost",
                "max_cost",
                "min_cost",
                "avg_unserved",
                "p0_all_served",
                "avg_no_takeoff",
                "avg_abort_return",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "distribution": summary.distribution,
                    "gamma": summary.gamma,
                    "scenario_count": summary.scenario_count,
                    "avg_cost": summary.avg_cost,
                    "std_cost": summary.std_cost,
                    "max_cost": summary.max_cost,
                    "min_cost": summary.min_cost,
                    "avg_unserved": summary.avg_unserved,
                    "p0_all_served": summary.p0_all_served,
                    "avg_no_takeoff": summary.avg_no_takeoff,
                    "avg_abort_return": summary.avg_abort_return,
                }
            )


def _validate_replay_config(config: ScenarioReplayConfig) -> None:
    if config.scenario_count <= 0:
        raise ValueError("scenario_count must be positive.")
    if config.safety_margin_kwh < 0:
        raise ValueError("safety_margin_kwh must be non-negative.")
    if config.clip_min_ratio <= 0 or config.clip_max_ratio <= 0:
        raise ValueError("Clip ratios must be positive.")
    if config.clip_min_ratio > config.clip_max_ratio:
        raise ValueError("clip_min_ratio cannot exceed clip_max_ratio.")
    if config.energy_unit_cost < 0:
        raise ValueError("energy_unit_cost must be non-negative.")


def _prepare_solutions(
    *,
    evaluator: Evaluator,
    energy_model: DroneEnergyModel,
    gamma_solutions: Sequence[GammaSolutionInput],
    include_base_cost: bool,
    time_tolerance: float,
) -> Tuple[Tuple[_PreparedSolution, ...], set[Tuple[int, int]]]:
    prepared: list[_PreparedSolution] = []
    all_arc_keys: set[Tuple[int, int]] = set()

    for item in gamma_solutions:
        gamma_key = str(item.gamma)
        details = evaluator.evaluate_with_details(item.solution)
        base_cost = item.base_cost
        if base_cost is None:
            base_cost = details.result.total_cost if include_base_cost else 0.0

        keys = _task_key_lookup(item.solution.drone_tasks)
        tasks: list[_PreparedTask] = []
        total_customers = 0
        for idx, task in enumerate(item.solution.drone_tasks):
            timing_key = keys[idx]
            timing = details.drone_timings.get(timing_key)
            if timing is None:
                continue
            assessment = assess_drone_task_energy(
                task=task,
                timing=timing,
                energy_model=energy_model,
                deviation_rate=0.0,
                uncertainty_budget=0.0,
                capacity=None,
                tolerance=time_tolerance,
                time_tolerance=time_tolerance,
            )
            if not assessment.segment_energies:
                continue

            nodes = [task.launch_node, *task.customers(), task.retrieve_node]
            arc_keys = tuple(zip(nodes, nodes[1:]))
            for arc in arc_keys:
                all_arc_keys.add((int(arc[0]), int(arc[1])))

            total_customers += len(task.customers())
            tasks.append(
                _PreparedTask(
                    gamma=gamma_key,
                    drone_id=task.drone_id,
                    customer_count=len(task.customers()),
                    launch_time=timing.launch_time,
                    nominal_total_energy=assessment.nominal_energy,
                    segment_nominal_energies=assessment.segment_energies,
                    segment_arcs=tuple(
                        (int(origin), int(dest)) for origin, dest in arc_keys
                    ),
                    segment_arc_indices=tuple(),
                )
            )

        tasks.sort(key=lambda t: (t.drone_id, t.launch_time))
        prepared.append(
            _PreparedSolution(
                gamma=gamma_key,
                base_cost=float(base_cost),
                total_customers=total_customers,
                tasks=tuple(tasks),
            )
        )

    return tuple(prepared), all_arc_keys


def _with_arc_indices(
    solution: _PreparedSolution,
    arc_to_index: Mapping[Tuple[int, int], int],
) -> _PreparedSolution:
    new_tasks: list[_PreparedTask] = []
    for task in solution.tasks:
        indices: list[int] = []
        for arc_key in task.segment_arcs:
            try:
                indices.append(arc_to_index[arc_key])
            except KeyError as exc:
                raise KeyError(f"Missing arc in scenario pool: {arc_key}") from exc
        new_tasks.append(
            _PreparedTask(
                gamma=task.gamma,
                drone_id=task.drone_id,
                customer_count=task.customer_count,
                launch_time=task.launch_time,
                nominal_total_energy=task.nominal_total_energy,
                segment_nominal_energies=task.segment_nominal_energies,
                segment_arcs=task.segment_arcs,
                segment_arc_indices=tuple(indices),
            )
        )

    return _PreparedSolution(
        gamma=solution.gamma,
        base_cost=solution.base_cost,
        total_customers=solution.total_customers,
        tasks=tuple(new_tasks),
    )


def _task_key_lookup(tasks: Sequence[DroneTask]) -> Dict[int, int]:
    counts: Dict[int, int] = defaultdict(int)
    for task in tasks:
        if task.task_id is not None:
            counts[int(task.task_id)] += 1

    keys: Dict[int, int] = {}
    for index, task in enumerate(tasks):
        if task.task_id is not None and counts.get(int(task.task_id), 0) == 1:
            keys[index] = int(task.task_id)
        else:
            keys[index] = index
    return keys


def _generate_scenario_multipliers(
    *,
    rng: random.Random,
    distribution: ScenarioDistributionConfig,
    scenario_count: int,
    unit_count: int,
    arc_keys: Sequence[Tuple[int, int]] | None,
    clip_min_ratio: float,
    clip_max_ratio: float,
) -> Tuple[Tuple[float, ...], ...]:
    kind = distribution.normalized_kind()
    rows: list[Tuple[float, ...]] = []
    for _ in range(scenario_count):
        if unit_count == 0:
            rows.append(tuple())
            continue
        raw = _sample_row(
            rng,
            distribution,
            kind,
            unit_count,
            arc_keys=arc_keys,
        )
        clipped = [
            min(clip_max_ratio, max(clip_min_ratio, value))
            for value in raw
        ]
        rows.append(tuple(clipped))
    return tuple(rows)


def _sample_row(
    rng: random.Random,
    distribution: ScenarioDistributionConfig,
    kind: str,
    unit_count: int,
    arc_keys: Sequence[Tuple[int, int]] | None = None,
) -> list[float]:
    if kind == "ND":
        if distribution.cv < 0:
            raise ValueError("ND distribution requires cv >= 0.")
        return [rng.gauss(1.0, distribution.cv) for _ in range(unit_count)]

    if kind == "UD":
        if distribution.delta < 0:
            raise ValueError("UD distribution requires delta >= 0.")
        lo = 1.0 - distribution.delta
        hi = 1.0 + distribution.delta
        return [rng.uniform(lo, hi) for _ in range(unit_count)]

    if kind == "NDC":
        if distribution.cv < 0:
            raise ValueError("NDC distribution requires cv >= 0.")
        if arc_keys is None or len(arc_keys) != unit_count:
            raise ValueError("NDC requires arc_keys aligned with unit_count.")
        # Node-driven correlation:
        # h_a = h_bar_a * (1 + z_i + z_j), z_i ~ N(0, (alpha/(2*sqrt(2)))^2).
        # Here alpha is `distribution.cv` (default 0.1).
        node_sigma = distribution.cv / (2.0 * math.sqrt(2.0))
        node_noise: dict[int, float] = {}

        def _node_z(node: int) -> float:
            value = node_noise.get(node)
            if value is None:
                value = rng.gauss(0.0, node_sigma)
                node_noise[node] = value
            return value

        row: list[float] = []
        for origin, dest in arc_keys:
            row.append(1.0 + _node_z(int(origin)) + _node_z(int(dest)))
        return row

    if kind == "LOGNORMAL":
        if distribution.cv < 0:
            raise ValueError("LOGNORMAL distribution requires cv >= 0.")
        sigma_sq = math.log(1.0 + distribution.cv * distribution.cv)
        sigma = math.sqrt(sigma_sq)
        mu = -0.5 * sigma_sq
        return [math.exp(mu + sigma * rng.gauss(0.0, 1.0)) for _ in range(unit_count)]

    if kind == "STUDENT_T":
        nu = int(distribution.degrees_of_freedom)
        if nu <= 2:
            raise ValueError("STUDENT_T requires degrees_of_freedom > 2.")
        if distribution.cv < 0:
            raise ValueError("STUDENT_T distribution requires cv >= 0.")
        std_t = math.sqrt(nu / (nu - 2.0))
        scale = distribution.cv / std_t
        return [1.0 + scale * _sample_student_t(rng, nu) for _ in range(unit_count)]

    if kind == "MIXTURE":
        if not (0.0 <= distribution.mixture_probability <= 1.0):
            raise ValueError("MIXTURE requires 0 <= mixture_probability <= 1.")
        if distribution.cv < 0 or distribution.stress_cv < 0:
            raise ValueError("MIXTURE requires non-negative cv values.")
        row: list[float] = []
        for _ in range(unit_count):
            if rng.random() < distribution.mixture_probability:
                value = rng.gauss(distribution.stress_mean, distribution.stress_cv)
            else:
                value = rng.gauss(1.0, distribution.cv)
            row.append(value)
        return row

    if kind == "DETERMINISTIC":
        return [distribution.deterministic_multiplier for _ in range(unit_count)]

    raise ValueError(f"Unsupported distribution kind: {distribution.kind}")


def _sample_student_t(rng: random.Random, dof: int) -> float:
    numerator = rng.gauss(0.0, 1.0)
    chi_square = rng.gammavariate(dof / 2.0, 2.0)
    if chi_square <= 0:
        return 0.0
    return numerator / math.sqrt(chi_square / dof)


def _replay_distribution(
    *,
    prepared_solutions: Sequence[_PreparedSolution],
    distribution_name: str,
    multipliers: Sequence[Sequence[float]],
    safety_margin_kwh: float,
    battery_capacity: float | Mapping[int, float] | None,
    energy_unit_cost: float,
    tolerance: float,
) -> Tuple[ScenarioReplayRecord, ...]:
    records: list[ScenarioReplayRecord] = []
    for scenario_id, row in enumerate(multipliers):
        for solution in prepared_solutions:
            metrics = _replay_one_solution(
                solution=solution,
                multiplier_row=row,
                safety_margin_kwh=safety_margin_kwh,
                battery_capacity=battery_capacity,
                energy_unit_cost=energy_unit_cost,
                tolerance=tolerance,
            )
            records.append(
                ScenarioReplayRecord(
                    distribution=distribution_name,
                    gamma=solution.gamma,
                    scenario_id=scenario_id,
                    cost=metrics["cost"],
                    unserved=metrics["unserved"],
                    no_takeoff=metrics["no_takeoff"],
                    abort_return=metrics["abort_return"],
                    served_customers=metrics["served_customers"],
                    total_customers=solution.total_customers,
                    all_served=(metrics["unserved"] == 0),
                )
            )
    return tuple(records)


def _replay_one_solution(
    *,
    solution: _PreparedSolution,
    multiplier_row: Sequence[float],
    safety_margin_kwh: float,
    battery_capacity: float | Mapping[int, float] | None,
    energy_unit_cost: float,
    tolerance: float,
) -> Dict[str, float | int]:
    realized_energy = 0.0
    served_customers = 0
    unserved = 0
    no_takeoff = 0
    abort_return = 0

    for task in solution.tasks:
        capacity = _resolve_per_drone_capacity(battery_capacity, task.drone_id)
        finite_capacity = capacity is not None and math.isfinite(capacity)
        if capacity is None:
            capacity_value = float("inf")
        else:
            capacity_value = float(capacity)

        required_nominal = task.nominal_total_energy + safety_margin_kwh
        if finite_capacity and (capacity_value + tolerance < required_nominal):
            no_takeoff += 1
            unserved += task.customer_count
            continue

        segment_realized = [
            nominal * multiplier_row[arc_index]
            for nominal, arc_index in zip(
                task.segment_nominal_energies, task.segment_arc_indices
            )
        ]
        suffix_energy = [0.0] * (len(segment_realized) + 1)
        for i in range(len(segment_realized) - 1, -1, -1):
            suffix_energy[i] = suffix_energy[i + 1] + segment_realized[i]

        remaining = capacity_value
        served_in_task = 0
        aborted = False
        for seg_idx, seg_energy in enumerate(segment_realized):
            if finite_capacity and seg_energy > remaining + tolerance:
                abort_return += 1
                unserved += (task.customer_count - served_in_task)
                aborted = True
                break

            remaining -= seg_energy
            realized_energy += seg_energy

            if seg_idx < task.customer_count:
                served_in_task += 1

            if finite_capacity and seg_idx < len(segment_realized) - 1:
                required_rest = suffix_energy[seg_idx + 1] + safety_margin_kwh
                if remaining + tolerance < required_rest:
                    abort_return += 1
                    unserved += (task.customer_count - served_in_task)
                    aborted = True
                    break

        if not aborted:
            served_customers += task.customer_count

    cost = solution.base_cost + energy_unit_cost * realized_energy
    return {
        "cost": cost,
        "unserved": unserved,
        "no_takeoff": no_takeoff,
        "abort_return": abort_return,
        "served_customers": served_customers,
    }


def _resolve_per_drone_capacity(
    capacity: float | Mapping[int, float] | None,
    drone_id: int,
) -> float | None:
    if capacity is None:
        return None
    if isinstance(capacity, Mapping):
        return capacity.get(drone_id)
    return capacity


def _summarize_records(
    records: Sequence[ScenarioReplayRecord],
) -> Tuple[ScenarioReplaySummary, ...]:
    grouped: Dict[Tuple[str, str], list[ScenarioReplayRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.distribution, record.gamma)].append(record)

    summaries: list[ScenarioReplaySummary] = []
    for (distribution, gamma), items in sorted(grouped.items()):
        costs = [r.cost for r in items]
        unserved = [r.unserved for r in items]
        no_takeoff = [r.no_takeoff for r in items]
        abort_return = [r.abort_return for r in items]
        n = len(items)
        if n == 0:
            continue

        avg_cost = statistics.fmean(costs)
        std_cost = statistics.pstdev(costs) if n > 1 else 0.0
        p0 = sum(1 for v in unserved if v == 0) / n
        summaries.append(
            ScenarioReplaySummary(
                distribution=distribution,
                gamma=gamma,
                scenario_count=n,
                avg_cost=avg_cost,
                std_cost=std_cost,
                max_cost=max(costs),
                min_cost=min(costs),
                avg_unserved=statistics.fmean(unserved),
                p0_all_served=p0,
                avg_no_takeoff=statistics.fmean(no_takeoff),
                avg_abort_return=statistics.fmean(abort_return),
            )
        )
    return tuple(summaries)
