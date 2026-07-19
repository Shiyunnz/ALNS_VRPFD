"""Shared tuning framework for ALNS, TS, and GA.

Provides:
- Objective construction (stability-aware: mean + λ·std)
- Class-based deadline generation (consistent across algorithms)
- Instance loading
- Result saving with manifest tracking
"""

from __future__ import annotations

import json
import time
import random
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from alns_vrpfd.utils.io_utils import read_instance
from alns_vrpfd.evaluation.evaluator import Evaluator, DelayBreakdown, NodeDelay, TimeWindowViolation
from alns_vrpfd.model.initializer import build_two_phase_initial_solution
from alns_vrpfd.deprivation import DEFAULT_SUPPLY_CLASS_SEQUENCE, WANG_SUPPLY_CLASSES, deprivation_cost
from revision.manifest import (
    config_hash, record_run, update_run, check_duplicate,
    TRAINING_INSTANCES, ALL_INSTANCE10,
    TRAINING_SEEDS_PHASE1, TRAINING_SEEDS_PHASE2, VALIDATION_SEEDS,
    REVISION_ROOT,
)

INFEASIBLE_PENALTY = 1e6
TIME_TOLERANCE_HOURS = 0.02
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ClassWeightedEvaluator(Evaluator):
    """Evaluator using class-based deprivation cost (consistent across ALNS/TS/GA)."""

    def __init__(self, instance, classes, **kwargs):
        super().__init__(instance, **kwargs)
        self._node_classes = classes

    def _compute_delay_penalty(self, truck_timings, drone_timings):
        delays = []
        violations = []
        for route_id, timing in truck_timings.items():
            for node_id, arrival in timing.arrival_times.items():
                if not self._is_customer(node_id):
                    continue
                customer = self._customer_lookup.get(node_id)
                optimal = customer.optimal_time if customer else None
                latest = customer.latest_time if customer else None
                if latest is not None and arrival - latest > self._time_tolerance:
                    violations.append(TimeWindowViolation(
                        node_id=node_id, arrival_time=arrival,
                        latest_time=latest, served_by="truck", route_id=route_id))
                delay_value = 0.0
                if optimal is not None and arrival - optimal > self._time_tolerance:
                    delay_value = arrival - optimal
                if delay_value > 0.0:
                    delays.append(NodeDelay(
                        node_id=node_id, arrival_time=arrival,
                        reference_time=optimal or 0.0, delay=delay_value,
                        served_by="truck", route_id=route_id))
        for task_key, timing in drone_timings.items():
            for node_id, arrival in timing.customer_arrival_times.items():
                customer = self._customer_lookup.get(node_id)
                optimal = customer.optimal_time if customer else None
                latest = customer.latest_time if customer else None
                if latest is not None and arrival - latest > self._time_tolerance:
                    violations.append(TimeWindowViolation(
                        node_id=node_id, arrival_time=arrival,
                        latest_time=latest, served_by="drone", route_id=int(task_key)))
                delay_value = 0.0
                if optimal is not None and arrival - optimal > self._time_tolerance:
                    delay_value = arrival - optimal
                if delay_value > 0.0:
                    delays.append(NodeDelay(
                        node_id=node_id, arrival_time=arrival,
                        reference_time=optimal or 0.0, delay=delay_value,
                        served_by="drone", route_id=int(task_key)))
        total_delay = 0.0
        for delay in delays:
            tau = delay.delay
            if tau <= 0.0:
                continue
            cost = deprivation_cost(
                tau,
                self._node_classes.get(delay.node_id, "water"),
                cost_lambda=self._cost_lambda,
                rho=self._cost_rho,
                normalized=self._cost_normalized,
            )
            total_delay += cost
        return DelayBreakdown(
            total_delay=total_delay, nodes=tuple(delays), violations=tuple(violations))


def generate_class_based_deadlines(instance, seed=42):
    """Generate class-based deadlines for an instance. Deterministic given seed."""
    rng_tw = np.random.default_rng(seed)
    customer_ids = instance.customer_manager.customer_ids()
    truck_speed = instance.vehicle_specs["truck"].speed
    drone_speed = instance.vehicle_specs["drone"].speed
    depot_id = instance.customer_manager.depot_start
    node_list = instance.all_node_ids()
    idx_map = {nid: i for i, nid in enumerate(node_list)}
    depot_idx = idx_map[depot_id]
    dist_truck = instance.distance_matrix("truck")
    dist_drone = instance.distance_matrix("drone")
    classes = {}
    supply_classes = list(DEFAULT_SUPPLY_CLASS_SEQUENCE)
    for offset, cid in enumerate(customer_ids):
        c = supply_classes[offset % len(supply_classes)]
        if offset >= len(supply_classes):
            c = str(rng_tw.choice(supply_classes))
        classes[cid] = c
        params = WANG_SUPPLY_CLASSES[c]
        ci = idx_map[cid]
        r_i = min(dist_truck[depot_idx][ci] / truck_speed,
                   dist_drone[depot_idx][ci] / drone_speed)
        delta_o = float(rng_tw.uniform(*params.deadline_optimal_delta_hours))
        delta_l = float(rng_tw.uniform(*params.deadline_latest_delta_hours))
        instance.customer_manager.assign_supply_class(cid, c)
        instance.customer_manager.assign_time_window(
            cid, r_i + delta_o, r_i + delta_o + delta_l)
    return classes


def load_instance_for_tuning(instance_name: str, seed: int = 42, instance_dir: str = "Instance10"):
    """Load and configure a VRPFD instance with class-based deadlines."""
    fpath = str(PROJECT_ROOT / "data" / instance_dir / f"{instance_name}.txt")
    from alns_vrpfd.utils.config_loader import ALNSConfig
    config = ALNSConfig(str(PROJECT_ROOT / "config" / "alns_config.yaml"))

    instance = read_instance(fpath, strategy="class_based", apply_time_windows=False)
    instance.vehicle_specs["drone"].endurance = float("inf")
    instance.configure_robustness(
        drone_battery_capacity=config.drone_battery_capacity,
        energy_uncertainty_budget=config.energy_uncertainty_budget,
        energy_deviation_rate=config.energy_deviation_rate,
        same_truck_retrieval=config.same_truck_retrieval,
    )
    classes = generate_class_based_deadlines(instance, seed=seed)
    evaluator = ClassWeightedEvaluator(
        instance, classes,
        rendezvous_tolerance=config.drone_rendezvous_tolerance,
        forced_drone_customers=config.forced_drone_customers,
        allow_multiple_launch_per_node=config.relax_allow_multiple_launch_per_node,
        cost_lambda=config.cost_lambda,
        cost_rho=config.cost_rho,
        cost_normalized=config.cost_normalized,
        time_tolerance=TIME_TOLERANCE_HOURS,
    )
    return instance, evaluator, classes


def stability_objective(costs: List[float], lambda_std: float = 1.0) -> float:
    """Mean + lambda * std objective."""
    arr = np.array(costs)
    return float(np.mean(arr) + lambda_std * np.std(arr))


def save_tuning_result(
    algorithm: str,
    phase: str,
    best_params: Dict[str, Any],
    best_value: float,
    all_trials_summary: List[Dict],
    output_dir: Optional[Path] = None,
) -> Path:
    """Save tuning result to revision directory."""
    if output_dir is None:
        output_dir = REVISION_ROOT / "tuning" / f"{algorithm}_bayesian"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result = {
        "algorithm": algorithm,
        "phase": phase,
        "timestamp": timestamp,
        "best_value": best_value,
        "best_params": best_params,
        "n_trials": len(all_trials_summary),
        "trials_summary": all_trials_summary,
    }

    path = output_dir / f"{algorithm}_{phase}_best_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    config_path = output_dir / f"{algorithm}_{phase}_config_{timestamp}.yaml" if algorithm == "alns" else output_dir / f"{algorithm}_{phase}_config_{timestamp}.json"

    if algorithm == "alns":
        with open(config_path, "w") as f:
            import yaml
            yaml.dump(best_params, f, default_flow_style=False)
    else:
        with open(config_path, "w") as f:
            json.dump(best_params, f, indent=2)

    logger.info(f"Saved {algorithm} {phase} result: best_value={best_value:.4f}")
    logger.info(f"  Config: {config_path}")
    logger.info(f"  Full result: {path}")
    return path
