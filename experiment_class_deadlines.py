"""
Standalone experiment: compare ALNS with OLD demand-based deadlines vs NEW class-based deadlines.

Creates a separate results directory. Does NOT modify any existing code.

Usage:
    cd code/
    python experiment_class_deadlines.py
"""

import sys, os, time, random, numpy as np, logging, datetime
from pathlib import Path
from copy import deepcopy

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from alns_vrpfd.utils.data_reader import InstanceDataReader, TimeWindowConfig
from alns_vrpfd.utils.io_utils import read_instance
from alns_vrpfd.evaluation.evaluator import (
    Evaluator, EvaluationResult, EvaluationDetails, DelayBreakdown,
    NodeDelay, TimeWindowViolation,
)
from alns_vrpfd.model.initializer import build_two_phase_initial_solution
from alns_vrpfd.core.operators import (
    DestroyRandom, DestroyShaw, DestroyWorstDistance,
    RepairCheapest, RepairDronePriorityRegret,
    RepairEqualPriority, RepairTruckFirst,
)
from alns_vrpfd.core.sa import SANNCfg, SimulatedAnnealingALNS
from alns_vrpfd.utils.config_loader import ALNSConfig
from alns_vrpfd.instance.customers import Customer
from alns_vrpfd.deprivation import DEFAULT_SUPPLY_CLASS_SEQUENCE, WANG_SUPPLY_CLASSES, deprivation_cost

CLASS_NAMES = {key: WANG_SUPPLY_CLASSES[key].label for key in DEFAULT_SUPPLY_CLASS_SEQUENCE}

INSTANCE_NAMES = [f"R_30_10_{i}" for i in range(1, 6)]
SEED = 42


def generate_class_based_deadlines(instance, rng):
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
            c = str(rng.choice(supply_classes))
        classes[cid] = c
        params = WANG_SUPPLY_CLASSES[c]
        ci = idx_map[cid]
        r_i = min(dist_truck[depot_idx][ci] / truck_speed,
                   dist_drone[depot_idx][ci] / drone_speed)
        delta_o = float(rng.uniform(*params.deadline_optimal_delta_hours))
        delta_l = float(rng.uniform(*params.deadline_latest_delta_hours))
        instance.customer_manager.assign_supply_class(cid, c)
        instance.customer_manager.assign_time_window(
            cid, r_i + delta_o, r_i + delta_o + delta_l)
    return classes


class ClassWeightedEvaluator(Evaluator):
    """Evaluator with class-weighted deprivation cost."""

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
            cost = deprivation_cost(tau, self._node_classes.get(delay.node_id, "water"), cost_lambda=30.0, rho=0.2083, normalized=True)
            total_delay += cost

        return DelayBreakdown(
            total_delay=total_delay,
            nodes=tuple(delays),
            violations=tuple(violations),
        )


def run_alns_once(instance, config, seed, evaluator):
    sa_config_dict = config.build_sa_config_dict()
    sa_config_dict["iterations"] = 2000
    sa_config_dict["size"] = "small"
    sa_config_dict["log_operator_metrics"] = False
    sa_cfg = SANNCfg(**sa_config_dict)
    alns_rng = random.Random(seed)

    destroy_ops = [
        DestroyRandom(instance, rng=random.Random(seed + 1000),
                      anchor_strategy="rebase_to_neighbor"),
        DestroyWorstDistance(instance, rng=random.Random(seed + 1004),
                            anchor_strategy="rebase_to_neighbor"),
        DestroyShaw(instance, rng=random.Random(seed + 1002),
                     anchor_strategy="rebase_to_neighbor"),
    ]
    repair_ops = [
        RepairCheapest(instance, rng=random.Random(seed + 2004),
                       drone_priority=2.2, robust_energy_mode="embedded"),
        RepairDronePriorityRegret(instance, rng=random.Random(seed + 2002),
                                   drone_priority=2.2, robust_energy_mode="embedded"),
        RepairTruckFirst(instance, rng=random.Random(seed + 2003),
                          drone_priority=2.2, robust_energy_mode="embedded"),
        RepairEqualPriority(instance, rng=random.Random(seed + 2001),
                            drone_priority=2.2, robust_energy_mode="embedded"),
    ]

    initial_solution = build_two_phase_initial_solution(instance)
    alns = SimulatedAnnealingALNS(
        instance=instance, destroy_ops=destroy_ops, repair_ops=repair_ops,
        evaluator=evaluator, cfg=sa_cfg, rng=alns_rng)

    start = time.time()
    best_sol = alns.run(initial_solution)
    runtime = time.time() - start
    details = evaluator.evaluate_with_details(best_sol)
    eval_res = details.result

    return {
        "cost": eval_res.total_cost,
        "truck_cost": eval_res.truck_distance_cost,
        "drone_cost": eval_res.drone_distance_cost,
        "delay_cost": eval_res.delay_penalty,
        "feasible": eval_res.feasible,
        "runtime": runtime,
        "truck_routes": [tr.nodes for tr in best_sol.truck_routes],
        "drone_tasks": [(dt.nodes, dt.launch_node, dt.retrieve_node)
                        for dt in best_sol.drone_tasks],
        "delay_nodes": len(details.delay_breakdown.nodes),
        "violations": len(details.delay_breakdown.violations),
    }


def print_comparison(old, new, classes, name):
    print(f"\n{'='*80}")
    print(f"  COMPARISON: {name}")
    print(f"{'='*80}")
    print(f"{'Metric':<30} {'OLD (demand)':>14} {'NEW (class)':>14} {'Change':>10}")
    print("-" * 70)
    for key, label in [("cost", "Total Cost"), ("truck_cost", "Truck Dist"),
                        ("drone_cost", "Drone Dist"), ("delay_cost", "Delay Cost"),
                        ("delay_nodes", "Delayed Nodes"), ("violations", "Violations"),
                        ("runtime", "Runtime (s)")]:
        ov, nv = old[key], new[key]
        if isinstance(ov, (int, float)) and isinstance(nv, (int, float)) and ov != 0:
            pct = (nv - ov) / ov * 100
            print(f"{label:<30} {ov:>14.2f} {nv:>14.2f} {pct:>+9.1f}%")
        else:
            print(f"{label:<30} {str(ov):>14} {str(nv):>14}")
    print(f"{'Feasible':<30} {str(old['feasible']):>14} {str(new['feasible']):>14}")

    print(f"\n  Class assignments:")
    print(f"  {'Node':>4} {'Class':>16} {'Beta':>8} {'Omega':>8}")
    print(f"  {'-'*42}")
    demands = {}
    for cid in sorted(classes.keys()):
        c = classes[cid]
        demands[cid] = c
        spec = WANG_SUPPLY_CLASSES[c]
        print(f"  {cid:>4} {CLASS_NAMES[c]:>16} {spec.beta:>8.4f} {spec.omega:>8.2f}")


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = PROJECT_ROOT / "results" / "class_deadlines"
    results_dir.mkdir(parents=True, exist_ok=True)

    config = ALNSConfig(str(PROJECT_ROOT / "config" / "alns_config.yaml"))

    all_old, all_new = [], []

    for inst_name in INSTANCE_NAMES:
        fpath = str(PROJECT_ROOT / "data" / "Instance10" / f"{inst_name}.txt")
        if not os.path.exists(fpath):
            print(f"SKIP: {fpath} not found")
            continue
        rng = np.random.default_rng(SEED)

        # ── OLD ────────────────────────────────────────────────────────
        print(f"\n>>> OLD (demand-based): {inst_name}")
        instance_old = read_instance(fpath, strategy="demand_based")
        instance_old.vehicle_specs["drone"].endurance = float("inf")
        instance_old.configure_robustness(
            drone_battery_capacity=config.drone_battery_capacity,
            energy_uncertainty_budget=config.energy_uncertainty_budget,
            energy_deviation_rate=config.energy_deviation_rate,
            same_truck_retrieval=config.same_truck_retrieval)
        ev_old = Evaluator(instance_old,
                           rendezvous_tolerance=config.drone_rendezvous_tolerance,
                           forced_drone_customers=config.forced_drone_customers,
                           allow_multiple_launch_per_node=config.relax_allow_multiple_launch_per_node)
        old_res = run_alns_once(instance_old, config, SEED, ev_old)

        # ── NEW ────────────────────────────────────────────────────────
        print(f">>> NEW (class-based): {inst_name}")
        instance_new = read_instance(fpath, strategy="demand_based", apply_time_windows=False)
        instance_new.vehicle_specs["drone"].endurance = float("inf")
        instance_new.configure_robustness(
drone_battery_capacity=config.drone_battery_capacity,
             energy_uncertainty_budget=config.energy_uncertainty_budget,
             energy_deviation_rate=config.energy_deviation_rate,
             same_truck_retrieval=config.same_truck_retrieval)
        classes = generate_class_based_deadlines(instance_new, rng)
        ev_new = ClassWeightedEvaluator(instance_new, classes,
                                         rendezvous_tolerance=config.drone_rendezvous_tolerance,
                                         forced_drone_customers=config.forced_drone_customers,
                                         allow_multiple_launch_per_node=config.relax_allow_multiple_launch_per_node)
        new_res = run_alns_once(instance_new, config, SEED, ev_new)

        print_comparison(old_res, new_res, classes, inst_name)
        all_old.append(old_res)
        all_new.append(new_res)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  SUMMARY ACROSS ALL INSTANCES")
    print(f"{'='*80}")
    old_feas = [r for r in all_old if r["feasible"]]
    new_feas = [r for r in all_new if r["feasible"]]
    if old_feas and new_feas:
        avg_old = sum(r["cost"] for r in old_feas) / len(old_feas)
        avg_new = sum(r["cost"] for r in new_feas) / len(new_feas)
        avg_old_delay = sum(r["delay_cost"] for r in old_feas) / len(old_feas)
        avg_new_delay = sum(r["delay_cost"] for r in new_feas) / len(new_feas)
        print(f"  OLD avg_cost={avg_old:.2f}  avg_delay={avg_old_delay:.2f}  feasible={len(old_feas)}/{len(all_old)}")
        print(f"  NEW avg_cost={avg_new:.2f}  avg_delay={avg_new_delay:.2f}  feasible={len(new_feas)}/{len(all_new)}")
        print(f"  Cost change: {(avg_new-avg_old)/avg_old*100:+.1f}%")
        if avg_old_delay > 0:
            print(f"  Delay change: {(avg_new_delay-avg_old_delay)/avg_old_delay*100:+.1f}%")

    # ── Save log ────────────────────────────────────────────────────────────
    log_path = results_dir / f"experiment_{timestamp}.log"
    with open(log_path, "w") as f:
        f.write(f"Experiment: {timestamp}\n")
        f.write(f"OLD instances: {len(all_old)}, NEW instances: {len(all_new)}\n")
        for i, (o, n) in enumerate(zip(all_old, all_new)):
            f.write(f"\nInstance {INSTANCE_NAMES[i] if i < len(INSTANCE_NAMES) else '?'}:\n")
            f.write(f"  OLD: cost={o['cost']:.2f}, delay={o['delay_cost']:.2f}, feasible={o['feasible']}\n")
            f.write(f"  NEW: cost={n['cost']:.2f}, delay={n['delay_cost']:.2f}, feasible={n['feasible']}\n")
    print(f"\nLog saved to: {log_path}")
