"""Quick test of the new exponential projection delay cost on Instance10."""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from alns_vrpfd.deprivation import (
    WANG_SUPPLY_CLASSES,
    DEFAULT_SUPPLY_CLASS_SEQUENCE,
    deprivation_cost,
    HOLGUIN_INTERCEPT,
    MAX_TARDINESS_HOURS,
)
from alns_vrpfd.evaluation import Evaluator
from alns_vrpfd.instance import InstanceManager, TimeWindowConfig
from alns_vrpfd.model.initializer import build_two_phase_initial_solution
from alns_vrpfd.core.operators import (
    DestroyRandom,
    DestroyWorstDistance,
    DestroyShaw,
    RepairCheapest,
    RepairDronePriorityRegret,
    RepairTruckFirst,
    RepairEqualPriority,
)
from alns_vrpfd.core.sa import SANNCfg, SimulatedAnnealingALNS
from alns_vrpfd.utils.config_loader import ALNSConfig
from alns_vrpfd.utils.io_utils import read_instance
import math
import random


def verify_deprivation_cost():
    """Verify the new exponential projection cost function."""
    print("=" * 60)
    print("Verifying new exponential projection delay cost function")
    print("=" * 60)

    print(f"\nHOLGUIN_INTERCEPT = {HOLGUIN_INTERCEPT}")
    print(f"MAX_TARDINESS_HOURS = {MAX_TARDINESS_HOURS}")

    print("\nSupply class parameters:")
    print(f"{'Class':>12} {'beta':>8} {'omega':>8} {'delta_o':>15} {'delta_l':>15}")
    for cls in DEFAULT_SUPPLY_CLASS_SEQUENCE:
        spec = WANG_SUPPLY_CLASSES[cls]
        print(f"{spec.label:>12} {spec.beta:>8.4f} {spec.omega:>8.2f} "
              f"{str(spec.deadline_optimal_delta_hours):>15} "
              f"{str(spec.deadline_latest_delta_hours):>15}")

    print("\nDelay cost f(tau) with lambda=12, rho=1.0, normalized=True:")
    print(f"{'tau(h)':>8} {'Medicine':>12} {'Water':>12} {'Food':>12} {'Tent':>12}")
    for tau in [0.1, 0.25, 0.5, 1.0, 2.0, 4.4947]:
        values = [deprivation_cost(tau, c, cost_lambda=12.0, rho=1.0, normalized=True)
                  for c in DEFAULT_SUPPLY_CLASS_SEQUENCE]
        print(f"{tau:8.4f} " + " ".join(f"{v:12.4f}" for v in values))

    print("\nDelay cost f(tau) with lambda=12, rho=1/24, normalized=True (no compression):")
    print(f"{'tau(h)':>8} {'Medicine':>12} {'Water':>12} {'Food':>12} {'Tent':>12}")
    for tau in [0.1, 0.25, 0.5, 1.0, 2.0, 4.4947]:
        values = [deprivation_cost(tau, c, cost_lambda=12.0, rho=1.0/24.0, normalized=True)
                  for c in DEFAULT_SUPPLY_CLASS_SEQUENCE]
        print(f"{tau:8.4f} " + " ".join(f"{v:12.4f}" for v in values))

    print("\nDelay cost f(tau) with lambda=12, rho=1.0, normalized=False (raw):")
    print(f"{'tau(h)':>8} {'Medicine':>12} {'Water':>12} {'Food':>12} {'Tent':>12}")
    for tau in [0.1, 0.25, 0.5, 1.0, 2.0, 4.4947]:
        values = [deprivation_cost(tau, c, cost_lambda=12.0, rho=1.0, normalized=False)
                  for c in DEFAULT_SUPPLY_CLASS_SEQUENCE]
        print(f"{tau:8.4f} " + " ".join(f"{v:12.4f}" for v in values))


def run_instance(
    instance_name: str,
    config: ALNSConfig,
    seed: int = 42,
    iterations: int = 2000,
):
    """Run ALNS on a single instance and return key metrics."""
    fpath = str(PROJECT_ROOT / "data" / "Instance10" / f"{instance_name}.txt")
    instance = read_instance(fpath, strategy=config.time_window_strategy)
    if 'drone' in instance.vehicle_specs:
        instance.vehicle_specs['drone'].endurance = float('inf')
    instance.configure_robustness(
        drone_battery_capacity=config.drone_battery_capacity,
        energy_uncertainty_budget=config.energy_uncertainty_budget,
        energy_deviation_rate=config.energy_deviation_rate,
        same_truck_retrieval=config.same_truck_retrieval,
    )

    evaluator = Evaluator(
        instance,
        rendezvous_tolerance=config.drone_rendezvous_tolerance,
        cost_lambda=config.cost_lambda,
        cost_rho=config.cost_rho,
        cost_normalized=config.cost_normalized,
    )

    sa_config_dict = config.build_sa_config_dict()
    sa_config_dict['iterations'] = iterations
    sa_config_dict['size'] = 'small'
    sa_cfg = SANNCfg(**sa_config_dict)
    rng = random.Random(seed)

    drone_priority = config.drone_priority
    drone_bonus_kwargs = config.drone_bonus

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
                       drone_priority=drone_priority, **drone_bonus_kwargs),
        RepairDronePriorityRegret(instance, rng=random.Random(seed + 2002),
                                  drone_priority=drone_priority, **drone_bonus_kwargs),
        RepairTruckFirst(instance, rng=random.Random(seed + 2003),
                        drone_priority=drone_priority, **drone_bonus_kwargs),
        RepairEqualPriority(instance, rng=random.Random(seed + 2001),
                           drone_priority=drone_priority, **drone_bonus_kwargs),
    ]

    initial_solution = build_two_phase_initial_solution(instance)
    alns = SimulatedAnnealingALNS(
        instance=instance,
        destroy_ops=destroy_ops,
        repair_ops=repair_ops,
        evaluator=evaluator,
        cfg=sa_cfg,
        rng=rng,
    )
    best_sol = alns.run(initial_solution)
    eval_res = evaluator.evaluate_solution(best_sol)
    return eval_res, evaluator, best_sol


def main():
    config = ALNSConfig(str(PROJECT_ROOT / "config" / "alns_config.yaml"))

    print(f"\nConfig: lambda={config.cost_lambda}, rho={config.cost_rho}, "
          f"normalized={config.cost_normalized}")

    R30_INSTANCES = ["R_30_10_1", "R_30_10_2", "R_30_10_3", "R_30_10_4", "R_30_10_5"]

    print("\n" + "=" * 80)
    print("Running ALNS on R_30_10 instances (2000 iters, seed=42)")
    print("=" * 80)

    results = []
    for inst_name in R30_INSTANCES:
        print(f"\n--- {inst_name} ---")
        start = time.time()
        eval_res, evaluator, best_sol = run_instance(inst_name, config, seed=42, iterations=2000)
        elapsed = time.time() - start

        details = evaluator.evaluate_with_details(best_sol)
        delay_bd = details.delay_breakdown

        delay_by_class = {}
        for nd in delay_bd.nodes:
            cust = evaluator._customer_lookup.get(nd.node_id)
            cls = cust.supply_class if cust else "unknown"
            c = deprivation_cost(nd.delay, cls,
                                  cost_lambda=config.cost_lambda,
                                  rho=config.cost_rho,
                                  normalized=config.cost_normalized)
            delay_by_class.setdefault(cls, []).append((nd.delay, c))

        print(f"  Feasible: {eval_res.feasible}")
        print(f"  Total Cost: {eval_res.total_cost:.2f}")
        print(f"  Truck Dist Cost: {eval_res.truck_distance_cost:.2f}")
        print(f"  Drone Dist Cost: {eval_res.drone_distance_cost:.2f}")
        print(f"  Delay Penalty: {eval_res.delay_penalty:.2f}")
        print(f"  Transport Cost: {eval_res.truck_distance_cost + eval_res.drone_distance_cost:.2f}")
        print(f"  Delay/Transport: {eval_res.delay_penalty / max(1, eval_res.truck_distance_cost + eval_res.drone_distance_cost) * 100:.1f}%")
        print(f"  Total Delays: {len(delay_bd.nodes)}, Violations: {len(delay_bd.violations)}")
        print(f"  Time: {elapsed:.1f}s")

        for cls in DEFAULT_SUPPLY_CLASS_SEQUENCE:
            class_delays = delay_by_class.get(cls, [])
            if class_delays:
                avg_delay = sum(d for d, c in class_delays) / len(class_delays)
                avg_cost = sum(c for d, c in class_delays) / len(class_delays)
                total_cost_cls = sum(c for d, c in class_delays)
                print(f"    {cls:>10}: {len(class_delays):>2} nodes, "
                      f"avg delay={avg_delay:.3f}h, avg cost={avg_cost:.2f}, "
                      f"total cost={total_cost_cls:.2f}")
            else:
                print(f"    {cls:>10}: 0 nodes (all on time)")

        results.append({
            "name": inst_name,
            "feasible": eval_res.feasible,
            "total_cost": eval_res.total_cost,
            "delay_penalty": eval_res.delay_penalty,
            "transport_cost": eval_res.truck_distance_cost + eval_res.drone_distance_cost,
        })

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Instance':>12} {'Feasible':>8} {'Total':>10} {'Delay':>10} {'Transport':>10} {'Delay%':>8}")
    feasible_results = []
    for r in results:
        dp = r["delay_penalty"]
        tc = r["transport_cost"]
        pct = dp / max(1, tc) * 100 if r["feasible"] else float('nan')
        print(f"{r['name']:>12} {str(r['feasible']):>8} {r['total_cost']:>10.2f} "
              f"{dp:>10.2f} {tc:>10.2f} {pct:>8.1f}%")
        if r["feasible"]:
            feasible_results.append(r)

    if feasible_results:
        avg_delay = sum(r["delay_penalty"] for r in feasible_results) / len(feasible_results)
        avg_transport = sum(r["transport_cost"] for r in feasible_results) / len(feasible_results)
        avg_pct = avg_delay / max(1, avg_transport) * 100
        print(f"\n  Avg delay penalty: {avg_delay:.2f}")
        print(f"  Avg transport cost: {avg_transport:.2f}")
        print(f"  Avg delay/transport: {avg_pct:.1f}%")
        print(f"  Feasibility: {len(feasible_results)}/{len(results)}")


if __name__ == "__main__":
    verify_deprivation_cost()
    main()