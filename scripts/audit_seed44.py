#!/usr/bin/env python3
"""Audit seed 44 solution from D group: full structure analysis."""

import sys, json, random, math
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from alns_vrpfd.utils.io_utils import read_instance
from alns_vrpfd.utils.config_loader import ALNSConfig
from alns_vrpfd.evaluation.evaluator import Evaluator
from alns_vrpfd.core.sa import SANNCfg, SimulatedAnnealingALNS
from alns_vrpfd.model.initializer import build_two_phase_initial_solution
from run_alns import build_operators, infer_size

from alns_vrpfd.core.operators.matheuristic_lns import MatheuristicLNSRepair
from alns_vrpfd.core.operators.truck_drone_rechain import TruckDroneRechainRepair

INSTANCE = "data/Instance10/R_30_10_2.txt"
SEED = 44


def audit_solution(sol, evaluator, instance, label):
    ev = evaluator.evaluate_solution(sol)
    print(f"\n{'='*70}")
    print(f"  {label} — Cost: {ev.total_cost:.4f}, Feasible: {ev.feasible}")
    print(f"{'='*70}")

    print(f"\n  Truck distance cost: {ev.truck_distance_cost:.4f}")
    print(f"  Drone distance cost: {ev.drone_distance_cost:.4f}")
    print(f"  Delay penalty:       {ev.delay_penalty:.4f}")
    print(f"  Total:               {ev.total_cost:.4f}")

    print(f"\n  --- Truck Routes ---")
    customer_ids = set(instance.customer_manager.customer_ids())
    for r in sol.truck_routes:
        demand = sum(
            instance.customer_manager.demands().get(n, 0)
            for n in r.nodes if n in customer_ids
        )
        print(f"  T{r.id}: {r.nodes}  (demand={demand:.1f}/{r.capacity:.0f})")

    print(f"\n  --- Drone Tasks ---")
    rob_cfg = instance.robust_config
    from alns_vrpfd.evaluation.energy import DroneEnergyModel
    node_index = {n: i for i, n in enumerate(instance.all_node_ids())}
    drone_energy_model = DroneEnergyModel()
    drone_time = instance.time_matrix("drone")
    battery_cap = rob_cfg.drone_battery_capacity

    for t in sol.drone_tasks:
        cs = t.customers()
        launch = f"T{t.launch_truck}@{t.launch_node}" if t.launch_truck is not None else f"@{t.launch_node}"
        land = f"T{t.land_truck}@{t.retrieve_node}" if t.land_truck is not None else f"@{t.retrieve_node}"
        print(f"  D{t.drone_id}: {launch} -> {cs} -> {land}")

        payloads = []
        running = 0.0
        demands_dict = instance.customer_manager.demands()
        for c in cs:
            p = max(0.001, running + demands_dict.get(c, 0))
            payloads.append(p)
            running = p - demands_dict.get(c, 0)

        nodes_seq = [t.launch_node] + cs + [t.retrieve_node]
        nom_energy = 0.0
        deviations = []
        for seg_i in range(len(nodes_seq) - 1):
            a, b = nodes_seq[seg_i], nodes_seq[seg_i + 1]
            try:
                flight_t = drone_time[node_index[a]][node_index[b]]
            except (KeyError, IndexError):
                flight_t = float('inf')
            p = payloads[seg_i] if seg_i < len(payloads) else 0.001
            e = drone_energy_model.energy_kwh(p, flight_t)
            nom_energy += e
            deviations.append(e * rob_cfg.energy_deviation_rate)

        budgeted = sum(deviations)
        robust_energy = nom_energy + budgeted
        margin = battery_cap - robust_energy
        print(f"    Nominal energy: {nom_energy:.3f} kWh, "
              f"Robust: {robust_energy:.3f} kWh, "
              f"Margin: {margin:+.3f} kWh")
        print(f"    Battery cap: {battery_cap:.3f} kWh, "
              f"{'OK' if margin >= 0 else 'VIOLATION'}")


def main():
    instance = read_instance(INSTANCE)
    evaluator = Evaluator(instance)
    config = ALNSConfig("config/alns_config.yaml")

    rng = random.Random(SEED)

    destroy_ops, repair_ops = build_operators(
        instance, SEED,
        drone_priority=config.drone_priority,
        repair_set="all",
        enable_composite=True,
        drone_bonus_kwargs=config.drone_bonus,
        forced_drone_customers=config.forced_drone_customers,
        robust_energy_mode="embedded",
    )

    initial = build_two_phase_initial_solution(
        instance,
        truck_forbidden_customers=config.forced_drone_customers,
        allow_multiple_launch_per_node=config.relax_allow_multiple_launch_per_node,
    )

    sa_config = config.build_sa_config_dict()
    sa_config["iterations"] = 4000
    sa_config["size"] = infer_size(instance)
    sa_config["drone_reanchor_ls_enabled"] = True
    sa_config["drone_composite_reanchor_enabled"] = False
    sa_config["drone_sortie_constructor_enabled"] = False
    sa_config["matheuristic_lns_enabled"] = True
    sa_cfg = SANNCfg(**sa_config)

    alns = SimulatedAnnealingALNS(
        instance=instance, destroy_ops=destroy_ops, repair_ops=repair_ops,
        evaluator=evaluator, cfg=sa_cfg, rng=rng,
    )

    print("Running ALNS with Step6 + MLNS for seed 44...")
    best_sol = alns.run(initial)

    audit_solution(best_sol, evaluator, instance, "ALNS + Step6 + MLNS (before Rechain)")

    print("\n\nApplying MLNS polish (5 trials)...")
    best_cost = evaluator.evaluate_solution(best_sol).total_cost
    original_cost = best_cost
    for trial in range(5):
        lns = MatheuristicLNSRepair(
            instance=instance, evaluator=evaluator,
            max_customers=3, max_anchor_dist_factor=2.0,
            energy_tolerance=1.0,
            rng=random.Random(rng.randint(0, 2**31)),
        )
        improved = lns.apply(best_sol)
        imp_cost = evaluator.evaluate_solution(improved).total_cost
        if math.isfinite(imp_cost) and imp_cost < best_cost - 1e-6:
            print(f"  MLNS trial {trial}: {best_cost:.4f} -> {imp_cost:.4f}")
            best_sol = improved
            best_cost = imp_cost

    audit_solution(best_sol, evaluator, instance, "After MLNS polish (before Rechain)")

    print("\n\nApplying Rechain polish (5 trials)...")
    best_cost = evaluator.evaluate_solution(best_sol).total_cost
    original_rechain = best_cost
    for trial in range(5):
        rechain = TruckDroneRechainRepair(
            instance=instance, evaluator=evaluator,
            max_customers=3, max_anchor_dist_factor=2.0,
            energy_tolerance=1.0, max_candidates=5000,
            rng=random.Random(rng.randint(0, 2**31)),
        )
        improved = rechain.apply(best_sol)
        imp_cost = evaluator.evaluate_solution(improved).total_cost
        if math.isfinite(imp_cost) and imp_cost < best_cost - 1e-6:
            print(f"  Rechain trial {trial}: {best_cost:.4f} -> {imp_cost:.4f}")
            print(f"    Rechain attempts={rechain.attempts}, created={rechain.created}, accepted={rechain.accepted}")
            best_sol = improved
            best_cost = imp_cost

    audit_solution(best_sol, evaluator, instance, "FINAL (after Rechain)")

    print("\n\n=== Oracle 53.02 comparison ===")
    print("  Oracle target: T0:[0,1,11] + T0:[0,3,5,11]")
    print("  Found (55.76): T0:[0,5,3,11] + T0:[0,1,11]")
    print("  Diff: Truck 0 order is [0,5,3,11] vs oracle [0,3,5,11]")
    print("  Key question: is [0,5,3,11] vs [0,3,5,11] significant for drone sorties?")


if __name__ == "__main__":
    main()