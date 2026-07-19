"""
ALNS vs MILP comparison script.
Runs both algorithms on the same instance with matching configuration,
saves run records, and prints a side-by-side summary.
"""
import sys
import time
import random
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from alns_vrpfd.utils.io_utils import read_instance
from alns_vrpfd.utils.config_loader import ALNSConfig
from alns_vrpfd.evaluation.evaluator import Evaluator
from alns_vrpfd.evaluation.run_record import save_run_record, reconstruct_solution_from_mip


def run_alns(instance, evaluator, config, seed, iterations):
    import time
    import random
    from copy import deepcopy
    from alns_vrpfd.core.sa import SANNCfg, SimulatedAnnealingALNS

    cfg = config
    single_truck = cfg.raw.get("single_truck", False)

    from run_alns import build_operators, infer_size

    sa_config_dict = cfg.build_sa_config_dict()
    sa_config_dict['iterations'] = iterations
    sa_config_dict['log_operator_metrics'] = False
    sa_config_dict['size'] = infer_size(instance)
    sa_cfg = SANNCfg(**sa_config_dict)
    rng = random.Random(seed)

    forced_drone_customers = cfg.forced_drone_customers
    drone_bonus_kwargs = cfg.drone_bonus

    destroy_ops, repair_ops = build_operators(
        instance, seed, drone_priority=cfg.drone_priority,
        repair_set="all", enable_composite=True,
        drone_bonus_kwargs=drone_bonus_kwargs,
        forced_drone_customers=forced_drone_customers,
        robust_energy_mode="embedded",
    )

    use_two_phase = cfg.raw.get("initial_solution", {}).get("two_phase", True)
    from alns_vrpfd.model.initializer import build_two_phase_initial_solution, build_initial_solution

    if use_two_phase:
        initial_solution = build_two_phase_initial_solution(
            instance,
            truck_forbidden_customers=forced_drone_customers,
            allow_multiple_launch_per_node=cfg.relax_allow_multiple_launch_per_node,
        )
    else:
        initial_solution = build_initial_solution(
            instance,
            truck_forbidden_customers=forced_drone_customers,
            allow_multiple_launch_per_node=cfg.relax_allow_multiple_launch_per_node,
        )

    alns = SimulatedAnnealingALNS(
        instance=instance,
        destroy_ops=destroy_ops,
        repair_ops=repair_ops,
        evaluator=evaluator,
        cfg=sa_cfg,
        rng=rng,
    )

    start = time.perf_counter()
    best_sol = alns.run(initial_solution)
    runtime = time.perf_counter() - start

    return best_sol, runtime


def run_mip(instance, evaluator, config, energy_budget=3, time_limit=600):
    import time
    from alns_vrpfd.mip.builder import build_mip_model

    cfg = config
    start = time.perf_counter()

    artifacts = build_mip_model(
        instance,
        epsilon=1e-3,
        energy_budget=energy_budget,
        num_segments=10,
        use_gurobi_pwl=True,
        robust_energy=True,
        big_m_time=1000,
        big_m_load=1000,
        big_m_energy=20.0,
        tardiness_weight=1.0,
        cost_lambda=cfg.cost_lambda,
        cost_rho=cfg.cost_rho,
        cost_normalized=cfg.cost_normalized,
    )

    model = artifacts.model
    model.setParam("TimeLimit", time_limit)
    model.setParam("Threads", 4)
    model.setParam("OutputFlag", 0)
    model.optimize()

    runtime = time.perf_counter() - start
    return artifacts, runtime


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ALNS vs MILP comparison")
    parser.add_argument("instance", type=str, help="Instance file path")
    parser.add_argument("--alns-iters", type=int, default=4000, help="ALNS iterations")
    parser.add_argument("--mip-timelimit", type=int, default=600, help="MILP time limit (s)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--energy-budget", type=int, default=3, help="Energy uncertainty budget")
    args = parser.parse_args()

    instance_path = Path(args.instance)
    seed = args.seed
    instance_name = instance_path.stem

    print(f"=== ALNS vs MILP Comparison: {instance_name} ===")
    print(f"ALNS iterations: {args.alns_iters}, MILP time limit: {args.mip_timelimit}s")
    print()

    # --- Load config & instance ---
    config = ALNSConfig("config/alns_config.yaml")
    instance = read_instance(str(instance_path), strategy=config.time_window_strategy)
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
        forced_drone_customers=config.forced_drone_customers,
        allow_multiple_launch_per_node=config.relax_allow_multiple_launch_per_node,
        cost_lambda=config.cost_lambda,
        cost_rho=config.cost_rho,
        cost_normalized=config.cost_normalized,
    )

    record_config = {
        "drone_battery_capacity": config.drone_battery_capacity,
        "energy_uncertainty_budget": config.energy_uncertainty_budget,
        "energy_deviation_rate": config.energy_deviation_rate,
        "cost_lambda": config.cost_lambda,
        "cost_rho": config.cost_rho,
        "cost_normalized": config.cost_normalized,
        "alns_iterations": args.alns_iters,
        "mip_time_limit": args.mip_timelimit,
        "seed": seed,
    }

    # --- Run ALNS ---
    print("--- ALNS ---")
    alns_sol, alns_runtime = run_alns(instance, evaluator, config, seed, args.alns_iters)
    alns_details = evaluator.evaluate_with_details(alns_sol)
    alns_record_path = save_run_record(
        instance=instance,
        algorithm="alns",
        solution=alns_sol,
        details=alns_details,
        runtime_seconds=alns_runtime,
        config=record_config,
        seed=seed,
        instance_name=instance_name,
    )
    alns_cost = alns_details.result.total_cost
    alns_energy_ok = alns_details.robustness.feasible
    print(f"  Cost: {alns_cost:.2f}")
    print(f"  Trucks: {len(alns_sol.truck_routes)}, Drones: {len({t.drone_id for t in alns_sol.drone_tasks})}")
    print(f"  Energy feasible: {alns_energy_ok}")
    print(f"  Runtime: {alns_runtime:.2f}s")
    print(f"  Record: {alns_record_path}")
    print()

    # --- Run MILP ---
    print("--- MILP ---")
    artifacts, mip_runtime = run_mip(instance, evaluator, config, args.energy_budget, args.mip_timelimit)
    mip_sol = reconstruct_solution_from_mip(artifacts)

    if mip_sol is not None:
        mip_details = evaluator.evaluate_with_details(mip_sol)
        mip_record_path = save_run_record(
            instance=instance,
            algorithm="mip",
            solution=mip_sol,
            details=mip_details,
            runtime_seconds=mip_runtime,
            config=record_config,
            seed=seed,
            instance_name=instance_name,
        )
        mip_cost = mip_details.result.total_cost
        mip_energy_ok = mip_details.robustness.feasible
        print(f"  Cost: {mip_cost:.2f}")
        print(f"  Trucks: {len(mip_sol.truck_routes)}, Drones: {len({t.drone_id for t in mip_sol.drone_tasks})}")
        print(f"  Energy feasible: {mip_energy_ok}")
        print(f"  Solver status: {artifacts.model.Status}")
        print(f"  Runtime: {mip_runtime:.2f}s")
        print(f"  Record: {mip_record_path}")
    else:
        mip_cost = None
        mip_energy_ok = None
        mip_record_path = None
        print(f"  No feasible solution found (status: {artifacts.model.Status})")
    print()

    # --- Comparison ---
    print("=== Comparison ===")
    print(f"{'Metric':<25} {'ALNS':<12} {'MILP':<12}")
    print("-" * 49)
    print(f"{'Total Cost':<25} {alns_cost:<12.2f} {mip_cost if mip_cost else 'N/A':<12}")
    print(f"{'Gap (MILP-ALNS)/ALNS':<25} {'-':<12} {(mip_cost - alns_cost) / alns_cost * 100 if mip_cost else 'N/A':<10.2f}%")
    print(f"{'Trucks':<25} {len(alns_sol.truck_routes):<12} {len(mip_sol.truck_routes) if mip_sol else 'N/A':<12}")
    print(f"{'Energy Feasible':<25} {str(alns_energy_ok):<12} {str(mip_energy_ok) if mip_energy_ok is not None else 'N/A':<12}")
    print(f"{'Runtime (s)':<25} {alns_runtime:<12.2f} {mip_runtime:<12.2f}")

    print()
    print("ALNS route record:", alns_record_path)
    if mip_record_path:
        print("MIP route record:", mip_record_path)


if __name__ == "__main__":
    main()
