"""Run ALNS with N seeds on instances where ALNS was ≳ MILP, to test convergence."""
import sys, json, random, time, argparse
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from alns_vrpfd.utils.io_utils import read_instance
from alns_vrpfd.utils.config_loader import ALNSConfig
from alns_vrpfd.evaluation.evaluator import Evaluator
from alns_vrpfd.core.sa import SANNCfg, SimulatedAnnealingALNS
from alns_vrpfd.model.initializer import build_two_phase_initial_solution
from run_alns import build_operators, infer_size

WEAK_INSTANCES = [
    "data/Instance10/R_30_10_2.txt",
    "data/Instance10/R_30_10_3.txt",
    "data/Instance10/R_30_10_4.txt",
    "data/Instance10/R_30_10_5.txt",
    "data/Instance10/R_40_10_1.txt",
    "data/Instance10/R_40_10_2.txt",
    "data/Instance10/R_40_10_3.txt",
    "data/Instance10/R_40_10_4.txt",
    "data/Instance10/R_50_10_1.txt",
    "data/Instance10/R_50_10_2.txt",
    "data/Instance10/R_50_10_3.txt",
]

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, default=15)
parser.add_argument("--iters", type=int, default=2000)
args = parser.parse_args()

config = ALNSConfig("config/alns_config.yaml")

results = {}
for inst_path in WEAK_INSTANCES:
    tag = Path(inst_path).stem
    print(f"\n{'='*50}")
    print(f"[{tag}] {args.seeds} seeds × {args.iters} iters")
    print(f"{'='*50}")

    instance = read_instance(inst_path, strategy="class_based")
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

    sa_config = config.build_sa_config_dict()
    sa_config["iterations"] = args.iters
    sa_config["size"] = infer_size(instance)
    sa_config["drone_reanchor_ls_enabled"] = True

    initial = build_two_phase_initial_solution(
        instance,
        truck_forbidden_customers=config.forced_drone_customers,
        allow_multiple_launch_per_node=config.relax_allow_multiple_launch_per_node,
    )

    costs = []
    best_cost = float('inf')
    best_seed = None
    total_time = 0.0

    for seed in range(args.seeds):
        sa_cfg = SANNCfg(**sa_config)
        rng = random.Random(seed)

        destroy_ops, repair_ops = build_operators(
            instance, seed,
            drone_priority=config.drone_priority,
            repair_set="all", enable_composite=True,
            drone_bonus_kwargs=config.drone_bonus,
            forced_drone_customers=config.forced_drone_customers,
            robust_energy_mode="embedded",
        )

        alns = SimulatedAnnealingALNS(
            instance=instance, destroy_ops=destroy_ops,
            repair_ops=repair_ops, evaluator=evaluator,
            cfg=sa_cfg, rng=rng,
        )

        t0 = time.perf_counter()
        best_sol = alns.run(initial)
        elapsed = time.perf_counter() - t0

        ev = evaluator.evaluate_solution(best_sol)
        if ev.feasible and ev.total_cost < best_cost:
            best_cost = ev.total_cost
            best_seed = seed
        if ev.feasible:
            costs.append(ev.total_cost)
        total_time += elapsed

    if costs:
        mean_cost = sum(costs) / len(costs)
        min_cost = min(costs)
    else:
        mean_cost = float('inf')
        min_cost = float('inf')

    results[tag] = {
        "best_cost": round(min_cost, 4),
        "mean_cost": round(mean_cost, 4),
        "best_seed": best_seed,
        "total_runs": args.seeds,
        "feasible_runs": len(costs),
        "total_time_s": round(total_time, 1),
    }

    print(f"  best={min_cost:.2f} (seed {best_seed}), mean={mean_cost:.2f}, "
          f"feasible={len(costs)}/{args.seeds}, time={total_time:.0f}s")

OUT = Path("results") / "alns_multi_seed_weak.json"
with OUT.open("w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT}")
