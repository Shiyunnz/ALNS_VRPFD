"""
稳定性验证：用最优配置在 10 个不同 seed 上重复实验，检验是否能稳定收敛到最优解。
使用 class-based deadline + 贝叶斯调优后的最优配置。

Usage:
    cd code/
    python scripts/stability_verification.py [--seeds N] [--iterations I]
"""
import sys, os, time, random, json, datetime
from pathlib import Path
from copy import deepcopy
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from alns_vrpfd.utils.io_utils import read_instance
from alns_vrpfd.evaluation.evaluator import Evaluator, DelayBreakdown, NodeDelay, TimeWindowViolation
from alns_vrpfd.model.initializer import build_two_phase_initial_solution
from alns_vrpfd.core.operators import (
    DestroyRandom, DestroyShaw, DestroyWorstDistance,
    RepairCheapest, RepairDronePriorityRegret,
    RepairEqualPriority, RepairTruckFirst,
)
from alns_vrpfd.core.sa import SANNCfg, SimulatedAnnealingALNS
from alns_vrpfd.utils.config_loader import ALNSConfig
from alns_vrpfd.deprivation import DEFAULT_SUPPLY_CLASS_SEQUENCE, WANG_SUPPLY_CLASSES, deprivation_cost

INSTANCES = ["R_30_10_1", "R_30_10_2", "R_30_10_3", "R_30_10_4", "R_30_10_5"]
N_SEEDS = 10
ITERATIONS = 4000

CLASS_NAMES = {key: WANG_SUPPLY_CLASSES[key].label for key in DEFAULT_SUPPLY_CLASS_SEQUENCE}

# Best params from Bayesian tuning (class-based)
BEST_PARAMS = {
    "w_percent": 34.18135621483163,
    "cooling_rate_initial": 0.9904822417276974,
    "cooling_rate_final": 0.994863669768,
    "cooling_transition_iters": 388,
    "eta": 0.6740836090955724,
    "alpha_credit": 0.6333249799718762,
    "weight_decay": 0.021797427346036503,
    "r_lower": 0.26273436326943395,
    "r_upper_small": 0.2944935614718466,
    "quota_base_cap": 36,
    "reheat_stall_trigger": 373,
    "reheat_acceptance_window": 136,
    "reheat_acceptance_min": 0.09162315228086913,
    "reheat_duration": 89,
    "reheat_temperature_scale": 0.22485148510586112,
    "reheat_quota_multiplier": 2.462667289138577,
    "reheat_shake_probability": 0.948368193601002,
    "reheat_random_repair_prob": 0.38257585021373836,
    "local_search_frequency": 6,
    "local_search_on_new_best": True,
    "depot_drone_probability": 0.3570080874241234,
    "intensify_frequency": 30,
    "escape_enabled": False,
    "dynamic_cooling_enabled": True,
    "cooling_slowdown_factor": 0.9965499562070199,
    "cooling_speedup_factor": 0.986142231072098,
    "recent_improvement_window": 74,
    "diversification_enabled": True,
    "diversification_trigger_stall": 674,
    "diversification_restart_best_prob": 0.39952737752083567,
    "diversification_destroy_ratio": 0.6097255129157266,
    "adaptive_quota_enabled": True,
    "drone_priority": 2.2850068227231564,
    "depot_bonus": 1.9335179191244873,
    "multi_customer_bonus": 1.0251924070670393,
    "reward_scale": {
        "global": 51.60538929207463,
        "better": 10.49707116869889,
        "slight_better": 12.946367448551914,
        "accepted_worse": 2.774432539475951,
    },
}


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
            cost = deprivation_cost(tau, self._node_classes.get(delay.node_id, "water"))
            total_delay += cost
        return DelayBreakdown(
            total_delay=total_delay,
            nodes=tuple(delays),
            violations=tuple(violations),
        )


def build_sa_config(params, iterations=ITERATIONS):
    config = ALNSConfig(str(PROJECT_ROOT / "config" / "alns_config.yaml"))
    base = config.build_sa_config_dict()
    base.update({
        "iterations": iterations,
        "size": "small",
        "log_operator_metrics": False,
        "w_percent": params["w_percent"],
        "cooling_rate_initial": params["cooling_rate_initial"],
        "cooling_rate_final": params["cooling_rate_final"],
        "cooling_transition_iters": params["cooling_transition_iters"],
        "eta": params["eta"],
        "alpha_credit": params["alpha_credit"],
        "reward_scale": params["reward_scale"],
        "weight_decay": params["weight_decay"],
        "r_lower": params["r_lower"],
        "r_upper_small": params["r_upper_small"],
        "quota_base_cap": params["quota_base_cap"],
        "reheat_stall_trigger": params["reheat_stall_trigger"],
        "reheat_acceptance_window": params["reheat_acceptance_window"],
        "reheat_acceptance_min": params["reheat_acceptance_min"],
        "reheat_duration": params["reheat_duration"],
        "reheat_temperature_scale": params["reheat_temperature_scale"],
        "reheat_quota_multiplier": params["reheat_quota_multiplier"],
        "reheat_shake_probability": params["reheat_shake_probability"],
        "reheat_random_repair_prob": params["reheat_random_repair_prob"],
        "local_search_frequency": params["local_search_frequency"],
        "local_search_on_new_best": params["local_search_on_new_best"],
        "depot_drone_probability": params["depot_drone_probability"],
        "intensify_frequency": params["intensify_frequency"],
        "escape_enabled": params["escape_enabled"],
        "dynamic_cooling_enabled": params["dynamic_cooling_enabled"],
        "diversification_enabled": params["diversification_enabled"],
        "adaptive_quota_enabled": params["adaptive_quota_enabled"],
    })
    if params.get("dynamic_cooling_enabled"):
        base["improvement_threshold"] = 0.01
        base["cooling_slowdown_factor"] = params.get("cooling_slowdown_factor", 0.998)
        base["cooling_speedup_factor"] = params.get("cooling_speedup_factor", 0.980)
        base["recent_improvement_window"] = params.get("recent_improvement_window", 50)
    if params.get("diversification_enabled"):
        base["diversification_trigger_stall"] = params.get("diversification_trigger_stall", 500)
        base["diversification_restart_best_prob"] = params.get("diversification_restart_best_prob", 0.7)
        base["diversification_destroy_ratio"] = params.get("diversification_destroy_ratio", 0.6)
    if not params.get("escape_enabled"):
        base["escape_trigger_stall"] = 100
        base["escape_duration"] = 20
    return SANNCfg(**base)


def run_once(instance_name, seed, iterations=ITERATIONS):
    fpath = str(PROJECT_ROOT / "data" / "Instance10" / f"{instance_name}.txt")
    config = ALNSConfig(str(PROJECT_ROOT / "config" / "alns_config.yaml"))

    rng_tw = np.random.default_rng(seed)
    instance = read_instance(fpath, strategy="class_based", apply_time_windows=False)
    instance.vehicle_specs["drone"].endurance = float("inf")
    instance.configure_robustness(
        drone_battery_capacity=config.drone_battery_capacity,
        energy_uncertainty_budget=config.energy_uncertainty_budget,
        energy_deviation_rate=config.energy_deviation_rate,
        same_truck_retrieval=config.same_truck_retrieval,
    )
    classes = generate_class_based_deadlines(instance, rng_tw)
    evaluator = ClassWeightedEvaluator(
        instance, classes,
        rendezvous_tolerance=config.drone_rendezvous_tolerance,
        forced_drone_customers=config.forced_drone_customers,
        allow_multiple_launch_per_node=config.relax_allow_multiple_launch_per_node,
    )

    sa_cfg = build_sa_config(BEST_PARAMS, iterations=iterations)
    alns_rng = random.Random(seed)

    drone_bonus_kwargs = {
        "depot_bonus": BEST_PARAMS["depot_bonus"],
        "multi_customer_bonus": BEST_PARAMS["multi_customer_bonus"],
        "multi_customer_threshold": 2,
        "wait_max": 20.0,
        "allow_multiple_launch_per_node": config.relax_allow_multiple_launch_per_node,
    }
    dp = BEST_PARAMS["drone_priority"]

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
                       drone_priority=dp, robust_energy_mode="embedded",
                       **drone_bonus_kwargs),
        RepairDronePriorityRegret(instance, rng=random.Random(seed + 2002),
                                   drone_priority=dp, robust_energy_mode="embedded",
                                   **drone_bonus_kwargs),
        RepairTruckFirst(instance, rng=random.Random(seed + 2003),
                         drone_priority=dp, robust_energy_mode="embedded",
                         **drone_bonus_kwargs),
        RepairEqualPriority(instance, rng=random.Random(seed + 2001),
                            drone_priority=dp, robust_energy_mode="embedded",
                            **drone_bonus_kwargs),
    ]

    initial = build_two_phase_initial_solution(instance)
    alns = SimulatedAnnealingALNS(
        instance=instance, destroy_ops=destroy_ops, repair_ops=repair_ops,
        evaluator=evaluator, cfg=sa_cfg, rng=alns_rng,
    )
    start = time.time()
    best = alns.run(initial)
    runtime = time.time() - start
    eval_res = evaluator.evaluate_solution(best)

    return {
        "cost": eval_res.total_cost,
        "truck_cost": eval_res.truck_distance_cost,
        "drone_cost": eval_res.drone_distance_cost,
        "delay_cost": eval_res.delay_penalty,
        "feasible": eval_res.feasible,
        "runtime": runtime,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=N_SEEDS, help="Number of seeds per instance")
    parser.add_argument("--iterations", type=int, default=ITERATIONS, help="ALNS iterations")
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "results" / "stability_verification"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    print(f"{'Instance':<12} {'Seed':>5} {'Cost':>8} {'Truck':>7} {'Drone':>7} {'Delay':>7} {'Feas':>5} {'Time':>6}")
    print("-" * 70)

    for inst in INSTANCES:
        costs = []
        print(f"\n--- {inst} ---")
        for s in range(args.seeds):
            seed = 100 + s
            r = run_once(inst, seed, args.iterations)
            costs.append(r["cost"])
            tag = "OK" if r["feasible"] else "INFEASIBLE"
            print(f"  {inst:<12} {seed:>5} {r['cost']:>8.2f} {r['truck_cost']:>7.2f} "
                  f"{r['drone_cost']:>7.2f} {r['delay_cost']:>7.2f} {tag:>5} {r['runtime']:>5.1f}s")

        costs_arr = np.array(costs)
        infeasible_count = sum(1 for c in costs if c >= 1e6)
        feas_costs = costs_arr[costs_arr < 1e6]
        print(f"\n  Summary for {inst}:")
        print(f"    mean={np.mean(feas_costs):.2f}  std={np.std(feas_costs):.2f}  "
              f"min={np.min(feas_costs):.2f}  max={np.max(feas_costs):.2f}  "
              f"gap={((np.max(feas_costs)-np.min(feas_costs))/np.min(feas_costs)*100):.1f}%  "
              f"feasible={len(feas_costs)}/{args.seeds}")
        all_results[inst] = {
            "costs": costs,
            "mean": float(np.mean(feas_costs)),
            "std": float(np.std(feas_costs)),
            "min": float(np.min(feas_costs)),
            "max": float(np.max(feas_costs)),
            "gap_pct": float((np.max(feas_costs) - np.min(feas_costs)) / np.min(feas_costs) * 100),
            "feasible_count": len(feas_costs),
            "total_seeds": args.seeds,
        }

    print(f"\n{'='*70}")
    print("  OVERALL STABILITY SUMMARY")
    print(f"{'='*70}")
    print(f"{'Instance':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Gap%':>7} {'Feas':>5}")
    print("-" * 55)
    all_means = []
    for inst in INSTANCES:
        r = all_results[inst]
        print(f"{inst:<12} {r['mean']:>8.2f} {r['std']:>8.2f} {r['min']:>8.2f} "
              f"{r['max']:>8.2f} {r['gap_pct']:>6.1f}% {r['feasible_count']:>3}/{r['total_seeds']}")
        all_means.append(r["mean"])
    print(f"\n  Overall mean={np.mean(all_means):.2f}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"stability_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved → {out_path}")
