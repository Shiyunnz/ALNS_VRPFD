"""
对比贝叶斯最优配置 vs 默认配置的路线差异。
在 5 个实例上分别运行，输出卡车路线 + 无人机任务详细对比。
"""
import sys, os, math, time, random, json
from pathlib import Path
from copy import deepcopy

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from alns_vrpfd.utils.io_utils import read_instance
from alns_vrpfd.evaluation.evaluator import Evaluator, EvaluationResult
from alns_vrpfd.model.initializer import build_two_phase_initial_solution
from alns_vrpfd.core.operators import (
    DestroyRandom, DestroyShaw, DestroyWorstDistance,
    RepairCheapest, RepairDronePriorityRegret,
    RepairEqualPriority, RepairTruckFirst,
)
from alns_vrpfd.core.sa import SANNCfg, SimulatedAnnealingALNS
from alns_vrpfd.utils.config_loader import ALNSConfig

INSTANCES = ["R_30_10_1", "R_30_10_2", "R_30_10_3", "R_30_10_4", "R_30_10_5"]
SEED = 42
ITERATIONS = 4000

BEST_PARAMS = {
    "w_percent": 19.548073458706902,
    "cooling_rate_initial": 0.9962659775135407,
    "cooling_rate_final": 0.9712912290677784,
    "cooling_transition_iters": 1373,
    "eta": 0.7849097093431012,
    "alpha_credit": 0.6742186637558922,
    "weight_decay": 0.003572834658417511,
    "r_lower": 0.18923092120644835,
    "r_upper_small": 0.49245284258236205,
    "quota_base_cap": 24,
    "reheat_stall_trigger": 596,
    "reheat_acceptance_window": 94,
    "reheat_acceptance_min": 0.018144113680048563,
    "reheat_duration": 69,
    "reheat_temperature_scale": 0.11958240622937452,
    "reheat_quota_multiplier": 2.2136012436359565,
    "reheat_shake_probability": 0.9439833549512975,
    "reheat_random_repair_prob": 0.4985010479742863,
    "local_search_frequency": 12,
    "local_search_on_new_best": True,
    "depot_drone_probability": 0.06795743377732272,
    "intensify_frequency": 49,
    "escape_enabled": True,
    "escape_trigger_stall": 51,
    "escape_duration": 34,
    "dynamic_cooling_enabled": False,
    "diversification_enabled": False,
    "adaptive_quota_enabled": False,
    "drone_priority": 3.1499059459016205,
    "depot_bonus": 1.1341856120022966,
    "multi_customer_bonus": 3.2958198415429667,
    "reward_scale": {
        "global": 33.46493728311663,
        "better": 15.291289877515922,
        "slight_better": 7.427667844505805,
        "accepted_worse": 3.4975205369293265,
    },
}


def build_tuned_config(iterations=ITERATIONS):
    config = ALNSConfig(str(PROJECT_ROOT / "config" / "alns_config.yaml"))
    base = config.build_sa_config_dict()
    base.update({
        "iterations": iterations,
        "size": "small",
        "log_operator_metrics": False,
        "w_percent": BEST_PARAMS["w_percent"],
        "cooling_rate_initial": BEST_PARAMS["cooling_rate_initial"],
        "cooling_rate_final": BEST_PARAMS["cooling_rate_final"],
        "cooling_transition_iters": BEST_PARAMS["cooling_transition_iters"],
        "eta": BEST_PARAMS["eta"],
        "alpha_credit": BEST_PARAMS["alpha_credit"],
        "reward_scale": BEST_PARAMS["reward_scale"],
        "weight_decay": BEST_PARAMS["weight_decay"],
        "r_lower": BEST_PARAMS["r_lower"],
        "r_upper_small": BEST_PARAMS["r_upper_small"],
        "quota_base_cap": BEST_PARAMS["quota_base_cap"],
        "reheat_stall_trigger": BEST_PARAMS["reheat_stall_trigger"],
        "reheat_acceptance_window": BEST_PARAMS["reheat_acceptance_window"],
        "reheat_acceptance_min": BEST_PARAMS["reheat_acceptance_min"],
        "reheat_duration": BEST_PARAMS["reheat_duration"],
        "reheat_temperature_scale": BEST_PARAMS["reheat_temperature_scale"],
        "reheat_quota_multiplier": BEST_PARAMS["reheat_quota_multiplier"],
        "reheat_shake_probability": BEST_PARAMS["reheat_shake_probability"],
        "reheat_random_repair_prob": BEST_PARAMS["reheat_random_repair_prob"],
        "local_search_frequency": BEST_PARAMS["local_search_frequency"],
        "local_search_on_new_best": BEST_PARAMS["local_search_on_new_best"],
        "depot_drone_probability": BEST_PARAMS["depot_drone_probability"],
        "intensify_frequency": BEST_PARAMS["intensify_frequency"],
        "escape_enabled": BEST_PARAMS["escape_enabled"],
        "escape_trigger_stall": BEST_PARAMS["escape_trigger_stall"],
        "escape_duration": BEST_PARAMS["escape_duration"],
        "dynamic_cooling_enabled": BEST_PARAMS["dynamic_cooling_enabled"],
        "diversification_enabled": BEST_PARAMS["diversification_enabled"],
        "adaptive_quota_enabled": BEST_PARAMS["adaptive_quota_enabled"],
    })
    return SANNCfg(**base)


def build_default_config(iterations=ITERATIONS):
    config = ALNSConfig(str(PROJECT_ROOT / "config" / "alns_config.yaml"))
    sa_config_dict = config.build_sa_config_dict()
    sa_config_dict["iterations"] = iterations
    sa_config_dict["size"] = "small"
    sa_config_dict["log_operator_metrics"] = False
    return SANNCfg(**sa_config_dict)


def run_alns(instance_name, cfg, label, drone_priority=2.2, drone_bonus_kwargs=None, seed=SEED):
    fpath = str(PROJECT_ROOT / "data" / "Instance10" / f"{instance_name}.txt")
    config = ALNSConfig(str(PROJECT_ROOT / "config" / "alns_config.yaml"))
    instance = read_instance(fpath, strategy="class_based")
    instance.vehicle_specs["drone"].endurance = float("inf")
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
    )

    rng = random.Random(seed)
    if drone_bonus_kwargs is None:
        drone_bonus_kwargs = config.drone_bonus
    # Remove drone_priority from bonus kwargs if present (passed separately)
    bonus = {k: v for k, v in drone_bonus_kwargs.items() if k != "drone_priority"}

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
                       drone_priority=drone_priority,
                       robust_energy_mode="embedded", **bonus),
        RepairDronePriorityRegret(instance, rng=random.Random(seed + 2002),
                                   drone_priority=drone_priority,
                                   robust_energy_mode="embedded", **bonus),
        RepairTruckFirst(instance, rng=random.Random(seed + 2003),
                         drone_priority=drone_priority,
                         robust_energy_mode="embedded", **bonus),
        RepairEqualPriority(instance, rng=random.Random(seed + 2001),
                            drone_priority=drone_priority,
                            robust_energy_mode="embedded", **bonus),
    ]

    initial = build_two_phase_initial_solution(instance)
    alns = SimulatedAnnealingALNS(
        instance=instance, destroy_ops=destroy_ops, repair_ops=repair_ops,
        evaluator=evaluator, cfg=cfg, rng=rng,
    )
    start = time.time()
    best = alns.run(initial)
    runtime = time.time() - start

    det = evaluator.evaluate_with_details(best)
    res = det.result

    return {
        "label": label,
        "cost": res.total_cost,
        "truck_cost": res.truck_distance_cost,
        "drone_cost": res.drone_distance_cost,
        "delay_cost": res.delay_penalty,
        "feasible": res.feasible,
        "runtime": runtime,
        "truck_routes": [
            {"id": tr.id, "nodes": tr.nodes} for tr in best.truck_routes
        ],
        "drone_tasks": [
            {
                "drone_id": dt.drone_id,
                "nodes": dt.nodes,
                "launch_node": dt.launch_node,
                "retrieve_node": dt.retrieve_node,
                "launch_truck": dt.launch_truck,
                "land_truck": dt.land_truck,
            }
            for dt in best.drone_tasks
        ],
        "delay_nodes": [(d.node_id, d.delay, d.served_by) for d in det.delay_breakdown.nodes],
        "violations": [(v.node_id, v.arrival_time, v.latest_time) for v in det.delay_breakdown.violations],
    }


def cfg_extra_param(kwargs, key, default):
    if key in kwargs:
        return kwargs[key]
    return default


def extract_route_summary(result):
    lines = []
    for tr in result["truck_routes"]:
        lines.append(f"  T{tr['id']}: {' → '.join(str(n) for n in tr['nodes'])}")
    for dt in result["drone_tasks"]:
        launch = f"T{dt['launch_truck']}@{dt['launch_node']}" if dt['launch_truck'] is not None else f"Depot@{dt['launch_node']}"
        land = f"T{dt['land_truck']}@{dt['retrieve_node']}" if dt['land_truck'] is not None else f"Depot@{dt['retrieve_node']}"
        lines.append(f"  D{dt['drone_id']}: {dt['nodes']} ({launch} → {land})")
    return "\n".join(lines)


if __name__ == "__main__":
    default_cfg = build_default_config()
    tuned_cfg = build_tuned_config()

    for inst in INSTANCES:
        print(f"\n{'='*80}")
        print(f"  INSTANCE: {inst}")
        print(f"{'='*80}")

        # --- Default config ---
        config = ALNSConfig(str(PROJECT_ROOT / "config" / "alns_config.yaml"))
        default_bonus = config.drone_bonus
        def_res = run_alns(inst, default_cfg, "DEFAULT", drone_priority=2.2)
        print(f"\n  DEFAULT: cost={def_res['cost']:.2f}  truck={def_res['truck_cost']:.2f}  "
              f"drone={def_res['drone_cost']:.2f}  delay={def_res['delay_cost']:.2f}  "
              f"feasible={def_res['feasible']}  ({def_res['runtime']:.1f}s)")
        print(f"  Routes ({len(def_res['truck_routes'])} trucks, {len(def_res['drone_tasks'])} drones):")
        print(extract_route_summary(def_res))
        if def_res['delay_nodes']:
            print(f"  Delayed nodes: {[(n, f'{d:.3f}h', s) for n, d, s in def_res['delay_nodes']]}")
        if def_res['violations']:
            print(f"  TW violations: {len(def_res['violations'])}")

        # --- Tuned config ---
        tuned_bonus = {
            "depot_bonus": BEST_PARAMS["depot_bonus"],
            "multi_customer_bonus": BEST_PARAMS["multi_customer_bonus"],
            "multi_customer_threshold": 2,
            "wait_max": 20.0,
        }
        tuned_bonus["drone_priority"] = BEST_PARAMS["drone_priority"]
        tuned_res = run_alns(inst, tuned_cfg, "TUNED", drone_priority=BEST_PARAMS["drone_priority"], drone_bonus_kwargs=tuned_bonus)
        print(f"\n  TUNED:  cost={tuned_res['cost']:.2f}  truck={tuned_res['truck_cost']:.2f}  "
              f"drone={tuned_res['drone_cost']:.2f}  delay={tuned_res['delay_cost']:.2f}  "
              f"feasible={tuned_res['feasible']}  ({tuned_res['runtime']:.1f}s)")
        print(f"  Routes ({len(tuned_res['truck_routes'])} trucks, {len(tuned_res['drone_tasks'])} drones):")
        print(extract_route_summary(tuned_res))
        if tuned_res['delay_nodes']:
            print(f"  Delayed nodes: {[(n, f'{d:.3f}h', s) for n, d, s in tuned_res['delay_nodes']]}")
        if tuned_res['violations']:
            print(f"  TW violations: {len(tuned_res['violations'])}")

        # --- Comparison ---
        cost_diff = tuned_res['cost'] - def_res['cost']
        pct = (cost_diff / def_res['cost'] * 100) if def_res['cost'] != 0 else 0
        print(f"\n  DIFF: cost={cost_diff:+.2f} ({pct:+.1f}%)  "
              f"truck={tuned_res['truck_cost']-def_res['truck_cost']:+.2f}  "
              f"drone={tuned_res['drone_cost']-def_res['drone_cost']:+.2f}  "
              f"delay={tuned_res['delay_cost']-def_res['delay_cost']:+.2f}")

        # Same routes check
        def_routes = set(tuple(tr['nodes']) for tr in def_res['truck_routes'])
        tuned_routes = set(tuple(tr['nodes']) for tr in tuned_res['truck_routes'])
        same = def_routes == tuned_routes
        def_drones = set((tuple(dt['nodes']), dt['launch_node'], dt['retrieve_node']) for dt in def_res['drone_tasks'])
        tuned_drones = set((tuple(dt['nodes']), dt['launch_node'], dt['retrieve_node']) for dt in tuned_res['drone_tasks'])
        same_d = def_drones == tuned_drones
        print(f"  Same truck routes: {same}  |  Same drone tasks: {same_d}")
        if not same:
            only_def = def_routes - tuned_routes
            only_tuned = tuned_routes - def_routes
            if only_def:
                print(f"    Only in DEFAULT: {only_def}")
            if only_tuned:
                print(f"    Only in TUNED:  {only_tuned}")
        if not same_d:
            only_def = def_drones - tuned_drones
            only_tuned = tuned_drones - def_drones
            if only_def:
                print(f"    Only in DEFAULT drones: {only_def}")
            if only_tuned:
                print(f"    Only in TUNED drones:  {only_tuned}")