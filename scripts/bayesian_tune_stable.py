"""
Round 3 Bayesian Optimization for ALNS with nonlinear deprivation cost.

Key changes from Round 2:
  - Nonlinear deprivation cost: f(τ) = exp(1.5031 + 7.032τ) - exp(1.5031)
  - Class-weighted deprivation: κ·f(τ) where κ depends on urgency class
  - Reward parameters scaled up to match new cost magnitudes
  - Wider reward hierarchy search range (σ₁: 200-2000)
  - Temperature search adjusted for larger cost range
  - Multi-seed stability objective retained

Usage:
    cd code/
    python scripts/bayesian_tune_stable.py [--phase {1,2,3,all}] [--trials N]
"""
import sys, os, time, random, json, datetime, logging, argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import optuna
from optuna.samplers import TPESampler

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

ResultT = Dict[str, Any]

PHASE1_INSTANCES = ["R_30_10_1", "R_30_10_3"]
PHASE2_INSTANCES = ["R_30_10_1", "R_30_10_3", "R_30_10_5"]
ALL_INSTANCES = ["R_30_10_1", "R_30_10_2", "R_30_10_3", "R_30_10_4", "R_30_10_5"]
INFEASIBLE_PENALTY = 1e6

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
            total_delay=total_delay, nodes=tuple(delays), violations=tuple(violations))


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


def suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Round 3: Reward hierarchy scaled for nonlinear deprivation cost.

    Nonlinear f(τ) = exp(1.5031 + 7.032τ) - exp(1.5031) produces costs
    in the range ~4-5000 for τ∈[0.05, 1.0]. Each κ-weighted delay can
    reach hundreds to thousands. This requires σ₁ in hundreds range to
    meaningfully reward good operator behaviour.
    """
    params = {}

    # === SA: Initial temperature scaled up for new cost magnitudes ===
    params["w_percent"] = trial.suggest_float("w_percent", 5.0, 30.0)
    params["cooling_rate_initial"] = trial.suggest_float("cooling_rate_initial", 0.990, 0.9999)
    params["cooling_rate_final"] = trial.suggest_float("cooling_rate_final", 0.970, 0.995)
    params["cooling_transition_iters"] = trial.suggest_int("cooling_transition_iters", 300, 1600)

    # === Adaptive: Enforce σ₁ > σ₂ > σ₃ > σ₄ — scaled for nonlinear costs ===
    params["eta"] = trial.suggest_float("eta", 0.05, 0.5)
    params["alpha_credit"] = trial.suggest_float("alpha_credit", 0.3, 0.8)

    sigma1 = trial.suggest_float("reward_global", 100.0, 2000.0)
    sigma4 = trial.suggest_float("reward_worse", 0.5, 30.0)
    sigma2 = trial.suggest_float("reward_better_ratio", 0.3, 0.8)
    sigma3_max = max(0.1, sigma2 - 0.05)
    sigma3 = trial.suggest_float("reward_slight_ratio", 0.1, sigma3_max)
    params["reward_scale"] = {
        "global": sigma1,
        "better": sigma1 * sigma2,
        "slight_better": sigma1 * sigma3,
        "accepted_worse": sigma4,
    }

    params["weight_decay"] = trial.suggest_float("weight_decay", 0.005, 0.06)

    # === Destroy: WIDER range for stability ===
    params["r_lower"] = trial.suggest_float("r_lower", 0.10, 0.25)
    params["r_upper_small"] = trial.suggest_float("r_upper_small", 0.35, 0.55)
    params["quota_base_cap"] = trial.suggest_int("quota_base_cap", 15, 40)

    # === Reheat ===
    params["reheat_stall_trigger"] = trial.suggest_int("reheat_stall_trigger", 200, 600)
    params["reheat_acceptance_window"] = trial.suggest_int("reheat_acceptance_window", 50, 180)
    params["reheat_acceptance_min"] = trial.suggest_float("reheat_acceptance_min", 0.02, 0.15)
    params["reheat_duration"] = trial.suggest_int("reheat_duration", 30, 80)
    params["reheat_temperature_scale"] = trial.suggest_float("reheat_temperature_scale", 0.15, 0.60)
    params["reheat_quota_multiplier"] = trial.suggest_float("reheat_quota_multiplier", 1.5, 3.5)
    params["reheat_shake_probability"] = trial.suggest_float("reheat_shake_probability", 0.4, 0.95)
    params["reheat_random_repair_prob"] = trial.suggest_float("reheat_random_repair_prob", 0.2, 0.7)

    # === Local Search ===
    params["local_search_frequency"] = trial.suggest_int("local_search_frequency", 4, 20)
    params["local_search_on_new_best"] = trial.suggest_categorical("local_search_on_new_best", [True])
    params["depot_drone_probability"] = trial.suggest_float("depot_drone_probability", 0.05, 0.35)
    params["intensify_frequency"] = trial.suggest_int("intensify_frequency", 20, 70)

    # === Escape: off by default, since Round 1 showed it hurts ===
    params["escape_enabled"] = False

    # === Convergence: enable diversification, tune carefully ===
    params["dynamic_cooling_enabled"] = trial.suggest_categorical("dynamic_cooling_enabled", [True])
    params["cooling_slowdown_factor"] = trial.suggest_float("cooling_slowdown_factor", 0.992, 0.9999)
    params["cooling_speedup_factor"] = trial.suggest_float("cooling_speedup_factor", 0.96, 0.99)
    params["recent_improvement_window"] = trial.suggest_int("recent_improvement_window", 30, 100)

    params["diversification_enabled"] = trial.suggest_categorical("diversification_enabled", [True, False])
    if params["diversification_enabled"]:
        params["diversification_trigger_stall"] = trial.suggest_int("diversification_trigger_stall", 400, 800)
        params["diversification_restart_best_prob"] = trial.suggest_float("diversification_restart_best_prob", 0.5, 0.9)
        params["diversification_destroy_ratio"] = trial.suggest_float("diversification_destroy_ratio", 0.4, 0.7)

    params["adaptive_quota_enabled"] = trial.suggest_categorical("adaptive_quota_enabled", [True, False])

    # === Drone ===
    params["drone_priority"] = trial.suggest_float("drone_priority", 1.5, 3.5)
    params["depot_bonus"] = trial.suggest_float("depot_bonus", 0.3, 2.5)
    params["multi_customer_bonus"] = trial.suggest_float("multi_customer_bonus", 0.5, 6.0)

    return params


def build_sa_config(params: Dict[str, Any], iterations: int = 6000) -> SANNCfg:
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
        "escape_enabled": params.get("escape_enabled", False),
        "dynamic_cooling_enabled": params["dynamic_cooling_enabled"],
        "diversification_enabled": params["diversification_enabled"],
        "adaptive_quota_enabled": params["adaptive_quota_enabled"],
    })
    base["escape_trigger_stall"] = 80
    base["escape_duration"] = 20
    if params.get("dynamic_cooling_enabled"):
        base["improvement_threshold"] = 0.01
        base["cooling_slowdown_factor"] = params.get("cooling_slowdown_factor", 0.998)
        base["cooling_speedup_factor"] = params.get("cooling_speedup_factor", 0.980)
        base["recent_improvement_window"] = params.get("recent_improvement_window", 50)
    if params.get("diversification_enabled"):
        base["diversification_trigger_stall"] = params.get("diversification_trigger_stall", 500)
        base["diversification_restart_best_prob"] = params.get("diversification_restart_best_prob", 0.7)
        base["diversification_destroy_ratio"] = params.get("diversification_destroy_ratio", 0.6)
    return SANNCfg(**base)


def run_single(instance_name, params, seed, iterations=6000):
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

    sa_cfg = build_sa_config(params, iterations=iterations)
    alns_rng = random.Random(seed)
    dp = params["drone_priority"]
    bonus = {
        "depot_bonus": params["depot_bonus"],
        "multi_customer_bonus": params["multi_customer_bonus"],
        "multi_customer_threshold": 2,
        "wait_max": 20.0,
        "allow_multiple_launch_per_node": config.relax_allow_multiple_launch_per_node,
    }

    destroy_ops = [
        DestroyRandom(instance, rng=random.Random(seed + 1000), anchor_strategy="rebase_to_neighbor"),
        DestroyWorstDistance(instance, rng=random.Random(seed + 1004), anchor_strategy="rebase_to_neighbor"),
        DestroyShaw(instance, rng=random.Random(seed + 1002), anchor_strategy="rebase_to_neighbor"),
    ]
    repair_ops = [
        RepairCheapest(instance, rng=random.Random(seed + 2004), drone_priority=dp, robust_energy_mode="embedded", **bonus),
        RepairDronePriorityRegret(instance, rng=random.Random(seed + 2002), drone_priority=dp, robust_energy_mode="embedded", **bonus),
        RepairTruckFirst(instance, rng=random.Random(seed + 2003), drone_priority=dp, robust_energy_mode="embedded", **bonus),
        RepairEqualPriority(instance, rng=random.Random(seed + 2001), drone_priority=dp, robust_energy_mode="embedded", **bonus),
    ]

    initial = build_two_phase_initial_solution(instance)
    alns = SimulatedAnnealingALNS(
        instance=instance, destroy_ops=destroy_ops, repair_ops=repair_ops,
        evaluator=evaluator, cfg=sa_cfg, rng=alns_rng)

    start = time.time()
    try:
        best = alns.run(initial)
    except Exception as e:
        return {"cost": INFEASIBLE_PENALTY, "feasible": False, "runtime": time.time() - start,
                "delay_cost": 0, "truck_cost": 0, "drone_cost": 0, "error": str(e)}
    runtime = time.time() - start
    eval_res = evaluator.evaluate_solution(best)
    return {
        "cost": eval_res.total_cost, "feasible": eval_res.feasible,
        "delay_cost": eval_res.delay_penalty, "truck_cost": eval_res.truck_distance_cost,
        "drone_cost": eval_res.drone_distance_cost, "runtime": runtime,
    }





def create_objective(instances, n_seeds, iterations, lambda_std):
    """Stability-aware objective: minimizes mean + λ·std."""
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        scores = []
        grand_costs = []
        for inst in instances:
            costs = []
            for s_off in range(n_seeds):
                seed = 100 + s_off + trial.number * 7
                result = run_single(inst, params, seed, iterations)
                cost = result["cost"] if result["feasible"] else INFEASIBLE_PENALTY
                costs.append(cost)
            mean_c = np.mean(costs)
            std_c = np.std(costs)
            scores.append(mean_c + lambda_std * std_c)
            grand_costs.extend(costs)
            if trial.should_prune():
                raise optuna.TrialPruned()
        overall = float(np.mean(scores))
        trial.set_user_attr("mean_cost", float(np.mean(grand_costs)))
        trial.set_user_attr("std_cost", float(np.std(grand_costs)))
        return overall
    return objective


def run_phase1(n_trials=20, lambda_std=1.5):
    print("=" * 80)
    print("  PHASE 1 (Round 3, nonlinear cost): Coarse Search")
    print(f"  Instances: 2, Seeds: 2, Iterations: 4000, λ_std={lambda_std}")
    print("=" * 80)
    sampler = TPESampler(seed=42, n_startup_trials=5)
    study = optuna.create_study(direction="minimize", sampler=sampler,
                                study_name="alns_stable_phase1")
    objective = create_objective(PHASE1_INSTANCES, n_seeds=2, iterations=4000,
                                  lambda_std=lambda_std)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\n  Phase 1 Best: score={study.best_value:.2f}")
    return study


def run_phase2(phase1_study, n_trials=15, lambda_std=1.5):
    print("=" * 80)
    print("  PHASE 2 (Round 3, nonlinear cost): Fine Search")
    print(f"  Instances: 3, Seeds: 3, Iterations: 3000, λ_std={lambda_std}")
    print("=" * 80)
    sampler = TPESampler(seed=123, n_startup_trials=5)
    study = optuna.create_study(direction="minimize", sampler=sampler,
                                study_name="alns_stable_phase2")
    warm_start_params = None
    try:
        warm_start_params = phase1_study.best_params
    except ValueError:
        for t in phase1_study.get_trials(deepcopy=False):
            if t.state == optuna.trial.TrialState.COMPLETE:
                warm_start_params = t.params
                break
    if warm_start_params is None:
        for t in phase1_study.get_trials(deepcopy=False):
            if t.state == optuna.trial.TrialState.WAITING:
                warm_start_params = t.params
                break
    if warm_start_params is not None:
        study.enqueue_trial(warm_start_params)
    objective = create_objective(PHASE2_INSTANCES, n_seeds=3, iterations=3000,
                                  lambda_std=lambda_std)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\n  Phase 2 Best: score={study.best_value:.2f}")
    return study


def run_validation(best_params, n_seeds=5, iterations=4000):
    print("=" * 80)
    print("  PHASE 3: Stability Validation")
    print(f"  All 5 instances, {n_seeds} seeds, {iterations} iters")
    print("=" * 80)
    results = []
    for inst in ALL_INSTANCES:
        for s in range(n_seeds):
            seed = 100 + s
            result = run_single(inst, best_params, seed=seed, iterations=iterations)
            result["instance"] = inst
            result["seed"] = seed
            results.append(result)
            tag = "OK" if result["feasible"] else "INFEASIBLE"
            print(f"  {inst} seed={seed}: cost={result['cost']:.2f} [{tag}] ({result['runtime']:.1f}s)")

    feasible = [r for r in results if r["feasible"]]
    print(f"\n  === Stability Results ===")
    print(f"  {'Instance':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Gap%':>7} {'Feas':>5}")
    print("  " + "-" * 55)
    all_means = []
    for inst in ALL_INSTANCES:
        costs = [r["cost"] for r in feasible if r["instance"] == inst]
        if costs:
            m, s, mn, mx = np.mean(costs), np.std(costs), np.min(costs), np.max(costs)
            gap = (mx - mn) / mn * 100 if mn > 0 else 0
            all_means.append(m)
            print(f"  {inst:<12} {m:>8.2f} {s:>8.2f} {mn:>8.2f} {mx:>8.2f} {gap:>6.1f}% {len(costs):>3}/{n_seeds}")
    if all_means:
        print(f"\n  Overall mean={np.mean(all_means):.2f}")
    return results


def save_results(phase, study, results_dir):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    df = study.trials_dataframe()
    df.to_csv(results_dir / f"stable_{phase}_{timestamp}.csv", index=False)

    best = dict(study.best_params)
    if "reward_scale" not in best:
        sigma1 = best.pop("reward_global")
        sigma2_ratio = best.pop("reward_better_ratio")
        sigma3_ratio = best.pop("reward_slight_ratio")
        best["reward_scale"] = {
            "global": sigma1,
            "better": sigma1 * sigma2_ratio,
            "slight_better": sigma1 * sigma3_ratio,
            "accepted_worse": best.pop("reward_worse"),
        }
    with open(results_dir / f"best_params_stable_{phase}_{timestamp}.json", "w") as f:
        json.dump(best, f, indent=2, default=str)
    print(f"\n  Results → {results_dir}")
    return best, timestamp


def config_from_best_params(p):
    lines = [
        "# === Stability-Tuned ALNS Config (Bayesian Round 2) ===",
        "",
        "simulated_annealing:",
        f"  w_percent: {p.get('w_percent', 15.0):.4f}",
        "  temperature_min: 1.0e-4",
        "  cooling:",
        f"    rate_initial: {p.get('cooling_rate_initial', 0.9995):.6f}",
        f"    rate_final: {p.get('cooling_rate_final', 0.985):.6f}",
        f"    transition_iters: {p.get('cooling_transition_iters', 800)}",
        "",
        "adaptive_selection:",
        f"  eta: {p.get('eta', 0.35):.4f}",
        f"  alpha_credit: {p.get('alpha_credit', 0.60):.4f}",
        "  rewards:",
    ]
    rs = p.get("reward_scale", {})
    lines += [
        f"    global: {rs.get('global', 33):.2f}",
        f"    better: {rs.get('better', 13):.2f}",
        f"    slight_better: {rs.get('slight_better', 9):.2f}",
        f"    accepted_worse: {rs.get('accepted_worse', 1):.2f}",
        f"  weight_decay: {p.get('weight_decay', 0.02):.4f}",
        "",
        "destroy_quota:",
        f"  r_lower: {p.get('r_lower', 0.15):.4f}",
        f"  r_upper_small: {p.get('r_upper_small', 0.50):.4f}",
        f"  base_cap: {p.get('quota_base_cap', 30)}",
        "",
        "reheat:",
        f"  stall_trigger: {p.get('reheat_stall_trigger', 450)}",
        f"  acceptance_window: {p.get('reheat_acceptance_window', 100)}",
        f"  acceptance_min: {p.get('reheat_acceptance_min', 0.08):.4f}",
        f"  duration: {p.get('reheat_duration', 50)}",
        f"  temperature_scale: {p.get('reheat_temperature_scale', 0.3):.4f}",
        f"  quota_multiplier: {p.get('reheat_quota_multiplier', 2.5):.4f}",
        f"  shake_probability: {p.get('reheat_shake_probability', 0.85):.4f}",
        f"  random_repair_prob: {p.get('reheat_random_repair_prob', 0.5):.4f}",
        "",
        "local_search:",
        f"  frequency: {p.get('local_search_frequency', 10)}",
        f"  on_new_best: {p.get('local_search_on_new_best', True)}",
        f"  depot_drone_probability: {p.get('depot_drone_probability', 0.18):.4f}",
        f"  intensify_frequency: {p.get('intensify_frequency', 35)}",
        "",
        "escape:",
        f"  enabled: {p.get('escape_enabled', False)}",
        "",
        "convergence_enhancement:",
        "  dynamic_cooling:",
        f"    enabled: {p.get('dynamic_cooling_enabled', True)}",
        f"    slowdown_factor: {p.get('cooling_slowdown_factor', 0.998):.6f}",
        f"    speedup_factor: {p.get('cooling_speedup_factor', 0.980):.4f}",
        f"    recent_improvement_window: {p.get('recent_improvement_window', 50)}",
        "  diversification:",
        f"    enabled: {p.get('diversification_enabled', True)}",
        f"    trigger_stall: {p.get('diversification_trigger_stall', 500)}",
        f"    restart_best_prob: {p.get('diversification_restart_best_prob', 0.7):.2f}",
        f"    random_destroy_ratio: {p.get('diversification_destroy_ratio', 0.6):.2f}",
        "  adaptive_quota:",
        f"    enabled: {p.get('adaptive_quota_enabled', True)}",
        "",
        "drone:",
        f"  priority: {p.get('drone_priority', 2.2):.2f}",
        "  bonus:",
        f"    depot: {p.get('depot_bonus', 0.6):.2f}",
        f"    multi_customer: {p.get('multi_customer_bonus', 5.5):.2f}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stability-focused Bayesian Tuning for ALNS")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--lambda-std", type=float, default=1.5,
                        help="Weight on std in objective: mean + λ*std")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        args.phase = None

    lambda_std = args.lambda_std
    results_dir = PROJECT_ROOT / "results" / "bayesian_tuning_r3_nonlinear"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = results_dir / f"tuning_{timestamp}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger(__name__)
    logger.info(f"Stability tuning started: phase={args.phase}, lambda_std={lambda_std}")

    best_p1 = None
    if args.phase == 1 or args.all or args.phase is None:
        n = args.trials or 50
        logger.info(f"Phase 1: {n} trials, lambda={lambda_std}")
        study1 = run_phase1(n_trials=n, lambda_std=lambda_std)
        best_p1, ts1 = save_results("phase1", study1, results_dir)
        yaml_out = config_from_best_params(best_p1)
        with open(results_dir / f"best_config_stable_phase1_{ts1}.yaml", "w") as f:
            f.write(yaml_out)
        logger.info(f"Phase 1 complete. Best score: {study1.best_value:.2f}")

    if args.phase == 2 or args.all:
        n2 = args.trials or 30
        phase1_study = None
        if args.all or args.phase is None:
            phase1_study = study1
        else:
            import glob
            jsons = sorted(glob.glob(str(results_dir / "best_params_stable_phase1_*.json")))
            if not jsons:
                logger.error("No Phase 1 data. Run Phase 1 first."); sys.exit(1)
            with open(jsons[-1]) as f:
                best_p1 = json.load(f)
            phase1_params = dict(best_p1)
            if "reward_scale" in phase1_params:
                rs = phase1_params.pop("reward_scale")
                phase1_params["reward_global"] = rs["global"]
                phase1_params["reward_better_ratio"] = rs["better"] / rs["global"] if rs["global"] != 0 else 0.5
                phase1_params["reward_slight_ratio"] = rs["slight_better"] / rs["global"] if rs["global"] != 0 else 0.3
                phase1_params["reward_worse"] = rs["accepted_worse"]
            phase1_study = optuna.create_study(direction="minimize", study_name="alns_stable_phase1_loaded")
            phase1_study.enqueue_trial(phase1_params)
        logger.info(f"Phase 2: {n2} trials, lambda={lambda_std}")
        study2 = run_phase2(phase1_study, n_trials=n2, lambda_std=lambda_std)
        best_p2, ts2 = save_results("phase2", study2, results_dir)
        yaml_out = config_from_best_params(best_p2)
        with open(results_dir / f"best_config_stable_phase2_{ts2}.yaml", "w") as f:
            f.write(yaml_out)
        logger.info(f"Phase 2 complete. Best score: {study2.best_value:.2f}")

    if args.phase == 3:
        import glob
        jsons = sorted(glob.glob(str(results_dir / "best_params_stable_phase2_*.json")))
        if not jsons:
            jsons = sorted(glob.glob(str(results_dir / "best_params_stable_phase1_*.json")))
        if not jsons:
            logger.error("No best params found. Run Phase 1/2 first."); sys.exit(1)
        with open(jsons[-1]) as f:
            best_params = json.load(f)
        # Convert reward ratios
        if "reward_scale" not in best_params and "reward_global" in best_params:
            s1 = best_params.pop("reward_global")
            s2r = best_params.pop("reward_better_ratio")
            s3r = best_params.pop("reward_slight_ratio")
            best_params["reward_scale"] = {
                "global": s1, "better": s1 * s2r,
                "slight_better": s1 * s3r, "accepted_worse": best_params.pop("reward_worse"),
            }
        n_seeds = 10
        logger.info(f"Phase 3 validation: {n_seeds} seeds")
        results = run_validation(best_params, n_seeds=n_seeds)
        val_path = results_dir / f"validation_{timestamp}.json"
        with open(val_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Validation → {val_path}")

    logger.info("Stability tuning done!")
