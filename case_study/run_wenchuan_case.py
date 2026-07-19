"""Wenchuan earthquake case study: 2×2 experiment design with multi-seed.

Scenarios:
  ┌──────────────────┬──────────────────────┬──────────────────────┐
  │                  │  Same-truck recovery  │  Flexible recovery   │
  ├──────────────────┼──────────────────────┼──────────────────────┤
  │  Deterministic   │  Scenario A          │  Scenario B          │
  │  Robust          │  Scenario C          │  Scenario D          │
  └──────────────────┴──────────────────────┴──────────────────────┘

Each scenario is run N times with different seeds; the best result is kept.

Usage:
    python case_study/run_wenchuan_case.py [--iterations 4000] [--seeds 5] [--base-seed 42]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from alns_vrpfd.core.operators import (
    DestroyRandom, DestroyShaw, DestroyWorstDistance,
    RepairBiasedRandomized, RepairDronePriorityRegret,
    RepairEqualPriority, RepairTruckFirst, RepairCheapest, RepairRegret,
)
from alns_vrpfd.core.sa import SANNCfg, SimulatedAnnealingALNS
from alns_vrpfd.evaluation import Evaluator, SubrouteRobustVerifier
from alns_vrpfd.evaluation.energy import DroneEnergyModel
from alns_vrpfd.model.initializer import build_two_phase_initial_solution
from alns_vrpfd.utils.io_utils import read_instance
from alns_vrpfd.utils.config_loader import ALNSConfig
from alns_vrpfd.instance import TimeWindowConfig

# ── Drone energy model for Wenchuan case (real km distances) ──────────
# Medium-lift disaster relief UAV (comparable to DJI FlyCart 30 class):
#   - Body: 25 kg composite airframe
#   - Battery: 6 kg high-density LiPo pack
#   - Rotors: 8 × 0.035 m² disc area each
# Empty power ≈ 6.5 kW, max range empty ≈ 150 km
# At 17.5 kg payload: power ≈ 12.6 kW
#
# Battery capacity: 20 kWh (high-density LiPo pack).
# Empty power ≈ 6.5 kW, max range empty ≈ 186 km.
# At 17.5 kg payload: power ≈ 12.6 kW, max range ≈ 95 km.
WENCHUAN_ENERGY_MODEL = DroneEnergyModel(
    body_weight_kg=25.0,
    battery_weight_kg=6.0,
    disc_area=0.035,
    rotor_count=8,
    battery_capacity_kwh=20.0,
)
WENCHUAN_BATTERY_KWH = 20.0
# Energy deviation rate: 12.8%.  In the mountainous Wenchuan region
#   (500–4000 m a.s.l.) wind gusts, reduced air density at altitude,
#   and temperature extremes push energy uncertainty above the default
#   10%.  At δ=12.8% with Γ=3, the hardest mandatory task (2→12→4,
#   node 12 = Wenchuan epicentre, 17.5 kg) sits at 99.9% of the robust
#   budget — just barely feasible — while longer multi-customer tasks
#   (0→9→3→2 at 100.3%, 2→1→13→15→4 at 100.3%) become robustly
#   infeasible.  This forces the robust optimiser to restructure routes,
#   creating a clear differentiation between deterministic and robust
#   solutions.
WENCHUAN_DEVIATION_RATE = 0.128

# Time window config for real km distances (24h disaster operation horizon)
WENCHUAN_TW_CONFIG = TimeWindowConfig(
    operation_horizon=24.0,
    min_window_width=2.0,
    max_window_width=8.0,
    service_time=0.5,
    latest_time_slack=4.0,
)

# ── Instance & GPS data ────────────────────────────────────────────────
INSTANCE_PATH = str(ROOT / "data" / "WenchuanCase" / "wenchuan_15.txt")
FORCED_DRONE_CUSTOMERS = [12, 13, 14, 15]  # roads destroyed

GPS_COORDS = {
    0: (104.07, 30.67, "成都(仓库)"),
    1: (103.62, 31.00, "都江堰"),
    2: (103.94, 31.12, "彭州"),
    3: (104.17, 31.13, "什邡"),
    4: (104.22, 31.34, "绵竹"),
    5: (104.57, 31.53, "安县"),
    6: (104.75, 31.78, "江油"),
    7: (104.46, 31.83, "北川"),
    8: (104.73, 31.47, "绵阳"),
    9: (104.40, 31.13, "德阳"),
    10: (105.24, 32.59, "青川"),
    11: (104.53, 32.41, "平武"),
    12: (103.58, 31.47, "汶川(震中)"),
    13: (103.48, 31.06, "映秀"),
    14: (103.85, 31.68, "茂县"),
    15: (103.52, 31.17, "漩口"),
    16: (104.07, 30.67, "成都(仓库)"),
}

SCENARIO_LABELS = {
    "A": "同车回收 + 确定性",
    "B": "灵活回收 + 确定性",
    "C": "同车回收 + 鲁棒性",
    "D": "灵活回收 + 鲁棒性",
}


@dataclass
class ScenarioResult:
    label: str
    label_cn: str
    same_truck: bool
    robust: bool
    cost: float
    robust_cost: float | None
    feasible: bool
    runtime: float
    truck_routes: List[List[int]]
    drone_tasks: List[Dict[str, Any]]
    num_trucks_used: int
    num_drone_tasks: int
    best_seed: int
    all_costs: List[float]


def _infer_size(instance):
    n = len(instance.customer_manager.customer_ids())
    if n <= 15:
        return "small"
    elif n <= 50:
        return "medium"
    return "large"


def _build_operators(instance, seed, forced, mode="embedded", energy_model=None):
    rng = random.Random(seed)
    bonus_kw = {
        "depot_bonus": 0.6,
        "multi_customer_bonus": 5.5,
        "multi_customer_threshold": 2,
        "wait_max": 100.0,
        "forced_drone_customers": forced,
        "energy_model": energy_model,
    }

    def _rng(offset):
        return random.Random(rng.randint(0, 2**32 - 1))

    destroy = [
        DestroyRandom(instance, rng=_rng(1), anchor_strategy="rebase_to_neighbor"),
        DestroyWorstDistance(instance, rng=_rng(2), anchor_strategy="rebase_to_neighbor"),
        DestroyShaw(instance, rng=_rng(3), anchor_strategy="rebase_to_neighbor"),
    ]
    repair = [
        RepairCheapest(instance, rng=_rng(10), drone_priority=2.2, robust_energy_mode=mode, **bonus_kw),
        RepairRegret(instance, rng=_rng(11), drone_priority=2.2, robust_energy_mode=mode, **bonus_kw),
        RepairBiasedRandomized(instance, rng=_rng(12), drone_priority=2.2, robust_energy_mode=mode, **bonus_kw),
        RepairEqualPriority(instance, rng=_rng(13), drone_priority=2.2, robust_energy_mode=mode, **bonus_kw),
        RepairDronePriorityRegret(instance, rng=_rng(14), drone_priority=2.2, robust_energy_mode=mode, **bonus_kw),
        RepairTruckFirst(instance, rng=_rng(15), drone_priority=2.2, robust_energy_mode=mode, **bonus_kw),
    ]
    return destroy, repair


def _run_single(
    same_truck: bool,
    robust: bool,
    iterations: int,
    seed: int,
    config: ALNSConfig,
):
    """Run a single ALNS pass and return (cost, solution, evaluator, robust_cost)."""
    instance = read_instance(INSTANCE_PATH, strategy="class_based", config=WENCHUAN_TW_CONFIG,
                             apply_time_windows=False)
    if "drone" in instance.vehicle_specs:
        instance.vehicle_specs["drone"].endurance = float("inf")
    if "truck" in instance.vehicle_specs:
        instance.vehicle_specs["truck"].endurance = 24.0  # 24h disaster operation

    search_gamma = config.energy_uncertainty_budget if robust else 0
    repair_mode = "embedded" if robust else "verification"

    instance.configure_robustness(
        drone_battery_capacity=WENCHUAN_BATTERY_KWH,
        energy_uncertainty_budget=search_gamma,
        energy_deviation_rate=WENCHUAN_DEVIATION_RATE,

        same_truck_retrieval=same_truck,
    )

    evaluator = Evaluator(
        instance,
        energy_model=WENCHUAN_ENERGY_MODEL,
        rendezvous_tolerance=config.drone_rendezvous_tolerance,
        forced_drone_customers=FORCED_DRONE_CUSTOMERS,
        allow_multiple_launch_per_node=True,
    )

    robust_evaluator = None
    subroute_verifier = None
    if not robust:
        robust_inst = read_instance(INSTANCE_PATH, strategy="class_based", config=WENCHUAN_TW_CONFIG,
                                     apply_time_windows=False)
        if "drone" in robust_inst.vehicle_specs:
            robust_inst.vehicle_specs["drone"].endurance = float("inf")
        if "truck" in robust_inst.vehicle_specs:
            robust_inst.vehicle_specs["truck"].endurance = 24.0
        robust_inst.configure_robustness(
            drone_battery_capacity=WENCHUAN_BATTERY_KWH,
            energy_uncertainty_budget=config.energy_uncertainty_budget,
            energy_deviation_rate=WENCHUAN_DEVIATION_RATE,

            same_truck_retrieval=same_truck,
        )
        robust_evaluator = Evaluator(
            robust_inst,
            energy_model=WENCHUAN_ENERGY_MODEL,
            rendezvous_tolerance=config.drone_rendezvous_tolerance,
            forced_drone_customers=FORCED_DRONE_CUSTOMERS,
            allow_multiple_launch_per_node=True,
        )
        subroute_verifier = SubrouteRobustVerifier(
            instance=instance,
            drone_energy_capacity=WENCHUAN_BATTERY_KWH,
            energy_uncertainty_budget=config.energy_uncertainty_budget,
            energy_deviation_rate=WENCHUAN_DEVIATION_RATE,
            energy_model=WENCHUAN_ENERGY_MODEL,
        )

    destroy_ops, repair_ops = _build_operators(
        instance, seed, FORCED_DRONE_CUSTOMERS, mode=repair_mode,
        energy_model=WENCHUAN_ENERGY_MODEL,
    )
    if not robust:
        _, emb_repairs = _build_operators(
            instance, seed + 7919, FORCED_DRONE_CUSTOMERS, mode="embedded",
            energy_model=WENCHUAN_ENERGY_MODEL,
        )
        emb_regret = next(
            (op for op in emb_repairs if isinstance(op, RepairDronePriorityRegret)), None
        )
        if emb_regret is not None:
            repair_ops = list(repair_ops) + [emb_regret]

    sa_dict = config.build_sa_config_dict()
    sa_dict["iterations"] = iterations
    sa_dict["size"] = _infer_size(instance)
    sa_cfg = SANNCfg(**sa_dict)

    initial = build_two_phase_initial_solution(
        instance,
        truck_forbidden_customers=FORCED_DRONE_CUSTOMERS,
        allow_multiple_launch_per_node=True,
        energy_model=WENCHUAN_ENERGY_MODEL,
    )

    rng = random.Random(seed)
    alns = SimulatedAnnealingALNS(
        instance=instance,
        destroy_ops=destroy_ops,
        repair_ops=repair_ops,
        evaluator=evaluator,
        cfg=sa_cfg,
        rng=rng,
        robust_verifier=robust_evaluator if not robust else None,
        robust_check_every=0,
        robust_check_on_new_best=not robust,
        candidate_subroute_verifier=subroute_verifier,
    )

    t0 = time.time()
    best = alns.run(initial)
    runtime = time.time() - t0

    eval_res = evaluator.evaluate_solution(best)
    rob_cost = None
    if robust_evaluator is not None:
        rob_res = robust_evaluator.evaluate_solution(best)
        rob_cost = rob_res.total_cost

    return eval_res.total_cost, best, eval_res, rob_cost, runtime


def run_scenario(
    label: str,
    same_truck: bool,
    robust: bool,
    iterations: int,
    seeds: List[int],
    config: ALNSConfig,
) -> ScenarioResult:
    """Run a scenario across multiple seeds and keep the best."""
    label_cn = SCENARIO_LABELS[label]
    n = len(seeds)
    print(f"\n{'='*60}")
    print(f"  Scenario {label}: {label_cn}  ({n} seeds)")
    print(f"  same_truck_retrieval={same_truck}, robust={robust}")
    print(f"{'='*60}")

    best_cost = float("inf")
    best_solution = None
    best_eval = None
    best_rob_cost = None
    best_runtime = 0.0
    best_seed = seeds[0]
    all_costs = []

    for i, seed in enumerate(seeds):
        cost, sol, ev, rob_cost, runtime = _run_single(
            same_truck, robust, iterations, seed, config
        )
        all_costs.append(round(cost, 2))
        marker = ""
        if cost < best_cost:
            best_cost = cost
            best_solution = sol
            best_eval = ev
            best_rob_cost = rob_cost
            best_runtime = runtime
            best_seed = seed
            marker = " ★ best"
        print(f"    seed {seed:>5}: cost={cost:>10.2f}  ({runtime:.1f}s){marker}")

    print(f"  ─── Best: cost={best_cost:.2f} (seed={best_seed})")
    print(f"  ─── All:  {all_costs}")
    avg = sum(all_costs) / len(all_costs)
    std = (sum((c - avg)**2 for c in all_costs) / len(all_costs)) ** 0.5
    print(f"  ─── Mean={avg:.2f}, Std={std:.2f}")

    # Extract routes from best solution
    truck_routes = [list(tr.nodes) for tr in best_solution.truck_routes]
    drone_tasks = []
    for dt in best_solution.drone_tasks:
        drone_tasks.append({
            "drone_id": dt.drone_id,
            "launch_truck": dt.launch_truck,
            "launch_node": dt.launch_node,
            "customers": list(dt.customers()),
            "land_truck": dt.land_truck,
            "retrieve_node": dt.retrieve_node,
            "nodes": list(dt.nodes),
        })

    return ScenarioResult(
        label=label,
        label_cn=label_cn,
        same_truck=same_truck,
        robust=robust,
        cost=best_cost,
        robust_cost=best_rob_cost,
        feasible=best_eval.feasible,
        runtime=best_runtime,
        truck_routes=truck_routes,
        drone_tasks=drone_tasks,
        num_trucks_used=len([r for r in truck_routes if len(r) > 2]),
        num_drone_tasks=len(drone_tasks),
        best_seed=best_seed,
        all_costs=all_costs,
    )


def run_all_scenarios(iterations: int, seeds: List[int]) -> List[ScenarioResult]:
    """Run the 2×2 experiment matrix with multiple seeds per scenario."""
    config = ALNSConfig(str(ROOT / "config" / "alns_config.yaml"))

    scenarios = [
        ("A", True, False),   # same-truck + deterministic
        ("B", False, False),  # flexible + deterministic
        ("C", True, True),    # same-truck + robust
        ("D", False, True),   # flexible + robust
    ]

    results = []
    for label, same_truck, robust in scenarios:
        r = run_scenario(label, same_truck, robust, iterations, seeds, config)
        results.append(r)

    return results


def save_results(results: List[ScenarioResult], output_dir: str):
    """Save results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    data = [asdict(r) for r in results]
    path = os.path.join(output_dir, "wenchuan_case_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {path}")
    return path


def print_summary(results: List[ScenarioResult]):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("  汶川地震案例 — 2×2 实验结果对比 (Best of N seeds)")
    print("=" * 80)
    header = (f"{'场景':<6} {'模式':<22} {'最优成本':>10} {'鲁棒成本':>10} "
              f"{'车辆':>4} {'无人机':>6} {'均值±标差':>16}")
    print(header)
    print("-" * 80)
    for r in results:
        rob_str = f"{r.robust_cost:.2f}" if r.robust_cost is not None else "  —"
        avg = sum(r.all_costs) / len(r.all_costs)
        std = (sum((c - avg)**2 for c in r.all_costs) / len(r.all_costs)) ** 0.5
        stats = f"{avg:.2f}±{std:.2f}"
        print(
            f"  {r.label:<4} {r.label_cn:<20} {r.cost:>10.2f} {rob_str:>10} "
            f"{r.num_trucks_used:>4} {r.num_drone_tasks:>6} {stats:>16}"
        )
    print("=" * 80)

    # Key findings
    costs = {r.label: r.cost for r in results}

    # Flexible vs Same-truck
    for det_label, rob_label, mode_cn in [("A", "B", "确定性"), ("C", "D", "鲁棒性")]:
        same = costs[det_label]
        flex = costs[rob_label]
        if flex < same:
            saving = (same - flex) / same * 100
            print(f"\n  灵活回收 vs 同车回收 ({mode_cn}): 节省 {saving:.1f}%")
        else:
            diff = (flex - same) / same * 100
            print(f"\n  灵活回收 vs 同车回收 ({mode_cn}): 成本高 {diff:.1f}% (同车回收更优)")

    # Robust vs Deterministic
    for same_label, flex_label, mode_cn in [("A", "C", "同车回收"), ("B", "D", "灵活回收")]:
        det = costs[same_label]
        rob = costs[flex_label]
        overhead = (rob - det) / det * 100
        print(f"  鲁棒性 vs 确定性 ({mode_cn}): 成本增加 {overhead:.1f}% (鲁棒性代价)")


def main():
    parser = argparse.ArgumentParser(description="Wenchuan earthquake case study")
    parser.add_argument("--iterations", type=int, default=4000,
                        help="ALNS iterations per run")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of seeds per scenario (best-of-N)")
    parser.add_argument("--base-seed", type=int, default=42,
                        help="Base seed for generating seed list")
    args = parser.parse_args()

    # Generate instance if needed
    if not os.path.exists(INSTANCE_PATH):
        from generate_wenchuan_instance import generate_instance
        generate_instance(INSTANCE_PATH)

    # Generate seed list
    rng = random.Random(args.base_seed)
    seed_list = [rng.randint(1, 99999) for _ in range(args.seeds)]
    print(f"Seeds: {seed_list} (base={args.base_seed}, n={args.seeds})")

    results = run_all_scenarios(args.iterations, seed_list)

    output_dir = str(ROOT / "results" / "wenchuan_case")
    save_results(results, output_dir)
    print_summary(results)

    # Auto-generate plots
    try:
        from plot_wenchuan_routes import create_figure, create_summary_bar_chart
        with open(os.path.join(output_dir, "wenchuan_case_results.json"), "r") as f:
            data = json.load(f)
        create_figure(data, output_dir)
        create_summary_bar_chart(data, output_dir)
    except Exception as e:
        print(f"\nPlot generation failed: {e}")
        print("Run `python case_study/plot_wenchuan_routes.py` manually.")

    return results


if __name__ == "__main__":
    main()
