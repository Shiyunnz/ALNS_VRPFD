#!/usr/bin/env python3
"""A/B/C/D/E ablation on R_30_10_2: verify TruckDroneRechain closes the gap.

A: ALNS baseline (no drone LS)
B: ALNS + Step6 (drone_reanchor_ls)
C: ALNS + Step6 + MLNS final polish
D: ALNS + Step6 + MLNS + TruckDroneRechain polish
E: MILP with ALNS verification

Usage: python ablation_rechain.py [--seeds 5] [--iters 4000]
"""

import sys, time, json, random, math, statistics, argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from alns_vrpfd.utils.io_utils import read_instance
from alns_vrpfd.utils.config_loader import ALNSConfig
from alns_vrpfd.evaluation.evaluator import Evaluator
from alns_vrpfd.core.sa import SANNCfg, SimulatedAnnealingALNS
from alns_vrpfd.model.initializer import build_two_phase_initial_solution
from run_alns import build_operators, infer_size

INSTANCE = "data/Instance10/R_30_10_2.txt"


def run_alns(instance, evaluator, config, seed, enable_step6,
             enable_mlns_polish, enable_rechain_polish):
    cfg = config
    sa_config = cfg.build_sa_config_dict()
    sa_config["iterations"] = args.iters
    sa_config["size"] = infer_size(instance)
    sa_config["drone_reanchor_ls_enabled"] = enable_step6
    sa_config["drone_composite_reanchor_enabled"] = False
    sa_config["drone_sortie_constructor_enabled"] = False
    sa_cfg = SANNCfg(**sa_config)
    rng = random.Random(seed)

    destroy_ops, repair_ops = build_operators(
        instance, seed,
        drone_priority=cfg.drone_priority,
        repair_set="all",
        enable_composite=True,
        drone_bonus_kwargs=cfg.drone_bonus,
        forced_drone_customers=cfg.forced_drone_customers,
        robust_energy_mode="embedded",
    )

    initial = build_two_phase_initial_solution(
        instance,
        truck_forbidden_customers=cfg.forced_drone_customers,
        allow_multiple_launch_per_node=cfg.relax_allow_multiple_launch_per_node,
    )

    alns = SimulatedAnnealingALNS(
        instance=instance, destroy_ops=destroy_ops, repair_ops=repair_ops,
        evaluator=evaluator, cfg=sa_cfg, rng=rng,
    )

    if enable_mlns_polish:
        alns._cfg.matheuristic_lns_enabled = True

    best_sol = alns.run(initial)

    best_sol, mlns_detail = _apply_mlns_if_needed(
        best_sol, evaluator, instance, config, rng,
        enable_mlns=enable_mlns_polish)

    best_sol, rechain_detail = _apply_rechain_if_needed(
        best_sol, evaluator, instance, config, rng,
        enable_rechain=enable_rechain_polish)

    ev = evaluator.evaluate_solution(best_sol)
    detail = {
        "cost": round(ev.total_cost, 2),
        "feasible": ev.feasible,
        "truck_cost": round(ev.truck_distance_cost, 2),
        "drone_cost": round(ev.drone_distance_cost, 2),
        "delay": round(ev.delay_penalty, 2),
        "runtime": 0,
        "trucks": [r.nodes for r in best_sol.truck_routes],
        "drone_tasks": [
            f"D{t.drone_id}: T{t.launch_truck}@{t.launch_node} -> "
            f"{t.customers()} -> T{t.land_truck}@{t.retrieve_node}"
            for t in best_sol.drone_tasks
        ],
    }
    if mlns_detail:
        detail["mlns_improvement"] = mlns_detail
    if rechain_detail:
        detail["rechain_improvement"] = rechain_detail
    return detail, best_sol


def _apply_mlns_if_needed(best_sol, evaluator, instance, config, rng, enable_mlns):
    if not enable_mlns:
        return best_sol, None

    from alns_vrpfd.core.operators.matheuristic_lns import MatheuristicLNSRepair

    best_cost = evaluator.evaluate_solution(best_sol).total_cost
    if not math.isfinite(best_cost):
        return best_sol, None

    trials = getattr(config, "matheuristic_lns_trials", 5)
    max_cust = getattr(config, "matheuristic_lns_max_customers", 3)
    original_cost = best_cost
    improvements = []

    for trial in range(trials):
        lns = MatheuristicLNSRepair(
            instance=instance,
            evaluator=evaluator,
            max_customers=max_cust,
            max_anchor_dist_factor=2.0,
            energy_tolerance=1.0,
            rng=random.Random(rng.randint(0, 2**31)),
        )
        improved = lns.apply(best_sol)
        imp_cost = evaluator.evaluate_solution(improved).total_cost
        if math.isfinite(imp_cost) and imp_cost < best_cost - 1e-6:
            improvements.append(f"{best_cost:.2f}->{imp_cost:.2f}")
            best_sol = improved
            best_cost = imp_cost

    detail = None
    if best_cost < original_cost:
        detail = {
            "before": round(original_cost, 2),
            "after": round(best_cost, 2),
            "trials": trials,
            "improvements": improvements,
        }
    return best_sol, detail


def _apply_rechain_if_needed(best_sol, evaluator, instance, config, rng, enable_rechain):
    if not enable_rechain:
        return best_sol, None

    from alns_vrpfd.core.operators.truck_drone_rechain import TruckDroneRechainRepair

    best_cost = evaluator.evaluate_solution(best_sol).total_cost
    if not math.isfinite(best_cost):
        return best_sol, None

    original_cost = best_cost
    improvements = []
    trials = 5

    for trial in range(trials):
        rechain = TruckDroneRechainRepair(
            instance=instance,
            evaluator=evaluator,
            max_customers=3,
            max_anchor_dist_factor=2.0,
            energy_tolerance=1.0,
            rng=random.Random(rng.randint(0, 2**31)),
        )
        improved = rechain.apply(best_sol)
        imp_cost = evaluator.evaluate_solution(improved).total_cost
        if math.isfinite(imp_cost) and imp_cost < best_cost - 1e-6:
            improvements.append(
                f"{best_cost:.2f}->{imp_cost:.2f} "
                f"(trucks={[r.nodes for r in improved.truck_routes]})"
            )
            best_sol = improved
            best_cost = imp_cost

    detail = None
    if best_cost < original_cost:
        detail = {
            "before": round(original_cost, 2),
            "after": round(best_cost, 2),
            "trials": trials,
            "improvements": improvements,
            "final_trucks": [r.nodes for r in best_sol.truck_routes],
            "final_drones": [
                f"D{t.drone_id}: T{t.launch_truck}@{t.launch_node} -> "
                f"{t.customers()} -> T{t.land_truck}@{t.retrieve_node}"
                for t in best_sol.drone_tasks
            ],
        }
    return best_sol, detail


def run_mip_with_verification(instance, evaluator, config):
    from alns_vrpfd.mip.builder import build_mip_model

    start = time.perf_counter()
    artifacts = build_mip_model(
        instance,
        epsilon=1e-3, energy_budget=3, num_segments=10,
        use_gurobi_pwl=True, robust_energy=True,
        big_m_time=1000, big_m_load=1000, big_m_energy=20.0,
        tardiness_weight=1.0,
        cost_lambda=config.cost_lambda, cost_rho=config.cost_rho,
        cost_normalized=config.cost_normalized,
    )
    model = artifacts.model
    model.setParam("TimeLimit", 3600)
    model.setParam("MIPGap", 0.001)
    model.setParam("Threads", 4)
    model.setParam("OutputFlag", 1)
    model.optimize()
    runtime = time.perf_counter() - start

    result = {
        "mip_objective": round(model.ObjVal, 4) if model.SolCount > 0 else None,
        "runtime": round(runtime, 2),
        "status": model.Status,
        "mip_gap": round(model.MIPGap, 4) if model.SolCount > 0 else None,
    }

    if model.SolCount == 0:
        result["error"] = "No feasible solution found"
        return result

    from alns_vrpfd.mip.run_mip import _verify_with_alns_evaluator
    data = artifacts.data
    vars = artifacts.variables

    truck_routes_raw = []
    for k in data.trucks:
        arcs = [(i, j) for (i, j) in data.arcs if vars.x_truck[i, j, k].X > 0.5]
        if arcs:
            from alns_vrpfd.mip.run_mip import _reconstruct_routes
            routes = _reconstruct_routes(arcs)
            route = routes[0] if routes else []
            distance = sum(data.truck_distance[(i, j)] for i, j in arcs)
            truck_routes_raw.append({"truck": k, "route": route, "distance": distance})

    drone_tasks_raw = []
    for d in data.drones:
        arcs = [(i, j) for (i, j) in data.arcs if vars.y_drone[i, j, d].X > 0.5]
        if arcs:
            arrival_times = {i: vars.arrival_drone[i, d].X for i in data.nodes}
            routes = _reconstruct_routes(arcs, arrival_times)
            routes = [r for r in routes if len(r) > 2]
            if routes:
                distance = sum(data.drone_distance[(i, j)] for i, j in arcs)
                drone_tasks_raw.append({"drone": d, "routes": routes, "distance": distance})

    alns_verify = _verify_with_alns_evaluator(
        instance, data, vars, truck_routes_raw, drone_tasks_raw, model.ObjVal)
    result["alns_verification"] = alns_verify

    verified_cost = alns_verify.get("alns_cost")
    result["verified_milp_cost"] = verified_cost
    result["verified_feasible"] = alns_verify.get("alns_feasible", False)

    if verified_cost and alns_verify.get("alns_feasible"):
        gap = (verified_cost - model.ObjVal) / model.ObjVal * 100
        result["mip_vs_verified_gap"] = round(gap, 2)
        print(f"  MILP obj={model.ObjVal:.4f}, ALNS verified={verified_cost:.4f}, "
              f"gap={gap:.2f}%")
    else:
        result["mip_vs_verified_gap"] = None
        print(f"  MILP obj={model.ObjVal:.4f}, ALNS verification: INFEASIBLE")
        for v in alns_verify.get("robustness_violations", []):
            print(f"    D{v['drone_id']}: energy={v['worst_case_energy']:.4f}, "
                  f"budget={v['capacity']:.4f}, margin={v['margin']:.4f}")

    return result


def summarize(label, results):
    costs = [r["cost"] for r in results if r["feasible"]]
    if not costs:
        print(f"  {label}: no feasible solutions")
        return {"label": label, "n": 0}
    mdl = {
        "label": label, "n": len(costs),
        "mean": round(statistics.mean(costs), 2),
        "median": round(statistics.median(costs), 2),
        "min": min(costs), "max": max(costs),
        "std": round(statistics.stdev(costs), 2) if len(costs) > 1 else 0,
        "min_solution": min(results, key=lambda r: r.get("cost", float("inf"))),
    }
    print(f"  {label}: mean={mdl['mean']} median={mdl['median']} "
          f"min={mdl['min']} max={mdl['max']}")
    return mdl


def main():
    global args
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--iters", type=int, default=4000)
    p.add_argument("--skip-mip", action="store_true", help="Skip MILP (long run)")
    args = p.parse_args()

    seeds = list(range(42, 42 + args.seeds))
    instance = read_instance(INSTANCE, strategy="class_based")
    evaluator = Evaluator(instance)
    config = ALNSConfig()

    results = {}
    print(f"Instance: {INSTANCE}  Seeds: {seeds}  Iters: {args.iters}")
    print("=" * 70)

    configs = [
        ("A_baseline", False, False, False),
        ("B_step6", True, False, False),
        ("C_step6_mlns", True, True, False),
        ("D_step6_mlns_rechain", True, True, True),
    ]

    for label, step6, mlns, rechain in configs:
        print(f"\n--- {label} ---")
        run_results = []
        for seed in seeds:
            t_start = time.perf_counter()
            detail, sol = run_alns(instance, evaluator, config, seed,
                                    enable_step6=step6,
                                    enable_mlns_polish=mlns,
                                    enable_rechain_polish=rechain)
            t_end = time.perf_counter()
            detail["runtime"] = round(t_end - t_start, 2)
            run_results.append(detail)
            print(f"  seed {seed}: cost={detail['cost']} "
                  f"feasible={detail['feasible']} t={detail['runtime']}s "
                  f"trucks={detail['trucks']}")
            if detail.get("rechain_improvement"):
                ri = detail["rechain_improvement"]
                print(f"    rechain: {ri['before']}->{ri['after']} "
                      f"trucks={ri['final_trucks']}")
        results[label] = run_results
        summarize(label, run_results)

    if not args.skip_mip:
        print(f"\n--- E: MILP + ALNS verification ---")
        mip_result = run_mip_with_verification(instance, evaluator, config)
        results["E_mip"] = mip_result

    # Save
    outfile = Path("results/ablation_rechain_R30_10_2.json")
    outfile.parent.mkdir(parents=True, exist_ok=True)

    # Remove solution objects (not JSON serializable)
    clean_results = {}
    for label, data in results.items():
        if isinstance(data, list):
            clean_results[label] = data
        else:
            clean_results[label] = data

    with open(outfile, "w") as f:
        json.dump(clean_results, f, indent=2, default=str)
    print(f"\nSaved to {outfile}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Config':25s}  {'Mean':>7s} {'Median':>7s} {'Min':>7s} {'Max':>7s}  {'n':>3s}")
    print("-" * 70)
    for label in ["A_baseline", "B_step6", "C_step6_mlns", "D_step6_mlns_rechain"]:
        if label in results:
            costs = [r["cost"] for r in results[label] if r["feasible"]]
            if costs:
                print(f"{label:25s}  {statistics.mean(costs):7.2f} "
                      f"{statistics.median(costs):7.2f} {min(costs):7.2f} "
                      f"{max(costs):7.2f}  {len(costs):3d}")

    if "E_mip" in results:
        mip = results["E_mip"]
        print(f"{'E_mip (raw objective)':25s}  {mip.get('mip_objective', 'N/A')}")
        print(f"{'E_mip (ALNS verified)':25s}  {mip.get('verified_milp_cost', 'N/A')}")
        print(f"{'E_mip (feasible?)':25s}  {mip.get('verified_feasible', 'N/A')}")

    print("=" * 70)


if __name__ == "__main__":
    main()
