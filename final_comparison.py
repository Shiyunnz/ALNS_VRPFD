#!/usr/bin/env python3
"""A/B/C/D comparison: R_40_10_1 multi-seed.

A: ALNS baseline (no drone LS)
B: ALNS + Step6 (drone_reanchor_ls)
C: ALNS + Step6 + MLNS final polish
D: MILP optimum

Usage: python scripts/final_comparison.py [--seeds 10] [--iters 4000]
"""

import sys, time, json, random, math, statistics, argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from alns_vrpfd.utils.io_utils import read_instance
from alns_vrpfd.utils.config_loader import ALNSConfig
from alns_vrpfd.evaluation.evaluator import Evaluator
from alns_vrpfd.core.sa import SANNCfg, SimulatedAnnealingALNS
from alns_vrpfd.mip.builder import build_mip_model
from alns_vrpfd.evaluation.run_record import reconstruct_solution_from_mip
from run_alns import build_operators, infer_size
from alns_vrpfd.model.initializer import build_two_phase_initial_solution

INSTANCE = "data/Instance10/R_40_10_1.txt"


def run_alns(instance, evaluator, config, seed, enable_step6, enable_mlns_polish):
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
        # Temporarily enable MLNS for the polish phase
        alns._cfg.matheuristic_lns_enabled = True
        start = time.perf_counter()
        best_sol = alns.run_with_matheuristic_lns_polish(initial)
    else:
        start = time.perf_counter()
        best_sol = alns.run(initial)
    runtime = time.perf_counter() - start

    ev = evaluator.evaluate_solution(best_sol)
    return {
        "cost": round(ev.total_cost, 2),
        "feasible": ev.feasible,
        "runtime": round(runtime, 2),
        "trucks": [r.nodes for r in best_sol.truck_routes],
        "drone_tasks": [
            f"D{t.drone_id}: [{t.launch_node}->{t.customers()}->{t.retrieve_node}] lt={t.launch_truck} rt={t.land_truck}"
            for t in best_sol.drone_tasks
        ],
    }


def run_mip(instance, evaluator, config):
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
    model.setParam("TimeLimit", 120)
    model.setParam("MIPGap", 0.001)
    model.setParam("Threads", 4)
    model.setParam("OutputFlag", 0)
    model.optimize()
    runtime = time.perf_counter() - start

    sol = reconstruct_solution_from_mip(artifacts)
    if sol:
        ev = evaluator.evaluate_solution(sol)
        return {
            "cost": round(ev.total_cost, 2),
            "feasible": ev.feasible,
            "runtime": round(runtime, 2),
            "status": model.Status,
            "obj_val": round(model.ObjVal, 4),
            "gap": round(model.MIPGap, 4),
            "trucks": [r.nodes for r in sol.truck_routes],
            "drone_tasks": [
                f"D{t.drone_id}: [{t.launch_node}->{t.customers()}->{t.retrieve_node}]"
                for t in sol.drone_tasks
            ],
        }
    return None


def summarize(label, results, mip_cost):
    costs = [r["cost"] for r in results if r["feasible"]]
    if not costs:
        print(f"  {label}: no feasible solutions")
        return {"label": label, "n": 0}
    hits_9718 = sum(1 for c in costs if abs(c - 97.18) < 0.01)
    hits_9760 = sum(1 for c in costs if abs(c - 97.60) < 0.01)
    below_98 = sum(1 for c in costs if c < 98)
    mdl = {"label": label, "n": len(costs), "mean": round(statistics.mean(costs), 2),
           "median": round(statistics.median(costs), 2), "min": min(costs), "max": max(costs),
           "std": round(statistics.stdev(costs), 2) if len(costs) > 1 else 0,
           "hits_97.18": hits_9718, "hits_97.60": hits_9760, "below_98": below_98,
           "gap_vs_milp": f"{(min(costs) - mip_cost) / mip_cost * 100:.2f}%",
           "avg_runtime": round(statistics.mean([r["runtime"] for r in results]), 2)}
    print(f"  {label}: mean={mdl['mean']} median={mdl['median']} min={mdl['min']} "
          f"97.18×{hits_9718} 97.60×{hits_9760} <98×{below_98} /{len(costs)}")
    return mdl


def main():
    global args
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--iters", type=int, default=4000)
    args = p.parse_args()

    seeds = list(range(42, 42 + args.seeds))
    instance = read_instance(INSTANCE, strategy="class_based")
    evaluator = Evaluator(instance)
    config = ALNSConfig()

    results = {}
    print(f"Instance: {INSTANCE}  Seeds: {list(seeds)}  Iters: {args.iters}")

    # A: ALNS baseline
    print("\n--- A: ALNS baseline ---")
    alns_base = []
    for seed in seeds:
        r = run_alns(instance, evaluator, config, seed, enable_step6=False, enable_mlns_polish=False)
        alns_base.append(r)
        print(f"  seed {seed}: cost={r['cost']} t={r['runtime']}s")
    results["A_baseline"] = alns_base

    # B: ALNS + Step6
    print("\n--- B: ALNS + Step6 ---")
    alns_step6 = []
    for seed in seeds:
        r = run_alns(instance, evaluator, config, seed, enable_step6=True, enable_mlns_polish=False)
        alns_step6.append(r)
        print(f"  seed {seed}: cost={r['cost']} t={r['runtime']}s")
    results["B_step6"] = alns_step6

    # C: ALNS + Step6 + MLNS final polish
    print("\n--- C: ALNS + Step6 + MLNS final polish ---")
    alns_mlns = []
    for seed in seeds:
        r = run_alns(instance, evaluator, config, seed, enable_step6=True, enable_mlns_polish=True)
        alns_mlns.append(r)
        print(f"  seed {seed}: cost={r['cost']} t={r['runtime']}s")
    results["C_step6_mlns"] = alns_mlns

    # D: MILP
    print("\n--- D: MILP ---")
    mip_result = run_mip(instance, evaluator, config)
    results["D_mip"] = mip_result
    if mip_result:
        print(f"  cost={mip_result['cost']} obj={mip_result['obj_val']} "
              f"gap={mip_result['gap']:.2%} time={mip_result['runtime']}s")
    mip_cost = mip_result["cost"] if mip_result else float("inf")

    # Summary
    print("\n" + "=" * 65)
    print(f"{'Config':20s}  {'Mean':>6s} {'Median':>7s} {'Min':>6s} {'97.18':>6s} {'97.60':>6s} {'<98':>5s}  Gap")
    print("-" * 65)
    out = {}
    for label, data, hit97 in [("A_baseline", alns_base, None),
                                 ("B_step6", alns_step6, None),
                                 ("C_step6_mlns", alns_mlns, None)]:
        costs = [r["cost"] for r in data if r["feasible"]]
        if not costs: continue
        h18 = sum(1 for c in costs if abs(c - 97.18) < 0.01)
        h60 = sum(1 for c in costs if abs(c - 97.60) < 0.01)
        lt98 = sum(1 for c in costs if c < 98)
        gap = (min(costs) - mip_cost) / mip_cost * 100
        print(f"{label:20s}  {statistics.mean(costs):6.2f} {statistics.median(costs):7.2f} {min(costs):6.2f} "
              f"{h18:6d} {h60:6d} {lt98:5d}  {gap:+.2f}%")
        out[label] = summarize(label, data, mip_cost)

    if mip_result:
        print(f"{'D_mip':20s}  {mip_cost:6.2f}")
        out["D_mip"] = mip_result

    out["_meta"] = {"instance": INSTANCE, "seeds": seeds, "iters": args.iters, "mip_cost": mip_cost}

    # Save
    outfile = Path("results/final_comparison.json")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        json.dump({"results": results, "summary": out}, f, indent=2)
    print(f"\nSaved to {outfile}")
    print("=" * 65)


if __name__ == "__main__":
    main()
