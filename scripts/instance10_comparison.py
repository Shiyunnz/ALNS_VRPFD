#!/usr/bin/env python3
"""Comprehensive ALNS vs MILP comparison for all 5 R_30_10 instances.

For each instance:
  1. Load existing MILP best from results/MIPresult/
  2. Run ALNS baseline (A) + Step6 (B) — 5 seeds × 4000 iters each
  3. If ALNS misses MILP best, run MLNS final polish on B's best
  4. Report whether ALNS (possibly +MLNS) matches MILP

Usage: python scripts/instance10_comparison.py [--seeds 5]
"""

import sys, time, json, random, math, statistics, argparse, re, os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from alns_vrpfd.utils.io_utils import read_instance
from alns_vrpfd.utils.config_loader import ALNSConfig
from alns_vrpfd.evaluation.evaluator import Evaluator
from alns_vrpfd.core.sa import SANNCfg, SimulatedAnnealingALNS
from run_alns import build_operators, infer_size
from alns_vrpfd.model.initializer import build_two_phase_initial_solution
from alns_vrpfd.core.operators.matheuristic_lns import MatheuristicLNSRepair

INSTANCES = [
    ("R_30_10_1", "data/Instance10/R_30_10_1.txt"),
    ("R_30_10_2", "data/Instance10/R_30_10_2.txt"),
    ("R_30_10_3", "data/Instance10/R_30_10_3.txt"),
    ("R_30_10_4", "data/Instance10/R_30_10_4.txt"),
    ("R_30_10_5", "data/Instance10/R_30_10_5.txt"),
]

MIP_DIR = project_root / "results" / "MIPresult"


def parse_mip_output(name: str):
    """Parse existing MIP output file. Returns None if not found."""
    path = MIP_DIR / f"output_{name}.txt"
    if not path.exists():
        return None
    text = path.read_text()
    obj = re.search(r"Best objective\s+([\d.e+\-]+)", text)
    gap = re.search(r"gap\s+([\d.]+)%", text)
    status = re.search(r"Status:\s*(\d+)", text)
    return {
        "cost": float(obj.group(1)) if obj else None,
        "gap": float(gap.group(1)) if gap else None,
        "status": int(status.group(1)) if status else None,
        "optimal": status and int(status.group(1)) == 2,
    }


def reconstruct_mip_solution(short_name: str, instance):
    """Parse MIP output and reconstruct solution, evaluating with exact cost.
    
    short_name: e.g. '30_10_5' (without R_ prefix — matches MIP output filename)
    """
    from alns_vrpfd.model import Solution, TruckRoute, DroneTask
    from alns_vrpfd.core.operators.base import _build_payloads

    path = MIP_DIR / f"output_{short_name}.txt"
    if not path.exists():
        return None
    text = path.read_text()

    # Parse truck routes
    truck_matches = re.findall(r"Truck (\d+): \[([^\]]+)\]", text)
    if not truck_matches:
        return None

    truck_routes_raw = {}
    for tid_str, nodes_str in truck_matches:
        nodes = [int(n.strip()) for n in nodes_str.split(",") if n.strip()]
        truck_routes_raw[int(tid_str)] = nodes

    # Parse drone tasks from "Drone Tasks:" section
    drone_section = re.search(r"Drone Tasks:(.*?)(?=Truck Routes:|\Z)", text, re.DOTALL)
    if not drone_section:
        return None

    task_data = []  # (drone_id, launch_node, customers, retrieve_node)
    for block in re.findall(
        r"Drone (\d+):(.*?)(?=Drone \d+:|$)", drone_section.group(), re.DOTALL
    ):
        did = int(block[0])
        for m in re.findall(r"Task: Launch (\d+) -> Serves \[([^\]]*)\] -> Land (\d+)", block[1]):
            ln = int(m[0])
            cc = [int(x.strip()) for x in m[1].split(",") if x.strip()]
            rn = int(m[2])
            task_data.append((did, ln, cc, rn))

    # Build truck route lookup: node -> (truck_id, position_index)
    node_to_truck = {}
    for tid, nodes in truck_routes_raw.items():
        for pos, nid in enumerate(nodes):
            node_to_truck[nid] = (tid, pos)

    # Build solution
    em = instance.customer_manager
    customers_list = list(em.customers())
    dm = {c.customer_id: c.demand for c in customers_list}

    # Collect all drone-served customers (to remove from truck routes)
    drone_customers = set()
    launch_retrieve_nodes = set()
    for did, ln, cc, rn in task_data:
        drone_customers.update(cc)
        launch_retrieve_nodes.add(ln)
        launch_retrieve_nodes.add(rn)

    # Only remove nodes that are exclusively drone-served (not launch/retrieve points)
    nodes_to_remove = drone_customers - launch_retrieve_nodes

    sol = Solution()
    nid_counter = 0

    for tid in sorted(truck_routes_raw.keys()):
        filtered_nodes = [n for n in truck_routes_raw[tid] if n not in nodes_to_remove]
        sol.add_truck_route(TruckRoute(route_id=tid, nodes=filtered_nodes, capacity=instance.vehicle_specs['truck'].capacity))

    for did, ln, cc, rn in task_data:
        if not cc:
            continue  # skip empty tasks
        # Determine launch truck and land truck
        lt_info = node_to_truck.get(ln)
        rt_info = node_to_truck.get(rn)
        lt = lt_info[0] if lt_info else None
        rt = rt_info[0] if rt_info else None
        payloads = _build_payloads(cc, dm)
        task = DroneTask(
            task_id=nid_counter, drone_id=did,
            launch_truck=lt, launch_node=ln,
            customers=cc, land_truck=rt, retrieve_node=rn,
            payloads=payloads,
        )
        nid_counter += 1
        sol.add_drone_task(task)

    return sol


def run_alns(instance, evaluator, config, seed, enable_step6):
    cfg = config
    sa_config = cfg.build_sa_config_dict()
    sa_config["iterations"] = 4000
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
        "solution": best_sol,
    }


def run_mlns_polish(instance, evaluator, solution, n_trials=16):
    """Apply MLNS final polish on a solution. Returns improved solution."""
    best = solution
    best_cost = evaluator.evaluate_solution(solution).total_cost
    start = time.perf_counter()
    for trial in range(n_trials):
        lns = MatheuristicLNSRepair(
            instance=evaluator._instance,
            evaluator=evaluator,
            max_customers=4,
            max_anchor_dist_factor=2.0,
            energy_tolerance=1.0,
            rng=random.Random(trial * 7 + 3),
        )
        improved = lns.apply(best)
        ic = evaluator.evaluate_solution(improved).total_cost
        if math.isfinite(ic) and ic < best_cost - 1e-6:
            best = improved
            best_cost = ic
    runtime = time.perf_counter() - start
    ev = evaluator.evaluate_solution(best)
    return {
        "cost": round(ev.total_cost, 2),
        "feasible": ev.feasible,
        "improved": best_cost < evaluator.evaluate_solution(solution).total_cost - 1e-6,
        "runtime": round(runtime, 2),
        "trucks": [r.nodes for r in best.truck_routes],
        "drone_tasks": [
            f"D{t.drone_id}: [{t.launch_node}->{t.customers()}->{t.retrieve_node}] lt={t.launch_truck} rt={t.land_truck}"
            for t in best.drone_tasks
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--mlns-trials", type=int, default=8)
    parser.add_argument("--output", default="results/instance10_comparison.json")
    args = parser.parse_args()

    seeds = list(range(42, 42 + args.seeds))
    all_results = {}

    for name, path in INSTANCES:
        short_name = name.replace("R_", "")  # e.g. R_30_10_1 -> 30_10_1
        print(f"\n{'=' * 60}")
        print(f"{name}")
        print(f"{'=' * 60}")

        config = ALNSConfig()
        instance = read_instance(path, strategy=config.time_window_strategy)
        evaluator = Evaluator(instance)
        inst_data = {"instance": name, "short_name": short_name, "seeds": seeds}

        # 1. MILP: reconstruct from existing output + evaluate with exact cost
        mip_sol = reconstruct_mip_solution(short_name, instance)
        mip_meta = parse_mip_output(short_name) if not None else None
        mip_pwl_cost = mip_meta["cost"] if mip_meta else None
        if mip_sol:
            mip_ev = evaluator.evaluate_solution(mip_sol)
            mip_cost_exact = round(mip_ev.total_cost, 2) if mip_ev.feasible else None
            mip_feasible = mip_ev.feasible
            optimal_str = "✓ optimal" if (mip_meta and mip_meta.get("optimal")) else f"gap={mip_meta['gap']:.2f}%" if (mip_meta and mip_meta.get("gap")) else "N/A"
            if mip_cost_exact is not None:
                print(f"  MILP exact: {mip_cost_exact:.2f} ({optimal_str})")
            else:
                print(f"  MILP: PWL={mip_pwl_cost:.2f} ({optimal_str}) but exact eval infeasible (cross-truck pattern)")
            print(f"    Trucks: {[r.nodes for r in mip_sol.truck_routes]}")
            print(f"    Tasks: {[(t.drone_id, t.launch_node, t.customers(), t.retrieve_node) for t in mip_sol.drone_tasks]}")
        else:
            mip_cost_exact = None
            mip_pwl_cost = None
            print(f"  MILP: no reconstructable solution")
        inst_data["mip_cost_exact"] = mip_cost_exact
        inst_data["mip_pwl_cost"] = mip_pwl_cost
        # Use exact if feasible, else PWL as reference
        benchmark_cost = mip_cost_exact if mip_cost_exact is not None else mip_pwl_cost

        # 2. ALNS A (baseline)
        print(f"  --- A: ALNS baseline ---")
        alns_a_seeds = []
        for seed in seeds:
            r = run_alns(instance, evaluator, config, seed, enable_step6=False)
            alns_a_seeds.append(r)
            print(f"    seed {seed}: cost={r['cost']} t={r['runtime']}s")
        inst_data["A_baseline"] = alns_a_seeds

        # 3. ALNS B (Step6)
        print(f"  --- B: ALNS + Step6 ---")
        alns_b_seeds = []
        for seed in seeds:
            r = run_alns(instance, evaluator, config, seed, enable_step6=True)
            alns_b_seeds.append(r)
            print(f"    seed {seed}: cost={r['cost']} t={r['runtime']}s")
        inst_data["B_step6"] = alns_b_seeds

        # Find best ALNS cost across all configs
        all_costs = [r["cost"] for r in alns_a_seeds + alns_b_seeds if r["feasible"]]
        alns_min = min(all_costs) if all_costs else float("inf")

        # 4. MLNS final polish if ALNS misses benchmark
        mlns_result = None
        if benchmark_cost is not None and alns_min > benchmark_cost + 0.01:
            # Find the best solution (from either A or B)
            best_sol = None
            best_cost = float("inf")
            for r in alns_b_seeds:  # prefer B (Step6 has better performance)
                if r["feasible"] and r["cost"] < best_cost:
                    best_cost = r["cost"]
                    best_sol = r["solution"]

            if best_sol is not None:
                print(f"  --- C: MLNS polish (ALNS min={alns_min:.2f} vs benchmark={benchmark_cost:.2f}) ---")
                mlns_result = run_mlns_polish(instance, evaluator, best_sol, args.mlns_trials)
                print(f"    After MLNS: cost={mlns_result['cost']} improved={mlns_result['improved']} t={mlns_result['runtime']}s")
        else:
            print(f"  ALNS already matches benchmark (min={alns_min:.2f} vs {benchmark_cost:.2f}) — skipping MLNS")

        inst_data["C_mlns"] = mlns_result
        all_results[name] = inst_data

    # Summary table
    print(f"\n{'=' * 85}")
    print(f"{'Instance':12s} {'Benchmark':>9s} {'A_min':>7s} {'B_min':>7s} {'C_after':>8s} {'Matched?':>8s} {'Source':>8s}")
    print(f"{'-' * 85}")

    for name, path in INSTANCES:
        d = all_results[name]
        mip_exact = d.get("mip_cost_exact")
        mip_pwl = d.get("mip_pwl_cost")
        benchmark_cost = mip_exact if mip_exact is not None else mip_pwl
        source = "exact" if mip_exact is not None else ("PWL" if mip_pwl else "N/A")
        a_costs = [r["cost"] for r in d["A_baseline"] if r["feasible"]]
        b_costs = [r["cost"] for r in d["B_step6"] if r["feasible"]]
        a_min = min(a_costs) if a_costs else None
        b_min = min(b_costs) if b_costs else None
        c = d["C_mlns"]
        c_cost = c["cost"] if c else None
        c_str = f"{c_cost:.2f}" if c_cost else "N/A"

        final_min = c_cost if c_cost else (b_min if b_min else a_min)
        matched = "YES" if (final_min is not None and benchmark_cost is not None and final_min <= benchmark_cost + 0.01) else "NO"
        bench_str = f"{benchmark_cost:.2f}" if benchmark_cost else "N/A"
        a_str = f"{a_min:.2f}" if a_min else "N/A"
        b_str = f"{b_min:.2f}" if b_min else "N/A"
        print(f"{name:12s} {bench_str:>9s} {a_str:>7s} {b_str:>7s} {c_str:>8s} {matched:>8s} {source:>8s}")

    # Save
    outfile = Path(args.output)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    # Convert solutions to serializable
    def clean(r):
        if isinstance(r, dict) and "solution" in r:
            r = {k: v for k, v in r.items() if k != "solution"}
        return r
    serializable = {}
    for name, data in all_results.items():
        serializable[name] = {k: v for k, v in data.items()}
        for key in ["A_baseline", "B_step6"]:
            if key in serializable[name]:
                serializable[name][key] = [clean(r) for r in serializable[name][key]]
    with open(outfile, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {outfile}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
