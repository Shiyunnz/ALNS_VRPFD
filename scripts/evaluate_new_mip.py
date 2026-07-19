#!/usr/bin/env python3
"""Evaluate the new tightened MIP solutions and compare with ALNS baseline."""

import sys, json, math, random
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from alns_vrpfd.utils.io_utils import read_instance
from alns_vrpfd.utils.config_loader import ALNSConfig
from alns_vrpfd.evaluation.evaluator import Evaluator
from alns_vrpfd.model import Solution, TruckRoute, DroneTask
from alns_vrpfd.core.operators.base import _build_payloads

MIP_NEW_DIR = project_root / "results" / "MIPresult_new"
MIP_OLD_DIR = project_root / "results" / "MIPresult"
COMPARISON_FILE = project_root / "results" / "instance10_comparison.json"

def load_mip_json(short_name):
    path = MIP_NEW_DIR / f"{short_name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def reconstruct_from_mip_json(data, instance):
    """Reconstruct Solution from corrected MIP JSON output."""
    from alns_vrpfd.core.operators.base import _build_payloads

    em = instance.customer_manager
    dm = {c.customer_id: c.demand for c in em.customers()}

    truck_routes_raw = {}
    for tr in data.get("truck_routes", []):
        truck_routes_raw[tr["truck"]] = tr["route"]

    drone_tasks_raw = data.get("drone_tasks", [])

    # Collect all drone-served customers
    drone_customers = set()
    for dt in drone_tasks_raw:
        for r in dt.get("routes", []):
            drone_customers.update(r)

    # Remove drone customers from truck routes
    nodes_to_remove = drone_customers.copy()
    for dt in drone_tasks_raw:
        launch = dt.get("launch_node")
        retrieve = dt.get("retrieve_node")
        if launch in nodes_to_remove:
            nodes_to_remove.remove(launch)
        if retrieve in nodes_to_remove:
            nodes_to_remove.remove(retrieve)

    sol = Solution()
    tid_counter = 0
    for tid in sorted(truck_routes_raw.keys()):
        filtered = [n for n in truck_routes_raw[tid] if n not in nodes_to_remove]
        sol.add_truck_route(TruckRoute(route_id=tid, nodes=filtered,
                                       capacity=instance.vehicle_specs['truck'].capacity))

    # Build node -> truck map
    node_to_truck = {}
    for tid, nodes in truck_routes_raw.items():
        for pos, n in enumerate(nodes):
            node_to_truck[n] = (tid, pos)

    nid_counter = 0
    for dt in drone_tasks_raw:
        for r in dt.get("routes", []):
            if len(r) < 2:
                continue
            launch_node = r[0]
            retrieve_node = r[-1]
            customers = r[1:-1]
            if not customers:
                continue
            lt_info = node_to_truck.get(launch_node)
            rt_info = node_to_truck.get(retrieve_node)
            lt = lt_info[0] if lt_info else None
            rt = rt_info[0] if rt_info else None
            payloads = _build_payloads(customers, dm)
            task = DroneTask(
                task_id=nid_counter, drone_id=dt["drone"],
                launch_truck=lt, launch_node=launch_node,
                customers=customers,
                land_truck=rt, retrieve_node=retrieve_node,
                payloads=payloads,
            )
            nid_counter += 1
            sol.add_drone_task(task)

    return sol

def reconstruct_r30_10_2_known(instance):
    """Manually reconstruct R_30_10_2 from known MIP structure."""
    from alns_vrpfd.core.operators.base import _build_payloads
    em = instance.customer_manager
    dm = {c.customer_id: c.demand for c in em.customers()}

    sol = Solution()
    sol.add_truck_route(TruckRoute(route_id=0, nodes=[0, 1, 11],
                                   capacity=instance.vehicle_specs['truck'].capacity))
    sol.add_truck_route(TruckRoute(route_id=1, nodes=[0, 3, 5, 11],
                                   capacity=instance.vehicle_specs['truck'].capacity))

    d0p = _build_payloads([7, 6], dm)
    sol.add_drone_task(DroneTask(task_id=0, drone_id=0,
        launch_truck=0, launch_node=1, customers=[7, 6],
        land_truck=0, retrieve_node=11, payloads=d0p))

    d1a = _build_payloads([4], dm)
    sol.add_drone_task(DroneTask(task_id=1, drone_id=1,
        launch_truck=0, launch_node=1, customers=[4],
        land_truck=1, retrieve_node=3, payloads=d1a))

    d1b = _build_payloads([10, 2], dm)
    sol.add_drone_task(DroneTask(task_id=2, drone_id=1,
        launch_truck=1, launch_node=3, customers=[10, 2],
        land_truck=1, retrieve_node=5, payloads=d1b))

    d1c = _build_payloads([9, 8], dm)
    sol.add_drone_task(DroneTask(task_id=3, drone_id=1,
        launch_truck=1, launch_node=5, customers=[9, 8],
        land_truck=0, retrieve_node=11, payloads=d1c))

    return sol

def evaluate_solution(sol, evaluator):
    ev = evaluator.evaluate_solution(sol)
    details = evaluator.evaluate_with_details(sol)
    margin = min((b.margin for b in details.robustness.task_breakdown), default=float('inf'))
    energy_violation = margin < 0
    energy_margin_over_cap = margin / 6.3 * 100 if hasattr(details.robustness.task_breakdown[0], 'capacity') and details.robustness.task_breakdown[0].capacity > 0 else 0
    return {
        "feasible": ev.feasible,
        "cost": round(ev.total_cost, 2) if ev.feasible else None,
        "energy_violation": energy_violation,
        "worst_margin": round(margin, 4),
        "truck_cost": round(ev.truck_distance_cost, 2),
        "drone_cost": round(ev.drone_distance_cost, 2),
        "delay": round(ev.delay_penalty, 2),
        "violations": {
            "task_violation": evaluator._has_drone_task_violations(sol),
            "coverage": evaluator._has_customer_coverage_violation(sol),
            "robustness": not details.robustness.feasible,
        },
        "truck_routes": [r.nodes for r in sol.truck_routes],
        "drone_tasks": [
            f"D{t.drone_id}: T{t.launch_truck}@{t.launch_node} -> {t.customers()} -> T{t.land_truck}@{t.retrieve_node}"
            for t in sol.drone_tasks
        ],
    }

def main():
    cfg = ALNSConfig()

    instances = [
        ("R_30_10_1", "data/Instance10/R_30_10_1.txt"),
        ("R_30_10_2", "data/Instance10/R_30_10_2.txt"),
        ("R_30_10_3", "data/Instance10/R_30_10_3.txt"),
        ("R_30_10_4", "data/Instance10/R_30_10_4.txt"),
        ("R_30_10_5", "data/Instance10/R_30_10_5.txt"),
    ]

    # Load old comparison data
    old_comparison = {}
    if COMPARISON_FILE.exists():
        with open(COMPARISON_FILE) as f:
            old_comparison = json.load(f)

    results = {}
    for name, path in instances:
        print(f"\n{'='*60}")
        print(f"{name}")
        print(f"{'='*60}")

        short_name = name.replace("R_", "")
        instance = read_instance(path, strategy=cfg.time_window_strategy)
        evaluator = Evaluator(instance)

        # Get MIP solution
        mip_data = load_mip_json(short_name)
        if mip_data and mip_data.get("status") in (2, 3):
            sol = reconstruct_from_mip_json(mip_data, instance)
            mip_source = "json"
        elif name == "R_30_10_2":
            sol = reconstruct_r30_10_2_known(instance)
            mip_source = "manual"
        else:
            mip_source = "none"

        mip_eval = None
        if mip_source != "none":
            mip_eval = evaluate_solution(sol, evaluator)
            r = mip_eval
            print(f"  MIP evaluation (source={mip_source}):")
            print(f"    Feasible: {r['feasible']}, Cost: {r['cost']}")
            print(f"    Violations: {r['violations']}")
            if r['energy_violation']:
                print(f"    ** Energy margin={r['worst_margin']:.4f} ({(r['worst_margin']/6.3*100):.1f}% over)")
            print(f"    Trucks: {r['truck_routes']}")
            for t in r['drone_tasks']:
                print(f"    {t}")

        # Compare with ALNS
        old_data = old_comparison.get(name, {})
        if old_data:
            alns_a = [s["cost"] for s in old_data.get("A_baseline", []) if s.get("feasible")]
            alns_b = [s["cost"] for s in old_data.get("B_step6", []) if s.get("feasible")]
            alns_min = min(alns_a + alns_b) if (alns_a + alns_b) else None
            mlns = old_data.get("C_mlns", {})
            mlns_cost = mlns.get("cost") if mlns else None

            print(f"  ALNS baseline min: {alns_min}")
            print(f"  MLNS after: {mlns_cost}")
            if mip_eval and mip_eval.get("cost") and alns_min:
                gap = (alns_min - mip_eval["cost"]) / mip_eval["cost"] * 100
                print(f"  ALNS vs MIP gap: {gap:.2f}%")

        results[name] = {
            "mip_source": mip_source,
        }

    print(f"\n{'='*60}")
    print("Done")

if __name__ == "__main__":
    main()
