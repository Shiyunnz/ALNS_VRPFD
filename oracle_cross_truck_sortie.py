#!/usr/bin/env python3
"""Offline oracle: exhaustive reconstruction of R_40_10_1 customers {6,8,10}.

Proves whether 97.18 is reachable from 97.60 via local drone sortie
reconstruction (Matheuristic LNS repair).

Set-partition enumeration over {6,8,10} with all anchor permutations,
pruned by quick energy bounds.
"""

import sys, math, itertools, json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from alns_vrpfd.utils.io_utils import read_instance
from alns_vrpfd.evaluation.evaluator import Evaluator
from alns_vrpfd.model import DroneTask, Solution, TruckRoute
from alns_vrpfd.core.operators.base import _build_payloads
from alns_vrpfd.evaluation.energy import DroneEnergyModel
from alns_vrpfd.evaluation.subroute_robust_verifier import _budgeted_sum

INSTANCE_PATH = "data/Instance10/R_40_10_1.txt"
NEIGHBORHOOD = {6, 8, 10}
TRUCK_NODES = {0: [0, 2, 1, 11], 1: [0, 7, 4, 5, 11]}
DRONES = [0, 1]


def truck_of(node: int):
    for tid, nodes in TRUCK_NODES.items():
        if node in nodes:
            return tid
    return None


def set_partitions(xs):
    """All set partitions preserving input order within each block."""
    xs = list(xs)
    if not xs:
        yield []
        return
    first = xs[0]
    for rest in set_partitions(xs[1:]):
        yield [[first]] + rest
        for i, blk in enumerate(rest):
            new = deepcopy(rest)
            new[i] = [first] + blk
            yield new


def quick_energy(inst, launch, customers, retrieve):
    """Worst-case energy for a drone sortie (no full timing)."""
    ni = {n: i for i, n in enumerate(inst.all_node_ids())}
    dt = inst.time_matrix("drone")
    em = DroneEnergyModel()
    rc = inst.robust_config
    dm = {c.customer_id: c.demand for c in inst.customer_manager._customers.values()}
    pl = _build_payloads(customers, dm)
    nodes = [launch] + customers + [retrieve]
    nom, devs = 0.0, []
    for p, a, b in zip(pl, nodes, nodes[1:]):
        try:
            t = dt[ni[a]][ni[b]]
        except (KeyError, IndexError):
            return float("inf")
        e = em.energy_kwh(p, t)
        nom += e
        devs.append(e * rc.energy_deviation_rate)
    return nom + _budgeted_sum(devs, rc.energy_uncertainty_budget)


def build_base(inst):
    """Reconstruct the ALNS 97.60 solution."""
    dm = {c.customer_id: c.demand for c in inst.customer_manager._customers.values()}
    s = Solution()
    for tid, nodes in TRUCK_NODES.items():
        s.add_truck_route(TruckRoute(route_id=tid, nodes=list(nodes), capacity=1000.0))
    tid = 0
    def t(di, lt, ln, cc, rt, rn):
        nonlocal tid
        t = DroneTask(task_id=tid, drone_id=di, launch_truck=lt, launch_node=ln,
                       customers=cc, land_truck=rt, retrieve_node=rn,
                       payloads=_build_payloads(cc, dm))
        tid += 1
        return t
    s.add_drone_task(t(0, 1, 7, [9],   1, 4))
    s.add_drone_task(t(0, 1, 4, [6,8], 1, 5))
    s.add_drone_task(t(1, None, 0, [3],   0, 2))
    s.add_drone_task(t(1, 0, 1, [10],   None, 11))
    return s


def main():
    print("=" * 70)
    print("ORACLE: Exhaustive Cross-Truck Sortie Reconstruction")
    print("=" * 70)

    inst = read_instance(INSTANCE_PATH, strategy="class_based")
    ev = Evaluator(inst)
    dm = {c.customer_id: c.demand for c in inst.customer_manager._customers.values()}
    dc = inst.vehicle_specs.get("drone", type("", (), {"capacity": 30})()).capacity
    bat = inst.robust_config.drone_battery_capacity

    print(f"Drone capacity={dc}  Battery={bat}")
    print(f"Demands: {dm}")
    print()

    base = build_base(inst)
    be = ev.evaluate_solution(base)
    print(f"  Base ALNS: {be.total_cost:.2f}  feasible={be.feasible}")
    for t in base.drone_tasks:
        print(f"    D{t.drone_id}: [{t.launch_node}->{t.customers()}->{t.retrieve_node}]")
    print()

    # ── Clean base: strip {6,8,10} ──
    clean = base.clone()
    for c in NEIGHBORHOOD:
        for t in list(clean.drone_tasks):
            try:
                t.remove_customer(c)
            except (ValueError, IndexError):
                pass
    clean.drone_tasks = [t for t in clean.drone_tasks if len(t.customers()) > 0]
    ce = ev.evaluate_solution(clean)
    print(f"  Clean base ({NEIGHBORHOOD} removed): {ce.total_cost:.2f}  feasible={ce.feasible}")
    print()

    # ── Verify MILP target ──
    print("─── Verify MILP Target (97.18) ───")
    sol = clean.clone()
    nid = max((t.task_id or 0) for t in sol.drone_tasks) + 1 if sol.drone_tasks else 0
    for di, lt, ln, cc, rt, rn in [
        (1, 1, 4, [6], 1, 5),
        (1, 1, 5, [8,10], 0, 1),
    ]:
        sol.add_drone_task(DroneTask(task_id=nid, drone_id=di, launch_truck=lt, launch_node=ln,
                                     customers=cc, land_truck=rt, retrieve_node=rn,
                                     payloads=_build_payloads(cc, dm)))
        nid += 1
    me = ev.evaluate_solution(sol)
    print(f"  MILP target: {me.total_cost:.2f}  feasible={me.feasible}")
    if me.feasible and abs(me.total_cost - 97.18) < 0.02:
        print("  ✓ MILP structure confirmed at 97.18!")
    print()

    # ── Pre-compute energy bounds for pruning ──
    all_anchors = sorted(set(n for nodes in TRUCK_NODES.values() for n in nodes))
    print(f"Anchors: {all_anchors}")

    # For each (block, launch, retrieve), check quick energy < 1.5×battery
    energy_ok = {}
    for block_key, block in [((6,), [6]), ((8,), [8]), ((10,), [10]),
                              ((6,8), [6,8]), ((6,8), [8,6]),
                              ((6,10), [6,10]), ((6,10), [10,6]),
                              ((8,10), [8,10]), ((8,10), [10,8]),
                              ((6,8,10), [6,8,10]), ((6,8,10), [6,10,8]),
                              ((6,8,10), [8,6,10]), ((6,8,10), [8,10,6]),
                              ((6,8,10), [10,6,8]), ((6,8,10), [10,8,6]),
                              ]:
        key_base = tuple(sorted(block))
        for ln in all_anchors:
            for rn in all_anchors:
                e = quick_energy(inst, ln, block, rn)
                if e < bat * 1.5:
                    energy_ok[(key_base, ln, rn)] = True

    print(f"  Energy-checked {len(energy_ok)} (block, launch, retrieve) combos")
    print(f"  Feasible anchors per block:")
    for bk in sorted(set(k[0] for k in energy_ok)):
        feasible_anchors = [(ln, rn) for (bk_, ln, rn) in energy_ok if bk_ == bk]
        print(f"    {list(bk)}: {len(feasible_anchors)} anchor pairs")

    print()
    print("─── Exhaustive Enumeration ───")

    best = float("inf")
    best_desc = ""
    results = []
    total = 0

    for partition in set_partitions(sorted(NEIGHBORHOOD)):
        # block_perms: for each block, all permutations
        block_perms = [itertools.permutations(blk) for blk in partition]
        for perm_combo in itertools.product(*block_perms):
            ordered_blocks = [list(p) for p in perm_combo]
            n = len(ordered_blocks)

            # Pre-collect feasible (launch, retrieve) pairs per block
            lr_opts_per_block = []
            feasible = True
            for blk in ordered_blocks:
                bk = tuple(sorted(blk))
                opts = [(ln, rn) for (bk_, ln, rn) in energy_ok if bk_ == bk]
                if not opts:
                    feasible = False
                    break
                lr_opts_per_block.append(opts)
            if not feasible:
                continue

            # Generate drone assignments + anchor choices
            for drone_ids in itertools.product(DRONES, repeat=n):
                for lr_combo in itertools.product(*lr_opts_per_block):
                    total += 1

                    # Build candidate solution
                    sol = clean.clone()
                    nid = max((t.task_id or 0) for t in sol.drone_tasks) + 1 if sol.drone_tasks else 0

                    for bi in range(n):
                        blk = ordered_blocks[bi]
                        ln, rn = lr_combo[bi]
                        di = drone_ids[bi]
                        lt = truck_of(ln)
                        rt = truck_of(rn)
                        if ln == 0: lt = None
                        if rn == 11: rt = None
                        sol.add_drone_task(DroneTask(
                            task_id=nid, drone_id=di,
                            launch_truck=lt, launch_node=ln,
                            customers=blk,
                            land_truck=rt, retrieve_node=rn,
                            payloads=_build_payloads(blk, dm),
                        ))
                        nid += 1

                    se = ev.evaluate_solution(sol)
                    if se.feasible and math.isfinite(se.total_cost):
                        desc = "; ".join([
                            f"D{drone_ids[bi]}: [{lr_combo[bi][0]}->{ordered_blocks[bi]}->{lr_combo[bi][1]}]"
                            for bi in range(n)
                        ])
                        results.append({"cost": round(se.total_cost, 2), "desc": desc})
                        if se.total_cost < best:
                            best = se.total_cost
                            best_desc = desc
                            print(f"  Best so far: {best:.2f}  |  {desc}")

    results.sort(key=lambda x: x["cost"])

    print()
    print(f"  Total enumerated: {total}")
    print(f"  Feasible found:   {len(results)}")
    print()
    print("=" * 70)
    print("TOP 10 SOLUTIONS")
    print("=" * 70)
    seen = set()
    cnt = 0
    for r in results:
        if r["desc"] not in seen:
            seen.add(r["desc"])
            print(f"  {r['cost']:8.2f}  |  {r['desc']}")
            cnt += 1
            if cnt >= 10:
                break

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Base ALNS:    {be.total_cost:.2f}")
    print(f"  MILP optimal: 97.18")
    print(f"  Best found:   {best:.2f}")
    print(f"  Gap to MILP:  {(best - 97.18) / 97.18 * 100:.2f}%")
    if best < 97.60:
        print(f"  ✓ Improved over ALNS by {97.60 - best:.2f}")
    else:
        print(f"  ✗ No improvement over ALNS")

    out = Path("results/oracle_cross_truck_sortie.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fp:
        json.dump({
            "base_alns": round(be.total_cost, 2),
            "milp_optimal": 97.18,
            "best_found": round(best, 2),
            "best_desc": best_desc,
            "total_enumerated": total,
            "feasible_found": len(results),
            "top_10": [{"cost": r["cost"], "desc": r["desc"]} for r in results[:10]],
        }, fp, indent=2)
    print(f"  Saved to {out}")


if __name__ == "__main__":
    main()