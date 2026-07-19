"""Analyze why MILP's 50.96 solution is infeasible in ALNS.

Sortie D1: 5 -> [9, 8] -> 11 on R_30_10_2.
ALNS exact robust energy = 6.419 kWh > battery (6.3 kWh) by 0.119 kWh.
"""
import sys, math, json
from pathlib import Path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from alns_vrpfd.evaluation.energy import DroneEnergyModel
from alns_vrpfd.utils.io_utils import read_instance
from alns_vrpfd.utils.config_loader import ALNSConfig

def pwl(omega, bps, pvs):
    if omega <= bps[0]: return pvs[0]
    if omega >= bps[-1]: return pvs[-1]
    for k in range(len(bps)-1):
        if bps[k] <= omega <= bps[k+1]:
            t = (omega-bps[k])/(bps[k+1]-bps[k]) if bps[k+1]>bps[k] else 0
            return pvs[k] + t*(pvs[k+1]-pvs[k])
    return pvs[-1]

def gamma_robust(energies, dev_rate, gamma_budget):
    devs = [e*dev_rate for e in energies]
    states = [0.0]*(gamma_budget+1)
    for e, d in zip(energies, devs):
        nxt = [0.0]*(gamma_budget+1)
        for g in range(gamma_budget+1):
            wv = states[g] + e
            if g > 0: wv = max(wv, states[g-1] + e + d)
            nxt[g] = wv
        states = nxt
    return sum(energies), states[gamma_budget]

def main():
    instance = read_instance(str(project_root/"data"/"Instance10"/"R_30_10_2.txt"), strategy="demand_based")
    cfg = ALNSConfig()
    em = DroneEnergyModel()
    BATTERY = cfg.drone_battery_capacity  # 6.3 kWh
    DEV_RATE = cfg.energy_deviation_rate   # 0.1
    GAMMA = 3
    DRONE_CAP = instance.vehicle_specs['drone'].capacity  # 30 kg

    # PWL breakpoints
    K = 10
    bps = [k*DRONE_CAP/K for k in range(K+1)]
    pvs = [em.power_kw(w) for w in bps]

    dm = instance.distance_matrix("drone")
    tm = instance.time_matrix("drone")
    nodes = instance.all_node_ids()
    idx = {n:i for i,n in enumerate(nodes)}
    demands = {c.customer_id: c.demand for c in instance.customer_manager.customers()}

    # The sortie: 5 -> [9, 8] -> 11
    arcs = [(5,9), (9,8), (8,11)]
    # Demand: cust 9=12, cust 8=6
    # ALNS payload sequence: [18, 6, 0]
    pay_alns = [demands[9]+demands[8], demands[8], 0.0]
    times = [tm[idx[i]][idx[j]] for i,j in arcs]
    dists = [dm[idx[i]][idx[j]] for i,j in arcs]

    def seg_energies(payloads):
        return [em.energy_kwh(p, t) for p,t in zip(payloads, times)]

    def pwl_energies(payloads):
        return [pwl(p, bps, pvs)*t for p,t in zip(payloads, times)]

    # ── 1. ALNS exact (correct payloads: [18, 6, 0]) ──
    e_exact = seg_energies(pay_alns)
    nom_exact, worst_exact = gamma_robust(e_exact, DEV_RATE, GAMMA)

    # ── 2. PWL with same payloads ──
    e_pwl = pwl_energies(pay_alns)
    nom_pwl, worst_pwl = gamma_robust(e_pwl, DEV_RATE, GAMMA)

    print("="*80)
    print("  MILP vs ALNS Energy Gap — Sortie D1: 5 → [9,8] → 11")
    print("="*80)

    print(f"\n  Instance: R_30_10_2")
    print(f"  Battery: {BATTERY} kWh, Γ={GAMMA}, dev={DEV_RATE}, drone_cap={DRONE_CAP} kg")
    print(f"  Demands: node 9={demands[9]:.0f} kg, node 8={demands[8]:.0f} kg")

    print(f"\n  ─── Table 1: ALNS Exact Energy ───")
    print(f"  {'Arc':>8}  {'d(km)':>7}  {'t(h)':>9}  {'payload':>8}  {'P(kW)':>8}  {'E(kWh)':>10}")
    for (i,j), d, t, p, e in zip(arcs, dists, times, pay_alns, e_exact):
        print(f"  {i}→{j:>8}  {d:7.2f}  {t:9.6f}  {p:8.2f}  {em.power_kw(p):8.4f}  {e:10.6f}")
    print(f"  {'':=>50}")
    print(f"  {'Total':>41}  {nom_exact:10.6f}")
    print(f"  {'Robust (Γ=3)':>41}  {worst_exact:10.6f}")
    print(f"  {'Battery':>41}  {BATTERY:10.4f}")
    print(f"  {'Margin':>41}  {worst_exact-BATTERY:+.6f}")
    print(f"  {'Feasible':>41}  {'NO' if worst_exact>BATTERY else 'YES'}")

    print(f"\n  ─── Table 2: How MILP Can Underestimate This Sortie ───")
    print(f"  The key: MILP's u launch event relaxes load_drone_plus at launch point 5.")
    print(f"  If MILP assigns load_drone_plus[5,D1] < 18 (the true total), energy is under-estimated.\n")
    print(f"  {'Scenario':>35}  {'loads':>20}  {'E_nom':>10}  {'E_worst':>10}  {'Margin':>10}  {'Feas?':>7}")

    scenarios = [
        ("Correct: cumulative [18,6,0]", [18,6,0]),
        ("Load=12 (5→9 only, rest=0)", [12,0,0]),
        ("Load=9 (7.5→1.5), rest=0", [9,3,0]),
        ("Load=15 (=12+3 spread)", [15,3,0]),
        ("Load=0 (fully relaxed)", [0,0,0]),
    ]
    for name, ld in scenarios:
        # At non-launch nodes, load tracks as: load[j] = load[i] - v_served*demand[j]
        # For node 9: ld[1] = ld[0] - 1*demand[9] → but ld[1] is given
        # For node 8: ld[2] = ld[1] - 1*demand[8]
        e_test = seg_energies(ld)
        n, w = gamma_robust(e_test, DEV_RATE, GAMMA)
        m = w - BATTERY
        print(f"  {name:>35}  {str([f'{x:.0f}' for x in ld]):>20}  {n:10.4f}  {w:10.4f}  {m:+10.4f}  {'NO' if w>BATTERY else 'YES':>7}")

    print(f"\n  ─── Table 3: PWL vs Exact Power at Each Payload ───")
    print(f"  {'ω(kg)':>8}  {'P_exact':>10}  {'P_pwl':>10}  {'diff':>10}")
    for w in [0,3,6,9,12,15,18,21,24,27,30]:
        pe = em.power_kw(w)
        pp = pwl(w, bps, pvs)
        print(f"  {w:8.2f}  {pe:10.4f}  {pp:10.4f}  {pp-pe:+10.4f}")

    print(f"\n  ─── Table 4: PWL Exact Matching at Sortie Payloads ───")
    for p in pay_alns:
        print(f"  payload={p:.0f} kg: P_exact={em.power_kw(p):.4f}, "
              f"P_pwl={pwl(p,bps,pvs):.4f}, "
              f"on_breakpoint={'YES' if p in bps else 'NO'}")

    # ── Find the minimal feasible payload ──
    print(f"\n  ─── Table 5: What Launch Load Makes Sortie Feasible? ───")
    print(f"  (Assuming MILP can set load_drone_plus[5,D1] arbitrarily)")
    print(f"  {'launch_load(kg)':>18}  {'payloads':>20}  {'E_worst':>10}  {'Margin vs 6.3':>16}")
    for ll in range(0, 19, 1):
        ld = [max(0, ll), max(0, ll-demands[9]), max(0, ll-demands[9]-demands[8])]
        e = seg_energies(ld)
        n, w = gamma_robust(e, DEV_RATE, GAMMA)
        print(f"  {ll:>18.0f}  {str([f'{x:.0f}' for x in ld]):>20}  {w:10.4f}  {w-BATTERY:+15.4f}" +
              ("  ← feasible" if w <= BATTERY else ""))

    # ── Identify root cause ──
    print(f"\n  {'='*70}")
    print(f"  ROOT CAUSE ANALYSIS")
    print(f"  {'='*70}")

    n_arc, w_arc, m_arc = None, None, None
    for ll in range(0, 19, 1):
        ld = [max(0, ll), max(0, ll-demands[9]), max(0, ll-demands[9]-demands[8])]
        e = seg_energies(ld)
        n, w = gamma_robust(e, DEV_RATE, GAMMA)
        if w <= BATTERY:
            n_arc, w_arc, m_arc, ld_arc = n, w, w-BATTERY, ld
            break

    print(f"\n  The MILP needs NOTHING to bind load_drone_plus[launch_point, D1]")
    print(f"  to the total sortie payload. The u[i,k,d]=1 event relaxes the load")
    print(f"  continuity constraints via +M*u_sum terms (Eq 34-35 in builder.py).")
    print(f"")
    print(f"  Solver action: minimize cost → minimize energy → minimize load")
    print(f"  Since: lower load → lower PWL power → lower energy_state_gamma")
    print(f"  → easier to satisfy battery constraint")
    print(f"")
    print(f"  To pass battery constraint (≤{BATTERY:.1f} kWh), the MILP needs")
    print(f"  load_drone_plus[5,D1] ≤ {ld_arc[0] if n_arc else 'N/A'} kg")
    print(f"  (max {ld_arc[0] if n_arc else 0} kg to be 'feasible')")
    print(f"  But the true physical payload is 18 kg.")
    print(f"")
    print(f"  Truth: sortie 5→[9,8]→11 has worst-case energy {worst_exact:.4f} kWh")
    print(f"  = nominal {nom_exact:.4f} + robust margin {worst_exact-nom_exact:.4f}")
    print(f"  Battery capacity = {BATTERY:.1f} kWh")
    print(f"  True margin = {worst_exact-BATTERY:.4f} kWh → INFEASIBLE")
    print(f"")
    print(f"  So the 0.119 kWh gap comes from:")
    print(f"  1. MILP underestimates launch load → lower power on first arc")
    print(f"  2. v_served constrains load AFTER first customer, but launch load is free")
    print(f"  3. The MILP has no constraint that total_launch_load = sum(sortie_demands)")
    print(f"  4. Solver exploits this: sets low launch load, low power, 'passes' energy check")
    print(f"")
    print(f"  FIX: Add constraint: load_drone_minus[launch_point, d] >= sum(served downstream demands)")
    print(f"  Or equivalently: when u[i,k,d]=1, bind load to the downstream task payload")

    # ── Check full-sortie energy breakdown ──
    print(f"\n  ─── Table 6: Full Sortie Energy Detail ───")
    print(f"  Correct ALNS computation:")
    for (i,j), p, e in zip(arcs, pay_alns, e_exact):
        print(f"    arc {i}→{j}: payload={p:.0f} kg, "
              f"P={em.power_kw(p):.4f} kW, "
              f"t={tm[idx[i]][idx[j]]:.6f} h, "
              f"E={e:.6f} kWh, "
              f"Robust_dev={e*DEV_RATE:.6f} kWh")

    print(f"\n  ─── Table 7: Gamma-layer Propagation ───")
    print(f"  {'seg':>4}  {'E_nom':>10}  {'dev':>10}  ", end="")
    for g in range(GAMMA+1): print(f"Γ={g:<10}  ", end="")
    print()
    states = [0.0]*(GAMMA+1)
    for sidx, (e, d) in enumerate(zip(e_exact, [x*DEV_RATE for x in e_exact])):
        nxt = [0.0]*(GAMMA+1)
        for g in range(GAMMA+1):
            wv = states[g] + e
            if g>0: wv = max(wv, states[g-1]+e+d)
            nxt[g] = wv
        print(f"  {sidx:4d}  {e:10.6f}  {d:10.6f}  ", end="")
        for g in range(GAMMA+1): print(f"{nxt[g]:<12.6f}", end="")
        print()
        states = nxt

    print(f"\n  Final Γ={GAMMA} worst-case: {states[GAMMA]:.6f} kWh")
    print(f"  Battery: {BATTERY:.1f} kWh")
    print(f"  OVER by: {states[GAMMA]-BATTERY:.4f} kWh")

if __name__ == "__main__":
    main()