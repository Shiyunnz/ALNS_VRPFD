"""R30 MILP incremental runner — fixed extraction + ALNS verification."""
import sys, json, time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT.parent))
sys.path.insert(0, str(SCRIPT.parent.parent))

from alns_vrpfd.mip.run_mip import run_single_mip

INSTANCES = [
    "data/Instance10/R_30_10_1.txt",
    "data/Instance10/R_30_10_2.txt",
    "data/Instance10/R_30_10_3.txt",
    "data/Instance10/R_30_10_4.txt",
    "data/Instance10/R_30_10_5.txt",
]

OUT = Path("results") / "MIPresult_new"
OUT.mkdir(parents=True, exist_ok=True)
AGG = OUT / "milp_r30_results.json"

all_results = []
if AGG.exists():
    with AGG.open() as f:
        all_results = json.load(f)
completed = {
    Path(r["instance"]).stem
    for r in all_results
    if "error" not in r and r.get("alns_verification", {}).get("alns_feasible") is not None
}
print(f"Already done (with ALNS verif): {sorted(completed) if completed else 'none'}")

for inst in INSTANCES:
    tag = Path(inst).stem
    if tag in completed:
        print(f"[SKIP] {tag}")
        continue

    print(f"\n{'='*60}")
    print(f"[{len(all_results)+1 - len(completed)}/{len(INSTANCES)}] Running {tag}")
    print(f"{'='*60}")
    t0 = time.time()

    config = {
        "instance": inst,
        "time_window_strategy": "class_based",
        "epsilon": 1e-3,
        "energy_budget": 3,
        "robust_energy": True,
        "bigm_time": 1000,
        "bigm_load": 1000,
        "bigm_energy": 20.0,
        "tardiness_weight": 1.0,
        "time_limit": 3600,
        "notes": f"{tag} — 3-segment PWL, fixed extraction",
    }

    result = {}
    try:
        result = run_single_mip(config)
    except Exception as e:
        import traceback; traceback.print_exc()
        result = {"instance": inst, "error": str(e), "runtime": time.time()-t0}

    all_results.append(result)
    with AGG.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - t0
    ver = result.get("alns_verification", {})
    obj = result.get("objective", "?")
    gap = result.get("mip_gap", 0)
    alns_ok = ver.get("alns_feasible")
    alns_cost = ver.get("alns_cost")
    print(f"  Done in {elapsed/60:.1f}min | obj={obj} | gap={gap*100:.1f}% | ALNS ok={alns_ok} | ALNS cost={alns_cost}")

print("\n=== FINAL SUMMARY ===")
fmt = "{:<18} {:>8} {:>8} {:>8} {:>8}"
print(fmt.format("Instance", "Obj", "Gap%", "ALNS ok", "ALNS cost"))
for r in all_results:
    inst = Path(r.get("instance","?")).stem
    obj = r.get("objective", None)
    gap = r.get("mip_gap", 0)*100 if isinstance(r.get("mip_gap"), (int,float)) else None
    ver = r.get("alns_verification", {})
    alns_ok = str(ver.get("alns_feasible"))
    alns_cost = ver.get("alns_cost")
    obj_s = f"{obj:.1f}" if isinstance(obj, (int,float)) else str(obj)
    gap_s = f"{gap:.1f}%" if gap is not None else "?"
    acost_s = f"{alns_cost:.1f}" if isinstance(alns_cost, (int,float)) else str(alns_cost)
    print(fmt.format(inst, obj_s, gap_s, alns_ok, acost_s))
