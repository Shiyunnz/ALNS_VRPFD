"""Run MILP for R30 n=10 instances one at a time, saving results per instance."""
import sys
from pathlib import Path
import json
import time

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from alns_vrpfd.mip.run_mip import run_single_mip

INSTANCES = [
    "data/Instance10/R_30_10_1.txt",
    "data/Instance10/R_30_10_2.txt",
    "data/Instance10/R_30_10_3.txt",
    "data/Instance10/R_30_10_4.txt",
    "data/Instance10/R_30_10_5.txt",
]

OUTPUT_DIR = Path("results") / "MIPresult_new"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

all_results = []
for idx, instance_path in enumerate(INSTANCES):
    tag = Path(instance_path).stem
    print(f"\n{'='*60}")
    print(f"[{idx+1}/{len(INSTANCES)}] Running {tag}")
    print(f"{'='*60}")
    start = time.time()

    config = {
        "instance": instance_path,
        "time_window_strategy": "class_based",
        "epsilon": 1e-3,
        "energy_budget": 3,
        "robust_energy": True,
        "bigm_time": 1000,
        "bigm_load": 1000,
        "bigm_energy": 20.0,
        "tardiness_weight": 1.0,
        "time_limit": 3600,
        "notes": f"{tag} — 3-segment PWL, 3600s",
    }

    try:
        result = run_single_mip(config)
        elapsed = time.time() - start
        print(f"[{tag}] done in {elapsed/60:.1f} min")
        all_results.append(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        elapsed = time.time() - start
        all_results.append({
            "instance": instance_path,
            "error": str(e),
            "runtime": elapsed,
        })

    # Write aggregate after each instance (incremental save)
    aggregate = OUTPUT_DIR / "mip_r30_runs.json"
    with aggregate.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  [saved aggregate to {aggregate}]")

print("\nAll R30 instances complete.")
print(f"Results: {OUTPUT_DIR / 'mip_r30_runs.json'}")

# Print summary table
print(f"\n{'Instance':<20} {'Obj':>12} {'Gap%':>8} {'Runtime':>8}")
print("-"*50)
for r in all_results:
    inst = Path(r.get("instance", "?")).stem
    obj = r.get("objective", "INF")
    gap = r.get("mip_gap", None)
    rt = r.get("runtime", 0)
    obj_str = f"{obj:.2f}" if isinstance(obj, (int, float)) else str(obj)
    gap_str = f"{gap*100:.2f}%" if isinstance(gap, (int, float)) and gap is not None else "N/A"
    rt_str = f"{rt:.0f}s" if isinstance(rt, (int, float)) else str(rt)
    print(f"{inst:<20} {obj_str:>12} {gap_str:>8} {rt_str:>8}")
