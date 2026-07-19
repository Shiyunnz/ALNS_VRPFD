"""续跑 R30 MILP，跳过已有结果的实例。"""
import sys
import json
import time
from pathlib import Path

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
AGGREGATE = OUTPUT_DIR / "mip_r30_runs.json"

# 加载已有结果
all_results = []
completed = set()
if AGGREGATE.exists():
    with AGGREGATE.open() as f:
        all_results = json.load(f)
    completed = {Path(r["instance"]).stem for r in all_results if "error" not in r}
    print(f"已完成的实例: {sorted(completed)}")

for inst_path in INSTANCES:
    tag = Path(inst_path).stem
    if tag in completed:
        print(f"[SKIP] {tag} — already done")
        continue

    print(f"\n{'='*60}")
    print(f"[RUNNING] {tag}")
    print(f"{'='*60}")
    start = time.time()

    config = {
        "instance": inst_path,
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
        all_results.append(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        all_results.append({
            "instance": inst_path,
            "error": str(e),
            "runtime": time.time() - start,
        })

    with AGGREGATE.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - start
    print(f"[{tag}] Done in {elapsed/60:.1f} min")

# 输出汇总
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"{'Instance':<20} {'Obj':>10} {'Gap%':>8} {'Status':>8}")
print("-"*48)
for r in all_results:
    inst = Path(r.get("instance", "?")).stem
    obj = r.get("objective", None)
    gap = r.get("mip_gap", None)
    status = r.get("status", "?")
    if obj is not None and gap is not None:
        print(f"{inst:<20} {obj:10.2f} {gap*100:7.2f}% {status:>8}")
    else:
        err = r.get("error", "?")
        print(f"{inst:<20} {'ERR':>10} {'N/A':>8} {str(err)[:40]}")
