#!/usr/bin/env python3
"""
批量运行所有敏感度分析 (embedded 鲁棒策略)

按顺序运行 8 项敏感度分析, 结果保存到 *_embedded 文件夹:
  1. docking_flexibility_comparison → drone_flexibility_embedded/
  2. battery_sensitivity           → battery_sensitivity_embedded/
  3. drone_speed_sensitivity       → drone_speed_embedded/
  4. drone_count_sensitivity       → drone_count_embedded/
  5. drone_payload_sensitivity     → drone_payload_embedded/
  6. gamma_sensitivity             → gamma_sensitivity_embedded/
  7. theta_sensitivity             → theta_sensitivity_embedded/
  8. time_window_sensitivity       → time_window_sensitivity_embedded/

用法:
    python sensitivity/run_all_sensitivity_embedded.py
    python sensitivity/run_all_sensitivity_embedded.py --instance-dir data/Instance25
    python sensitivity/run_all_sensitivity_embedded.py --skip flexibility  # 跳过已完成的分析
    python sensitivity/run_all_sensitivity_embedded.py --only battery speed  # 只运行指定分析
"""

from __future__ import annotations

import sys
import argparse
import time
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

RESULTS_BASE = Path(__file__).parent / "results_new"

# Analysis name → (module name, embedded output dir name, output prefix/csv stem)
ANALYSES = [
    ("flexibility",  "docking_flexibility_comparison", "drone_flexibility_embedded",   "docking_flexibility"),
    ("battery",      "battery_sensitivity",            "battery_sensitivity_embedded",  "battery_sensitivity"),
    ("speed",        "drone_speed_sensitivity",        "drone_speed_embedded",          "drone_speed_sensitivity"),
    ("count",        "drone_count_sensitivity",        "drone_count_embedded",          "drone_count_sensitivity"),
    ("payload",      "drone_payload_sensitivity",      "drone_payload_embedded",        "drone_payload_sensitivity"),
    ("gamma",        "gamma_sensitivity",              "gamma_sensitivity_embedded",    "gamma_sensitivity"),
    ("theta",        "theta_sensitivity",              "theta_sensitivity_embedded",    "theta_sensitivity"),
    ("timewindow",   "time_window_sensitivity",        "time_window_sensitivity_embedded", "time_window_sensitivity"),
]


def _patch_and_run_standard(module_name: str, embedded_dir: str, csv_stem: str, extra_argv: list[str]):
    """Import a sensitivity module, patch OUTPUT_DIR/OUTPUT_CSV, and run main()."""
    import importlib

    embedded_path = RESULTS_BASE / embedded_dir
    embedded_path.mkdir(parents=True, exist_ok=True)

    mod = importlib.import_module(f"sensitivity.{module_name}")

    # Patch module-level output paths
    mod.OUTPUT_DIR = embedded_path
    mod.OUTPUT_CSV = embedded_path / f"{csv_stem}_results.csv"

    # Set sys.argv for argparse
    sys.argv = [module_name + ".py"] + extra_argv

    mod.main()


def _run_flexibility(embedded_dir: str, extra_argv: list[str]):
    """Run docking_flexibility_comparison with --output-dir flag."""
    import importlib

    embedded_path = RESULTS_BASE / embedded_dir
    embedded_path.mkdir(parents=True, exist_ok=True)

    mod = importlib.import_module("sensitivity.docking_flexibility_comparison")

    # This script supports --output-dir and --output-prefix natively
    sys.argv = [
        "docking_flexibility_comparison.py",
        "--output-dir", str(embedded_path),
        "--output-prefix", "docking_flexibility_embedded",
    ] + extra_argv

    mod.main()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量运行所有敏感度分析 (embedded 策略)")
    parser.add_argument(
        "--instance-dir", "-i",
        action="append",
        dest="instance_dirs",
        help="算例目录 (传递给各子脚本), 例如 --instance-dir data/Instance25",
    )
    parser.add_argument(
        "--trials", type=int, default=5,
        help="每组合独立试验次数 (默认: 5)",
    )
    parser.add_argument(
        "--skip", nargs="*", default=[],
        metavar="NAME",
        help="跳过指定的分析, 可选: flexibility battery speed count payload gamma theta timewindow",
    )
    parser.add_argument(
        "--only", nargs="*", default=None,
        metavar="NAME",
        help="只运行指定的分析 (互斥于 --skip)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Build the common extra argv for each sub-script
    extra_argv: list[str] = []
    if args.instance_dirs:
        for d in args.instance_dirs:
            extra_argv.extend(["--instance-dir", str(d)])
    extra_argv.extend(["--trials", str(args.trials)])

    skip_set = set(args.skip) if args.skip else set()
    only_set = set(args.only) if args.only else None

    total_start = time.perf_counter()

    for name, module_name, embedded_dir, csv_stem in ANALYSES:
        if only_set is not None and name not in only_set:
            print(f"\n{'='*60}")
            print(f"⏭️  跳过 {name} (不在 --only 列表中)")
            print(f"{'='*60}")
            continue
        if name in skip_set:
            print(f"\n{'='*60}")
            print(f"⏭️  跳过 {name} (在 --skip 列表中)")
            print(f"{'='*60}")
            continue

        print(f"\n{'='*60}")
        print(f"🚀 开始运行: {name} → {embedded_dir}/")
        print(f"{'='*60}")
        
        analysis_start = time.perf_counter()

        try:
            if name == "flexibility":
                _run_flexibility(embedded_dir, extra_argv)
            else:
                _patch_and_run_standard(module_name, embedded_dir, csv_stem, extra_argv)
            elapsed = time.perf_counter() - analysis_start
            print(f"\n✅ {name} 完成! 耗时: {elapsed/60:.1f} 分钟")
        except Exception as exc:
            elapsed = time.perf_counter() - analysis_start
            print(f"\n❌ {name} 失败 ({elapsed/60:.1f}分钟): {exc}")
            import traceback
            traceback.print_exc()

    total_elapsed = time.perf_counter() - total_start
    print(f"\n{'='*60}")
    print(f"🏁 全部完成! 总耗时: {total_elapsed/60:.1f} 分钟")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
