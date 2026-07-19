#!/usr/bin/env python3
"""
Replace rows in the detailed results CSV for specific instance(s) and speed(s) by keeping only the last occurrence per pair.
Also recomputes the summary CSV from the detailed file.
"""
from pathlib import Path
import csv
import argparse
from statistics import mean
import math
import shutil


def parse_args():
    p = argparse.ArgumentParser(
        description='Update specific results rows and recompute summary')
    p.add_argument(
        '--results', default='sensitivity/results_new/drone_speed/drone_speed_sensitivity_results.csv')
    p.add_argument('--instance', action='append', dest='instances',
                   help='instance paths (can specify multiple)')
    p.add_argument('--speeds', type=str, default=None,
                   help='comma-separated speeds to update')
    return p.parse_args()


def update_rows(results_path: Path, instances: list[str] | None, speeds_str: str | None):
    if not results_path.exists():
        raise FileNotFoundError(f'Results file not found: {results_path}')

    speeds = None
    if speeds_str:
        speeds = set(float(s.strip()) for s in speeds_str.split(','))
    instances_set = set(instances) if instances else None

    # Backup original file
    bak = results_path.with_suffix('.csv.bak')
    shutil.copy(results_path, bak)
    print('Backed up original results to', bak)

    # Read all rows, keep last occurrence for targeted keys
    rows = []
    with results_path.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Build index of last occurrence for targeted keys
    last_index = {}
    for idx, r in enumerate(rows):
        inst = r.get('instance')
        try:
            speed = float(r.get('drone_speed', 0))
        except Exception:
            speed = None
        key = (inst, speed)
        if instances_set is None and speeds is None:
            # No filters: update all -> same behaviour as dedupe to keep last occurrence globally
            last_index[key] = idx
        else:
            # If this row matches filter, track last index
            if instances_set is not None and inst not in instances_set:
                continue
            if speeds is not None and speed not in speeds:
                continue
            last_index[key] = idx

    if not last_index:
        print('No matching rows found for update. Nothing to do.')
        return

    # Build new rows: include all rows excluding prior occurrences of keys that are targeted; keep the last occurrence
    keys_to_keep = set(last_index.keys())
    new_rows = []
    seen_for_keys = set()
    for idx, r in enumerate(rows):
        inst = r.get('instance')
        try:
            speed = float(r.get('drone_speed', 0))
        except Exception:
            speed = None
        key = (inst, speed)
        if key in keys_to_keep:
            # if this is last occurrence, keep; otherwise skip
            if last_index[key] == idx and key not in seen_for_keys:
                new_rows.append(r)
                seen_for_keys.add(key)
            else:
                # skip older occurrence
                continue
        else:
            new_rows.append(r)

    # Write updated detailed CSV
    fieldnames = None
    if rows:
        fieldnames = list(rows[0].keys())
    results_path.with_suffix('').parent.mkdir(parents=True, exist_ok=True)
    tmp = results_path.with_suffix('.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)
    tmp.replace(results_path)
    print('Updated detailed results written to', results_path)

    # Recompute summary from detailed CSV
    recompute_summary(results_path.parent)


def recompute_summary(results_dir: Path):
    detailed = results_dir / 'drone_speed_sensitivity_results.csv'
    if not detailed.exists():
        print('Detailed results do not exist; cannot recompute summary')
        return
    rows = []
    with detailed.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Group by scale (from instance) and speed
    groups = {}

    def extract_scale(instance_path):
        try:
            return Path(instance_path).parent.name
        except Exception:
            return 'unknown'

    for r in rows:
        inst = r.get('instance')
        scale = extract_scale(inst) if inst else 'unknown'
        try:
            speed = float(r.get('drone_speed', 0))
        except Exception:
            speed = 0.0
        groups.setdefault((scale, speed), []).append(r)

    summary_rows = []
    for (scale, speed), members in sorted(groups.items()):
        cs = []
        dc = []
        for m in members:
            try:
                c = float(m.get('cost_saving_vs_baseline')
                          or m.get('cost_saving') or 0.0)
                if math.isfinite(c):
                    cs.append(c)
            except Exception:
                pass
            try:
                d = float(m.get('best_drone_customers')
                          or m.get('best_drone') or 0.0)
                if math.isfinite(d):
                    dc.append(d)
            except Exception:
                pass
        summary_rows.append({'scale': scale, 'drone_speed': speed, 'avg_cost_saving_vs_baseline': mean(
            cs) if cs else 0.0, 'avg_best_drone_customers': mean(dc) if dc else 0.0})

    summary = results_dir / 'drone_speed_summary.csv'
    fieldnames = ['scale', 'drone_speed',
                  'avg_cost_saving_vs_baseline', 'avg_best_drone_customers']
    with summary.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)
    print('Recomputed summary:', summary)


if __name__ == '__main__':
    args = parse_args()
    res_path = Path(args.results)
    update_rows(res_path, args.instances, args.speeds)
