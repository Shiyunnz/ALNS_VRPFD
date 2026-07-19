"""Run manifest and config hashing for revision experiments.

Every run gets a deterministic experiment_id derived from its configuration.
The manifest tracks all runs for deduplication and traceability.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

REVISION_ROOT = Path(__file__).resolve().parent.parent / "results" / "revision_20260610"
MANIFEST_DIR = REVISION_ROOT / "manifest"
MANIFEST_CSV = MANIFEST_DIR / "run_manifest.csv"
MANIFEST_JSONL = MANIFEST_DIR / "run_manifest.jsonl"


def config_hash(config: Dict[str, Any]) -> str:
    """Deterministic SHA-1 hash of a config dict (sorted keys, stable JSON)."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha1(canonical.encode()).hexdigest()[:12]


def experiment_id(
    algorithm: str,
    instance: str,
    seed: int,
    config_hash_val: str,
    experiment_family: str,
) -> str:
    """Deterministic experiment ID from run identity fields + git commit."""
    try:
        import subprocess
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=str(Path(__file__).resolve().parent.parent),
        ).decode().strip()
    except Exception:
        commit = "unknown"
    raw = f"{algorithm}|{instance}|{seed}|{config_hash_val}|{commit}|{experiment_family}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def init_manifest() -> None:
    """Create manifest directory and CSV header if they don't exist."""
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST_CSV.exists():
        header = [
            "experiment_id", "experiment_family", "algorithm", "instance", "seed",
            "config_hash", "code_commit", "command", "start_time", "end_time",
            "runtime_seconds", "status", "result_path", "stdout_path", "stderr_path", "notes",
        ]
        with open(MANIFEST_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()


def load_manifest() -> list:
    """Load existing manifest entries."""
    if not MANIFEST_CSV.exists():
        return []
    with open(MANIFEST_CSV, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def check_duplicate(
    algorithm: str,
    instance: str,
    seed: int,
    cfg_hash: str,
    experiment_family: str,
) -> Optional[str]:
    """Check if a successful run with this identity already exists. Returns experiment_id or None."""
    entries = load_manifest()
    for row in entries:
        if (
            row["algorithm"] == algorithm
            and row["instance"] == instance
            and row["seed"] == str(seed)
            and row["config_hash"] == cfg_hash
            and row["experiment_family"] == experiment_family
            and row["status"] == "success"
        ):
            return row["experiment_id"]
    return None


def record_run(
    algorithm: str,
    instance: str,
    seed: int,
    cfg: Dict[str, Any],
    experiment_family: str,
    status: str = "started",
    command: str = "",
    runtime_seconds: float = 0.0,
    result_path: str = "",
    notes: str = "",
) -> str:
    """Register a run in the manifest. Returns experiment_id."""
    init_manifest()
    cfg_h = config_hash(cfg)
    eid = experiment_id(algorithm, instance, seed, cfg_h, experiment_family)

    row = {
        "experiment_id": eid,
        "experiment_family": experiment_family,
        "algorithm": algorithm,
        "instance": instance,
        "seed": seed,
        "config_hash": cfg_h,
        "code_commit": _git_commit(),
        "command": command,
        "start_time": datetime.now().isoformat(),
        "end_time": datetime.now().isoformat() if status != "started" else "",
        "runtime_seconds": f"{runtime_seconds:.2f}",
        "status": status,
        "result_path": result_path,
        "stdout_path": "",
        "stderr_path": "",
        "notes": notes,
    }

    with open(MANIFEST_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writerow(row)

    with open(MANIFEST_JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")

    return eid


def update_run(eid: str, status: str, runtime_seconds: float = 0.0, result_path: str = "", notes: str = "") -> None:
    """Update an existing manifest row by experiment_id."""
    rows = load_manifest()
    updated = False
    for row in rows:
        if row["experiment_id"] == eid:
            row["status"] = status
            row["end_time"] = datetime.now().isoformat()
            row["runtime_seconds"] = f"{runtime_seconds:.2f}"
            if result_path:
                row["result_path"] = result_path
            if notes:
                row["notes"] = notes
            updated = True
            break

    if updated:
        header = rows[0].keys() if rows else []
        with open(MANIFEST_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)


def _git_commit() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=str(Path(__file__).resolve().parent.parent),
        ).decode().strip()
    except Exception:
        return "unknown"


TRAINING_INSTANCES = ["R_30_10_1", "R_30_10_3", "R_30_10_5"]
ALL_INSTANCE10 = ["R_30_10_1", "R_30_10_2", "R_30_10_3", "R_30_10_4", "R_30_10_5"]
TRAINING_SEEDS_PHASE1 = [100, 101]
TRAINING_SEEDS_PHASE2 = [100, 101, 102]
VALIDATION_SEEDS = list(range(100, 110))