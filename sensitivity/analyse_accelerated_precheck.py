"""Analyse accelerated precheck_guarded vs embedded trial CSV."""

import argparse
import csv
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results_new/verification_strategy_compare"

TRIAL_FILES = {
    "instance10": RESULTS_DIR / "instance10_all_t3_i2000_accelerated_precheck_guarded_trials.csv",
    "instance25": RESULTS_DIR / "instance25_all_t3_i2000_accelerated_precheck_guarded_trials.csv",
    "instance25_round2": RESULTS_DIR / "instance25_round2_accelerated_trials.csv",
}


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)) ** 0.5


def analyse(trials_path: Path, label: str):
    rows = list(csv.DictReader(trials_path.open()))

    emb = [r for r in rows if r["method"] == "embedded"]
    pcg = [r for r in rows if r["method"] == "precheck_guarded"]

    emb_rt = [float(r["runtime_sec"]) for r in emb]
    pcg_rt = [float(r["runtime_sec"]) for r in pcg]
    emb_cost = [float(r["robust_cost"]) for r in emb]
    pcg_cost = [float(r["robust_cost"]) for r in pcg]

    print("=" * 60)
    print(f"OVERALL  ({label})")
    print("=" * 60)
    print(
        f"Embedded:          mean_rt={mean(emb_rt):.3f}s  "
        f"std={stdev(emb_rt):.3f}  "
        f"mean_cost={mean(emb_cost):.3f}  "
        f"std={stdev(emb_cost):.3f}"
    )
    print(
        f"Precheck_Guarded:  mean_rt={mean(pcg_rt):.3f}s  "
        f"std={stdev(pcg_rt):.3f}  "
        f"mean_cost={mean(pcg_cost):.3f}  "
        f"std={stdev(pcg_cost):.3f}"
    )
    rt_pct = (mean(pcg_rt) - mean(emb_rt)) / mean(emb_rt) * 100
    cost_pct = (mean(pcg_cost) - mean(emb_cost)) / mean(emb_cost) * 100
    print(f"Runtime delta: {rt_pct:+.1f}%")
    print(f"Cost delta:    {cost_pct:+.2f}%")
    print()

    # Per-region breakdown
    print("-" * 60)
    print("PER-REGION BREAKDOWN")
    print("-" * 60)
    for region in ["30", "40", "50"]:
        e = [r for r in emb if f"R_{region}_" in r["instance"]]
        p = [r for r in pcg if f"R_{region}_" in r["instance"]]
        if not e or not p:
            continue
        e_rt = [float(r["runtime_sec"]) for r in e]
        p_rt = [float(r["runtime_sec"]) for r in p]
        e_cost = [float(r["robust_cost"]) for r in e]
        p_cost = [float(r["robust_cost"]) for r in p]

        # Pair by (instance, seed)
        e_map = {(r["instance"], r["seed"]): r for r in e}
        p_map = {(r["instance"], r["seed"]): r for r in p}
        keys = sorted(set(e_map) & set(p_map))

        faster = sum(
            1
            for k in keys
            if float(p_map[k]["runtime_sec"]) < float(e_map[k]["runtime_sec"])
        )
        noninferior = sum(
            1
            for k in keys
            if float(p_map[k]["robust_cost"])
            <= float(e_map[k]["robust_cost"]) + 1e-6
        )
        speedups = [
            float(e_map[k]["runtime_sec"]) / float(p_map[k]["runtime_sec"])
            for k in keys
        ]

        rd = (mean(p_rt) - mean(e_rt)) / mean(e_rt) * 100
        cd = (mean(p_cost) - mean(e_cost)) / mean(e_cost) * 100
        print(
            f"R{region}: rt_delta={rd:+.1f}%  cost_delta={cd:+.2f}%  "
            f"faster={faster}/{len(keys)}  noninferior={noninferior}/{len(keys)}  "
            f"mean_speedup={mean(speedups):.3f}x"
        )
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instance",
        type=str,
        default="all",
        choices=["all", "instance10", "instance25", "instance25_round2"],
        help="Which instance set to analyse.",
    )
    args = parser.parse_args()

    targets = list(TRIAL_FILES.keys()) if args.instance == "all" else [args.instance]
    for key in targets:
        path = TRIAL_FILES[key]
        if not path.exists():
            print(f"[SKIP] {key}: {path} not found")
            continue
        analyse(path, key)


if __name__ == "__main__":
    main()
