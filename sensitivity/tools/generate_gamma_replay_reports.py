"""Generate gamma trend plots and Table-6 style replay tables for Instance25.

Default behavior:
- Include Region 30/40/50 data (15 instances total).
- Apply rerun override for Region-30 gamma=1 if override files exist.
- Visualize the all-region average trend (single dual-axis chart).
- Export three Table-6 style replay tables (ND / UD / NDC).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_BASE_DIR = Path("sensitivity/results_new/scenario_replay")
DEFAULT_PREFIXES = ["instance25_r30_gamma", "instance25_r40_gamma", "instance25_r50_gamma"]
DEFAULT_R30_G1_BEST_OVERRIDE = (
    Path("sensitivity/results_new/scenario_replay_rerun_gamma1")
    / "instance25_r30_gamma1_rerun_best_all.csv"
)
DEFAULT_R30_G1_REPLAY_OVERRIDE = (
    Path("sensitivity/results_new/scenario_replay_rerun_gamma1")
    / "instance25_r30_gamma1_rerun_replay_summary_all.csv"
)
DEFAULT_G0_BEST_OVERRIDES = [
    Path("sensitivity/results_new/scenario_replay_rerun_gamma0")
    / "instance25_r30_gamma0_rerun_best_all.csv",
    Path("sensitivity/results_new/scenario_replay_rerun_gamma0")
    / "instance25_r40_gamma0_rerun_best_all.csv",
    Path("sensitivity/results_new/scenario_replay_rerun_gamma0")
    / "instance25_r50_gamma0_rerun_best_all.csv",
]
DEFAULT_G0_REPLAY_OVERRIDES = [
    Path("sensitivity/results_new/scenario_replay_rerun_gamma0")
    / "instance25_r30_gamma0_rerun_replay_summary_all.csv",
    Path("sensitivity/results_new/scenario_replay_rerun_gamma0")
    / "instance25_r40_gamma0_rerun_replay_summary_all.csv",
    Path("sensitivity/results_new/scenario_replay_rerun_gamma0")
    / "instance25_r50_gamma0_rerun_replay_summary_all.csv",
]
DEFAULT_G0_TRIAL_OVERRIDES = [
    Path("sensitivity/results_new/scenario_replay_rerun_gamma0")
    / "instance25_r30_gamma0_rerun_trials_all.csv",
    Path("sensitivity/results_new/scenario_replay_rerun_gamma0")
    / "instance25_r40_gamma0_rerun_trials_all.csv",
    Path("sensitivity/results_new/scenario_replay_rerun_gamma0")
    / "instance25_r50_gamma0_rerun_trials_all.csv",
]
DEFAULT_R30_G1_TRIAL_OVERRIDE = (
    Path("sensitivity/results_new/scenario_replay_rerun_gamma1")
    / "instance25_r30_gamma1_rerun_trials_all.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate gamma trend plots and Table-6 style replay tables."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(DEFAULT_BASE_DIR),
        help="Directory containing *_best_all.csv and *_replay_summary_all.csv",
    )
    parser.add_argument(
        "--prefixes",
        type=str,
        default=",".join(DEFAULT_PREFIXES),
        help="Comma-separated prefixes, e.g. instance25_r40_gamma,instance25_r50_gamma",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: same as --base-dir",
    )
    parser.add_argument(
        "--r30-gamma1-best-override",
        type=str,
        default=str(DEFAULT_R30_G1_BEST_OVERRIDE),
        help="Optional Region-30 gamma=1 best_all override CSV path.",
    )
    parser.add_argument(
        "--r30-gamma1-replay-override",
        type=str,
        default=str(DEFAULT_R30_G1_REPLAY_OVERRIDE),
        help="Optional Region-30 gamma=1 replay_summary override CSV path.",
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        choices=["cost_min", "drone_max"],
        default="cost_min",
        help="Representative solution selection per (instance,gamma): cost_min or drone_max.",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        choices=["best_of_trials", "mean_of_trials", "seed_paired_mean"],
        default="best_of_trials",
        help=(
            "Trend aggregation mode: best_of_trials (legacy), "
            "mean_of_trials, or seed_paired_mean."
        ),
    )
    parser.add_argument(
        "--apply-default-overrides",
        action="store_true",
        help="Apply built-in rerun override CSVs (disabled by default).",
    )
    return parser.parse_args()


def annotate_smart(
    ax,
    x_vals,
    y_vals,
    labels,
    *,
    color: str,
    used_positions: list[tuple[float, float]],
    dpi: float,
    prefer_above: bool = True,
    avoid_boxes: list[tuple[float, float, float, float]] | None = None,
) -> None:
    """Annotate points with simple overlap-avoidance in display coordinates."""
    offsets_pt = [10, -12, 16, -18, 22, -24, 28, -30]
    if not prefer_above:
        offsets_pt = [-12, 10, -18, 16, -24, 22, -30, 28]
    x_nudges_pt = [0, 8, -8, 12, -12]
    x_gap_px = 34
    y_gap_px = 20
    px_per_pt = dpi / 72.0

    for x, y, label in zip(x_vals, y_vals, labels):
        x_px, y_px = ax.transData.transform((x, y))
        chosen_offset = offsets_pt[-1]
        chosen_dx = 0
        found = False
        for off in offsets_pt:
            for dx in x_nudges_pt:
                candidate_x_px = x_px + dx * px_per_pt
                candidate_y_px = y_px + off * px_per_pt
                collision = any(
                    abs(candidate_x_px - ux) < x_gap_px and abs(candidate_y_px - uy) < y_gap_px
                    for ux, uy in used_positions
                )
                in_avoid_box = False
                if avoid_boxes:
                    for x0, y0, x1, y1 in avoid_boxes:
                        if x0 <= candidate_x_px <= x1 and y0 <= candidate_y_px <= y1:
                            in_avoid_box = True
                            break
                if not collision and not in_avoid_box:
                    chosen_offset = off
                    chosen_dx = dx
                    found = True
                    break
            if found:
                break

        va = "bottom" if chosen_offset > 0 else "top"
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(chosen_dx, chosen_offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=10,
            fontweight="bold",
            color=color,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.2),
        )
        used_positions.append((x_px + chosen_dx * px_per_pt, y_px + chosen_offset * px_per_pt))


def expand_axis_limits(ax, values, *, pad_ratio: float = 0.12, min_pad: float = 0.5) -> None:
    """Add vertical headroom/footroom so annotations stay inside plot area."""
    vals = [float(v) for v in values]
    if not vals:
        return
    vmin = min(vals)
    vmax = max(vals)
    span = max(vmax - vmin, 1e-9)
    pad = max(span * pad_ratio, min_pad)
    ax.set_ylim(vmin - pad, vmax + pad)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    return pd.read_csv(path)


def _instance_stem(path_text: str) -> str:
    return Path(path_text).stem


def _instance_region(stem: str) -> str:
    # Expected: R_<region>_<size>_<id>
    parts = stem.split("_")
    if len(parts) >= 2 and parts[1].isdigit():
        return parts[1]
    return "unknown"


def _load_best_frames(base_dir: Path, prefixes: Iterable[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for prefix in prefixes:
        path = base_dir / f"{prefix}_best_all.csv"
        df = _read_csv(path).copy()
        df["instance_name"] = df["instance"].map(_instance_stem)
        df["region"] = df["instance_name"].map(_instance_region)
        df["gamma"] = df["gamma"].astype(float).astype(int)
        df["best_cost"] = df["best_cost"].astype(float)
        df["best_drone_customers"] = df["best_drone_customers"].astype(float)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _load_trial_frames(base_dir: Path, prefixes: Iterable[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for prefix in prefixes:
        path = base_dir / f"{prefix}_trials_all.csv"
        df = _read_csv(path).copy()
        df["instance_name"] = df["instance"].map(_instance_stem)
        df["region"] = df["instance_name"].map(_instance_region)
        df["gamma"] = df["gamma"].astype(float).astype(int)
        df["best_cost"] = df["best_cost"].astype(float)
        df["best_drone_customers"] = df["best_drone_customers"].astype(float)
        df["run_time"] = pd.to_numeric(df.get("run_time", 0.0), errors="coerce").fillna(0.0)
        df["feasible_flag"] = df.get("feasible", 0).astype(str).str.lower().isin(
            ["1", "true", "t", "yes", "y"]
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _apply_trial_overrides(trial_df: pd.DataFrame, override_paths: Iterable[Path]) -> pd.DataFrame:
    base = trial_df.copy()
    for override_path in override_paths:
        if not override_path.exists():
            continue
        ov = _read_csv(override_path).copy()
        ov["instance_name"] = ov["instance"].map(_instance_stem)
        ov["region"] = ov["instance_name"].map(_instance_region)
        ov["gamma"] = ov["gamma"].astype(float).astype(int)
        ov["best_cost"] = ov["best_cost"].astype(float)
        ov["best_drone_customers"] = ov["best_drone_customers"].astype(float)
        ov["run_time"] = pd.to_numeric(ov.get("run_time", 0.0), errors="coerce").fillna(0.0)
        ov["feasible_flag"] = ov.get("feasible", 0).astype(str).str.lower().isin(
            ["1", "true", "t", "yes", "y"]
        )
        if ov.empty:
            continue

        keys = set(zip(ov["instance_name"], ov["gamma"]))
        keep_mask = ~base.apply(
            lambda r: (r["instance_name"], int(r["gamma"])) in keys,
            axis=1,
        )
        base = pd.concat([base[keep_mask].copy(), ov], ignore_index=True)
    return base


def _select_representative_from_trials(
    trial_df: pd.DataFrame,
    *,
    selection_mode: str,
) -> pd.DataFrame:
    selected_rows: list[pd.Series] = []
    grouped = trial_df.groupby(["instance_name", "gamma"], sort=False)

    for _, group in grouped:
        feasible = group[group["feasible_flag"] & group["best_cost"].map(pd.notna)].copy()
        feasible = feasible[feasible["best_cost"].map(lambda v: pd.notna(v) and float(v) < float("inf"))]
        candidates = feasible if not feasible.empty else group

        if selection_mode == "drone_max":
            candidates = candidates.sort_values(
                ["best_drone_customers", "best_cost", "run_time"],
                ascending=[False, True, True],
            )
        else:
            candidates = candidates.sort_values(
                ["best_cost", "best_drone_customers", "run_time"],
                ascending=[True, False, True],
            )
        selected_rows.append(candidates.iloc[0])

    cols = [
        "instance",
        "instance_name",
        "region",
        "gamma",
        "best_cost",
        "best_drone_customers",
    ]
    return pd.DataFrame(selected_rows)[cols].reset_index(drop=True)


def _apply_r30_gamma1_best_override(best_df: pd.DataFrame, override_path: Path) -> pd.DataFrame:
    if not override_path.exists():
        return best_df

    ov = _read_csv(override_path).copy()
    ov["instance_name"] = ov["instance"].map(_instance_stem)
    ov["region"] = ov["instance_name"].map(_instance_region)
    ov["gamma"] = ov["gamma"].astype(float).astype(int)
    ov["best_cost"] = ov["best_cost"].astype(float)
    ov["best_drone_customers"] = ov["best_drone_customers"].astype(float)
    ov = ov[ov["gamma"] == 1]

    if ov.empty:
        return best_df

    base = best_df.copy()
    to_replace = set(ov["instance_name"].tolist())
    keep_mask = ~((base["region"] == "30") & (base["gamma"] == 1) & (base["instance_name"].isin(to_replace)))
    base = base[keep_mask].copy()

    cols = [
        "instance",
        "instance_name",
        "region",
        "gamma",
        "best_cost",
        "best_drone_customers",
    ]
    missing_cols = [c for c in cols if c not in ov.columns]
    if missing_cols:
        raise ValueError(f"Best override missing columns: {missing_cols}")

    merged = pd.concat([base, ov[cols]], ignore_index=True)
    return merged


def _apply_best_overrides(best_df: pd.DataFrame, override_paths: Iterable[Path]) -> pd.DataFrame:
    base = best_df.copy()
    for override_path in override_paths:
        if not override_path.exists():
            continue
        ov = _read_csv(override_path).copy()
        ov["instance_name"] = ov["instance"].map(_instance_stem)
        ov["region"] = ov["instance_name"].map(_instance_region)
        ov["gamma"] = ov["gamma"].astype(float).astype(int)
        ov["best_cost"] = ov["best_cost"].astype(float)
        ov["best_drone_customers"] = ov["best_drone_customers"].astype(float)
        if ov.empty:
            continue

        cols = [
            "instance",
            "instance_name",
            "region",
            "gamma",
            "best_cost",
            "best_drone_customers",
        ]
        missing_cols = [c for c in cols if c not in ov.columns]
        if missing_cols:
            raise ValueError(f"Best override missing columns: {missing_cols} in {override_path}")

        keys = set(zip(ov["instance_name"], ov["gamma"]))
        keep_mask = ~base.apply(lambda r: (r["instance_name"], int(r["gamma"])) in keys, axis=1)
        base = pd.concat([base[keep_mask].copy(), ov[cols]], ignore_index=True)
    return base


def _build_gamma_trend(best_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = (
        best_df[best_df["gamma"] == 0][["instance_name", "best_cost"]]
        .rename(columns={"best_cost": "baseline_cost"})
        .copy()
    )
    merged = best_df.merge(baseline, on="instance_name", how="left")
    merged["cost_saving_vs_g0_pct"] = (
        (merged["baseline_cost"] - merged["best_cost"]) / merged["baseline_cost"] * 100.0
    )

    by_region = (
        merged.groupby(["region", "gamma"], as_index=False)
        .agg(
            avg_cost_saving_vs_g0_pct=("cost_saving_vs_g0_pct", "mean"),
            avg_best_drone_served_nodes=("best_drone_customers", "mean"),
            avg_best_cost=("best_cost", "mean"),
            num_instances=("instance_name", "nunique"),
        )
        .sort_values(["region", "gamma"])
    )

    overall = (
        merged.groupby(["gamma"], as_index=False)
        .agg(
            avg_cost_saving_vs_g0_pct=("cost_saving_vs_g0_pct", "mean"),
            avg_best_drone_served_nodes=("best_drone_customers", "mean"),
            avg_best_cost=("best_cost", "mean"),
            num_instances=("instance_name", "nunique"),
        )
        .sort_values(["gamma"])
    )
    overall["region"] = "30+40+50"

    cols = [
        "region",
        "gamma",
        "num_instances",
        "avg_best_cost",
        "avg_cost_saving_vs_g0_pct",
        "avg_best_drone_served_nodes",
    ]
    return (
        overall[cols].reset_index(drop=True),
        by_region[cols].sort_values(["region", "gamma"]).reset_index(drop=True),
    )


def _build_gamma_trend_from_trials(
    trial_df: pd.DataFrame,
    *,
    aggregation: str,
    selection_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if aggregation == "best_of_trials":
        best_df = _select_representative_from_trials(
            trial_df,
            selection_mode=selection_mode,
        )
        return _build_gamma_trend(best_df)

    feasible = trial_df[trial_df["feasible_flag"] & trial_df["best_cost"].map(pd.notna)].copy()
    feasible = feasible[feasible["best_cost"].map(lambda v: pd.notna(v) and float(v) < float("inf"))]
    if feasible.empty:
        raise ValueError("No feasible rows in trials for trend aggregation.")

    if aggregation == "mean_of_trials":
        grouped = (
            feasible.groupby(["instance_name", "region", "gamma"], as_index=False)
            .agg(
                best_cost=("best_cost", "mean"),
                best_drone_customers=("best_drone_customers", "mean"),
                instance=("instance", "first"),
            )
        )
        return _build_gamma_trend(grouped)

    if aggregation == "seed_paired_mean":
        if "seed" not in feasible.columns:
            raise ValueError("seed_paired_mean requires 'seed' column in trials CSV.")
        feasible["seed"] = pd.to_numeric(feasible["seed"], errors="coerce")
        feasible = feasible[feasible["seed"].map(pd.notna)].copy()

        g0 = feasible[feasible["gamma"] == 0][["instance_name", "region", "seed", "best_cost"]].copy()
        g0 = g0.rename(columns={"best_cost": "baseline_cost"})
        paired = feasible.merge(g0, on=["instance_name", "region", "seed"], how="inner")
        if paired.empty:
            raise ValueError("seed_paired_mean found no (instance,seed,gamma=0) pairs.")
        paired["cost_saving_vs_g0_pct"] = (
            (paired["baseline_cost"] - paired["best_cost"]) / paired["baseline_cost"] * 100.0
        )

        by_region = (
            paired.groupby(["region", "gamma"], as_index=False)
            .agg(
                avg_cost_saving_vs_g0_pct=("cost_saving_vs_g0_pct", "mean"),
                avg_best_drone_served_nodes=("best_drone_customers", "mean"),
                avg_best_cost=("best_cost", "mean"),
                num_instances=("instance_name", "nunique"),
            )
            .sort_values(["region", "gamma"])
        )
        overall = (
            paired.groupby(["gamma"], as_index=False)
            .agg(
                avg_cost_saving_vs_g0_pct=("cost_saving_vs_g0_pct", "mean"),
                avg_best_drone_served_nodes=("best_drone_customers", "mean"),
                avg_best_cost=("best_cost", "mean"),
                num_instances=("instance_name", "nunique"),
            )
            .sort_values(["gamma"])
        )
        overall["region"] = "30+40+50"
        cols = [
            "region",
            "gamma",
            "num_instances",
            "avg_best_cost",
            "avg_cost_saving_vs_g0_pct",
            "avg_best_drone_served_nodes",
        ]
        return (
            overall[cols].reset_index(drop=True),
            by_region[cols].reset_index(drop=True),
        )

    raise ValueError(f"Unsupported aggregation mode: {aggregation}")


def _plot_gamma_trend_all_avg(trend_df: pd.DataFrame, output_png: Path, output_pdf: Path) -> None:
    data = trend_df.sort_values("gamma")
    x = data["gamma"]
    y_cost = data["avg_cost_saving_vs_g0_pct"]
    y_drone = data["avg_best_drone_served_nodes"]

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    # Match battery/drone_count plotting style
    bar_red_fill = "#F6D8E6"
    bar_red_border = "#E38D83"
    bar_blue_fill = "#CFEEF6"
    bar_blue_border = "#3886C2"
    color_cost = bar_blue_border
    color_drone = bar_red_border

    line1 = ax1.plot(
        x,
        y_cost,
        marker="s",
        markerfacecolor=bar_blue_fill,
        markeredgewidth=2,
        markersize=10,
        linewidth=3,
        color=color_cost,
        label="Cost Saving (%)",
    )
    ax1.axhline(0.0, color="#999999", linewidth=1.0, linestyle="--")
    ax1.set_xlabel("Gamma (Energy Uncertainty Budget)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Avg Cost Saving vs Baseline (%)", color=color_cost, fontsize=14, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_cost, labelsize=12)
    ax1.tick_params(axis="x", labelsize=12)
    ax1.set_xticks(x)
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2 = ax1.twinx()
    line2 = ax2.plot(
        x,
        y_drone,
        marker="o",
        markerfacecolor=bar_red_fill,
        markeredgewidth=2,
        markersize=10,
        linewidth=3,
        color=color_drone,
        linestyle="--",
        label="Drone Customers",
    )
    ax2.set_ylabel("Avg Drone Served Customers", color=color_drone, fontsize=14, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_drone, labelsize=12)

    # Same headroom/footroom policy as battery/drone_count scripts.
    expand_axis_limits(ax1, y_cost, pad_ratio=0.10, min_pad=0.6)
    expand_axis_limits(ax2, y_drone, pad_ratio=0.08, min_pad=0.25)

    # Keep legend in upper-left inside plot, and let annotations avoid legend area.
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    legend = ax1.legend(
        lines,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        ncol=1,
        frameon=True,
        shadow=True,
        fontsize=11,
    )

    # Same smart annotation style as battery/drone_count scripts.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    lb = legend.get_window_extent(renderer=renderer)
    legend_margin_px = 10.0
    avoid_boxes = [
        (
            lb.x0 - legend_margin_px,
            lb.y0 - legend_margin_px,
            lb.x1 + legend_margin_px,
            lb.y1 + legend_margin_px,
        )
    ]
    used_positions: list[tuple[float, float]] = []
    annotate_smart(
        ax1,
        x,
        y_cost,
        [f"{v:.2f}%" for v in y_cost],
        color=color_cost,
        used_positions=used_positions,
        dpi=fig.dpi,
        prefer_above=False,
        avoid_boxes=avoid_boxes,
    )
    annotate_smart(
        ax2,
        x,
        y_drone,
        [f"{v:.2f}" for v in y_drone],
        color=color_drone,
        used_positions=used_positions,
        dpi=fig.dpi,
        prefer_above=True,
        avoid_boxes=avoid_boxes,
    )
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300)
    plt.savefig(output_pdf, dpi=300)
    plt.close(fig)


def _load_replay_frames(base_dir: Path, prefixes: Iterable[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for prefix in prefixes:
        path = base_dir / f"{prefix}_replay_summary_all.csv"
        df = _read_csv(path).copy()
        df["gamma"] = df["gamma"].astype(float).astype(int)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _apply_r30_gamma1_replay_override(replay_df: pd.DataFrame, override_path: Path) -> pd.DataFrame:
    if not override_path.exists():
        return replay_df

    ov = _read_csv(override_path).copy()
    ov["gamma"] = ov["gamma"].astype(float).astype(int)
    ov = ov[ov["gamma"] == 1]
    if ov.empty:
        return replay_df

    base = replay_df.copy()
    key_cols = ["instance", "distribution", "gamma"]
    ov_keys = set(zip(ov["instance"], ov["distribution"], ov["gamma"]))
    keep_mask = ~base.apply(
        lambda r: (r["instance"], r["distribution"], int(r["gamma"])) in ov_keys,
        axis=1,
    )
    base = base[keep_mask].copy()
    merged = pd.concat([base, ov], ignore_index=True)
    return merged


def _apply_replay_overrides(replay_df: pd.DataFrame, override_paths: Iterable[Path]) -> pd.DataFrame:
    base = replay_df.copy()
    for override_path in override_paths:
        if not override_path.exists():
            continue
        ov = _read_csv(override_path).copy()
        ov["gamma"] = ov["gamma"].astype(float).astype(int)
        if ov.empty:
            continue
        keys = set(zip(ov["instance"], ov["distribution"], ov["gamma"]))
        keep_mask = ~base.apply(
            lambda r: (r["instance"], r["distribution"], int(r["gamma"])) in keys,
            axis=1,
        )
        base = pd.concat([base[keep_mask].copy(), ov], ignore_index=True)
    return base


def _export_table6_by_distribution(replay_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_paths: list[Path] = []
    dists = ["ND", "UD", "NDC"]
    for dist in dists:
        sub = replay_df[replay_df["distribution"] == dist].copy()
        if sub.empty:
            continue
        table = sub.rename(
            columns={
                "instance": "Instance",
                "gamma": "gamma",
                "avg_cost": "AvgCost",
                "std_cost": "StdCost",
                "max_cost": "MaxCost",
                "min_cost": "MinCost",
                "avg_unserved": "AvgUnserved",
                "p0_all_served": "P(Unserved=0)%",
                "avg_no_takeoff": "AvgNoTakeoff",
                "avg_abort_return": "AvgAbortReturn",
            }
        )
        table["P(Unserved=0)%"] = table["P(Unserved=0)%"] * 100.0
        table = table[
            [
                "Instance",
                "gamma",
                "AvgCost",
                "StdCost",
                "MaxCost",
                "MinCost",
                "AvgUnserved",
                "P(Unserved=0)%",
                "AvgNoTakeoff",
                "AvgAbortReturn",
            ]
        ].sort_values(["Instance", "gamma"])
        table = table.round(
            {
                "AvgCost": 4,
                "StdCost": 4,
                "MaxCost": 4,
                "MinCost": 4,
                "AvgUnserved": 4,
                "P(Unserved=0)%": 2,
                "AvgNoTakeoff": 4,
                "AvgAbortReturn": 4,
            }
        )

        out_path = output_dir / f"table6_instance25_r30_r40_r50_{dist}.csv"
        table.to_csv(out_path, index=False)
        output_paths.append(out_path)
    return output_paths


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    prefixes = [x.strip() for x in args.prefixes.split(",") if x.strip()]
    if not prefixes:
        raise ValueError("No valid prefixes provided.")

    trial_df = _load_trial_frames(base_dir, prefixes)
    if args.apply_default_overrides:
        trial_overrides: list[Path] = [*DEFAULT_G0_TRIAL_OVERRIDES, DEFAULT_R30_G1_TRIAL_OVERRIDE]
        trial_df = _apply_trial_overrides(trial_df, trial_overrides)
    trend_all_df, trend_by_region_df = _build_gamma_trend_from_trials(
        trial_df,
        aggregation=args.aggregation,
        selection_mode=args.selection_mode,
    )

    suffix_parts: list[str] = []
    if args.selection_mode != "cost_min":
        suffix_parts.append(args.selection_mode)
    if args.aggregation != "best_of_trials":
        suffix_parts.append(args.aggregation)
    suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ""
    trend_csv = output_dir / f"instance25_r30_r40_r50_gamma_trends_all_avg{suffix}.csv"
    trend_all_df.round(
        {
            "avg_best_cost": 4,
            "avg_cost_saving_vs_g0_pct": 4,
            "avg_best_drone_served_nodes": 4,
        }
    ).to_csv(trend_csv, index=False)
    trend_region_csv = output_dir / f"instance25_r30_r40_r50_gamma_trends_by_region{suffix}.csv"
    trend_by_region_df.round(
        {
            "avg_best_cost": 4,
            "avg_cost_saving_vs_g0_pct": 4,
            "avg_best_drone_served_nodes": 4,
        }
    ).to_csv(trend_region_csv, index=False)

    trend_png = output_dir / f"instance25_r30_r40_r50_gamma_trends_all_avg{suffix}.png"
    trend_pdf = output_dir / f"instance25_r30_r40_r50_gamma_trends_all_avg{suffix}.pdf"
    _plot_gamma_trend_all_avg(trend_all_df, trend_png, trend_pdf)

    replay_df = _load_replay_frames(base_dir, prefixes)
    if args.apply_default_overrides:
        replay_df = _apply_replay_overrides(replay_df, DEFAULT_G0_REPLAY_OVERRIDES)
        replay_df = _apply_r30_gamma1_replay_override(replay_df, Path(args.r30_gamma1_replay_override))
    table_paths = _export_table6_by_distribution(replay_df, output_dir)

    print(f"trend csv: {trend_csv}")
    print(f"trend by-region csv: {trend_region_csv}")
    print(f"trend plot png: {trend_png}")
    print(f"trend plot pdf: {trend_pdf}")
    for p in table_paths:
        print(f"table6 csv: {p}")


if __name__ == "__main__":
    main()
