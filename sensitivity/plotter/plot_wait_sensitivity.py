#!/usr/bin/env python3
"""Plot truck-drone waiting-time sensitivity with the paper sensitivity style."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def annotate_smart(
    ax,
    x_vals,
    y_vals,
    labels,
    *,
    color: str,
    used_positions: list[tuple[float, float]],
    dpi: float,
) -> None:
    """Annotate points with simple overlap-avoidance in display coordinates."""
    offsets_pt = [10, -12, 16, -18, 22, -24]
    x_gap_px = 22
    y_gap_px = 14
    px_per_pt = dpi / 72.0

    for x, y, label in zip(x_vals, y_vals, labels):
        x_px, y_px = ax.transData.transform((x, y))
        chosen_offset = offsets_pt[-1]

        for off in offsets_pt:
            candidate_y_px = y_px + off * px_per_pt
            collision = any(
                abs(x_px - ux) < x_gap_px and abs(candidate_y_px - uy) < y_gap_px
                for ux, uy in used_positions
            )
            if not collision:
                chosen_offset = off
                break

        va = "bottom" if chosen_offset > 0 else "top"
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(0, chosen_offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=10,
            fontweight="bold",
            color=color,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.2),
        )
        used_positions.append((x_px, y_px + chosen_offset * px_per_pt))


def expand_axis_limits(ax, values, *, pad_ratio: float = 0.12, min_pad: float = 0.5) -> None:
    vals = [float(v) for v in values]
    if not vals:
        return
    vmin = min(vals)
    vmax = max(vals)
    span = max(vmax - vmin, 1e-9)
    pad = max(span * pad_ratio, min_pad)
    ax.set_ylim(vmin - pad, vmax + pad)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot wait sensitivity using average or best-of-trials values."
    )
    parser.add_argument(
        "--trials-csv",
        type=Path,
        default=Path(
            "sensitivity/results_new/wait_sensitivity_deadline_cost_rerun_20260613/trials.csv"
        ),
        help="Input incremental trials CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "sensitivity/results_new/wait_sensitivity_deadline_cost_rerun_20260613/"
            "wait_sensitivity_best_paper_style.pdf"
        ),
        help="Output PDF path; PNG will be written with the same stem.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional output CSV for the display table.",
    )
    parser.add_argument(
        "--mode",
        choices=["average", "best"],
        default="average",
        help="Display average repeated-run values or best-of-trials values.",
    )
    parser.add_argument(
        "--baseline-wait",
        type=float,
        default=20.0,
        help="Waiting tolerance used as the cost-saving baseline.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title. Default is no title, matching paper plots.",
    )
    parser.add_argument(
        "--legend-loc",
        type=str,
        default="upper left",
        help="Matplotlib legend location.",
    )
    return parser.parse_args()


def _load_trials(trials_csv: Path) -> pd.DataFrame:
    if not trials_csv.exists():
        raise FileNotFoundError(f"Trials CSV not found: {trials_csv}")

    df = pd.read_csv(trials_csv)
    required = {
        "wait_tolerance",
        "seed",
        "trial",
        "best_cost",
        "best_drone_customers",
        "max_rendezvous_deviation",
        "avg_rendezvous_deviation",
        "rendezvous_count",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns in {trials_csv}: {missing}")

    df = df.copy()
    numeric_cols = [
        "wait_tolerance",
        "best_cost",
        "best_drone_customers",
        "max_rendezvous_deviation",
        "avg_rendezvous_deviation",
        "rendezvous_count",
        "run_time",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["wait_tolerance", "best_cost"])


def build_average_summary(trials_csv: Path, baseline_wait: float) -> pd.DataFrame:
    df = _load_trials(trials_csv)
    grouped = (
        df.groupby("wait_tolerance", as_index=False)
        .agg(
            n_trials=("best_cost", "size"),
            avg_cost=("best_cost", "mean"),
            std_cost=("best_cost", "std"),
            min_cost=("best_cost", "min"),
            max_cost=("best_cost", "max"),
            avg_drone_customers=("best_drone_customers", "mean"),
            std_drone_customers=("best_drone_customers", "std"),
            avg_max_rendezvous_deviation=("max_rendezvous_deviation", "mean"),
            avg_rendezvous_deviation=("avg_rendezvous_deviation", "mean"),
            avg_rendezvous_count=("rendezvous_count", "mean"),
            avg_run_time=("run_time", "mean"),
        )
        .sort_values("wait_tolerance")
    )

    baseline_rows = grouped[grouped["wait_tolerance"].sub(baseline_wait).abs() < 1e-9]
    if baseline_rows.empty:
        raise ValueError(f"Baseline wait={baseline_wait:g} not found in average summary.")
    baseline_avg_cost = float(baseline_rows.iloc[0]["avg_cost"])

    grouped["baseline_wait_tolerance"] = baseline_wait
    grouped["baseline_avg_cost"] = baseline_avg_cost
    grouped["avg_cost_saving_vs_baseline"] = (
        (baseline_avg_cost - grouped["avg_cost"]) / baseline_avg_cost * 100.0
    )
    return grouped[
        [
            "wait_tolerance",
            "n_trials",
            "avg_cost",
            "std_cost",
            "min_cost",
            "max_cost",
            "baseline_avg_cost",
            "avg_cost_saving_vs_baseline",
            "avg_drone_customers",
            "std_drone_customers",
            "avg_max_rendezvous_deviation",
            "avg_rendezvous_deviation",
            "avg_rendezvous_count",
            "avg_run_time",
        ]
    ]


def build_best_summary(trials_csv: Path, baseline_wait: float) -> pd.DataFrame:
    df = _load_trials(trials_csv)
    idx = df.groupby("wait_tolerance")["best_cost"].idxmin()
    best = df.loc[idx].copy().sort_values("wait_tolerance")

    baseline_rows = best[best["wait_tolerance"].sub(baseline_wait).abs() < 1e-9]
    if baseline_rows.empty:
        raise ValueError(f"Baseline wait={baseline_wait:g} not found in best summary.")
    baseline_cost = float(baseline_rows.iloc[0]["best_cost"])

    best["baseline_wait_tolerance"] = baseline_wait
    best["baseline_best_cost"] = baseline_cost
    best["best_cost_saving_vs_baseline"] = (
        (baseline_cost - best["best_cost"]) / baseline_cost * 100.0
    )

    display_cols = [
        "wait_tolerance",
        "seed",
        "trial",
        "best_cost",
        "baseline_best_cost",
        "best_cost_saving_vs_baseline",
        "best_drone_customers",
        "max_rendezvous_deviation",
        "avg_rendezvous_deviation",
        "rendezvous_count",
        "run_time",
    ]
    return best[[c for c in display_cols if c in best.columns]]


def plot_summary(
    summary: pd.DataFrame,
    output_pdf: Path,
    title: str | None,
    *,
    mode: str,
    legend_loc: str,
) -> None:
    output_png = output_pdf.with_suffix(".png")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Same palette and visual grammar as the existing paper sensitivity plots.
    bar_red_fill = "#F6D8E6"
    bar_red_border = "#E38D83"
    bar_blue_fill = "#CFEEF6"
    bar_blue_border = "#3886C2"
    color_cost = bar_blue_border
    color_drone = bar_red_border

    x_vals = summary["wait_tolerance"].astype(float)
    if mode == "average":
        y_cost = summary["avg_cost_saving_vs_baseline"].astype(float)
        y_drone = summary["avg_drone_customers"].astype(float)
        cost_ylabel = "Avg Cost Saving vs Baseline (%)"
        drone_ylabel = "Avg Drone Served Customers"
        drone_labels = [f"{y:.2f}" for y in y_drone]
    else:
        y_cost = summary["best_cost_saving_vs_baseline"].astype(float)
        y_drone = summary["best_drone_customers"].astype(float)
        cost_ylabel = "Best Cost Saving vs Baseline (%)"
        drone_ylabel = "Best Drone Served Customers"
        drone_labels = [f"{y:.0f}" for y in y_drone]

    ax1.set_xlabel("Mutual Waiting Tolerance", fontsize=14, fontweight="bold")
    ax1.set_ylabel(
        cost_ylabel,
        color=color_cost,
        fontsize=14,
        fontweight="bold",
    )
    line1 = ax1.plot(
        x_vals,
        y_cost,
        color=color_cost,
        marker="s",
        markerfacecolor=bar_blue_fill,
        markeredgewidth=2,
        markersize=10,
        linewidth=3,
        label="Cost Saving (%)",
    )
    ax1.tick_params(axis="y", labelcolor=color_cost, labelsize=12)
    ax1.tick_params(axis="x", labelsize=12)
    ax1.set_xticks(x_vals)
    ax1.set_xticklabels([f"{x:.0f}" for x in x_vals])
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2 = ax1.twinx()
    ax2.set_ylabel(
        drone_ylabel,
        color=color_drone,
        fontsize=14,
        fontweight="bold",
    )
    line2 = ax2.plot(
        x_vals,
        y_drone,
        color=color_drone,
        marker="o",
        markerfacecolor=bar_red_fill,
        markeredgewidth=2,
        markersize=10,
        linewidth=3,
        linestyle="--",
        label="Drone Customers",
    )
    ax2.tick_params(axis="y", labelcolor=color_drone, labelsize=12)

    expand_axis_limits(ax1, y_cost, pad_ratio=0.12, min_pad=0.8)
    expand_axis_limits(ax2, y_drone, pad_ratio=0.12, min_pad=0.6)

    fig.canvas.draw()
    used_positions: list[tuple[float, float]] = []
    annotate_smart(
        ax1,
        x_vals,
        y_cost,
        [f"{y:.2f}%" for y in y_cost],
        color=color_cost,
        used_positions=used_positions,
        dpi=fig.dpi,
    )
    annotate_smart(
        ax2,
        x_vals,
        y_drone,
        drone_labels,
        color=color_drone,
        used_positions=used_positions,
        dpi=fig.dpi,
    )

    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc=legend_loc, frameon=True, shadow=True, fontsize=12)

    if title:
        ax1.set_title(title, fontsize=16, pad=15)

    plt.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_pdf, dpi=300)
    plt.savefig(output_png, dpi=300)
    plt.close(fig)
    print(f"Plot saved to: {output_pdf}")
    print(f"Plot saved to: {output_png}")


def main() -> int:
    args = parse_args()
    if args.mode == "average":
        summary = build_average_summary(args.trials_csv, args.baseline_wait)
        default_summary_name = "wait_sensitivity_average_summary.csv"
    else:
        summary = build_best_summary(args.trials_csv, args.baseline_wait)
        default_summary_name = "wait_sensitivity_best_summary.csv"

    summary_output = (
        args.summary_output
        if args.summary_output
        else args.output.with_name(default_summary_name)
    )
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_output, index=False)
    print(f"{args.mode.title()} summary saved to: {summary_output}")

    plot_summary(summary, args.output, args.title, mode=args.mode, legend_loc=args.legend_loc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
