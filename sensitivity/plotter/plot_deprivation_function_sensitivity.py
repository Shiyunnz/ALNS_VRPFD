#!/usr/bin/env python3
"""Plot deprivation-cost function sensitivity in the paper's dual-axis style."""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


CONDITION_ORDER = [
    "exponential_rho_0.15",
    "exponential_rho_0.208333",
    "exponential_rho_0.30",
    "linear",
    "quadratic",
]

CONDITION_LABELS = {
    "exponential_rho_0.15": "Exp.\n$\\rho^{\\mathrm{dep}}=0.15$",
    "exponential_rho_0.208333": "Exp. baseline\n$\\rho^{\\mathrm{dep}}=0.2083$",
    "exponential_rho_0.30": "Exp.\n$\\rho^{\\mathrm{dep}}=0.30$",
    "linear": "Linear",
    "quadratic": "Quadratic",
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Plot deprivation-cost specification sensitivity."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=project_root
        / "results"
        / "revision_experiments"
        / "deprivation_sensitivity"
        / "summary.csv",
        help="Input experiment summary CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root
        / "results"
        / "revision_experiments"
        / "deprivation_sensitivity"
        / "deprivation_cost_sensitivity_plot.pdf",
        help="Output PDF path; a PNG is written with the same stem.",
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
) -> None:
    """Annotate points while avoiding overlaps in display coordinates."""
    offsets_pt = [10, -12, 16, -18, 22, -24]
    x_gap_px = 28
    y_gap_px = 14
    px_per_pt = dpi / 72.0

    for x, y, label in zip(x_vals, y_vals, labels):
        x_px, y_px = ax.transData.transform((x, y))
        chosen_offset = offsets_pt[-1]
        for offset in offsets_pt:
            candidate_y_px = y_px + offset * px_per_pt
            collision = any(
                abs(x_px - used_x) < x_gap_px
                and abs(candidate_y_px - used_y) < y_gap_px
                for used_x, used_y in used_positions
            )
            if not collision:
                chosen_offset = offset
                break

        ax.annotate(
            label,
            xy=(x, y),
            xytext=(0, chosen_offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if chosen_offset > 0 else "top",
            fontsize=10,
            fontweight="bold",
            color=color,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.2),
        )
        used_positions.append((x_px, y_px + chosen_offset * px_per_pt))


def expand_axis_limits(
    ax, values, *, pad_ratio: float = 0.15, min_pad: float = 0.25
) -> None:
    vals = [float(value) for value in values]
    vmin = min(vals)
    vmax = max(vals)
    span = max(vmax - vmin, 1e-9)
    pad = max(span * pad_ratio, min_pad)
    ax.set_ylim(vmin - pad, vmax + pad)


def main() -> None:
    args = parse_args()
    if not args.summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {args.summary_csv}")

    summary = pd.read_csv(args.summary_csv).set_index("condition")
    missing = [condition for condition in CONDITION_ORDER if condition not in summary.index]
    if missing:
        raise ValueError(f"Missing sensitivity conditions: {missing}")

    data = summary.loc[CONDITION_ORDER].reset_index()
    x_vals = list(range(len(data)))
    cost_saving = data["mean_paired_saving_vs_baseline_pct"].astype(float)
    total_delay = data["mean_total_delay_hours"].astype(float)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Match the battery-capacity and drone-fleet sensitivity figures.
    blue_fill = "#CFEEF6"
    blue_border = "#3886C2"
    red_fill = "#F6D8E6"
    red_border = "#E38D83"

    ax1.set_xlabel(
        "Deprivation-cost Specification", fontsize=14, fontweight="bold"
    )
    ax1.set_ylabel(
        "Avg Cost Saving vs Baseline (%)",
        color=blue_border,
        fontsize=14,
        fontweight="bold",
    )
    cost_line = ax1.plot(
        x_vals,
        cost_saving,
        color=blue_border,
        marker="s",
        markerfacecolor=blue_fill,
        markeredgewidth=2,
        markersize=10,
        linewidth=3,
        label="Cost Saving (%)",
    )
    ax1.axhline(0, color="#808080", linewidth=1, alpha=0.65)
    ax1.set_xticks(x_vals)
    ax1.set_xticklabels([CONDITION_LABELS[item] for item in CONDITION_ORDER])
    ax1.tick_params(axis="x", labelsize=11)
    ax1.tick_params(axis="y", labelcolor=blue_border, labelsize=12)
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2 = ax1.twinx()
    ax2.set_ylabel(
        "Avg Total Tardiness (h)",
        color=red_border,
        fontsize=14,
        fontweight="bold",
    )
    delay_line = ax2.plot(
        x_vals,
        total_delay,
        color=red_border,
        marker="o",
        markerfacecolor=red_fill,
        markeredgewidth=2,
        markersize=10,
        linewidth=3,
        linestyle="--",
        label="Total Tardiness (h)",
    )
    ax2.tick_params(axis="y", labelcolor=red_border, labelsize=12)

    expand_axis_limits(ax1, cost_saving, pad_ratio=0.16, min_pad=0.8)
    expand_axis_limits(ax2, total_delay, pad_ratio=0.16, min_pad=0.15)

    fig.canvas.draw()
    used_positions: list[tuple[float, float]] = []
    annotate_smart(
        ax1,
        x_vals,
        cost_saving,
        [f"{value:.2f}%" for value in cost_saving],
        color=blue_border,
        used_positions=used_positions,
        dpi=fig.dpi,
    )
    annotate_smart(
        ax2,
        x_vals,
        total_delay,
        [f"{value:.2f} h" for value in total_delay],
        color=red_border,
        used_positions=used_positions,
        dpi=fig.dpi,
    )

    lines = cost_line + delay_line
    ax1.legend(
        lines,
        [line.get_label() for line in lines],
        loc="upper left",
        frameon=True,
        shadow=True,
        fontsize=12,
    )

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_png = args.output.with_suffix(".png")
    fig.savefig(args.output, dpi=300)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)
    print(f"Plot saved to: {args.output}")
    print(f"Plot saved to: {output_png}")


if __name__ == "__main__":
    main()
