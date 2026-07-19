"""Plot raw literature deprivation functions for four supply classes over 8h.

Water uses the original Holguin-Veras exponential function with time in hours.
Medicine, food, and tent use Wang et al.'s fitted logistic deprivation-level
functions with time in days; each curve is baseline-adjusted by subtracting
its value at zero delay.
"""

from __future__ import annotations

from pathlib import Path
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

WANG_LOGISTIC_PARAMS = {
    "medicine": {
        "label": "Medicine (Wang DLF)",
        "k": 9.772697,
        "a": 3.9031,
        "b": 0.7919,
    },
    "food": {
        "label": "Food (Wang DLF)",
        "k": 9.745492,
        "a": 4.2280,
        "b": 0.7407,
    },
    "tent": {
        "label": "Tent (Wang DLF)",
        "k": 9.752874,
        "a": 4.2047,
        "b": 0.7437,
    },
}

COLORS = {
    "medicine": "#D55E00",
    "water": "#0072B2",
    "food": "#009E73",
    "tent": "#CC79A7",
}

LINESTYLES = {
    "medicine": "-",
    "water": "--",
    "food": "-.",
    "tent": ":",
}


def raw_holguin_water_cost(tau_hours: np.ndarray) -> np.ndarray:
    """Original water deprivation cost function with tau measured in hours."""

    return np.exp(1.5031 + 0.1172 * tau_hours) - math.exp(1.5031)


def raw_wang_logistic_increment(tau_hours: np.ndarray, supply_class: str) -> np.ndarray:
    """Baseline-adjusted Wang logistic DLF increment with X measured in days."""

    params = WANG_LOGISTIC_PARAMS[supply_class]
    x_days = tau_hours / 24.0
    value = params["k"] / (1.0 + params["a"] * np.exp(-params["b"] * x_days))
    baseline = params["k"] / (1.0 + params["a"])
    return value - baseline


def main() -> None:
    output_dir = PROJECT_ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    tau = np.linspace(0.0, 8.0, 800)
    curves = {
        "medicine": raw_wang_logistic_increment(tau, "medicine"),
        "water": raw_holguin_water_cost(tau),
        "food": raw_wang_logistic_increment(tau, "food"),
        "tent": raw_wang_logistic_increment(tau, "tent"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), constrained_layout=True)

    for key in ["medicine", "water", "food", "tent"]:
        label = "Water (Holguin-Veras DCF)" if key == "water" else WANG_LOGISTIC_PARAMS[key]["label"]
        axes[0].plot(
            tau,
            curves[key],
            color=COLORS[key],
            linestyle=LINESTYLES[key],
            linewidth=2.0,
            label=label,
        )

    for key in ["medicine", "food", "tent"]:
        axes[1].plot(
            tau,
            curves[key],
            color=COLORS[key],
            linestyle=LINESTYLES[key],
            linewidth=2.0,
            label=WANG_LOGISTIC_PARAMS[key]["label"],
        )

    axes[0].set_title("(a) Raw functions, 0-8h")
    axes[0].set_xlabel("Delay time $\\tau$ (hours)")
    axes[0].set_ylabel("Raw baseline-adjusted value")
    axes[0].set_xlim(0.0, 8.0)
    axes[0].set_ylim(bottom=0.0)
    axes[0].grid(True, linewidth=0.4, alpha=0.3)
    axes[0].legend(frameon=False, loc="upper left")

    axes[1].set_title("(b) Wang DLF zoom")
    axes[1].set_xlabel("Delay time $\\tau$ (hours)")
    axes[1].set_ylabel("DLF increment")
    axes[1].set_xlim(0.0, 8.0)
    axes[1].set_ylim(bottom=0.0)
    axes[1].grid(True, linewidth=0.4, alpha=0.3)
    axes[1].legend(frameon=False, loc="upper left")

    png_path = output_dir / "raw_four_supply_deprivation_costs_8h.png"
    pdf_path = output_dir / "raw_four_supply_deprivation_costs_8h.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print()
    print("Raw baseline-adjusted values:")
    print(f"{'tau(h)':>8} {'Medicine':>12} {'Water':>12} {'Food':>12} {'Tent':>12}")
    for hour in [0, 1, 2, 4, 6, 8]:
        values = {
            "medicine": raw_wang_logistic_increment(np.array([hour], dtype=float), "medicine")[0],
            "water": raw_holguin_water_cost(np.array([hour], dtype=float))[0],
            "food": raw_wang_logistic_increment(np.array([hour], dtype=float), "food")[0],
            "tent": raw_wang_logistic_increment(np.array([hour], dtype=float), "tent")[0],
        }
        print(
            f"{hour:8.1f}"
            f" {values['medicine']:12.4f}"
            f" {values['water']:12.4f}"
            f" {values['food']:12.4f}"
            f" {values['tent']:12.4f}"
        )


if __name__ == "__main__":
    main()
