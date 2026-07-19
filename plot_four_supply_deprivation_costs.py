"""Plot four supply-class deprivation cost functions."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from alns_vrpfd.deprivation import (  # noqa: E402
    DEFAULT_SUPPLY_CLASS_SEQUENCE,
    WANG_SUPPLY_CLASSES,
    deprivation_cost,
)


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


def _series(tau: np.ndarray, supply_class: str) -> np.ndarray:
    return np.array([deprivation_cost(float(t), supply_class, cost_lambda=30.0, rho=0.2083, normalized=True) for t in tau])


def main() -> None:
    output_dir = PROJECT_ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
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
    })

    tau_short = np.linspace(0.001, 2.0, 500)
    tau_wang = np.linspace(0.0, 2.0, 500)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), constrained_layout=True)

    for supply_class in DEFAULT_SUPPLY_CLASS_SEQUENCE:
        spec = WANG_SUPPLY_CLASSES[supply_class]
        label = f"{spec.label} ($\\beta$={spec.beta:.4f}, $\\omega$={spec.omega:.2f})"
        axes[0].plot(
            tau_short,
            _series(tau_short, supply_class),
            color=COLORS[supply_class],
            linestyle=LINESTYLES[supply_class],
            linewidth=2.0,
            label=label,
        )
        if supply_class != "water":
            axes[1].plot(
                tau_wang,
                _series(tau_wang, supply_class),
                color=COLORS[supply_class],
                linestyle=LINESTYLES[supply_class],
                linewidth=2.0,
                label=f"{spec.label}: $\\beta$={spec.beta:.4f}, $\\omega$={spec.omega:.2f}",
            )

    axes[0].set_title("(a) Four classes, operational scale")
    axes[0].set_xlabel("Delay $\\tau$ (hours)")
    axes[0].set_ylabel("Deprivation cost")
    axes[0].set_xlim(0.0, 2.0)
    axes[0].set_ylim(bottom=0.0)
    axes[0].grid(True, linewidth=0.4, alpha=0.3)
    axes[0].legend(frameon=False, loc="upper left")

    axes[1].set_title("(b) Wang logistic classes")
    axes[1].set_xlabel("Delay $\\tau$ (hours)")
    axes[1].set_ylabel("Baseline-adjusted DLF")
    axes[1].set_xlim(0.0, 2.0)
    axes[1].set_ylim(bottom=0.0)
    axes[1].grid(True, linewidth=0.4, alpha=0.3)
    axes[1].legend(frameon=False, loc="upper left")

    png_path = output_dir / "four_supply_deprivation_costs.png"
    pdf_path = output_dir / "four_supply_deprivation_costs.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print()
    print("Cost values:")
    print(f"{'tau(h)':>8} {'Medicine':>12} {'Water':>12} {'Food':>12} {'Tent':>12}")
    for tau in [0.025, 0.1, 0.5, 0.85, 1.0, 2.0]:
        values = [deprivation_cost(tau, c, cost_lambda=30.0, rho=0.2083, normalized=True) for c in DEFAULT_SUPPLY_CLASS_SEQUENCE]
        print(f"{tau:8.2f} " + " ".join(f"{v:12.3f}" for v in values))


if __name__ == "__main__":
    main()
