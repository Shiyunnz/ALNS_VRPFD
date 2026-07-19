"""Plot four supply-class projected deprivation cost functions.

Generates publication-quality figures showing:
  (a) Normalized cost functions (lambda=30, rho=0.2083)
  (b) Normalized, no compression (lambda=30, rho=1/24)
  (c) Raw exponential projection (lambda=1, rho=0.2083, normalized=False)
  (d) Comparison with original Holguin-Veras water baseline
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from alns_vrpfd.deprivation import (
    WANG_SUPPLY_CLASSES,
    DEFAULT_SUPPLY_CLASS_SEQUENCE,
    deprivation_cost,
    HOLGUIN_INTERCEPT,
    MAX_TARDINESS_HOURS,
)
import math


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

LABELS = {
    "medicine": f"Medicine ($\\beta$=0.4558, $\\omega$=1.35)",
    "water": f"Water ($\\beta$=0.4525, $\\omega$=1.35)",
    "food": f"Food ($\\beta$=0.4464, $\\omega$=1.00)",
    "tent": f"Tent ($\\beta$=0.4469, $\\omega$=0.75)",
}


def _series(tau: np.ndarray, supply_class: str, **kwargs) -> np.ndarray:
    return np.array([deprivation_cost(float(t), supply_class, **kwargs) for t in tau])


def main() -> None:
    output_dir = PROJECT_ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    tau_max = MAX_TARDINESS_HOURS
    tau = np.linspace(0.001, tau_max, 500)

    # ==================================================================
    # Figure 1: Multi-panel figure (main result)
    # ==================================================================
    fig, axes = plt.subplots(2, 2, figsize=(8, 6.5), constrained_layout=True)

    # Panel (a): Normalized, recommended parameters lambda=30, rho=0.2083
    ax = axes[0, 0]
    for cls in DEFAULT_SUPPLY_CLASS_SEQUENCE:
        spec = WANG_SUPPLY_CLASSES[cls]
        ax.plot(tau, _series(tau, cls, cost_lambda=30.0, rho=0.2083, normalized=True),
                color=COLORS[cls], linestyle=LINESTYLES[cls], linewidth=1.8,
                label=LABELS[cls])
    ax.axvline(tau_max, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Tardiness $\\tau$ (hours)")
    ax.set_ylabel("Normalized cost $f_{c_i}(\\tau)$")
    ax.set_title("(a) Normalized, $\\lambda{=}30, \\rho{=}0.2083$")
    ax.set_xlim(0, tau_max)
    ax.set_ylim(bottom=0)
    ax.grid(True, linewidth=0.4, alpha=0.3)
    ax.legend(frameon=False, loc="upper left", fontsize=7.5)

    # Panel (b): Normalized, rho=1/24 (no compression)
    ax = axes[0, 1]
    for cls in DEFAULT_SUPPLY_CLASS_SEQUENCE:
        ax.plot(tau, _series(tau, cls, cost_lambda=30.0, rho=1/24, normalized=True),
                color=COLORS[cls], linestyle=LINESTYLES[cls], linewidth=1.8,
                label=LABELS[cls])
    ax.axvline(tau_max, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Tardiness $\\tau$ (hours)")
    ax.set_ylabel("Normalized cost $f_{c_i}(\\tau)$")
    ax.set_title("(b) Normalized, $\\lambda{=}30, \\rho{=}1/24$")
    ax.set_xlim(0, tau_max)
    ax.set_ylim(bottom=0)
    ax.grid(True, linewidth=0.4, alpha=0.3)
    ax.legend(frameon=False, loc="upper left", fontsize=7.5)

    # Panel (c): Raw (non-normalized), lambda=1, rho=0.2083
    ax = axes[1, 0]
    for cls in DEFAULT_SUPPLY_CLASS_SEQUENCE:
        ax.plot(tau, _series(tau, cls, cost_lambda=1.0, rho=0.2083, normalized=False),
                color=COLORS[cls], linestyle=LINESTYLES[cls], linewidth=1.8,
                label=LABELS[cls])
    ax.axvline(tau_max, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Tardiness $\\tau$ (hours)")
    ax.set_ylabel("Raw cost $\\alpha_c(e^{a+\\beta_c\\rho\\tau}-e^a)$")
    ax.set_title("(c) Raw, $\\lambda{=}1, \\rho{=}0.2083$ (no norm.)")
    ax.set_xlim(0, tau_max)
    ax.set_ylim(bottom=0, top=100)
    ax.grid(True, linewidth=0.4, alpha=0.3)
    ax.legend(frameon=False, loc="upper left", fontsize=7.5)

    # Panel (d): Comparison with original Holguin-Veras water
    ax = axes[1, 1]
    exp_a = math.exp(HOLGUIN_INTERCEPT)
    # Original HV water: g(t) = exp(1.5031 + 0.1172*t) - exp(1.5031)
    hv_water = np.array([math.exp(HOLGUIN_INTERCEPT + 0.1172 * t) - exp_a for t in tau])
    # Old paper rescaling: f(t) = exp(1.5031 + 7.032*t/12) - exp(1.5031)
    old_paper = np.array([math.exp(HOLGUIN_INTERCEPT + 7.032 * t / 12.0) - exp_a for t in tau])
    # New projected water (rho=0.2083, normalized)
    new_water_norm = _series(tau, 'water', cost_lambda=30.0, rho=0.2083, normalized=True)
    # New projected water (rho=1/24, normalized)
    new_water_r124 = _series(tau, 'water', cost_lambda=30.0, rho=1/24, normalized=True)

    ax.plot(tau, new_water_r124, color=COLORS['water'], linestyle='-', linewidth=2.0,
            label="New: $\\rho{=}1/24$, norm.")
    ax.plot(tau, new_water_norm, color=COLORS['water'], linestyle='--', linewidth=2.0,
            label="New: $\\rho{=}0.2083$, norm.")
    ax.plot(tau, hv_water / hv_water.max() * 16.2, color='gray', linestyle='-.', linewidth=1.2,
            label="HV orig. (scaled to max 16.2)")
    ax.set_xlabel("Tardiness $\\tau$ (hours)")
    ax.set_ylabel("Cost")
    ax.set_title("(d) Water class comparison")
    ax.set_xlim(0, tau_max)
    ax.set_ylim(bottom=0)
    ax.grid(True, linewidth=0.4, alpha=0.3)
    ax.legend(frameon=False, loc="upper left", fontsize=7.5)

    png_path = output_dir / "projected_deprivation_cost_4panel.png"
    pdf_path = output_dir / "projected_deprivation_cost_4panel.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

    # ==================================================================
    # Figure 2: Single-panel focused view (recommended config)
    # ==================================================================
    fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)

    for cls in DEFAULT_SUPPLY_CLASS_SEQUENCE:
        spec = WANG_SUPPLY_CLASSES[cls]
        y = _series(tau, cls, cost_lambda=30.0, rho=0.2083, normalized=True)
        ax.plot(tau, y, color=COLORS[cls], linestyle=LINESTYLES[cls], linewidth=2.0,
                label=f"{spec.label} ($\\beta_c$={spec.beta:.4f}, $\\omega_c$={spec.omega:.2f})")

    ax.axvline(tau_max, color='gray', linestyle=':', linewidth=0.8, alpha=0.6,
               label=f"$H_\\tau = {tau_max:.2f}$ h")
    ax.set_xlabel("Tardiness $\\tau_i$ beyond soft deadline (hours)")
    ax.set_ylabel("Delay cost $f_{c_i}(\\tau_i)$")
    ax.set_title("Class-specific projected deprivation cost\n$\\lambda=30,\\rho=0.2083$, normalized")
    ax.set_xlim(0, tau_max + 0.2)
    ax.set_ylim(bottom=0)
    ax.grid(True, linewidth=0.4, alpha=0.3)
    ax.legend(frameon=False, loc="upper left", fontsize=8)

    png_path = output_dir / "projected_cost_main.png"
    pdf_path = output_dir / "projected_cost_main.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

    # ==================================================================
    # Print cost table
    # ==================================================================
    print("\n=== Cost table: lambda=30, rho=0.2083, normalized=True ===")
    print(f"{'tau(h)':>8} {'Medicine':>12} {'Water':>12} {'Food':>12} {'Tent':>12} {'Med/Tent':>10}")
    for t in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 4.4947]:
        vals = {c: deprivation_cost(t, c, cost_lambda=30.0, rho=0.2083, normalized=True)
                for c in DEFAULT_SUPPLY_CLASS_SEQUENCE}
        ratio = vals['medicine'] / vals['tent'] if vals['tent'] > 0 else float('inf')
        print(f"{t:8.4f} {vals['medicine']:12.4f} {vals['water']:12.4f} "
              f"{vals['food']:12.4f} {vals['tent']:12.4f} {ratio:10.2f}")

    print("\n=== Cost table: lambda=30, rho=1/24, normalized=True ===")
    print(f"{'tau(h)':>8} {'Medicine':>12} {'Water':>12} {'Food':>12} {'Tent':>12}")
    for t in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 4.4947]:
        vals = {c: deprivation_cost(t, c, cost_lambda=30.0, rho=1/24, normalized=True)
                for c in DEFAULT_SUPPLY_CLASS_SEQUENCE}
        print(f"{t:8.4f} {vals['medicine']:12.4f} {vals['water']:12.4f} "
              f"{vals['food']:12.4f} {vals['tent']:12.4f}")


if __name__ == "__main__":
    main()