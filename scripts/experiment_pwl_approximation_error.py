#!/usr/bin/env python3
"""Quantify the current three-segment PWL approximation errors.

The experiment follows the implementation used by the MILP:

* delay cost: equally spaced breakpoints and class-specific normalized
  deprivation costs;
* drone power: equally spaced payload breakpoints over [0, 30] kg.

Two delay domains are reported. ``paper_horizon`` uses the normalization
horizon H_tau=4.4947 h, while ``mip_conservative`` uses the 6 h upper cap in
``alns_vrpfd.mip.builder``.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alns_vrpfd.deprivation import (  # noqa: E402
    DEFAULT_SUPPLY_CLASS_SEQUENCE,
    MAX_TARDINESS_HOURS,
    deprivation_cost,
)
from alns_vrpfd.mip.piecewise_energy import (  # noqa: E402
    PiecewiseLinearEnergyBuilder,
)
from alns_vrpfd.utils.config_loader import ALNSConfig  # noqa: E402


def interpolate(xs: np.ndarray, ys: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, xs, ys)


def error_metrics(
    exact: np.ndarray,
    approx: np.ndarray,
    *,
    endpoint_scale: float,
) -> dict[str, float]:
    error = approx - exact
    abs_error = np.abs(error)
    nonzero = exact > max(1e-12, abs(endpoint_scale) * 1e-9)
    relative = np.abs(error[nonzero] / exact[nonzero]) if np.any(nonzero) else np.array([0.0])
    return {
        "max_abs_error": float(np.max(abs_error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_relative_error_pct": float(np.max(relative) * 100.0),
        "max_endpoint_normalized_error_pct": (
            float(np.max(abs_error) / abs(endpoint_scale) * 100.0)
            if endpoint_scale
            else 0.0
        ),
        "mean_signed_error": float(np.mean(error)),
    }


def delay_rows(config: ALNSConfig, segments: int, grid_size: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    domains = {
        "paper_horizon": MAX_TARDINESS_HOURS,
        "mip_conservative": 6.0,
    }
    for domain_name, upper in domains.items():
        xs = np.linspace(0.0, upper, segments + 1)
        grid = np.linspace(0.0, upper, grid_size)
        for supply_class in DEFAULT_SUPPLY_CLASS_SEQUENCE:
            exact = np.array(
                [
                    deprivation_cost(
                        value,
                        supply_class,
                        cost_lambda=config.cost_lambda,
                        rho=config.cost_rho,
                        normalized=config.cost_normalized,
                    )
                    for value in grid
                ],
                dtype=float,
            )
            ys = np.array(
                [
                    deprivation_cost(
                        value,
                        supply_class,
                        cost_lambda=config.cost_lambda,
                        rho=config.cost_rho,
                        normalized=config.cost_normalized,
                    )
                    for value in xs
                ],
                dtype=float,
            )
            approx = interpolate(xs, ys, grid)
            idx = int(np.argmax(np.abs(approx - exact)))
            metrics = error_metrics(exact, approx, endpoint_scale=float(exact[-1]))
            rows.append(
                {
                    "function": "delay",
                    "domain": domain_name,
                    "supply_class": supply_class,
                    "segments": segments,
                    "domain_upper": upper,
                    "max_error_at": float(grid[idx]),
                    **metrics,
                }
            )
    return rows


def energy_rows(segments: int, grid_size: int) -> list[dict[str, object]]:
    builder = PiecewiseLinearEnergyBuilder(num_segments=segments)
    payload_max = 30.0
    breakpoints, values = builder.compute_breakpoints(payload_max)
    xs = np.asarray(breakpoints, dtype=float)
    ys = np.asarray(values, dtype=float)
    grid = np.linspace(0.0, payload_max, grid_size)
    exact = ((builder.W + builder.m + grid) ** 1.5) * builder.constant
    approx = interpolate(xs, ys, grid)
    idx = int(np.argmax(np.abs(approx - exact)))
    metrics = error_metrics(exact, approx, endpoint_scale=float(exact[-1]))
    return [
        {
            "function": "energy_power",
            "domain": "payload",
            "supply_class": "",
            "segments": segments,
            "domain_upper": payload_max,
            "max_error_at": float(grid[idx]),
            **metrics,
        }
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    config = ALNSConfig(str(PROJECT_ROOT / "config" / "alns_config.yaml"))
    segments = int(config.piecewise_delay_segments)
    energy_segments = int(config.piecewise_energy_segments)
    grid_size = 200_001
    rows = delay_rows(config, segments, grid_size)
    rows.extend(energy_rows(energy_segments, grid_size))

    output_dir = PROJECT_ROOT / "results" / "revision_experiments" / "pwl_error"
    write_csv(output_dir / "pwl_error_metrics.csv", rows)
    summary = {
        "delay_segments": segments,
        "energy_segments": energy_segments,
        "cost_lambda": config.cost_lambda,
        "cost_rho": config.cost_rho,
        "cost_normalized": config.cost_normalized,
        "grid_size": grid_size,
        "rows": rows,
    }
    (output_dir / "pwl_error_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Saved: {output_dir}")
    for row in rows:
        label = f"{row['function']}:{row['domain']}:{row['supply_class'] or 'all'}"
        print(
            f"{label:42s} max_abs={row['max_abs_error']:.8f} "
            f"max_rel={row['max_relative_error_pct']:.4f}% "
            f"endpoint_norm={row['max_endpoint_normalized_error_pct']:.4f}%"
        )


if __name__ == "__main__":
    main()
