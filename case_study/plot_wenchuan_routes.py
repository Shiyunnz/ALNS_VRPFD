"""Visualize Wenchuan earthquake case study routes on geographic map.

Creates a 2×2 subplot figure comparing 4 scenarios:
  - Same-truck vs Flexible recovery
  - Deterministic vs Robust

Uses terrain-like background shading, approximate river paths, and
geographic styling to mimic a real map.

Usage:
    python case_study/plot_wenchuan_routes.py [results_json_path]
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ── GPS data ───────────────────────────────────────────────────────────
GPS_COORDS = {
    0: (104.07, 30.67),   # 成都(仓库)
    1: (103.62, 31.00),   # 都江堰
    2: (103.94, 31.12),   # 彭州
    3: (104.17, 31.13),   # 什邡
    4: (104.22, 31.34),   # 绵竹
    5: (104.57, 31.53),   # 安县
    6: (104.75, 31.78),   # 江油
    7: (104.46, 31.83),   # 北川
    8: (104.73, 31.47),   # 绵阳
    9: (104.40, 31.13),   # 德阳
    10: (105.24, 32.59),  # 青川
    11: (104.53, 32.41),  # 平武
    12: (103.58, 31.47),  # 汶川(震中)
    13: (103.48, 31.06),  # 映秀
    14: (103.85, 31.68),  # 茂县
    15: (103.52, 31.17),  # 漩口
    16: (104.07, 30.67),  # 成都(仓库)
}

NODE_LABELS = {
    0: "成都(仓库)", 1: "都江堰", 2: "彭州", 3: "什邡", 4: "绵竹",
    5: "安县", 6: "江油", 7: "北川", 8: "绵阳", 9: "德阳",
    10: "青川", 11: "平武", 12: "汶川(震中)", 13: "映秀",
    14: "茂县", 15: "漩口",
}

DEMANDS = {
    1: 5, 2: 5, 3: 7, 4: 6, 5: 4, 6: 6, 7: 7,
    8: 27, 9: 28, 10: 9, 11: 14,
    12: 35, 13: 17, 14: 6, 15: 6,
}

DRONE_ONLY = {12, 13, 14, 15}

# Truck route colors – slightly more saturated for visibility on terrain
TRUCK_COLORS = ["#1565C0", "#2E7D32", "#E65100", "#7B1FA2"]
# Drone task color
DRONE_COLOR = "#C62828"
DRONE_ALPHA = 0.85

# ── Terrain colormap (green lowlands → brown highlands) ──
_TERRAIN_COLORS = [
    (0.85, 0.93, 0.82),  # light green (plains ~500 m)
    (0.78, 0.88, 0.72),  # green
    (0.82, 0.85, 0.68),  # yellow-green (foothills)
    (0.88, 0.82, 0.65),  # tan
    (0.82, 0.74, 0.58),  # brown (mountains)
    (0.75, 0.67, 0.55),  # dark brown (high mountains)
    (0.92, 0.90, 0.88),  # light grey (snow line)
]
TERRAIN_CMAP = LinearSegmentedColormap.from_list("sichuan_terrain", _TERRAIN_COLORS, N=256)

# ── Approximate river paths (lon, lat waypoints) ──
RIVER_MINJIANG = [
    (103.35, 32.70), (103.45, 32.30), (103.50, 31.90),
    (103.55, 31.50), (103.50, 31.20), (103.55, 31.05),
    (103.62, 30.98), (103.75, 30.82), (104.07, 30.67),
]
RIVER_FUJIANG = [
    (104.20, 32.80), (104.30, 32.50), (104.45, 32.20),
    (104.50, 31.85), (104.60, 31.60), (104.68, 31.50),
    (104.73, 31.30), (104.75, 31.00), (104.70, 30.75),
]
RIVER_JIALING = [
    (105.40, 32.80), (105.20, 32.55), (105.00, 32.30),
    (104.85, 32.05), (104.78, 31.80), (104.75, 31.50),
]

# ── Label offset customisation (pixels) ──
# (dx, dy, ha, va)
LABEL_OFFSETS = {
    0:  (12, -6, "left", "top"),
    1:  (-10, 8, "right", "bottom"),
    2:  (0, 10, "center", "bottom"),
    3:  (0, -10, "center", "top"),
    4:  (10, 4, "left", "center"),
    5:  (10, 4, "left", "center"),
    6:  (10, 0, "left", "center"),
    7:  (-10, 4, "right", "center"),
    8:  (10, -4, "left", "top"),
    9:  (0, -10, "center", "top"),
    10: (10, 0, "left", "center"),
    11: (-10, 0, "right", "center"),
    12: (-10, 4, "right", "center"),
    13: (-10, -4, "right", "top"),
    14: (10, 4, "left", "center"),
    15: (-10, 4, "right", "center"),
}


def _get_lonlat(node_id: int):
    """Return (longitude, latitude) for a node, mapping 16→0 for depot end."""
    nid = 0 if node_id == 16 else node_id
    return GPS_COORDS[nid]


def _make_terrain(ax, xlim, ylim, seed=42):
    """Draw a synthetic terrain background that mimics western Sichuan topography."""
    rng = np.random.RandomState(seed)
    nx, ny = 300, 300
    xs = np.linspace(xlim[0], xlim[1], nx)
    ys = np.linspace(ylim[0], ylim[1], ny)
    X, Y = np.meshgrid(xs, ys)

    # Base elevation: higher to the west/northwest (mountains), lower east (plains)
    elev = (xlim[1] - X) * 1.2 + (Y - ylim[0]) * 0.4

    # Add ridges along Longmenshan fault direction (SW → NE)
    ridge_x = np.linspace(103.3, 105.3, 100)
    ridge_y = np.linspace(30.9, 32.7, 100)
    for rx, ry in zip(ridge_x, ridge_y):
        dist = np.sqrt((X - rx) ** 2 + (Y - ry) ** 2)
        elev += 0.6 * np.exp(-dist ** 2 / 0.03)

    # Random bumps for realism
    for _ in range(40):
        cx, cy = rng.uniform(xlim[0], xlim[1]), rng.uniform(ylim[0], ylim[1])
        amp = rng.uniform(0.2, 0.8)
        sig = rng.uniform(0.05, 0.2)
        elev += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sig ** 2))

    # Smooth with uniform kernel (no scipy dependency)
    kernel_size = 17
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    # Pad and convolve manually using numpy
    pad = kernel_size // 2
    padded = np.pad(elev, pad, mode="edge")
    from numpy.lib.stride_tricks import as_strided
    shape = elev.shape + (kernel_size, kernel_size)
    strides = padded.strides * 2
    windows = as_strided(padded, shape=shape, strides=strides)
    elev = np.einsum("ijkl,kl->ij", windows, kernel)

    # Normalize
    elev = (elev - elev.min()) / (elev.max() - elev.min())

    # Draw terrain
    ax.imshow(elev, extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
              origin="lower", cmap=TERRAIN_CMAP, alpha=0.55, aspect="auto",
              zorder=0, interpolation="bilinear")

    # Hillshade overlay for 3D relief effect
    dx_elev = np.gradient(elev, axis=1)
    dy_elev = np.gradient(elev, axis=0)
    slope = np.sqrt(dx_elev ** 2 + dy_elev ** 2)
    aspect_angle = np.arctan2(-dy_elev, dx_elev)
    # Sun from upper-left (azimuth 315°, altitude 45°)
    az = np.radians(315)
    alt = np.radians(45)
    shade = np.sin(alt) * np.cos(np.arctan(slope)) + \
            np.cos(alt) * np.sin(np.arctan(slope)) * np.cos(az - aspect_angle)
    shade = np.clip(shade, 0, 1)
    ax.imshow(shade, extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
              origin="lower", cmap="gray", alpha=0.18, aspect="auto",
              zorder=0, interpolation="bilinear")


def _draw_rivers(ax):
    """Draw approximate river paths."""
    river_style = dict(color="#5B9BD5", linewidth=1.2, alpha=0.5, zorder=1,
                       solid_capstyle="round")
    for river in [RIVER_MINJIANG, RIVER_FUJIANG, RIVER_JIALING]:
        lons, lats = zip(*river)
        ax.plot(lons, lats, **river_style)


def plot_scenario(ax, scenario: Dict[str, Any], panel_label: str = ""):
    """Plot a single scenario on the given axes."""
    cost = scenario["cost"]
    truck_routes = scenario["truck_routes"]
    drone_tasks = scenario["drone_tasks"]

    xlim = (103.15, 105.55)
    ylim = (30.45, 32.85)

    # ── Terrain background ──
    _make_terrain(ax, xlim, ylim)
    _draw_rivers(ax)

    # ── Fault line ──
    fault_lons = [103.4, 103.6, 103.9, 104.2, 104.5, 104.8, 105.3]
    fault_lats = [30.95, 31.10, 31.35, 31.55, 31.80, 32.10, 32.60]
    ax.plot(fault_lons, fault_lats, color="#E57373", linewidth=6, alpha=0.25,
            zorder=1, solid_capstyle="round")
    ax.plot(fault_lons, fault_lats, color="#D32F2F", linewidth=1.5, alpha=0.55,
            zorder=1, linestyle="--")

    # ── Truck routes ──
    for i, route in enumerate(truck_routes):
        if len(route) <= 2:
            continue
        color = TRUCK_COLORS[i % len(TRUCK_COLORS)]
        lons = [_get_lonlat(n)[0] for n in route]
        lats = [_get_lonlat(n)[1] for n in route]
        # Shadow for depth
        ax.plot(lons, lats, color="black", linewidth=3.5, alpha=0.10,
                zorder=2, solid_capstyle="round")
        ax.plot(lons, lats, color=color, linewidth=2.5, alpha=0.9,
                zorder=3, solid_capstyle="round")
        # Direction arrows
        for j in range(len(route) - 1):
            x0, y0 = _get_lonlat(route[j])
            x1, y1 = _get_lonlat(route[j + 1])
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            dx, dy = x1 - x0, y1 - y0
            if abs(dx) + abs(dy) > 0.01:
                ax.annotate("", xy=(mx + dx * 0.015, my + dy * 0.015),
                            xytext=(mx - dx * 0.015, my - dy * 0.015),
                            arrowprops=dict(arrowstyle="-|>", color=color,
                                            lw=1.8, mutation_scale=12),
                            zorder=4)

    # ── Drone tasks ──
    for dt in drone_tasks:
        launch_node = dt["launch_node"]
        retrieve_node = dt["retrieve_node"]
        customers = dt["customers"]
        path_nodes = [launch_node] + customers + [retrieve_node]
        lons = [_get_lonlat(n)[0] for n in path_nodes]
        lats = [_get_lonlat(n)[1] for n in path_nodes]
        ax.plot(lons, lats, color=DRONE_COLOR, linewidth=1.8, alpha=DRONE_ALPHA,
                linestyle=(0, (4, 3)), zorder=5)
        if len(path_nodes) >= 2:
            x0, y0 = _get_lonlat(path_nodes[0])
            x1, y1 = _get_lonlat(path_nodes[1])
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            dx, dy = x1 - x0, y1 - y0
            if abs(dx) + abs(dy) > 0.01:
                ax.annotate("", xy=(mx + dx * 0.015, my + dy * 0.015),
                            xytext=(mx - dx * 0.015, my - dy * 0.015),
                            arrowprops=dict(arrowstyle="-|>", color=DRONE_COLOR,
                                            lw=1.4, mutation_scale=10),
                            zorder=6)

    # ── Node markers ──
    # Depot
    lon0, lat0 = GPS_COORDS[0]
    ax.scatter([lon0], [lat0], marker="s", s=220, c="#D32F2F",
               edgecolors="white", linewidths=2.0, zorder=10)
    dx0, dy0, ha0, va0 = LABEL_OFFSETS[0]
    ax.annotate(NODE_LABELS[0], (lon0, lat0), fontsize=10,
                ha=ha0, va=va0, xytext=(dx0, dy0),
                textcoords="offset points", fontweight="bold",
                fontfamily="sans-serif", zorder=11,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          alpha=0.7, edgecolor="none"))

    # Customer nodes
    for nid in range(1, 16):
        lon, lat = GPS_COORDS[nid]
        demand = DEMANDS.get(nid, 0)
        size = 60 + demand * 4

        if nid in DRONE_ONLY:
            ax.scatter([lon], [lat], marker="^", s=size, c="#FF6F00",
                       edgecolors="white", linewidths=1.2, zorder=10)
        else:
            ax.scatter([lon], [lat], marker="o", s=size, c="#1565C0",
                       edgecolors="white", linewidths=1.2, zorder=10)

        dx_l, dy_l, ha_l, va_l = LABEL_OFFSETS.get(nid, (0, 8, "center", "bottom"))
        ax.annotate(NODE_LABELS.get(nid, str(nid)), (lon, lat),
                    fontsize=9, ha=ha_l, va=va_l,
                    xytext=(dx_l, dy_l), textcoords="offset points",
                    fontfamily="sans-serif", zorder=11,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                              alpha=0.65, edgecolor="none"))

    # ── Info box (no title) ──
    n_trucks = len([r for r in truck_routes if len(r) > 2])
    n_drones = len(drone_tasks)
    robust_cost = scenario.get("robust_cost")

    info = f"成本={cost:.1f}  卡车={n_trucks}  无人机={n_drones}"
    if robust_cost is not None:
        info += f"  鲁棒成本={robust_cost:.1f}"

    ax.text(0.03, 0.03, info, transform=ax.transAxes, fontsize=10,
            va="bottom", ha="left", fontfamily="sans-serif",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      alpha=0.85, edgecolor="#cccccc", linewidth=0.5))

    # ── Panel label (a), (b), etc. ──
    if panel_label:
        ax.text(0.02, 0.97, panel_label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="left",
                fontfamily="sans-serif",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.8, edgecolor="none"))

    # ── Axis styling ──
    ax.set_xlabel("经度 (°E)", fontsize=11, fontfamily="sans-serif")
    ax.set_ylabel("纬度 (°N)", fontsize=11, fontfamily="sans-serif")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    # Degree-based grid resembling geographic maps
    ax.set_xticks(np.arange(103.5, 105.6, 0.5))
    ax.set_yticks(np.arange(30.5, 33.0, 0.5))
    ax.tick_params(labelsize=9, direction="in", length=4)
    ax.grid(True, alpha=0.25, linewidth=0.4, color="#666666", linestyle=":")
    # Subtle border
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#444444")


def create_figure(results: List[Dict[str, Any]], output_dir: str):
    """Create the 2×2 comparison figure."""
    # Configure matplotlib for Chinese fonts
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "STHeiti", "SimHei",
        "Microsoft YaHei", "Arial Unicode MS", "sans-serif",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(16, 18))

    # Map label → position: row0=deterministic, row1=robust, col0=same-truck, col1=flexible
    pos_map = {"A": (0, 0), "B": (0, 1), "C": (1, 0), "D": (1, 1)}
    panel_labels = {"A": "(a)", "B": "(b)", "C": "(c)", "D": "(d)"}

    for sc in results:
        r, c = pos_map[sc["label"]]
        plot_scenario(axes[r, c], sc, panel_label=panel_labels[sc["label"]])

    # ── Shared legend ──
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#D32F2F",
               markersize=11, label="仓库 (成都)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1565C0",
               markersize=9, label="卡车可达节点"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#FF6F00",
               markersize=9, label="仅无人机可达节点 (道路损毁)"),
        Line2D([0], [0], color="#1565C0", linewidth=2.5, label="卡车路线"),
        Line2D([0], [0], color=DRONE_COLOR, linewidth=1.8, linestyle="--",
               label="无人机任务"),
        Line2D([0], [0], color="#D32F2F", linewidth=1.5, linestyle="--",
               alpha=0.55, label="龙门山断裂带 (示意)"),
    ]

    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=12, frameon=True, fancybox=True,
               bbox_to_anchor=(0.5, -0.005),
               edgecolor="#cccccc", facecolor="white")

    # Column headers
    fig.text(0.30, 0.965, "同车回收 (Same-truck)", ha="center", fontsize=15,
             fontweight="bold", fontfamily="sans-serif")
    fig.text(0.73, 0.965, "灵活回收 (Flexible)", ha="center", fontsize=15,
             fontweight="bold", fontfamily="sans-serif")
    # Row headers
    fig.text(0.012, 0.72, "确\n定\n性", ha="center", fontsize=14,
             fontweight="bold", fontfamily="sans-serif", va="center")
    fig.text(0.012, 0.30, "鲁\n棒\n性", ha="center", fontsize=14,
             fontweight="bold", fontfamily="sans-serif", va="center")

    # No suptitle — removed per request

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)

    # Save high-res
    png_path = os.path.join(output_dir, "wenchuan_routes_comparison.png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Figure saved: {png_path}")

    pdf_path = os.path.join(output_dir, "wenchuan_routes_comparison.pdf")
    fig.savefig(pdf_path, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Figure saved: {pdf_path}")

    plt.close(fig)
    return png_path


def create_summary_bar_chart(results: List[Dict[str, Any]], output_dir: str):
    """Create a bar chart comparing costs across scenarios with seed distribution."""
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "STHeiti", "SimHei",
        "Microsoft YaHei", "Arial Unicode MS", "sans-serif",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    labels = [r["label"] for r in results]
    labels_cn = [r["label_cn"] for r in results]
    costs = [r["cost"] for r in results]
    n_trucks = [r["num_trucks_used"] for r in results]
    n_drones = [r["num_drone_tasks"] for r in results]

    x = np.arange(len(labels))
    colors = ["#1976D2", "#43A047", "#F57C00", "#7B1FA2"]

    # Cost bar with seed distribution
    bars = ax1.bar(x, costs, 0.6, color=colors, alpha=0.85, edgecolor="white")
    for i, (bar, c, r) in enumerate(zip(bars, costs, results)):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{c:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        # Show all seed costs as scatter points
        all_costs = r.get("all_costs", [c])
        if len(all_costs) > 1:
            jitter = np.random.RandomState(42).uniform(-0.12, 0.12, len(all_costs))
            ax1.scatter(x[i] + jitter, all_costs, c="gray", s=25, alpha=0.6,
                        zorder=5, edgecolors="white", linewidths=0.5)
            avg = sum(all_costs) / len(all_costs)
            ax1.hlines(avg, x[i] - 0.25, x[i] + 0.25, colors="red",
                       linewidths=1.5, linestyles="--", zorder=6)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{l}\n{cn}" for l, cn in zip(labels, labels_cn)], fontsize=8)
    ax1.set_ylabel("总配送成本", fontsize=10)
    ax1.set_title("成本对比 (柱=最优, 点=各seed, 虚线=均值)", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # Truck & drone usage
    width = 0.3
    bars1 = ax2.bar(x - width / 2, n_trucks, width, color="#1565C0", alpha=0.85, label="卡车数")
    bars2 = ax2.bar(x + width / 2, n_drones, width, color=DRONE_COLOR, alpha=0.85, label="无人机任务数")
    for b in bars1:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1,
                 str(int(b.get_height())), ha="center", va="bottom", fontsize=9)
    for b in bars2:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1,
                 str(int(b.get_height())), ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{l}\n{cn}" for l, cn in zip(labels, labels_cn)], fontsize=8)
    ax2.set_ylabel("数量", fontsize=10)
    ax2.set_title("车辆使用对比", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("汶川地震案例 — 场景对比统计 (Best of N seeds)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "wenchuan_cost_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Bar chart saved: {path}")
    plt.close(fig)
    return path


def main():
    root = Path(__file__).resolve().parent.parent
    output_dir = str(root / "results" / "wenchuan_case")

    # Load results
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        json_path = os.path.join(output_dir, "wenchuan_case_results.json")

    if not os.path.exists(json_path):
        print(f"Results file not found: {json_path}")
        print("Run `python case_study/run_wenchuan_case.py` first.")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"Loaded {len(results)} scenario results.")

    create_figure(results, output_dir)
    create_summary_bar_chart(results, output_dir)

    print("\nDone! All figures saved to:", output_dir)


if __name__ == "__main__":
    main()
