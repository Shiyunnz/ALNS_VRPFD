"""Generate Wenchuan earthquake case study instance file.

Data sources:
- Liu et al. (2018a) Applied Mathematical Modelling - 15 affected areas, demands
- Liu et al. (2019) Computers & Industrial Engineering - damage assessment
- Yin et al. (2023) Transportation Research Part B - truck-drone collaborative mode
- 中国地震局, 汶川地震烈度分布图 (2008)

Geographic reference: Chengdu depot, 15 demand nodes along Longmenshan fault.
Drone-only node determination: seismic intensity (烈度) ≥ X度 + road damage reports.

Distance units: real kilometres (Euclidean projection from GPS).
Demand units: kg of relief cargo per drone sortie (scaled from Liu 2018a Table 2).
Energy model calibration: see case_study/run_wenchuan_case.py for drone parameters.
"""

import math
import os

# ── GPS coordinates (longitude, latitude) ──────────────────────────────
# Seismic intensity (烈度) data from China Earthquake Administration
# Official isoseismal map: https://www.cea.gov.cn
#
# Determination of drone-only nodes:
#   1. 烈度 ≥ X度: 极重灾区, 道路桥梁大面积损毁, 地面交通几乎完全中断
#   2. 结合实际路况报告:
#      - 都汶公路(G213)全线中断 → 汶川、映秀、漩口不可达
#      - 茂北公路严重受损、山体滑坡 → 茂县不可达
#   3. 北川虽烈度X度但有替代路线(经绵阳方向)可达, 故保留为卡车节点
#
# Reference: Liu et al. (2018a) Table 2 的15个需求节点选取也基于与
#            中国地震局专家讨论后确定

GPS_COORDS = {
    # name:      (longitude, latitude)
    "depot":      (104.07, 30.67),  # 成都 Chengdu — 仓库/配送中心
    # ── 卡车可达节点 (Truck-accessible, 烈度 VI-IX度, 有替代路线) ──
    "dujiangyan": (103.62, 31.00),  # A 都江堰  烈度IX  (经成灌高速可达)
    "pengzhou":   (103.94, 31.12),  # B 彭州    烈度VIII
    "shifang":    (104.17, 31.13),  # C 什邡    烈度VIII (高需求)
    "mianzhu":    (104.22, 31.34),  # D 绵竹    烈度IX
    "anxian":     (104.57, 31.53),  # E 安县    烈度VIII
    "jiangyou":   (104.75, 31.78),  # F 江油    烈度VII
    "beichuan":   (104.46, 31.83),  # G 北川    烈度X   (高需求, 经绵阳可达)
    "mianyang":   (104.73, 31.47),  # H 绵阳    烈度VII  (高需求)
    "deyang":     (104.40, 31.13),  # I 德阳    烈度VII  (高需求)
    "qingchuan":  (105.24, 32.59),  # J 青川    烈度VIII (经广元方向可达)
    "pingwu":     (104.53, 32.41),  # K 平武    烈度VIII
    # ── 仅无人机可达节点 (Drone-only, 烈度 ≥ X度 且 地面交通完全中断) ──
    "wenchuan":   (103.58, 31.47),  # L 汶川  烈度XI (震中, G213中断)
    "yingxiu":    (103.48, 31.06),  # M 映秀  烈度XI (百花大桥坍塌, 紧邻震中)
    "maoxian":    (103.85, 31.68),  # N 茂县  烈度IX (茂北公路山体滑坡阻断)
    "xuankou":    (103.52, 31.17),  # O 漩口  烈度X  (都汶公路中断段)
}

# Seismic intensity for each node (used for drone-only determination & visualization)
SEISMIC_INTENSITY = {
    "depot": "V", "dujiangyan": "IX", "pengzhou": "VIII", "shifang": "VIII",
    "mianzhu": "IX", "anxian": "VIII", "jiangyou": "VII", "beichuan": "X",
    "mianyang": "VII", "deyang": "VII", "qingchuan": "VIII", "pingwu": "VIII",
    "wenchuan": "XI", "yingxiu": "XI", "maoxian": "IX", "xuankou": "X",
}

# Node ordering: 0=depot, 1-11=truck nodes, 12-15=drone-only, 16=depot_end
NODE_ORDER = [
    "depot",      # 0
    "dujiangyan", # 1
    "pengzhou",   # 2
    "shifang",    # 3
    "mianzhu",    # 4
    "anxian",     # 5
    "jiangyou",   # 6
    "beichuan",   # 7
    "mianyang",   # 8
    "deyang",     # 9
    "qingchuan",  # 10
    "pingwu",     # 11
    "wenchuan",   # 12  drone-only (烈度XI, 震中)
    "yingxiu",    # 13  drone-only (烈度XI, 桥梁坍塌)
    "maoxian",    # 14  drone-only (烈度IX, 滑坡阻断)
    "xuankou",    # 15  drone-only (烈度X, 道路中断)
    "depot",      # 16  depot end
]

# Chinese labels for visualization
NODE_LABELS_CN = {
    0: "成都(仓库)", 1: "都江堰", 2: "彭州", 3: "什邡", 4: "绵竹",
    5: "安县", 6: "江油", 7: "北川", 8: "绵阳", 9: "德阳",
    10: "青川", 11: "平武", 12: "汶川(震中)", 13: "映秀",
    14: "茂县", 15: "漩口", 16: "成都(仓库)",
}

DRONE_ONLY_NODES = [12, 13, 14, 15]

# Demands: kg of relief cargo per drone sortie.
# Proportions from Liu 2018a Table 2 (nominal relief personnel / 60),
# scaled ×0.5 to represent realistic single-sortie drone payload weights.
DEMANDS = {
    1: 2.5, 2: 2.5, 3: 3.5, 4: 3.0, 5: 2.0, 6: 3.0, 7: 3.5,
    8: 13.5, 9: 14.0, 10: 4.5, 11: 7.0,
    12: 17.5, 13: 8.5, 14: 3.0, 15: 3.0,
}

# ── Projection: GPS → planar km (real distances, no scaling) ──────────
LAT_REF = 30.67
KM_PER_DEG_LON = math.cos(math.radians(LAT_REF)) * 111.32  # ~95.8 km
KM_PER_DEG_LAT = 111.0


def gps_to_xy(lon, lat):
    """Convert GPS to planar km coordinates (offset from Chengdu depot)."""
    depot_lon, depot_lat = GPS_COORDS["depot"]
    x = (lon - depot_lon) * KM_PER_DEG_LON
    y = (lat - depot_lat) * KM_PER_DEG_LAT
    return round(x, 2), round(y, 2)


def euclidean(x1, y1, x2, y2):
    return round(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2), 1)


def generate_instance(output_path: str):
    """Write the Wenchuan earthquake instance file."""
    n_nodes = len(NODE_ORDER)  # 17 (0..16)

    # Compute XY coordinates
    coords = {}
    for idx, name in enumerate(NODE_ORDER):
        lon, lat = GPS_COORDS[name]
        coords[idx] = gps_to_xy(lon, lat)

    # Compute distance matrices
    drone_dists = []
    truck_dists = []
    road_factor = 1.3  # road detour factor for trucks

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            x1, y1 = coords[i]
            x2, y2 = coords[j]
            d_euc = euclidean(x1, y1, x2, y2)
            drone_dists.append((i, j, d_euc))
            drone_dists.append((j, i, d_euc))
            d_truck = round(d_euc * road_factor, 1)
            truck_dists.append((i, j, d_truck))
            truck_dists.append((j, i, d_truck))

    # Add depot self-loop (0 → 16)
    drone_dists.append((0, n_nodes - 1, 0))
    truck_dists.append((0, n_nodes - 1, 0))

    lines = []

    # Vehicle information
    # Vehicle specs calibrated for Wenchuan disaster relief scenario:
    # Truck: 40 km/h on damaged mountain roads, 35 kg cargo per trip
    #   (total demand ≈ 91 kg → need all 3 trucks, avoiding degenerate single-truck case)
    # Drone: 60 km/h medium-lift delivery UAV, 25 kg max payload, 2h endurance
    # Energy constraint (battery=19 kWh) handled by DroneEnergyModel in runner
    lines.append("VEHICLE INFORMATION")
    lines.append("Type\tNumber\tCapacity\tEndurance\t Speed\t Unit cost")
    lines.append("Truck\t    5\t       35\t    24.00\t   40\t   1.0")
    lines.append("Drone\t    3\t       25\t     2.00\t   60\t   1.0")
    lines.append("")

    # Customer information
    lines.append("CUSTOMER INFORMATION")
    lines.append("Id \tX\tY\tDemand_D\tDemand_P")
    for idx in range(n_nodes):
        x, y = coords[idx]
        d = DEMANDS.get(idx, 0)
        lines.append(f"{idx}  \t{x:.2f}\t{y:.2f}\t{d:8.2f}\t{0:8.2f}")
    lines.append("")

    # Drone distances
    lines.append("Distance For Drone")
    for i, j, d in sorted(drone_dists):
        lines.append(f"    {i}\t   {j}\t    {d}")
    lines.append("")

    # Truck distances
    lines.append("Distance For Truck")
    for i, j, d in sorted(truck_dists):
        lines.append(f"    {i}\t   {j}\t    {d}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Instance written to {output_path}")
    print(f"  Nodes: {n_nodes} (depot + {n_nodes - 2} customers + depot_end)")
    print(f"  Drone-only nodes: {DRONE_ONLY_NODES}")
    print(f"  Total demand: {sum(DEMANDS.values())}")

    # Print drone-only determination logic
    print("\n  Drone-only node determination (烈度 + 道路损毁报告):")
    for idx in DRONE_ONLY_NODES:
        name = NODE_ORDER[idx]
        intensity = SEISMIC_INTENSITY[name]
        print(f"    Node {idx} ({NODE_LABELS_CN[idx]}): 烈度{intensity}")

    return coords


if __name__ == "__main__":
    out = os.path.join(
        os.path.dirname(__file__), "..", "data", "WenchuanCase", "wenchuan_15.txt"
    )
    generate_instance(os.path.abspath(out))
