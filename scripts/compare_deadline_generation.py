"""
Compare original demand-based deadline generation vs new class-based generation.

Original (demand-based):
  - q_i normalized to [0,1] as demand_ratio
  - window_width = w_max - demand_ratio * (w_max - w_min)
  - l_i = o_i + window_width + latest_time_slack

New (class-based):
  - Each node randomly assigned a relief-item class c_i in {1,2,3,4}
  - Class determines deadline parameters independently of q_i
  - r_i = min travel time from depot
  - o_i = r_i + Delta^o_{c_i}
  - l_i = o_i + Delta^l_{c_i}
  - Deprivation cost uses the class-specific Wang/Holguin function

Runs on Instance10 (R_30_10_1) for visual comparison.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from alns_vrpfd.utils.data_reader import InstanceDataReader, TimeWindowConfig
from alns_vrpfd.deprivation import DEFAULT_SUPPLY_CLASS_SEQUENCE, WANG_SUPPLY_CLASSES, deprivation_cost

CLASS_NAMES = {key: WANG_SUPPLY_CLASSES[key].label for key in DEFAULT_SUPPLY_CLASS_SEQUENCE}


def deprival_cost_old(tau):
    return deprivation_cost(tau, "water", cost_lambda=30.0, rho=0.2083, normalized=True)


def deprival_cost_new(tau, supply_class):
    return deprivation_cost(tau, supply_class, cost_lambda=30.0, rho=0.2083, normalized=True)


def generate_class_based_deadlines(instance, rng):
    customer_ids = instance.customer_manager.customer_ids()
    demands = instance.customer_manager.demands()
    truck_speed = instance.vehicle_specs["truck"].speed
    drone_speed = instance.vehicle_specs["drone"].speed

    depot_id = instance.customer_manager.depot_start

    dist_truck = instance.distance_matrix("truck")
    dist_drone = instance.distance_matrix("drone")

    node_list = instance.all_node_ids()
    idx_map = {nid: i for i, nid in enumerate(node_list)}
    depot_idx = idx_map[depot_id]

    classes = {}
    optimal_new = {}
    latest_new = {}

    supply_classes = list(DEFAULT_SUPPLY_CLASS_SEQUENCE)
    for offset, cid in enumerate(customer_ids):
        c = supply_classes[offset % len(supply_classes)]
        if offset >= len(supply_classes):
            c = str(rng.choice(supply_classes))
        classes[cid] = c
        params = WANG_SUPPLY_CLASSES[c]

        ci = idx_map[cid]
        d_t = dist_truck[depot_idx][ci]
        d_d = dist_drone[depot_idx][ci]
        t_t = d_t / truck_speed
        t_d = d_d / drone_speed
        r_i = min(t_t, t_d)

        delta_o = rng.uniform(*params.deadline_optimal_delta_hours)
        delta_l = rng.uniform(*params.deadline_latest_delta_hours)
        instance.customer_manager.assign_supply_class(cid, c)

        optimal_new[cid] = r_i + delta_o
        latest_new[cid] = optimal_new[cid] + delta_l

    return classes, optimal_new, latest_new


def main():
    rng = np.random.default_rng(42)

    instance_file = os.path.join(
        os.path.dirname(__file__), "..", "data", "Instance10", "R_30_10_1.txt"
    )
    instance_file = os.path.abspath(instance_file)

    reader_old = InstanceDataReader(
        time_window_strategy="demand_based",
        time_window_config=TimeWindowConfig(),
        apply_time_windows=True,
    )
    instance_old = reader_old.read_instance(instance_file)

    reader_new = InstanceDataReader(
        time_window_strategy="demand_based",
        time_window_config=TimeWindowConfig(),
        apply_time_windows=False,
    )
    instance_new = reader_new.read_instance(instance_file)

    classes, optimal_new, latest_new = generate_class_based_deadlines(instance_new, rng)

    for cid in instance_new.customer_manager.customer_ids():
        instance_new.customer_manager.assign_time_window(
            cid, optimal_new[cid], latest_new[cid]
        )

    customer_ids = instance_old.customer_manager.customer_ids()
    demands = instance_old.customer_manager.demands()

    min_d = min(demands.values())
    max_d = max(demands.values())

    print("=" * 100)
    print("DEADLINE GENERATION COMPARISON (Instance R_30_10_1)")
    print("=" * 100)
    print()
    print(f"{'Node':>4} {'q_i':>6} {'Class':>16} {'Beta':>8} {'Omega':>8}")
    print(f"{'':>4} {'':>6} {'':>16} {'':>8} {'':>8}")
    print("-" * 46)
    for cid in customer_ids:
        c = classes[cid]
        spec = WANG_SUPPLY_CLASSES[c]
        print(f"{cid:>4} {demands[cid]:>6.0f} {CLASS_NAMES[c]:>16} {spec.beta:>8.4f} {spec.omega:>8.2f}")

    print()
    print("=" * 100)
    print("OLD demand-based deadlines vs NEW class-based deadlines")
    print("=" * 100)
    print(
            f"{'Node':>4} {'q_i':>6} {'Class':>16} "
        f"{'o_i_old':>9} {'l_i_old':>9} {'w_old':>7} "
        f"{'o_i_new':>9} {'l_i_new':>9} {'w_new':>7} "
        f"{'r_i':>6}"
    )
    print("-" * 100)

    for cid in customer_ids:
        c = classes[cid]
        o_old, l_old = instance_old.customer_manager.time_window(cid)
        o_new, l_new = instance_new.customer_manager.time_window(cid)

        w_old = l_old - o_old
        w_new = l_new - o_new

        depot_id = instance_old.customer_manager.depot_start
        node_list = instance_old.all_node_ids()
        idx_map = {nid: i for i, nid in enumerate(node_list)}
        depot_idx = idx_map[depot_id]
        ci = idx_map[cid]
        truck_speed = instance_old.vehicle_specs["truck"].speed
        drone_speed = instance_old.vehicle_specs["drone"].speed
        d_t = instance_old.distance_matrix("truck")[depot_idx][ci]
        d_d = instance_old.distance_matrix("drone")[depot_idx][ci]
        r_i = min(d_t / truck_speed, d_d / drone_speed)

        print(
            f"{cid:>4} {demands[cid]:>6.0f} {CLASS_NAMES[c]:>16} "
            f"{o_old:>9.3f} {l_old:>9.3f} {w_old:>7.3f} "
            f"{o_new:>9.3f} {l_new:>9.3f} {w_new:>7.3f} "
            f"{r_i:>6.3f}"
        )

    print()
    print("=" * 100)
    print("DEPRIVATION COST COMPARISON (assuming arrival at o_i + 0.1h tardiness)")
    print("=" * 100)
    tau_example = 0.1
    print(
        f"{'Node':>4} {'Class':>16} {'Beta':>8} "
        f"{'f_old(0.1)':>12} {'f_new(0.1)':>12} {'ratio':>7}"
    )
    print("-" * 65)
    for cid in customer_ids:
        c = classes[cid]
        f_old = deprival_cost_old(tau_example)
        f_new = deprival_cost_new(tau_example, c)
        spec = WANG_SUPPLY_CLASSES[c]
        print(
            f"{cid:>4} {CLASS_NAMES[c]:>16} {spec.beta:>8.4f} "
            f"{f_old:>12.2f} {f_new:>12.2f} {f_new / f_old:>7.2f}x"
        )

    print()
    print("=" * 100)
    print("KEY DIFFERENCES SUMMARY")
    print("=" * 100)

    old_widths = [instance_old.customer_manager.time_window(cid)[1] - instance_old.customer_manager.time_window(cid)[0] for cid in customer_ids]
    new_widths = [instance_new.customer_manager.time_window(cid)[1] - instance_new.customer_manager.time_window(cid)[0] for cid in customer_ids]

    demand_list = [demands[cid] for cid in customer_ids]
    old_corr = np.corrcoef(demand_list, old_widths)[0, 1]
    new_corr = np.corrcoef(demand_list, new_widths)[0, 1]

    print(f"  Old (demand-based):")
    print(f"    Correlation q_i vs window_width: {old_corr:.4f}")
    print(f"    Window width range: [{min(old_widths):.3f}, {max(old_widths):.3f}]")
    print(f"  New (class-based):")
    print(f"    Correlation q_i vs window_width: {new_corr:.4f}")
    print(f"    Window width range: [{min(new_widths):.3f}, {max(new_widths):.3f}]")
    print()
    print("  Class distribution:")
    from collections import Counter
    class_counts = Counter(classes[cid] for cid in customer_ids)
    for c in sorted(class_counts):
        spec = WANG_SUPPLY_CLASSES[c]
        print(f"    {CLASS_NAMES[c]}: {class_counts[c]} nodes (beta={spec.beta:.4f}, omega={spec.omega:.2f})")


if __name__ == "__main__":
    main()
