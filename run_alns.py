from alns_vrpfd.core.operators import (
    DestroyRandom,
    DestroyRouteRemoval,
    DestroyShaw,
    DestroyWorstDistance,
    RepairBiasedRandomized,
    RepairDronePriorityRegret,
    RepairEqualPriority,
    RepairTruckFirst,
    RepairCheapest,
    RepairRegret,
)
from alns_vrpfd.core.sa import SANNCfg, SimulatedAnnealingALNS
from alns_vrpfd.evaluation import Evaluator, SubrouteRobustVerifier
from alns_vrpfd.instance.manager import InstanceManager
from alns_vrpfd.model import DroneTask, Solution
from alns_vrpfd.model.initializer import build_initial_solution, build_two_phase_initial_solution
from alns_vrpfd.utils.io_utils import read_instance
from alns_vrpfd.utils.config_loader import ALNSConfig
import sys
import os
import time
import random
import argparse
from copy import deepcopy
from pathlib import Path
from typing import Optional

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def build_operators(
    instance,
    seed,
    drone_priority,
    repair_set,
    enable_composite,
    master_rng=None,
    drone_bonus_kwargs=None,
    forced_drone_customers=None,
    operator_profile="lite",
    robust_energy_mode="embedded",
    repair_weights=None,
):
    rng = master_rng or random.Random(seed)

    # 使用传入的 drone_bonus_kwargs 或默认值
    if drone_bonus_kwargs is None:
        drone_bonus_kwargs = {
            "depot_bonus": 1.5,
            "multi_customer_bonus": 5.0,
            "multi_customer_threshold": 2,
            "wait_max": 20.0,
        }
    repair_kwargs = dict(drone_bonus_kwargs)

    # 添加 forced_drone_customers 到 kwargs
    if forced_drone_customers:
        repair_kwargs["forced_drone_customers"] = forced_drone_customers
    if repair_weights is not None:
        repair_kwargs["weights"] = tuple(repair_weights)

    def _rng_for(offset: int):
        if master_rng is not None:
            # derive a per-operator seed from the master RNG to avoid sharing the same RNG instance
            return random.Random(master_rng.randint(0, 2 ** 32 - 1))
        return random.Random(seed + offset)

    # Keep a single operator set for controlled experiments.
    # operator_profile is accepted for compatibility but does not change operators.
    profile = "lite"

    # 使用固定的基础销毁算子集合
    destroy_ops = [
        DestroyRandom(instance, rng=_rng_for(1000),
                      anchor_strategy="rebase_to_neighbor"),
        DestroyWorstDistance(instance, rng=_rng_for(
            1004), anchor_strategy="rebase_to_neighbor"),
        DestroyShaw(instance, rng=_rng_for(1002),
                    anchor_strategy="rebase_to_neighbor"),
        DestroyRouteRemoval(instance, rng=_rng_for(1008),
                            anchor_strategy="drop_tasks"),
    ]
    if repair_set == "legacy":
        repair_ops = [
            RepairCheapest(instance, rng=_rng_for(2004),
                           drone_priority=drone_priority, robust_energy_mode=robust_energy_mode, **repair_kwargs),
            RepairRegret(instance, rng=_rng_for(2005), k=3,
                         drone_priority=drone_priority, robust_energy_mode=robust_energy_mode, **repair_kwargs),
            RepairBiasedRandomized(
                instance,
                beta=1.5,
                rng=_rng_for(2000),
                drone_priority=drone_priority,
                robust_energy_mode=robust_energy_mode,
                **repair_kwargs,
            ),
        ]
    elif repair_set == "all":
        repair_ops = [
            RepairCheapest(instance, rng=_rng_for(2004),
                           drone_priority=drone_priority, robust_energy_mode=robust_energy_mode, **repair_kwargs),
            RepairRegret(instance, rng=_rng_for(2005), k=3,
                         drone_priority=drone_priority, robust_energy_mode=robust_energy_mode, **repair_kwargs),
            RepairBiasedRandomized(
                instance,
                beta=1.5,
                rng=_rng_for(2000),
                drone_priority=drone_priority,
                robust_energy_mode=robust_energy_mode,
                **repair_kwargs,
            ),
            RepairEqualPriority(
                instance,
                rng=_rng_for(2001),
                drone_priority=drone_priority,
                robust_energy_mode=robust_energy_mode,
                **repair_kwargs,
            ),
            RepairDronePriorityRegret(
                instance,
                rng=_rng_for(2002),
                drone_priority=drone_priority,
                robust_energy_mode=robust_energy_mode,
                **repair_kwargs,
            ),
            RepairTruckFirst(
                instance,
                rng=_rng_for(2003),
                drone_priority=drone_priority,
                robust_energy_mode=robust_energy_mode,
                **repair_kwargs,
            ),
        ]
    else:  # "new"
        # 测试模式：只保留4个基础修复算子
        repair_ops = [
            # 无人机能耗插入（最便宜插入）
            RepairCheapest(instance, rng=_rng_for(2004),
                           drone_priority=drone_priority, robust_energy_mode=robust_energy_mode, **repair_kwargs),
            # 无人机后悔值插入
            RepairDronePriorityRegret(
                instance,
                rng=_rng_for(2002),
                drone_priority=drone_priority,
                robust_energy_mode=robust_energy_mode,
                **repair_kwargs,
            ),
            # 卡车优先插入
            RepairTruckFirst(
                instance,
                rng=_rng_for(2003),
                drone_priority=drone_priority,
                robust_energy_mode=robust_energy_mode,
                **repair_kwargs,
            ),
            # 等机会插入
            RepairEqualPriority(
                instance,
                rng=_rng_for(2001),
                drone_priority=drone_priority,
                robust_energy_mode=robust_energy_mode,
                **repair_kwargs,
            ),
        ]

    # 以下修复算子暂时禁用，用于测试基础算子效果
    # repair_ops.append(
    #     RepairMergeRoutes(
    #         instance,
    #         rng=_rng_for(2006),
    #         merge_bonus=2.0,
    #         **drone_bonus_kwargs,
    #     )
    # )
    # if enable_composite:
    #     # Append composite operator with modest preference for multi-customer tasks
    #     repair_ops.append(
    #         RepairCompositeDrone(
    #             instance,
    #             rng=_rng_for(2100),
    #             drone_priority=drone_priority + 0.15,
    #             max_task_customers=4,
    #             min_task_customers=2,
    #             neighbor_radius=55.0,
    #             neighbor_pool_limit=12,
    #             min_gain=-1.0,
    #             **drone_bonus_kwargs,
    #         )
    #     )

    # # Add new operators for intermediate sync and sequential tasks
    # repair_ops.extend([
    #     RepairIntermediateSync(
    #         instance,
    #         rng=_rng_for(2200),
    #         drone_priority=drone_priority + 0.2,
    #         intermediate_bonus=0.5,
    #         depot_penalty=0.3,
    #         neighbor_radius=30.0,
    #         position_weight=0.2,
    #     ),
    #     RepairSequentialDroneTasks(
    #         instance,
    #         rng=_rng_for(2300),
    #         drone_priority=drone_priority + 0.15,
    #         sequence_bonus=2.0,
    #         min_customers_per_sequence=2,
    #         max_tasks_per_node=3,
    #     ),
    # ])

    # # Add depot cluster repair for depot-launched drone tasks (key for MILP-like solutions)
    # # 获取 forced_drone_customers
    # forced_drone = drone_bonus_kwargs.get("forced_drone_customers", [])
    # repair_ops.extend([
    #     DepotClusterRepair(
    #         instance,
    #         max_cluster_size=4,  # Allow larger clusters
    #         distance_threshold=25.0,  # Wider clustering radius
    #         truck_insertion_threshold=10.0,  # Very low threshold - always consider depot
    #         rng=_rng_for(2400),
    #         forced_drone_customers=forced_drone,
    #     ),
    #     RepairParallelFlights(
    #         instance,
    #         rng=_rng_for(2500),
    #         drone_priority=drone_priority + 0.5,  # High priority
    #         **drone_bonus_kwargs
    #     ),
    #     RepairAggressiveDrone(
    #         instance,
    #         rng=_rng_for(2600),
    #         drone_priority=drone_priority + 1.0,  # Very High priority
    #         **drone_bonus_kwargs
    #     ),
    #     # Truck-to-drone conversion local search - helps shift customers from truck to drone
    #     LocalSearchTruckToDrone(
    #         instance,
    #         rng=_rng_for(2700),
    #         max_customers_per_task=3,
    #         energy_margin=0.1,
    #         min_savings_ratio=-5.0,  # Accept slight losses to increase drone usage
    #         **drone_bonus_kwargs
    #     ),
    # ])

    return destroy_ops, repair_ops


# Backward-compatible alias used by older scripts.
_build_operators = build_operators


def infer_size(instance) -> str:
    """Infer size bucket from customer count for SANNCfg defaults."""
    num_customers = len(instance.customer_manager.customer_ids())
    if num_customers <= 15:
        return "small"
    if num_customers <= 50:
        return "medium"
    return "large"


# Backward-compatible alias used by older scripts.
_infer_size = infer_size


def main_alns():
    parser = argparse.ArgumentParser(description="Run Single ALNS Experiment")
    parser.add_argument("instance", type=str, help="Path to instance file")
    parser.add_argument("--iterations", type=int,
                        default=None, help="Number of iterations")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: random based on time)")
    parser.add_argument('--same-truck-retrieval', type=str, choices=['true', 'false'], default=None,
                        help='Override same_truck_retrieval config (true/false)')
    parser.add_argument(
        "--robust-strategy",
        type=str,
        choices=["embedded", "precheck_guarded"],
        default="embedded",
        help=(
            "embedded: robust in search; "
            "precheck_guarded: deterministic search + changed-subroute robust pre-check "
            "+ exact robust check only for new-best updates."
        ),
    )
    args = parser.parse_args()

    instance_path = args.instance
    if not os.path.exists(instance_path):
        print(f"Error: Instance {instance_path} not found.")
        return

    config = ALNSConfig("config/alns_config.yaml")

    # Seed 处理：命令行参数 > 配置文件 > 系统时间
    seed = args.seed
    if seed is None:
        seed = config.seed if config.seed is not None else int(time.time())

    # 迭代次数：命令行参数 > 配置文件默认值
    iterations = args.iterations if args.iterations else config.iterations_default

    print(f"Running ALNS on {instance_path}")
    print(f"  Iterations: {iterations}, Seed: {seed}")

    start_time = time.time()

    strategy = args.robust_strategy
    search_gamma = config.energy_uncertainty_budget
    repair_mode = "embedded"
    use_precheck_guarded_strategy = strategy == "precheck_guarded"
    if use_precheck_guarded_strategy:
        # Deterministic ALNS search for diversification; robust checks are selective.
        search_gamma = 0
        repair_mode = "verification"

    instance = read_instance(instance_path, strategy=config.time_window_strategy)
    # Align drone endurance to infinity for standard comparison (MIP assumption)
    if 'drone' in instance.vehicle_specs:
        instance.vehicle_specs['drone'].endurance = float('inf')

    # Determine same_truck_retrieval: CLI override > config
    if args.same_truck_retrieval is not None:
        same_truck = True if args.same_truck_retrieval == 'true' else False
    else:
        same_truck = config.same_truck_retrieval

    instance.configure_robustness(
        drone_battery_capacity=config.drone_battery_capacity,
        energy_uncertainty_budget=search_gamma,
        energy_deviation_rate=config.energy_deviation_rate,

        same_truck_retrieval=same_truck,
    )
    print(
        "Robustness config: "
        f"battery={config.drone_battery_capacity}kWh, "
        f"gamma(search)={search_gamma}, "
        f"gamma(robust)={config.energy_uncertainty_budget}, "
        f"deviation={config.energy_deviation_rate}, "
        f"strategy={strategy}"
    )

    # 强制无人机客户
    forced_drone_customers = config.forced_drone_customers
    if forced_drone_customers:
        print(f"Forced drone customers: {forced_drone_customers}")

    evaluator = Evaluator(
        instance,
        rendezvous_tolerance=config.drone_rendezvous_tolerance,
        forced_drone_customers=forced_drone_customers,
        allow_multiple_launch_per_node=config.relax_allow_multiple_launch_per_node,
        cost_lambda=config.cost_lambda,
        cost_rho=config.cost_rho,
        cost_normalized=config.cost_normalized,
    )
    robust_evaluator = None
    candidate_subroute_verifier = None
    collect_robust_route_pool = False
    if use_precheck_guarded_strategy:
        robust_instance = read_instance(instance_path, strategy=config.time_window_strategy)
        if 'drone' in robust_instance.vehicle_specs:
            robust_instance.vehicle_specs['drone'].endurance = float('inf')
        robust_instance.configure_robustness(
            drone_battery_capacity=config.drone_battery_capacity,
            energy_uncertainty_budget=config.energy_uncertainty_budget,
            energy_deviation_rate=config.energy_deviation_rate,

            same_truck_retrieval=same_truck,
        )
        robust_evaluator = Evaluator(
            robust_instance,
            rendezvous_tolerance=config.drone_rendezvous_tolerance,
            forced_drone_customers=forced_drone_customers,
            allow_multiple_launch_per_node=config.relax_allow_multiple_launch_per_node,
            cost_lambda=config.cost_lambda,
            cost_rho=config.cost_rho,
            cost_normalized=config.cost_normalized,
        )
        candidate_subroute_verifier = SubrouteRobustVerifier(
            instance=instance,
            drone_energy_capacity=robust_instance.robust_config.drone_battery_capacity,
            energy_uncertainty_budget=robust_instance.robust_config.energy_uncertainty_budget,
            energy_deviation_rate=robust_instance.robust_config.energy_deviation_rate,
        )
        # Single-path evaluation: deterministic evaluator + robust feasibility gate.
        collect_robust_route_pool = False

    sa_config_dict = config.build_sa_config_dict()

    # 迭代次数：命令行参数 > 配置文件默认值
    sa_config_dict['iterations'] = iterations

    # 算子日志设置从配置文件读取
    sa_config_dict['log_operator_metrics'] = config.log_operators
    sa_config_dict['operator_log_interval'] = config.operator_log_interval

    # Adjust size based on customer count
    sa_config_dict['size'] = infer_size(instance)

    sa_cfg = SANNCfg(**sa_config_dict)
    rng = random.Random(seed)

    # 从配置文件获取无人机奖励参数
    drone_bonus_kwargs = config.drone_bonus

    destroy_ops, repair_ops = build_operators(
        instance, seed, drone_priority=config.drone_priority,
        repair_set="all", enable_composite=True,
        drone_bonus_kwargs=drone_bonus_kwargs,
        forced_drone_customers=forced_drone_customers,
        robust_energy_mode=repair_mode,
    )
    if use_precheck_guarded_strategy:
        # Hybrid repair pool:
        # primary verification-mode repairs + one embedded-mode
        # drone-priority regret operator for robust-cost quality.
        _, embedded_repairs = build_operators(
            instance,
            seed + 7919,
            drone_priority=config.drone_priority,
            repair_set="all",
            enable_composite=True,
            drone_bonus_kwargs=drone_bonus_kwargs,
            forced_drone_customers=forced_drone_customers,
            robust_energy_mode="embedded",
        )
        embedded_regret = next(
            (
                op
                for op in embedded_repairs
                if isinstance(op, RepairDronePriorityRegret)
            ),
            None,
        )
        if embedded_regret is not None:
            repair_ops = list(repair_ops) + [embedded_regret]

    # 构建初始解时直接排除禁止卡车访问的节点
    # 可选择使用两阶段启发式 (two_phase=True) 或原始算法 (two_phase=False)
    use_two_phase = config.raw.get(
        "initial_solution", {}).get("two_phase", True)

    if use_two_phase:
        print("  Using two-phase heuristic for initial solution...")
        initial_solution = build_two_phase_initial_solution(
            instance,
            truck_forbidden_customers=forced_drone_customers,
            allow_multiple_launch_per_node=config.relax_allow_multiple_launch_per_node,
        )
    else:
        print("  Using original algorithm for initial solution...")
        initial_solution = build_initial_solution(
            instance,
            truck_forbidden_customers=forced_drone_customers,
            allow_multiple_launch_per_node=config.relax_allow_multiple_launch_per_node,
        )

    if forced_drone_customers:
        print(
            f"  Initial solution: truck-forbidden customers {forced_drone_customers} excluded from truck routes")

    alns = SimulatedAnnealingALNS(
        instance=instance,
        destroy_ops=destroy_ops,
        repair_ops=repair_ops,
        evaluator=evaluator,
        cfg=sa_cfg,
        rng=rng,
        robust_verifier=robust_evaluator if use_precheck_guarded_strategy else None,
        robust_check_every=0,
        robust_check_on_new_best=use_precheck_guarded_strategy,
        candidate_subroute_verifier=candidate_subroute_verifier,
        conservative_cost_evaluator=(
            robust_evaluator if collect_robust_route_pool else None
        ),
        collect_robust_route_pool=collect_robust_route_pool,
    )

    best_sol = alns.run(initial_solution)
    runtime = time.time() - start_time
    eval_res = evaluator.evaluate_solution(best_sol)
    robust_eval_res = robust_evaluator.evaluate_solution(
        best_sol) if robust_evaluator is not None else None

    print("\n=== ALNS Result ===")
    if robust_eval_res is not None:
        print(f"Search Cost (deterministic): {eval_res.total_cost:.2f}")
        print(f"Conservative Robust Cost: {robust_eval_res.total_cost:.2f}")
        print(f"Robust Feasible: {robust_eval_res.feasible}")
    else:
        print(f"Cost: {eval_res.total_cost:.2f}")
    print(f"Time: {runtime:.2f}s")
    print(f"Feasible: {eval_res.feasible}")

    print("\nTruck Routes:")
    for tr in best_sol.truck_routes:
        print(f"  Truck {tr.id}: {tr.nodes}")

    print("\nDrone Tasks:")
    for dt in best_sol.drone_tasks:
        launch = f"T{dt.launch_truck}@{dt.launch_node}" if dt.launch_truck is not None else f"Depot@{dt.launch_node}"
        land = f"T{dt.land_truck}@{dt.retrieve_node}" if dt.land_truck is not None else f"Depot@{dt.retrieve_node}"
        print(f"  Drone {dt.drone_id}: {dt.nodes} ({launch} -> {land})")

    # --- Save run record ---
    try:
        details = evaluator.evaluate_with_details(best_sol)
        record_config = {
            "drone_battery_capacity": config.drone_battery_capacity,
            "energy_uncertainty_budget": config.energy_uncertainty_budget,
            "energy_deviation_rate": config.energy_deviation_rate,
            "cost_lambda": config.cost_lambda,
            "cost_rho": config.cost_rho,
            "cost_normalized": config.cost_normalized,
        }
        from alns_vrpfd.evaluation.run_record import save_run_record
        instance_name = Path(instance_path).stem
        rec_path = save_run_record(
            instance=instance,
            algorithm="alns",
            solution=best_sol,
            details=details,
            runtime_seconds=runtime,
            config=record_config,
            seed=seed,
            instance_name=instance_name,
        )
        print(f"\nRun record saved to: {rec_path}")
    except Exception as exc:
        print(f"\nWarning: failed to save run record: {exc}")

    # --- Post-processing: try to consolidate routes ---
    improved_sol = _try_consolidate_routes(best_sol, instance, evaluator)
    if improved_sol is not None:
        best_sol = improved_sol
        eval_res = evaluator.evaluate_solution(best_sol)
        print("\n=== After Route Consolidation ===")
        if robust_eval_res is not None:
            print(f"Search Cost (deterministic): {eval_res.total_cost:.2f}")
        else:
            print(f"Cost: {eval_res.total_cost:.2f}")
        print(f"Feasible: {eval_res.feasible}")
        print("\nTruck Routes:")
        for tr in best_sol.truck_routes:
            print(f"  Truck {tr.id}: {tr.nodes}")
        print("\nDrone Tasks:")
        for dt in best_sol.drone_tasks:
            launch = f"T{dt.launch_truck}@{dt.launch_node}" if dt.launch_truck is not None else f"Depot@{dt.launch_node}"
            land = f"T{dt.land_truck}@{dt.retrieve_node}" if dt.land_truck is not None else f"Depot@{dt.retrieve_node}"
            print(f"  Drone {dt.drone_id}: {dt.nodes} ({launch} -> {land})")


def _try_consolidate_routes(
    solution: Solution,
    instance: InstanceManager,
    evaluator: Evaluator,
) -> Optional[Solution]:
    best_sol = solution
    best_cost = evaluator.evaluate_solution(best_sol).total_cost
    demands = instance.customer_manager.demands()
    drone_spec = instance.vehicle_specs.get('drone', None)
    drone_capacity = drone_spec.capacity if drone_spec else 30.0
    depot_start = instance.customer_manager.depot_start
    depot_end = instance.customer_manager.depot_end or depot_start
    if depot_start is None:
        return None

    changed = True
    while changed:
        changed = False
        for route in sorted(best_sol.truck_routes, key=lambda r: len(r.customers())):
            custs = route.customers()
            if not custs:
                continue
            # Skip if any drone task depends on this truck
            depends = any(
                (dt.launch_truck == route.id or dt.land_truck == route.id)
                for dt in best_sol.drone_tasks
            )
            if depends:
                continue

            cand = deepcopy(best_sol)
            target = next((tr for tr in cand.truck_routes if tr.id == route.id), None)
            if target is None:
                continue
            cand.truck_routes.remove(target)

            new_drone_id = 0
            used_ids = {t.drone_id for t in cand.drone_tasks}
            while new_drone_id in used_ids:
                new_drone_id += 1

            drone_groups = []
            current_group = []
            current_load = 0.0
            for c in custs:
                d = demands.get(c, 0.0)
                if current_load + d <= drone_capacity + 1e-6:
                    current_group.append(c)
                    current_load += d
                else:
                    if current_group:
                        drone_groups.append(current_group)
                    current_group = [c]
                    current_load = d
            if current_group:
                drone_groups.append(current_group)

            for group in drone_groups:
                task = DroneTask(
                    drone_id=new_drone_id,
                    launch_truck=None,
                    launch_node=depot_start,
                    customers=group,
                    land_truck=None,
                    retrieve_node=depot_end,
                )
                cand.add_drone_task(task)

            cand_eval = evaluator.evaluate_solution(cand)
            if cand_eval.feasible and cand_eval.total_cost < best_cost - 1e-6:
                best_sol = cand
                best_cost = cand_eval.total_cost
                changed = True
                print(f"  Consolidation: removed Truck {route.id} (customers {custs}), new cost {best_cost:.3f}")
                break
    return best_sol if best_sol is not solution else None


if __name__ == "__main__":
    main_alns()
