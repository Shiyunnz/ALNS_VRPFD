"""Simulated annealing acceptance schedule with adaptive destroy/repair selection."""

from __future__ import annotations

import logging
import math
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

from alns_vrpfd.core.adaptive import AdaptiveOperatorManager
from alns_vrpfd.core.operators import (
    DestroyOperator,
    DestroyRandom,
    DestroySegmentShuffle,
    RepairOperator,
    RepairBiasedRandomized,
)
from alns_vrpfd.evaluation import Evaluator
from alns_vrpfd.evaluation.energy import DroneEnergyModel
from alns_vrpfd.evaluation.subroute_robust_verifier import SubrouteRobustVerifier
from alns_vrpfd.model import Solution

RewardLabel = str


def _finite_or_zero(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if math.isfinite(numeric) else 0.0


@dataclass
class SANNCfg:
    size: str = "small"  # "small" or "large"
    iterations: Optional[int] = None
    w_percent: float = 30.0
    cooling_rate: float = 0.97
    cooling_rate_initial: Optional[float] = None
    cooling_rate_final: Optional[float] = None
    cooling_transition_iters: int = 800
    temperature_min: float = 1e-4
    eta: float = 0.6
    alpha_credit: float = 0.65
    reward_scale: Optional[dict[str, float]] = None
    max_non_improve: Optional[int] = None
    r_lower: float = 0.15  # r_L: 移除比例下限 (15%)
    r_upper_small: float = 0.5  # r_U: 小规模实例移除比例上限 (50%)
    r_upper_large: float = 0.3  # r_U: 大规模实例移除比例上限 (30%)
    reheat_stall_trigger: int = 300
    reheat_acceptance_window: int = 80
    reheat_acceptance_min: float = 0.05
    reheat_duration: int = 40
    reheat_recovery: int = 30
    reheat_cooldown: int = 60
    reheat_quota_multiplier: float = 1.3
    reheat_quota_upper_cap: float = 0.6
    reheat_quota_absolute_cap: Optional[int] = 50
    reheat_p_floor: float = 0.05
    reheat_random_repair_prob: float = 0.5
    reheat_shake_fraction: float = 0.2
    reheat_shake_probability: float = 0.6
    reheat_temperature_scale: float = 1.0
    # 逃脱算法参数 (从配置文件读取)
    escape_enabled: bool = True          # 是否启用逃脱算法
    escape_trigger_stall: int = 100      # 触发逃脱的停滞次数
    escape_duration: int = 20            # 逃脱持续的迭代次数
    quota_base_cap: int = 30  # 移除数量绝对上限 (公式中的30)
    weight_history: int = 30
    weight_decay: float = 0.02
    log_operator_metrics: bool = False
    operator_log_interval: int = 250
    measure_operator_time: bool = True
    # Local search parameters
    local_search_frequency: int = 10  # Apply LS every N iterations
    local_search_on_new_best: bool = True  # Apply LS when new best found
    depot_drone_probability: float = 0.15  # Probability to try depot drone task
    # Enhanced local search parameters
    intensify_frequency: int = 50  # Full intensification every N iterations
    cross_exchange_prob: float = 0.3  # Probability to try cross-exchange
    path_relinking_prob: float = 0.2  # Probability to try path relinking to best
    # Cache robust-feasibility checks to avoid repeated expensive evaluations.
    robust_cache_enabled: bool = True
    robust_cache_size: int = 4096

    # 收敛增强参数 (新增)
    # 动态冷却率
    dynamic_cooling_enabled: bool = True
    improvement_threshold: float = 0.01  # 判定改善的阈值
    cooling_slowdown_factor: float = 0.998  # 改善时减慢冷却
    cooling_speedup_factor: float = 0.980  # 停滞时加速冷却
    recent_improvement_window: int = 50  # 监控改善的窗口

    # 多样化重启
    diversification_enabled: bool = True
    diversification_trigger_stall: int = 500  # 触发多样化的停滞次数
    diversification_restart_best_prob: float = 0.7  # 从最优解重启概率
    diversification_destroy_ratio: float = 0.6  # 多样化销毁比例

    # 自适应配额
    adaptive_quota_enabled: bool = True
    quota_increase_on_stall: int = 100  # 停滞多少次后增加配额
    quota_decrease_on_improve: bool = True  # 改善时减少配额

    # Drone re-anchor local search (Step 6)
    drone_reanchor_ls_enabled: bool = True
    drone_reanchor_ls_max_moves: int = 10

    # Composite drone re-anchor destroy-repair (Step 7)
    drone_composite_reanchor_enabled: bool = False
    drone_composite_reanchor_max_tasks: int = 3

    # Multi-customer sortie constructor (Step 8)
    drone_sortie_constructor_enabled: bool = True
    drone_sortie_max_customers: int = 3
    drone_sortie_top_k: int = 5
    drone_sortie_max_sorties: int = 20

    # Mini-MILP truck route polish (Step 10)
    mini_milp_polish_enabled: bool = False
    mini_milp_polish_time_limit: float = 60.0
    mini_milp_polish_gap: float = 0.005

    # Track best zero-cross-truck solution alongside global best
    track_no_cross_truck: bool = False

    # Matheuristic LNS exhaustive sortie reconstruction (Step 11)
    matheuristic_lns_enabled: bool = False
    matheuristic_lns_frequency: int = 0  # In-loop (0 = disabled, rely on final polish)
    matheuristic_lns_max_customers: int = 3
    matheuristic_lns_trials: int = 5  # Number of random neighborhoods for final polish

    def iterations_for(self) -> int:
        if self.iterations is not None:
            return self.iterations
        return 2000 if self.size.lower() == "small" else 4000

    def r_upper(self) -> float:
        return self.r_upper_small if self.size.lower() == "small" else self.r_upper_large


# Preferred explicit name kept as an alias for backwards compatibility.
SAConfig = SANNCfg


class SimulatedAnnealingALNS:
    """ALNS loop with simulated annealing acceptance and adaptive operator choice."""

    def __init__(
        self,
        *,
        instance,
        destroy_ops: Iterable[DestroyOperator],
        repair_ops: Iterable[RepairOperator],
        evaluator: Evaluator,
        cfg: SANNCfg,
        rng: Optional[random.Random] = None,
        verbose: bool = True,
        robust_verifier: Optional[Evaluator] = None,
        robust_check_every: int = 0,
        robust_check_on_new_best: bool = False,
        candidate_subroute_verifier: Optional[SubrouteRobustVerifier] = None,
        conservative_cost_evaluator: Optional[Evaluator] = None,
        collect_robust_route_pool: bool = False,
    ) -> None:
        self._verbose = verbose
        self._destroy_ops = list(destroy_ops)
        self._repair_ops = list(repair_ops)
        if not self._destroy_ops or not self._repair_ops:
            raise ValueError(
                "ALNS requires at least one destroy and one repair operator.")

        self._evaluator = evaluator
        self._robust_verifier = robust_verifier
        self._robust_check_every = max(0, int(robust_check_every))
        self._robust_check_on_new_best = bool(robust_check_on_new_best)
        self._candidate_subroute_verifier = candidate_subroute_verifier
        self._conservative_cost_evaluator = conservative_cost_evaluator
        self._collect_robust_route_pool = bool(collect_robust_route_pool)
        self._robust_cache_enabled = bool(cfg.robust_cache_enabled)
        self._robust_cache_size = max(0, int(cfg.robust_cache_size))
        self._robust_feasible_cache: dict[tuple, bool] = {}
        self._robust_cache_order: Deque[tuple] = deque()
        self._robust_eval_calls = 0
        self._robust_cache_hits = 0
        self._best_robust_certified = False
        self._cfg = cfg
        self._rng = rng or random.Random(random.getrandbits(32))
        self._instance = instance
        self._n_customers = len(instance.customer_manager.customer_ids())
        if self._n_customers == 0:
            raise ValueError("Instance must contain at least one customer.")
        self._energy_model = DroneEnergyModel()
        self._cooling_rate_initial = cfg.cooling_rate_initial or cfg.cooling_rate
        self._cooling_rate_final = cfg.cooling_rate_final or cfg.cooling_rate
        self._cooling_transition_iters = max(1, cfg.cooling_transition_iters)
        self._base_quota_upper_ratio = self._cfg.r_upper()
        self._quota_scale = 1.0
        self._quota_upper_ratio = self._base_quota_upper_ratio
        self._aos = AdaptiveOperatorManager(
            self._destroy_ops,
            self._repair_ops,
            eta=cfg.eta,
            reward_scale=cfg.reward_scale,
            alpha=cfg.alpha_credit,
            rng=self._rng,
            history=cfg.weight_history,
            decay=cfg.weight_decay,
        )
        self._aos_base_p_floor = self._aos.probability_floor()
        self._biased_random_repair = next(
            (op for op in self._repair_ops if isinstance(op, RepairBiasedRandomized)),
            None,
        )
        self._shake_destroy = next(
            (op for op in self._destroy_ops if isinstance(op, DestroySegmentShuffle)),
            next(
                (op for op in self._destroy_ops if isinstance(op, DestroyRandom)),
                self._destroy_ops[0],
            ),
        )
        self._shake_repair = self._biased_random_repair or self._repair_ops[0]
        self._log_operator_metrics = cfg.log_operator_metrics
        self._operator_log_interval = max(1, cfg.operator_log_interval)
        self._track_no_cross_truck = cfg.track_no_cross_truck
        self._best_no_cross_truck_solution: Optional[Solution] = None
        self._best_no_cross_truck_cost = float("inf")

    def _try_update_no_cross_truck(self, solution: Solution, cost: float) -> None:
        """Update best zero-cross-truck solution if this candidate qualifies."""
        if not self._track_no_cross_truck:
            return
        if cost >= self._best_no_cross_truck_cost:
            return
        for task in solution.drone_tasks:
            if (task.launch_truck is not None
                    and task.land_truck is not None
                    and task.launch_truck != task.land_truck):
                return  # has cross-truck sortie, skip
        self._best_no_cross_truck_solution = solution.clone()
        self._best_no_cross_truck_cost = cost

    def run(self, initial: Solution, time_limit: float | None = None) -> Solution:
        current = initial.clone()

        # Pre-processing: remove energy-violating drone tasks from initial solution
        initial_details = self._evaluator.evaluate_with_details(current)
        if not initial_details.robustness.feasible:
            violating_ids = {b.task_id for b in initial_details.robustness.task_breakdown if not b.feasible}
            if violating_ids:
                removed_customers = []
                for t in current.drone_tasks:
                    if t.task_id in violating_ids:
                        removed_customers.extend(t.customers())
                current.drone_tasks = [t for t in current.drone_tasks if t.task_id not in violating_ids]
                # Insert removed customers into truck routes
                truck_dist = self._instance.distance_matrix("truck")
                node_index = {n: i for i, n in enumerate(self._instance.all_node_ids())}
                for cust in removed_customers:
                    best_pos, best_route, best_delta = None, None, float('inf')
                    for route in current.truck_routes:
                        for pos in range(1, len(route.nodes)):
                            prev, nxt = route.nodes[pos-1], route.nodes[pos]
                            i_prev, i_cust, i_nxt = node_index.get(prev), node_index.get(cust), node_index.get(nxt)
                            if i_prev is not None and i_cust is not None and i_nxt is not None:
                                delta = truck_dist[i_prev][i_cust] + truck_dist[i_cust][i_nxt] - truck_dist[i_prev][i_nxt]
                                if delta < best_delta:
                                    best_delta, best_pos, best_route = delta, pos, route
                    if best_route and best_pos:
                        best_route.nodes.insert(best_pos, cust)

        current_eval = self._evaluator.evaluate_solution(current)
        current_cost = self._search_cost(current, current_eval)
        best = current.clone()
        best_cost = current_cost
        best_feasible_solution: Optional[Solution] = current.clone() if current_eval.feasible else None
        best_feasible_cost = current_eval.total_cost if current_eval.feasible else float("inf")
        robust_route_pool: dict[tuple, tuple[float, Solution]] = {}
        robust_pool_best_cost = float("inf")
        robust_pool_best_solution: Optional[Solution] = None

        if self._collect_robust_route_pool and self._conservative_cost_evaluator is not None:
            robust_pool_best_cost, robust_pool_best_solution = self._maybe_update_robust_pool(
                robust_route_pool,
                robust_pool_best_cost,
                robust_pool_best_solution,
                best,
            )
        if self._robust_check_on_new_best:
            if self._candidate_subroute_verifier is not None:
                # Certify baseline best once so later best checks can use
                # incremental changed-subroute verification.
                self._best_robust_certified = self._lightweight_robust_check_with_cache(
                    best
                )
            else:
                self._best_robust_certified = self._robust_feasible_with_cache(best)

        configured_iterations = self._cfg.iterations_for()
        iterations = configured_iterations
        recovering_initial_feasibility = not math.isfinite(current_eval.total_cost)
        if recovering_initial_feasibility:
            iterations = max(
                iterations,
                self._feasibility_recovery_iteration_limit(configured_iterations),
            )
        termination_reason = "iterations_completed"
        executed_iterations = 0
        initial_temperature = self._initial_temperature(current_eval.total_cost)
        T = initial_temperature
        stall = 0
        total_stall = 0  # 总停滞计数 (用于多样化重启)
        acceptance_window: Deque[bool] = deque()
        accepted_worse_count = 0
        self._base_quota_upper_ratio = self._cfg.r_upper()
        self._quota_scale = 1.0
        self._quota_upper_ratio = self._base_quota_upper_ratio
        base_p_floor = self._aos.probability_floor()
        self._aos_base_p_floor = base_p_floor
        self._aos.set_probability_floor(base_p_floor)
        reheat_active = False
        reheat_end = -1
        recovery_active = False
        recovery_end = -1
        quota_recovery_start = 1.0
        upper_recovery_start = self._base_quota_upper_ratio
        p_floor_recovery_start = base_p_floor
        last_reheat_iter = -self._cfg.reheat_cooldown
        last_diversification_iter = -self._cfg.diversification_trigger_stall
        start_time = time.perf_counter()

        # 收敛增强: 改善追踪
        recent_improvements: Deque[float] = deque(
            maxlen=self._cfg.recent_improvement_window)
        dynamic_cooling_rate = self._cooling_rate_initial
        adaptive_quota_bonus = 0  # 自适应配额奖励
        last_best_iter = 0  # 上次找到最优解的迭代

        # Escape Mechanism State
        escape_active = False
        escape_counter = 0

        # --- Convergence & operator tracking ---
        self._convergence_history: List[Dict[str, Any]] = []
        self._operator_usage: Dict[str, int] = {}
        self._operator_weight_history: List[Dict[str, Any]] = []
        weight_log_interval = max(1, iterations // 200)  # ~200 snapshots

        for iteration in range(iterations):
            executed_iterations = iteration + 1
            if time_limit is not None and time.perf_counter() - start_time >= time_limit:
                print(
                    f"Time limit reached at iteration {iteration}; terminating search.")
                termination_reason = "time_limit"
                executed_iterations = iteration
                break

            # Check Escape Trigger (仅当启用时)
            if self._cfg.escape_enabled and not escape_active and stall >= self._cfg.escape_trigger_stall:
                escape_active = True
                escape_counter = 0
                stall = 0  # Reset stall to avoid re-triggering immediately after
                print(
                    f"Escape mechanism ACTIVATED at iteration {iteration + 1}")

            if reheat_active and iteration >= reheat_end:
                reheat_active = False
                if self._cfg.reheat_recovery > 0:
                    recovery_active = True
                    recovery_end = iteration + self._cfg.reheat_recovery
                    quota_recovery_start = self._quota_scale
                    upper_recovery_start = self._quota_upper_ratio
                    p_floor_recovery_start = self._aos.probability_floor()
                else:
                    self._quota_scale = 1.0
                    self._quota_upper_ratio = self._base_quota_upper_ratio
                    self._aos.set_probability_floor(base_p_floor)

            if recovery_active:
                steps_remaining = recovery_end - iteration
                total = self._cfg.reheat_recovery
                if total <= 0 or steps_remaining <= 0:
                    recovery_active = False
                    self._quota_scale = 1.0
                    self._quota_upper_ratio = self._base_quota_upper_ratio
                    self._aos.set_probability_floor(base_p_floor)
                else:
                    progress = (total - steps_remaining) / total
                    self._quota_scale = quota_recovery_start + \
                        (1.0 - quota_recovery_start) * progress
                    self._quota_upper_ratio = upper_recovery_start + (
                        self._base_quota_upper_ratio - upper_recovery_start
                    ) * progress
                    new_p_floor = p_floor_recovery_start + \
                        (base_p_floor - p_floor_recovery_start) * progress
                    self._aos.set_probability_floor(new_p_floor)

            if escape_active:
                # Randomly select operators for diversity (Escape Mode)
                destroy = self._rng.choice(self._destroy_ops)
                repair = self._rng.choice(self._repair_ops)
            else:
                destroy = self._aos.select_destroy()
                repair = self._aos.select_repair()
                if reheat_active and self._biased_random_repair is not None:
                    probability = max(
                        0.0, min(1.0, self._cfg.reheat_random_repair_prob))
                    if self._rng.random() < probability:
                        repair = self._biased_random_repair

            # 自适应配额调整
            quota = self._sample_quota(adaptive_bonus=adaptive_quota_bonus)
            destroy_time = 0.0
            repair_time = 0.0
            candidate_eval = None
            try:
                if self._cfg.measure_operator_time:
                    start = time.perf_counter()
                    destroyed, pool = destroy.apply(current, quota)
                    destroy_time = time.perf_counter() - start
                else:
                    destroyed, pool = destroy.apply(current, quota)
                    destroy_time = 1.0

                if self._cfg.measure_operator_time:
                    start = time.perf_counter()
                    candidate = repair.apply(destroyed, pool.customers)
                    repair_time = time.perf_counter() - start
                else:
                    candidate = repair.apply(destroyed, pool.customers)
                    repair_time = 1.0

                if (
                    self._candidate_subroute_verifier is not None
                    and not self._candidate_subroute_verifier.verify_candidate(
                        base=current,
                        candidate=candidate,
                    )
                ):
                    candidate_cost = float("inf")
                else:
                    candidate_eval = self._evaluator.evaluate_solution(candidate)
                    candidate_cost = self._search_cost(candidate, candidate_eval)
                if (
                    self._collect_robust_route_pool
                    and self._conservative_cost_evaluator is not None
                    and candidate_eval is not None
                    and candidate_eval.feasible
                ):
                    robust_pool_best_cost, robust_pool_best_solution = self._maybe_update_robust_pool(
                        robust_route_pool,
                        robust_pool_best_cost,
                        robust_pool_best_solution,
                        candidate,
                    )
                if (
                    self._robust_verifier is not None
                    and self._robust_check_every > 0
                    and (iteration + 1) % self._robust_check_every == 0
                    and candidate_eval is not None
                    and candidate_eval.feasible
                ):
                    if not self._robust_feasible_with_cache(candidate):
                        candidate_cost = float("inf")
            except Exception:
                self._aos.update(
                    destroy,
                    repair,
                    "rejected",
                    0.0,
                    destroy_time,
                    repair_time,
                )
                continue

            if escape_active:
                # Force acceptance in escape mode
                accepted = True
                improvement = 0.0
                reward = "accepted_worse"  # Dummy reward
                escape_counter += 1

                # Check for global improvement (Exit condition 1)
                if candidate_cost < best_cost:
                    print(
                        f"Escape successful! Found new best at iter {iteration+1}")
                    escape_active = False
                    stall = 0
                    # Continue to normal update logic which handles best update

                # Check duration limit (Exit condition 2)
                elif escape_counter >= self._cfg.escape_duration:
                    print(
                        f"Escape finished at iter {iteration+1} (duration reached)")
                    escape_active = False
                    stall = 0  # Reset stall
            else:
                reward, accepted, improvement = self._acceptance(
                    current_cost,
                    candidate_cost,
                    T,
                    best_cost,
                )

            # 记录改善情况
            recent_improvements.append(improvement)

            if accepted:
                current = candidate
                current_cost = candidate_cost
                candidate_improves_best = candidate_cost < best_cost
                if candidate_improves_best and not self._is_robust_feasible_best(
                    candidate,
                    candidate_cost=candidate_cost,
                    best_cost=best_cost,
                    base_solution=best,
                ):
                    candidate_improves_best = False
                stall = 0 if candidate_improves_best else stall + 1

                # Apply local search when finding new best
                is_new_best = candidate_improves_best
                if is_new_best:
                    best = candidate.clone()
                    best_cost = candidate_cost
                    self._try_update_no_cross_truck(best, best_cost)
                    self._best_robust_certified = True
                    if (
                        candidate_eval is not None
                        and candidate_eval.feasible
                        and candidate_eval.total_cost < best_feasible_cost
                    ):
                        best_feasible_solution = candidate.clone()
                        best_feasible_cost = candidate_eval.total_cost
                    last_best_iter = iteration
                    total_stall = 0  # 重置总停滞

                    # 自适应配额: 改善时减少配额
                    if self._cfg.adaptive_quota_enabled and self._cfg.quota_decrease_on_improve:
                        adaptive_quota_bonus = max(0, adaptive_quota_bonus - 2)

                    # Local search on new best
                    if self._cfg.local_search_on_new_best:
                        try:
                            ls_solution = self._local_search(best.clone())
                            ls_eval = self._evaluator.evaluate_solution(
                                ls_solution)
                            if (
                                math.isfinite(ls_eval.total_cost)
                                and ls_eval.total_cost < best_cost
                                and self._is_robust_feasible_best(
                                    ls_solution,
                                    candidate_cost=ls_eval.total_cost,
                                    best_cost=best_cost,
                                    base_solution=best,
                                )
                            ):
                                best = ls_solution
                                best_cost = ls_eval.total_cost
                                self._try_update_no_cross_truck(best, best_cost)
                                if ls_eval.total_cost < best_feasible_cost:
                                    best_feasible_solution = ls_solution.clone()
                                    best_feasible_cost = ls_eval.total_cost
                                self._best_robust_certified = True
                                current = ls_solution.clone()
                                current_cost = best_cost
                        except Exception:
                            pass
            else:
                stall += 1
                total_stall += 1

                # 自适应配额: 停滞时增加配额
                if self._cfg.adaptive_quota_enabled and stall > 0 and stall % self._cfg.quota_increase_on_stall == 0:
                    adaptive_quota_bonus = min(10, adaptive_quota_bonus + 1)

            # Periodic local search
            if self._cfg.local_search_frequency > 0 and (iteration + 1) % self._cfg.local_search_frequency == 0:
                try:
                    ls_solution = self._local_search(current.clone())
                    ls_eval = self._evaluator.evaluate_solution(ls_solution)
                    if (
                        math.isfinite(ls_eval.total_cost)
                        and ls_eval.total_cost < current_cost
                        and self._is_robust_feasible(ls_solution)
                    ):
                        current = ls_solution
                        current_cost = ls_eval.total_cost
                        if current_cost < best_cost and self._is_robust_feasible_best(
                            current,
                            candidate_cost=current_cost,
                            best_cost=best_cost,
                            base_solution=best,
                        ):
                            best = current.clone()
                            best_cost = current_cost
                            self._try_update_no_cross_truck(best, best_cost)
                            if current_cost < best_feasible_cost:
                                best_feasible_solution = current.clone()
                                best_feasible_cost = current_cost
                            self._best_robust_certified = True
                            stall = 0
                            total_stall = 0
                            last_best_iter = iteration
                except Exception:
                    pass

            # Matheuristic LNS: exhaustive sortie reconstruction (Step 11)
            if self._cfg.matheuristic_lns_enabled and self._cfg.matheuristic_lns_frequency > 0 and (
                iteration + 1
            ) % self._cfg.matheuristic_lns_frequency == 0:
                try:
                    mlns_sol = self._apply_matheuristic_lns(current.clone())
                    mlns_eval = self._evaluator.evaluate_solution(mlns_sol)
                    if (
                        math.isfinite(mlns_eval.total_cost)
                        and mlns_eval.total_cost < current_cost
                        and self._is_robust_feasible(mlns_sol)
                    ):
                        current = mlns_sol
                        current_cost = mlns_eval.total_cost
                        if current_cost < best_cost and self._is_robust_feasible_best(
                            current,
                            candidate_cost=current_cost,
                            best_cost=best_cost,
                            base_solution=best,
                        ):
                            best = current.clone()
                            best_cost = current_cost
                            self._try_update_no_cross_truck(best, best_cost)
                            if current_cost < best_feasible_cost:
                                best_feasible_solution = current.clone()
                                best_feasible_cost = current_cost
                            self._best_robust_certified = True
                            stall = 0
                            total_stall = 0
                            last_best_iter = iteration
                except Exception:
                    pass

            # Full intensification periodically
            if self._cfg.intensify_frequency > 0 and (iteration + 1) % self._cfg.intensify_frequency == 0:
                try:
                    intense_solution = self._intensify_search(best.clone())
                    intense_eval = self._evaluator.evaluate_solution(
                        intense_solution)
                    if (
                        math.isfinite(intense_eval.total_cost)
                        and intense_eval.total_cost < best_cost
                        and self._is_robust_feasible_best(
                            intense_solution,
                            candidate_cost=intense_eval.total_cost,
                            best_cost=best_cost,
                            base_solution=best,
                        )
                    ):
                        best = intense_solution
                        best_cost = intense_eval.total_cost
                        self._try_update_no_cross_truck(best, best_cost)
                        if intense_eval.total_cost < best_feasible_cost:
                            best_feasible_solution = intense_solution.clone()
                            best_feasible_cost = intense_eval.total_cost
                        self._best_robust_certified = True
                        current = intense_solution.clone()
                        current_cost = best_cost
                        stall = 0
                        total_stall = 0
                        last_best_iter = iteration
                except Exception:
                    pass

            # Path relinking to best solution occasionally
            if self._cfg.path_relinking_prob > 0 and self._rng.random() < self._cfg.path_relinking_prob / 10:
                try:
                    pr_solution = self._path_relinking(
                        current.clone(), best.clone())
                    pr_eval = self._evaluator.evaluate_solution(pr_solution)
                    if (
                        math.isfinite(pr_eval.total_cost)
                        and pr_eval.total_cost < current_cost
                        and self._is_robust_feasible(pr_solution)
                    ):
                        current = pr_solution
                        current_cost = pr_eval.total_cost
                        if current_cost < best_cost and self._is_robust_feasible_best(
                            current,
                            candidate_cost=current_cost,
                            best_cost=best_cost,
                            base_solution=best,
                        ):
                            best = current.clone()
                            best_cost = current_cost
                            self._try_update_no_cross_truck(best, best_cost)
                            if current_cost < best_feasible_cost:
                                best_feasible_solution = current.clone()
                                best_feasible_cost = current_cost
                            self._best_robust_certified = True
                            stall = 0
                            total_stall = 0
                            last_best_iter = iteration
                except Exception:
                    pass

            self._aos.update(
                destroy,
                repair,
                reward,
                improvement,
                destroy_time,
                repair_time,
            )

            # --- Record convergence & operator data ---
            destroy_name = getattr(destroy, "name", destroy.__class__.__name__)
            repair_name = getattr(repair, "name", repair.__class__.__name__)
            self._convergence_history.append({
                "iteration": iteration + 1,
                "current_cost": current_cost,
                "best_cost": best_cost,
                "temperature": T,
                "reward": reward,
                "destroy": destroy_name,
                "repair": repair_name,
            })
            self._operator_usage[destroy_name] = self._operator_usage.get(destroy_name, 0) + 1
            self._operator_usage[repair_name] = self._operator_usage.get(repair_name, 0) + 1
            if (iteration + 1) % weight_log_interval == 0 or iteration == 0:
                snap = self._aos.snapshot()
                record: Dict[str, Any] = {"iteration": iteration + 1}
                for entry in snap.get("destroy", []):
                    record[f"d_{entry['name']}"] = entry["weight"]
                for entry in snap.get("repair", []):
                    record[f"r_{entry['name']}"] = entry["weight"]
                self._operator_weight_history.append(record)

            # 动态冷却率调整
            if self._cfg.dynamic_cooling_enabled:
                cooling_rate = self._dynamic_cooling_rate(
                    iteration, recent_improvements, stall)
            else:
                cooling_rate = self._cooling_rate_for(iteration)
            T = max(self._cfg.temperature_min, T * cooling_rate)

            accepted_worse = reward == "accepted_worse"
            if self._cfg.reheat_acceptance_window > 0:
                acceptance_window.append(accepted_worse)
                accepted_worse_count += int(accepted_worse)
                if len(acceptance_window) > self._cfg.reheat_acceptance_window:
                    removed = acceptance_window.popleft()
                    accepted_worse_count -= int(removed)

            trigger_allowed = iteration - \
                last_reheat_iter >= max(0, self._cfg.reheat_cooldown)
            acceptance_ready = (
                self._cfg.reheat_acceptance_window > 0
                and len(acceptance_window) >= self._cfg.reheat_acceptance_window
            )
            acceptance_ratio = (
                accepted_worse_count / len(acceptance_window)
                if acceptance_ready and len(acceptance_window) > 0
                else None
            )
            acceptance_threshold = max(
                0.0, min(1.0, self._cfg.reheat_acceptance_min))

            # Use iters_since_best instead of stall for reheat trigger
            # This avoids the issue where escape mechanism resets stall
            iters_since_best_for_reheat = iteration - last_best_iter
            stall_trigger = (
                trigger_allowed
                and not reheat_active
                and not recovery_active
                and iters_since_best_for_reheat >= max(0, self._cfg.reheat_stall_trigger)
            )
            acceptance_trigger = (
                trigger_allowed
                and not reheat_active
                and not recovery_active
                and acceptance_ready
                and acceptance_ratio is not None
                and acceptance_ratio < acceptance_threshold
            )

            if stall_trigger or acceptance_trigger:
                reheat_active = True
                reheat_end = iteration + self._cfg.reheat_duration
                last_reheat_iter = iteration
                stall = 0
                self._quota_scale = max(1.0, self._cfg.reheat_quota_multiplier)
                adjusted_upper = min(
                    1.0,
                    max(
                        self._base_quota_upper_ratio,
                        self._base_quota_upper_ratio * self._cfg.reheat_quota_multiplier,
                    ),
                )
                cap_ratio = self._cfg.reheat_quota_upper_cap
                if cap_ratio is not None:
                    self._quota_upper_ratio = min(cap_ratio, adjusted_upper)
                else:
                    self._quota_upper_ratio = adjusted_upper
                self._quota_upper_ratio = max(
                    self._quota_upper_ratio, self._base_quota_upper_ratio)
                self._aos.set_probability_floor(self._cfg.reheat_p_floor)
                T = initial_temperature * max(0.01, self._cfg.reheat_temperature_scale)
                acceptance_window.clear()
                accepted_worse_count = 0
                recovery_active = False
                msg_reason = "stall" if stall_trigger else "acceptance"
                if self._verbose:
                    print(
                        f"Reheating triggered ({msg_reason}) at iteration {iteration + 1}; temperature reset to {T:.4f}."
                    )

                if self._cfg.reheat_shake_probability > 0 and self._rng.random() < self._cfg.reheat_shake_probability:
                    shaken = self._shake(current)
                    if shaken is not None:
                        shaken_eval = self._evaluator.evaluate_solution(shaken)
                        shaken_cost = shaken_eval.total_cost
                        if math.isfinite(shaken_cost) and self._is_robust_feasible(shaken):
                            current = shaken
                            current_cost = shaken_cost
                            stall = 0
                            if shaken_cost < best_cost and self._is_robust_feasible_best(
                                shaken,
                                candidate_cost=shaken_cost,
                                best_cost=best_cost,
                                base_solution=best,
                            ):
                                best = shaken.clone()
                                best_cost = shaken_cost
                                self._try_update_no_cross_truck(best, best_cost)
                                if shaken_cost < best_feasible_cost:
                                    best_feasible_solution = shaken.clone()
                                    best_feasible_cost = shaken_cost
                                self._best_robust_certified = True
                                last_best_iter = iteration

            # 多样化重启 (长期停滞时触发)
            diversification_allowed = (
                self._cfg.diversification_enabled
                and iteration - last_diversification_iter >= self._cfg.diversification_trigger_stall
                and total_stall >= self._cfg.diversification_trigger_stall
                and not reheat_active
            )
            if diversification_allowed:
                last_diversification_iter = iteration
                if self._verbose:
                    print(
                        f"Diversification restart at iteration {iteration + 1} (total_stall={total_stall})")

                # 决定从哪里重启
                if self._rng.random() < self._cfg.diversification_restart_best_prob:
                    restart_from = best.clone()
                else:
                    restart_from = current.clone()

                # 强力扰动
                diversified = self._diversify(restart_from)
                if diversified is not None:
                    div_eval = self._evaluator.evaluate_solution(diversified)
                    if math.isfinite(div_eval.total_cost) and self._is_robust_feasible(diversified):
                        current = diversified
                        current_cost = div_eval.total_cost
                        stall = 0
                        # 不重置total_stall，避免频繁多样化

                        # 温度部分重置
                        T = initial_temperature * 0.5

                        if current_cost < best_cost and self._is_robust_feasible_best(
                            current,
                            candidate_cost=current_cost,
                            best_cost=best_cost,
                            base_solution=best,
                        ):
                            best = current.clone()
                            best_cost = current_cost
                            self._try_update_no_cross_truck(best, best_cost)
                            if current_cost < best_feasible_cost:
                                best_feasible_solution = current.clone()
                                best_feasible_cost = current_cost
                            self._best_robust_certified = True
                            total_stall = 0
                            last_best_iter = iteration

            if self._verbose and ((iteration + 1) % 100 == 0 or iteration == 0):
                iters_since_best = iteration - last_best_iter
                print(
                    f"Iteration {iteration + 1}: current cost {current_cost:.3f}, best {best_cost:.3f}, "
                    f"T {T:.4f}, stall {stall}, since_best {iters_since_best}"
                )
            if (
                self._log_operator_metrics
                and self._cfg.operator_log_interval > 0
                and (iteration + 1) % self._operator_log_interval == 0
            ):
                self._print_operator_snapshot(iteration + 1)
            if (
                recovering_initial_feasibility
                and best_feasible_solution is not None
                and executed_iterations >= configured_iterations
            ):
                termination_reason = "feasibility_recovered"
                break
            if (
                self._cfg.max_non_improve is not None
                and best_feasible_solution is not None
                and stall >= self._cfg.max_non_improve
            ):
                termination_reason = "max_non_improve"
                break

        self.last_run_stats = {
            "configured_iterations": configured_iterations,
            "effective_iterations": iterations,
            "executed_iterations": executed_iterations,
            "termination_reason": termination_reason,
            "best_cost": best_cost,
            "best_feasible_cost": best_feasible_cost,
            "elapsed_time": time.perf_counter() - start_time,
            "robust_eval_calls": self._robust_eval_calls,
            "robust_cache_hits": self._robust_cache_hits,
            "robust_cache_size": len(self._robust_feasible_cache),
        }
        if self._collect_robust_route_pool and self._conservative_cost_evaluator is not None:
            self.last_run_stats["robust_route_pool_size"] = len(robust_route_pool)
            self.last_run_stats["robust_route_pool_best_cost"] = robust_pool_best_cost
            if robust_pool_best_solution is not None:
                return robust_pool_best_solution
        if best_feasible_solution is not None:
            return best_feasible_solution
        return best

    def _feasibility_recovery_iteration_limit(self, configured_iterations: int) -> int:
        """Return a bounded recovery budget for runs that start infeasible."""
        scaled = 20 * self._n_customers * self._n_customers
        return max(configured_iterations, min(4000, max(200, scaled)))

    def run_with_milp_polish(
        self,
        initial: Solution,
        time_limit: float | None = None,
        milp_time_limit: float = 30.0,
        milp_gap: float = 0.005,
    ) -> Solution:
        """Run ALNS, then polish the best solution with mini-MILP."""
        best = self.run(initial, time_limit=time_limit)
        from alns_vrpfd.core.operators.mini_milp_polish import polish_with_mini_milp

        result = polish_with_mini_milp(
            solution=best,
            instance=self._evaluator._instance,
            evaluator=self._evaluator,
            time_limit=milp_time_limit,
            mip_gap=milp_gap,
            verbose=False,
        )
        if result.improved:
            logger.info(
                f"Mini-MILP final polish: {result.original_cost:.2f} -> "
                f"{result.polished_cost:.2f} (delta={result.improvement:.2f})"
            )
            return result.polished_solution
        return best

    def run_with_full_milp_polish(
        self,
        initial: Solution,
        time_limit: float | None = None,
        milp_time_limit: float = 30.0,
        milp_gap: float = 0.005,
    ) -> Solution:
        """Run ALNS, then polish with the full MILP warm-start."""
        best = self.run(initial, time_limit=time_limit)
        from alns_vrpfd.core.operators.milp_warm_start import polish_with_full_milp_warm_start

        result = polish_with_full_milp_warm_start(
            solution=best,
            instance=self._evaluator._instance,
            evaluator=self._evaluator,
            time_limit=milp_time_limit,
            mip_gap=milp_gap,
            verbose=False,
        )
        if result.improved and result.polished_solution is not None:
            logger.info(
                f"Full MILP polish: {result.original_cost:.2f} -> "
                f"{result.polished_cost:.2f} (delta={result.improvement:.2f}, "
                f"gap={result.gap:.2%}, status={result.status})"
            )
            return result.polished_solution
        return best

    def run_with_matheuristic_lns_polish(
        self,
        initial: Solution,
        time_limit: float | None = None,
    ) -> Solution:
        """Run ALNS, then polish with matheuristic LNS (multiple neighborhoods).

        Tries multiple random neighborhoods for exhaustive sortie
        reconstruction on the best solution found by ALNS.
        """
        best = self.run(initial, time_limit=time_limit)
        if not self._cfg.matheuristic_lns_enabled:
            return best

        from alns_vrpfd.core.operators.matheuristic_lns import MatheuristicLNSRepair

        best_cost = self._evaluator.evaluate_solution(best).total_cost
        if not math.isfinite(best_cost):
            return best

        trials = getattr(self._cfg, "matheuristic_lns_trials", 5)
        max_cust = getattr(self._cfg, "matheuristic_lns_max_customers", 3)

        for trial in range(trials):
            lns = MatheuristicLNSRepair(
                instance=self._evaluator._instance,
                evaluator=self._evaluator,
                max_customers=max_cust,
                max_anchor_dist_factor=2.0,
                energy_tolerance=1.0,
                rng=self._rng,
            )
            improved = lns.apply(best)
            imp_cost = self._evaluator.evaluate_solution(improved).total_cost
            if (
                math.isfinite(imp_cost)
                and imp_cost < best_cost - 1e-6
            ):
                logger.info(
                    f"Matheuristic LNS polish (trial {trial + 1}/{trials}): "
                    f"{best_cost:.2f} -> {imp_cost:.2f} "
                    f"(delta={best_cost - imp_cost:.2f}, "
                    f"candidates={lns.created})"
                )
                best = improved
                best_cost = imp_cost
                self._try_update_no_cross_truck(best, best_cost)
            else:
                logger.debug(
                    f"Matheuristic LNS polish trial {trial + 1}/{trials}: "
                    f"no improvement ({best_cost:.2f} -> {imp_cost:.2f}, "
                    f"candidates={lns.created})"
                )

        return best

    @property
    def convergence_history(self) -> List[Dict[str, Any]]:
        return getattr(self, "_convergence_history", [])

    @property
    def operator_usage(self) -> Dict[str, int]:
        return getattr(self, "_operator_usage", {})

    @property
    def operator_weight_history(self) -> List[Dict[str, Any]]:
        return getattr(self, "_operator_weight_history", [])

    def _is_robust_feasible(self, solution: Solution) -> bool:
        if self._robust_verifier is None or self._robust_check_every <= 0:
            # Even when periodic checks are disabled, use lightweight verifier
            # if available.  This prevents non-robust solutions from silently
            # becoming `current` via periodic local search or path relinking.
            if self._candidate_subroute_verifier is not None:
                return self._lightweight_robust_check_with_cache(solution)
            return True
        return self._robust_feasible_with_cache(solution)

    def _maybe_update_robust_pool(
        self,
        pool: dict[tuple, tuple[float, Solution]],
        best_cost: float,
        best_solution: Optional[Solution],
        candidate: Solution,
    ) -> tuple[float, Optional[Solution]]:
        if self._conservative_cost_evaluator is None:
            return best_cost, best_solution

        signature = self._solution_signature(candidate)
        if signature in pool:
            return best_cost, best_solution

        try:
            robust_eval = self._conservative_cost_evaluator.evaluate_solution(
                candidate)
        except Exception:
            return best_cost, best_solution

        robust_cost = robust_eval.total_cost
        if not math.isfinite(robust_cost):
            return best_cost, best_solution

        snapshot = candidate.clone()
        pool[signature] = (robust_cost, snapshot)
        if robust_cost < best_cost:
            return robust_cost, snapshot
        return best_cost, best_solution

    @staticmethod
    def _solution_signature(solution: Solution) -> tuple:
        truck_signature = tuple(
            (route.id, tuple(route.nodes))
            for route in sorted(solution.truck_routes, key=lambda r: r.id)
        )
        drone_signature = tuple(
            (
                task.drone_id,
                task.launch_truck,
                task.launch_node,
                tuple(task.customers()),
                task.land_truck,
                task.retrieve_node,
                tuple(round(float(payload), 8) for payload in task.payloads),
            )
            for task in solution.drone_tasks
        )
        return truck_signature, drone_signature

    @staticmethod
    def _drone_only_signature(solution: Solution) -> tuple:
        """Signature based solely on drone tasks.

        Robust energy feasibility depends only on drone task structure
        (launch/retrieve nodes, customers, payloads), not on truck routes.
        Using this narrower key dramatically improves cache hit rate.
        """
        return tuple(
            (
                task.drone_id,
                task.launch_node,
                tuple(task.customers()),
                task.retrieve_node,
                tuple(round(float(payload), 8) for payload in task.payloads),
            )
            for task in solution.drone_tasks
        )

    def _is_robust_feasible_best(
        self,
        solution: Solution,
        candidate_cost: float | None = None,
        best_cost: float | None = None,
        base_solution: Solution | None = None,
    ) -> bool:
        if self._robust_verifier is None:
            return True
        if self._robust_check_every <= 0 and not self._robust_check_on_new_best:
            return True
        # Fast path: if current best is already certified robust-feasible,
        # only changed sub-routes need to be re-checked for a new candidate.
        if self._candidate_subroute_verifier is not None:
            if base_solution is not None and self._best_robust_certified:
                return self._lightweight_incremental_best_check_with_cache(
                    base_solution=base_solution,
                    candidate=solution,
                )
            return self._lightweight_robust_check_with_cache(solution)
        return self._robust_feasible_with_cache(solution)

    def _lightweight_incremental_best_check_with_cache(
        self,
        *,
        base_solution: Solution,
        candidate: Solution,
    ) -> bool:
        signature = self._drone_only_signature(candidate)
        if self._robust_cache_enabled and self._robust_cache_size > 0:
            cached = self._robust_feasible_cache.get(signature)
            if cached is not None:
                self._robust_cache_hits += 1
                return cached
        try:
            feasible = self._candidate_subroute_verifier.verify_candidate(
                base=base_solution,
                candidate=candidate,
            )
        except Exception:
            feasible = False
        self._robust_eval_calls += 1
        if self._robust_cache_enabled and self._robust_cache_size > 0:
            self._remember_robust_feasible(signature, feasible)
        return feasible

    def _lightweight_robust_check_with_cache(self, solution: Solution) -> bool:
        """Check robust energy feasibility using the lightweight verifier.

        Uses drone-only signature for cache keys since robust energy
        feasibility depends only on drone task structure.
        """
        signature = self._drone_only_signature(solution)
        if self._robust_cache_enabled and self._robust_cache_size > 0:
            cached = self._robust_feasible_cache.get(signature)
            if cached is not None:
                self._robust_cache_hits += 1
                return cached
        try:
            feasible = self._candidate_subroute_verifier.verify_all_tasks(solution)
        except Exception:
            feasible = False
        self._robust_eval_calls += 1
        if self._robust_cache_enabled and self._robust_cache_size > 0:
            self._remember_robust_feasible(signature, feasible)
        return feasible

    def _robust_feasible_with_cache(self, solution: Solution) -> bool:
        """Fallback: full Evaluator-based robust check with cache."""
        if self._robust_verifier is None:
            return True
        signature = self._drone_only_signature(solution)
        if self._robust_cache_enabled and self._robust_cache_size > 0:
            cached = self._robust_feasible_cache.get(signature)
            if cached is not None:
                self._robust_cache_hits += 1
                return cached
        try:
            robust_eval = self._robust_verifier.evaluate_solution(solution)
        except Exception:
            return False
        self._robust_eval_calls += 1
        feasible = math.isfinite(robust_eval.total_cost)
        if self._robust_cache_enabled and self._robust_cache_size > 0:
            self._remember_robust_feasible(signature, feasible)
        return feasible

    def _remember_robust_feasible(self, signature: tuple, feasible: bool) -> None:
        if signature in self._robust_feasible_cache:
            self._robust_feasible_cache[signature] = feasible
            return
        if len(self._robust_feasible_cache) >= self._robust_cache_size:
            oldest = self._robust_cache_order.popleft()
            self._robust_feasible_cache.pop(oldest, None)
        self._robust_feasible_cache[signature] = feasible
        self._robust_cache_order.append(signature)

    def _dynamic_cooling_rate(
        self,
        iteration: int,
        recent_improvements: Deque[float],
        stall: int
    ) -> float:
        """动态调整冷却率

        - 如果最近有显著改善，减慢冷却（保持探索）
        - 如果长期停滞，加速冷却（促进收敛）
        - 否则使用标准冷却率
        """
        base_rate = self._cooling_rate_for(iteration)

        if not recent_improvements:
            return base_rate

        # 计算最近平均改善
        avg_improvement = sum(recent_improvements) / len(recent_improvements)

        # 如果有显著改善，减慢冷却
        if avg_improvement > self._cfg.improvement_threshold:
            return min(self._cfg.cooling_slowdown_factor, base_rate * 1.005)

        # 如果停滞，加速冷却
        if stall > self._cfg.reheat_stall_trigger // 2:
            return max(self._cfg.cooling_speedup_factor, base_rate * 0.995)

        return base_rate

    def _diversify(self, solution: Solution) -> Optional[Solution]:
        """执行强力多样化扰动"""
        destroy_ratio = self._cfg.diversification_destroy_ratio
        quota = max(3, int(round(destroy_ratio * self._n_customers)))

        try:
            # 使用shake_destroy进行大规模破坏
            destroyed, pool = self._shake_destroy.apply(
                solution.clone(), quota)

            # 使用随机修复
            if self._biased_random_repair:
                repaired = self._biased_random_repair.apply(
                    destroyed, pool.customers)
            else:
                repaired = self._shake_repair.apply(destroyed, pool.customers)

            return repaired
        except Exception:
            return None

    def _cooling_rate_for(self, iteration: int) -> float:
        if self._cooling_transition_iters <= 0:
            return self._cooling_rate_final
        progress = min(1.0, max(0.0, iteration /
                       self._cooling_transition_iters))
        return self._cooling_rate_initial + (
            self._cooling_rate_final - self._cooling_rate_initial
        ) * progress

    def _acceptance(
        self,
        current_cost: float,
        candidate_cost: float,
        temperature: float,
        best_cost: float,
    ) -> tuple[RewardLabel, bool, float]:
        current_finite = math.isfinite(current_cost)
        candidate_finite = math.isfinite(candidate_cost)
        best_finite = math.isfinite(best_cost)

        if not current_finite and candidate_finite:
            return "better", True, 1.0
        if current_finite and not candidate_finite:
            return "accepted_worse", False, 0.0
        if not candidate_finite and not current_finite:
            return "accepted_worse", False, 0.0

        delta = candidate_cost - current_cost
        improvement = 0.0

        # σ₁: 找到新的全局最优解
        if candidate_finite and best_finite and candidate_cost < best_cost:
            denom = max(abs(best_cost), 1.0)
            improvement = max(0.0, (best_cost - candidate_cost) / denom)
            return "global", True, improvement

        # σ₂ 和 σ₃: 区分显著改善和略有改善
        if delta <= 0:
            denom = max(abs(current_cost), 1.0)
            improvement = max(0.0, -delta / denom)

            # 设置阈值区分显著改善和略有改善
            # 如果改善幅度 > 1%(可调整),则为显著改善(σ₂)
            # 否则为略有改善(σ₃)
            improvement_threshold = 0.01
            if improvement > improvement_threshold:
                return "better", True, improvement  # σ₂: 显著改善
            else:
                return "slight_better", True, improvement  # σ₃: 略有改善

        # σ₄: 接受较差解(模拟退火准则)
        if temperature > 0 and self._rng.random() < math.exp(-delta / max(temperature, 1e-9)):
            return "accepted_worse", True, 0.0

        # 拒绝解(不计入奖励更新,但在统计中仍可追踪)
        return "accepted_worse", False, 0.0

    def _search_cost(self, solution: Solution, eval_result: Any) -> float:
        """Return the objective used by the search acceptance rule.

        Feasible solutions keep the evaluator's true objective. Infeasible
        solutions receive a large finite penalty so feasibility recovery can
        accept candidates that reduce violations instead of getting stuck at
        inf -> inf.
        """
        feasible = bool(getattr(eval_result, "feasible", False))
        total_cost = getattr(eval_result, "total_cost", float("inf"))
        if feasible and math.isfinite(total_cost):
            return total_cost

        truck_cost = _finite_or_zero(getattr(eval_result, "truck_distance_cost", 0.0))
        drone_cost = _finite_or_zero(getattr(eval_result, "drone_distance_cost", 0.0))
        delay_penalty = _finite_or_zero(getattr(eval_result, "delay_penalty", 0.0))
        energy_penalty = _finite_or_zero(getattr(eval_result, "energy_penalty", 0.0))

        time_window_violations = 0
        time_window_excess = 0.0
        robustness_violations = 0
        try:
            details = self._evaluator.evaluate_with_details(solution)
            violations = getattr(getattr(details, "delay_breakdown", None), "violations", ()) or ()
            time_window_violations = len(violations)
            time_window_excess = sum(
                max(
                    0.0,
                    _finite_or_zero(getattr(v, "arrival_time", 0.0))
                    - _finite_or_zero(getattr(v, "latest_time", 0.0)),
                )
                for v in violations
            )
            robustness = getattr(details, "robustness", None)
            if robustness is not None and not getattr(robustness, "feasible", True):
                robustness_violations = sum(
                    1
                    for item in getattr(robustness, "task_breakdown", ()) or ()
                    if not getattr(item, "feasible", True)
                )
        except Exception:
            pass

        coverage_penalty = 100_000.0 if self._has_coverage_gap(solution) else 0.0
        structural_penalty = 100_000.0 * self._hard_solution_violation_count(solution)

        return (
            1_000_000.0
            + coverage_penalty
            + structural_penalty
            + 100_000.0 * time_window_violations
            + 10_000.0 * time_window_excess
            + 50_000.0 * robustness_violations
            + truck_cost
            + drone_cost
            + delay_penalty
            + energy_penalty
        )

    def _has_coverage_gap(self, solution: Solution) -> bool:
        try:
            customer_ids = set(self._instance.customer_manager.customer_ids())
        except Exception:
            return False
        served = set()
        for route in solution.truck_routes:
            served.update(node for node in route.nodes if node in customer_ids)
        for task in solution.drone_tasks:
            served.update(node for node in task.customers() if node in customer_ids)
        return served != customer_ids

    def _hard_solution_violation_count(self, solution: Solution) -> int:
        checks = (
            "_has_drone_task_violations",
            "_has_drone_anchor_conflicts",
            "_has_forced_drone_violation",
            "_has_drone_limit_violations",
            "_has_depot_start_retrieve_violation",
        )
        count = 0
        for name in checks:
            check = getattr(self._evaluator, name, None)
            if check is None:
                continue
            try:
                if check(solution):
                    count += 1
            except Exception:
                continue
        return count

    def _initial_temperature(self, cost: float) -> float:
        if not math.isfinite(cost):
            # 当初始解不可行时，使用较高的默认温度以允许更多探索
            return max(self._cfg.temperature_min, 50.0)
        delta = (self._cfg.w_percent / 100.0) * abs(cost)
        if delta <= 0.0:
            return max(self._cfg.temperature_min, 50.0)
        return max(self._cfg.temperature_min, -delta / math.log(0.5))

    def _sample_quota(self, adaptive_bonus: int = 0) -> int:
        """
        采样移除客户数量 β。

        基础公式: β ∈ [max{3, r_L|C|}, min{30, r_U|C|}]

        其中:
        - |C| = self._n_customers (客户总数)
        - r_L = self._cfg.r_lower (移除比例下限, 默认0.15)
        - r_U = self._cfg.r_upper() (移除比例上限, 小规模0.5/大规模0.3)
        - 30 = self._cfg.quota_base_cap (绝对上限)

        在重热(reheat)阶段，配额会动态放大以增加探索性。
        """
        # 基础下限: max{3, r_L|C|}
        lower = max(3, int(self._cfg.r_lower * self._n_customers))

        # 基础上限: min{quota_base_cap, r_U|C|}
        base_cap = max(1, self._cfg.quota_base_cap)  # 默认30
        base_upper = min(base_cap, int(
            self._cfg.r_upper() * self._n_customers))
        dynamic_upper = int(round(self._quota_upper_ratio * self._n_customers))
        upper = base_upper
        if self._quota_scale > 1.0 or dynamic_upper > base_upper:
            upper = max(base_upper, dynamic_upper)
            absolute_cap = self._cfg.reheat_quota_absolute_cap
            if absolute_cap is not None:
                upper = min(upper, absolute_cap)
        if upper < lower:
            upper = lower
        upper = max(lower, upper)

        # 应用自适应配额奖励
        if adaptive_bonus > 0:
            upper = min(upper + adaptive_bonus, self._n_customers)

        quota = self._rng.randint(lower, upper)
        if self._quota_scale != 1.0:
            scaled = int(round(quota * self._quota_scale))
            cap = dynamic_upper if dynamic_upper > 0 else upper
            absolute_cap = self._cfg.reheat_quota_absolute_cap
            if absolute_cap is not None:
                cap = min(cap, absolute_cap)
            cap = max(lower, cap)
            quota = max(lower, min(cap, scaled))
        return quota

    def _shake(self, solution: Solution) -> Optional[Solution]:
        shake_fraction = max(0.0, min(1.0, self._cfg.reheat_shake_fraction))
        quota = max(1, int(round(shake_fraction * self._n_customers)))
        if quota <= 0:
            return None

        destroy_op = self._shake_destroy
        repair_op = self._shake_repair
        try:
            shaken_base = solution.clone()
            destroyed, pool = destroy_op.apply(shaken_base, quota)
            shaken = repair_op.apply(destroyed, pool.customers)
            return shaken
        except Exception:
            return None

    def _print_operator_snapshot(self, iteration: int) -> None:
        snapshot = self._aos.snapshot()
        print(f"Operator weights at iteration {iteration}:")
        for kind, entries in snapshot.items():
            summary = ", ".join(
                f"{entry['name']}={entry['weight']:.2f}" for entry in entries
            )
            print(f"  {kind}: {summary}")

    # ========== Local Search Methods ==========

    def _local_search(self, solution: Solution) -> Solution:
        """Apply local search to improve solution quality.

        Combines multiple improvement strategies inspired by Tabu Search:
        1. Drone-truck swap (exchange customers between drone and truck)
        2. Route merging (convert multi-truck to single truck)
        3. Aggressive depot drone tasks  
        4. Truck-launched drone tasks
        5. Full route optimization (2-opt + Or-opt + best insertion)
        6. Drone task optimization
        """
        improved = solution.clone()

        # Step 0: Try swapping customers between drone and truck
        improved = self._try_drone_truck_swap(improved)

        # Step 1: Try to merge truck routes into one
        improved = self._try_merge_truck_routes(improved)

        # Step 2: Aggressively try depot-launched drone tasks
        for _ in range(3):
            improved = self._try_depot_drone_tasks(improved)

        # Step 2.5: Try truck-launched drone tasks
        improved = self._try_truck_launched_drone(improved)

        # Step 3: Full truck route optimization
        improved = self._optimize_truck_route(improved)

        # Step 4: Optimize drone tasks (add more customers)
        improved = self._optimize_drone_tasks(improved)

        # Step 5: Mobile Hub Optimization
        improved = self._optimize_mobile_hub(improved)

        # Step 6: Drone task split/merge/reanchor local search
        if self._cfg.drone_reanchor_ls_enabled:
            improved = self._optimize_drone_reanchor(improved)

        # Step 7: Composite drone task re-anchor destroy-repair
        if self._cfg.drone_composite_reanchor_enabled:
            improved = self._composite_drone_reanchor(improved)

        # Step 8: Multi-customer sortie constructor (with composite acceptance)
        if self._cfg.drone_sortie_constructor_enabled:
            improved = self._construct_multi_customer_sorties(improved)

        # Step 10: Mini-MILP truck route polish — REMOVED from per-iteration LS.
        # Run as final polish only via run_with_milp_polish() instead.

        return improved

    def _optimize_mobile_hub(self, solution: Solution) -> Solution:
        """Mobile Hub Optimization: Offload truck segments to drones.

        Identify segments A -> B -> ... -> C in truck route where B... can be
        served by a drone launching at A and retrieving at C.
        """
        truck_dist = self._evaluator._instance.distance_matrix('truck')
        drone_dist = self._evaluator._instance.distance_matrix('drone')
        node_index = self._build_node_index()
        demands = self._evaluator._instance.customer_manager.demands()
        drone_cap = self._evaluator._instance.vehicle_specs['drone'].capacity
        drone_endurance = self._evaluator._instance.vehicle_specs['drone'].endurance
        drone_speed = self._evaluator._instance.vehicle_specs['drone'].speed
        drone_count = self._evaluator._instance.vehicle_specs['drone'].number

        best_solution = solution
        best_cost = self._evaluator.evaluate_solution(solution).total_cost
        if not math.isfinite(best_cost):
            return solution

        # Identify available drones
        used_drones = {task.drone_id for task in solution.drone_tasks}
        available_drones = [d for d in range(
            drone_count) if d not in used_drones]

        if not available_drones:
            # Can we repurpose a drone? Maybe later.
            return solution

        for route in solution.truck_routes:
            nodes = route.nodes
            if len(nodes) < 4:  # Need at least Launch -> Cust -> Retrieve
                continue

            # Try to find a segment to offload
            # A (Launch) -> [M1, M2...] (Drone) -> B (Retrieve)
            # Truck goes A -> B directly

            for i in range(len(nodes) - 2):
                launch_node = nodes[i]

                # Look ahead for retrieval node
                # We can skip up to 5 customers
                for j in range(i + 2, min(i + 7, len(nodes))):
                    retrieve_node = nodes[j]

                    # Candidates to offload
                    middle_nodes = nodes[i+1:j]

                    # Check capacity
                    total_demand = sum(demands.get(c, 0) for c in middle_nodes)
                    if total_demand > drone_cap:
                        continue

                    # Check drone feasibility
                    drone_dist_val = self._calc_drone_distance(
                        launch_node, middle_nodes, retrieve_node, drone_dist, node_index)
                    est_time = drone_dist_val / \
                        drone_speed + len(middle_nodes) * 0.1

                    if est_time > drone_endurance:
                        continue

                    # Check if truck distance reduction is worth it
                    # Truck currently: A -> M1 -> ... -> B
                    # Truck proposed: A -> B
                    current_truck_dist = 0.0
                    for k in range(i, j):
                        n1, n2 = nodes[k], nodes[k+1]
                        current_truck_dist += truck_dist[node_index[n1]
                                                         ][node_index[n2]]

                    proposed_truck_dist = truck_dist[node_index[launch_node]
                                                     ][node_index[retrieve_node]]

                    # Heuristic: If truck saves significantly OR drone is very efficient
                    if proposed_truck_dist < current_truck_dist * 0.8 or est_time < drone_endurance * 0.5:
                        # Try this move
                        cand = solution.clone()

                        # Modify truck route: remove middle nodes
                        # Find the route in candidate (by ID matches usually safe, but index is safer)
                        # We assume 1 truck for now or find by ID
                        cand_route = None
                        for r in cand.truck_routes:
                            if r.id == route.id:
                                cand_route = r
                                break

                        if not cand_route:
                            continue

                        # Safe removal
                        new_nodes = []
                        skip = False
                        for n in cand_route.nodes:
                            if n == launch_node:
                                new_nodes.append(n)
                                skip = True  # Skip middle
                            elif n == retrieve_node:
                                new_nodes.append(n)
                                skip = False  # Resume
                            elif not skip:
                                new_nodes.append(n)
                            # If skip is True, we ignore n (middle nodes)

                        cand_route.nodes = new_nodes

                        # Add drone task
                        from alns_vrpfd.model.route import DroneTask
                        task_id = max((t.task_id or 0)
                                      for t in cand.drone_tasks) + 1 if cand.drone_tasks else 1

                        # Calculate payloads
                        payloads = self._build_payloads(middle_nodes)

                        new_task = DroneTask(
                            task_id=task_id,
                            drone_id=available_drones[0],
                            launch_truck=route.id,
                            launch_node=launch_node,
                            customers=middle_nodes,
                            land_truck=route.id,
                            retrieve_node=retrieve_node,
                            payloads=payloads,
                        )
                        cand.drone_tasks.append(new_task)

                        try:
                            cand_eval = self._evaluator.evaluate_solution(cand)
                            if math.isfinite(cand_eval.total_cost) and cand_eval.total_cost < best_cost - 1e-6:
                                best_cost = cand_eval.total_cost
                                best_solution = cand
                                # Use up a drone?
                                # For simplicity, just return first improvement or keep searching?
                                # If we return, we restart search from improved solution next time.
                                return cand
                        except Exception:
                            continue

        return best_solution

    def _optimize_drone_reanchor(self, solution: Solution) -> Solution:
        """Apply drone task split/merge/reanchor local search."""
        from alns_vrpfd.core.operators.drone_reanchor import DroneTaskSplitMergeLocalSearch

        ls = DroneTaskSplitMergeLocalSearch(
            instance=self._evaluator._instance,
            evaluator=self._evaluator,
            max_moves=self._cfg.drone_reanchor_ls_max_moves,
        )
        return ls.apply(solution)

    def _composite_drone_reanchor(self, solution: Solution) -> Solution:
        """Composite drone re-anchor: dissolve 1-3 drone tasks and re-insert freely.

        This explores neighborhoods that single-customer ALNS moves cannot reach,
        e.g. creating multi-customer drone sorties with cross-truck retrieval.
        """
        from alns_vrpfd.core.operators.drone_reanchor import DroneTaskReanchorRepair

        if not solution.drone_tasks:
            return solution

        repair_ops = self._repair_ops if self._repair_ops else []
        if not repair_ops:
            return solution

        composite = DroneTaskReanchorRepair(
            instance=self._evaluator._instance,
            repair_operators=repair_ops,
            evaluator=self._evaluator,
            max_tasks=self._cfg.drone_composite_reanchor_max_tasks,
            rng=self._rng,
        )
        return composite.apply(solution)

    def _apply_matheuristic_lns(self, solution: Solution) -> Solution:
        """Exhaustive drone sortie reconstruction (Step 11 - Matheuristic LNS).

        Selects a critical neighborhood of 2-4 drone-served customers,
        removes them, and exhaustively enumerates all possible drone
        sortie assignments to find the optimal local structure.
        """
        from alns_vrpfd.core.operators.matheuristic_lns import MatheuristicLNSRepair

        lns = MatheuristicLNSRepair(
            instance=self._evaluator._instance,
            evaluator=self._evaluator,
            max_customers=self._cfg.matheuristic_lns_max_customers,
            rng=self._rng,
        )
        return lns.apply(solution)

    def _construct_multi_customer_sorties(self, solution: Solution) -> Solution:
        """Construct multi-customer cross-truck drone sorties (Step 8)."""
        from alns_vrpfd.core.operators.drone_reanchor import MultiCustomerSortieConstructor

        constructor = MultiCustomerSortieConstructor(
            instance=self._evaluator._instance,
            evaluator=self._evaluator,
            max_customers=self._cfg.drone_sortie_max_customers,
            top_k=self._cfg.drone_sortie_top_k,
            max_sorties=self._cfg.drone_sortie_max_sorties,
            rng=self._rng,
            polish_after=self._cfg.drone_sortie_constructor_enabled,
            reanchor_after=self._cfg.drone_reanchor_ls_enabled,
        )
        return constructor.apply(solution)

    def _mini_milp_polish(self, solution: Solution) -> Solution:
        """Polish truck routes using mini-MILP (Step 10).

        Fixes drone task assignments and re-optimizes truck customer
        visit ordering and assignment via Gurobi. This is an exact
        polishing step that can close the gap to the full MILP optimum
        that ALNS local search cannot reach.
        """
        from alns_vrpfd.core.operators.mini_milp_polish import polish_with_mini_milp

        result = polish_with_mini_milp(
            solution=solution,
            instance=self._evaluator._instance,
            evaluator=self._evaluator,
            time_limit=self._cfg.mini_milp_polish_time_limit,
            mip_gap=self._cfg.mini_milp_polish_gap,
            verbose=False,
        )
        if result.improved:
            logger.info(
                f"Mini-MILP polish improved: {result.original_cost:.2f} -> "
                f"{result.polished_cost:.2f} (delta={result.improvement:.2f}, "
                f"time={result.runtime_seconds:.1f}s)"
            )
            return result.polished_solution
        return solution

    def _synchronized_truck_polish(self, solution: Solution) -> Solution:
        """Synchronized truck-route polish after drone sortie changes (Step 9)."""
        from alns_vrpfd.core.operators.truck_route_polish import SynchronizedTruckRoutePolish

        polisher = SynchronizedTruckRoutePolish(
            instance=self._evaluator._instance,
            evaluator=self._evaluator,
            max_iterations=20,
        )
        result = polisher.apply(solution)
        # Also re-run drone reanchor LS after polishing
        if self._cfg.drone_reanchor_ls_enabled:
            from alns_vrpfd.core.operators.drone_reanchor import DroneTaskSplitMergeLocalSearch
            ls = DroneTaskSplitMergeLocalSearch(
                instance=self._evaluator._instance,
                evaluator=self._evaluator,
                max_moves=5,
            )
            result = ls.apply(result)
        return result

    def _try_drone_truck_swap(self, solution: Solution) -> Solution:
        """Try swapping customers between drone tasks and truck routes.

        This explores different delivery mode assignments to find better solutions.
        """
        if not solution.drone_tasks or not solution.truck_routes:
            return solution

        truck_dist = self._evaluator._instance.distance_matrix('truck')
        drone_dist = self._evaluator._instance.distance_matrix('drone')
        node_index = self._build_node_index()
        demands = self._evaluator._instance.customer_manager.demands()
        drone_cap = self._evaluator._instance.vehicle_specs['drone'].capacity
        drone_endurance = self._evaluator._instance.vehicle_specs['drone'].endurance
        drone_speed = self._evaluator._instance.vehicle_specs['drone'].speed
        drone_count = self._evaluator._instance.vehicle_specs['drone'].number

        # Get current solution cost
        current_eval = self._evaluator.evaluate_solution(solution)
        if not math.isfinite(current_eval.total_cost):
            return solution
        best_cost = current_eval.total_cost
        best_solution = solution

        # Get drone customers
        drone_customers = []
        for task in solution.drone_tasks:
            drone_customers.extend(task.customers())

        # Get truck customers that are drone-eligible
        truck_drone_eligible = []
        for route in solution.truck_routes:
            for c in route.customers():
                if demands.get(c, float('inf')) <= drone_cap:
                    truck_drone_eligible.append(c)

        # Try swapping: move drone customer to truck, move truck customer to drone
        for drone_cust in drone_customers[:5]:  # Limit to 5 for speed
            for truck_cust in truck_drone_eligible[:10]:  # Limit to 10
                if drone_cust == truck_cust:
                    continue

                # Create candidate solution
                cand = solution.clone()

                # 1. Remove drone_cust from drone tasks
                for task in cand.drone_tasks:
                    if drone_cust in task.customers():
                        new_custs = [
                            c for c in task.customers() if c != drone_cust]
                        if new_custs:
                            task.nodes = [task.launch_node] + \
                                new_custs + [task.retrieve_node]
                        else:
                            cand.drone_tasks.remove(task)
                        break

                # 2. Add drone_cust to truck
                for route in cand.truck_routes:
                    if drone_cust in route.nodes:
                        continue  # Prevent duplicates

                    if route.customers():  # Add to first non-empty route
                        # Find best insertion position
                        best_pos = 1
                        best_delta = float('inf')
                        for pos in range(1, len(route.nodes)):
                            prev = route.nodes[pos - 1]
                            next_n = route.nodes[pos]
                            idx_p = node_index.get(prev, -1)
                            idx_c = node_index.get(drone_cust, -1)
                            idx_n = node_index.get(next_n, -1)
                            if idx_p >= 0 and idx_c >= 0 and idx_n >= 0:
                                delta = truck_dist[idx_p][idx_c] + \
                                    truck_dist[idx_c][idx_n] - \
                                    truck_dist[idx_p][idx_n]
                                if delta < best_delta:
                                    best_delta = delta
                                    best_pos = pos
                        route.nodes.insert(best_pos, drone_cust)
                        break

                # 3. Remove truck_cust from truck
                for route in cand.truck_routes:
                    if truck_cust in route.nodes:
                        route.nodes = [
                            n for n in route.nodes if n != truck_cust]
                        break

                # 4. Create depot drone task for truck_cust
                used_drones = {task.drone_id for task in cand.drone_tasks}
                available_drone = None
                for d in range(drone_count):
                    if d not in used_drones:
                        available_drone = d
                        break

                if available_drone is None:
                    continue  # No drone available

                depot_start = self._instance.customer_manager.depot_start
                depot_end = self._instance.customer_manager.depot_end

                # Check if drone can serve truck_cust
                dist = self._calc_drone_distance(
                    depot_start, [truck_cust], depot_end, drone_dist, node_index)
                est_time = dist / drone_speed + 0.1
                if est_time > drone_endurance:
                    continue  # Infeasible

                from alns_vrpfd.model.route import DroneTask
                task_id = max((t.task_id or 0)
                              for t in cand.drone_tasks) + 1 if cand.drone_tasks else 1
                # Calculate payloads
                payloads = self._build_payloads([truck_cust])

                new_task = DroneTask(
                    task_id=task_id,
                    drone_id=available_drone,
                    launch_truck=None,
                    launch_node=depot_start,
                    customers=[truck_cust],
                    land_truck=None,
                    retrieve_node=depot_end,
                    payloads=payloads,
                )
                cand.drone_tasks.append(new_task)

                # Evaluate candidate
                try:
                    cand_eval = self._evaluator.evaluate_solution(cand)
                    if math.isfinite(cand_eval.total_cost) and cand_eval.total_cost < best_cost - 1e-6:
                        best_cost = cand_eval.total_cost
                        best_solution = cand
                except Exception:
                    continue

        return best_solution

    def _try_merge_truck_routes(self, solution: Solution) -> Solution:
        """Try merging multiple truck routes into one if beneficial."""
        if len(solution.truck_routes) <= 1:
            return solution

        truck_dist = self._evaluator._instance.distance_matrix('truck')
        node_index = self._build_node_index()
        demands = self._evaluator._instance.customer_manager.demands()
        truck_cap = self._evaluator._instance.vehicle_specs['truck'].capacity

        # Calculate current total truck distance
        current_total = sum(
            self._route_distance(r.nodes, truck_dist, node_index)
            for r in solution.truck_routes
        )

        # Get all truck customers
        all_customers = []
        for route in solution.truck_routes:
            all_customers.extend(route.customers())

        # Check capacity constraint
        total_demand = sum(demands.get(c, 0) for c in all_customers)
        if total_demand > truck_cap:
            return solution  # Cannot merge due to capacity

        depot_start = solution.truck_routes[0].nodes[0]
        depot_end = solution.truck_routes[0].nodes[-1]

        # Try nearest neighbor construction for merged route
        best_nodes = None
        best_dist = current_total

        for start_cust in all_customers[:min(5, len(all_customers))]:
            remaining = set(all_customers) - {start_cust}
            order = [start_cust]
            current = start_cust

            while remaining:
                nearest = None
                nearest_dist = float('inf')

                idx_curr = node_index.get(current, -1)
                for cand in remaining:
                    idx_cand = node_index.get(cand, -1)
                    if idx_curr >= 0 and idx_cand >= 0:
                        d = truck_dist[idx_curr][idx_cand]
                        if d < nearest_dist:
                            nearest_dist = d
                            nearest = cand

                if nearest is None:
                    break

                order.append(nearest)
                remaining.remove(nearest)
                current = nearest

            if len(order) == len(all_customers):
                merged_nodes = [depot_start] + order + [depot_end]
                merged_dist = self._route_distance(
                    merged_nodes, truck_dist, node_index)

                if merged_dist < best_dist - 1e-6:
                    best_dist = merged_dist
                    best_nodes = merged_nodes

        # Also try 2-opt on the naive merge
        naive_nodes = [depot_start] + all_customers + [depot_end]
        naive_dist = self._route_distance(naive_nodes, truck_dist, node_index)
        if naive_dist < best_dist - 1e-6:
            best_dist = naive_dist
            best_nodes = naive_nodes

        if best_nodes:
            # Create merged solution
            from alns_vrpfd.model import TruckRoute
            new_route = TruckRoute(
                route_id=solution.truck_routes[0].id,
                nodes=best_nodes,
                capacity=truck_cap
            )
            solution.truck_routes = [new_route]

            # Fix drone task references to use the merged route's ID
            merged_id = new_route.id
            for task in solution.drone_tasks:
                if task.launch_truck is not None:
                    task.launch_truck = merged_id
                if task.land_truck is not None:
                    task.land_truck = merged_id

        return solution

    def _optimize_truck_route(self, solution: Solution) -> Solution:
        """Comprehensive truck route optimization."""
        truck_dist = self._evaluator._instance.distance_matrix('truck')
        node_index = self._build_node_index()

        for route in solution.truck_routes:
            customers = route.customers()
            if len(customers) < 2:
                continue

            # Phase 1: 2-opt until no improvement
            improved = True
            max_iterations = 100
            iteration = 0
            while improved and iteration < max_iterations:
                improved = False
                iteration += 1
                nodes = route.nodes

                for i in range(1, len(nodes) - 2):
                    for j in range(i + 2, len(nodes) - 1):
                        c1, c2 = nodes[i], nodes[i + 1]
                        c3, c4 = nodes[j], nodes[j + 1]

                        idx1 = node_index.get(c1, -1)
                        idx2 = node_index.get(c2, -1)
                        idx3 = node_index.get(c3, -1)
                        idx4 = node_index.get(c4, -1)

                        if any(idx < 0 for idx in [idx1, idx2, idx3, idx4]):
                            continue

                        current_dist = truck_dist[idx1][idx2] + \
                            truck_dist[idx3][idx4]
                        new_dist = truck_dist[idx1][idx3] + \
                            truck_dist[idx2][idx4]

                        if new_dist < current_dist - 1e-6:
                            new_nodes = nodes[:i + 1] + \
                                nodes[i + 1:j + 1][::-1] + nodes[j + 1:]
                            route.nodes = new_nodes
                            improved = True
                            break
                    if improved:
                        break

            # Phase 2: Or-opt (relocate single customers)
            improved = True
            iteration = 0
            while improved and iteration < max_iterations:
                improved = False
                iteration += 1
                nodes = route.nodes
                best_gain = 0
                best_move = None

                for i in range(1, len(nodes) - 1):  # Customer positions
                    cust = nodes[i]
                    prev = nodes[i - 1]
                    next_node = nodes[i + 1]

                    # Current cost of customer in position
                    idx_prev = node_index.get(prev, -1)
                    idx_cust = node_index.get(cust, -1)
                    idx_next = node_index.get(next_node, -1)

                    if any(idx < 0 for idx in [idx_prev, idx_cust, idx_next]):
                        continue

                    removal_cost = truck_dist[idx_prev][idx_cust] + \
                        truck_dist[idx_cust][idx_next]
                    reconnect_cost = truck_dist[idx_prev][idx_next]

                    # Try inserting at other positions
                    for j in range(1, len(nodes) - 1):
                        if j == i or j == i - 1 or j == i + 1:
                            continue

                        insert_prev = nodes[j - 1]
                        insert_next = nodes[j]

                        idx_ins_prev = node_index.get(insert_prev, -1)
                        idx_ins_next = node_index.get(insert_next, -1)

                        if idx_ins_prev < 0 or idx_ins_next < 0:
                            continue

                        old_insert_cost = truck_dist[idx_ins_prev][idx_ins_next]
                        new_insert_cost = truck_dist[idx_ins_prev][idx_cust] + \
                            truck_dist[idx_cust][idx_ins_next]

                        gain = (removal_cost - reconnect_cost) - \
                            (new_insert_cost - old_insert_cost)

                        if gain > best_gain + 1e-6:
                            best_gain = gain
                            best_move = (i, j)

                if best_move:
                    i, j = best_move
                    cust = nodes[i]
                    new_nodes = nodes[:i] + nodes[i + 1:]  # Remove
                    insert_pos = j if j < i else j - 1
                    new_nodes.insert(insert_pos, cust)  # Insert
                    route.nodes = new_nodes
                    improved = True

            # Phase 3: Try nearest neighbor reconstruction from scratch
            if len(route.customers()) >= 4:
                route = self._try_nearest_neighbor(
                    route, truck_dist, node_index)

        return solution

    def _try_nearest_neighbor(self, route, truck_dist, node_index):
        """Try nearest neighbor heuristic and keep if better."""
        customers = list(route.customers())
        if len(customers) < 4:
            return route

        # Current distance
        current_dist = self._route_distance(
            route.nodes, truck_dist, node_index)

        # Try NN from each customer as starting point
        best_dist = current_dist
        best_order = None
        depot_start = route.nodes[0]
        depot_end = route.nodes[-1]

        for start_cust in customers[:5]:  # Try first 5 as potential starts
            remaining = set(customers) - {start_cust}
            order = [start_cust]
            current = start_cust

            while remaining:
                # Find nearest unvisited
                nearest = None
                nearest_dist = float('inf')

                idx_curr = node_index.get(current, -1)
                for cand in remaining:
                    idx_cand = node_index.get(cand, -1)
                    if idx_curr >= 0 and idx_cand >= 0:
                        d = truck_dist[idx_curr][idx_cand]
                        if d < nearest_dist:
                            nearest_dist = d
                            nearest = cand

                if nearest is None:
                    break

                order.append(nearest)
                remaining.remove(nearest)
                current = nearest

            if len(order) == len(customers):
                nn_nodes = [depot_start] + order + [depot_end]
                nn_dist = self._route_distance(
                    nn_nodes, truck_dist, node_index)

                if nn_dist < best_dist - 1e-6:
                    best_dist = nn_dist
                    best_order = nn_nodes

        if best_order:
            route.nodes = best_order

        return route

    def _route_distance(self, nodes, truck_dist, node_index):
        """Calculate total route distance."""
        total = 0.0
        for i in range(len(nodes) - 1):
            idx1 = node_index.get(nodes[i], -1)
            idx2 = node_index.get(nodes[i + 1], -1)
            if idx1 >= 0 and idx2 >= 0:
                total += truck_dist[idx1][idx2]
        return total

    def _apply_2opt(self, solution: Solution) -> Solution:
        """Apply 2-opt improvement to truck routes."""
        truck_dist = self._evaluator._instance.distance_matrix('truck')
        node_index = self._build_node_index()

        for route in solution.truck_routes:
            customers = route.customers()  # Use customers() method, excludes depot
            if len(customers) < 3:
                continue

            improved = True
            while improved:
                improved = False
                nodes = route.nodes  # Full nodes including depot

                for i in range(1, len(nodes) - 2):  # Skip depot at start
                    for j in range(i + 2, len(nodes) - 1):  # Skip depot at end
                        # Calculate current distance
                        c1, c2 = nodes[i], nodes[i + 1]
                        c3, c4 = nodes[j], nodes[j + 1]

                        idx1 = node_index.get(c1, -1)
                        idx2 = node_index.get(c2, -1)
                        idx3 = node_index.get(c3, -1)
                        idx4 = node_index.get(c4, -1)

                        if any(idx < 0 for idx in [idx1, idx2, idx3, idx4]):
                            continue

                        current_dist = truck_dist[idx1][idx2] + \
                            truck_dist[idx3][idx4]
                        new_dist = truck_dist[idx1][idx3] + \
                            truck_dist[idx2][idx4]

                        if new_dist < current_dist - 1e-6:
                            # Reverse segment between i+1 and j
                            new_nodes = nodes[:i + 1] + \
                                nodes[i + 1:j + 1][::-1] + nodes[j + 1:]
                            route.nodes = new_nodes
                            improved = True
                            break
                    if improved:
                        break

        return solution

    def _build_node_index(self) -> dict:
        """Build mapping from node ID to matrix index."""
        node_ids = self._evaluator._instance.all_node_ids()
        return {node: idx for idx, node in enumerate(node_ids)}

    def _build_payloads(self, customers: list) -> list:
        """Build payloads list for a sequence of customers.

        Payloads start with total demand and decrease after each customer.
        Returns list of length len(customers) + 1.
        """
        demands = self._evaluator._instance.customer_manager.demands()
        payloads = []
        remaining = sum(demands.get(c, 0) for c in customers)
        payloads.append(remaining)
        for c in customers:
            remaining -= demands.get(c, 0)
            payloads.append(max(remaining, 0.0))
        return payloads

    def _optimize_drone_tasks(self, solution: Solution) -> Solution:
        """Optimize drone tasks by converting single to multi-customer where possible."""
        if not solution.drone_tasks:
            return solution

        drone_dist = self._evaluator._instance.distance_matrix('drone')
        node_index = self._build_node_index()
        demands = self._evaluator._instance.customer_manager.demands()
        drone_cap = self._evaluator._instance.vehicle_specs['drone'].capacity
        drone_endurance = self._evaluator._instance.vehicle_specs['drone'].endurance
        drone_speed = self._evaluator._instance.vehicle_specs['drone'].speed

        # Get drone-eligible customers on truck routes
        truck_customers = set()
        for route in solution.truck_routes:
            for cust in route.customers():  # Use customers() method
                if demands.get(cust, float('inf')) <= drone_cap:
                    truck_customers.add(cust)

        # Try to expand single-customer drone tasks
        for task in solution.drone_tasks:
            if len(task.customers()) >= 2:  # Use customers() method
                continue

            launch = task.launch_node
            retrieve = task.retrieve_node
            current_custs = list(task.customers())  # Use customers() method

            # Find nearby customers that could be added
            candidates = [
                c for c in truck_customers if c not in set(current_custs)]

            for cand in candidates[:5]:  # Try up to 5 candidates
                test_custs = current_custs + [cand]

                # Check capacity
                total_demand = sum(demands.get(c, 0) for c in test_custs)
                if total_demand > drone_cap:
                    continue

                # Check time/distance
                total_dist = self._calc_drone_distance(
                    launch, test_custs, retrieve, drone_dist, node_index)
                est_time = total_dist / drone_speed + len(test_custs) * 0.1

                if est_time <= drone_endurance:
                    # Update task nodes: launch -> customers -> retrieve
                    task.nodes = [launch] + test_custs + [retrieve]
                    self._remove_customer_from_truck(solution, cand)
                    truck_customers.discard(cand)
                    current_custs = test_custs
                    break

        return solution

    def _calc_drone_distance(self, launch: int, customers: list, retrieve: int, dist_matrix, node_index) -> float:
        """Calculate total drone distance."""
        nodes = [launch] + customers + [retrieve]
        total = 0.0
        for i in range(len(nodes) - 1):
            i_idx = node_index.get(nodes[i], -1)
            j_idx = node_index.get(nodes[i + 1], -1)
            if i_idx < 0 or j_idx < 0:
                return float('inf')
            total += dist_matrix[i_idx][j_idx]
        return total

    def _remove_customer_from_truck(self, solution: Solution, customer_id: int) -> None:
        """Remove a customer from truck routes."""
        for route in solution.truck_routes:
            route.nodes = [n for n in route.nodes if n != customer_id]

    def _try_depot_drone_tasks(self, solution: Solution) -> Solution:
        """Try creating depot-launched drone tasks (0 -> customers -> 11).

        Enhanced version that evaluates cost reduction before creating tasks.
        Iteratively attempts to assign tasks to available drones.
        """
        demands = self._evaluator._instance.customer_manager.demands()
        drone_cap = self._evaluator._instance.vehicle_specs['drone'].capacity
        drone_endurance = self._evaluator._instance.vehicle_specs['drone'].endurance
        drone_speed = self._evaluator._instance.vehicle_specs['drone'].speed
        drone_count = self._evaluator._instance.vehicle_specs['drone'].number

        drone_dist = self._evaluator._instance.distance_matrix('drone')
        truck_dist = self._evaluator._instance.distance_matrix('truck')
        node_index = self._build_node_index()

        # Work on a clone that we update iteratively
        current_solution = solution.clone()

        # Get current cost baseline
        current_eval = self._evaluator.evaluate_solution(current_solution)
        if not math.isfinite(current_eval.total_cost):
            return solution
        current_cost = current_eval.total_cost

        # Keep track of improvements
        any_improvement = False

        # Get eligible customers on truck routes
        # We need to refresh this list as we remove customers
        def get_truck_customers(sol):
            custs = []
            for route in sol.truck_routes:
                for cust in route.customers():
                    if demands.get(cust, float('inf')) <= drone_cap:
                        custs.append(cust)
            return custs

        # Try to fill available drones one by one
        # We iterate up to drone_count times
        for _ in range(drone_count):
            used_drones = {
                task.drone_id for task in current_solution.drone_tasks}
            available_drone = None
            for d in range(drone_count):
                if d not in used_drones:
                    available_drone = d
                    break

            if available_drone is None:
                break

            truck_customers = get_truck_customers(current_solution)
            if not truck_customers:
                break

            best_task_move = None  # (task_custs, score, dist, est_time)
            best_score = 0.0

            # Evaluate candidates for this drone
            candidates = []

            depot_start = self._instance.customer_manager.depot_start
            depot_end = self._instance.customer_manager.depot_end

            # Try single customers
            for cust in truck_customers[:15]:
                dist = self._calc_drone_distance(
                    depot_start, [cust], depot_end, drone_dist, node_index)
                est_time = dist / drone_speed + 0.1

                if est_time > drone_endurance:
                    continue

                truck_saved = self._truck_distance_saved(
                    current_solution, [cust], truck_dist, node_index)

                utilization = est_time / drone_endurance
                score = truck_saved - dist + \
                    (2.0 if utilization > 0.6 else 0.5)
                candidates.append(([cust], score, dist, est_time))

            # Try two customers
            for i, c1 in enumerate(truck_customers[:12]):
                for c2 in truck_customers[i + 1:15]:
                    total_demand = demands.get(c1, 0) + demands.get(c2, 0)
                    if total_demand > drone_cap:
                        continue

                    for custs in [[c1, c2], [c2, c1]]:
                        dist = self._calc_drone_distance(
                            depot_start, custs, depot_end, drone_dist, node_index)
                        est_time = dist / drone_speed + 0.2

                        if est_time > drone_endurance:
                            continue

                        truck_saved = self._truck_distance_saved(
                            current_solution, custs, truck_dist, node_index)
                        utilization = est_time / drone_endurance
                        score = truck_saved - dist + \
                            (3.0 if utilization > 0.6 else 1.0)
                        candidates.append((custs, score, dist, est_time))

            # Sort and pick best
            candidates.sort(key=lambda x: x[1], reverse=True)

            # Try the best valid move
            for task_custs, score, dist, est_time in candidates[:3]:
                # Apply to a temp clone to verify cost
                cand = current_solution.clone()

                for c in task_custs:
                    self._remove_customer_from_truck(cand, c)

                from alns_vrpfd.model.route import DroneTask
                task_id = max((t.task_id or 0)
                              for t in cand.drone_tasks) + 1 if cand.drone_tasks else 1

                # Calculate payloads
                payloads = self._build_payloads(task_custs)

                new_task = DroneTask(
                    task_id=task_id,
                    drone_id=available_drone,
                    launch_truck=None,
                    launch_node=depot_start,
                    customers=task_custs,
                    land_truck=None,
                    retrieve_node=depot_end,
                    payloads=payloads,
                )
                cand.drone_tasks.append(new_task)

                try:
                    cand_eval = self._evaluator.evaluate_solution(cand)
                    if math.isfinite(cand_eval.total_cost) and cand_eval.total_cost < current_cost - 1e-6:
                        current_solution = cand
                        current_cost = cand_eval.total_cost
                        any_improvement = True
                        break  # Move to next drone
                except Exception:
                    continue

        return current_solution if any_improvement else solution

    def _truck_distance_saved(self, solution: Solution, customers: list, truck_dist, node_index) -> float:
        """Calculate truck distance saved by removing customers."""
        saved = 0.0
        for route in solution.truck_routes:
            nodes = route.nodes
            for cust in customers:
                if cust in nodes:
                    idx = nodes.index(cust)
                    if idx > 0 and idx < len(nodes) - 1:
                        prev = nodes[idx - 1]
                        next_n = nodes[idx + 1]
                        idx_p = node_index.get(prev, -1)
                        idx_c = node_index.get(cust, -1)
                        idx_n = node_index.get(next_n, -1)
                        if idx_p >= 0 and idx_c >= 0 and idx_n >= 0:
                            # Distance removed - new direct distance
                            saved += truck_dist[idx_p][idx_c] + \
                                truck_dist[idx_c][idx_n] - \
                                truck_dist[idx_p][idx_n]
        return saved

    def _try_truck_launched_drone(self, solution: Solution) -> Solution:
        """Try creating truck-launched drone tasks for better coordination."""
        demands = self._evaluator._instance.customer_manager.demands()
        drone_cap = self._evaluator._instance.vehicle_specs['drone'].capacity
        drone_endurance = self._evaluator._instance.vehicle_specs['drone'].endurance
        drone_speed = self._evaluator._instance.vehicle_specs['drone'].speed
        drone_count = self._evaluator._instance.vehicle_specs['drone'].number

        drone_dist = self._evaluator._instance.distance_matrix('drone')
        node_index = self._build_node_index()

        # Get current cost
        current_eval = self._evaluator.evaluate_solution(solution)
        if not math.isfinite(current_eval.total_cost):
            return solution
        best_cost = current_eval.total_cost
        best_solution = solution

        # Get used drones
        used_drones = {task.drone_id for task in solution.drone_tasks}

        for route in solution.truck_routes:
            nodes = route.nodes
            if len(nodes) < 4:  # Need at least depot-cust1-cust2-depot
                continue

            # Try creating tasks from each position
            for launch_pos in range(len(nodes) - 2):
                launch_node = nodes[launch_pos]

                for retrieve_pos in range(launch_pos + 2, min(launch_pos + 5, len(nodes))):
                    retrieve_node = nodes[retrieve_pos]

                    # Get customers between launch and retrieve
                    task_custs = []
                    total_demand = 0
                    for mid_pos in range(launch_pos + 1, retrieve_pos):
                        cust = nodes[mid_pos]
                        if cust in (self._depot_start, self._depot_end):  # Skip depots
                            continue
                        demand = demands.get(cust, float('inf'))
                        if total_demand + demand > drone_cap:
                            break
                        task_custs.append(cust)
                        total_demand += demand

                    if not task_custs:
                        continue

                    # Check drone feasibility
                    dist = self._calc_drone_distance(
                        launch_node, task_custs, retrieve_node, drone_dist, node_index)
                    est_time = dist / drone_speed + len(task_custs) * 0.1

                    if est_time > drone_endurance:
                        continue

                    # Find available drone
                    available_drone = None
                    for d in range(drone_count):
                        if d not in used_drones:
                            available_drone = d
                            break

                    if available_drone is None:
                        continue

                    # Create candidate solution
                    cand = solution.clone()

                    # Remove customers from truck
                    cand_route = cand.truck_routes[solution.truck_routes.index(
                        route)]
                    cand_route.nodes = [
                        n for n in cand_route.nodes if n not in task_custs]

                    # Create drone task
                    from alns_vrpfd.model.route import DroneTask
                    task_id = max((t.task_id or 0)
                                  for t in cand.drone_tasks) + 1 if cand.drone_tasks else 1

                    # Calculate payloads
                    payloads = self._build_payloads(task_custs)

                    new_task = DroneTask(
                        task_id=task_id,
                        drone_id=available_drone,
                        launch_truck=route.id,
                        launch_node=launch_node,
                        customers=task_custs,
                        land_truck=route.id,
                        retrieve_node=retrieve_node,
                        payloads=payloads,
                    )
                    cand.drone_tasks.append(new_task)

                    try:
                        cand_eval = self._evaluator.evaluate_solution(cand)
                        if math.isfinite(cand_eval.total_cost) and cand_eval.total_cost < best_cost - 1e-6:
                            best_cost = cand_eval.total_cost
                            best_solution = cand
                    except Exception:
                        continue

        return best_solution

    # ========== Enhanced Local Search Methods ==========

    def _intensify_search(self, solution: Solution) -> Solution:
        """Full intensification search - more thorough than regular local search.

        Applies multiple neighborhood structures exhaustively:
        1. Cross-exchange between routes
        2. String relocation (move sequences)
        3. Complete route re-optimization
        4. Aggressive drone task creation
        """
        improved = solution.clone()

        # Full local search first
        improved = self._local_search(improved)

        # Cross-exchange between truck routes
        improved = self._cross_exchange(improved)

        # String relocation
        improved = self._string_relocate(improved)

        # Re-optimize all routes
        improved = self._optimize_truck_route(improved)

        # One more round of drone optimization
        for _ in range(2):
            improved = self._try_depot_drone_tasks(improved)

        return improved

    def _cross_exchange(self, solution: Solution) -> Solution:
        """Cross-exchange: swap segments between two routes."""
        if len(solution.truck_routes) < 2:
            return solution

        truck_dist = self._evaluator._instance.distance_matrix('truck')
        node_index = self._build_node_index()

        best_solution = solution
        best_cost = self._evaluator.evaluate_solution(solution).total_cost
        if not math.isfinite(best_cost):
            return solution

        routes = solution.truck_routes

        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                route1 = routes[i]
                route2 = routes[j]

                custs1 = route1.customers()
                custs2 = route2.customers()

                if not custs1 or not custs2:
                    continue

                # Try exchanging segments of length 1-2
                for seg_len1 in range(1, min(3, len(custs1) + 1)):
                    for start1 in range(len(custs1) - seg_len1 + 1):
                        seg1 = custs1[start1:start1 + seg_len1]

                        for seg_len2 in range(1, min(3, len(custs2) + 1)):
                            for start2 in range(len(custs2) - seg_len2 + 1):
                                seg2 = custs2[start2:start2 + seg_len2]

                                # Create new solution with exchanged segments
                                cand = solution.clone()

                                new_custs1 = custs1[:start1] + \
                                    list(seg2) + custs1[start1 + seg_len1:]
                                new_custs2 = custs2[:start2] + \
                                    list(seg1) + custs2[start2 + seg_len2:]

                                depot_start = route1.nodes[0]
                                depot_end = route1.nodes[-1]

                                cand.truck_routes[i].nodes = [
                                    depot_start] + new_custs1 + [depot_end]
                                cand.truck_routes[j].nodes = [
                                    depot_start] + new_custs2 + [depot_end]

                                try:
                                    cand_eval = self._evaluator.evaluate_solution(
                                        cand)
                                    if math.isfinite(cand_eval.total_cost) and cand_eval.total_cost < best_cost - 1e-6:
                                        best_cost = cand_eval.total_cost
                                        best_solution = cand
                                except Exception:
                                    continue

        return best_solution

    def _string_relocate(self, solution: Solution) -> Solution:
        """Relocate a string (consecutive sequence) of customers to another position."""
        if not solution.truck_routes:
            return solution

        truck_dist = self._evaluator._instance.distance_matrix('truck')
        node_index = self._build_node_index()

        best_solution = solution
        best_cost = self._evaluator.evaluate_solution(solution).total_cost
        if not math.isfinite(best_cost):
            return solution

        for route_idx, route in enumerate(solution.truck_routes):
            custs = route.customers()
            if len(custs) < 3:
                continue

            # Try relocating strings of length 2-3
            for string_len in range(2, min(4, len(custs))):
                for start in range(len(custs) - string_len + 1):
                    string = custs[start:start + string_len]
                    remaining = custs[:start] + custs[start + string_len:]

                    if not remaining:
                        continue

                    # Try inserting string at different positions in remaining
                    for insert_pos in range(len(remaining) + 1):
                        if insert_pos == start:
                            continue  # Same position

                        new_custs = remaining[:insert_pos] + \
                            string + remaining[insert_pos:]

                        cand = solution.clone()
                        depot_start = route.nodes[0]
                        depot_end = route.nodes[-1]
                        cand.truck_routes[route_idx].nodes = [
                            depot_start] + new_custs + [depot_end]

                        try:
                            cand_eval = self._evaluator.evaluate_solution(cand)
                            if math.isfinite(cand_eval.total_cost) and cand_eval.total_cost < best_cost - 1e-6:
                                best_cost = cand_eval.total_cost
                                best_solution = cand
                        except Exception:
                            continue

        return best_solution

    def _try_relocate_customer(self, solution: Solution) -> Solution:
        """Try relocating single customers to better positions (inter-route)."""
        if len(solution.truck_routes) < 2:
            return solution

        truck_dist = self._evaluator._instance.distance_matrix('truck')
        node_index = self._build_node_index()

        best_solution = solution
        best_cost = self._evaluator.evaluate_solution(solution).total_cost
        if not math.isfinite(best_cost):
            return solution

        routes = solution.truck_routes

        for src_idx in range(len(routes)):
            src_route = routes[src_idx]
            src_custs = src_route.customers()

            if not src_custs:
                continue

            for cust in src_custs[:5]:  # Limit for speed
                for dst_idx in range(len(routes)):
                    if src_idx == dst_idx:
                        continue

                    dst_route = routes[dst_idx]

                    # Find best insertion position
                    for insert_pos in range(1, len(dst_route.nodes)):
                        cand = solution.clone()

                        # Remove from source
                        cand.truck_routes[src_idx].nodes = [
                            n for n in cand.truck_routes[src_idx].nodes if n != cust
                        ]

                        # Insert to destination
                        cand.truck_routes[dst_idx].nodes.insert(
                            insert_pos, cust)

                        try:
                            cand_eval = self._evaluator.evaluate_solution(cand)
                            if math.isfinite(cand_eval.total_cost) and cand_eval.total_cost < best_cost - 1e-6:
                                best_cost = cand_eval.total_cost
                                best_solution = cand
                        except Exception:
                            continue

        return best_solution

    def _path_relinking(self, current: Solution, target: Solution) -> Solution:
        """Path relinking: generate intermediate solutions between current and target."""
        if not current.truck_routes or not target.truck_routes:
            return current

        # Get customer sequences
        current_seq = []
        for route in current.truck_routes:
            current_seq.extend(route.customers())

        target_seq = []
        for route in target.truck_routes:
            target_seq.extend(route.customers())

        if set(current_seq) != set(target_seq):
            return current  # Different customer sets, can't do path relinking

        best_solution = current
        best_cost = self._evaluator.evaluate_solution(current).total_cost
        if not math.isfinite(best_cost):
            return current

        # Simple path relinking: move customers from current sequence toward target sequence
        working = current.clone()

        for i, target_cust in enumerate(target_seq[:5]):  # Limit iterations
            # Find where target_cust is in working solution
            found_route = None
            found_pos = None

            for r_idx, route in enumerate(working.truck_routes):
                if target_cust in route.nodes:
                    found_route = r_idx
                    found_pos = route.nodes.index(target_cust)
                    break

            if found_route is None:
                continue

            # Try moving it to match target position
            # (simplified: just try some moves and evaluate)
            cand = working.clone()

            # Remove from current position
            cand.truck_routes[found_route].nodes.remove(target_cust)

            # Insert at a position that brings it closer to target order
            # For simplicity, insert into first route at position i+1
            if cand.truck_routes:
                insert_route = 0
                insert_pos = min(i + 1, len(cand.truck_routes[0].nodes) - 1)
                insert_pos = max(1, insert_pos)
                cand.truck_routes[insert_route].nodes.insert(
                    insert_pos, target_cust)

                try:
                    cand_eval = self._evaluator.evaluate_solution(cand)
                    if math.isfinite(cand_eval.total_cost) and cand_eval.total_cost < best_cost - 1e-6:
                        best_cost = cand_eval.total_cost
                        best_solution = cand
                        working = cand.clone()
                except Exception:
                    pass

        return best_solution
