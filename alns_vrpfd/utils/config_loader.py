"""YAML配置文件加载工具"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def find_config_file(config_name: str = "alns_config.yaml") -> Path:
    """查找配置文件路径

    搜索顺序:
    1. 当前工作目录
    2. 当前工作目录的config子目录
    3. 项目根目录
    4. 项目根目录的config子目录
    """
    search_paths = [
        Path.cwd() / config_name,
        Path.cwd() / "config" / config_name,
    ]

    # 尝试找到项目根目录 (包含alns_vrpfd的目录)
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "alns_vrpfd").exists():
            search_paths.extend([
                parent / config_name,
                parent / "config" / config_name,
            ])
            break

    for path in search_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"配置文件 '{config_name}' 未找到。搜索路径: {[str(p) for p in search_paths]}"
    )


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """加载YAML配置文件

    Args:
        config_path: 配置文件路径，如果为None则自动查找

    Returns:
        配置字典
    """
    if config_path is None:
        config_path = find_config_file()
    else:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def get_nested(config: Dict, *keys, default=None):
    """安全获取嵌套配置值

    Args:
        config: 配置字典
        *keys: 嵌套键路径
        default: 默认值

    Returns:
        配置值或默认值
    """
    result = config
    for key in keys:
        if isinstance(result, dict) and key in result:
            result = result[key]
        else:
            return default
    return result


class ALNSConfig:
    """ALNS配置类，提供便捷的配置访问接口"""

    def __init__(self, config_path: Optional[str | Path] = None):
        self._config = load_config(config_path)
        self._validate()

    def _validate(self):
        """验证必要的配置项"""
        required_sections = ["simulated_annealing", "adaptive_selection"]
        for section in required_sections:
            if section not in self._config:
                print(f"警告: 配置缺少 '{section}' 部分，将使用默认值")

    @property
    def relax_allow_anchor_conflict(self) -> bool:
        return get_nested(self._config, "relaxation", "allow_anchor_conflict", default=True)

    @property
    def relax_allow_multiple_launch_per_node(self) -> bool:
        return get_nested(self._config, "relaxation", "allow_multiple_launch_per_node", default=True)

    @property
    def raw(self) -> Dict[str, Any]:
        """返回原始配置字典"""
        return self._config

    # ========== 基础设置 ==========
    @property
    def seed(self) -> Optional[int]:
        """返回随机种子，None 表示使用系统时间"""
        return get_nested(self._config, "general", "seed", default=None)

    @property
    def deterministic(self) -> bool:
        return get_nested(self._config, "general", "deterministic", default=False)

    @property
    def time_window_strategy(self) -> str:
        return get_nested(self._config, "general", "time_window_strategy", default="class_based")

    # ========== 迭代次数 ==========
    @property
    def iterations_default(self) -> int:
        """默认迭代次数（命令行不指定时使用）"""
        return get_nested(self._config, "iterations", "default", default=2000)

    @property
    def iterations_small(self) -> int:
        return get_nested(self._config, "iterations", "small", default=2000)

    @property
    def iterations_large(self) -> int:
        return get_nested(self._config, "iterations", "large", default=4000)

    @property
    def iterations(self) -> int:
        """默认迭代次数（小规模）"""
        return self.iterations_small

    @property
    def max_non_improve(self) -> Optional[int]:
        return get_nested(self._config, "iterations", "max_non_improve", default=None)

    @property
    def time_limit(self) -> int:
        """时间限制（秒）"""
        return get_nested(self._config, "iterations", "time_limit", default=1800)

    # ========== 模拟退火 ==========
    @property
    def w_percent(self) -> float:
        return get_nested(self._config, "simulated_annealing", "w_percent", default=30.0)

    @property
    def temperature_min(self) -> float:
        return get_nested(self._config, "simulated_annealing", "temperature_min", default=1e-4)

    @property
    def sa_min_temp(self) -> float:
        """模拟退火最低温度（temperature_min的别名）"""
        return self.temperature_min

    @property
    def cooling_rate_initial(self) -> float:
        return get_nested(self._config, "simulated_annealing", "cooling", "rate_initial", default=0.998)

    @property
    def cooling_rate_final(self) -> float:
        return get_nested(self._config, "simulated_annealing", "cooling", "rate_final", default=0.978)

    @property
    def cooling_rate_default(self) -> float:
        return get_nested(self._config, "simulated_annealing", "cooling", "rate_default", default=0.982)

    @property
    def cooling_transition_iters(self) -> int:
        return get_nested(self._config, "simulated_annealing", "cooling", "transition_iters", default=1400)

    # ========== 重热机制 ==========
    @property
    def reheat_stall_trigger(self) -> int:
        return get_nested(self._config, "reheat", "stall_trigger", default=300)

    @property
    def reheat_acceptance_window(self) -> int:
        return get_nested(self._config, "reheat", "acceptance_window", default=80)

    @property
    def reheat_acceptance_min(self) -> float:
        return get_nested(self._config, "reheat", "acceptance_min", default=0.05)

    @property
    def reheat_duration(self) -> int:
        return get_nested(self._config, "reheat", "duration", default=40)

    @property
    def reheat_recovery(self) -> int:
        return get_nested(self._config, "reheat", "recovery", default=30)

    @property
    def reheat_cooldown(self) -> int:
        return get_nested(self._config, "reheat", "cooldown", default=60)

    @property
    def reheat_quota_multiplier(self) -> float:
        return get_nested(self._config, "reheat", "quota_multiplier", default=2.2)

    @property
    def reheat_quota_upper_cap(self) -> float:
        return get_nested(self._config, "reheat", "quota_upper_cap", default=0.95)

    @property
    def reheat_quota_absolute_cap(self) -> Optional[int]:
        return get_nested(self._config, "reheat", "quota_absolute_cap", default=90)

    @property
    def reheat_p_floor(self) -> float:
        return get_nested(self._config, "reheat", "p_floor", default=0.1)

    @property
    def reheat_random_repair_prob(self) -> float:
        return get_nested(self._config, "reheat", "random_repair_prob", default=0.5)

    @property
    def reheat_shake_fraction(self) -> float:
        return get_nested(self._config, "reheat", "shake_fraction", default=0.4)

    @property
    def reheat_shake_probability(self) -> float:
        return get_nested(self._config, "reheat", "shake_probability", default=0.85)

    @property
    def reheat_temperature_scale(self) -> float:
        return get_nested(self._config, "reheat", "temperature_scale", default=1.2)

    # ========== 销毁配额 ==========
    # β ∈ [max{3, r_L|C|}, min{30, r_U|C|}]
    @property
    def r_lower(self) -> float:
        """r_L: 移除比例下限 (默认15%)"""
        return get_nested(self._config, "destroy_quota", "r_lower", default=0.15)

    @property
    def r_upper_small(self) -> float:
        """r_U: 小规模实例移除比例上限 (默认50%)"""
        return get_nested(self._config, "destroy_quota", "r_upper_small", default=0.50)

    @property
    def r_upper_large(self) -> float:
        """r_U: 大规模实例移除比例上限 (默认30%)"""
        return get_nested(self._config, "destroy_quota", "r_upper_large", default=0.30)

    @property
    def quota_base_cap(self) -> int:
        """移除数量绝对上限 (公式中的30)"""
        return get_nested(self._config, "destroy_quota", "base_cap", default=30)

    # ========== 自适应算子选择 ==========
    @property
    def eta(self) -> float:
        return get_nested(self._config, "adaptive_selection", "eta", default=0.6)

    @property
    def alpha_credit(self) -> float:
        return get_nested(self._config, "adaptive_selection", "alpha_credit", default=0.65)

    @property
    def reward_scale(self) -> Dict[str, float]:
        default = {"global": 40.0, "better": 18.0,
                   "slight_better": 12.0, "accepted_worse": 2.0}
        return get_nested(self._config, "adaptive_selection", "rewards", default=default)

    @property
    def weight_bounds(self) -> tuple:
        bounds = get_nested(self._config, "adaptive_selection",
                            "weight_bounds", default={})
        return (bounds.get("min", 0.1), bounds.get("max", 100.0))

    @property
    def probability_floor(self) -> float:
        return get_nested(self._config, "adaptive_selection", "probability_floor", default=0.02)

    @property
    def weight_history(self) -> int:
        return get_nested(self._config, "adaptive_selection", "weight_history", default=30)

    @property
    def weight_decay(self) -> float:
        return get_nested(self._config, "adaptive_selection", "weight_decay", default=0.02)

    # ========== 局部搜索 ==========
    @property
    def local_search_frequency(self) -> int:
        return get_nested(self._config, "local_search", "frequency", default=8)

    @property
    def local_search_on_new_best(self) -> bool:
        return get_nested(self._config, "local_search", "on_new_best", default=True)

    @property
    def depot_drone_probability(self) -> float:
        return get_nested(self._config, "local_search", "depot_drone_probability", default=0.15)

    @property
    def intensify_frequency(self) -> int:
        return get_nested(self._config, "local_search", "intensify_frequency", default=40)

    @property
    def cross_exchange_prob(self) -> float:
        return get_nested(self._config, "local_search", "cross_exchange_prob", default=0.3)

    @property
    def path_relinking_prob(self) -> float:
        return get_nested(self._config, "local_search", "path_relinking_prob", default=0.15)

    @property
    def drone_reanchor_ls_enabled(self) -> bool:
        return get_nested(self._config, "local_search", "drone_reanchor_ls_enabled", default=True)

    @property
    def drone_reanchor_ls_max_moves(self) -> int:
        return get_nested(self._config, "local_search", "drone_reanchor_ls_max_moves", default=10)

    @property
    def drone_composite_reanchor_enabled(self) -> bool:
        return get_nested(self._config, "local_search", "drone_composite_reanchor_enabled", default=False)

    @property
    def drone_composite_reanchor_max_tasks(self) -> int:
        return get_nested(self._config, "local_search", "drone_composite_reanchor_max_tasks", default=3)

    @property
    def drone_sortie_constructor_enabled(self) -> bool:
        return get_nested(self._config, "local_search", "drone_sortie_constructor_enabled", default=False)

    @property
    def drone_sortie_max_customers(self) -> int:
        return get_nested(self._config, "local_search", "drone_sortie_max_customers", default=3)

    @property
    def drone_sortie_top_k(self) -> int:
        return get_nested(self._config, "local_search", "drone_sortie_top_k", default=5)

    @property
    def drone_sortie_max_sorties(self) -> int:
        return get_nested(self._config, "local_search", "drone_sortie_max_sorties", default=20)

    @property
    def mini_milp_polish_enabled(self) -> bool:
        return get_nested(self._config, "local_search", "mini_milp_polish_enabled", default=False)

    @property
    def mini_milp_polish_time_limit(self) -> float:
        return get_nested(self._config, "local_search", "mini_milp_polish_time_limit", default=60.0)

    @property
    def mini_milp_polish_gap(self) -> float:
        return get_nested(self._config, "local_search", "mini_milp_polish_gap", default=0.005)

    @property
    def matheuristic_lns_enabled(self) -> bool:
        return get_nested(self._config, "local_search", "matheuristic_lns_enabled", default=False)

    @property
    def matheuristic_lns_frequency(self) -> int:
        return get_nested(self._config, "local_search", "matheuristic_lns_frequency", default=0)

    @property
    def matheuristic_lns_max_customers(self) -> int:
        return get_nested(self._config, "local_search", "matheuristic_lns_max_customers", default=3)

    @property
    def matheuristic_lns_trials(self) -> int:
        return get_nested(self._config, "local_search", "matheuristic_lns_trials", default=5)

    # ========== 无人机设置 ==========
    @property
    def drone_priority(self) -> float:
        return get_nested(self._config, "drone", "priority", default=2.0)

    @property
    def drone_bonus(self) -> Dict[str, float]:
        bonus = get_nested(self._config, "drone", "bonus", default={})
        return {
            "depot_bonus": bonus.get("depot", 0.5),
            "multi_customer_bonus": bonus.get("multi_customer", 5.0),
            "multi_customer_threshold": bonus.get("multi_customer_threshold", 2),
            "wait_max": bonus.get("wait_max", 20.0),
            # MIP allows multiple drones to launch/land at the same node
            "allow_multiple_launch_per_node": self.relax_allow_multiple_launch_per_node,
        }

    @property
    def drone_rendezvous_tolerance(self) -> float:
        """Maximum wait time for drone rendezvous."""
        return get_nested(self._config, "drone", "rendezvous_tolerance", default=0.5)

    @property
    def forced_drone_customers(self) -> List[int]:
        """客户点列表，这些点必须由无人机配送"""
        customers = get_nested(self._config, "drone",
                               "forced_drone_customers", default=[])
        return [int(c) for c in customers] if customers else []

    # ========== 日志设置 ==========
    @property
    def log_operators(self) -> bool:
        return get_nested(self._config, "logging", "log_operators", default=False)

    @property
    def operator_log_interval(self) -> int:
        return get_nested(self._config, "logging", "operator_log_interval", default=250)

    # ========== 鲁棒性设置 ==========
    @property
    def drone_battery_capacity(self) -> float:
        return get_nested(self._config, "robustness", "drone_battery_capacity", default=6.3)

    @property
    def energy_uncertainty_budget(self) -> int:
        return get_nested(self._config, "robustness", "energy_uncertainty_budget", default=3)

    @property
    def energy_deviation_rate(self) -> float:
        return get_nested(self._config, "robustness", "energy_deviation_rate", default=0.1)

    @property
    def same_truck_retrieval(self) -> bool:
        return get_nested(self._config, "robustness", "same_truck_retrieval", default=True)

    # NOTE: Parallel repair operator is enabled by default, no toggle in config
    # ========== 延误成本参数 ==========
    @property
    def cost_lambda(self) -> float:
        return get_nested(self._config, "delay_cost", "cost_lambda", default=12.0)

    @property
    def cost_rho(self) -> float:
        return get_nested(self._config, "delay_cost", "rho", default=1.0)

    @property
    def cost_normalized(self) -> bool:
        return get_nested(self._config, "delay_cost", "normalized", default=True)

    # ========== MIP / 分段线性化设置 ==========
    @property
    def piecewise_energy_segments(self) -> int:
        """返回能耗分段数（用于 Gurobi 分段近似）。"""
        return get_nested(self._config, "mip", "piecewise", "energy_num_segments", default=10)

    @property
    def piecewise_delay_segments(self) -> int:
        """返回延误函数的分段数（保留以备将来使用）。"""
        return get_nested(self._config, "mip", "piecewise", "delay_num_segments", default=10)

    # ========== 逃脱算法 ==========
    @property
    def escape_enabled(self) -> bool:
        return get_nested(self._config, "escape", "enabled", default=True)

    @property
    def escape_trigger_stall(self) -> int:
        return get_nested(self._config, "escape", "trigger_stall", default=100)

    @property
    def escape_duration(self) -> int:
        return get_nested(self._config, "escape", "duration", default=20)

    # ========== 收敛增强 ==========
    @property
    def diversification_enabled(self) -> bool:
        return get_nested(self._config, "convergence_enhancement", "diversification", "enabled", default=True)

    @property
    def diversification_trigger_stall(self) -> int:
        return get_nested(self._config, "convergence_enhancement", "diversification", "trigger_stall", default=500)

    @property
    def dynamic_cooling_enabled(self) -> bool:
        return get_nested(self._config, "convergence_enhancement", "dynamic_cooling", "enabled", default=True)

    @property
    def adaptive_quota_enabled(self) -> bool:
        return get_nested(self._config, "convergence_enhancement", "adaptive_quota", "enabled", default=True)

    # ========== MIP 设置 ==========
    @property
    def mip_time_limit(self) -> int:
        return get_nested(self._config, "mip", "time_limit", default=1800)

    @property
    def mip_gap(self) -> float:
        return get_nested(self._config, "mip", "gap", default=0.001)

    @property
    def mip_threads(self) -> int:
        return get_nested(self._config, "mip", "threads", default=0)

    @property
    def mip_output_flag(self) -> int:
        return get_nested(self._config, "mip", "output_flag", default=1)

    def iterations_for(self, size: str) -> int:
        """根据规模获取迭代次数"""
        return self.iterations_small if size.lower() == "small" else self.iterations_large

    def build_sa_config_dict(self, size: str = "small", iterations: Optional[int] = None) -> Dict[str, Any]:
        """构建SANNCfg所需的参数字典"""
        return {
            "size": size,
            "iterations": iterations,
            "w_percent": self.w_percent,
            "cooling_rate": self.cooling_rate_default,
            "cooling_rate_initial": self.cooling_rate_initial,
            "cooling_rate_final": self.cooling_rate_final,
            "cooling_transition_iters": self.cooling_transition_iters,
            "temperature_min": self.temperature_min,
            "eta": self.eta,
            "alpha_credit": self.alpha_credit,
            "reward_scale": self.reward_scale,
            "max_non_improve": self.max_non_improve,
            "r_lower": self.r_lower,
            "r_upper_small": self.r_upper_small,
            "r_upper_large": self.r_upper_large,
            "reheat_stall_trigger": self.reheat_stall_trigger,
            "reheat_acceptance_window": self.reheat_acceptance_window,
            "reheat_acceptance_min": self.reheat_acceptance_min,
            "reheat_duration": self.reheat_duration,
            "reheat_recovery": self.reheat_recovery,
            "reheat_cooldown": self.reheat_cooldown,
            "reheat_quota_multiplier": self.reheat_quota_multiplier,
            "reheat_quota_upper_cap": self.reheat_quota_upper_cap,
            "reheat_quota_absolute_cap": self.reheat_quota_absolute_cap,
            "reheat_p_floor": self.reheat_p_floor,
            "reheat_random_repair_prob": self.reheat_random_repair_prob,
            "reheat_shake_fraction": self.reheat_shake_fraction,
            "reheat_shake_probability": self.reheat_shake_probability,
            "reheat_temperature_scale": self.reheat_temperature_scale,
            "quota_base_cap": self.quota_base_cap,
            "weight_history": self.weight_history,
            "weight_decay": self.weight_decay,
            "log_operator_metrics": self.log_operators,
            "operator_log_interval": self.operator_log_interval,
            "local_search_frequency": self.local_search_frequency,
            "local_search_on_new_best": self.local_search_on_new_best,
            "depot_drone_probability": self.depot_drone_probability,
            "intensify_frequency": self.intensify_frequency,
            "cross_exchange_prob": self.cross_exchange_prob,
            "path_relinking_prob": self.path_relinking_prob,
            # 逃脱算法参数
            "escape_enabled": self.escape_enabled,
            "escape_trigger_stall": self.escape_trigger_stall,
            "escape_duration": self.escape_duration,
            # 收敛增强参数
            "diversification_enabled": self.diversification_enabled,
            "diversification_trigger_stall": self.diversification_trigger_stall,
            "diversification_restart_best_prob": get_nested(self._config, "convergence_enhancement", "diversification", "restart_best_prob", default=0.7),
            "diversification_destroy_ratio": get_nested(self._config, "convergence_enhancement", "diversification", "random_destroy_ratio", default=0.6),
            "dynamic_cooling_enabled": self.dynamic_cooling_enabled,
            "improvement_threshold": get_nested(self._config, "convergence_enhancement", "dynamic_cooling", "improvement_threshold", default=0.01),
            "cooling_slowdown_factor": get_nested(self._config, "convergence_enhancement", "dynamic_cooling", "slowdown_factor", default=0.998),
            "cooling_speedup_factor": get_nested(self._config, "convergence_enhancement", "dynamic_cooling", "speedup_factor", default=0.980),
            "recent_improvement_window": get_nested(self._config, "convergence_enhancement", "dynamic_cooling", "recent_improvement_window", default=50),
            "adaptive_quota_enabled": self.adaptive_quota_enabled,
            # Drone re-anchor local search
            "drone_reanchor_ls_enabled": self.drone_reanchor_ls_enabled,
            "drone_reanchor_ls_max_moves": self.drone_reanchor_ls_max_moves,
            "drone_composite_reanchor_enabled": self.drone_composite_reanchor_enabled,
            "drone_composite_reanchor_max_tasks": self.drone_composite_reanchor_max_tasks,
            "drone_sortie_constructor_enabled": self.drone_sortie_constructor_enabled,
            "drone_sortie_max_customers": self.drone_sortie_max_customers,
            "drone_sortie_top_k": self.drone_sortie_top_k,
            "drone_sortie_max_sorties": self.drone_sortie_max_sorties,
            "mini_milp_polish_enabled": self.mini_milp_polish_enabled,
            "mini_milp_polish_time_limit": self.mini_milp_polish_time_limit,
            "mini_milp_polish_gap": self.mini_milp_polish_gap,
            "matheuristic_lns_enabled": self.matheuristic_lns_enabled,
            "matheuristic_lns_frequency": self.matheuristic_lns_frequency,
            "matheuristic_lns_max_customers": self.matheuristic_lns_max_customers,
            "matheuristic_lns_trials": self.matheuristic_lns_trials,
        }


# 便捷函数: 直接获取默认配置实例
_default_config: Optional[ALNSConfig] = None


def get_default_config() -> ALNSConfig:
    """获取默认配置实例（单例模式）"""
    global _default_config
    if _default_config is None:
        try:
            _default_config = ALNSConfig()
        except FileNotFoundError:
            # 如果找不到配置文件，创建一个使用默认值的配置
            _default_config = ALNSConfig.__new__(ALNSConfig)
            _default_config._config = {}
    return _default_config
