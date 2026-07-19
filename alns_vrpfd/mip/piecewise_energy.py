"""分段线性化能耗约束实现模块。

此模块提供将非线性能耗函数 E(ω) = (W+m+ω)^1.5 * const 
转换为线性约束的功能，保持 MILP 模型的线性特性。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    gp = None
    GRB = None

from alns_vrpfd.evaluation.energy import DroneEnergyModel


class PiecewiseLinearEnergyBuilder:
    """构建分段线性化的能耗约束。"""

    def __init__(
        self,
        energy_model: DroneEnergyModel | None = None,
        num_segments: int = 10,
        use_gurobi_pwl: bool = True,
    ):
        """
        Parameters
        ----------
        energy_model : DroneEnergyModel, optional
            能耗模型实例，默认使用标准参数
        num_segments : int
            分段数量，越多越精确但变量越多，默认10
        use_gurobi_pwl : bool
            是否使用 Gurobi 的 addGenConstrPWL（推荐），否则手动实现
        """
        self.energy_model = energy_model or DroneEnergyModel()
        self.num_segments = num_segments
        self.use_gurobi_pwl = use_gurobi_pwl

        # 提取能耗模型参数
        self.W = self.energy_model.body_weight_kg
        self.m = self.energy_model.battery_weight_kg
        self.constant = math.sqrt(
            (self.energy_model.gravitational_accel ** 3)
            / (2.0 * self.energy_model.air_density
               * self.energy_model.disc_area
               * self.energy_model.rotor_count)
        ) / 1000.0  # kW

    def compute_breakpoints(
        self,
        omega_max: float
    ) -> Tuple[list[float], list[float]]:
        """计算分段线性化的分段点和对应的功率值。

        Parameters
        ----------
        omega_max : float
            最大负载（通常是无人机容量）

        Returns
        -------
        breakpoints : list[float]
            负载分段点 [0, omega_max/K, 2*omega_max/K, ..., omega_max]
        power_values : list[float]
            对应的功率值 [f(0), f(omega_max/K), ..., f(omega_max)]
        """
        K = self.num_segments
        breakpoints = [k * omega_max / K for k in range(K + 1)]

        power_values = []
        for omega in breakpoints:
            effective_mass = self.W + self.m + omega
            power_kw = (effective_mass ** 1.5) * self.constant
            power_values.append(power_kw)

        return breakpoints, power_values

    def add_piecewise_energy_constraints(
        self,
        model,
        data,  # ProblemData
        vars,  # VariableContainer
        big_m_energy: float,
        *,
        robust_energy: bool = False,
    ) -> Dict[str, any]:
        """添加分段线性化的能耗约束到模型中。

        Parameters
        ----------
        model : gp.Model
            Gurobi 模型实例
        data : ProblemData
            问题数据
        vars : VariableContainer
            决策变量容器
        big_m_energy : float
            能耗约束的 Big-M 值

        Returns
        -------
        new_vars : dict
            新增的辅助变量字典，包含:
            - 'power_approx': 近似功率变量
            - 'omega_active': 活跃负载变量（仅在弧被使用时非零）
        """
        print("  构建分段线性化能耗约束...")

        omega_max = data.drone_capacity
        breakpoints, power_values = self.compute_breakpoints(omega_max)

        print(f"    分段数: {self.num_segments}")
        print(f"    负载范围: [0, {omega_max}] kg")
        print(f"    分段点数: {len(breakpoints)}")

        arc_set = data.arcs
        max_power = max(power_values) if power_values else 0.0
        finite_times = [
            data.drone_time[(i, j)]
            for (i, j) in arc_set
            if not math.isinf(data.drone_time[(i, j)])
        ]
        max_travel_time = max(finite_times) if finite_times else 0.0

        # 创建功率近似变量
        power_approx = model.addVars(
            arc_set,
            data.drones,
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            name="power_approx",
        )

        # 创建活跃负载变量 (omega_active = load_drone_plus * y)
        omega_active = model.addVars(
            arc_set,
            data.drones,
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            ub=omega_max,
            name="omega_active",
        )

        energy_active = model.addVars(
            arc_set,
            data.drones,
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            name="energy_active",
        )

        print(
            f"    新增变量: {len(arc_set) * len(data.drones) * 3}"
        )

        # 为每条弧添加分段线性约束
        constraint_count = 0
        for (i, j) in arc_set:
            travel_time = data.drone_time[(i, j)]
            for d in data.drones:
                # Step 1: omega_active = load_drone_minus[i,d] * y[i,j,d]
                # 使用 McCormick 松弛
                # 注意: load_drone_minus[i,d]表示无人机到达i时的累积负载，
                # 也就是离开i沿弧(i,j)飞行时携带的负载(已扣除i的服务需求)
                self._add_mccormick_product(
                    model,
                    omega_active[i, j, d],
                    vars.load_drone_plus[i, d],  # 离开i时的累积负载
                    vars.y_drone[i, j, d],
                    omega_max,
                    name_prefix=f"mccormick[{i},{j},{d}]",
                )
                constraint_count += 4

                # Step 2: power_approx = f(omega_active)
                if self.use_gurobi_pwl and hasattr(model, 'addGenConstrPWL'):
                    # 使用 Gurobi 的分段线性约束
                    model.addGenConstrPWL(
                        omega_active[i, j, d],
                        power_approx[i, j, d],
                        breakpoints,
                        power_values,
                        name=f"pwl_power[{i},{j},{d}]",
                    )
                else:
                    # 手动实现分段线性化（使用 SOS2 或凸组合）
                    self._add_manual_pwl(
                        model,
                        omega_active[i, j, d],
                        power_approx[i, j, d],
                        breakpoints,
                        power_values,
                        name_prefix=f"pwl[{i},{j},{d}]",
                    )
                    constraint_count += len(breakpoints) + 2

                energy_nom_expr = power_approx[i, j, d] * travel_time
                max_energy_arc = max_power * travel_time
                model.addConstr(
                    energy_active[i, j, d] - energy_nom_expr
                    <= big_m_energy * (1 - vars.y_drone[i, j, d]),
                    name=f"energy_link_ub[{i},{j},{d}]",
                )
                model.addConstr(
                    energy_nom_expr - energy_active[i, j, d]
                    <= big_m_energy * (1 - vars.y_drone[i, j, d]),
                    name=f"energy_link_lb[{i},{j},{d}]",
                )
                model.addConstr(
                    energy_active[i, j, d]
                    <= max_energy_arc * vars.y_drone[i, j, d],
                    name=f"energy_usage_gate[{i},{j},{d}]",
                )

        print(f"    新增约束: ~{constraint_count}")

        # 添加能耗流约束（使用分段线性化的功率）
        self._add_energy_flow_constraints(
            model,
            data,
            vars,
            energy_active,
            big_m_energy,
            robust_energy=robust_energy,
        )

        return {
            'power_approx': power_approx,
            'omega_active': omega_active,
            'energy_active': energy_active,
        }

    def _add_mccormick_product(
        self,
        model,
        z: Any,
        x: Any,
        y: Any,
        x_max: float,
        name_prefix: str,
    ):
        """添加 McCormick 松弛约束: z = x * y, 其中 y ∈ {0,1}, x ∈ [0, x_max]。

        McCormick 不等式:
        1. z >= 0
        2. z >= x - x_max * (1 - y)
        3. z <= x_max * y
        4. z <= x
        """
        model.addConstr(z >= 0, name=f"{name_prefix}_lb")
        model.addConstr(
            z >= x - x_max * (1 - y),
            name=f"{name_prefix}_lb_active"
        )
        model.addConstr(
            z <= x_max * y,
            name=f"{name_prefix}_ub_inactive"
        )
        model.addConstr(z <= x, name=f"{name_prefix}_ub")

    def _add_manual_pwl(
        self,
        model,
        x: Any,
        y: Any,
        xpts: list[float],
        ypts: list[float],
        name_prefix: str,
    ):
        """手动实现分段线性约束（适用于不支持 addGenConstrPWL 的 Gurobi 版本）。

        使用凸组合方法:
        y = Σ λ_k * ypts[k]
        x = Σ λ_k * xpts[k]
        Σ λ_k = 1
        λ_k >= 0
        最多两个相邻的 λ_k 非零 (SOS2)
        """
        K = len(xpts) - 1

        # 创建插值权重变量
        lambda_vars = model.addVars(
            range(K + 1),
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            ub=1.0,
            name=f"{name_prefix}_lambda",
        )

        # 约束: Σ λ_k = 1
        model.addConstr(
            gp.quicksum(lambda_vars[k] for k in range(K + 1)) == 1,
            name=f"{name_prefix}_sum",
        )

        # 约束: x = Σ λ_k * xpts[k]
        model.addConstr(
            x == gp.quicksum(lambda_vars[k] * xpts[k] for k in range(K + 1)),
            name=f"{name_prefix}_x_interp",
        )

        # 约束: y = Σ λ_k * ypts[k]
        model.addConstr(
            y == gp.quicksum(lambda_vars[k] * ypts[k] for k in range(K + 1)),
            name=f"{name_prefix}_y_interp",
        )

        # SOS2 约束: 最多两个相邻的 λ 非零
        model.addSOS(GRB.SOS_TYPE2, [lambda_vars[k] for k in range(K + 1)])

    def _add_energy_flow_constraints(
        self,
        model,
        data,
        vars,
        energy_active,
        big_m_energy: float,
        *,
        robust_energy: bool = False,
    ):
        """使用 gamma-indexed 状态与分段能耗联动。

        修改版本：当无人机从发射点出发时，能耗从0开始。
        这与 ALNS 的实现一致：每个 DroneTask 的能耗是独立检查的，
        假设无人机在回收后会在卡车上充满电。
        """
        arc_set = data.arcs
        gamma_range = data.gamma_range
        gamma_min = gamma_range[0]

        for (i, j) in arc_set:
            for d in data.drones:
                energy_use = energy_active[i, j, d]
                # 检查是否从发射点出发（如果 u[i,k,d]=1 for any k）
                u_sum = gp.quicksum(vars.u[i, k, d] for k in data.trucks)

                for gamma in gamma_range:
                    # 标准能耗流约束（当不是发射点时有效）
                    # 当 u_sum=1 时，这个约束被松弛
                    model.addConstr(
                        vars.energy_state_gamma[j, d, gamma]
                        >= vars.energy_state_gamma[i, d, gamma]
                        + energy_use
                        - big_m_energy * (1 - vars.y_drone[i, j, d])
                        - big_m_energy * u_sum,  # 发射点松弛
                        name=f"energy_flow_pwl_nominal[{i},{j},{d},{gamma}]",
                    )
                    # 发射点能耗流约束（当是发射点时有效，能耗从0开始）
                    # 只有当 u_sum=1 且 y[i,j]=1 时约束才有效
                    model.addConstr(
                        vars.energy_state_gamma[j, d, gamma]
                        >= energy_use
                        - big_m_energy * (1 - vars.y_drone[i, j, d])
                        - big_m_energy * (1 - u_sum),  # 只在发射点有效
                        name=f"energy_flow_pwl_launch[{i},{j},{d},{gamma}]",
                    )

                    if gamma > gamma_min:
                        energy_dev = data.energy_deviation_rate * energy_use
                        # 标准偏差约束（非发射点）
                        model.addConstr(
                            vars.energy_state_gamma[j, d, gamma]
                            >= vars.energy_state_gamma[i, d, gamma - 1]
                            + energy_use
                            + energy_dev
                            - big_m_energy * (1 - vars.y_drone[i, j, d])
                            - big_m_energy * u_sum,  # 发射点松弛
                            name=f"energy_flow_pwl_deviation[{i},{j},{d},{gamma}]",
                        )
                        # 发射点偏差约束
                        model.addConstr(
                            vars.energy_state_gamma[j, d, gamma]
                            >= energy_use + energy_dev
                            - big_m_energy * (1 - vars.y_drone[i, j, d])
                            - big_m_energy * (1 - u_sum),
                            name=f"energy_flow_pwl_launch_dev[{i},{j},{d},{gamma}]",
                        )

        # # 原始版本（已注释）：不处理发射点重置，能耗跨任务累积
        # for (i, j) in arc_set:
        #     for d in data.drones:
        #         energy_use = energy_active[i, j, d]
        #         for gamma in gamma_range:
        #             model.addConstr(
        #                 vars.energy_state_gamma[j, d, gamma]
        #                 >= vars.energy_state_gamma[i, d, gamma]
        #                 + energy_use
        #                 - big_m_energy * (1 - vars.y_drone[i, j, d]),
        #                 name=f"energy_flow_pwl_nominal[{i},{j},{d},{gamma}]",
        #             )
        #             if gamma > gamma_min:
        #                 energy_dev = data.energy_deviation_rate * energy_use
        #                 model.addConstr(
        #                     vars.energy_state_gamma[j, d, gamma]
        #                     >= vars.energy_state_gamma[i, d, gamma - 1]
        #                     + energy_use
        #                     + energy_dev
        #                     - big_m_energy * (1 - vars.y_drone[i, j, d]),
        #                     name=f"energy_flow_pwl_deviation[{i},{j},{d},{gamma}]",
        #                 )


def add_piecewise_linear_energy_constraints(
    model,
    data,
    vars,
    big_m_energy: float,
    num_segments: int = 10,
    use_gurobi_pwl: bool = True,
    *,
    robust_energy: bool = False,
) -> Dict[str, any]:
    """便捷函数：添加分段线性化能耗约束。

    Parameters
    ----------
    model : gp.Model
        Gurobi 模型
    data : ProblemData
        问题数据
    vars : VariableContainer
        决策变量
    big_m_energy : float
        Big-M 常数
    num_segments : int
        分段数量，默认 10
    use_gurobi_pwl : bool
        是否使用 Gurobi 的 PWL 功能，默认 True

    Returns
    -------
    new_vars : dict
        新增变量字典
    """
    builder = PiecewiseLinearEnergyBuilder(
        num_segments=num_segments,
        use_gurobi_pwl=use_gurobi_pwl,
    )
    return builder.add_piecewise_energy_constraints(
        model, data, vars, big_m_energy, robust_energy=robust_energy
    )
