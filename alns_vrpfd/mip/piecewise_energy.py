"""。

 E(ω) = (W+m+ω)^1.5 * const
， MILP 。
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
    """。"""

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
            ，
        num_segments : int
            ，，10
        use_gurobi_pwl : bool
             Gurobi  addGenConstrPWL（），
        """
        self.energy_model = energy_model or DroneEnergyModel()
        self.num_segments = num_segments
        self.use_gurobi_pwl = use_gurobi_pwl


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
        """。

        Parameters
        ----------
        omega_max : float
            （）

        Returns
        -------
        breakpoints : list[float]
             [0, omega_max/K, 2*omega_max/K, ..., omega_max]
        power_values : list[float]
             [f(0), f(omega_max/K), ..., f(omega_max)]
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
        """。

        Parameters
        ----------
        model : gp.Model
            Gurobi 
        data : ProblemData
            
        vars : VariableContainer
            
        big_m_energy : float
             Big-M 

        Returns
        -------
        new_vars : dict
            ，:
            - 'power_approx': 
            - 'omega_active': （）
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


        power_approx = model.addVars(
            arc_set,
            data.drones,
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            name="power_approx",
        )

        # (omega_active = load_drone_plus * y)
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


        constraint_count = 0
        for (i, j) in arc_set:
            travel_time = data.drone_time[(i, j)]
            for d in data.drones:
                # Step 1: omega_active = load_drone_minus[i,d] * y[i,j,d]
                # McCormick
                # : load_drone_minus[i,d]i，
                # i(i,j)(i)
                self._add_mccormick_product(
                    model,
                    omega_active[i, j, d],
                    vars.load_drone_plus[i, d],  # i
                    vars.y_drone[i, j, d],
                    omega_max,
                    name_prefix=f"mccormick[{i},{j},{d}]",
                )
                constraint_count += 4

                # Step 2: power_approx = f(omega_active)
                if self.use_gurobi_pwl and hasattr(model, 'addGenConstrPWL'):
                    # Gurobi
                    model.addGenConstrPWL(
                        omega_active[i, j, d],
                        power_approx[i, j, d],
                        breakpoints,
                        power_values,
                        name=f"pwl_power[{i},{j},{d}]",
                    )
                else:
                    # （ SOS2 ）
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

        # （）
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
        """ McCormick : z = x * y,  y ∈ {0,1}, x ∈ [0, x_max]。

        McCormick :
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
        """（ addGenConstrPWL  Gurobi ）。

        :
        y = Σ λ_k * ypts[k]
        x = Σ λ_k * xpts[k]
        Σ λ_k = 1
        λ_k >= 0
         λ_k  (SOS2)
        """
        K = len(xpts) - 1


        lambda_vars = model.addVars(
            range(K + 1),
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            ub=1.0,
            name=f"{name_prefix}_lambda",
        )

        # : Σ λ_k = 1
        model.addConstr(
            gp.quicksum(lambda_vars[k] for k in range(K + 1)) == 1,
            name=f"{name_prefix}_sum",
        )

        # : x = Σ λ_k * xpts[k]
        model.addConstr(
            x == gp.quicksum(lambda_vars[k] * xpts[k] for k in range(K + 1)),
            name=f"{name_prefix}_x_interp",
        )

        # : y = Σ λ_k * ypts[k]
        model.addConstr(
            y == gp.quicksum(lambda_vars[k] * ypts[k] for k in range(K + 1)),
            name=f"{name_prefix}_y_interp",
        )

        # SOS2 :  λ
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
        """ gamma-indexed 。

        ：，0。
         ALNS ： DroneTask ，
        。
        """
        arc_set = data.arcs
        gamma_range = data.gamma_range
        gamma_min = gamma_range[0]

        for (i, j) in arc_set:
            for d in data.drones:
                energy_use = energy_active[i, j, d]
                # （ u[i,k,d]=1 for any k）
                u_sum = gp.quicksum(vars.u[i, k, d] for k in data.trucks)

                for gamma in gamma_range:
                    # （）
                    # u_sum=1 ，
                    model.addConstr(
                        vars.energy_state_gamma[j, d, gamma]
                        >= vars.energy_state_gamma[i, d, gamma]
                        + energy_use
                        - big_m_energy * (1 - vars.y_drone[i, j, d])
                        - big_m_energy * u_sum,
                        name=f"energy_flow_pwl_nominal[{i},{j},{d},{gamma}]",
                    )
                    # （，0）
                    # u_sum=1  y[i,j]=1
                    model.addConstr(
                        vars.energy_state_gamma[j, d, gamma]
                        >= energy_use
                        - big_m_energy * (1 - vars.y_drone[i, j, d])
                        - big_m_energy * (1 - u_sum),
                        name=f"energy_flow_pwl_launch[{i},{j},{d},{gamma}]",
                    )

                    if gamma > gamma_min:
                        energy_dev = data.energy_deviation_rate * energy_use
                        # （）
                        model.addConstr(
                            vars.energy_state_gamma[j, d, gamma]
                            >= vars.energy_state_gamma[i, d, gamma - 1]
                            + energy_use
                            + energy_dev
                            - big_m_energy * (1 - vars.y_drone[i, j, d])
                            - big_m_energy * u_sum,
                            name=f"energy_flow_pwl_deviation[{i},{j},{d},{gamma}]",
                        )

                        model.addConstr(
                            vars.energy_state_gamma[j, d, gamma]
                            >= energy_use + energy_dev
                            - big_m_energy * (1 - vars.y_drone[i, j, d])
                            - big_m_energy * (1 - u_sum),
                            name=f"energy_flow_pwl_launch_dev[{i},{j},{d},{gamma}]",
                        )

        # # （）：，
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
    """：。

    Parameters
    ----------
    model : gp.Model
        Gurobi 
    data : ProblemData
        
    vars : VariableContainer
        
    big_m_energy : float
        Big-M 
    num_segments : int
        ， 10
    use_gurobi_pwl : bool
         Gurobi  PWL ， True

    Returns
    -------
    new_vars : dict
        
    """
    builder = PiecewiseLinearEnergyBuilder(
        num_segments=num_segments,
        use_gurobi_pwl=use_gurobi_pwl,
    )
    return builder.add_piecewise_energy_constraints(
        model, data, vars, big_m_energy, robust_energy=robust_energy
    )
