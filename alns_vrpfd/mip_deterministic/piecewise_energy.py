"""Deterministic piecewise linear energy constraints (without robust optimization).

此模块提供将非线性能耗函数 E(ω) = (W+m+ω)^1.5 * const
转换为线性约束的功能，用于确定性MILP模型。
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
    """Build piecewise linearized energy constraints for deterministic model."""

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
            Energy model instance with default parameters
        num_segments : int
            Number of segments for piecewise approximation
        use_gurobi_pwl : bool
            Whether to use Gurobi's addGenConstrPWL (recommended)
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
        ) / 1000.0

    def compute_breakpoints(
        self,
        omega_max: float
    ) -> Tuple[list[float], list[float]]:
        """Compute breakpoints for piecewise linearization.

        Parameters
        ----------
        omega_max : float
            Maximum payload (typically drone capacity)

        Returns
        -------
        breakpoints : list[float]
            Load breakpoints [0, omega_max/K, 2*omega_max/K, ..., omega_max]
        power_values : list[float]
            Corresponding power values [f(0), f(omega_max/K), ..., f(omega_max)]
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
        data,
        vars,
        big_m_energy: float,
    ) -> Dict[str, any]:
        """Add piecewise linear energy constraints to deterministic model.

        Parameters
        ----------
        model : gp.Model
            Gurobi model instance
        data : ProblemData
            Problem data
        vars : VariableContainer
            Decision variable container
        big_m_energy : float
            Big-M value for energy constraints

        Returns
        -------
        new_vars : dict
            New auxiliary variables including:
            - 'power_approx': approximate power variable
            - 'omega_active': active load variable
            - 'energy_active': active energy variable
        """
        print("  Building deterministic piecewise linear energy constraints...")

        omega_max = data.drone_capacity
        breakpoints, power_values = self.compute_breakpoints(omega_max)

        print(f"    Segments: {self.num_segments}")
        print(f"    Load range: [0, {omega_max}] kg")
        print(f"    Breakpoints: {len(breakpoints)}")

        arc_set = data.arcs
        max_power = max(power_values) if power_values else 0.0

        power_approx = model.addVars(
            arc_set,
            data.drones,
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            name="power_approx",
        )

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
            f"    New variables: {len(arc_set) * len(data.drones) * 3}"
        )

        constraint_count = 0
        for (i, j) in arc_set:
            travel_time = data.drone_time[(i, j)]
            for d in data.drones:
                self._add_mccormick_product(
                    model,
                    omega_active[i, j, d],
                    vars.load_drone_plus[i, d],
                    vars.y_drone[i, j, d],
                    omega_max,
                    name_prefix=f"mccormick[{i},{j},{d}]",
                )
                constraint_count += 4

                if self.use_gurobi_pwl and hasattr(model, 'addGenConstrPWL'):
                    model.addGenConstrPWL(
                        omega_active[i, j, d],
                        power_approx[i, j, d],
                        breakpoints,
                        power_values,
                        name=f"pwl_power[{i},{j},{d}]",
                    )
                else:
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

        print(f"    New constraints: ~{constraint_count}")

        self._add_deterministic_energy_flow_constraints(
            model,
            data,
            vars,
            energy_active,
            big_m_energy,
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
        """Add McCormick envelope constraints: z = x * y, where y ∈ {0,1}, x ∈ [0, x_max].

        McCormick inequalities:
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
        """Manually implement piecewise linear constraints using convex combination.

        y = Σ λ_k * ypts[k]
        x = Σ λ_k * xpts[k]
        Σ λ_k = 1
        λ_k >= 0
        At most two adjacent λ_k are non-zero (SOS2)
        """
        K = len(xpts) - 1

        lambda_vars = model.addVars(
            range(K + 1),
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            ub=1.0,
            name=f"{name_prefix}_lambda",
        )

        model.addConstr(
            gp.quicksum(lambda_vars[k] for k in range(K + 1)) == 1,
            name=f"{name_prefix}_sum",
        )

        model.addConstr(
            x == gp.quicksum(lambda_vars[k] * xpts[k] for k in range(K + 1)),
            name=f"{name_prefix}_x_interp",
        )

        model.addConstr(
            y == gp.quicksum(lambda_vars[k] * ypts[k] for k in range(K + 1)),
            name=f"{name_prefix}_y_interp",
        )

        model.addSOS(GRB.SOS_TYPE2, [lambda_vars[k] for k in range(K + 1)])

    def _add_deterministic_energy_flow_constraints(
        self,
        model,
        data,
        vars,
        energy_active,
        big_m_energy: float,
    ):
        """Add deterministic energy flow constraints (without gamma states).

        When a drone launches from a truck, energy resets to 0 (each task is independent).
        """
        arc_set = data.arcs

        for (i, j) in arc_set:
            for d in data.drones:
                energy_use = energy_active[i, j, d]
                u_sum = gp.quicksum(vars.u[i, k, d] for k in data.trucks)

                # Standard energy flow constraint (when not a launch point)
                # When u_sum=1, this constraint is relaxed
                model.addConstr(
                    vars.energy_state[j, d]
                    >= vars.energy_state[i, d]
                    + energy_use
                    - big_m_energy * (1 - vars.y_drone[i, j, d])
                    - big_m_energy * u_sum,
                    name=f"energy_flow_det_nominal[{i},{j},{d}]",
                )

                # Launch point energy flow constraint (energy starts from 0)
                # Only active when u_sum=1 and y[i,j]=1
                model.addConstr(
                    vars.energy_state[j, d]
                    >= energy_use
                    - big_m_energy * (1 - vars.y_drone[i, j, d])
                    - big_m_energy * (1 - u_sum),
                    name=f"energy_flow_det_launch[{i},{j},{d}]",
                )


def add_piecewise_linear_energy_constraints(
    model,
    data,
    vars,
    big_m_energy: float,
    num_segments: int = 10,
    use_gurobi_pwl: bool = True,
) -> Dict[str, any]:
    """Convenience function to add piecewise linear energy constraints.

    Parameters
    ----------
    model : gp.Model
        Gurobi model
    data : ProblemData
        Problem data
    vars : VariableContainer
        Decision variables
    big_m_energy : float
        Big-M constant
    num_segments : int
        Number of segments, default 10
    use_gurobi_pwl : bool
        Whether to use Gurobi's PWL feature, default True

    Returns
    -------
    new_vars : dict
        New variables dictionary
    """
    builder = PiecewiseLinearEnergyBuilder(
        num_segments=num_segments,
        use_gurobi_pwl=use_gurobi_pwl,
    )
    return builder.add_piecewise_energy_constraints(
        model, data, vars, big_m_energy
    )