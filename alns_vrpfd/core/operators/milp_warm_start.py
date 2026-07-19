"""MILP warm-start polishing for ALNS solutions.

Builds the full MILP model but seeds it with the ALNS solution as a
warm start, then solves with a short time limit. This allows the MILP
to simultaneously optimize both truck routes AND drone task assignments,
which ALNS local search cannot do.

For R_40_10_1 (~900 vars), with a warm start at cost 97.60,
Gurobi should converge toward 97.18 in 30-60 seconds.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from alns_vrpfd.evaluation.evaluator import Evaluator
from alns_vrpfd.evaluation.run_record import reconstruct_solution_from_mip
from alns_vrpfd.instance.manager import InstanceManager
from alns_vrpfd.mip.builder import MIPArtifacts, build_mip_model
from alns_vrpfd.model import Solution

logger = logging.getLogger(__name__)


@dataclass
class WarmStartResult:
    original_cost: float
    polished_cost: float
    improvement: float
    runtime_seconds: float
    improved: bool
    status: int
    gap: float
    polished_solution: Optional[Solution]


def _set_warm_start(
    artifacts: MIPArtifacts,
    solution: Solution,
    instance: InstanceManager,
) -> None:
    """Set warm start values from an ALNS solution onto MILP variables."""
    data = artifacts.data
    vars = artifacts.variables

    for route in solution.truck_routes:
        k = route.id
        if k not in data.trucks:
            continue
        for pos in range(len(route.nodes) - 1):
            i = route.nodes[pos]
            j = route.nodes[pos + 1]
            if (i, j) in data.arcs:
                var = vars.x_truck.get((i, j, k))
                if var is not None:
                    var.Start = 1

    for dt in solution.drone_tasks:
        d = dt.drone_id
        if d not in data.drones:
            continue
        seg_nodes = [dt.launch_node] + list(dt.customers()) + [dt.retrieve_node]
        for pos in range(len(seg_nodes) - 1):
            i = seg_nodes[pos]
            j = seg_nodes[pos + 1]
            if (i, j) in data.arcs:
                var = vars.y_drone.get((i, j, d))
                if var is not None:
                    var.Start = 1

        if dt.launch_node in data.nodes and dt.launch_truck is not None:
            u_var = vars.u.get((dt.launch_node, dt.launch_truck, d))
            if u_var is not None:
                u_var.Start = 1

        if dt.retrieve_node in data.nodes and dt.land_truck is not None:
            u_var = vars.u.get((dt.retrieve_node, dt.land_truck, d))
            if u_var is not None:
                u_var.Start = 1

        for customer in dt.customers():
            if customer in data.customers and d in data.drones:
                v_var = vars.v_served.get((customer, d))
                if v_var is not None:
                    v_var.Start = 1


def polish_with_full_milp_warm_start(
    solution: Solution,
    instance: InstanceManager,
    evaluator: Evaluator,
    time_limit: float = 30.0,
    mip_gap: float = 0.001,
    verbose: bool = False,
    energy_budget: int = 3,
) -> WarmStartResult:
    """Polish an ALNS solution by solving the full MILP with warm start."""
    start_time = time.perf_counter()
    original_eval = evaluator.evaluate_solution(solution)
    original_cost = original_eval.total_cost

    if not original_eval.feasible:
        return WarmStartResult(
            original_cost=original_cost, polished_cost=original_cost,
            improvement=0.0, runtime_seconds=time.perf_counter() - start_time,
            improved=False, status=-1, gap=1.0, polished_solution=None,
        )

    from alns_vrpfd.utils.config_loader import ALNSConfig
    cfg = ALNSConfig()

    artifacts = build_mip_model(
        instance,
        epsilon=1e-3,
        energy_budget=energy_budget,
        num_segments=10,
        use_gurobi_pwl=True,
        robust_energy=True,
        big_m_time=1000,
        big_m_load=1000,
        big_m_energy=20.0,
        tardiness_weight=1.0,
        cost_lambda=cfg.cost_lambda,
        cost_rho=cfg.cost_rho,
        cost_normalized=cfg.cost_normalized,
    )

    _set_warm_start(artifacts, solution, instance)

    model = artifacts.model
    model.setParam("TimeLimit", time_limit)
    model.setParam("MIPGap", mip_gap)
    model.setParam("Threads", 4)
    model.setParam("OutputFlag", 1 if verbose else 0)
    model.optimize()

    runtime = time.perf_counter() - start_time

    # Use MILP objective (more reliable than reconstruction)
    if model.SolCount > 0:
        milp_obj = model.ObjVal
    else:
        milp_obj = float("inf")

    # Also try reconstruction
    polished_solution = reconstruct_solution_from_mip(artifacts)
    if polished_solution is not None:
        polished_eval = evaluator.evaluate_solution(polished_solution)
        polished_cost_from_eval = polished_eval.total_cost if polished_eval.feasible else float("inf")
        if polished_cost_from_eval < original_cost - 0.01:
            polished_cost = polished_cost_from_eval
        else:
            polished_cost = original_cost
    else:
        # Fall back: MILP objective approximation
        if milp_obj < original_cost - 0.01:
            polished_cost = milp_obj
        else:
            polished_cost = original_cost

    improvement = original_cost - polished_cost
    status = model.Status
    gap = model.MIPGap if hasattr(model, 'MIPGap') else 1.0

    status_desc = {1: "loaded", 2: "optimal", 3: "infeasible", 4: "inf_or_unbd",
                   5: "unbounded", 6: "cutoff", 7: "iteration_limit",
                   8: "node_limit", 9: "time_limit", 10: "solution_limit",
                   11: "interrupted", 12: "numeric", 13: "suboptimal"}

    if status == GRB.OPTIMAL:
        logger.info(f"MILP proven optimal: {original_cost:.2f} -> {polished_cost:.2f} (delta={improvement:.2f}, time={runtime:.1f}s)")
    elif improvement > 0.01:
        logger.info(f"MILP improved: {original_cost:.2f} -> {polished_cost:.2f} (delta={improvement:.2f}, gap={gap:.2%}, status={status_desc.get(status, status)}, time={runtime:.1f}s)")
    else:
        logger.info(f"MILP no improvement: {polished_cost:.2f} (gap={gap:.2%}, status={status_desc.get(status, status)}, time={runtime:.1f}s)")

    return WarmStartResult(
        original_cost=original_cost,
        polished_cost=polished_cost,
        improvement=improvement,
        runtime_seconds=runtime,
        improved=improvement > 0.01,
        status=status,
        gap=gap,
        polished_solution=polished_solution or solution.clone(),
    )