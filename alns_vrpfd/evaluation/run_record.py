from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from alns_vrpfd.evaluation.evaluator import EvaluationDetails, NodeDelay
from alns_vrpfd.evaluation.robustness import DroneEnergyAssessment
from alns_vrpfd.instance.manager import InstanceManager
from alns_vrpfd.model.route import DroneTask, TruckRoute
from alns_vrpfd.model.solution import Solution


def _fmt(val: Any) -> str:
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _path_string(nodes: Sequence[int]) -> str:
    return " → ".join(str(n) for n in nodes)


def _path_string_with_drone_markers(
    path_nodes: Sequence[int],
    launch_nodes: set[int],
) -> str:
    parts: list[str] = []
    for n in path_nodes:
        label = f"{n}(drone↑)" if n in launch_nodes else str(n)
        parts.append(label)
    return " → ".join(parts)


def _get_truck_arrival_map(
    details: EvaluationDetails,
) -> dict[int, dict[int, float]]:
    result: dict[int, dict[int, float]] = {}
    for tid, timing in details.truck_timings.items():
        arrival_map: dict[int, float] = {}
        for node, arr in zip(
            timing.arrival_times.keys(), timing.arrival_times.values()
        ):
            arrival_map[node] = arr
        result[tid] = arrival_map
    return result


def build_run_record(
    instance: InstanceManager,
    algorithm: str,
    solution: Solution,
    details: EvaluationDetails,
    runtime_seconds: float,
    config: dict,
    seed: int | None = None,
    instance_name: str | None = None,
) -> dict:
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%dT%H:%M:%S")
    ts_compact = now.strftime("%Y-%m-%d_%H-%M-%S")
    if instance_name is None:
        instance_name = "unknown"

    rec: dict = {}
    result = details.result
    rob = details.robustness

    # ── run ──
    rec["run"] = {
        "id": f"{instance_name}_{algorithm}_{ts_compact}",
        "algorithm": algorithm,
        "instance": instance_name,
        "seed": seed,
        "timestamp": ts,
        "runtime_seconds": round(runtime_seconds, 2),
        "config": {k: v for k, v in config.items() if not callable(v)},
    }

    # ── summary ──
    truck_count = len(solution.truck_routes)
    drone_ids = sorted({t.drone_id for t in solution.drone_tasks})
    truck_custs = {c for tr in solution.truck_routes for c in tr.customers()}
    drone_custs = {c for dt in solution.drone_tasks for c in dt.customers()}
    customer_count = len(truck_custs | drone_custs)

    delay_violations = sum(
        1 for nd in details.delay_breakdown.nodes if nd.delay > 1e-9
    )

    rec["summary"] = {
        "total_cost": round(result.total_cost, 4),
        "transport_cost": round(result.truck_distance_cost + result.drone_distance_cost, 4),
        "delay_cost": round(result.delay_penalty, 4),
        "service_completion_hour": round(_compute_service_completion(details), 4),
        "energy_feasible": rob.feasible,
        "num_trucks": truck_count,
        "num_drones": len(drone_ids),
        "num_customers_served": customer_count,
        "delay_violations": delay_violations,
    }

    # ── routes ──
    truck_arrival_map = _get_truck_arrival_map(details)

    all_drone_launch_nodes: set[int] = set()
    launch_per_truck: dict[int, set[int]] = {}
    for dt in solution.drone_tasks:
        if dt.launch_truck is not None:
            launch_per_truck.setdefault(dt.launch_truck, set()).add(dt.launch_node)
            all_drone_launch_nodes.add(dt.launch_node)

    routes_list: list[dict] = []
    for tr in solution.truck_routes:
        tid = tr.id
        launch_nodes = launch_per_truck.get(tid, set())
        path_str = _path_string_with_drone_markers(tr.nodes, launch_nodes)
        arrival_map = truck_arrival_map.get(tid, {})
        arrival_times = [arrival_map.get(n, 0.0) for n in tr.nodes]
        routes_list.append({
            "truck": tid,
            "path": path_str,
            "node_sequence": list(tr.nodes),
            "arrival_times": [round(t, 4) for t in arrival_times],
            "drone_launches": sorted(launch_nodes),
            "drone_launch_count": len(launch_nodes),
        })
    rec["routes"] = routes_list

    # ── drone_flights ──
    energy_by_task: dict[int | None, DroneEnergyAssessment] = {}
    for ass in rob.task_breakdown:
        energy_by_task[ass.task_id] = ass

    drone_tasks_by_drone: dict[int, list[DroneTask]] = {}
    for dt in solution.drone_tasks:
        drone_tasks_by_drone.setdefault(dt.drone_id, []).append(dt)

    flights_list: list[dict] = []
    for did in sorted(drone_tasks_by_drone.keys()):
        tasks = drone_tasks_by_drone[did]
        seg_strings: list[str] = []
        for dt in tasks:
            path_nodes = list(dt.nodes)
            served = dt.customers()
            ass = energy_by_task.get(dt.task_id)
            if ass is not None:
                energy_str = f"{round(ass.nominal_energy, 2)}/{round(ass.capacity, 1) if ass.capacity else '?'} kWh"
                feasible_mark = "✓" if ass.feasible else "✗"
            else:
                energy_str = "?/?"
                feasible_mark = "?"

            launch_str = f"T{dt.launch_truck}@{dt.launch_node}" if dt.launch_truck is not None else f"Depot@{dt.launch_node}"
            land_str = f"T{dt.land_truck}@{dt.retrieve_node}" if dt.land_truck is not None else f"Depot@{dt.retrieve_node}"

            seg = (
                f"{_path_string(path_nodes)}  "
                f"(customers: {','.join(str(c) for c in served)},  "
                f"{launch_str}→{land_str},  "
                f"energy: {energy_str} {feasible_mark})"
            )
            seg_strings.append(seg)

        flights_list.append({
            "drone": did,
            "parent_truck": tasks[0].launch_truck if tasks else None,
            "segments": seg_strings,
        })
    rec["drone_flights"] = flights_list

    # ── details.customers ──
    customer_details_list: list[dict] = []
    delay_by_node: dict[int, NodeDelay] = {
        nd.node_id: nd for nd in details.delay_breakdown.nodes
    }

    for cust in instance.customer_manager.customers():
        cid = cust.customer_id
        nd = delay_by_node.get(cid)
        tw_opt, tw_late = instance.customer_manager.time_window(cid)
        sclass = instance.customer_manager.supply_class(cid)

        drone_id = None
        parent_truck = None
        for dt in solution.drone_tasks:
            if cid in dt.customers():
                drone_id = dt.drone_id
                parent_truck = dt.launch_truck
                break

        if nd is not None:
            arrival = nd.arrival_time
            delay = nd.delay
        else:
            arrival = None
            delay = 0.0
            # fallback: look up from truck timings
            for tr in solution.truck_routes:
                if cid in tr.customers() and tr.id in truck_arrival_map:
                    arrival = truck_arrival_map[tr.id].get(cid)
                    if arrival is not None:
                        break
            # fallback: look up from drone timings
            if arrival is None:
                for dt in solution.drone_tasks:
                    if cid in dt.customers() and dt.task_id is not None:
                        dtime = details.drone_timings.get(dt.task_id)
                        if dtime is not None:
                            arrival = dtime.customer_arrival_times.get(cid)
                            if arrival is not None:
                                break

        delay_cost = 0.0
        if delay > 0 and cid in {n.node_id for n in details.delay_breakdown.nodes}:
            from alns_vrpfd.deprivation import deprivation_cost
            try:
                delay_cost = deprivation_cost(
                    delay,
                    supply_class=sclass,
                    cost_lambda=config.get("cost_lambda", 12.0),
                    rho=config.get("cost_rho", 1.0),
                    normalized=config.get("cost_normalized", True),
                )
            except Exception:
                delay_cost = 0.0

        customer_details_list.append({
            "id": cid,
            "class": sclass or "?",
            "demand": cust.demand,
            "drone": drone_id,
            "parent_truck": parent_truck,
            "arrival": round(arrival, 4) if arrival is not None else None,
            "due": round(tw_opt, 4) if tw_opt is not None else None,
            "deadline": round(tw_late, 4) if tw_late is not None else None,
            "delay_hours": round(delay, 4),
            "delay_cost": round(delay_cost, 4),
            "early": arrival is not None and tw_opt is not None and arrival < tw_opt - 1e-9,
        })

    rec.setdefault("details", {})["customers"] = customer_details_list

    # ── details.energy_per_segment ──
    seg_details: list[dict] = []
    for ass in rob.task_breakdown:
        seg_details.append({
            "drone": ass.drone_id,
            "path": ass.task_id,
            "nominal_kwh": round(ass.nominal_energy, 4),
            "worst_case_kwh": round(ass.worst_case_energy, 4),
            "battery": round(ass.capacity, 4) if ass.capacity is not None else None,
            "feasible": ass.feasible,
            "margin_kwh": round(ass.margin, 4),
            "segment_energies": [round(e, 4) for e in ass.segment_energies],
        })
    rec["details"]["energy_per_segment"] = seg_details

    # ── details.timing ──
    total_delay = sum(nd.delay for nd in details.delay_breakdown.nodes)
    max_delay = max((nd.delay for nd in details.delay_breakdown.nodes), default=0.0)
    rec["details"]["timing"] = {
        "total_travel_hours": rec["summary"]["service_completion_hour"],
        "total_delay_hours": round(total_delay, 4),
        "max_delay_hours": round(max_delay, 4),
        "time_window_violations": delay_violations,
    }

    return rec


def reconstruct_solution_from_mip(
    artifacts: "MIPArtifacts",
) -> Solution | None:
    import gurobipy as gp

    data = artifacts.data
    vars = artifacts.variables
    model = artifacts.model

    if model.SolCount == 0:
        return None

    sol = Solution()

    # Truck routes
    for k in data.trucks:
        truck_arcs = [
            (i, j)
            for (i, j) in data.arcs
            if vars.x_truck[i, j, k].X > 0.5
        ]
        if not truck_arcs:
            continue
        from alns_vrpfd.mip.run_mip import _reconstruct_routes
        routes = _reconstruct_routes(truck_arcs)
        if not routes:
            continue
        path = routes[0]
        from alns_vrpfd.model.route import TruckRoute
        tr = TruckRoute(route_id=k, nodes=path, capacity=data.truck_capacity)
        sol.add_truck_route(tr)

    # Build node→truck map
    node_to_truck: dict[int, int] = {}
    for k in data.trucks:
        truck_arcs_k = [
            (i, j)
            for (i, j) in data.arcs
            if vars.x_truck[i, j, k].X > 0.5
        ]
        for i, j in truck_arcs_k:
            node_to_truck[i] = k
            node_to_truck[j] = k  # last node overwrite is fine

    # Drone tasks
    next_task_id = 0
    for d in data.drones:
        drone_arcs = [
            (i, j)
            for (i, j) in data.arcs
            if vars.y_drone[i, j, d].X > 0.5
        ]
        if not drone_arcs:
            continue

        # Build u launch map: (node, truck) for this drone
        u_trucks: dict[int, int] = {}  # launch_node -> truck
        for (i_key, k_key, d_key), var in vars.u.items():
            if d_key == d and var.X > 0.5:
                u_trucks[i_key] = k_key

        # v_served customers
        served = {j for (j, dd), var in vars.v_served.items() if dd == d and var.X > 0.5}

        # Reconstruct chain(s) sorted by arrival time
        arrival_times = {}
        for i in data.nodes:
            v = vars.arrival_drone[i, d].X
            if v > 0:
                arrival_times[i] = v
        from alns_vrpfd.mip.run_mip import _reconstruct_routes
        chains = _reconstruct_routes(drone_arcs, arrival_times)

        for chain in chains:
            seg_nodes = list(chain)
            # Split at launch nodes (excluding start node)
            split_indices = [0]
            for idx, node in enumerate(seg_nodes):
                if idx > 0 and node in u_trucks:
                    split_indices.append(idx)
            split_indices.append(len(seg_nodes) - 1)

            for si in range(len(split_indices) - 1):
                start_idx = split_indices[si]
                end_idx = split_indices[si + 1] + 1
                if end_idx - start_idx < 2:
                    continue
                seg = seg_nodes[start_idx:end_idx]
                launch_node = seg[0]
                retrieve_node = seg[-1]

                seg_customers = [n for n in seg[1:-1] if n in served]
                if not seg_customers:
                    continue

                launch_truck = u_trucks.get(launch_node)
                land_truck = node_to_truck.get(retrieve_node)

                payloads = []
                for arc_i, arc_j in zip(seg, seg[1:]):
                    p = float(vars.load_drone_minus[arc_j, d].X)
                    payloads.append(p)

                from alns_vrpfd.model.route import DroneTask
                dt = DroneTask(
                    drone_id=d,
                    launch_truck=launch_truck,
                    launch_node=launch_node,
                    customers=seg_customers,
                    land_truck=land_truck,
                    retrieve_node=retrieve_node,
                    payloads=payloads,
                    task_id=next_task_id,
                )
                next_task_id += 1
                sol.add_drone_task(dt)

    return sol


def _compute_service_completion(details: EvaluationDetails) -> float:
    latest = 0.0
    for timing in details.truck_timings.values():
        if timing.arrival_times:
            vals = list(timing.arrival_times.values())
            latest = max(latest, vals[-1])
    for timing in details.drone_timings.values():
        latest = max(latest, timing.retrieve_time)
    return latest


def save_run_record(
    instance: InstanceManager,
    algorithm: str,
    solution: Solution,
    details: EvaluationDetails,
    runtime_seconds: float,
    config: dict,
    seed: int | None = None,
    instance_name: str | None = None,
    output_dir: str | Path = "results/records",
) -> str:
    record = build_run_record(
        instance=instance,
        algorithm=algorithm,
        solution=solution,
        details=details,
        runtime_seconds=runtime_seconds,
        config=config,
        seed=seed,
        instance_name=instance_name,
    )

    instance_name = record["run"]["instance"]
    ts = record["run"]["timestamp"].replace(":", "-").replace("T", "_")
    algo = algorithm
    filename = f"{algo}_{ts}.json"
    out_path = Path(output_dir) / instance_name
    out_path.mkdir(parents=True, exist_ok=True)
    filepath = out_path / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    return str(filepath)
