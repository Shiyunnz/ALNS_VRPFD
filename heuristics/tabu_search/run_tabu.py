"""Run Tabu Search experiments for comparison with ALNS."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

from alns_vrpfd.evaluation import Evaluator
from alns_vrpfd.model.initializer import build_initial_solution
from alns_vrpfd.utils.io_utils import read_instance
from tabu_search import TabuSearch

# Configuration for Tabu Search runs
RUNS: List[Dict[str, Any]] = [
    {
        "instance": "data/Instance10/R_30_10_1.txt",
        "tabu_tenure": 10,
        "max_iterations": 1000,
        "time_limit": 300,  # 5 minutes
        "seed": 0,
        "notes": "Tabu Search on small instance for comparison",
    },
    {
        "instance": "data/Instance10/R_30_10_2.txt",
        "tabu_tenure": 10,
        "max_iterations": 1000,
        "time_limit": 300,
        "seed": 0,
        "notes": "Tabu Search comparison run",
    },
    {
        "instance": "data/Instance10/R_30_10_3.txt",
        "tabu_tenure": 10,
        "max_iterations": 1000,
        "time_limit": 300,
        "seed": 0,
        "notes": "Tabu Search comparison run",
    },
    {
        "instance": "data/Instance10/R_30_10_4.txt",
        "tabu_tenure": 10,
        "max_iterations": 1000,
        "time_limit": 300,
        "seed": 0,
        "notes": "Tabu Search comparison run",
    },
    {
        "instance": "data/Instance10/R_30_10_5.txt",
        "tabu_tenure": 10,
        "max_iterations": 1000,
        "time_limit": 300,
        "seed": 0,
        "notes": "Tabu Search comparison run",
    },
    {
        "instance": "data/Instance10/R_40_10_1.txt",
        "tabu_tenure": 10,
        "max_iterations": 1000,
        "time_limit": 300,
        "seed": 0,
        "notes": "Tabu Search comparison run",
    },
    {
        "instance": "data/Instance10/R_40_10_2.txt",
        "tabu_tenure": 10,
        "max_iterations": 1000,
        "time_limit": 300,
        "seed": 0,
        "notes": "Tabu Search comparison run",
    },
    {
        "instance": "data/Instance10/R_40_10_3.txt",
        "tabu_tenure": 10,
        "max_iterations": 1000,
        "time_limit": 300,
        "seed": 0,
        "notes": "Tabu Search comparison run",
    },
    {
        "instance": "data/Instance10/R_40_10_4.txt",
        "tabu_tenure": 10,
        "max_iterations": 1000,
        "time_limit": 300,
        "seed": 0,
        "notes": "Tabu Search comparison run",
    },
    {
        "instance": "data/Instance10/R_40_10_5.txt",
        "tabu_tenure": 10,
        "max_iterations": 1000,
        "time_limit": 300,
        "seed": 0,
        "notes": "Tabu Search comparison run",
    },
]

OUTPUT_PATH = Path("results") / "tabu_search_results.json"


def run_tabu_with_profile(
    tabu_search: TabuSearch,
    evaluator: Evaluator,
    initial_solution,
    time_limit: float | None = None,
):
    """Run Tabu Search with profiling."""
    start_time = time.perf_counter()
    best_solution = tabu_search.run(initial_solution, time_limit=time_limit)
    total_time = time.perf_counter() - start_time

    profile = {
        "total": total_time,
        "initial_cost": evaluator.evaluate_solution(initial_solution).total_cost,
        "best_cost": evaluator.evaluate_solution(best_solution).total_cost,
    }

    return best_solution, profile


def main() -> None:
    results = []

    for config in RUNS:
        instance_path = config["instance"]
        print(f"=== Running {instance_path} ===")

        instance = read_instance(instance_path)
        evaluator = Evaluator(instance)

        # Use greedy initial solution (consistent with ALNS)
        initial_solution = build_initial_solution(instance)

        tabu_search = TabuSearch(
            evaluator=evaluator,
            tabu_tenure=config["tabu_tenure"],
            max_iterations=config["max_iterations"],
            rng=random.Random(config["seed"]),
        )

        best_solution, profile = run_tabu_with_profile(
            tabu_search, evaluator, initial_solution, time_limit=config["time_limit"]
        )

        result = {
            "config": config,
            "profile": profile,
        }
        results.append(result)

        print(f"Initial cost: {profile['initial_cost']:.3f}")
        print(f"Best cost: {profile['best_cost']:.3f}")
        print(f"Run time: {profile['total']:.2f}s")
        print()

    # Save results
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
