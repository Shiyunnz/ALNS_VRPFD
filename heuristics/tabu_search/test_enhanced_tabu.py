"""Test script for enhanced Tabu Search implementation."""

from __future__ import annotations
from algorithms.tabu_search import EnhancedTabuSearch
from alns_vrpfd.utils.io_utils import read_instance
from alns_vrpfd.model.initializer import build_initial_solution
from alns_vrpfd.evaluation import Evaluator

import time
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in [str(p) for p in __import__('sys').path]:
    __import__('sys').path.insert(0, str(project_root))

import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent
for _p in [Path(__file__).resolve().parent] + list(Path(__file__).resolve().parents):
    if (_p / 'run_alns.py').exists():
        _project_root = _p
        break
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
del _p, _project_root


def test_enhanced_tabu_search():
    """Test the enhanced Tabu Search implementation."""
    instance_path = "data/Instance10/R_30_10_1.txt"

    print(f"Testing Enhanced Tabu Search on {instance_path}")
    print("=" * 60)

    # Load instance and create components
    instance = read_instance(instance_path)
    evaluator = Evaluator(instance)
    initial_solution = build_initial_solution(instance)

    initial_cost = evaluator.evaluate_solution(initial_solution).total_cost
    print(f"Initial solution cost: {initial_cost:.3f}")

    # Create enhanced Tabu Search
    tabu_search = EnhancedTabuSearch(
        instance=instance,
        evaluator=evaluator,
        tabu_tenure=15,
        max_iterations=1000,
        adaptive_tabu=True
    )

    # Run search
    start_time = time.perf_counter()
    best_solution = tabu_search.run(
        initial_solution, time_limit=60.0)  # 1 minute limit
    end_time = time.perf_counter()

    best_cost = evaluator.evaluate_solution(best_solution).total_cost
    run_time = end_time - start_time

    print(f"Best solution cost: {best_cost:.3f}")
    print(
        f"Improvement: {initial_cost - best_cost:.3f} ({((initial_cost - best_cost) / initial_cost * 100):.1f}%)")
    print(f"Run time: {run_time:.2f} seconds")
    print(f"Feasible: {evaluator.evaluate_solution(best_solution).feasible}")

    # Solution summary
    print(f"Truck routes: {len(best_solution.truck_routes)}")
    print(f"Drone tasks: {len(best_solution.drone_tasks)}")

    return best_solution, best_cost


if __name__ == "__main__":
    test_enhanced_tabu_search()
