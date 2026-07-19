"""Command-line entrypoint to run the GA search on a given instance."""

from __future__ import annotations
from algorithms.ga.ga import GeneticAlgorithm, GAConfig
from alns_vrpfd.utils.io_utils import read_instance
from alns_vrpfd.model.initializer import build_initial_solution
from alns_vrpfd.evaluation import Evaluator

import argparse
import json
import random
import time
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to sys.path for setup_path import
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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


def load_config(config_path: Path) -> GAConfig:
    """Load GA configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    return GAConfig(**config_dict)


def run_ga_single(
    instance_path: Path,
    config: GAConfig,
    seed: int = 42,
    output_stats: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run GA on a single instance."""
    start_time = time.time()

    # Load instance
    instance = read_instance(str(instance_path), strategy="class_based")

    # Initialize components
    evaluator = Evaluator(instance)
    rng = random.Random(seed)

    # Create initial solution (optional)
    initial_solution = build_initial_solution(instance)

    # Run GA
    ga = GeneticAlgorithm(instance, config, evaluator, rng)
    best_individual = ga.run(initial_solution)

    total_time = time.time() - start_time

    # Prepare results
    result = {
        "instance": str(instance_path),
        "algorithm": "GA",
        "seed": seed,
        "config": {
            "population_size": config.population_size,
            "generations": config.generations,
            "crossover_rate": config.crossover_rate,
            "mutation_rate": config.mutation_rate,
            "elite_size": config.elite_size,
        },
        "results": {
            "total_cost": best_individual.fitness,
            "feasible": best_individual.feasible,
            "truck_distance": best_individual.truck_distance,
            "drone_distance": best_individual.drone_distance,
            "delay_penalty": best_individual.delay_penalty,
            "solve_time": total_time,
            "generations_completed": ga.generation + 1,
        },
        "solution_summary": {
            "truck_routes_count": len(best_individual.solution.truck_routes),
            "drone_tasks_count": len(best_individual.solution.drone_tasks),
            "truck_route_lengths": [len(route.nodes) for route in best_individual.solution.truck_routes],
        }
    }

    # Save detailed statistics if requested
    if output_stats:
        stats = ga.get_statistics()
        stats_result = {
            "instance": str(instance_path),
            "algorithm": "GA",
            "seed": seed,
            "statistics": stats,
            "config": result["config"],
        }

        with open(output_stats, 'w', encoding='utf-8') as f:
            json.dump(stats_result, f, indent=2, ensure_ascii=False)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Genetic Algorithm on VRPFD instances."
    )
    parser.add_argument(
        "instance",
        type=Path,
        help="Path to the instance file",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config_ga.json",
        help="Path to GA configuration JSON file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        help="Output JSON file for detailed statistics",
    )

    # Override config parameters
    parser.add_argument("--population-size", type=int, help="Population size")
    parser.add_argument("--generations", type=int,
                        help="Number of generations")
    parser.add_argument("--crossover-rate", type=float, help="Crossover rate")
    parser.add_argument("--mutation-rate", type=float, help="Mutation rate")
    parser.add_argument("--elite-size", type=int, help="Elite size")
    parser.add_argument("--time-limit", type=float,
                        help="Time limit in seconds")

    args = parser.parse_args()

    # Load base configuration
    config = load_config(args.config)

    # Override configuration with command line arguments
    if args.population_size is not None:
        config.population_size = args.population_size
    if args.generations is not None:
        config.generations = args.generations
    if args.crossover_rate is not None:
        config.crossover_rate = args.crossover_rate
    if args.mutation_rate is not None:
        config.mutation_rate = args.mutation_rate
    if args.elite_size is not None:
        config.elite_size = args.elite_size
    if args.time_limit is not None:
        config.time_limit = args.time_limit

    # Run GA
    result = run_ga_single(
        instance_path=args.instance,
        config=config,
        seed=args.seed,
        output_stats=args.stats_output,
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"GA Results for {args.instance.name}")
    print(f"{'='*60}")
    print(f"Total Cost: {result['results']['total_cost']:.2f}")
    print(f"Feasible: {result['results']['feasible']}")
    print(f"Truck Distance: {result['results']['truck_distance']:.2f}")
    print(f"Drone Distance: {result['results']['drone_distance']:.2f}")
    print(f"Delay Penalty: {result['results']['delay_penalty']:.2f}")
    print(f"Solve Time: {result['results']['solve_time']:.2f}s")
    print(f"Generations: {result['results']['generations_completed']}")
    print(f"Truck Routes: {result['solution_summary']['truck_routes_count']}")
    print(f"Drone Tasks: {result['solution_summary']['drone_tasks_count']}")
    print(f"{'='*60}")

    # Save results if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")

    if args.stats_output:
        print(f"Detailed statistics saved to {args.stats_output}")


if __name__ == "__main__":
    main()
