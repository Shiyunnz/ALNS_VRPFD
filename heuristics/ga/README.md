# Genetic Algorithm for VRPFD

This directory contains a research-grade Genetic Algorithm (GA) implementation for the Vehicle Routing Problem with Drones and Time Windows (VRPFD), designed for fair comparison with the existing Adaptive Large Neighborhood Search (ALNS) algorithm.

## Features

- **Population-based Evolutionary Algorithm**: Tournament selection, elitism, crossover, and mutation operators
- **Solution Encoding**: Direct use of existing Solution class (truck_routes + drone_tasks) for compatibility
- **Fitness Evaluation**: Integrated with existing Evaluator for cost/feasible assessment
- **Comprehensive Statistics**: Generation-by-generation tracking of fitness, diversity, and feasibility
- **Reproducibility**: Seed-controlled random number generation with time limits and early stopping
- **Configurable Parameters**: JSON-based configuration system with CLI overrides

## Files

- `ga.py`: Core GA implementation with genetic operators and population management
- `run_ga.py`: Command-line interface for running GA experiments
- `config_ga.json`: Default GA parameter configuration

## Usage

### Basic Run

```bash
python heuristics/ga/run_ga.py data/Instance10/R_30_10_1.txt --seed 42
```

### With Custom Parameters

```bash
python heuristics/ga/run_ga.py data/Instance10/R_30_10_1.txt \
  --generations 200 \
  --population-size 150 \
  --crossover-rate 0.9 \
  --mutation-rate 0.05 \
  --seed 42 \
  --stats-output results.json
```

### Batch Comparison with ALNS

```bash
python compare_ga_alns.py \
  --instances data/Instance10/R_30_10_*.txt \
  --algorithms ga alns \
  --seeds 42 123 456 \
  --output comparison_results.json
```

### Visualization

```bash
# Text-based convergence analysis
python visualization/plot_ga_simple.py --stats results.json

# Algorithm comparison
python visualization/plot_ga_simple.py --comparison comparison_results.json
```

## Configuration Parameters

| Parameter                    | Default | Description                             |
| ---------------------------- | ------- | --------------------------------------- |
| `population_size`            | 100     | Number of individuals in population     |
| `generations`                | 200     | Maximum number of generations           |
| `tournament_size`            | 5       | Tournament selection size               |
| `crossover_rate`             | 0.8     | Probability of crossover                |
| `mutation_rate`              | 0.1     | Probability of mutation                 |
| `elite_size`                 | 5       | Number of elite individuals preserved   |
| `truck_route_crossover_rate` | 0.7     | Rate of truck route crossover           |
| `drone_task_mutation_rate`   | 0.3     | Rate of drone task mutation             |
| `route_segment_swap_rate`    | 0.4     | Rate of route segment swaps in mutation |

## Genetic Operators

### Selection

- **Tournament Selection**: Selects best individual from random tournament

### Crossover

- **Route-based Crossover**: Exchanges truck routes between parent solutions
- **Preserves Solution Structure**: Maintains depot-customer-depot format

### Mutation

- **Route Segment Swap**: Swaps customer positions within truck routes
- **Drone Task Modification**: Adds/removes drone tasks probabilistically

## Output Format

### Console Output

```
============================================================
GA Results for R_30_10_1.txt
============================================================
Total Cost: 129.70
Feasible: True
Truck Distance: 129.70
Drone Distance: 0.00
Delay Penalty: 0.00
Solve Time: 0.35s
Generations: 25
Truck Routes: 2
Drone Tasks: 0
============================================================
```

### Statistics JSON

Contains generation-by-generation history of:

- Best fitness values
- Average fitness values
- Feasible solution counts
- Population diversity metrics

## Integration with Existing Codebase

The GA implementation is fully integrated with the existing VRPFD framework:

- Uses `alns_vrpfd.evaluation.Evaluator` for fitness assessment
- Compatible with `alns_vrpfd.model.solution.Solution` class
- Leverages `alns_vrpfd.utils.io_utils.read_instance()` for data loading
- Follows repository conventions for path management and imports

## Research Applications

This GA implementation supports:

- **Algorithmic Comparison**: Direct performance comparison with ALNS
- **Parameter Sensitivity Analysis**: Systematic parameter tuning studies
- **Statistical Analysis**: Reproducible experiments with multiple seeds
- **Scalability Testing**: Performance evaluation across instance sizes
- **Hybrid Approaches**: Foundation for combining GA with local search

## Performance Notes

- GA typically finds good solutions but may converge slower than ALNS on this problem
- Population size and generations are key parameters affecting solution quality
- Early stopping based on stagnation detection prevents unnecessary computation
- Parallel evaluation of population enables efficient use of computational resources
