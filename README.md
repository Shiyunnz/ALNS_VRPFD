# RVRPFDD — Robust Vehicle Routing Problem with Flexible Docking Drones

This repository implements the **Robust Vehicle Routing Problem with Flexible Docking Drones in Humanitarian Logistics** (RVRPFDD). The framework coordinates multi-truck, multi-drone operations for post-disaster relief distribution.

## Features

- **Flexible docking**: Drones launched from one truck can be recovered by another, enabling cross-truck coordination without requiring pre-assigned pairing.
- **Hierarchical deadlines**: Combined hard and soft deadlines with convex deprivation-based penalties reflecting heterogeneous supply urgency.
- **Robust energy**: Bertsimas-Sim budgeted uncertainty sets ensure drone routes remain feasible under worst-case energy consumption scenarios.
- **Exact MILP formulation**: Gurobi-based model with piecewise-linear approximation of convex delay cost and nonlinear energy consumption (3 segments).
- **ALNS heuristic**: Adaptive Large Neighborhood Search with 14 tailored destroy/repair operators, including drone reanchor local search, matheuristic LNS, and truck-drone rechaining.
- **Benchmark TS/GA**: Tabu Search and Genetic Algorithm implementations for comparison.

## Repository structure

```
alns_vrpfd/          — Core algorithm package
  core/              — ALNS engine, SA cooling, operators
  evaluation/        — Evaluator, energy model, robustness checker
  instance/          — Instance reader, customer manager, time windows
  mip/               — MILP builder (Gurobi)
  model/             — Solution, TruckRoute, DroneTask
  utils/             — Config loader, I/O utilities
config/              — YAML configuration
data/                — Benchmark instances (n=10–100)
heuristics/          — TS and GA baseline implementations
sensitivity/         — Sensitivity analysis scripts
run_alns.py          — Main ALNS entry point
```

## Requirements

- Python 3.10+
- Gurobi (for MILP only; ALNS runs without it)
- numpy

## Usage

```bash
# Run ALNS on a single instance
python run_alns.py data/Instance10/R_30_10_1.txt --seed 42

# Compare ALNS with MILP on small instances
python run_alns_milp_comparison.py

# Run TS or GA benchmark
python heuristics/tabu_search/run_tabu.py data/Instance10/R_30_10_1.txt
```

Configuration is managed through `config/alns_config.yaml`.
