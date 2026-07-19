# Revision Code and Experiment Plan

This document records code-side work only. Manuscript and response text changes are tracked separately in `revision_paper_response_plan.md`.

## Scope

The revision experiments should be rerun under a reproducible protocol while avoiding duplicate computation. The main targets are:

1. Recalibrate ALNS, TS, and GA with a systematic method rather than manual tuning.
2. Rerun all ALNS, TS, and GA comparison tables in the manuscript.
3. Rerun sensitivity experiments.
4. Rerun Monte Carlo robustness simulations and add runtime columns.
5. Preserve all raw logs, configs, and result tables for traceability.

## Proposed Output Root

Create a new result root for this revision:

```text
code/results/revision_20260610/
  manifest/
  configs/
  logs/
  tuning/
  main_tables/
  sensitivity/
  monte_carlo/
  derived_tables/
```

Recommended files:

```text
manifest/run_manifest.csv
manifest/run_manifest.jsonl
manifest/result_index.csv
configs/alns/final_config.yaml
configs/ga/final_config.json
configs/ts/final_config.json
logs/<experiment_id>.log
derived_tables/table_alns_milp_n10.csv
derived_tables/table_metaheuristics_n25_n50.csv
derived_tables/table_metaheuristics_n75_n100.csv
derived_tables/table_mc_nd_with_runtime.csv
derived_tables/table_mc_ndc_with_runtime.csv
```

## Run Identity and Deduplication

Every run should have a deterministic identity:

```text
experiment_id = sha1(
  algorithm
  + instance
  + seed
  + config_hash
  + code_commit
  + experiment_family
)
```

Before launching a run:

1. Compute `config_hash` from the complete algorithm config snapshot.
2. Check `manifest/run_manifest.csv`.
3. Reuse a previous successful result only if `algorithm`, `instance`, `seed`, `config_hash`, `code_commit`, and `experiment_family` all match.
4. Mark interrupted or failed runs as `failed` or `partial`, never silently overwrite them.

Minimum manifest columns:

```text
experiment_id, experiment_family, algorithm, instance, seed, config_hash,
code_commit, command, start_time, end_time, runtime_seconds, status,
result_path, stdout_path, stderr_path, notes
```

## Parameter Calibration Protocol

### ALNS

Use a two-stage calibration protocol:

1. RSM/DOE screening on a reduced parameter set.
2. Bayesian optimization for final selection.

RSM is useful for low-dimensional continuous factors, not for the full 30+ dimensional ALNS space. Use it only for the most influential parameters:

```text
w_percent
cooling_rate_initial
cooling_rate_final
eta
r_lower
r_upper_small
local_search_frequency
depot_bonus
```

Candidate implementation:

- Use Central Composite Design or Box-Behnken Design.
- Fit a quadratic response model with main effects and pairwise interactions.
- Objective: multi-seed mean cost plus a stability penalty.
- Report fitted importance qualitatively in the response, not as main manuscript results.

Bayesian optimization:

- Continue using Optuna TPE.
- Use training/tuning instances only, e.g. `R_30_10_1`, `R_30_10_3`, `R_30_10_5`.
- Validate on independent seeds and all five `Instance10` cases.
- Objective should be stability-aware:

```text
score = mean_cost + lambda_std * std_cost + infeasibility_penalty
```

Recommended `lambda_std`: 1.0 to 1.5.

### TS and GA

TS and GA must be tuned before comparison with ALNS.

TS parameters to tune:

```text
tabu_tenure
max_iterations or time_limit
neighborhood_sample_size
diversification trigger
aspiration setting
drone-specific move probability
```

GA parameters to tune:

```text
population_size
generations
tournament_size
crossover_rate
mutation_rate
elite_size
truck_route_crossover_rate
drone_task_mutation_rate
route_segment_swap_rate
max_stagnation
```

Use the same training instances and validation protocol as ALNS. The final comparison must use fixed tuned parameters for each algorithm.

## Main Experiments to Rerun

### ALNS vs MILP

Target manuscript table: `tab:algo_performance`.

Instances:

```text
Instance10/R_30_10_1 ... R_50_10_5
```

Protocol:

- MILP: record incumbent objective, runtime, status, MIP gap, and ALNS evaluator verification result.
- ALNS: 5 independent seeds per instance.
- Report best objective, mean objective, average runtime.
- Preserve route records under `results/records/` or the revision result root.

### ALNS vs GA vs TS

Target manuscript tables:

```text
tab:medium_scale
tab:large_scale
```

Instances:

```text
Instance25, Instance50, Instance75, Instance100
R_30, R_40, R_50 regions
5 instances per region
```

Protocol:

- Use the final tuned config for each algorithm.
- Use identical instance sets and seed lists.
- Record best, mean, std, feasible ratio, and runtime in raw data.
- Manuscript table may keep objective-only layout unless runtime is requested by reviewers.

### Sensitivity Experiments

Target manuscript figures:

```text
battery_sensitivity_plot.pdf
drone_count_sensitivity_plot.pdf
```

Rerun:

- Battery capacity sensitivity on all 15 `n=25` instances.
- Drone fleet size sensitivity on all 15 `n=25` instances.

Protocol:

- Use final ALNS config.
- Use the same seed list for all parameter levels.
- Store both trial-level and aggregated CSV files.

### Monte Carlo Robustness

Target manuscript tables:

```text
tab:replay_table6_nd
tab:replay_table6_ndc
```

Rerun:

- `n=25`, regions `30x30`, `40x40`, `50x50`.
- Gamma values: `0, 1, 2, 3`.
- Distributions: ND and NDC.
- 1000 replay scenarios per instance.

Add runtime data:

```text
solve_runtime_seconds
replay_runtime_seconds
total_runtime_seconds
```

For the manuscript tables, add one concise column:

```text
AvgTime(s)
```

Use `total_runtime_seconds` if the goal is to show the full computational effect of robustness. Keep separate solve/replay runtime columns in raw CSV for later discussion.

## Existing ALNS Tuning Assessment

### Existing Evidence

The repository already contains several ALNS tuning result families:

```text
code/results/bayesian_tuning/
code/results/bayesian_tuning_class/
code/results/bayesian_tuning_stable/
code/results/bayesian_tuning_r3_nonlinear/
```

Important observed validation summaries:

| Result family | Validation runs | Mean cost | Min | Max | Mean runtime |
|---|---:|---:|---:|---:|---:|
| `bayesian_tuning` | 15 | 72.208 | 56.24 | 100.88 | not selected for current deadline model |
| `bayesian_tuning_class` | 15 | 73.522 | 56.24 | 103.66 | 5.81s |
| `bayesian_tuning_stable` | 50 | 74.664 | 56.24 | 127.84 | 5.02s |
| `bayesian_tuning_r3_nonlinear` | 50 | 75.237 | 56.24 | 126.88 | 5.05s |

The current `code/config/alns_config.yaml` matches the stable-tuned family more closely than the class-only phase-2 config:

```text
w_percent = 23.72
cooling_rate_initial = 0.995709
cooling_rate_final = 0.983861
local_search.frequency = 6
drone.priority = 2.57
drone.bonus.depot = 2.29
drone.bonus.multi_customer = 4.08
```

These match `code/results/bayesian_tuning_stable/best_config_stable_phase2_20260605_144902.yaml`.

The class-only Bayesian phase-2 config had a lower 15-run mean on its validation set:

```text
mean = 73.522 over 5 instances x 3 seeds
```

The stable config used a stronger stability validation:

```text
mean = 74.664 over 5 instances x 10 seeds
```

This is a more defensible final config for the paper, even if its mean is slightly worse on the smaller 3-seed validation.

### Is the Current ALNS Tuning Optimal?

No. The current evidence does not support claiming global or near-global optimality of the ALNS parameter setting.

What the evidence supports:

1. The tuned ALNS is materially better than the previous manual/default setting under the revised class-based deadline model.
2. The stable-tuned configuration is more defensible than the class-only 3-seed winner because it uses a stability-aware objective and a 50-run validation.
3. The current config is likely a good practical setting for the current codebase and small-instance calibration set.

Main limitations:

1. Tuning instances are only `Instance10`; the main heuristic comparison includes `n=25, 50, 75, 100`.
2. The tuning objective uses only a small number of representative instances.
3. Several tuning families use different deprivation-cost assumptions, so they should not be mixed in the response without qualification.
4. Search budgets are modest: class phase 2 has 40 trials, stable phase 2 has 30 trials, and r3 nonlinear phase 2 has 15 trials.
5. RSM has not yet been run, so the response cannot honestly claim response-surface calibration until we add that experiment.
6. Current validation reports cost and runtime, but not enough statistical comparison against the manual/default config under the exact same seeds.

Recommended conclusion:

```text
The current ALNS parameters should be treated as a stability-tuned Bayesian configuration,
not as a proven global optimum. For the revision, we can state that manual tuning was
replaced by a systematic calibration protocol, and that the selected configuration was
validated on independent seeds before all main experiments were rerun.
```

## Code Tasks

### New or Updated Scripts

1. Add a revision runner that writes manifest rows before and after each run.
2. Add config hashing utilities.
3. Add ALNS RSM screening script.
4. Add TS tuning script.
5. Add GA tuning script.
6. Add metaheuristic comparison rerun script using fixed tuned configs.
7. Update Monte Carlo replay aggregation to compute runtime columns.
8. Add table export scripts that generate manuscript-ready CSV and LaTeX snippets.

### Existing Scripts to Reuse

```text
code/scripts/bayesian_tune_alns.py
code/scripts/bayesian_tune_stable.py
code/heuristics/ga/run_ga.py
code/heuristics/tabu_search/run_tabu.py
code/sensitivity/battery_sensitivity.py
code/sensitivity/drone_count_sensitivity.py
code/sensitivity/tools/run_replay_from_best_bank.py
code/sensitivity/tools/export_table6_csv_to_tex.py
```

### Quality Checks

Before using outputs in the manuscript:

1. Confirm all planned runs have `status=success`.
2. Confirm no result row mixes old and new config hashes.
3. Confirm all algorithms use the same instance and seed sets.
4. Confirm all reported objective values are feasible under the ALNS evaluator.
5. Confirm Monte Carlo table runtime is averaged over the same rows as the cost/reliability metrics.
6. Keep raw trial-level CSVs even if the manuscript reports only aggregated values.

