# Revision Paper and Response Plan

This document records manuscript-side and reviewer-response-side changes only. Code and experiment pipeline tasks are tracked separately in `revision_code_experiment_plan.md`.

## Reviewer Issue

The reviewer is concerned that algorithm parameters were manually tuned and that the experimental evidence should be rerun or strengthened using a more formal calibration procedure.

The response should be transparent:

1. Acknowledge that the original submission used preliminary/manual calibration.
2. State that the revision adds systematic parameter calibration.
3. Explain that tuning experiments are separated from final evaluation experiments.
4. Rerun the comparison and sensitivity data using fixed tuned parameters.
5. Preserve tuning details in the response and appendix, not in the main result tables unless necessary.

## Response Strategy

Recommended response wording:

```text
We thank the reviewer for pointing out the need for a more transparent parameter
calibration procedure. In the original submission, the parameters of ALNS and the
benchmark heuristics were selected through preliminary manual calibration. In the
revision, we replaced this with a systematic calibration protocol. Specifically,
we first used response surface methodology on a reduced set of influential
continuous parameters to identify stable parameter ranges, and then applied
Bayesian optimization based on Optuna TPE for final configuration selection.
The calibration was conducted on a small set of representative instances, while
all reported comparison experiments were rerun using fixed calibrated parameters
on the benchmark instances. We also retained all run logs, configuration snapshots,
and trial-level results to improve reproducibility.
```

If RSM is not actually run, do not claim it was run. Use this wording instead:

```text
We designed the revised calibration protocol around response-surface screening
and Bayesian optimization. The Bayesian optimization component has been implemented
and validated; response-surface screening is reported as part of the revised
parameter-calibration methodology once the corresponding screening runs are completed.
```

## Main Manuscript Changes

### Section: Computational Experiments

Add a short paragraph before performance tables:

```text
To ensure a fair comparison, all metaheuristic parameters were fixed before
the final computational experiments. The parameters were selected using a
separate calibration set and were not adjusted on the test instances reported
in the following tables. Details of the calibration protocol are provided in
Appendix~X.
```

This paragraph should be added near the start of Section `sec:experiments`, before `Performance of the ALNS Algorithm`.

### Table: ALNS vs MILP

Target:

```text
Table~\ref{tab:algo_performance}
```

Required after rerun:

1. Replace all MILP objective, runtime, and status-dependent text if values change.
2. Replace all ALNS best, mean, and average runtime values.
3. Ensure the discussion distinguishes proven MILP optima from time-limit incumbents.
4. Mention ALNS evaluator verification if MILP-PWL inconsistency remains relevant.

Do not cite a MILP raw incumbent as an optimal benchmark unless Gurobi status proves optimality and ALNS evaluator verification confirms feasibility.

### Tables: ALNS vs GA vs TS

Targets:

```text
Table~\ref{tab:medium_scale}
Table~\ref{tab:large_scale}
```

Required after rerun:

1. Replace all ALNS, GA, and TS values.
2. Update the paragraph that states how many instances ALNS wins.
3. Update average improvement percentages over GA and TS.
4. State that TS and GA were independently calibrated under the same protocol.

Suggested wording:

```text
The benchmark TS and GA algorithms were calibrated using the same training
instances and validation criterion as ALNS. The final comparison uses fixed
parameters for all algorithms, and no parameter was adjusted based on the test
instances in Tables~\ref{tab:medium_scale} and~\ref{tab:large_scale}.
```

### Monte Carlo Robustness Tables

Targets:

```text
Table~\ref{tab:replay_table6_nd}
Table~\ref{tab:replay_table6_ndc}
```

Add one runtime column:

```text
AvgTime(s)
```

Current table layout:

```latex
\begin{tabular}{crrrrrrrr}
Region & $\Gamma$ & AvgCost & StdCost & MaxCost & MinCost & AvgUnserved & CompRate\% & AvgAbortReturn \\
```

Revised layout:

```latex
\begin{tabular}{crrrrrrrrr}
Region & $\Gamma$ & AvgCost & StdCost & MaxCost & MinCost & AvgUnserved & CompRate\% & AvgAbortReturn & AvgTime(s) \\
```

Update explanatory text:

```text
AvgTime(s) reports the average total computational time, including route
optimization and scenario replay evaluation, for each robustness setting.
```

Discussion to add after the two tables:

```text
The runtime column shows the computational price of robustness. Although larger
values of $\Gamma$ introduce stricter robust feasibility checks and can increase
route-construction time, the average runtime remains within a practical range
for the tested instance scale. Thus, the observed reliability gains are not
obtained at the cost of prohibitive computational effort.
```

Only use this conclusion if the rerun actually supports it.

### Sensitivity Analysis

Targets:

```text
Figure~\ref{fig:battery_sensitivity}
Figure~\ref{fig:drone_count_sensitivity}
```

Required after rerun:

1. Replace figure PDFs if regenerated.
2. Update text values for battery savings, drone-served nodes, and fleet-size savings.
3. Avoid overclaiming monotonicity unless all rerun values support it.
4. If results are noisier after rerun, phrase findings as "overall trend" rather than strict monotonic behavior.

## Appendix Methodology Section

Add an appendix subsection, tentatively titled:

```latex
\subsection{Parameter Calibration Protocol}
```

Recommended content structure:

1. Original submission used preliminary manual calibration.
2. Revision uses systematic calibration.
3. Tuning instances and evaluation instances are separated.
4. RSM screens a small set of continuous ALNS parameters.
5. Bayesian optimization selects final ALNS, TS, and GA configurations.
6. The objective minimizes multi-seed mean objective with a stability penalty.
7. Final experiments use fixed parameters.
8. All logs and configuration snapshots are stored for reproducibility.

Suggested appendix prose:

```text
The parameter calibration was conducted independently from the final benchmark
evaluation. Let $C_{a,i,s}(\theta)$ denote the objective value obtained by
algorithm $a$ on calibration instance $i$ and seed $s$ under parameter vector
$\theta$. The calibration objective minimized
$\bar{C}_{a}(\theta)+\lambda \sigma_{a}(\theta)$, where $\bar{C}_{a}$ and
$\sigma_{a}$ are the mean and standard deviation over calibration runs. Infeasible
solutions received a large penalty. This criterion favors parameter settings that
achieve both low objective values and stable performance across random seeds.
```

For ALNS:

```text
For ALNS, response-surface screening was first applied to a reduced set of
influential continuous parameters, including initial temperature, cooling rates,
adaptive learning rate, destruction ratios, local-search frequency, and drone
insertion bonuses. The screened ranges were then refined using Bayesian
optimization with a tree-structured Parzen estimator.
```

For TS and GA:

```text
For TS and GA, the same calibration framework was used with algorithm-specific
parameter spaces. TS parameters included tabu tenure, diversification settings,
and neighborhood sampling controls. GA parameters included population size,
generation limit, tournament size, crossover rate, mutation rate, elite size,
and drone-specific mutation probabilities.
```

## Response-Only Tuning Results

The tuning test results should be reported in the response letter, not in the main manuscript tables.

Recommended response-level table:

```text
Method | Calibration instances | Trials | Seeds | Objective | Best validation mean | Notes
Manual/default | same instances | - | 3 | mean cost | 80.46 | original preliminary setting
Bayesian TPE class | 3 instances | 40 phase-2 trials | 3 | mean cost | 73.52 | lower mean on short validation
Bayesian TPE stable | 3 instances | 30 phase-2 trials | 10 | mean + 1.5 std | 74.66 | selected for stability
```

Do not present these as final benchmark performance. Present them as calibration evidence only.

## Current ALNS Tuning Interpretation for Paper

Do not write:

```text
The ALNS parameters are optimal.
```

Write:

```text
The ALNS parameters were selected through a systematic calibration procedure
and validated on independent random seeds before being fixed for the final
experiments.
```

If more detail is needed:

```text
The selected configuration is not claimed to be globally optimal in the
hyperparameter space. Rather, it is a calibrated and stability-validated setting
that replaces the preliminary manual tuning used in the original submission.
```

This is technically defensible and avoids overclaiming.

## LaTeX Editing Rules

When applying these changes to `paper/main.tex`:

1. Do not delete original paragraphs.
2. Comment out replaced paragraphs with `%`.
3. Wrap new or modified text in `\textcolor{red}{...}`.
4. In `paper/R1_response.tex`, quote manuscript changes inside:

```latex
\begin{changes}
...
\end{changes}
```

## Checklist Before Applying to Manuscript

1. Confirm final ALNS config path and config hash.
2. Confirm final TS and GA config paths and config hashes.
3. Confirm all comparison tables have been rerun.
4. Confirm Monte Carlo tables include runtime.
5. Confirm sensitivity values match regenerated figures.
6. Confirm response text does not claim RSM results before RSM runs exist.
7. Confirm all new manuscript text is red and old text is retained as comments.

