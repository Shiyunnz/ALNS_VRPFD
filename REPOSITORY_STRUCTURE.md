# ResearchProject Repository Structure

This project is split into two independent Git repositories:

- `paper/`: Overleaf paper repository.
- `code/`: GitHub code and experiment repository.

The top-level `ResearchProject/` directory is a workspace container, not a Git
repository. Commit, pull, and push must be done inside `paper/` and `code/`
separately.

## Top-Level Workspace

```text
ResearchProject/
├── AGENTS.md                 # Workspace conventions for agents and scripts
├── paper/                    # LaTeX manuscript and response letter repo
└── code/                     # Python implementation, experiments, and results repo
```

## `paper/` Repository

```text
paper/
├── main.tex                  # Main manuscript source
├── R1_response.tex           # Reviewer response source
├── ref.bib                   # Bibliography database
├── figs/                     # Figures included by the manuscript
├── thumbnails/               # Elsevier/CAS template image assets
├── cas-*.{cls,bst,sty}       # Elsevier/CAS template files
├── reviewresponse.sty        # Response letter formatting helpers
└── *.pdf, *.aux, *.bbl, ...  # Local/Overleaf LaTeX build outputs
```

Notes:

- Per `AGENTS.md`, manuscript edits should keep old text commented with `%`.
- New or revised manuscript text should be wrapped with `\textcolor{red}{...}`.
- Reviewer-response excerpts should use the `changes` environment.

## `code/` Repository

```text
code/
├── alns_vrpfd/               # Main Python package
│   ├── core/                 # ALNS search logic and operators
│   ├── evaluation/           # Cost, timing, robustness, replay, search evaluators
│   ├── experiments/          # Experiment orchestration utilities
│   ├── instance/             # Instance and customer data abstractions
│   ├── mip/                  # MILP model and MILP runner utilities
│   ├── mip_deterministic/    # Deterministic MILP variants
│   ├── model/                # Solution, route, and initializer models
│   ├── tests/                # Pytest regression and integration tests
│   └── utils/                # IO and configuration helpers
├── heuristics/               # Non-ALNS baselines and metaheuristics
│   ├── ga/                   # Genetic algorithm implementation
│   └── tabu_search/          # Constraint-aware tabu search implementation
├── revision/                 # ALNS/TS/GA validation and tuning entry points
├── scripts/                  # One-off analyses, ablations, tuning, plotting scripts
├── sensitivity/              # Sensitivity experiment runners and plotters
├── case_study/               # Wenchuan case-study generation, runs, and plotting
├── config/                   # Runtime configuration files
├── data/                     # Benchmark and case-study instances
│   ├── Instance10/
│   ├── Instance25/
│   ├── Instance50/
│   ├── Instance75/
│   ├── Instance100/
│   ├── Solomon/
│   └── WenchuanCase/
├── docs/                     # Design notes, experiment plans, and method notes
├── figures/                  # Generated standalone paper/method figures
├── results/                  # Experiment outputs, checkpoints, manifests, records
├── software_copyright_doc/   # Software copyright documentation assets
├── run_alns.py               # Main ALNS command-line entry point
├── run_alns_milp_comparison.py
├── run_convergence_analysis.py
├── setup_path.py
└── .gitignore
```

## Important Runtime Paths

- Python interpreter: `/Users/minz/anaconda3/bin/python`
- ALNS config: `code/config/alns_config.yaml`
- Main TS implementation: `code/heuristics/tabu_search/tabu_search.py`
- Main ALNS entry point: `code/run_alns.py`
- ALNS/TS/GA validation runner: `code/revision/validate_alns_ts_ga.py`

## Result Organization

`code/results/` contains persistent experiment artifacts. Current naming
patterns are:

- `ablation_*`: operator or algorithm ablation results.
- `microbenchmark_*`: performance and verifier microbenchmarks.
- `revision_20260610/`: revision experiment checkpoints, manifests, and main
  comparison tables.
- `records/`: detailed per-run route and timing records.
- `convergence_analysis/`: convergence traces by instance.

`code/sensitivity/results_new/` contains sensitivity-study reruns and plot data.

## Git Synchronization

Use separate Git commands for each repository:

```bash
cd /Users/minz/Desktop/ResearchProject/code
git status
git add -A
git commit -m "<message>"
git pull --rebase origin main
git push origin main

cd /Users/minz/Desktop/ResearchProject/paper
git status
git pull --rebase origin master
git push origin master
```

Do not run Git commands from `/Users/minz/Desktop/ResearchProject` expecting both
repositories to be handled together; it is only the workspace container.
