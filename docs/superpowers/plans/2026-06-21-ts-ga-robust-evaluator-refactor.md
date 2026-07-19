# TS/GA Robust Evaluator Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make TS and GA reuse an ALNS-style robust search evaluator with caching and candidate verification, reducing repeated full evaluator calls while preserving final feasibility.

**Architecture:** Add a small shared `SearchEvaluator` wrapper under `alns_vrpfd/evaluation/`. TS and GA will call this wrapper for penalized costs, robust feasibility caching, and optional candidate verification while continuing to use the canonical `Evaluator` for final costs.

**Tech Stack:** Python, pytest, existing `Evaluator`, existing `SubrouteRobustVerifier`, existing TS/GA implementations.

---

### Task 1: Shared SearchEvaluator

**Files:**
- Create: `alns_vrpfd/evaluation/search_evaluator.py`
- Test: `alns_vrpfd/tests/test_search_evaluator.py`

- [ ] Write failing tests that verify `SearchEvaluator.evaluate_solution()` returns the same feasible result as `Evaluator.evaluate_solution()`.
- [ ] Write failing tests that verify repeated `robust_feasible_cached()` calls hit the cache for the same drone-task signature.
- [ ] Implement `SearchEvaluator` with matrix cache, full evaluation delegation, robust cache, and penalized cost.
- [ ] Run `pytest alns_vrpfd/tests/test_search_evaluator.py -q`.

### Task 2: TS Integration

**Files:**
- Modify: `heuristics/tabu_search/tabu_search.py`
- Test: `alns_vrpfd/tests/test_tabu_search_time_limit.py`

- [ ] Add constructor argument `search_evaluator=None`.
- [ ] Route `_penalized_cost()` through `SearchEvaluator.penalized_cost()` when provided.
- [ ] Gate candidate acceptance with `SearchEvaluator.verify_candidate(base=current, candidate=neighbor)` before full scoring.
- [ ] Add regression test proving repeated TS scoring uses shared robust cache.
- [ ] Run `pytest alns_vrpfd/tests/test_tabu_search_time_limit.py -q`.

### Task 3: GA Integration

**Files:**
- Modify: `heuristics/ga/ga.py`
- Test: `alns_vrpfd/tests/test_tabu_search_time_limit.py`

- [ ] Add constructor argument `search_evaluator=None`.
- [ ] Route `_evaluate_individual()` through `SearchEvaluator.penalized_cost()` when provided.
- [ ] Replace mutation-time full evaluator calls with `SearchEvaluator.evaluate_solution()` and cache-aware feasibility checks.
- [ ] Add regression test proving GA can evaluate individuals through the shared evaluator and still preserves best feasible individual.
- [ ] Run `pytest alns_vrpfd/tests/test_tabu_search_time_limit.py -q`.

### Task 4: Smoke Benchmark

**Files:**
- No production files.

- [ ] Run `pytest alns_vrpfd/tests/test_search_evaluator.py alns_vrpfd/tests/test_tabu_search_time_limit.py -q`.
- [ ] Run a short `R_30_10_2`, seed `44` smoke for TS and GA and report feasibility, cost, runtime, and cache hits.
