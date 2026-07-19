"""Tests for embedded-vs-verification benchmark aggregation helpers."""

from __future__ import annotations

import math

from sensitivity.robust_embedding_vs_verification_speed_cost import (
    TrialRow,
    build_pair_rows,
    summarize_pairs,
)


def test_build_pair_rows_and_summary():
    rows = [
        TrialRow(
            instance="a.txt",
            seed=1,
            gamma=3,
            method="embedded",
            runtime_sec=10.0,
            best_cost=100.0,
            feasible=1,
            iterations_executed=200,
            termination_reason="iterations_completed",
        ),
        TrialRow(
            instance="a.txt",
            seed=1,
            gamma=3,
            method="verification",
            runtime_sec=8.0,
            best_cost=102.0,
            feasible=1,
            iterations_executed=200,
            termination_reason="iterations_completed",
        ),
        TrialRow(
            instance="a.txt",
            seed=2,
            gamma=3,
            method="embedded",
            runtime_sec=9.0,
            best_cost=99.0,
            feasible=1,
            iterations_executed=200,
            termination_reason="iterations_completed",
        ),
        TrialRow(
            instance="a.txt",
            seed=2,
            gamma=3,
            method="verification",
            runtime_sec=10.0,
            best_cost=98.0,
            feasible=1,
            iterations_executed=200,
            termination_reason="iterations_completed",
        ),
    ]

    pairs = build_pair_rows(rows)
    assert len(pairs) == 2
    assert math.isclose(pairs[0].speedup_verification_over_embedded, 1.25, rel_tol=1e-9)
    assert math.isclose(pairs[0].cost_delta_verification_minus_embedded, 2.0, rel_tol=1e-9)

    summary = summarize_pairs(pairs)
    overall = summary[0]
    assert overall["scope"] == "overall"
    assert overall["pair_count"] == 2
    assert overall["mean_speedup_verification_over_embedded"] > 1.0
