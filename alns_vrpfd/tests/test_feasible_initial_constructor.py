"""Tests for the deadline-aware feasible initial solution constructor."""

from __future__ import annotations

from alns_vrpfd.model.feasible_initializer import (
    build_feasible_initial_solution,
)
from alns_vrpfd.model.initializer import build_two_phase_initial_solution
from revision.tune_base import load_instance_for_tuning


def _violation_count(evaluator, solution) -> int:
    return len(evaluator.evaluate_with_details(solution).delay_breakdown.violations)


def test_feasible_initial_constructor_finds_feasible_r50_50_case():
    instance, evaluator, _ = load_instance_for_tuning(
        "R_50_50_4",
        seed=7381,
        instance_dir="Instance50",
    )

    solution, diagnostics = build_feasible_initial_solution(instance, evaluator)
    result = evaluator.evaluate_solution(solution)

    assert result.feasible
    assert diagnostics.feasible
    assert diagnostics.final_violations == 0
    assert diagnostics.initial_violations > diagnostics.final_violations


def test_feasible_initial_constructor_finds_feasible_multi_customer_r50_50_case():
    instance, evaluator, _ = load_instance_for_tuning(
        "R_50_50_2",
        seed=7381,
        instance_dir="Instance50",
    )

    solution, diagnostics = build_feasible_initial_solution(instance, evaluator)
    result = evaluator.evaluate_solution(solution)

    assert result.feasible
    assert diagnostics.feasible
    assert diagnostics.final_violations == 0
    assert any(len(task.customers()) > 1 for task in solution.drone_tasks)


def test_feasible_initial_constructor_improves_hard_r50_50_case():
    instance, evaluator, _ = load_instance_for_tuning(
        "R_50_50_1",
        seed=7381,
        instance_dir="Instance50",
    )
    old_initial = build_two_phase_initial_solution(instance)
    old_violations = _violation_count(evaluator, old_initial)

    solution, diagnostics = build_feasible_initial_solution(instance, evaluator)
    new_violations = _violation_count(evaluator, solution)

    assert diagnostics.initial_violations == old_violations
    assert new_violations < old_violations
    assert diagnostics.final_violations == new_violations
    assert diagnostics.final_lateness < diagnostics.initial_lateness


def test_feasible_initial_constructor_keeps_drone_structure_valid_on_hard_case():
    instance, evaluator, _ = load_instance_for_tuning(
        "R_50_50_1",
        seed=7381,
        instance_dir="Instance50",
    )

    solution, _ = build_feasible_initial_solution(instance, evaluator)
    details = evaluator.evaluate_with_details(solution)

    assert not evaluator._has_drone_limit_violations(solution)
    assert not evaluator._has_drone_task_violations(solution)
    assert details.robustness.feasible
