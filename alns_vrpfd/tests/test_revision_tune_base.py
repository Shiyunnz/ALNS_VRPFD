"""Regression tests for revision comparison helpers."""

from __future__ import annotations

from types import SimpleNamespace

from revision.tune_base import ClassWeightedEvaluator


def test_class_weighted_evaluator_uses_configured_deprivation_parameters(monkeypatch):
    calls = []

    def fake_deprivation_cost(tau, supply_class, **kwargs):
        calls.append((tau, supply_class, kwargs))
        return 1.0

    monkeypatch.setattr("revision.tune_base.deprivation_cost", fake_deprivation_cost)

    evaluator = ClassWeightedEvaluator.__new__(ClassWeightedEvaluator)
    evaluator._node_classes = {1: "medicine"}
    evaluator._customer_lookup = {
        1: SimpleNamespace(optimal_time=1.0, latest_time=5.0),
    }
    evaluator._time_tolerance = 1e-6
    evaluator._cost_lambda = 30.0
    evaluator._cost_rho = 0.20833333333333334
    evaluator._cost_normalized = True
    evaluator._is_customer = lambda node_id: node_id == 1

    evaluator._compute_delay_penalty(
        {0: SimpleNamespace(arrival_times={1: 2.0})},
        {},
    )

    assert calls == [
        (
            1.0,
            "medicine",
            {
                "cost_lambda": 30.0,
                "rho": 0.20833333333333334,
                "normalized": True,
            },
        )
    ]
