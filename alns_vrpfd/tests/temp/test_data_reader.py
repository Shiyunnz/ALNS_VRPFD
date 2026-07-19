"""Temporary tests for the legacy instance data reader."""

from __future__ import annotations

from pathlib import Path

import pytest

from ...utils.io_utils import read_instance


def test_read_instance_dataset() -> None:
    """Ensure the reader loads the sample dataset into dense matrices."""
    path = Path("data/Instance10/R_30_10_1.txt")
    if not path.exists():
        pytest.skip("Sample dataset not available in workspace.")

    instance = read_instance(str(path), strategy="emergency")

    assert instance.customer_manager.depot_start == 0
    assert instance.customer_manager.depot_end == 11
    assert len(instance.customer_manager.customer_ids()) == 10

    matrix = instance.distance_matrix("truck")
    nodes = instance.distances.nodes()

    assert len(matrix) == len(nodes)
    assert matrix[nodes.index(0)][nodes.index(1)] == pytest.approx(12.6, rel=1e-6)

    time_matrix = instance.time_matrix("truck")
    expected_time = 12.6 / instance.vehicle_types()["truck"].speed
    assert time_matrix[nodes.index(0)][nodes.index(1)] == pytest.approx(expected_time, rel=1e-6)
