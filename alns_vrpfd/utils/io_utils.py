"""Utilities for reading instance data from disk."""

from __future__ import annotations

from alns_vrpfd.instance import InstanceManager, TimeWindowConfig

from .data_reader import InstanceDataReader

__all__ = ["read_instance"]


def read_instance(
    path: str,
    *,
    strategy: str = "class_based",
    config: TimeWindowConfig | None = None,
    apply_time_windows: bool = True,
) -> InstanceManager:
    """Read an instance file and return a populated InstanceManager."""
    reader = InstanceDataReader(
        time_window_strategy=strategy,
        time_window_config=config,
        apply_time_windows=apply_time_windows,
    )
    return reader.read_instance(path)
