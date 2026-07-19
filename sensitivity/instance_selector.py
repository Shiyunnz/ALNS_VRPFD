"""Shared instance selection helpers for sensitivity scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

VALID_SCOPES = {"all", "region", "single"}
VALID_REGIONS = {"30", "40", "50"}


def _normalize_regions(regions_text: str | None) -> set[str]:
    if regions_text is None:
        return set(VALID_REGIONS)
    values = [x.strip() for x in str(regions_text).split(",") if x.strip()]
    if not values:
        return set(VALID_REGIONS)
    normalized = set(values)
    invalid = sorted(normalized - VALID_REGIONS)
    if invalid:
        raise ValueError(
            f"Invalid region value(s): {invalid}. Allowed: {sorted(VALID_REGIONS)}"
        )
    return normalized


def _extract_region(stem: str) -> str | None:
    # Expected pattern: R_<region>_<size>_<id>
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0] == "R" and parts[1].isdigit():
        return parts[1]
    return None


def collect_instance_paths_with_scope(
    instance_dirs: Iterable[str | Path],
    *,
    scope: str = "all",
    regions_text: str | None = None,
    instance_name: str | None = None,
    excluded_stems: Iterable[str] | None = None,
) -> list[str]:
    """Collect instance files with unified scope filters.

    scope:
    - all: include all *.txt files under instance dirs
    - region: include files whose name pattern has region in --regions
    - single: include exactly one target instance by --instance-name
    """
    if scope not in VALID_SCOPES:
        raise ValueError(f"Invalid scope: {scope}. Allowed: {sorted(VALID_SCOPES)}")

    regions = _normalize_regions(regions_text) if scope == "region" else set()

    target_raw = (instance_name or "").strip()
    target_stem = Path(target_raw).stem if target_raw else ""
    if scope == "single" and not target_raw:
        raise ValueError("--instance-name is required when --instance-scope single")

    excluded = set(excluded_stems or [])
    collected: list[str] = []

    for directory in instance_dirs:
        dir_path = Path(directory)
        if not dir_path.exists():
            if dir_path.is_file() and dir_path.suffix == ".txt":
                candidates = [dir_path]
            else:
                raise FileNotFoundError(f"算例目录不存在: {dir_path}")
        elif dir_path.is_file():
            candidates = [dir_path] if dir_path.suffix == ".txt" else []
        else:
            candidates = sorted(dir_path.glob("*.txt"))

        for instance_file in candidates:
            stem = instance_file.stem
            name = instance_file.name
            full = str(instance_file)

            if stem in excluded:
                continue

            if scope == "all":
                collected.append(full)
                continue

            if scope == "region":
                region = _extract_region(stem)
                if region in regions:
                    collected.append(full)
                continue

            # scope == single
            if (
                stem == target_stem
                or name == target_raw
                or full == target_raw
            ):
                collected.append(full)

    unique_sorted = sorted(set(collected))

    if scope == "single":
        if not unique_sorted:
            raise ValueError(f"未找到指定算例: {instance_name}")
        if len(unique_sorted) > 1:
            raise ValueError(
                f"找到多个同名算例，请用完整路径指定 --instance-name。匹配数: {len(unique_sorted)}"
            )

    if not unique_sorted:
        if scope == "region":
            raise ValueError(f"未找到区域 {sorted(regions)} 的算例文件。")
        raise ValueError("未在指定目录中找到任何 .txt 算例文件，请检查目录路径是否正确。")

    return unique_sorted
