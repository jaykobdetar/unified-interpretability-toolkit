"""Strict clients of optional SVD availability and numeric result records."""

from collections.abc import Mapping, Sequence
from typing import Any
from analytics import svd


def computed(backend: Any, values: Sequence[float]) -> tuple[list[float], float | None]:
    original: svd.Fit = svd.compute_with_numpy(backend, values, 2, 2, 7)["original"]
    return original["singular_values"], original["rank_one_residual_energy_fraction"]


def optional(
    values: Sequence[float], shape: Sequence[int], region: Mapping[str, int]
) -> tuple[bool, str]:
    report = svd.run(values, shape, region, seed=7)
    if report["available"]:
        return True, report["scope"]
    return False, report["reason"]
