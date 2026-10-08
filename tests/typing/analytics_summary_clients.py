"""Strict clients of summary records and identity-preserving validation."""

from collections.abc import Mapping
from typing import Any
from analytics import svd_summary as summary


def packet(
    model: Mapping[str, Any], tensor: Mapping[str, Any], region: Mapping[str, int]
) -> tuple[str, int, list[int]]:
    result = summary.skeleton(model, tensor, region, 7)
    key: str = result["cache_key"]
    count: int = result["coverage"]["visited_values"]
    positions: list[int] = result["preview"]["positions"]
    return key, count, positions


def validated(
    report: summary.Summary,
    model: Mapping[str, Any],
    tensor: Mapping[str, Any],
    region: Mapping[str, int],
) -> summary.Summary:
    return summary.validate(report, model=model, tensor=tensor, region=region, seed=7)


def energy(report: summary.Summary) -> tuple[float, float | None]:
    original: summary.Fit = report["results"]["original"]
    return original["frobenius_energy"], original["rank_one_residual_energy_fraction"]
