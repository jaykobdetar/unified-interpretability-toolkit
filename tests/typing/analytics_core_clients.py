"""Strict clients of concrete core records and generic ranked mapping identity."""

from collections.abc import Mapping, Sequence
from analytics import core


def geometry(
    shape: Sequence[int], region: Mapping[str, int]
) -> tuple[int, int, int, int]:
    return core.geometry(shape, region)


def values(
    data: Sequence[float], shape: Sequence[int], region: Mapping[str, int]
) -> tuple[int, float, list[int]]:
    result = core.statistics(data, shape, region)
    row: core.AxisStatistic = core.ranked(result["rows"], 1)[0]
    count: int = row["count"]
    mean: float = row["mean_abs"]
    order: list[int] = result["row_order"]
    return count, mean, order


def report(
    data: Sequence[float], shape: Sequence[int], region: Mapping[str, int]
) -> tuple[str, int, bool]:
    result = core.analyze(
        data, source_identity="fixture", tensor="matrix", shape=shape, region=region
    )
    identity: str = result["cache_key"]
    visited: int = result["coverage"]["visited_values"]
    ready: bool = result["heads"]["available"]
    return identity, visited, ready


def head_dimension(heads: core.HeadLayout | core.Unavailable) -> int | None:
    if heads["available"]:
        return heads["head_dim"]
    return None
