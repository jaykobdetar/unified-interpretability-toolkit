"""Strict clients preserve selected catalog records and complete worker outcomes."""

from collections.abc import Mapping
from typing import Any, TypedDict
from analytics.source import Catalog, ModelOutliers
from analytics.svd_summary import Summary
from analytics.worker import (
    RegionReport,
    analyze_request,
    catalog_from_payload,
    validate_request,
)


class TensorMetadata(TypedDict):
    id: int
    name: str
    dtype: str
    shape: list[int]


def selected(data: Any, tensors: list[TensorMetadata]) -> TensorMetadata | None:
    return validate_request(data, tensors)


def catalog(payload: Mapping[str, Any]) -> Catalog:
    return catalog_from_payload(payload)


def report(payload: Mapping[str, Any]) -> RegionReport | ModelOutliers | Summary:
    return analyze_request(payload)
