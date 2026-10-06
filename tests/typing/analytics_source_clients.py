"""Strict clients of trusted catalog and model-summary records."""

from collections.abc import Mapping
from analytics.core import AnalyticsReport
from analytics.source import (
    Catalog,
    ModelOutliers,
    LayoutEvidence,
    read_layout_evidence,
)


def selected(catalog: Catalog, region: Mapping[str, int]) -> AnalyticsReport:
    return catalog.region("matrix", region, top=2, seed=7)


def cached(catalog: Catalog, report: AnalyticsReport) -> AnalyticsReport:
    return catalog.validate_cached(report)


def model(catalog: Catalog) -> tuple[ModelOutliers, int]:
    report = catalog.model_outliers(top=2, value_budget=8, seed=7)
    visited: int = report["coverage"]["visited_values"]
    return report, visited


def evidence(config: str, implementation: str) -> LayoutEvidence:
    return read_layout_evidence(config, implementation)[1]
