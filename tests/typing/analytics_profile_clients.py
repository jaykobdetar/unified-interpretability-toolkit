"""Strict clients of profile options and their catalog call boundary."""

from collections.abc import Mapping
from pathlib import Path
from analytics.core import AnalyticsReport
from analytics.profiles import ProfileOptions, resolve_local
from analytics.source import Catalog


def resolved(root: str | Path) -> tuple[ProfileOptions, str | None]:
    return resolve_local(root)


def selected(catalog: Catalog, region: Mapping[str, int]) -> AnalyticsReport:
    options, _ = resolve_local(catalog.root)
    return catalog.region("matrix", region, **options)
