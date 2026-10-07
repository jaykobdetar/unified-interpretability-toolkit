"""The catalogue retains mutable, ordered containers and exact element shapes."""

from typing import assert_type

from atlas_host.host_assets import ASSETS, BUNDLE


def asset_entry(path: str) -> tuple[str, str]:
    assert_type(ASSETS, dict[str, tuple[str, str]])
    return ASSETS[path]


def bundle_entry(index: int) -> str:
    assert_type(BUNDLE, list[str])
    return BUNDLE[index]
