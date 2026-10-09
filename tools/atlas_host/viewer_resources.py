"""Strict standalone-viewer budgets; validation never starts work or applies limits."""

import json

from . import limits as _limits

MIB = 1024**2
GIB = 1024**3
DEFAULTS = {
    "version": 1,
    "cpu_count": _limits.VIEWER_DEFAULT_CPU_COUNT,
    "address_space_bytes": _limits.VIEWER_DEFAULT_ADDRESS_SPACE_BYTES,
    "available_floor_bytes": _limits.VIEWER_DEFAULT_AVAILABLE_FLOOR_BYTES,
    "disk_reserve_bytes": _limits.VIEWER_DEFAULT_DISK_RESERVE_BYTES,
    "workspace_bytes": _limits.VIEWER_DEFAULT_WORKSPACE_BYTES,
    "tile_cache_bytes": _limits.VIEWER_DEFAULT_TILE_CACHE_BYTES,
    "tile_cache_files": _limits.VIEWER_DEFAULT_TILE_CACHE_FILES,
}
RANGES = {
    "version": (1, 1),
    "cpu_count": (_limits.VIEWER_MIN_CPU_COUNT, _limits.VIEWER_MAX_CPU_COUNT),
    "address_space_bytes": (
        _limits.VIEWER_MIN_ADDRESS_SPACE_BYTES,
        _limits.VIEWER_MAX_ADDRESS_SPACE_BYTES,
    ),
    "available_floor_bytes": (
        _limits.VIEWER_MIN_AVAILABLE_FLOOR_BYTES,
        _limits.VIEWER_MAX_AVAILABLE_FLOOR_BYTES,
    ),
    "disk_reserve_bytes": (
        _limits.VIEWER_MIN_DISK_RESERVE_BYTES,
        _limits.VIEWER_MAX_DISK_RESERVE_BYTES,
    ),
    "workspace_bytes": (
        _limits.VIEWER_MIN_WORKSPACE_BYTES,
        _limits.VIEWER_MAX_WORKSPACE_BYTES,
    ),
    "tile_cache_bytes": (
        _limits.VIEWER_MIN_TILE_CACHE_BYTES,
        _limits.VIEWER_MAX_TILE_CACHE_BYTES,
    ),
    "tile_cache_files": (
        _limits.VIEWER_MIN_TILE_CACHE_FILES,
        _limits.VIEWER_MAX_TILE_CACHE_FILES,
    ),
}


def validate(value: object) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError("resources must contain only supported budget fields")
    result: dict[str, int] = {**DEFAULTS, **value}
    for key, number in result.items():
        low, high = RANGES[key]
        if type(number) is not int or not low <= number <= high:
            raise ValueError(f"resources.{key} must be an integer from {low} to {high}")
    if result["workspace_bytes"] > result["address_space_bytes"] // 2:
        raise ValueError(
            "workspace budget must fit within half the address-space budget"
        )
    return result


def encode(value: object) -> str:
    return json.dumps(validate(value), separators=(",", ":"), sort_keys=True)
