"""Strict standalone-viewer budgets; validation never starts work or applies limits."""

import json

MIB = 1024**2
GIB = 1024**3
DEFAULTS = {
    "version": 1,
    "cpu_count": 1,
    "address_space_bytes": 768 * MIB,
    "available_floor_bytes": 3 * GIB,
    "disk_reserve_bytes": 25 * GIB,
    "workspace_bytes": 64 * MIB,
    "tile_cache_bytes": 2 * GIB,
    "tile_cache_files": 1000,
}
RANGES = {
    "version": (1, 1),
    "cpu_count": (1, 8),
    "address_space_bytes": (256 * MIB, 8 * GIB),
    "available_floor_bytes": (512 * MIB, 128 * GIB),
    "disk_reserve_bytes": (GIB, 1024 * GIB),
    "workspace_bytes": (64 * MIB, GIB),
    "tile_cache_bytes": (16 * MIB, 64 * GIB),
    "tile_cache_files": (64, 100000),
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
