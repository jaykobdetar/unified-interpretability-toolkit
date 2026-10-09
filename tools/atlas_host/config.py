"""Owner machine configuration. Validation never applies limits or starts work."""

from copy import deepcopy
from os import PathLike
from pathlib import Path
from typing import Any

from . import limits as _limits

from .common import fields, integer, label, read_json, require

# Byte values are explicit. These reproduce, rather than replace, runtime guards.
GIB = 1024**3
MIB = 1024**2
LOCAL_LIMITS = {
    "cpu_count": _limits.HOST_CONFIG_CPU_COUNT,
    "numeric_workers": _limits.HOST_CONFIG_NUMERIC_WORKERS,
    "heavy_jobs": _limits.HOST_CONFIG_HEAVY_JOBS,
    "rust_address_space_bytes": _limits.HOST_CONFIG_RUST_ADDRESS_SPACE_BYTES,
    "rust_available_floor_bytes": _limits.HOST_CONFIG_RUST_AVAILABLE_FLOOR_BYTES,
    "build_address_space_bytes": _limits.HOST_CONFIG_BUILD_ADDRESS_SPACE_BYTES,
    "build_jobs": _limits.HOST_CONFIG_BUILD_JOBS,
    "build_start_available_bytes": _limits.HOST_CONFIG_BUILD_START_AVAILABLE_BYTES,
    "browser_tree_rss_bytes": _limits.HOST_CONFIG_BROWSER_TREE_RSS_BYTES,
    "browser_start_available_bytes": _limits.HOST_CONFIG_BROWSER_START_AVAILABLE_BYTES,
    "stop_available_bytes": _limits.HOST_CONFIG_STOP_AVAILABLE_BYTES,
    "inference_rss_bytes": _limits.HOST_CONFIG_INFERENCE_RSS_BYTES,
    "inference_address_space_bytes": _limits.HOST_CONFIG_INFERENCE_ADDRESS_SPACE_BYTES,
    "inference_start_available_bytes": _limits.HOST_CONFIG_INFERENCE_START_AVAILABLE_BYTES,
    "inference_wall_ms": _limits.HOST_CONFIG_INFERENCE_WALL_MS,
    "inference_cpu_ms": _limits.HOST_CONFIG_INFERENCE_CPU_MS,
    "prompt_tokens": _limits.HOST_CONFIG_PROMPT_TOKENS,
    "new_tokens": _limits.HOST_CONFIG_NEW_TOKENS,
    "client_lease_ms": _limits.HOST_CONFIG_CLIENT_LEASE_MS,
    "inference_queue": _limits.HOST_CONFIG_INFERENCE_QUEUE,
    "analytics_address_space_bytes": _limits.HOST_CONFIG_ANALYTICS_ADDRESS_SPACE_BYTES,
    "analytics_rss_bytes": _limits.HOST_CONFIG_ANALYTICS_RSS_BYTES,
    "analytics_start_available_bytes": _limits.HOST_CONFIG_ANALYTICS_START_AVAILABLE_BYTES,
    "analytics_wall_ms": _limits.HOST_CONFIG_ANALYTICS_WALL_MS,
    "analytics_cpu_ms": _limits.HOST_CONFIG_ANALYTICS_CPU_MS,
    "disk_reserve_bytes": _limits.HOST_CONFIG_DISK_RESERVE_BYTES,
    "tile_disk_bytes": _limits.HOST_CONFIG_TILE_DISK_BYTES,
    "tile_files": _limits.HOST_CONFIG_TILE_FILES,
    "pending_headers": _limits.HOST_CONFIG_PENDING_HEADERS,
    "header_bytes": _limits.HOST_CONFIG_HEADER_BYTES,
    "header_deadline_ms": _limits.HOST_CONFIG_HEADER_DEADLINE_MS,
    "dispatch_queue": _limits.HOST_CONFIG_DISPATCH_QUEUE,
    "numeric_queue": _limits.HOST_CONFIG_NUMERIC_QUEUE,
    "write_deadline_ms": _limits.HOST_CONFIG_WRITE_DEADLINE_MS,
    "coordinator_body_bytes": _limits.HOST_CONFIG_COORDINATOR_BODY_BYTES,
    "upstream_response_bytes": _limits.HOST_CONFIG_UPSTREAM_RESPONSE_BYTES,
}


def validate_config(value: dict[str, Any], base: str | PathLike[str]) -> dict[str, Any]:
    fields(value, ("version", "profile", "bind", "ports", "paths"), ("limits",))
    integer(value["version"], 1, 1)
    require(value["profile"] == "local-v1", "Only unchanged local-v1 is enabled")
    require(value["bind"] == "127.0.0.1", "Only existing loopback binding is enabled")
    fields(value["ports"], ("coordinator", "renderer"))
    ports = [integer(p, 1, 65535) for p in value["ports"].values()]
    require(len(set(ports)) == len(ports), "Ports must be distinct")
    # Match the current coordinator's reserved viewer ports.
    require(not set(ports) & {8774, 8775, 8785}, "Port reserved by current viewer")
    fields(value["paths"], ("registry", "cache"))
    base = Path(base).resolve()
    paths: dict[str, str] = {}
    for name, raw in value["paths"].items():
        label(raw, 4096)
        path = Path(raw).expanduser()
        paths[name] = str((base / path).resolve())
    require(paths["registry"] != paths["cache"], "Registry and cache paths must differ")
    supplied = value.get("limits", {})
    fields(supplied, (), LOCAL_LIMITS)
    for key, number in supplied.items():
        integer(number)
        require(number == LOCAL_LIMITS[key], "Resource changes require staged review")
    return {**deepcopy(value), "paths": paths, "limits": deepcopy(LOCAL_LIMITS)}


def load_config(path: str | PathLike[str]) -> dict[str, Any]:
    path = Path(path)
    return validate_config(read_json(path, 16384), path.parent)


def capabilities(config: dict[str, Any]) -> dict[str, Any]:
    """Sanitized informational projection, not enforcement or authorization."""
    return {
        "version": 1,
        "profile": config["profile"],
        "limits": deepcopy(config["limits"]),
        "downloads": False,
        "gpu": False,
        "resident_activation": False,
        "http_routes_active": False,
    }
