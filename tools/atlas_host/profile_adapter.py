"""Pure native progress translation. No worker, routes, snapshots or admission.

The caller supplies an authenticated binding and *remaining* host work ledger
after accounting for admission, waits and all owned worker CPU. This function
never authorizes a native grant or interprets partial results as resumable.
"""

import math
from copy import deepcopy
from typing import Any, Mapping, cast

from .cache import binding
from .common import canonical, digest, fields, integer, require
from .progress import validate_progress


def native_progress(
    model_id: str,
    selected: dict[str, Any],
    native: dict[str, Any],
    remaining: Mapping[str, Any],
) -> dict[str, Any]:
    selected = binding(selected)
    require(
        type(native) is dict and len(canonical(native)) <= 16384,
        "Native progress exceeds small status bound",
    )
    require(
        native.get("schema") == "weight-atlas.strength.v1"
        and binding(cast(dict[str, Any], native.get("binding"))) == selected,
        "Native profile source mismatch",
    )
    digest(native.get("identity"))
    total = math.prod(selected["shape"][-2:])
    integer(cast(int, native.get("total_values")), total, total)
    visited = integer(cast(int, native.get("visited_values")), 0, total)
    available = integer(
        cast(int, native.get("authorized_remaining_values")), 0, total - visited
    )
    integer(cast(int, native.get("allocated_state_bytes")), 0, 32 * 1024**2)
    state = native.get("state")
    require(state in ("ready", "paused", "complete", "invalid"), "Unknown native state")
    require(
        type(native.get("complete")) is bool
        and native["complete"] == (state == "complete")
        and (state != "complete" or visited == total),
        "Inconsistent native completion",
    )
    require(
        (native.get("error") is not None) == (state == "invalid"),
        "Inconsistent native error",
    )
    for key in ("active_seconds", "remaining_active_ms"):
        require(
            type(native.get(key)) in (int, float)
            and math.isfinite(native[key])
            and 0 <= native[key] <= 2**53 / 1000,
            "Invalid native time",
        )
    control = native.get("control")
    require(
        type(control) is dict and control.get("algorithm") == "swap-or-not-8-v1",
        "Unknown paired control",
    )
    integer(cast(int, cast(dict[str, Any], control).get("seed")), 0, 2**32 - 1)
    fields(remaining, ("values", "wall_ms", "cpu_ms"))
    host = {key: integer(value) for key, value in remaining.items()}
    require(
        host["wall_ms"] <= 5000 and host["cpu_ms"] <= 4000,
        "Remaining work exceeds existing total limits",
    )
    work = {
        "values": min(available, host["values"]),
        "wall_ms": min(math.floor(native["remaining_active_ms"]), host["wall_ms"]),
        "cpu_ms": host["cpu_ms"],
    }
    error = None
    if state == "invalid":
        public_state, error = "error", "worker_error"
    elif state == "complete" and not (host["wall_ms"] and host["cpu_ms"]):
        public_state, error = "error", "resource_limit"
    elif state == "complete":
        public_state = "complete"
    elif state == "paused" or not all(work.values()):
        public_state, error = "partial", "budget_exhausted"
    else:
        public_state = "running"
    if public_state in ("error", "complete", "partial"):
        work = {"values": 0, "wall_ms": 0, "cpu_ms": 0}
    return validate_progress(
        {
            "model_id": model_id,
            "kind": "profile",
            "binding": deepcopy(selected),
            "state": public_state,
            "visited_values": visited,
            "total_values": total,
            "elapsed_active_ms": math.ceil(native["active_seconds"] * 1000),
            "remaining_authorized_work": work,
            "complete": public_state == "complete",
            "error": error,
        }
    )
