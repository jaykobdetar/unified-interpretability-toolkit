"""Use the concrete pure ledger with the existing native JSON projection."""

from typing import Any, Callable, assert_type
from atlas_host.budget import RemainingWork, WorkGrant
from atlas_host.profile_adapter import native_progress


def ledger_client(clock: Callable[[], float], cpu: Callable[[], float]) -> WorkGrant:
    grant = WorkGrant(12, 5000, 4000, clock=clock, cpu_clock=cpu)
    assert_type(grant.remaining(), RemainingWork)
    assert_type(grant.heartbeat(), None)
    assert_type(grant.advance(3, lambda count, deadline: count), int)
    return grant


def projection_client(
    grant: WorkGrant, model: str, selected: dict[str, Any], native: dict[str, Any]
) -> dict[str, Any]:
    result = native_progress(model, selected, native, grant.remaining())
    assert_type(result, dict[str, Any])
    return result
