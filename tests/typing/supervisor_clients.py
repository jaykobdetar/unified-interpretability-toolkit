"""Strict callers of the real supervisor and existing legacy implementations."""

from collections.abc import Callable, Mapping
from typing import assert_type

from analytics.service import AnalyticsJobs
from atlas_host.live_inference import Session
from atlas_host.supervisor import (
    AdmissionGrant,
    ComputeToken,
    CpuClock,
    CpuLedger,
    HostedLegacyLane,
    NativeCommand,
    ProfileOwner,
    RemainingGrant,
    Supervisor,
)


def accounting(
    meters: dict[str, Callable[[], float]], clock: Callable[[], float], cpu: CpuClock
) -> RemainingGrant:
    ledger = CpuLedger(meters)
    assert_type(ledger.total(), float)
    grant = AdmissionGrant(clock, cpu, wall_ms=4000, cpu_ms=3000)
    grant.sample_child(0.5)
    remaining = assert_type(grant.remaining(250), RemainingGrant)
    assert_type(remaining["wall_ms"], int)
    grant.cancel()
    return remaining


def snapshots(supervisor: Supervisor, owner: ProfileOwner) -> int:
    supervisor.retain_profile(owner, 16)
    charge = assert_type(supervisor.charged_snapshot_bytes(), int)
    supervisor.close_profile(owner, all_handles_closed=True)
    return charge


def native(
    supervisor: Supervisor, payload: object, acknowledgment: Mapping[str, object]
) -> NativeCommand:
    token = assert_type(supervisor.acquire("tile", "context"), ComputeToken)
    assert_type(token.current(), bool)
    command = assert_type(
        supervisor.native_command(token, payload, 1000), NativeCommand
    )
    assert_type(command["request"], object)
    supervisor.transport_lost(token)
    supervisor.native_ack(token, acknowledgment)
    supervisor.renderer_reaped(token, reaped=True, descendants_clear=True)
    token.release()
    assert_type(supervisor.busy(), bool)
    return command


def actual_legacy_implementations(
    supervisor: Supervisor, session: Session, jobs: AnalyticsJobs
) -> HostedLegacyLane:
    lane = HostedLegacyLane(supervisor, session, jobs)
    assert_type(lane.busy(), bool)
    token = assert_type(lane.reserve("inference", "context"), ComputeToken)
    lane.finish(token, reaped=True, descendants_clear=True, finalized=True)
    return lane
