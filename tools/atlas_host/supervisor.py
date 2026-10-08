"""Inactive single-supervisor admission and accounting. No standalone/global lock.

No launcher imports this module. All clocks, process/receipt observations and
scheduling are supplied by the future supervised runtime and qualified separately.
"""

import math
import secrets
import threading

from .common import fields, integer, require

from collections.abc import Callable, Mapping
from typing import Protocol, TypedDict


class CpuClock(Protocol):
    def total(self) -> float: ...


class SnapshotStorage(Protocol):
    @property
    def owned_storage_bytes(self) -> int: ...


class ProfileJobOwner(Protocol):
    @property
    def store(self) -> SnapshotStorage: ...


class ProfileOwner(Protocol):
    @property
    def job(self) -> ProfileJobOwner: ...


class InferenceLane(Protocol):
    @property
    def process(self) -> object | None: ...

    @property
    def status(self) -> str: ...


class AnalyticsLane(Protocol):
    @property
    def busy(self) -> bool: ...


class RemainingGrant(TypedDict):
    wall_ms: int
    cpu_ms: int


class NativeCommand(TypedDict):
    version: int
    operation_id: str
    kind: str
    remaining_ms: int
    request: object


KINDS = {
    "profile",
    "inference",
    "analytics",
    "tile",
    "calibration",
    "overview",
    "inspect",
    "metadata",
}
NATIVE_KINDS = {"tile", "calibration", "overview", "inspect", "metadata"}


class CpuLedger:
    """Registered cumulative context clocks, queryable from ANY calling thread.

    Providers must freeze a final value before a context's OS clock disappears.
    Child user+system cumulative CPU is a separate series, replaced at reap.
    """

    def __init__(self, meters: dict[str, Callable[[], float]]) -> None:
        require(
            type(meters) is dict and set(meters) == {"admission", "owner", "watchdog"},
            "All job execution contexts require CPU meters",
        )
        self.meters = dict(meters)
        self.last: dict[str, float] = {}
        self.lock = threading.Lock()

    def total(self) -> float:
        with self.lock:
            total = 0.0
            for name, read in self.meters.items():
                value = read()
                require(
                    type(value) in (int, float)
                    and math.isfinite(value)
                    and value >= self.last.get(name, 0),
                    "Job CPU clock regressed/unavailable",
                )
                self.last[name] = value
                total += value
            return total


class AdmissionGrant:
    """Construct at HTTP admission BEFORE parsing, binding and job construction."""

    def __init__(
        self,
        clock: Callable[[], float],
        cpu: CpuClock,
        *,
        wall_ms: int = 5000,
        cpu_ms: int = 4000,
    ) -> None:
        self.clock, self.cpu = clock, cpu
        self.started = clock()
        self.cpu_start = cpu.total()
        integer(wall_ms, 1, 5000)
        integer(cpu_ms, 1, 4000)
        self.deadline = self.started + wall_ms / 1000
        self.cpu_limit = cpu_ms / 1000
        self.child_cpu = 0.0
        self.cancelled = False
        self.lock = threading.RLock()

    def sample_child(self, cumulative: float) -> None:
        with self.lock:
            require(
                type(cumulative) in (int, float)
                and math.isfinite(cumulative)
                and cumulative >= self.child_cpu,
                "Child CPU clock regressed",
            )
            self.child_cpu = cumulative

    def remaining(self, reserve_ms: int = 0) -> RemainingGrant:
        with self.lock:
            require(not self.cancelled, "Original admission cancelled")
            wall = self.deadline - self.clock() - reserve_ms / 1000
            cpu = self.cpu_limit - (self.cpu.total() - self.cpu_start) - self.child_cpu
            require(wall > 0 and cpu > 0, "Original admission grant exhausted")
            return {
                "wall_ms": math.floor(wall * 1000),
                "cpu_ms": math.floor(cpu * 1000),
            }

    def cancel(self) -> None:
        with self.lock:
            self.cancelled = True


class ComputeToken:
    def __init__(
        self, supervisor: "Supervisor", operation: str, kind: str, context: str
    ) -> None:
        self._supervisor = supervisor
        self.operation, self.kind, self.context = operation, kind, context
        self.poisoned = False
        self._completion = False

    def current(self) -> bool:
        with self._supervisor.lock:
            return self._supervisor.active is self and not self.poisoned

    def poison(self) -> None:
        with self._supervisor.lock:
            require(self._supervisor.active is self, "Stale compute token")
            self.poisoned = True

    def release(self) -> None:
        with self._supervisor.lock:
            require(
                self._supervisor.active is self and self._completion,
                "Confirmed completion/reap required before release",
            )
            self._supervisor.active = None


class Supervisor:
    """Exactly one numeric operation in THIS hosted workflow; never a job queue.

    Completion evidence methods are trusted adapter APIs, never visitor routes.
    Existing standalone launchers are outside this object and cannot use profiles.
    """

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.active: ComputeToken | None = None
        self.profile_session: ProfileOwner | None = None
        self.snapshot_reservation = 0
        self.stopping = False

    def prevent_admission(self) -> None:
        with self.lock:
            self.stopping = True
            if self.active is not None:
                self.active.poisoned = True  # Retain exact ownership until cleanup.

    def charged_snapshot_bytes(self) -> int:
        with self.lock:
            integer(self.snapshot_reservation)
            if self.profile_session is None:
                require(self.snapshot_reservation == 0, "Snapshot owner unavailable")
                return 0
            actual = self.profile_session.job.store.owned_storage_bytes
            integer(actual)
            return max(actual, self.snapshot_reservation)

    def retain_profile(self, owner: ProfileOwner, frame_bytes: int) -> None:
        with self.lock:
            require(not self.stopping, "Hosted provider is stopping")
            require(
                self.profile_session is None,
                "Another profile session retains its snapshot",
            )
            integer(frame_bytes, 1, 16 * 1024**2)
            self.profile_session = owner
            # Conservatively reserve BOTH old/new frames through retiring close.
            self.snapshot_reservation = 2 * frame_bytes

    def close_profile(self, owner: object, *, all_handles_closed: bool) -> None:
        with self.lock:
            require(
                self.profile_session is owner and all_handles_closed is True,
                "Profile storage cleanup unconfirmed",
            )
            self.profile_session = None
            self.snapshot_reservation = 0

    def acquire(
        self,
        kind: str,
        context: str,
        *,
        legacy_busy: Callable[[], bool] = lambda: False,
    ) -> ComputeToken:
        require(
            kind in KINDS and type(context) is str and bool(context),
            "Invalid hosted operation",
        )
        with self.lock:
            # Inference/analytics cleanup-uncertain state remains authoritative.
            require(not self.stopping, "Hosted provider is stopping")
            require(
                self.active is None and not legacy_busy(),
                "Hosted compute busy; no queue",
            )
            token = ComputeToken(self, secrets.token_hex(16), kind, context)
            self.active = token
            return token

    def _owns(self, token: ComputeToken) -> None:
        require(self.active is token, "Stale compute operation")

    def child_finished(
        self,
        token: ComputeToken,
        *,
        reaped: bool,
        descendants_clear: bool,
        finalized: bool,
    ) -> None:
        with self.lock:
            self._owns(token)
            require(
                token.kind not in NATIVE_KINDS,
                "Native work requires explicit acknowledgment",
            )
            require(
                reaped is True and descendants_clear is True and finalized is True,
                "Owned work is not finalized",
            )
            token._completion = True

    def admission_aborted(self, token: ComputeToken, *, no_child_created: bool) -> None:
        with self.lock:
            self._owns(token)
            require(no_child_created is True, "Admission cleanup is uncertain")
            token._completion = True

    def native_command(
        self, token: ComputeToken, payload: object, remaining_ms: int
    ) -> NativeCommand:
        with self.lock:
            self._owns(token)
            require(
                token.kind in NATIVE_KINDS and token.current(),
                "Native operation unavailable",
            )
            integer(remaining_ms, 1, 2**31 - 1)
            # Internal channel only. No owner capabilities or visitor paths.
            return {
                "version": 1,
                "operation_id": token.operation,
                "kind": token.kind,
                "remaining_ms": remaining_ms,
                "request": payload,
            }

    def native_ack(
        self, token: ComputeToken, acknowledgment: Mapping[str, object]
    ) -> None:
        with self.lock:
            self._owns(token)
            fields(
                acknowledgment, ("version", "operation_id", "complete", "numeric_idle")
            )
            require(
                token.kind in NATIVE_KINDS
                and acknowledgment["version"] == 1
                and type(acknowledgment["version"]) is int
                and acknowledgment["operation_id"] == token.operation
                and acknowledgment["complete"] is True
                and acknowledgment["numeric_idle"] is True,
                "Native completion acknowledgment mismatch",
            )
            token._completion = True

    def renderer_reaped(
        self, token: ComputeToken, *, reaped: bool, descendants_clear: bool
    ) -> None:
        with self.lock:
            self._owns(token)
            require(
                token.kind in NATIVE_KINDS
                and reaped is True
                and descendants_clear is True,
                "Owned renderer reap not confirmed",
            )
            token._completion = True

    def transport_lost(self, token: ComputeToken) -> None:
        # Neither a response timeout nor disconnected client proves native idle.
        token.poison()

    def busy(self) -> bool:
        with self.lock:
            return self.active is not None


class HostedLegacyLane:
    """Adapter for supervised Session/AnalyticsJobs, preserving old busy checks.

    No existing launcher is modified. Hooks must cover every hosted start and
    completion path before activation; this wrapper is not process-global proof.
    """

    def __init__(
        self, supervisor: Supervisor, inference: InferenceLane, analytics: AnalyticsLane
    ) -> None:
        self.supervisor, self.inference, self.analytics = (
            supervisor,
            inference,
            analytics,
        )

    def busy(self) -> bool:
        return (
            self.inference.process is not None
            or self.inference.status in ("loading", "running", "stopping")
            or self.analytics.busy
        )

    def reserve(self, kind: str, context: str) -> ComputeToken:
        require(kind in ("inference", "analytics"), "Wrong hosted legacy lane")
        return self.supervisor.acquire(kind, context, legacy_busy=self.busy)

    def finish(
        self,
        token: ComputeToken,
        *,
        reaped: bool,
        descendants_clear: bool,
        finalized: bool,
    ) -> None:
        require(not self.busy(), "Inference/analytics cleanup remains pending")
        self.supervisor.child_finished(
            token,
            reaped=reaped,
            descendants_clear=descendants_clear,
            finalized=finalized,
        )
        token.release()
