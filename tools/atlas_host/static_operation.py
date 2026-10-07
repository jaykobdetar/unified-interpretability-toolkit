"""Nominal static-operation ownership and bookkeeping, without host wiring."""

import math
from pathlib import Path
import shutil
import threading
from typing import Callable, Mapping, Protocol, TypeAlias, cast

from .common import require
from .dense_static_admission import BoundDenseStaticAdmission
from .profile_observation import SpawnObservationPending
from .profile_os import FrozenBinary, OwnedProcess, ProcessBook
from .profile_service import ProfileService, WatchSlot
from .registry import reservation
from .static_models import BoundStatic, PreparedStatic
from .supervisor import AdmissionGrant, Supervisor
from .validation_policy import check_memory

ChildSeries: TypeAlias = list[OwnedProcess | float]


class OperationStream(Protocol):
    def close(self) -> object: ...


class OperationChannel(Protocol):
    @property
    def child(self) -> OwnedProcess: ...
    @property
    def stream(self) -> OperationStream: ...

    failed: bool
    settled_cpu: float


class OperationReader(Protocol):
    @property
    def child(self) -> OwnedProcess: ...
    @property
    def channel(self) -> OperationChannel: ...


class OperationHost(Protocol):
    @property
    def reader(self) -> object | None: ...
    @property
    def context(self) -> str | None: ...
    @property
    def static_prepared(self) -> PreparedStatic | None: ...
    @property
    def static_bound(self) -> BoundStatic | None: ...
    @property
    def dense_policy(self) -> BoundDenseStaticAdmission | None: ...
    @property
    def leases(self) -> dict[str, float]: ...
    @property
    def clock(self) -> Callable[[], float]: ...
    @property
    def cache_root(self) -> Path: ...

    stopping: bool

    def _revoke_readiness(self) -> None: ...


class OperationApp(Protocol):
    @property
    def host(self) -> OperationHost: ...
    @property
    def supervisor(self) -> Supervisor: ...
    @property
    def profiles(self) -> ProfileService: ...
    @property
    def book(self) -> ProcessBook: ...
    @property
    def binary(self) -> FrozenBinary: ...


class OperationOwner(Protocol):
    @property
    def static_operations(self) -> list["StaticOperation"]: ...


class StaticOperation:
    """One original HTTP grant, existing token/watch, through publication.

    Callback never takes app.lock; it only observes/signals its exact children.
    Handler joins the watch outside this lock. Failed ownership stays reachable
    from the app until the independent callback confirms exact reap/closure.
    """

    def __init__(self, app: OperationApp, grant: AdmissionGrant, kind: str) -> None:
        require(grant is not None, "Original HTTP grant required")
        self.app, self.grant = app, grant
        self.lock = threading.RLock()
        self.children: list[ChildSeries] = []
        self.channels: list[OperationChannel] = []
        self.prepared: PreparedStatic | None = None
        self.context: str | None = None
        self.mutated = self.failed = self.aborting = self.finished = self.sent = False
        self.token = app.supervisor.acquire(kind, "dense-static")
        self.slot: WatchSlot | None = None
        if hasattr(app, "static_operations"):
            cast(OperationOwner, app).static_operations.append(self)
        try:
            old = app.host.reader
            if old is not None:
                self.add_reader(cast(OperationReader, old))
            self.slot = app.profiles.watchdog.bind(self.pulse, grant.deadline)
            self.check()
        except BaseException:
            self.mutated = bool(self.children)
            self.abort()
            raise

    def add_child(self, child: OwnedProcess, baseline: float = 0.0) -> None:
        with self.lock:
            require(
                not any(c is child for c, _, _ in self.children),
                "Duplicate static child",
            )
            self.children.append([child, baseline, baseline])

    def add_reader(self, reader: OperationReader) -> None:
        channel = reader.channel
        if not any(c is channel.child for c, _, _ in self.children):
            self.add_child(channel.child, channel.settled_cpu)
        if channel not in self.channels:
            self.channels.append(channel)

    def _observe(self, selected: OwnedProcess | None = None) -> None:
        total = 0.0
        for series in self.children:
            child, base, seen = series
            sample = cast(OwnedProcess, child).sample()
            cpu = sample["cpu_seconds"]
            if child is selected:
                require(
                    sample["reaped"] is False
                    and not cast(OwnedProcess, child).unexpected
                    and getattr(child, "stop_at", None) is None,
                    "Selected static reader is not live",
                )
            require(
                type(cpu) in (int, float)
                and math.isfinite(cpu)
                and cpu >= cast(float, seen) >= cast(float, base),
                "Static child CPU uncertain/regressed",
            )
            if not sample["reaped"] and hasattr(child, "stats"):
                raw = cast(OwnedProcess, child).stats(cast(OwnedProcess, child).pid)
                require(
                    type(raw["cpu"]) in (int, float)
                    and math.isfinite(raw["cpu"])
                    and raw["cpu"] >= cast(float, seen),
                    "Static raw CPU regressed/unavailable",
                )
                require(
                    cast(OwnedProcess, child).start is None
                    or raw["start"] == cast(OwnedProcess, child).start,
                    "Static reader identity differs",
                )
            series[2] = cpu
            total += cpu - cast(float, base)
        self.grant.sample_child(total)

    def check(self) -> None:
        with self.lock:
            require(
                not self.failed and not self.finished and self.token.current(),
                "Static operation unavailable",
            )
            selected: OwnedProcess | None = None
            if self.context is not None:
                host = self.app.host
                require(
                    not host.stopping
                    and host.context == self.context
                    and host.reader is not None
                    and any(
                        cast(OperationReader, host.reader).channel is ch
                        for ch in self.channels
                    )
                    and any(end > host.clock() for end in host.leases.values()),
                    "Static publication lease changed",
                )
                require(
                    self.prepared is not None and self.prepared is host.static_prepared,
                    "Current selected static preparation required",
                )
                cast(PreparedStatic, self.prepared).check()
                policy = getattr(self.app, "dense_policy", None)
                require(
                    type(policy) is BoundDenseStaticAdmission
                    and policy is host.dense_policy,
                    "Current same-owner sealed static policy required",
                )
                cast(BoundDenseStaticAdmission, policy).check_binary(self.app.binary)
                channel = cast(OperationReader, host.reader).channel
                require(
                    not channel.failed
                    and channel.child is cast(OperationReader, host.reader).child,
                    "Selected static reader ownership changed",
                )
                selected = channel.child
            elif self.prepared is not None and self.mutated:
                self.prepared.check()  # A prospective refusal must not stop the old reader.
            # Sample selected liveness/identity and CPU after source/seal work.
            # Retired old children may be reaped and stay in the CPU ledger.
            require(
                selected is None
                or any(child is selected for child, _, _ in self.children),
                "Selected reader CPU ownership unavailable",
            )
            self._observe(selected)
            try:
                observed: Mapping[str, object] = self.app.book.resources()
                pending = False
            except SpawnObservationPending as error:
                observed = error.observed
                pending = True
            check_memory(
                observed,
                self.app.supervisor.charged_snapshot_bytes(),
                allow_pending=pending,
            )
            require(
                self.app.supervisor.profile_session is None,
                "Static snapshot permission unavailable",
            )
            growth = 2 * 1024**2 if self.token.kind == "tile" else 65536
            reservation(
                shutil.disk_usage(self.app.host.cache_root).free, cache_growth=growth
            )
            self.grant.remaining()

    def check_publication(self) -> None:
        with self.lock:
            self.check()
            if self.context is not None:
                bound = self.app.host.static_bound
                require(
                    bound is not None and bound.prepared is self.prepared,
                    "Complete current selected static binding required",
                )

    def _fail(self) -> None:
        if self.context is not None:
            self.mutated = True  # Selected reuse also needs owned cleanup.
        self.failed = True
        self.grant.cancel()
        if self.app.supervisor.active is self.token:
            self.token.poison()
        if self.mutated or self.sent:
            for channel in self.channels:
                channel.failed = True
            host = self.app.host
            if host.reader is not None and any(
                cast(OperationReader, host.reader).channel is ch for ch in self.channels
            ):
                host.stopping = True
                host.leases.clear()
                host._revoke_readiness()
            for child, _, _ in self.children:
                try:
                    cast(OwnedProcess, child).stop()
                except (ValueError, OSError):
                    pass  # Retain ownership; subsequent reap still required.

    def pulse(self) -> None:
        with self.lock:
            if self.finished:
                return
            try:
                self.check()
            except BaseException:
                # A failed resource/CPU/grant observation must also stop the old
                # reader, even before replacement; ordinary prospective policy
                # refusal uses abort() without changing the valid old reader.
                self.mutated = True
                self._fail()
            if self.aborting:
                self._reap()

    def _reap(self) -> bool:
        if self.finished:
            return True
        if not self.mutated and not self.sent:
            self.app.supervisor.admission_aborted(self.token, no_child_created=True)
        else:
            try:
                for child, _, _ in self.children:
                    cast(OwnedProcess, child).stop()
                    if (
                        not cast(OwnedProcess, child).sample()["reaped"]
                        or cast(OwnedProcess, child).unexpected
                    ):
                        return False
                for channel in self.channels:
                    channel.stream.close()
                require(self.app.book.settled(), "Static owned cleanup pending")
                observed: Mapping[str, object] = self.app.book.resources()
                check_memory(observed, 0)
                self.app.supervisor.renderer_reaped(
                    self.token, reaped=True, descendants_clear=True
                )
            except (ValueError, OSError):
                return False
        self.token.release()
        self.finished = True
        if self.slot is not None:
            self.slot.disarm()
        return True

    def abort(self) -> bool:
        with self.lock:
            if self.finished:
                return True
            self.aborting = True
            self._fail()
            done = self._reap()
            slot = self.slot
        if done and slot is not None:
            slot.close()
        return done

    def publish(self, write: Callable[[], object]) -> None:
        try:
            self.check_publication()
            write()
            self.check_publication()
            # No callback may race final settlement; never join under self.lock.
            cast(WatchSlot, self.slot).close()
            with self.lock:
                self.check_publication()
                if not self.sent:
                    self.app.supervisor.admission_aborted(
                        self.token, no_child_created=True
                    )
                self.token.release()
                self.finished = True
                for channel in self.channels:
                    if not channel.failed:
                        channel.settled_cpu = next(
                            cast(float, s[2])
                            for s in self.children
                            if s[0] is channel.child
                        )
        except BaseException:
            self.abort()
            raise


StaticOperation.__module__ = __package__ + ".hosted_runtime"
StaticOperation.__init__.__module__ = __package__ + ".hosted_runtime"
StaticOperation.add_child.__module__ = __package__ + ".hosted_runtime"
StaticOperation.add_reader.__module__ = __package__ + ".hosted_runtime"
StaticOperation._observe.__module__ = __package__ + ".hosted_runtime"
StaticOperation.check.__module__ = __package__ + ".hosted_runtime"
StaticOperation.check_publication.__module__ = __package__ + ".hosted_runtime"
StaticOperation._fail.__module__ = __package__ + ".hosted_runtime"
StaticOperation.pulse.__module__ = __package__ + ".hosted_runtime"
StaticOperation._reap.__module__ = __package__ + ".hosted_runtime"
StaticOperation.abort.__module__ = __package__ + ".hosted_runtime"
StaticOperation.publish.__module__ = __package__ + ".hosted_runtime"
