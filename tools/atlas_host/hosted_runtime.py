"""Unregistered hosted workflow wiring. No listener, thread or child on import."""

from copy import deepcopy
import json
import math
import os
import shutil
import select
import socket
import struct
import threading
import time
from urllib.parse import parse_qs, urlsplit

from .cache import binding
from .common import canonical, fields, integer, require
from .profile_api import PrivateProfileAPI
from .profile_os import (
    LinuxProfileHooks,
    ProcessBook,
    FrozenBinary,
    strict_json,
    available_bytes,
    require_platform,
)
from .profile_service import ProfileService
from .runtime_adapter import FixtureHost, check_fixture
from .static_models import StaticPolicy
from .supervisor import Supervisor
from .validation_policy import check_start, check_memory, BoundValidationPolicy
from .registry import reservation
from .profile_observation import SpawnObservationPending
from .dense_static_admission import BoundDenseStaticAdmission, check_cache_available
from .lifetime_guard import LifetimeGuard
from .static_operation import StaticOperation as StaticOperation

ROUTE_KIND = {
    "/tile": ("tile", "tile"),
    "/api/calibrate": ("calibration", "calibration"),
    "/api/model": ("model", "metadata"),
    "/api/progress": ("progress", "metadata"),
    "/api/tensor-status": ("tensor-status", "metadata"),
    "/api/view": ("view", "metadata"),
    "/api/binding": ("binding", "metadata"),
    "/api/inspect": ("inspect", "inspect"),
}


class NativeChannel:
    def __init__(
        self,
        stream,
        child,
        supervisor,
        service,
        book,
        *,
        clock=time.monotonic,
        wait=select.select,
    ):
        self.stream, self.child, self.supervisor, self.service, self.book = (
            stream,
            child,
            supervisor,
            service,
            book,
        )
        self.clock, self.wait = clock, wait
        self.failed = False
        self.settled_cpu = 0.0
        self.cpu_seen = 0.0
        self.token = None
        self.slot = None
        self.lock = threading.RLock()

    def _io(self, data, size, deadline, *, check=None):
        result = bytearray()
        while data if data is not None else len(result) < size:
            if check is not None:
                check()
            remaining = deadline - self.clock()
            require(remaining > 0, "Native transport deadline expired")
            self.wait(
                [self.stream] if data is None else [],
                [self.stream] if data is not None else [],
                [],
                remaining,
            )
            if check is not None:
                check()
            try:
                if data is not None:
                    n = self.stream.send(data)
                    require(n > 0, "Native channel closed")
                    data = data[n:]
                else:
                    part = self.stream.recv(size - len(result))
                    require(bool(part), "Native channel closed")
                    result.extend(part)
            except BlockingIOError:
                continue
        return bytes(result)

    def _reap_owned(self, token):
        # All token transitions are serialized by self.lock. Repeated/late
        # observations of a completed operation are no-ops, not new watchers.
        if self.token is not token:
            return self.token is None
        sample = self.child.sample()
        if not sample["reaped"] or self.child.unexpected:
            return False
        observed = self.book.resources()
        if not observed["all_owned_accounted"] or not observed["descendants_clear"]:
            return False
        self.supervisor.renderer_reaped(token, reaped=True, descendants_clear=True)
        token.release()
        self.token = None
        if self.slot is not None:
            self.slot.disarm()  # Nonwaiting: safe inside the current callback.
        return True

    def _fail_owned(self, token):
        self.failed = True
        self.child.stop()
        if self.token is token:
            token.poison()
            self._reap_owned(token)

    def idle_pulse(self):
        with self.lock:
            if self.failed:
                return
            cpu = self.child.sample()["cpu_seconds"]
            require(
                type(cpu) in (int, float)
                and math.isfinite(cpu)
                and cpu >= self.cpu_seen
                and cpu >= self.settled_cpu,
                "Idle CPU observation uncertain",
            )
            if hasattr(self.child, "stats") and not self.child.reaped:
                raw = self.child.stats(self.child.pid)
                require(
                    math.isfinite(raw["cpu"])
                    and raw["cpu"] >= self.cpu_seen
                    and raw["start"] == self.child.start,
                    "Idle reader identity/CPU uncertain",
                )
            self.cpu_seen = cpu
            require(
                cpu - self.settled_cpu < 4.0, "Unsettled idle CPU allowance exhausted"
            )

    def _static_read(self, path, context, operation):
        require(
            type(operation) is StaticOperation
            and operation.app.supervisor is self.supervisor,
            "Private static operation required",
        )
        parsed = urlsplit(path)
        require(
            not parsed.scheme
            and not parsed.netloc
            and not parsed.fragment
            and parsed.path in ROUTE_KIND,
            "Native route unavailable",
        )
        route, kind = ROUTE_KIND[parsed.path]
        query = parse_qs(parsed.query, keep_blank_values=True, max_num_fields=12)
        require(
            all(len(v) == 1 for v in query.values()) and operation.token.kind == kind,
            "Native route/token kind differs",
        )
        operation.check()
        require(not self.failed, "Native renderer cleanup pending")
        command = self.supervisor.native_command(
            operation.token,
            {"route": route, "query": {k: v[0] for k, v in query.items()}},
            operation.grant.remaining()["wall_ms"],
        )
        raw = canonical(command)
        require(len(raw) <= 16384, "Native command exceeds bound")
        operation.sent = True
        self._io(
            struct.pack("!I", len(raw)) + raw,
            None,
            operation.grant.deadline,
            check=operation.check,
        )
        length = struct.unpack(
            "!I", self._io(None, 4, operation.grant.deadline, check=operation.check)
        )[0]
        require(0 < length <= 16384, "Native header exceeds bound")
        header = strict_json(
            self._io(None, length, operation.grant.deadline, check=operation.check)
        )
        fields(header, ("ack", "status", "mime", "body_bytes"))
        integer(header["body_bytes"], 0, 2 * 1024**2)
        require(
            header["status"] in (200, 400)
            and header["mime"] in ("image/png", "application/json"),
            "Invalid native response",
        )
        body = self._io(
            None, header["body_bytes"], operation.grant.deadline, check=operation.check
        )
        operation.check()
        self.supervisor.native_ack(operation.token, header["ack"])
        # ACK is native idle, not handler publication. Token/watch remain owned.
        return header["status"], body, header["mime"]

    def read(self, path, context, *, operation=None):
        if operation is not None:
            return self._static_read(path, context, operation)
        with self.lock:
            require(not self.failed, "Native renderer cleanup pending")
        parsed = urlsplit(path)
        require(
            not parsed.scheme and not parsed.netloc and not parsed.fragment,
            "Local native route required",
        )
        require(parsed.path in ROUTE_KIND, "Native route unavailable")
        route, kind = ROUTE_KIND[parsed.path]
        query = parse_qs(parsed.query, keep_blank_values=True, max_num_fields=12)
        require(all(len(v) == 1 for v in query.values()), "Duplicate native query")
        grant, meter = self.service.admission()
        try:
            token = self.supervisor.acquire(kind, context)
            with self.lock:
                if self.failed:
                    # This operation has sent no command and created no child.
                    self.supervisor.admission_aborted(token, no_child_created=True)
                    token.release()
                    raise ValueError("Native renderer cleanup pending")
                self.token = token
        except BaseException:
            meter.freeze()
            raise
        baseline = self.child.cpu

        def guard():
            with self.lock:
                if self.token is not token:
                    return
                try:
                    sample = self.child.sample()
                    grant.sample_child(max(0, sample["cpu_seconds"] - baseline))
                    resources = self.book.resources()
                    require(
                        resources["all_owned_accounted"]
                        and resources["descendants_clear"]
                        and resources["rss_bytes"]
                        + self.supervisor.charged_snapshot_bytes()
                        <= 768 * 1024**2
                        and resources["available_bytes"] >= 13 * 1024**3 // 4,
                        "Native resource envelope",
                    )
                    grant.remaining()
                    require(
                        not self.failed and token.current(),
                        "Native transport uncertain",
                    )
                except BaseException:
                    self._fail_owned(token)

        try:
            with self.lock:
                # A callback cannot race assignment of its own watch slot.
                self.slot = self.service.watchdog.bind(guard, grant.deadline)
                command = self.supervisor.native_command(
                    token,
                    {"route": route, "query": {k: v[0] for k, v in query.items()}},
                    grant.remaining()["wall_ms"],
                )
            raw = canonical(command)
            require(len(raw) <= 16384, "Native command exceeds bound")
            self._io(struct.pack("!I", len(raw)) + raw, None, grant.deadline)
            length = struct.unpack("!I", self._io(None, 4, grant.deadline))[0]
            require(0 < length <= 16384, "Native header exceeds bound")
            header = strict_json(self._io(None, length, grant.deadline))
            fields(header, ("ack", "status", "mime", "body_bytes"))
            integer(header["body_bytes"], 0, 2 * 1024**2)
            require(
                header["status"] in (200, 400)
                and header["mime"] in ("image/png", "application/json"),
                "Invalid native response",
            )
            body = self._io(None, header["body_bytes"], grant.deadline)
            guard()
            with self.lock:
                require(
                    not self.failed and self.token is token, "Native operation expired"
                )
                slot = self.slot
            # Never wait for a callback while holding the lock it needs.
            slot.close()
            with self.lock:
                require(
                    not self.failed and self.token is token, "Native operation expired"
                )
                grant.remaining()
                self.supervisor.native_ack(token, header["ack"])
                token.release()
                self.token = None
            return header["status"], body, header["mime"]
        except BaseException:
            with self.lock:
                self._fail_owned(token)
                if self.token is token and (self.slot is None or self.slot.closed):
                    # Only unresolved ownership needs continued escalation/reap.
                    # A proven terminal operation never receives a replacement.
                    self.slot = self.service.watchdog.bind(guard, grant.deadline)
            raise
        finally:
            meter.freeze()

    def stop(self):
        with self.lock:
            self.failed = True
            self.child.stop()
            sample = self.child.sample()
            if not sample["reaped"] or self.child.unexpected:
                return False
            if self.token is not None and not self._reap_owned(self.token):
                return False
            slot = self.slot
        if slot is not None:
            slot.close()  # Genuine watchdog failure is still a cleanup refusal.
        self.stream.close()
        return True


class HostedRenderer:
    def __init__(
        self,
        entry,
        cache,
        binary,
        book,
        supervisor,
        service,
        diagnostics=None,
        *,
        operation=None,
    ):
        require(book.settled(), "Previous owned process cleanup pending")
        self.stream, peer = socket.socketpair()
        self.stream.setblocking(False)
        self.channel = None
        self.child = None
        try:
            binary.check()
            argv = [
                str(binary.path),
                "hosted-renderer",
                "--model",
                entry["root"],
                "--cache",
                str(cache),
                "--name",
                entry["name"],
                "--revision",
                entry["manifest"]["revision"],
                "--channel-fd",
                str(peer.fileno()),
            ]
            kwargs = {"pass_fds": (peer.fileno(),), "receipt": False}
            if diagnostics is not None:
                kwargs["diagnostics"] = diagnostics
            self.child = book.spawn(argv, **kwargs)
            if operation is not None:
                operation.add_child(self.child)
            self.channel = NativeChannel(
                self.stream, self.child, supervisor, service, book
            )
            if operation is not None:
                operation.add_reader(self)
        except BaseException:
            if self.child is not None:
                self.child.stop()
            self.stream.close()
            raise
        finally:
            peer.close()

    def initialize(self):
        self.child.initialize()

    def alive(self):
        return not self.child.sample()["reaped"] and not self.channel.failed

    def ready(self):
        return self.alive()  # First bounded command validates actual startup.

    def read(self, path, *, operation=None):
        return self.channel.read(path, "hosted-renderer", operation=operation)

    def stop(self):
        return self.channel.stop()


class HostedContexts:
    def __init__(self, host, lock):
        self.host, self.lock = host, lock
        self.bindings = {}

    def authorize(self, model, context, tab):
        with self.lock:
            host = self.host
            require(
                not host.stopping
                and host.entry is not None
                and host.context == context
                and host.entry["model_id"] == model
                and host._owns(context, tab)
                and host.leases.get(tab, 0) > host.clock(),
                "Current live tab ownership required",
            )

    def remember(self, model, context, value):
        selected = binding(value)
        with self.lock:
            require(
                context == self.host.context
                and model == self.host.entry["model_id"]
                and selected["source_identity"] == self.host.source_identity
                and selected["model_identity"] == self.host.model_identity,
                "Native context binding mismatch",
            )
            # A bounded current selection cache, never a file/model catalog scan.
            self.bindings = {canonical(selected): deepcopy(selected)}

    def resolve(self, data):
        with self.lock:
            self.authorize(data["model_id"], data["context_id"], data["tab_capability"])
            selected = binding(data["binding"])
            require(
                self.bindings.get(canonical(selected)) == selected,
                "Read current native binding before admission",
            )
            entry = deepcopy(self.host.entry)
            check_fixture(entry)

            def check():
                check_fixture(entry)
                require(
                    self.host.context == data["context_id"]
                    and self.host.entry == entry,
                    "Source context changed",
                )

            return {
                "root": entry["root"],
                "revision": entry["manifest"]["revision"],
                "cache": str(self.host.cache_root),
                "binding": selected,
                "check": check,
            }


class HostedApplication:
    """Complete assembly, intentionally not constructed by an installed launcher.

    start_threads starts only the coordinator contexts. A future reviewed launcher
    supplies the qualified binary receipt, binds HTTPServer to 127.0.0.1, and owns
    shutdown until close() confirms every registered process/handle is gone.
    """

    def __init__(
        self,
        registry,
        cache,
        binary_path,
        binary_sha,
        *,
        startup_diagnostics=None,
        launch_policy=None,
        dense_policy=None,
    ):
        # Fail before process inventory, binary I/O, threads or child creation.
        self.platform = require_platform()
        self.diagnostics = startup_diagnostics
        self.lock = threading.RLock()
        self.supervisor = Supervisor()
        self.book = ProcessBook(diagnostics=startup_diagnostics)
        self.binary = FrozenBinary(binary_path, binary_sha)
        self.launch_policy = launch_policy
        require(
            dense_policy is None
            or type(dense_policy) is BoundDenseStaticAdmission
            and dense_policy.registry is registry
            and launch_policy is None,
            "Same-registry sealed static policy required",
        )
        self.dense_policy = dense_policy
        if dense_policy is not None:
            dense_policy.check_binary(self.binary)
        self.static_operations = []
        if launch_policy is not None:
            require(
                type(launch_policy) is BoundValidationPolicy,
                "Owner-reviewed recipe required",
            )
            launch_policy.check_binary(self.binary)
        # Surface registered candidates without granting dense launches. No
        # static_admission hook is installed; the fixture recipe remains exact.
        self.host = FixtureHost(
            registry,
            cache,
            self._renderer,
            static_policy=StaticPolicy(registry),
            static_admission=self._admit_static if dense_policy is not None else None,
            dense_policy=dense_policy,
        )
        self.contexts = HostedContexts(self.host, self.lock)
        self.profiles = ProfileService(
            self.supervisor,
            self.contexts,
            lambda source, slot: LinuxProfileHooks(
                self.book, self.binary, source, slot, launch_policy=self.launch_policy
            ),
            diagnostics=startup_diagnostics,
        )
        self.lifetime = LifetimeGuard(
            self.book,
            self.supervisor,
            self.profiles.request_shutdown,
            diagnostics=startup_diagnostics,
        )
        self.profiles.watchdog.set_lifetime(
            self._lifetime_pulse if dense_policy is not None else self.lifetime.pulse
        )
        self.profiles.owner_cleanup = self._owned_shutdown_step
        self.api = PrivateProfileAPI(self.profiles, self.contexts.authorize)

    def _admit_static(self, prepared):
        require(
            type(self.dense_policy) is BoundDenseStaticAdmission
            and not self.lifetime.stopping,
            "Static admission disabled",
        )
        token = self.supervisor.active
        op = next(
            (
                op
                for op in self.static_operations
                if op.token is token and op.prepared is prepared
            ),
            None,
        )
        require(
            op is not None
            and not op.failed
            and token.kind == "metadata"
            and token.current(),
            "Exact pending startup reservation required",
        )
        op.check()
        self.dense_policy.admit(prepared)
        self.dense_policy.check_binary(self.binary)
        reuse = (
            self.host.reader is not None
            and self.host.entry == prepared.entry
            and self.host.view_kind == "static"
            and self.host.static_prepared is prepared
        )
        if reuse:
            # The exact owned live reader already holds the native cache lock.
            # No new cold-start reservation or competing lock probe is needed.
            require(
                self.host.reader.channel in op.channels,
                "Current static reader ownership required",
            )
        else:
            check_cache_available(self.host.cache_root / prepared.entry["model_id"])
            check_start(
                None, available_bytes(), shutil.disk_usage(self.host.cache_root).free
            )
        reservation(shutil.disk_usage(self.host.cache_root).free, metadata=65536)
        return True

    def begin_static(self, grant, kind):
        require(
            self.dense_policy is not None and not self.lifetime.stopping,
            "Static owner admission disabled",
        )
        op = StaticOperation(self, grant, kind)
        return op

    def _lifetime_pulse(self):
        self.lifetime.pulse()
        if self.dense_policy is None:
            return
        try:
            reader = self.host.reader
            try:
                if (
                    reader is not None
                    and self.host.view_kind == "static"
                    and not self.supervisor.busy()
                ):
                    reader.channel.idle_pulse()
            except BaseException:
                self.host.stopping = True
                self.host.leases.clear()
                self.host._revoke_readiness()
                reader.channel.failed = True
                reader.child.stop()
            # Lifetime execution also owns failed settlement after watch.close().
            # No replacement watch or renewed deadline is created.
            for op in self.static_operations:
                if op.aborting and not op.finished:
                    op.pulse()
            self.static_operations = [
                op for op in self.static_operations if not op.finished
            ]
        except BaseException:
            self.lifetime.failed = self.lifetime.stopping = True
            self.lifetime.pulse()  # Existing shutdown/reap owner retains uncertainty.

    def _renderer(self, entry, path, *, prepared=None, operation=None):
        require(not self.lifetime.stopping, "Hosted provider shutdown pending")
        if prepared is None:
            check_fixture(entry)
            if self.launch_policy is not None:
                self.launch_policy.check_entry(entry)
        else:
            require(
                type(getattr(self, "dense_policy", None)) is BoundDenseStaticAdmission
                and type(operation) is StaticOperation
                and operation.app is self
                and self.supervisor.active is operation.token
                and operation.token.kind == "metadata"
                and operation.prepared is prepared
                and prepared.entry == entry,
                "Exact private startup reservation required",
            )
            operation.check()
            self.dense_policy.admit(prepared)
            self.dense_policy.check_binary(self.binary)
            check_cache_available(path)
            require(
                self.book.settled() and self.supervisor.profile_session is None,
                "Competing owned work remains",
            )
            check_start(
                None, available_bytes(), shutil.disk_usage(self.host.cache_root).free
            )
            reservation(shutil.disk_usage(self.host.cache_root).free, metadata=65536)
            operation.mutated = True
            operation.check()
        return HostedRenderer(
            entry,
            path,
            self.binary,
            self.book,
            self.supervisor,
            self.profiles,
            self.diagnostics,
            operation=operation,
        )

    def start_threads(self):
        require_platform()
        check_start(
            self.launch_policy,
            available_bytes(),
            shutil.disk_usage(self.host.cache_root).free,
        )
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        self.profiles.start_threads()  # Both new contexts inherit this one CPU.

    def tick(self):
        self.lifetime.pulse()
        self.book.cleanup(stopping_only=True)
        with self.lock:
            if self.lifetime.stopping:
                self.host.close()
                return
            # A profile owns the source renderer lease until cleanup. Source
            # changes/cancel/expiry are handled by the independent profile owner.
            if (
                self.supervisor.active is None
                or self.supervisor.active.kind != "profile"
            ):
                self.host.tick()

    def _owned_shutdown_step(self):
        # Owner context/harness, never the independent watchdog. Nonblocking reap;
        # source handles and storage remain charged until their close is proven.
        for op in self.static_operations:
            if not op.finished:
                op.abort()
        with self.lock:
            host_closed = self.host.close()
        owned_closed = self.book.cleanup()
        return (
            host_closed
            and owned_closed
            and not self.supervisor.busy()
            and self.supervisor.charged_snapshot_bytes() == 0
        )

    def close(self):
        self.lifetime.request_shutdown()
        self._owned_shutdown_step()
        # ProfileService cannot stop its independent lifetime watch until all
        # source/child/snapshot cleanup above is confirmed, even when idle.
        return self.profiles.close()
