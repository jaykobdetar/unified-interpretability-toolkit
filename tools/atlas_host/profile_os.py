"""Linux ownership hooks for the inactive hosted runtime; imports start nothing."""

import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import signal
import subprocess
import sys
import fcntl
import resource
import threading
import time

from .common import canonical, fields, require, _unique
from .memory import available_bytes
from .registry import fingerprint
from .profile_observation import SpawnObservationPending
from .startup_diagnostics import (
    StartupDiagnostics,
    StartupRecord,
    safe_error,
    error_chain,
    diagnose,
    diagnosed,
)

from collections.abc import Callable, Mapping, Sequence
from types import ModuleType
from typing import Any, IO, Protocol, TypedDict, cast


class PlatformFacts(TypedDict):
    executable: str
    version: str
    platform: str
    apis: dict[str, bool]


class ProcessStats(TypedDict):
    start: int
    cpu: float
    rss: int


class ResourceSnapshot(TypedDict):
    rss_bytes: int
    available_bytes: int
    all_owned_accounted: bool
    descendants_clear: bool


class ProfileSource(TypedDict):
    root: str | Path
    revision: str
    cache: str | Path
    binding: dict[str, Any]
    check: Callable[[], object]


class WatchRegistration(Protocol):
    def close(self) -> object: ...


class ProfileWatchdog(Protocol):
    def register(
        self, callback: Callable[[], object], deadline: float, /
    ) -> WatchRegistration: ...
    def close(self) -> object: ...


class SnapshotOwner(Protocol):
    def require_settled(self) -> object: ...
    @property
    def owned_storage_bytes(self) -> int: ...


class BinaryPolicy(Protocol):
    def check_binary(self, binary: "FrozenBinary", /) -> object: ...


MIB = 1024**2
GIB = 1024**3


# An explicit interpreter build must expose these APIs; version alone is not proof.
PLATFORM_APIS: dict[str, tuple[ModuleType, tuple[str, ...]]] = {
    "os": (
        os,
        (
            "pidfd_open",
            "wait4",
            "memfd_create",
            "MFD_ALLOW_SEALING",
            "MFD_CLOEXEC",
            "sched_getaffinity",
            "sched_setaffinity",
            "set_blocking",
            "pread",
        ),
    ),
    "signal": (signal, ("pidfd_send_signal",)),
    "time": (time, ("pthread_getcpuclockid", "clock_gettime")),
    "fcntl": (
        fcntl,
        (
            "F_ADD_SEALS",
            "F_GET_SEALS",
            "F_SEAL_WRITE",
            "F_SEAL_GROW",
            "F_SEAL_SHRINK",
            "F_SEAL_SEAL",
        ),
    ),
    "resource": (resource, ("RLIMIT_AS", "RLIMIT_CPU")),
    "hashlib": (hashlib, ("file_digest",)),
}


def platform_capabilities() -> PlatformFacts:
    """Metadata only: no descriptors, clock registration or kernel probes."""
    return {
        "executable": sys.executable,
        "version": sys.version,
        "platform": sys.platform,
        "apis": {
            module + "." + name: hasattr(owner, name)
            for module, (owner, names) in PLATFORM_APIS.items()
            for name in names
        },
    }


def require_platform() -> PlatformFacts:
    facts = platform_capabilities()
    missing = sorted(name for name, present in facts["apis"].items() if not present)
    require(
        facts["platform"] == "linux" and not missing,
        "Hosted provider requires an explicit compatible Linux Python; missing APIs: "
        + ", ".join(missing),
    )
    return facts


def strict_json(raw: str | bytes | bytearray) -> Any:
    return json.loads(
        raw,
        object_pairs_hook=_unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")),
    )


def proc_stat(pid: int) -> ProcessStats:
    raw = Path(f"/proc/{pid}/stat").read_text()
    parts = raw[raw.rindex(")") + 2 :].split()
    return {
        "start": int(parts[19]),
        "cpu": (int(parts[11]) + int(parts[12])) / os.sysconf("SC_CLK_TCK"),
        "rss": int(parts[21]) * os.sysconf("SC_PAGE_SIZE"),
    }


def children(pid: int) -> set[int]:
    tasks = list(Path(f"/proc/{pid}/task").iterdir())
    require(len(tasks) <= 64, "Owned thread inventory exceeds bound")
    found: set[int] = set()
    for task in tasks:
        raw = (task / "children").read_text()
        require(len(raw) <= 4096, "Owned child inventory exceeds bound")
        found.update(int(p) for p in raw.split())
    return found


class ThreadMeter:
    """Registered thread CPU clock, frozen by that thread before it exits."""

    def __init__(self, *, diagnostics: StartupDiagnostics | None = None) -> None:
        self.diagnostics = diagnostics
        self.clock_id: int | None = None
        self.final: float | None = None
        self.lock = threading.Lock()

    @diagnosed("profile_os.ThreadMeter.register")
    def register(self) -> None:
        with self.lock:
            require(
                self.clock_id is None and self.final is None,
                "CPU context already registered",
            )
            self.clock_id = time.pthread_getcpuclockid(threading.get_ident())

    @diagnosed("profile_os.ThreadMeter.read")
    def read(self) -> float:
        with self.lock:
            if self.final is not None:
                return self.final
            require(self.clock_id is not None, "CPU context unavailable")
            return time.clock_gettime(cast(int, self.clock_id))

    @diagnosed("profile_os.ThreadMeter.freeze")
    def freeze(self) -> None:
        with self.lock:
            self.final = time.clock_gettime(cast(int, self.clock_id))
            self.clock_id = None


class OwnedProcess:
    """One serialized wait4 owner. Never call Popen.poll()/wait()/communicate().

    The Popen handle is recorded before initialize can fail. Until wait4 reaps,
    the direct child PID cannot be reused; signals use pidfd after initialization.
    Unexpected descendants permanently mark cleanup uncertain for this runtime.
    """

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        *,
        clock: Callable[[], float] = time.monotonic,
        stats: Callable[[int], ProcessStats] = proc_stat,
        descendants: Callable[[int], set[int]] = children,
        wait4: Callable[[int, int], tuple[int, int, resource.struct_rusage]] = os.wait4,
        read: Callable[[int, int], bytes] = os.read,
        diagnostic: StartupRecord | None = None,
        diagnostics: StartupDiagnostics | None = None,
    ) -> None:
        self.process, self.pid = process, process.pid
        self.clock, self.stats, self.descendants = clock, stats, descendants
        self.wait4, self.read = wait4, read
        self.lock = threading.RLock()
        self.pidfd: int | None = None
        self.start: int | None = None
        self.reaped = False
        self.exit_code: int | None = None
        self.cpu = 0.0
        self.receipt: Any = None
        self.raw = bytearray()
        self.eof = process.stdout is None
        self.drainable = self.eof
        self.invalid = False
        self.stop_at: float | None = None
        self.kill_sent = False
        self.unexpected: set[int] = set()
        self.descendant_fds: dict[int, int] = {}
        self.diagnostics = diagnostics
        self.diagnostic = diagnostic
        self.stderr_eof = diagnostic is None
        self.stderr_ready = False
        if diagnostic is not None:
            diagnostic.update(pid=self.pid)

    @diagnosed("profile_os.OwnedProcess.initialize")
    def initialize(self) -> None:
        try:
            self._initialize()
        except BaseException as error:
            if self.diagnostic is not None:
                self.diagnostic.update(
                    initialize_error={
                        **safe_error(error),
                        "error_chain": error_chain(error),
                    }
                )
            raise

    def _initialize(self) -> None:
        with self.lock:
            self._drain_stderr()
            if self.process.stdout is not None:
                os.set_blocking(self.process.stdout.fileno(), False)
                self.drainable = True
            # A fast child may already have been reaped by the watchdog.
            if not self.reaped:
                self.start = self.stats(self.pid)["start"]
                self.pidfd = os.pidfd_open(self.pid)
                require(
                    self.stats(self.pid)["start"] == self.start,
                    "Child identity changed",
                )

    def _observe_descendants(self) -> None:
        found = self.descendants(self.pid)
        for pid in found - self.unexpected:
            self.unexpected.add(pid)  # Retain uncertainty even if pidfd setup fails.
            identity = self.stats(pid)["start"]
            fd = os.pidfd_open(pid)
            if self.stats(pid)["start"] != identity:
                os.close(fd)
                raise ValueError("Descendant identity changed")
            self.descendant_fds[pid] = fd

    def _signal(self, sig: int) -> None:
        for fd in self.descendant_fds.values():
            try:
                signal.pidfd_send_signal(fd, sig)
            except ProcessLookupError:
                pass
        if self.reaped:
            return
        if self.pidfd is not None:
            signal.pidfd_send_signal(self.pidfd, sig)
        else:
            # Direct, unreaped child; wait4 is owned exclusively under this lock.
            os.kill(self.pid, sig)

    @diagnosed("profile_os.OwnedProcess.stop")
    def stop(self) -> None:
        with self.lock:
            if self.reaped and not self.descendant_fds:
                return
            try:
                if self.stop_at is None:
                    self.stop_at = self.clock()
                    self._signal(signal.SIGTERM)
                elif not self.kill_sent and self.clock() - self.stop_at >= 0.2:
                    self._signal(signal.SIGKILL)
                    self.kill_sent = True
            except ProcessLookupError:
                pass  # Still wait4; ESRCH does not release ownership.

    def _drain(self) -> None:
        if self.eof or not self.drainable:
            return
        try:
            raw = self.read(
                cast(IO[bytes], self.process.stdout).fileno(), 16385 - len(self.raw)
            )
        except BlockingIOError:
            return
        self.raw.extend(raw)
        if len(self.raw) > 16384:
            self.invalid = True
            self.stop()
            cast(IO[bytes], self.process.stdout).close()
            self.eof = True
        elif not raw:
            self.eof = True
            cast(IO[bytes], self.process.stdout).close()

    @property
    def streams_closed(self) -> bool:
        return self.eof and self.stderr_eof

    def _drain_stderr(self) -> None:
        if self.stderr_eof:
            return
        stream = cast(IO[bytes], self.process.stderr)
        try:
            if not self.stderr_ready:
                os.set_blocking(stream.fileno(), False)
                self.stderr_ready = True
            # One bounded read per call, also after reap. Excess bytes are hashed
            # and discarded, never allowed to grow the retained prefix.
            raw = self.read(stream.fileno(), 4096)
            if raw:
                cast(StartupRecord, self.diagnostic).feed(raw)
            else:
                stream.close()
                self.stderr_eof = True
                cast(StartupRecord, self.diagnostic).update(stderr_eof=True)
        except BlockingIOError:
            pass
        except OSError as error:
            cast(StartupRecord, self.diagnostic).capture_error(
                "stderr_read_or_setup", error
            )
            # Do not hide the first capture failure. Later cleanup can reap;
            # the receipt explicitly distinguishes closed from EOF-confirmed.
            stream.close()
            self.stderr_eof = True
            raise

    @diagnosed("profile_os.OwnedProcess.sample")
    def sample(self) -> dict[str, Any]:
        with self.lock:
            if self.stop_at is not None:
                self.stop()  # Escalation continues after cancellation.
            self._drain()
            self._drain_stderr()
            if not self.reaped:
                # Before reap, detect any unexpected descendant; never infer that
                # root exit proves an unobserved descendant is gone.
                try:
                    observed = self.stats(self.pid)
                    if self.start is not None:
                        require(
                            observed["start"] == self.start,
                            "Owned PID identity changed",
                        )
                    self.cpu = max(self.cpu, observed["cpu"])
                    self._observe_descendants()
                    if self.unexpected:
                        self.invalid = True
                        self.stop()
                except FileNotFoundError:
                    pass  # wait4 is authoritative, not /proc disappearance.
                pid, status, usage = self.wait4(self.pid, os.WNOHANG)
                if pid:
                    require(pid == self.pid, "Foreign reap")
                    self.reaped = True
                    self.cpu = max(self.cpu, usage.ru_utime + usage.ru_stime)
                    self.exit_code = os.waitstatus_to_exitcode(status)
                    self.process.returncode = self.exit_code
                    diagnose(
                        self,
                        "profile_os.process_reaped",
                        pid=self.pid,
                        reaped=True,
                        wait_status=status,
                        exit_code=self.exit_code,
                    )
                    if self.diagnostic is not None:
                        self.diagnostic.update(
                            reaped=True, wait_status=status, exit_code=self.exit_code
                        )
                    if self.pidfd is not None:
                        os.close(self.pidfd)
                        self.pidfd = None
            if self.reaped and not self.drainable and self.stop_at is not None:
                self.invalid = True
                cast(IO[bytes], self.process.stdout).close()
                self.eof = True
            self._drain()
            if (
                self.reaped
                and self.eof
                and self.receipt is None
                and self.raw
                and not self.invalid
            ):
                try:
                    self.receipt = strict_json(self.raw)
                except (ValueError, UnicodeError) as error:
                    diagnose(self, "profile_os.invalid_receipt", error)
                    self.invalid = True
                if isinstance(self.receipt, dict) and self.diagnostics is not None:
                    facts = {
                        key: value
                        for key, value in self.receipt.items()
                        if key
                        in ("visited_values", "new_values", "frame_bytes", "live_bytes")
                    }
                    diagnose(
                        self,
                        "profile_os.worker_receipt",
                        receipt_json_parsed=True,
                        **facts,
                    )
            # Pipe EOF is part of owned completion; inherited pipe writers retain
            # busy rather than publishing an incomplete receipt.
            return {
                "cpu_seconds": self.cpu,
                "reaped": self.reaped and self.streams_closed,
                "exit_code": -1 if self.invalid else self.exit_code,
                "receipt": self.receipt,
            }

    @diagnosed("profile_os.OwnedProcess.wait")
    def wait(self, timeout: float) -> None:
        with self.lock:
            fds = [
                fd
                for fd in [
                    self.pidfd,
                    None if self.eof else cast(IO[bytes], self.process.stdout).fileno(),
                ]
                if fd is not None
            ]
            if not self.stderr_eof and self.stderr_ready:
                fds.append(cast(IO[bytes], self.process.stderr).fileno())
            # Hold descriptor ownership across the bounded wait; another thread
            # cannot close/reuse a pidfd between the snapshot and select.
            select.select(fds, [], [], min(timeout, 0.05))


class ProcessBook:
    """Whole coordinator/renderer/worker RSS; uncertainty is a refusal."""

    def __init__(
        self,
        *,
        stats: Callable[[int], ProcessStats] = proc_stat,
        descendants: Callable[[int], set[int]] = children,
        available: Callable[[], int] = available_bytes,
        diagnostics: StartupDiagnostics | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.diagnostics = diagnostics
        self.stats, self.descendants, self.available = stats, descendants, available
        self.clock = clock
        self.spawn_started: float | None = None
        self.root = os.getpid()
        self.root_start = stats(self.root)["start"]
        self.owned: list[OwnedProcess] = []
        self.lock = threading.RLock()
        self.spawn_uncertain = False
        self.spawning = False
        self.stop_requested = False

    def request_stop_all(self) -> None:
        with self.lock:
            self.stop_requested = True
            owned = list(self.owned)
        first_error: BaseException | None = None
        for child in owned:
            try:
                child.stop()
            except BaseException as error:
                diagnose(self, "profile_os.lifetime_signal_pending", error)
                first_error = first_error or error
        if first_error is not None:
            raise first_error

    def spawn(
        self,
        argv: Sequence[str],
        *,
        pass_fds: tuple[int, ...] = (),
        receipt: bool = True,
        diagnostics: StartupDiagnostics | None = None,
    ) -> OwnedProcess:
        with self.lock:
            self.owned = [
                c
                for c in self.owned
                if not (c.reaped and c.streams_closed and not c.unexpected)
            ]
            require(not self.stop_requested, "Owned provider shutdown pending")
            require(
                not self.spawning and len(self.owned) < 4,
                "Owned process inventory exceeds bound",
            )
            self.spawning = True
            self.spawn_started = self.clock()
        diagnostics = diagnostics or self.diagnostics
        diagnostic = None
        try:
            if diagnostics is not None:
                diagnostic = diagnostics.prepare(argv, pass_fds)
        except BaseException:
            with self.lock:
                self.spawning = False
            raise  # No Popen call; no new child ownership uncertainty.
        try:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE if receipt else subprocess.DEVNULL,
                stderr=(
                    subprocess.PIPE if diagnostic is not None else subprocess.DEVNULL
                ),
                close_fds=True,
                pass_fds=pass_fds,
                start_new_session=True,
            )
            child = OwnedProcess(
                process, diagnostic=diagnostic, diagnostics=diagnostics
            )
        except BaseException as error:
            if diagnostic is not None:
                diagnostic.update(
                    launch_error={
                        "phase": "popen_or_ownership",
                        **safe_error(error),
                        "error_chain": error_chain(error),
                    }
                )
            with self.lock:
                self.spawn_uncertain = True
                self.spawning = False
            raise
        with self.lock:
            self.owned.append(child)  # Before fallible pidfd/pipe setup.
            self.spawning = False
        return child

    def settled(self) -> bool:
        with self.lock:
            return (
                not self.spawn_uncertain
                and not self.spawning
                and all(
                    c.reaped and c.streams_closed and not c.unexpected
                    for c in self.owned
                )
            )

    def cleanup(self, *, stopping_only: bool = False) -> bool:
        with self.lock:
            owned = list(self.owned)
        for child in owned:
            if not stopping_only or child.stop_at is not None:
                child.stop()
                child.sample()
        return self.settled()

    def resources(self) -> ResourceSnapshot:
        with self.lock:
            probe, probe_pid = "root_stat", self.root
            try:
                root = self.stats(self.root)
                require(
                    root["start"] == self.root_start, "Coordinator identity changed"
                )
                rss, accounted, clear = (
                    root["rss"],
                    not (self.spawn_uncertain or self.spawning),
                    True,
                )
                expected = {child.pid for child in self.owned if not child.reaped}
                probe = "root_children"
                unregistered = self.descendants(self.root) - expected
                pending = self.spawning and not self.spawn_uncertain
                require(
                    not unregistered or pending and len(unregistered) == 1,
                    "Unregistered child",
                )
                for child in self.owned:
                    with child.lock:
                        clear = clear and not child.unexpected
                        if child.reaped:
                            continue
                        probe, probe_pid = "child_stat", child.pid
                        stat = self.stats(child.pid)
                        require(
                            child.start is None or stat["start"] == child.start,
                            "Child identity changed",
                        )
                        rss += stat["rss"]
                        probe = "child_children"
                        found = self.descendants(child.pid)
                        if found:
                            child._observe_descendants()
                        if found:
                            child.stop()
                            clear = False
                if unregistered:
                    # Read-only accounting of the sole candidate during Popen.
                    # Never adopt, signal, reap or publish using this observation.
                    candidate = next(iter(unregistered))
                    probe, probe_pid = "spawn_candidate", candidate
                    stat = self.stats(candidate)
                    require(
                        not self.descendants(candidate),
                        "Owned spawn candidate has descendants",
                    )
                    require(
                        self.stats(candidate)["start"] == stat["start"],
                        "Owned spawn identity changed",
                    )
                    rss += stat["rss"]
                probe, probe_pid = "available_memory", self.root
                observed: ResourceSnapshot = {
                    "rss_bytes": rss,
                    "available_bytes": self.available(),
                    "all_owned_accounted": accounted,
                    "descendants_clear": clear,
                }
                if pending:
                    require(
                        self.spawn_started is not None
                        and 0 <= self.clock() - self.spawn_started < 0.05,
                        "Owned spawn registration deadline exceeded",
                    )
                    diagnose(
                        self,
                        "profile_os.spawn_observation_pending",
                        spawning=True,
                        owned_count=len(self.owned),
                        **observed,
                    )
                    raise SpawnObservationPending(cast(Mapping[str, int], observed))
                return observed
            except (OSError, ValueError) as error:
                diagnose(
                    self,
                    "profile_os.resource_observation_failed",
                    error,
                    spawning=self.spawning,
                    spawn_uncertain=self.spawn_uncertain,
                    owned_count=len(self.owned),
                    probe=probe,
                    probe_pid=probe_pid,
                )
                return {
                    "rss_bytes": 768 * MIB + 1,
                    "available_bytes": 0,
                    "all_owned_accounted": False,
                    "descendants_clear": False,
                }


class FrozenBinary:
    def __init__(self, path: str | Path, expected_sha: str) -> None:
        self.path = Path(path).resolve()
        with self.path.open("rb") as stream:
            stat = os.fstat(stream.fileno())
            require(stat.st_size <= 32 * MIB, "Executable exceeds receipt bound")
            require(
                hashlib.file_digest(stream, "sha256").hexdigest() == expected_sha,
                "Qualified executable receipt mismatch",
            )
            self.identity = fingerprint(stat)
        self.check()

    def check(self) -> None:
        require(fingerprint(self.path.stat()) == self.identity, "Executable changed")


class LinuxProfileHooks:
    def __init__(
        self,
        book: ProcessBook,
        binary: FrozenBinary,
        source: ProfileSource,
        watchdog: ProfileWatchdog,
        *,
        launch_policy: BinaryPolicy | None = None,
    ) -> None:
        self.launch_policy = launch_policy
        self.book, self.binary, self.source, self.watchdog = (
            book,
            binary,
            source,
            watchdog,
        )
        self.diagnostics = getattr(book, "diagnostics", None)
        self.child: OwnedProcess | None = None
        self.attempted = False

    def start_gate(self) -> tuple[int, int]:
        return available_bytes(), shutil.disk_usage(self.source["cache"]).free

    def source_check(self) -> None:
        if self.launch_policy is not None:
            self.launch_policy.check_binary(self.binary)
        self.source["check"]()
        self.binary.check()

    def resources(self) -> ResourceSnapshot:
        return self.book.resources()

    def spawn(
        self, request: Mapping[str, Any], output_fd: int, input_fd: int | None
    ) -> OwnedProcess:
        fields(request, ("binding", "seed", "values", "wall_ms", "cpu_ms"))
        require(input_fd is None and not self.attempted, "Resume unavailable")
        self.attempted = True
        self.source_check()
        selected = request["binding"]
        require(selected == self.source["binding"], "Trusted source binding differs")
        opts = {
            "model": self.source["root"],
            "revision": self.source["revision"],
            "tensor": selected["tensor"],
            "slice": ",".join(map(str, selected["slice"]["leading_indices"])),
            "seed": request["seed"],
            "values": request["values"],
            "wall-ms": request["wall_ms"],
            "cpu-ms": request["cpu_ms"],
            "binding": canonical(selected).decode(),
            "output-fd": output_fd,
        }
        argv = [str(self.binary.path), "profile-worker"]
        for key, value in opts.items():
            argv.extend(["--" + key, str(value)])
        self.child = self.book.spawn(argv, pass_fds=(output_fd,))
        return self.child

    def watch(
        self, callback: Callable[[], object], deadline: float
    ) -> WatchRegistration:
        return self.watchdog.register(callback, deadline)

    def close_admission_watch(self) -> None:
        self.watchdog.close()

    def no_child_created(self) -> bool:
        return not self.book.spawn_uncertain and self.child is None

    def snapshot_closed(self, store: SnapshotOwner) -> bool:
        store.require_settled()
        return store.owned_storage_bytes == 0
