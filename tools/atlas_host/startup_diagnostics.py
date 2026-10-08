"""Opt-in, owner-local startup evidence. Never attach to an HTTP response.

Only known static errors and OS errno numbers leave the bounded private buffer.
Unrecognized stderr is represented by byte counts and hashes, never echoed.
"""

from atlas_host import limits as _limits

from copy import deepcopy
import fcntl
import hashlib
import json
from functools import wraps
import os
from pathlib import Path
import sys
import re
import stat
import threading

from .common import require

from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from typing import (
    Concatenate,
    Literal,
    NotRequired,
    ParamSpec,
    Protocol,
    TypeVar,
    TypedDict,
    Unpack,
    cast,
)


class SafeError(TypedDict):
    type: str
    errno: int | None


class TraceFrame(TypedDict):
    file: str
    function: str
    line: int


class ErrorNode(SafeError):
    message: str
    frames: list[TraceFrame]
    frames_truncated: bool
    cause: int | None
    context: int | None
    suppress_context: bool


class ErrorChain(TypedDict):
    exceptions: list[ErrorNode]
    chain_truncated: bool
    exception_limit: int
    total_frame_limit: int


class DiagnosticError(SafeError):
    error_chain: ErrorChain


class CaptureError(DiagnosticError):
    phase: str


class PassedFd(TypedDict):
    fd: int
    role: str
    parent_fd_flags: int
    parent_cloexec: bool
    parent_inheritable: bool
    parent_is_socket: bool


class ChildFdExpectations(TypedDict, total=False):
    passed_channel_survives_exec: bool
    passed_channel_cloexec_after_exec: bool
    rust_duplicate_uses: str
    rust_duplicate_cloexec: bool
    passed_output_survives_exec: bool
    passed_output_cloexec_after_exec: bool
    observed_in_child: bool


class RecordData(TypedDict):
    argv: list[str]
    argv_sanitized: bool
    passed_fds: list[PassedFd]
    close_fds: bool
    start_new_session: bool
    stdout: str
    stderr: str
    child_fd_expectations: ChildFdExpectations
    pid: int | None
    reaped: bool
    wait_status: int | None
    exit_code: int | None
    stderr_bytes_seen: int
    stderr_eof: bool
    capture_errors: list[CaptureError]
    launch_error: CaptureError | None
    initialize_error: NotRequired[DiagnosticError]


class RecordUpdates(TypedDict, total=False):
    passed_fds: list[PassedFd]
    pid: int | None
    reaped: bool
    wait_status: int | None
    exit_code: int | None
    stderr_eof: bool
    launch_error: CaptureError | None
    initialize_error: DiagnosticError


class RecordSnapshot(RecordData):
    stderr_prefix_bytes: int
    stderr_truncated: bool
    stderr_sha256_seen: str
    stderr_sha256_complete: bool
    stderr_redaction_policy: str


class StoredEvent(TypedDict):
    json: str
    count: int
    first_sequence: int
    last_sequence: int


class EventPayload(TypedDict):
    site: str
    thread: str
    facts: dict[str, object]
    error_chain: NotRequired[ErrorChain]


class EventSnapshot(EventPayload):
    count: int
    first_sequence: int
    last_sequence: int


class DiagnosticsSnapshot(TypedDict):
    version: int
    attempt_limit: int
    stderr_prefix_limit: int
    attempts: list[RecordSnapshot]
    events: list[EventSnapshot]
    event_limit: int
    event_byte_limit: int
    event_bytes: int
    events_dropped: int


class DiagnosticCollector(Protocol):
    def event(
        self, site: str, error: BaseException | None = None, **facts: object
    ) -> None: ...


_Owner = TypeVar("_Owner")
_Result = TypeVar("_Result")
_Arguments = ParamSpec("_Arguments")

STDERR_LIMIT = _limits.STARTUP_STDERR_LIMIT
ATTEMPT_LIMIT = _limits.STARTUP_ATTEMPT_LIMIT
STATIC_ERRORS = frozenset(
    (
        "Inherited private channel required",
        "Cannot duplicate hosted channel",
        "Connected private channel required",
        "Private channel required",
        "Options require values",
        "Expected --option value",
        "Duplicate CLI option",
        "--model DIRECTORY is required",
        "Hosted command exceeds limit",
        "Invalid hosted operation",
        "Hosted operation kind mismatch",
        "Hosted response deadline expired",
        "MemAvailable missing",
        "MemAvailable invalid",
        "PAUSED: fewer than 3 GiB available RAM; retry when memory is available",
        "Cannot check disk reserve",
        "PAUSED: fewer than 25 GiB disk reserve",
        "Cannot read affinity",
        "No allowed CPU",
        "ATLAS_CPU is outside allowed affinity",
        "Cannot set one CPU affinity",
        "Cannot read address-space limit",
        "Cannot set address-space limit",
    )
)

# Owner diagnostic messages only; visitor APIs retain their existing fixed errors.
STATIC_ERRORS = STATIC_ERRORS | frozenset(
    (
        "Unregistered child",
        "Owned spawn candidate has descendants",
        "Owned spawn identity changed",
        "Owned spawn registration deadline exceeded",
        "Owned spawn registration observation pending",
        "Coordinator identity changed",
        "Owned PID identity changed",
        "Child identity changed",
        "Descendant identity changed",
        "Foreign reap",
        "Owned thread inventory exceeds bound",
        "Owned child inventory exceeds bound",
        "Owned resources exceeded",
        "Profile resource envelope exceeded",
        "Final resource envelope exceeded",
        "Shared ownership uncertain",
        "Owned child CPU clock regressed",
        "Child CPU clock regressed",
        "Job CPU clock regressed/unavailable",
        "Invalid job-local owner CPU clock",
        "CPU context unavailable",
        "Wrong job-owning execution context",
        "Profile cancelled/expired",
        "Total profile grant exhausted",
        "Original admission cancelled",
        "Original admission grant exhausted",
        "Shared heavy-slot ownership lost",
        "Worker failed",
        "Invalid candidate receipt",
        "Candidate allowance/layout mismatch",
        "Admission consumed child allowance",
        "Existing profile launch gate refused",
        "No child budget remains",
        "Snapshot close disposition uncertain; do not retry descriptor",
        "Snapshot close disposition uncertain",
        "Unclosed snapshot storage remains",
        "All snapshot handles, including retiring storage, must be closed",
        "Worker cleanup must finish before session close",
        "Owner lease/cancel",
        "Independent watchdog callback failed",
        "Independent watchdog cleanup uncertain",
    )
)


def diagnose(
    owner: object, site: str, error: BaseException | None = None, **facts: object
) -> None:
    collector: DiagnosticCollector | None = getattr(owner, "diagnostics", None)
    if collector is not None:
        collector.event(site, error, **facts)


def diagnosed(
    site: str,
) -> Callable[
    [Callable[Concatenate[_Owner, _Arguments], _Result]],
    Callable[Concatenate[_Owner, _Arguments], _Result],
]:
    def decorate(
        function: Callable[Concatenate[_Owner, _Arguments], _Result],
    ) -> Callable[Concatenate[_Owner, _Arguments], _Result]:
        @wraps(function)
        def call(
            self: _Owner, *args: _Arguments.args, **kwargs: _Arguments.kwargs
        ) -> _Result:
            try:
                return function(self, *args, **kwargs)
            except BaseException as error:
                diagnose(self, site, error)
                raise

        return cast(Callable[Concatenate[_Owner, _Arguments], _Result], call)

    return decorate


def safe_error(error: BaseException) -> SafeError:
    # Exception text/filenames can contain private paths or capabilities.
    name = type(error).__name__
    known = {
        "RuntimeError",
        "AssertionError",
        "TypeError",
        "KeyError",
        "IndexError",
        "AttributeError",
        "StopIteration",
        "GeneratorExit",
        "KeyboardInterrupt",
        "SystemExit",
        "TimeoutError",
        "HostError",
        "ValueError",
        "OverflowError",
    }
    return {
        "type": name if isinstance(error, OSError) or name in known else "OtherError",
        "errno": error.errno if isinstance(error, OSError) else None,
    }


def sanitized_stderr(raw: bytes | bytearray) -> str:
    lines = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        message = line.removeprefix("ERROR: ")
        if line.startswith("ERROR: ") and message in STATIC_ERRORS:
            lines.append(line)
        else:
            native = re.fullmatch(
                r"ERROR: Hosted channel (peer_addr|F_DUPFD_CLOEXEC) failed: "
                r"kind=([A-Za-z]{1,40}); errno=(none|-?[0-9]{1,10})",
                line,
            )
            if native and native[2] in {
                "NotFound",
                "PermissionDenied",
                "ConnectionRefused",
                "ConnectionReset",
                "HostUnreachable",
                "NetworkUnreachable",
                "ConnectionAborted",
                "NotConnected",
                "AddrInUse",
                "AddrNotAvailable",
                "NetworkDown",
                "BrokenPipe",
                "AlreadyExists",
                "WouldBlock",
                "NotADirectory",
                "IsADirectory",
                "DirectoryNotEmpty",
                "ReadOnlyFilesystem",
                "FilesystemLoop",
                "StaleNetworkFileHandle",
                "InvalidInput",
                "InvalidData",
                "TimedOut",
                "WriteZero",
                "StorageFull",
                "NotSeekable",
                "QuotaExceeded",
                "FileTooLarge",
                "ResourceBusy",
                "ExecutableFileBusy",
                "Deadlock",
                "CrossesDevices",
                "TooManyLinks",
                "InvalidFilename",
                "ArgumentListTooLong",
                "Interrupted",
                "Unsupported",
                "UnexpectedEof",
                "OutOfMemory",
                "InProgress",
                "Other",
                "Uncategorized",
            }:
                lines.append(line)
                continue
            # Rust std::io Display; retain errno without trusting arbitrary text.
            match = re.fullmatch(r"ERROR: [^\r\n]* \(os error ([0-9]{1,5})\)", line)
            lines.append(
                "ERROR: [redacted OS error text] (os error " + match[1] + ")"
                if match
                else "[redacted unrecognized stderr line]"
            )
    return "\n".join(lines)[:STDERR_LIMIT]


ERROR_CHAIN_LIMIT = _limits.STARTUP_ERROR_CHAIN_LIMIT
TRACE_FRAME_LIMIT = _limits.STARTUP_TRACE_FRAME_LIMIT


def error_chain(
    error: BaseException, *, locations: Mapping[str | Path, str] | None = None
) -> ErrorChain:
    """Bounded cause/context graph, safe messages and frame locations; no locals.

    Explicit owner mappings identify external harness code without exporting its
    absolute pathname. Unknown code names/paths and arbitrary messages stay out.
    """
    trusted = {
        str(Path(name)): label
        for name, label in (locations or {}).items()
        if re.fullmatch(r"[A-Za-z0-9_./-]{1,128}", label)
    }
    repo = Path(__file__).parent.parent.parent
    stdlib = (
        Path(sys.base_prefix)
        / "lib"
        / ("python" + str(sys.version_info.major) + "." + str(sys.version_info.minor))
    )
    nodes: list[ErrorNode]
    pending: list[BaseException]
    indexes: dict[int, int]
    nodes, pending, indexes = [], [error], {id(error): 0}
    frames_left = TRACE_FRAME_LIMIT
    chain_truncated = False
    for current in pending:
        meta = safe_error(current)
        raw_message = (
            str(current)
            if isinstance(current, (OSError, ValueError, RuntimeError, AssertionError))
            else ""
        )
        message = sanitized_stderr(("ERROR: " + raw_message[:4096]).encode())[:1024]
        node: ErrorNode = {
            **meta,
            "message": message,
            "frames": [],
            "frames_truncated": False,
            "cause": None,
            "context": None,
            "suppress_context": bool(current.__suppress_context__),
        }
        tb = current.__traceback__
        while tb is not None and frames_left:
            code = tb.tb_frame.f_code
            path = Path(code.co_filename)
            label = trusted.get(str(path))
            if label is None:
                for root, prefix in ((repo, "repo/"), (stdlib, "stdlib/")):
                    try:
                        relative = str(path.relative_to(root))
                    except ValueError:
                        continue
                    if re.fullmatch(r"[A-Za-z0-9_./-]{1,112}", relative):
                        label = prefix + relative
                    break
            function = code.co_name
            node["frames"].append(
                {
                    "file": label or "<external-code>",
                    "function": (
                        function
                        if label and re.fullmatch(r"[A-Za-z0-9_<>]{1,64}", function)
                        else "<withheld>"
                    ),
                    "line": min(max(0, tb.tb_lineno), 10**9),
                }
            )
            frames_left -= 1
            tb = tb.tb_next
        node["frames_truncated"] = tb is not None
        relation: Literal["cause", "context"]
        for relation in ("cause", "context"):
            linked: BaseException | None = getattr(current, "__" + relation + "__")
            if linked is None:
                continue
            if id(linked) not in indexes:
                if len(pending) >= ERROR_CHAIN_LIMIT:
                    chain_truncated = True
                    continue
                indexes[id(linked)] = len(pending)
                pending.append(linked)
            node[relation] = indexes[id(linked)]
        nodes.append(node)
    return {
        "exceptions": nodes,
        "chain_truncated": chain_truncated,
        "exception_limit": ERROR_CHAIN_LIMIT,
        "total_frame_limit": TRACE_FRAME_LIMIT,
    }


class StartupRecord:
    def __init__(self, fd: int, *, worker: bool = False) -> None:
        self.lock = threading.RLock()
        self.raw = bytearray()
        self.digest = hashlib.sha256()
        self.data: RecordData = {
            "argv": [
                "<qualified-renderer>",
                "hosted-renderer",
                "--model",
                "<model-root>",
                "--cache",
                "<cache-root>",
                "--name",
                "<display-name>",
                "--revision",
                "<pinned-revision>",
                "--channel-fd",
                str(fd),
            ],
            "argv_sanitized": True,
            "passed_fds": [],
            "close_fds": True,
            "start_new_session": True,
            "stdout": "DEVNULL",
            "stderr": "bounded nonblocking PIPE",
            "child_fd_expectations": {
                "passed_channel_survives_exec": True,
                "passed_channel_cloexec_after_exec": False,
                "rust_duplicate_uses": "F_DUPFD_CLOEXEC",
                "rust_duplicate_cloexec": True,
                "observed_in_child": False,
            },
            "pid": None,
            "reaped": False,
            "wait_status": None,
            "exit_code": None,
            "stderr_bytes_seen": 0,
            "stderr_eof": False,
            "capture_errors": [],
            "launch_error": None,
        }
        if worker:
            cast(MutableMapping[str, object], self.data).update(
                argv=[
                    "<qualified-worker>",
                    "profile-worker",
                    "--model",
                    "<model-root>",
                    "--revision",
                    "<pinned-revision>",
                    "--tensor",
                    "<tensor-index>",
                    "--slice",
                    "<slice-indexes>",
                    "--seed",
                    "<seed>",
                    "--values",
                    "<values>",
                    "--wall-ms",
                    "<wall-ms>",
                    "--cpu-ms",
                    "<cpu-ms>",
                    "--binding",
                    "<trusted-binding>",
                    "--output-fd",
                    str(fd),
                ],
                stdout="bounded receipt PIPE",
                child_fd_expectations={
                    "passed_output_survives_exec": True,
                    "passed_output_cloexec_after_exec": False,
                    "observed_in_child": False,
                },
            )

    def update(self, **values: Unpack[RecordUpdates]) -> None:
        with self.lock:
            cast(MutableMapping[str, object], self.data).update(values)

    def capture_error(self, phase: str, error: BaseException) -> None:
        with self.lock:
            self.data["capture_errors"].append(
                {"phase": phase, **safe_error(error), "error_chain": error_chain(error)}
            )

    def feed(self, raw: bytes | bytearray) -> None:
        with self.lock:
            self.data["stderr_bytes_seen"] += len(raw)
            self.digest.update(raw)
            self.raw.extend(raw[: max(0, STDERR_LIMIT - len(self.raw))])

    def snapshot(self) -> RecordSnapshot:
        with self.lock:
            return {
                **deepcopy(self.data),
                "stderr": sanitized_stderr(self.raw),
                "stderr_prefix_bytes": len(self.raw),
                "stderr_truncated": self.data["stderr_bytes_seen"] > len(self.raw),
                "stderr_sha256_seen": self.digest.hexdigest(),
                "stderr_sha256_complete": self.data["stderr_eof"]
                and not self.data["capture_errors"],
                "stderr_redaction_policy": "static error allowlist or OS errno; all other text withheld",
            }


class StartupDiagnostics:
    """Four attempts, 16 KiB prefix each; refuse further starts, never evict evidence.

    The trusted owner opts in for a qualification session. No automatic file I/O,
    callbacks, background threads, command/response payloads or capabilities.
    """

    def __init__(self) -> None:
        self.records: list[StartupRecord] = []
        self.lock = threading.RLock()
        self.events: list[StoredEvent] = []
        self.event_bytes = 0
        self.events_dropped = 0
        self.sequence = 0

    def event(
        self, site: str, error: BaseException | None = None, **facts: object
    ) -> None:
        # Only fixed sites and scalar accounting facts; never copy request dicts.
        require(
            re.fullmatch(r"[A-Za-z0-9_.]{1,80}", site) is not None,
            "Invalid diagnostic site",
        )
        allowed = {
            "spawning",
            "spawn_uncertain",
            "owned_count",
            "unexpected_count",
            "frame_bytes",
            "reservation_bytes",
            "rss_bytes",
            "available_bytes",
            "all_owned_accounted",
            "descendants_clear",
            "pid",
            "reaped",
            "exit_code",
            "wait_status",
            "visited_values",
            "new_values",
            "live_bytes",
            "receipt_json_parsed",
            "probe_pid",
        }
        probe = facts.get("probe")
        facts = {
            key: value
            for key, value in facts.items()
            if key in allowed
            and (
                type(value) is bool or type(value) is int and -(2**63) <= value < 2**63
            )
        }
        if probe in (
            "root_stat",
            "root_children",
            "child_stat",
            "child_children",
            "spawn_candidate",
            "available_memory",
        ):
            facts["probe"] = probe
        thread = threading.current_thread().name
        payload: EventPayload = {
            "site": site,
            "thread": (
                thread
                if thread in ("atlas-profile-owner", "atlas-profile-watchdog")
                else "caller"
            ),
            "facts": facts,
        }
        if error is not None:
            payload["error_chain"] = error_chain(error)
        serialized = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        with self.lock:
            self.sequence += 1
            for entry in self.events:
                if entry["json"] == serialized:
                    entry["count"] += 1
                    entry["last_sequence"] = self.sequence
                    return
            if (
                len(self.events) >= 64
                or self.event_bytes + len(serialized) > 512 * 1024
            ):
                self.events_dropped += 1
                return
            self.events.append(
                {
                    "json": serialized,
                    "count": 1,
                    "first_sequence": self.sequence,
                    "last_sequence": self.sequence,
                }
            )
            self.event_bytes += len(serialized)

    def prepare(self, argv: Sequence[str], pass_fds: Iterable[int]) -> StartupRecord:
        renderer = (
            len(argv) == 12
            and argv[1] == "hosted-renderer"
            and argv[2::2]
            == ["--model", "--cache", "--name", "--revision", "--channel-fd"]
        )
        worker = (
            len(argv) == 22
            and argv[1] == "profile-worker"
            and argv[2::2]
            == [
                "--model",
                "--revision",
                "--tensor",
                "--slice",
                "--seed",
                "--values",
                "--wall-ms",
                "--cpu-ms",
                "--binding",
                "--output-fd",
            ]
        )
        require(
            renderer or worker, "Diagnostic capture only accepts fixed owned commands"
        )
        fd = int(argv[-1])
        require(fd > 2 and tuple(pass_fds) == (fd,), "Diagnostic inherited FD mismatch")
        with self.lock:
            require(
                len(self.records) < ATTEMPT_LIMIT, "Startup diagnostic inventory full"
            )
            record = StartupRecord(fd, worker=worker)
            self.records.append(record)
        try:
            flags = fcntl.fcntl(fd, fcntl.F_GETFD)
            record.update(
                passed_fds=[
                    {
                        "fd": fd,
                        "role": (
                            "profile output memfd"
                            if worker
                            else "private renderer channel"
                        ),
                        "parent_fd_flags": flags,
                        "parent_cloexec": bool(flags & fcntl.FD_CLOEXEC),
                        "parent_inheritable": os.get_inheritable(fd),
                        "parent_is_socket": stat.S_ISSOCK(os.fstat(fd).st_mode),
                    }
                ]
            )
        except BaseException as error:
            record.update(
                launch_error={
                    "phase": "fd_metadata",
                    **safe_error(error),
                    "error_chain": error_chain(error),
                }
            )
            raise
        return record

    def snapshot(self) -> DiagnosticsSnapshot:
        with self.lock:
            return {
                "version": 1,
                "attempt_limit": ATTEMPT_LIMIT,
                "stderr_prefix_limit": STDERR_LIMIT,
                "attempts": [record.snapshot() for record in self.records],
                "events": cast(
                    list[EventSnapshot],
                    [
                        {
                            **json.loads(e["json"]),
                            **{k: v for k, v in e.items() if k != "json"},
                        }
                        for e in self.events
                    ],
                ),
                "event_limit": 64,
                "event_byte_limit": 512 * 1024,
                "event_bytes": self.event_bytes,
                "events_dropped": self.events_dropped,
            }
