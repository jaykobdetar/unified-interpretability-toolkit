#!/usr/bin/env python3
"""Loopback-only prototype coordinator: static Rust atlas + isolated CPU inference.

No external network clients, downloads, prompt logs or persisted sessions. One
HTTP coordinator and one inference subprocess; numerical work uses one CPU.
"""

import argparse
from dataclasses import replace
from collections import deque
from contextlib import contextmanager
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import io
import os
from pathlib import Path
import secrets
import signal
import stat
import subprocess
import sys
import time
from typing import (
    Any,
    IO,
    NoReturn,
    Protocol,
    TypedDict,
    TypeAlias,
    TYPE_CHECKING,
    cast,
)
from collections.abc import Callable, Iterator, Mapping
import socket
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from analytics.service import AnalyticsJobs, Handler as AnalyticsHandler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from atlas_host.inference_architecture import (
    ARCH,
    WIDTH,
    LAYERS,
    HEADS,
    KV_HEADS,
    HEAD_DIM,
    VOCAB,
    CAPTURE_SITES,
    head_layout_descriptor,
    bind_viewer_head_layout,
)
from atlas_host.memory import available_bytes as available
from atlas_host.inference_edits import (
    SOURCE_MODEL,
    schema,
    validate_edits,
    validate_pair,
)
from atlas_host.inference_observations import (
    schema as observation_schema,
    validate_observation,
    validate_record,
)
from atlas_host import inference_prompt_pair as prompt_pair
from atlas_host import inference_sweep as sweep
from atlas_host.inference_experiments import Kind, REGISTRY, for_coordinator
from atlas_host.inference_architecture import architecture
from atlas_host.inference_geometry import (
    head_layout_descriptor as bound_head_layout_descriptor,
)
from atlas_host.inference_services import InferenceContracts

MANIFEST = json.loads((ROOT / "docs/models/smollm2-135m.json").read_text())
GIB = 1024**3
MAX_BODY = 8192
MAX_TRACE = 32
REQUEST_DEADLINE = 0.5
UPSTREAM_DEADLINE = 5.0
IO_TICK = 0.1


class TickOwner(Protocol):
    def tick(self) -> object: ...


class OwnedProcess(Protocol):
    def poll(self) -> int | None: ...
    def wait(self, *, timeout: float) -> int: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...


class ClosableServer(Protocol):
    def server_close(self) -> None: ...


class HashState(Protocol):
    def update(self, data: bytes, /) -> None: ...
    def hexdigest(self) -> str: ...


Fingerprint: TypeAlias = tuple[int, int, int, int, int]


class LayoutReceipt(TypedDict):
    directory: str
    files: dict[str, Fingerprint]


class Snapshot(TypedDict):
    session: str | None
    status: str
    steps: list[dict[str, Any]]
    details: dict[str, Any]
    peak_worker_rss_mib: float
    minimum_available_gib: float
    worker_alive: bool


class BackendError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code = status, code


class OperationDeadline:
    def __init__(
        self, session: TickOwner, clock: Callable[[], float] | None = None
    ) -> None:
        self.clock = clock or time.monotonic
        self.end = self.clock() + UPSTREAM_DEADLINE
        self.last_tick = -float("inf")
        self.session = session

    def remaining(self) -> float:
        now = self.clock()
        if now - self.last_tick >= IO_TICK:
            self.session.tick()
            self.last_tick = now
            if available() < 3.25 * GIB:
                raise BackendError(
                    503, "resource_limit", "Local memory reserve reached"
                )
        remaining = self.end - self.clock()
        if remaining <= 0:
            raise BackendError(
                504, "backend_timeout", "Local renderer operation timed out"
            )
        return remaining


class DeadlineReader(io.RawIOBase):
    def __init__(self, connection: socket.socket, deadline: OperationDeadline) -> None:
        super().__init__()
        self.connection, self.deadline = connection, deadline

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        while True:
            self.connection.settimeout(min(IO_TICK, self.deadline.remaining()))
            try:
                size = self.connection.recv_into(buffer)
                self.deadline.remaining()
                return size
            except TimeoutError:
                continue

    def close(self) -> None:
        try:
            self.connection.close()
        finally:
            super().close()


class DeadlineSocket:
    def __init__(self, connection: socket.socket, deadline: OperationDeadline) -> None:
        self.connection, self.deadline = connection, deadline

    def sendall(self, data: bytes | bytearray | memoryview) -> None:
        data = memoryview(data)
        while data:
            self.connection.settimeout(min(IO_TICK, self.deadline.remaining()))
            try:
                sent = self.connection.send(data)
                if not sent:
                    raise ConnectionError("Local renderer connection closed")
                data = data[sent:]
            except TimeoutError:
                continue
        self.deadline.remaining()

    def makefile(self, mode: str) -> io.BufferedReader:
        if mode != "rb":
            raise ValueError("Only upstream response reads are supported")
        # HTTPConnection closes its socket upon a Connection: close response.
        # Keep a separately owned descriptor alive until HTTPResponse closes.
        return io.BufferedReader(DeadlineReader(self.connection.dup(), self.deadline))

    def close(self) -> None:
        self.connection.close()


class ProxyConnection(Protocol):
    sock: socket.socket | DeadlineSocket | None

    def connect(self) -> None: ...
    def request(
        self, method: str, path: str, *, headers: Mapping[str, str]
    ) -> None: ...
    def getresponse(self) -> http.client.HTTPResponse: ...
    def close(self) -> None: ...


def proxy_request(
    port: int, method: str, path: str, session: TickOwner
) -> tuple[int, bytes, str]:
    deadline = OperationDeadline(session)
    conn = cast(
        ProxyConnection,
        http.client.HTTPConnection("127.0.0.1", port, timeout=REQUEST_DEADLINE),
    )
    try:
        deadline.remaining()
        conn.connect()  # Loopback connection establishment is bounded separately.
        deadline.remaining()
        conn.sock = DeadlineSocket(cast(socket.socket, conn.sock), deadline)
        headers = {"X-Atlas-Local": "1"} if method == "POST" else {}
        conn.request(method, path, headers=headers)
        response = conn.getresponse()
        try:
            body = response.read(2 * 1024**2 + 1)
            deadline.remaining()
            if len(body) > 2 * 1024**2:
                raise BackendError(
                    502,
                    "backend_response_limit",
                    "Local renderer response exceeds limit",
                )
            return (
                response.status,
                body,
                response.getheader("Content-Type", "application/octet-stream"),
            )
        finally:
            response.close()
    except TimeoutError:
        raise BackendError(
            504, "backend_timeout", "Local renderer connection timed out"
        ) from None
    except (OSError, http.client.HTTPException):
        raise BackendError(
            503, "backend_unavailable", "Local renderer unavailable; retry shortly"
        ) from None
    finally:
        conn.close()


def process_alive(process: OwnedProcess) -> bool:
    try:
        return process.poll() is None
    except OSError:
        return True  # An unknown reap result must retain ownership/admission.


def signal_and_reap(
    process: OwnedProcess, *, terminate: bool = False, timeout: float = 0.2
) -> bool:
    """One bounded attempt. False retains ownership; no unrelated PID is touched."""
    try:
        if process_alive(process):
            try:
                (process.terminate if terminate else process.kill)()
            except ProcessLookupError:
                pass
        process.wait(timeout=timeout)
        return True
    except (OSError, subprocess.TimeoutExpired):
        return False


class SweepAdmissionBudget:
    """One job budget; interruptible verification on this Linux main-thread owner."""

    def __init__(self, deadline: float, cpu_deadline: float) -> None:
        self.deadline, self.cpu_deadline = deadline, cpu_deadline

    def check(self) -> None:
        if time.monotonic() >= self.deadline:
            raise ValueError(
                "Total sweep wall budget exhausted during verification; no further work"
            )
        if time.process_time() >= self.cpu_deadline:
            raise ValueError(
                "Total sweep CPU budget exhausted during verification; no further work"
            )
        if available() < 3.25 * GIB:
            raise ValueError("Sweep verification stopped by available-memory reserve")

    def remaining_cpu(self) -> int:
        self.check()
        # RLIMIT_CPU uses integer seconds. Round down, never grant a fresh 90 s.
        remaining = int(self.cpu_deadline - time.process_time())
        if remaining < 1:
            raise ValueError("Less than one CPU second remains; no worker started")
        return min(90, remaining)

    @contextmanager
    def verification(self) -> Iterator[None]:
        self.check()
        # An existing timer is another owner's deadline, never ours to replace.
        if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
            raise ValueError("An existing deadline timer prevents sweep verification")
        previous = signal.getsignal(signal.SIGALRM)

        def arm() -> None:
            signal.setitimer(
                signal.ITIMER_REAL,
                max(0.000001, min(IO_TICK, self.deadline - time.monotonic())),
            )

        def alarm(_signal: int, _frame: object) -> None:
            # Raising interrupts a blocked regular-file open/read on this POSIX
            # main thread; a returning handler would permit automatic syscall retry.
            self.check()
            arm()

        signal.signal(signal.SIGALRM, alarm)
        try:
            arm()
            yield
            self.check()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)


def verify_model(directory: Path, check: Callable[[], object] | None = None) -> None:
    def checkpoint() -> None:
        if check is not None:
            check()

    for name, expected in MANIFEST["files"].items():
        checkpoint()
        with (directory / name).open("rb") as source:
            digest: HashState | str = hashlib.sha256()
            while True:
                checkpoint()
                block = source.read(1024 * 1024)
                checkpoint()
                if not block:
                    break
                cast(HashState, digest).update(block)
                checkpoint()
            digest = cast(HashState, digest).hexdigest()
        checkpoint()
        if digest != expected:
            raise ValueError(f"Pinned file hash mismatch: {name}")


def layout_fingerprints(directory: Path) -> LayoutReceipt | None:
    """Private correspondence to freshly verified regular pinned files."""
    try:
        directory = directory.resolve(strict=True)
        files: dict[str, Fingerprint] = {}
        for name in MANIFEST["files"]:
            value = (directory / name).lstat()
            if not stat.S_ISREG(value.st_mode):
                return None
            files[name] = (
                value.st_dev,
                value.st_ino,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )
        return {"directory": str(directory), "files": files}
    except OSError:
        return None


def verified_layout_receipt(directory: Path, *, required: bool) -> LayoutReceipt | None:
    """Reuse pinned verification; optional analytics evidence fails closed."""
    before = layout_fingerprints(directory)
    check: Callable[[], object] | None = None
    if not required:
        # Ordinary analytics sources remain unbound. Only the exact bounded
        # pinned configuration can trigger this small-model startup verifier.
        if (
            before is None
            or before["files"]["config.json"][2] > 65536
            or sum(value[2] for value in before["files"].values()) > 384 * 1024**2
        ):
            return None
        try:
            with (directory / "config.json").open("rb") as source:
                config = source.read(65537)
        except OSError:
            return None
        if hashlib.sha256(config).hexdigest() != MANIFEST["files"]["config.json"]:
            return None
        deadline = time.monotonic() + UPSTREAM_DEADLINE

        def check() -> None:
            if time.monotonic() >= deadline or available() < 3.25 * GIB:
                raise ValueError("Configuration receipt verification budget reached")

    try:
        verify_model(directory, check=check)
    except (OSError, ValueError):
        if required:
            raise
        return None
    return (
        before
        if before is not None and before == layout_fingerprints(directory)
        else None
    )


def current_layout_binding(
    model_info: Mapping[str, Any], session: "Session"
) -> dict[str, Any] | None:
    receipt = session.head_layout_receipt
    if receipt is None:
        return session.head_layout_binding
    if (
        receipt != layout_fingerprints(session.model)
        or model_info.get("source_directory") != receipt["directory"]
        or model_info.get("revision") != MANIFEST["revision"]
        or not isinstance(model_info.get("source_identity"), str)
    ):
        return None
    expected_model = hashlib.sha256(
        json.dumps(
            [
                "weight-atlas-model-v1",
                model_info["source_identity"],
                MANIFEST["revision"],
            ],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if model_info.get("model_identity") != expected_model:
        return None
    return {
        "source_identity": model_info["source_identity"],
        "model_identity": expected_model,
        "weights_sha256": MANIFEST["files"]["model.safetensors"],
        "config_sha256": MANIFEST["files"]["config.json"],
    }


def _contracts() -> InferenceContracts:
    """Capture the current compatibility defaults once, without model work."""
    from atlas_host import inference_edits as edits
    from atlas_host import inference_observations as observations
    from atlas_host import inference_prompt_pair as pair
    from atlas_host import inference_sweep as sweep

    value = replace(
        architecture(),
        description=ARCH,
        width=WIDTH,
        layers=LAYERS,
        query_heads=HEADS,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        vocab_size=VOCAB,
        capture_sites=CAPTURE_SITES,
    )
    return InferenceContracts(
        value,
        edits._bindings(value),
        observations._bindings(value),
        pair._bindings(value),
        sweep._bindings(value),
    )


class Session:
    def __init__(self, python: str | os.PathLike[str], model: Path) -> None:
        self.python, self.model = python, model
        self.process: subprocess.Popen[bytes] | None = None
        self.analytics: "AnalyticsJobs | None" = None
        self.inference_enabled = True
        # Internal validated correspondence only; never take from HTTP. Direct
        # Session construction does no file I/O and remains unbound by default.
        self.head_layout_binding: dict[str, Any] | None = None
        self.head_layout_receipt: LayoutReceipt | None = None
        self.stop_reason: str | None = None
        self.id: str | None = None
        self.status = "idle"
        self.steps: deque[dict[str, Any]] = deque(maxlen=MAX_TRACE)
        self.details: dict[str, Any] = {}
        self.mode = "generation"
        self.pair_request: dict[str, Any] | None = None
        self.sweep_plan: dict[str, Any] | None = None
        self.observation: dict[str, Any] | None = None
        self.capture_layer: int | None = None
        self.buffer = b""
        self.last_seen = self.started = time.monotonic()
        self.peak_rss = 0
        self.minimum_available = available()

    def snapshot(self) -> Snapshot:
        return {
            "session": self.id,
            "status": self.status,
            "steps": list(self.steps),
            "details": self.details,
            "peak_worker_rss_mib": round(self.peak_rss / 1024**2, 2),
            "minimum_available_gib": round(self.minimum_available / GIB, 3),
            "worker_alive": self.process is not None and process_alive(self.process),
        }

    def metadata(self) -> dict[str, Any]:
        contracts = _contracts()
        value = contracts.architecture
        # Shared across tabs: never include the capability or any session data.
        inference_busy = self.process is not None or self.status in (
            "loading",
            "running",
            "stopping",
        )
        analytics_busy = self.analytics is not None and self.analytics.busy
        return {
            "model": MANIFEST["repo"],
            "revision": MANIFEST["revision"],
            "engine": "CPU PyTorch / Transformers sidecar",
            "limits": {
                "prompt_tokens": 128,
                "new_tokens": 32,
                "worker_rss_mib": 1536,
                "sessions": 1,
                "capture_layers": list(range(value.layers)),
                "activation_sites": list(value.capture_sites),
                "vector_width": value.width,
            },
            "busy": inference_busy or analytics_busy,
            "busy_owner": (
                "inference session"
                if inference_busy
                else "analytics job" if analytics_busy else None
            ),
            "queue_capacity": 0,
            "architecture": value.description,
            "head_layout": bound_head_layout_descriptor(value),
            "comparison": contracts.comparison_schema(),
            "observations": contracts.observations_schema(),
            "prompt_pair": {
                "modes": list(prompt_pair.MODES),
                "prompts": 2,
                "max_prompt_bytes_each": 2048,
                "max_positions": 8,
                "vector_width": value.width,
                "max_vector_equivalents": 24,
            },
            "sweep": contracts.sweep_schema(),
        }

    def owns(self, capability: object) -> bool:
        return (
            isinstance(capability, str)
            and self.id is not None
            and secrets.compare_digest(
                capability.encode("utf-8"), self.id.encode("ascii")
            )
        )

    def stop(self, reason: str = "cancelled") -> bool:
        if self.mode == "sweep" and self.sweep_plan:
            current = self.details.get("sweep_current")
            remaining = sweep.record_ids(self.sweep_plan)[
                len(self.steps) : len(self.steps) + 1
            ]
            self.details["sweep_coverage"] = sweep.coverage(
                self.sweep_plan,
                len(self.steps),
                current if current in remaining else None,
            )
        if self.process is not None:
            if self.stop_reason is None:
                self.stop_reason = (
                    self.status if self.status in ("complete", "error") else reason
                )
            self.status = "stopping"
            self.buffer = b""
            if not signal_and_reap(self.process):
                self.details["cleanup_pending"] = True
                return False
            for stream in (self.process.stdin, self.process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
            self.process = None
            self.status, self.stop_reason = self.stop_reason, None
            self.details.pop("cleanup_pending", None)
        self.buffer = b""
        if self.status in ("loading", "running"):
            self.status = reason
        return True

    def start(self, data: dict[str, Any]) -> Snapshot:
        job_started = time.monotonic()
        admission_cpu_started = time.process_time()
        self.tick()
        if not self.inference_enabled:
            raise ValueError("Inference disabled in analytics-only mode")
        if self.analytics is not None and self.analytics.busy:
            raise ValueError(
                "Analysis owns local compute; wait for cleanup before inference"
            )
        if self.process is not None or self.status in ("loading", "running"):
            raise ValueError("A session is active; cancel or reset it first")
        contracts = _contracts()
        value = contracts.architecture
        mode = data.get("mode", "generation")
        experiment = for_coordinator(data, mode)
        sweep_plan: dict[str, Any] | None = None
        observation: dict[str, Any] | None
        layer: int | None
        request: dict[str, Any]
        if experiment is REGISTRY[Kind.SWEEP]:
            sweep_plan = contracts.build_plan(data)
            request = dict(data)
            observation, layer = None, data["capture_layer"]
        elif experiment in (REGISTRY[Kind.PREVIEW], REGISTRY[Kind.PROMPT_PAIR]):
            request = contracts.validate_request(data)
            observation, layer = None, request.get("layer")
        elif experiment in (REGISTRY[Kind.GENERATION], REGISTRY[Kind.COMPARISON]):
            prompt = data.get("prompt")
            limit, layer = data.get("max_new_tokens", 16), data.get("layer", 0)
            if (
                not isinstance(prompt, str)
                or not prompt.strip()
                or len(prompt.encode()) > 4096
            ):
                raise ValueError("Enter a nonempty prompt of at most 4096 UTF-8 bytes")
            if type(limit) is not int or not 1 <= limit <= MAX_TRACE:
                raise ValueError("Generate 1–32 tokens")
            if type(layer) is not int or not 0 <= layer < value.layers:
                raise ValueError(f"Choose activation layer 0–{value.layers-1}")
            activation_site = data.get("activation_site", "block")
            if (
                type(activation_site) is not str
                or activation_site not in value.capture_sites
            ):
                raise ValueError("Choose activation site block, attention, or mlp")
            observation = (
                contracts.validate_observation(data["observation"], activation_site)
                if "observation" in data
                else None
            )
            comparison = {}
            if "edits" in data:
                comparison = {
                    "edits": contracts.validate_edits(
                        data["edits"], data.get("source_model")
                    ),
                    "source_model": contracts.edits.source_model,
                }
            elif "source_model" in data:
                raise ValueError("source_model requires an explicit edits list")
            request = {
                "prompt": prompt,
                "max_new_tokens": limit,
                "layer": layer,
                "activation_site": activation_site,
                **comparison,
                **({"observation": observation} if observation is not None else {}),
            }
        else:
            raise ValueError("Unknown inference mode")
        if available() < 4.75 * GIB:
            raise ValueError(
                "Need 4.75 GiB available RAM before loading the inference model"
            )
        # Verify again each run, bounded streaming read; fail before model execution.
        budget = (
            SweepAdmissionBudget(
                job_started + sweep.WALL_SECONDS, admission_cpu_started + 90
            )
            if mode == "sweep"
            else None
        )
        if budget:
            with budget.verification():
                verify_model(self.model, check=budget.check)
        else:
            verify_model(self.model)
        self.mode = mode
        self.sweep_plan = sweep_plan
        self.pair_request = request if mode == "prompt_pair" else None
        self.observation, self.capture_layer = observation, layer
        self.id = secrets.token_hex(16)
        self.status, self.steps, self.details, self.buffer = (
            "loading",
            deque(maxlen=MAX_TRACE),
            {},
            b"",
        )
        self.last_seen = time.monotonic()
        self.started = job_started if mode == "sweep" else self.last_seen
        if sweep_plan:
            self.details.update(
                sweep_plan=sweep_plan, sweep_coverage=sweep.coverage(sweep_plan, 0)
            )
        self.peak_rss, self.minimum_available = 0, available()
        try:
            worker_budget = (
                [str(budget.deadline), str(budget.remaining_cpu())] if budget else []
            )
            self.process = subprocess.Popen(
                [
                    self.python,
                    "-B",
                    str(ROOT / "tools/inference_worker.py"),
                    str(self.model),
                    *worker_budget,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                bufsize=0,
            )
            cast(IO[bytes], self.process.stdin).write(
                json.dumps(request, allow_nan=False, ensure_ascii=False).encode()
                + b"\n"
            )
            cast(IO[bytes], self.process.stdin).close()
            os.set_blocking(cast(IO[bytes], self.process.stdout).fileno(), False)
        except Exception:
            self.stop("error")
            raise ValueError(
                "Could not start the configured Python inference runtime"
            ) from None
        return self.snapshot()

    def tick(self) -> None:
        if self.analytics is not None:
            self.analytics.tick()
        if self.process is None:
            return
        if self.stop_reason is not None:
            self.stop()
            return
        now = time.monotonic()
        mem = available()
        self.minimum_available = min(self.minimum_available, mem)
        try:
            status = Path(f"/proc/{self.process.pid}/status").read_text()
            rss = (
                int(
                    next(
                        (
                            s.split()[1]
                            for s in status.splitlines()
                            if s.startswith("VmRSS:")
                        ),
                        0,
                    )
                )
                * 1024
            )
            self.peak_rss = max(self.peak_rss, rss)
        except OSError:
            pass
        reason = (
            "resource_limit"
            if mem < 3.25 * GIB or self.peak_rss > 1.5 * GIB
            else (
                "client_timeout"
                if now - self.last_seen > 15
                else "time_limit" if now - self.started > 120 else None
            )
        )
        if reason:
            self.details["error"] = reason
            self.stop(reason)
            return
        contracts = _contracts()
        value = contracts.architecture
        # Bounded drain. A full pipe blocks the worker; no unbounded queue/thread.
        eof = False
        for _ in range(8):
            try:
                data = os.read(cast(IO[bytes], self.process.stdout).fileno(), 65536)
            except BlockingIOError:
                break
            if not data:
                eof = True
                break
            self.buffer += data
            if len(self.buffer) > 131072:
                self.details["error"] = "Worker output exceeded bound"
                self.stop("error")
                return
            while b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                try:
                    event = json.loads(
                        line,
                        parse_constant=lambda value: (_ for _ in ()).throw(
                            ValueError("Nonfinite JSON")
                        ),
                    )
                    kind = event.pop("type")
                    if kind == "step":
                        if len(self.steps) >= MAX_TRACE or event["index"] != len(
                            self.steps
                        ):
                            raise ValueError("Invalid step sequence")
                        if len(event["activation"]) != value.width:
                            raise ValueError("Invalid activation width")
                        if "baseline" in event or "edited" in event:
                            contracts.validate_pair(event)
                        if self.mode == "sweep":
                            contracts.validate_sweep_step(
                                event, cast(dict[str, Any], self.sweep_plan)
                            )
                        elif self.mode == "prompt_pair":
                            contracts.validate_pair_step(
                                event, cast(dict[str, Any], self.pair_request)
                            )
                        elif self.mode == "prompt_pair_preview":
                            raise ValueError(
                                "Preview cannot produce activation records"
                            )
                        else:
                            contracts.validate_record(
                                event, self.observation, self.capture_layer
                            )
                        self.steps.append(event)
                        if self.mode == "sweep":
                            self.details["sweep_coverage"] = sweep.coverage(
                                cast(dict[str, Any], self.sweep_plan), len(self.steps)
                            )
                        self.status = "running"
                    elif kind == "error":
                        self.status = "error"
                        self.details.update(event)
                        self.stop()
                        return
                    elif kind == "sweep_done":
                        if (
                            self.mode != "sweep"
                            or event.get("status") not in ("complete", "time_limit")
                            or event.get("coverage")
                            != sweep.coverage(
                                cast(dict[str, Any], self.sweep_plan), len(self.steps)
                            )
                            or (event["status"] == "complete")
                            != (
                                len(self.steps)
                                == cast(dict[str, Any], self.sweep_plan)["records"]
                            )
                        ):
                            raise ValueError(
                                "Invalid sweep completion/partial coverage"
                            )
                        self.status = event.pop("status")
                        self.details.update(event)
                        self.details["sweep_coverage"] = event["coverage"]
                        self.stop(self.status)
                        return
                    elif kind == "preview_done":
                        if (
                            self.mode != "prompt_pair_preview"
                            or self.steps
                            or set(event) != {"preview", "reason"}
                            or event["reason"] != "token_preview"
                        ):
                            raise ValueError("Unexpected tokenizer completion")
                        contracts.validate_preview(event["preview"])
                        self.details.update(event)
                        self.status = "complete"
                        self.stop()
                        return
                    elif kind == "done":
                        if self.mode in ("prompt_pair_preview", "sweep"):
                            raise ValueError(
                                "This mode requires its specific completion event"
                            )
                        if self.mode == "prompt_pair" and event.get(
                            "record_count"
                        ) != len(cast(dict[str, Any], self.pair_request)["positions"]):
                            raise ValueError("Prompt-pair coverage incomplete")
                        count = (
                            event.get("record_count")
                            if self.mode == "prompt_pair"
                            else event.get("generated_tokens")
                        )
                        if (
                            type(count) is not int
                            or count != len(self.steps)
                            or not self.steps
                        ):
                            raise ValueError("Invalid completion count")
                        self.status = "complete"
                        self.details.update(event)
                        self.stop()  # Reap before reporting completion; release model/KV now.
                        return
                    elif kind in ("prefill", "loaded"):
                        self.details.update(event)
                    else:
                        raise ValueError("Unknown worker record")
                except (ValueError, KeyError, TypeError):
                    self.details["error"] = "Invalid worker output"
                    self.stop("error")
                    return
        exited = not process_alive(self.process)
        if exited and eof:
            # Exit does not imply an empty pipe: EAGAIN or the per-tick read
            # budget can leave final records unread. Keep the pipe for later
            # ticks until EOF (or an explicit terminal record) is observed.
            if self.status not in ("complete", "error"):
                self.details["error"] = "Inference worker exited before completion"
                self.status = "error"
            self.stop()


class CoordinatorServer(Protocol):
    session: Session
    analytics: "AnalyticsJobs"
    server_port: int
    atlas_port: int
    timeout: float | None

    def handle_request(self) -> None: ...
    def server_close(self) -> None: ...


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(REQUEST_DEADLINE)

    def handle(self) -> None:
        # A complete request has an absolute 0.5 s receive deadline, so a slow
        # socket cannot starve the single coordinator's resource/lease checks.
        needed: int | None
        deadline, raw, needed = time.monotonic() + REQUEST_DEADLINE, bytearray(), None
        try:
            while needed is None or len(raw) < needed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                self.connection.settimeout(remaining)
                chunk = self.connection.recv(4096)
                if not chunk:
                    return
                raw.extend(chunk)
                if len(raw) > 8192 + MAX_BODY:
                    return
                boundary = raw.find(b"\r\n\r\n")
                if boundary < 0:
                    if len(raw) > 8192:
                        return
                    continue
                if boundary + 4 > 8192:
                    return
                headers = http.client.parse_headers(
                    io.BytesIO(bytes(raw).split(b"\r\n", 1)[1])
                )
                length = int(headers.get("Content-Length", "0"))
                if not 0 <= length <= MAX_BODY:
                    return
                needed = boundary + 4 + length
            self.rfile = io.BytesIO(raw[:needed])
            self.connection.settimeout(REQUEST_DEADLINE)
            super().handle()
        except (ValueError, OSError, http.client.HTTPException):
            return

    def log_message(self, *_args: object) -> None:
        pass  # URLs, prompts and errors are not logged.

    def send(self, status: int, body: object, mime: str = "application/json") -> None:
        if not isinstance(body, bytes):
            body = json.dumps(body, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data: blob:; script-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def validated(self) -> tuple[str, int]:
        port = cast(CoordinatorServer, self.server).server_port
        host = self.headers.get("Host")
        if host not in (f"localhost:{port}", f"127.0.0.1:{port}"):
            raise ValueError("Host must be loopback")
        if (
            self.headers.get("Origin", f"http://{host}") != f"http://{host}"
            or self.headers.get("Sec-Fetch-Site") == "cross-site"
        ):
            raise ValueError("Same-origin requests required")
        if (
            self.headers.get("Transfer-Encoding")
            or len(self.headers.get_all("Content-Length", [])) > 1
        ):
            raise ValueError("Invalid request framing")
        if sum(len(k) + len(v) for k, v in self.headers.items()) > 8192:
            raise ValueError("Headers exceed limit")
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 <= length <= MAX_BODY:
            raise ValueError("Request body exceeds limit")
        path = urlsplit(self.path).path
        if (
            not self.path.startswith("/")
            or self.path.startswith("//")
            or len(self.path) > 8192
        ):
            raise ValueError("Invalid local URL")
        return path, length

    def handle_action(self) -> None:
        session = cast(CoordinatorServer, self.server).session
        try:
            path, length = self.validated()
            if self.command == "POST" and self.headers.get("X-Atlas-Local") != "1":
                raise ValueError("Local action header required")
            if path.startswith("/api/analytics"):
                from analytics.service import route

                return route(
                    cast("AnalyticsHandler[None]", self),
                    path,
                    length,
                    cast(CoordinatorServer, self.server).analytics,
                )
            if path.startswith("/api/inference"):
                if not session.inference_enabled:
                    return self.send(
                        503, {"error": "Inference disabled in analytics-only mode"}
                    )
                if self.command == "GET" and path == "/api/inference":
                    session.tick()
                    return self.send(200, session.metadata())
                if self.command != "POST":
                    return self.send(404, {"error": "Not found"})
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("JSON object required")
                if path == "/api/inference/sweep-plan":
                    return self.send(
                        200, {"plan": sweep.build_plan(data, require_digest=False)}
                    )
                if path == "/api/inference/start":
                    return self.send(202, session.start(data))
                if not session.owns(data.get("session")):
                    return self.send(409, {"error": "Stale session"})
                if path == "/api/inference/poll":
                    session.last_seen = time.monotonic()
                    session.tick()
                elif path == "/api/inference/cancel":
                    session.stop()
                elif path == "/api/inference/reset":
                    if not session.stop():
                        return self.send(
                            503,
                            {
                                "error": "Worker cleanup pending; retry reset",
                                "code": "cleanup_pending",
                            },
                        )
                    session.id, session.status = None, "idle"
                    session.steps.clear()
                    session.details.clear()
                else:
                    return self.send(404, {"error": "Not found"})
                return self.send(200, session.snapshot())
            # Exact, task-owned assets only; no filesystem path from HTTP input.
            analytics_assets = {
                "/analytics-panel.js": "text/javascript",
                "/analytics-mount.js": "text/javascript",
                "/analytics-panel.css": "text/css",
                "/app.js": "text/javascript",
            }
            if self.command == "GET" and path in analytics_assets:
                return self.send(
                    200, (ROOT / "web" / path[1:]).read_bytes(), analytics_assets[path]
                )
            if self.command == "GET" and path in ("/", "/index.html"):
                body = (ROOT / "web/index.html").read_bytes()
                extras = b'<link rel="stylesheet" href="/analytics-panel.css"><script type="module" src="/analytics-mount.js"></script>'
                return self.send(
                    200,
                    body.replace(b"</head>", extras + b"</head>", 1),
                    "text/html; charset=utf-8",
                )
            if self.command == "GET" and path == "/inference.js":
                return self.send(
                    200, (ROOT / "web/inference.js").read_bytes(), "text/javascript"
                )
            # Preserve Rust's fixed asset/API allowlist and validation. No user-selected upstream.
            if length:
                raise ValueError("Only inference actions accept request bodies")
            status, body, mime = proxy_request(
                cast(CoordinatorServer, self.server).atlas_port,
                self.command,
                self.path,
                session,
            )
            if self.command == "GET" and path == "/api/model" and status == 200:
                body = json.dumps(
                    bind_inference_source(json.loads(body), session), allow_nan=False
                ).encode()
            self.send(status, body, mime)
        except BackendError as exc:
            self.send(exc.status, {"error": str(exc), "code": exc.code})
        except (ValueError, TimeoutError, OSError, http.client.HTTPException) as exc:
            message = (
                str(exc)
                if isinstance(exc, ValueError)
                else "Local backend unavailable or request timed out"
            )
            self.send(400, {"error": message})

    do_GET = handle_action
    do_POST = handle_action


def bind_inference_source(
    model_info: Mapping[str, Any], session: Session
) -> dict[str, Any]:
    # Only a native, single-source view of the pinned inference directory can
    # supply edit coordinates. A paired checkpoint view has a different space.
    model_info = dict(model_info)
    model_info.pop("inference_source_model", None)
    if (
        session.inference_enabled
        and model_info.get("inference_editable", True) is True
        and "comparison_identity" not in model_info
        and "coordinate_space" not in model_info
        and isinstance(model_info.get("source_directory"), str)
        and Path(model_info["source_directory"]).resolve() == session.model.resolve()
        and model_info.get("revision") == MANIFEST["revision"]
    ):
        model_info["inference_source_model"] = SOURCE_MODEL
    return bind_viewer_head_layout(
        model_info, current_layout_binding(model_info, session)
    )


def cleanup_owned(
    session: Session, atlas: OwnedProcess, server: ClosableServer
) -> bool:
    worker_done = atlas_done = False
    try:
        # Repeated bounded attempts retain the same child; no replacement work.
        for _ in range(8):
            worker_done = session.stop()
            analytics: "AnalyticsJobs | None" = getattr(session, "analytics", None)
            if analytics is not None:
                worker_done = analytics.stop() and worker_done
            if worker_done:
                break
            time.sleep(0.05)
    finally:
        try:
            atlas_done = signal_and_reap(atlas, terminate=True, timeout=0.5)
            if not atlas_done:
                atlas_done = signal_and_reap(atlas, timeout=1.5)
        finally:
            server.server_close()
    return worker_done and atlas_done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--python", required=True, help="Existing CPU PyTorch/Transformers Python"
    )
    parser.add_argument(
        "--analytics-only",
        action="store_true",
        help="Disable inference; permit bounded analytics of the configured source",
    )
    parser.add_argument("--name")
    parser.add_argument("--revision")
    parser.add_argument("--port", type=int, default=8796)
    parser.add_argument("--atlas-port", type=int, default=8797)
    args = parser.parse_args()
    args.name = args.name or (
        args.model.name + " · bounded analytics"
        if args.analytics_only
        else "SmolLM2-135M · fixed weights"
    )
    args.revision = args.revision or (
        "local source; revision not supplied"
        if args.analytics_only
        else MANIFEST["revision"]
    )
    if args.port == args.atlas_port or {args.port, args.atlas_port} & {
        8774,
        8775,
        8785,
    }:
        parser.error("Use distinct fresh ports; existing viewer ports are reserved")
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    if available() < 4.75 * GIB:
        raise SystemExit("Need 4.75 GiB available RAM")
    receipt = verified_layout_receipt(args.model, required=not args.analytics_only)
    session = Session(args.python, args.model.resolve())
    session.inference_enabled = not args.analytics_only
    session.head_layout_receipt = receipt
    server = cast(CoordinatorServer, HTTPServer(("127.0.0.1", args.port), Handler))
    server.timeout = 0.1
    server.session, server.atlas_port = session, args.atlas_port
    from analytics.service import AnalyticsJobs

    def analysis_model() -> Any:
        status, body, _mime = proxy_request(
            args.atlas_port, "GET", "/api/model", session
        )
        if status != 200:
            raise ValueError(
                "Validated model metadata unavailable; retry after renderer startup"
            )
        return json.loads(body)

    server.analytics = AnalyticsJobs(
        args.python,
        args.model,
        inference_busy=lambda: session.process is not None
        or session.status in ("loading", "running", "stopping"),
        fetch_model=analysis_model,
        available=available,
        reap=signal_and_reap,
    )
    session.analytics = server.analytics
    atlas = subprocess.Popen(
        [
            str(ROOT / "target/release/weight-atlas-rust"),
            "serve",
            "--model",
            str(args.model.resolve()),
            "--cache",
            str(ROOT / "cache-inference"),
            "--name",
            args.name,
            "--revision",
            args.revision,
            "--port",
            str(args.atlas_port),
        ],
        stdout=subprocess.DEVNULL,
    )

    def shutdown(*_args: object) -> NoReturn:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    label = (
        "Bounded analytics (inference disabled)"
        if args.analytics_only
        else "Experimental live inference and bounded analytics"
    )
    print(f"{label}: http://127.0.0.1:{args.port}", flush=True)
    try:
        while atlas.poll() is None:
            server.handle_request()
            session.tick()
            if available() < 3.25 * GIB:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if not cleanup_owned(session, atlas, server):
            raise SystemExit(
                "Shutdown incomplete: owned process cleanup remains pending"
            )


if __name__ == "__main__":
    main()
