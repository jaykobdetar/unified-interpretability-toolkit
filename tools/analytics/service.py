"""Single-coordinator-thread admission and owned analysis child management."""

import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from typing import Any, IO, Protocol, TypedDict, TypeVar, cast

from .source import Catalog, Tensor
from . import svd_summary
from .worker import MAX_INPUT, MAX_OUTPUT, parse_json, validate_request

GIB = 1024**3
WORKER_ROOT = Path(__file__).resolve().parent.parent


class SummaryLimits(TypedDict):
    schema: str
    axis: int
    values: int
    preview_axis: int
    body_bytes: int


class Limits(TypedDict):
    jobs: int
    queue: int
    values: int
    axis: int
    svd_axis: int
    svd_values: int
    wall_seconds: int
    svd_summary: SummaryLimits
    output_bytes: int
    worker_address_space_mib: int


class Metadata(TypedDict):
    available: bool
    busy: bool
    limits: Limits
    coverage: str


class Snapshot(TypedDict):
    job: str | None
    status: str
    result: dict[str, Any] | None
    error: str | None
    cleanup_pending: bool
    worker_alive: bool
    peak_worker_rss_mib: float


class ModelIdentity(TypedDict, total=False):
    model_identity: str
    revision: str


class ModelMetadata(ModelIdentity):
    source_identity: str
    catalog: list[dict[str, Any]]


class SummaryContract(TypedDict):
    model: ModelMetadata
    tensor: Mapping[str, Any]
    region: Mapping[str, int]
    seed: int


_Reply = TypeVar("_Reply", covariant=True)


class Handler(Protocol[_Reply]):
    @property
    def command(self) -> str: ...

    @property
    def rfile(self) -> IO[bytes]: ...

    def send(self, status: int, body: object) -> _Reply: ...


class AnalyticsJobs:
    def __init__(
        self,
        python: str | os.PathLike[str],
        model: str | os.PathLike[str],
        *,
        inference_busy: Callable[[], bool],
        fetch_model: Callable[[], Mapping[str, Any]],
        available: Callable[[], int],
        reap: Callable[[subprocess.Popen[bytes]], bool],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.python, self.model = python, Path(model).resolve()
        self.inference_busy, self.fetch_model = inference_busy, fetch_model
        self.available, self.reap, self.clock = available, reap, clock
        self.process: subprocess.Popen[bytes] | None = None
        self.id: str | None = None
        self.status = "idle"
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        self.buffer = bytearray()
        self.stop_reason: str | None = None
        self.started = self.last_seen = clock()
        self.peak_rss = 0
        self.summary_contract: SummaryContract | None = None

    @property
    def busy(self) -> bool:
        return self.process is not None or self.status in (
            "starting",
            "running",
            "stopping",
        )

    def metadata(self) -> Metadata:
        return {
            "available": True,
            "busy": self.busy or self.inference_busy(),
            "limits": {
                "jobs": 1,
                "queue": 0,
                "values": 65536,
                "axis": 4096,
                "svd_axis": 64,
                "svd_values": 4096,
                "wall_seconds": 5,
                "svd_summary": {
                    "schema": svd_summary.SCHEMA,
                    "axis": 128,
                    "values": 16384,
                    "preview_axis": 16,
                    "body_bytes": svd_summary.MAX_BODY,
                },
                "output_bytes": MAX_OUTPUT,
                "worker_address_space_mib": 768,
            },
            "coverage": "Explicit selected window or bounded catalog prefix; no full-model claim by default",
        }

    def owns(self, job: object) -> bool:
        return (
            isinstance(job, str)
            and self.id is not None
            and secrets.compare_digest(job.encode(), self.id.encode())
        )

    def snapshot(self) -> Snapshot:
        # Caller must authorize with owns() before returning this owner-private data.
        return {
            "job": self.id,
            "status": self.status,
            "result": self.result if self.status == "complete" else None,
            "error": self.error,
            "cleanup_pending": self.stop_reason is not None,
            "worker_alive": self.process is not None,
            "peak_worker_rss_mib": self.peak_rss / 1024**2,
        }

    def _model(self) -> ModelMetadata:
        metadata = self.fetch_model()
        if (
            "comparison_identity" in metadata
            or metadata.get("inference_editable") is False
        ):
            raise ValueError(
                "Analytics requires one original BF16 source, not a checkpoint comparison"
            )
        if any(t.get("dtype") != "BF16" for t in metadata.get("catalog", [])):
            raise ValueError(
                "Analytics requires an all-BF16 source catalog; F16/F32 analysis is unsupported"
            )
        if Path(metadata["source_directory"]).resolve() != self.model:
            raise ValueError(
                "Validated renderer model differs from configured analytics source"
            )
        return {
            **cast(
                ModelIdentity,
                {
                    key: metadata[key]
                    for key in ("model_identity", "revision")
                    if key in metadata
                },
            ),
            "source_identity": metadata["source_identity"],
            "catalog": [
                {
                    key: t[key]
                    for key in ("id", "name", "shape", "dtype", "shard", "byte_offset")
                }
                for t in metadata["catalog"]
            ],
        }

    def start(self, data: Any) -> Snapshot:
        self.tick()
        if self.busy or self.inference_busy():
            raise ValueError("Local compute busy; wait for inference/analysis cleanup")
        if (
            self.available() < 3.75 * GIB
            or shutil.disk_usage(WORKER_ROOT).free < 25 * GIB
        ):
            raise ValueError("Analytics paused by memory/disk reserve gates")
        self.status = "starting"
        self.result = None
        self.error = None
        self.buffer.clear()
        self.summary_contract = None
        try:
            model = self._model()
            chosen = validate_request(data, model["catalog"])
            if data.get("scope") == "svd_summary":
                svd_summary.binding(
                    model, cast(Mapping[str, Any], chosen), data["region"], data["seed"]
                )
                self.summary_contract = {
                    "model": model,
                    "tensor": cast(Mapping[str, Any], chosen),
                    "region": data["region"],
                    "seed": data["seed"],
                }
            catalog = Catalog(
                self.model,
                [
                    Tensor(t["name"], tuple(t["shape"]), t["shard"], t["byte_offset"])
                    for t in model["catalog"]
                ],
                model["source_identity"],
            )
            # Source.check() in this second trusted metadata read binds our local
            # fingerprint snapshot to the renderer's existing validated identity.
            if self._model() != model:
                raise ValueError("Source identity changed during admission")
            catalog.verify()
            payload = {
                "root": str(self.model),
                "model": model,
                "fingerprints": catalog.files,
                "index_fingerprint": catalog.index_fingerprint,
                "request": data,
            }
            raw = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
            if len(raw) > MAX_INPUT:
                raise ValueError("Analysis input/catalog exceeds 512 KiB cap")
            # Regular anonymous input file avoids blocking pipe writes on admission.
            # Child inherits only this input and its output pipe, never source FDs.
            with tempfile.TemporaryFile() as request:
                request.write(raw)
                request.seek(0)
                self.id = secrets.token_hex(16)
                self.started = self.last_seen = self.clock()
                self.peak_rss = 0
                self.process = subprocess.Popen(
                    [self.python, "-B", "-m", "analytics.worker"],
                    cwd=WORKER_ROOT,
                    stdin=request,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    env={
                        **os.environ,
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "NUMEXPR_NUM_THREADS": "1",
                        "VECLIB_MAXIMUM_THREADS": "1",
                    },
                    bufsize=0,
                )
            os.set_blocking(cast(IO[bytes], self.process.stdout).fileno(), False)
            self.status = "running"
        except Exception:
            self.status = "error"
            self.stop("error")
            raise
        return self.snapshot()

    def stop(self, reason: str = "cancelled") -> bool:
        if self.process is not None:
            if self.stop_reason is None:
                self.stop_reason = (
                    self.status if self.status in ("complete", "error") else reason
                )
            self.status = "stopping"
            if not self.reap(self.process):
                return False
            if self.process.stdout is not None:
                try:
                    self.process.stdout.close()
                except OSError:
                    pass  # Child is already reaped; no ownership remains.
            self.process = None
            self.status, self.stop_reason = self.stop_reason, None
        elif self.status in ("starting", "running"):
            self.status = reason
        self.buffer.clear()
        return True

    def tick(self) -> None:
        if self.process is None:
            if self.result is not None and self.clock() - self.last_seen > 15:
                self.result = None
                self.id = None
                self.status = "expired"
            return
        if self.stop_reason is not None:
            self.stop()
            return
        try:
            raw: str | bytes = Path(f"/proc/{self.process.pid}/status").read_text()
            rss = (
                int(
                    next(
                        (
                            line.split()[1]
                            for line in cast(str, raw).splitlines()
                            if line.startswith("VmRSS:")
                        ),
                        "0",
                    )
                )
                * 1024
            )
            self.peak_rss = max(self.peak_rss, rss)
        except OSError:
            pass
        reason = (
            "resource_limit"
            if self.available() < 3.25 * GIB or self.peak_rss > 768 * 1024**2
            else (
                "time_limit"
                if self.clock() - self.started > 5
                else "client_timeout" if self.clock() - self.last_seen > 15 else None
            )
        )
        if reason:
            self.error = reason
            self.stop(reason)
            return
        eof = False
        for _ in range(4):  # 256 KiB maximum drain per tick; no output reader thread.
            try:
                chunk = os.read(cast(IO[bytes], self.process.stdout).fileno(), 65536)
            except BlockingIOError:
                break
            except OSError:
                self.error = "Analysis pipe unavailable"
                self.status = "error"
                self.stop()
                return
            if not chunk:
                eof = True
                break
            self.buffer.extend(chunk)
            output_limit = (
                svd_summary.MAX_BODY + 2048
                if self.summary_contract is not None
                else MAX_OUTPUT
            )
            if len(self.buffer) > output_limit + 1:
                self.error = "Analysis output cap exceeded"
                self.status = "error"
                self.stop()
                return
            if b"\n" in self.buffer:
                try:
                    raw, rest = bytes(self.buffer).split(b"\n", 1)
                    if rest or len(raw) > output_limit:
                        raise ValueError("Unexpected worker output")
                    event = parse_json(raw)
                    if event.get("ok") is True and isinstance(
                        event.get("result"), dict
                    ):
                        if self.summary_contract is not None:
                            svd_summary.validate(
                                event["result"], **self.summary_contract
                            )
                        self.result = event["result"]
                        self.status = "complete"
                    elif event.get("ok") is False and isinstance(
                        event.get("error"), str
                    ):
                        self.error = event["error"][:512]
                        self.status = "error"
                    else:
                        raise ValueError("Invalid worker result")
                except (ValueError, TypeError, AttributeError):
                    self.error = "Invalid worker result"
                    self.status = "error"
                self.stop()
                return  # Always reap before releasing compute admission.
        if eof:
            self.error = "Worker exited without a complete result"
            self.status = "error"
            self.stop()


def route(
    handler: Handler[_Reply], path: str, length: int, jobs: AnalyticsJobs
) -> _Reply:
    """Called only AFTER the coordinator's existing framing/origin/header checks."""
    jobs.tick()
    if handler.command == "GET" and path == "/api/analytics":
        return handler.send(200, jobs.metadata())
    if handler.command != "POST":
        return handler.send(404, {"error": "Not found"})
    data = parse_json(handler.rfile.read(length))
    if not isinstance(data, dict):
        raise ValueError("JSON object required")
    if path == "/api/analytics/start":
        if jobs.busy or jobs.inference_busy():
            return handler.send(
                409,
                {
                    "error": "Local compute busy; wait for the active job to finish",
                    "code": "compute_busy",
                },
            )
        return handler.send(202, jobs.start(data))
    if set(data) != {"job"} or not jobs.owns(data.get("job")):
        return handler.send(409, {"error": "Stale analysis job", "code": "stale_job"})
    if path == "/api/analytics/poll":
        jobs.last_seen = jobs.clock()
        jobs.tick()
    elif path == "/api/analytics/cancel":
        jobs.stop()
    else:
        return handler.send(404, {"error": "Not found"})
    return handler.send(200, jobs.snapshot())
