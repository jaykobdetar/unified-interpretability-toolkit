"""Strict clients of real observation and diagnostic leaves."""

from collections.abc import Mapping
from pathlib import Path
from typing import assert_type

from atlas_host.profile_observation import SpawnObservationPending
from atlas_host.startup_diagnostics import (
    CaptureError,
    DiagnosticsSnapshot,
    ErrorChain,
    RecordSnapshot,
    SafeError,
    StartupDiagnostics,
    StartupRecord,
    diagnose,
    diagnosed,
    error_chain,
    safe_error,
    sanitized_stderr,
)


@diagnosed("typed.fixture")
def decorated(owner: object, value: int, *, label: str) -> tuple[int, str]:
    return value, label


def observations(observed: Mapping[str, int]) -> Mapping[str, int]:
    pending = SpawnObservationPending(observed)
    return assert_type(pending.observed, Mapping[str, int])


def diagnostics(owner: object, error: BaseException, raw: bytes) -> ErrorChain:
    diagnose(owner, "typed.fixture", error, rss_bytes=3)
    assert_type(decorated(owner, 7, label="typed"), tuple[int, str])
    assert_type(safe_error(error), SafeError)
    assert_type(sanitized_stderr(raw), str)
    locations: dict[str | Path, str] = {Path("fixture.py"): "fixture"}
    return assert_type(error_chain(error, locations=locations), ErrorChain)


def records(record: StartupRecord, error: BaseException) -> RecordSnapshot:
    record.update(pid=42, reaped=True, exit_code=0, wait_status=0, stderr_eof=True)
    record.feed(b"inert")
    record.capture_error("read", error)
    snapshot = assert_type(record.snapshot(), RecordSnapshot)
    assert_type(snapshot["capture_errors"], list[CaptureError])
    assert_type(snapshot["stderr_bytes_seen"], int)
    assert_type(snapshot["stderr_sha256_complete"], bool)
    return snapshot


def collector(
    value: StartupDiagnostics, argv: list[str], fd: int
) -> DiagnosticsSnapshot:
    assert_type(value.prepare(argv, (fd,)), StartupRecord)
    value.event("typed.fixture", rss_bytes=31, ignored=object())
    snapshot = assert_type(value.snapshot(), DiagnosticsSnapshot)
    assert_type(snapshot["attempts"], list[RecordSnapshot])
    assert_type(snapshot["events"][0]["count"], int)
    return snapshot
