"""Opt-in real ownership observations and independent bounded watchdog fixture."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any, cast
from threading import Thread

from atlas_host.profile_os import (
    FrozenBinary,
    ProcessStats,
    OwnedProcess,
    LinuxProfileHooks,
    ProcessBook,
    ThreadMeter,
    proc_stat,
    require_platform,
)
from atlas_host.profile_platform import ProfilePlatform, SupervisedProfile
from atlas_host.profile_service import WatchdogLoop
from atlas_host.supervisor import AdmissionGrant, CpuLedger, Supervisor


from profile_provider_lifecycle import until


def aggregate() -> dict[str, Any]:
    readings: list[tuple[int, dict[str, Any]]] = []

    def observe(pid: int) -> ProcessStats:
        value = proc_stat(pid)
        readings.append((pid, dict(value)))
        return value

    book = ProcessBook(stats=observe)
    result: dict[str, Any] = {}
    try:
        for _ in range(2):
            child = book.spawn(
                [sys.executable, "-c", "import time; time.sleep(10)"], receipt=False
            )
            child.initialize()
        until(
            lambda: all(proc_stat(c.pid)["rss"] > 0 for c in book.owned),
            0.5,
            "Quiet children did not reach resident startup",
        )
        readings.clear()
        measured = book.resources()
        assert {pid for pid, _ in readings} == {book.root, *(c.pid for c in book.owned)}
        assert len(readings) == 3
        assert measured["rss_bytes"] == sum(value["rss"] for _, value in readings)
        assert (
            measured["all_owned_accounted"] is True
            and measured["descendants_clear"] is True
        )
        assert all(value["rss"] > 0 for _, value in readings)
        assert not book.settled()
        result.update(
            observed=measured,
            process_readings=readings.copy(),
            live_children=[c.pid for c in book.owned],
        )
    finally:
        until(book.cleanup, 3.0, "Aggregate fixture children not reaped")
        assert all(
            c.reaped and c.streams_closed and not c.unexpected for c in book.owned
        )
        assert not book.spawn_uncertain
        result["cleanup"] = {"book_settled": book.settled(), "reaped": True}
    return result


class QuietHooks(LinuxProfileHooks):
    """Only the test worker body is a quiet bounded child; ownership hooks are real."""

    def spawn(self, request: Any, output_fd: int, input_fd: int | None) -> Any:
        assert input_fd is None and not self.attempted
        self.attempted = True
        self.source_check()
        assert request["binding"] == self.source["binding"]
        self.child = self.book.spawn(
            [sys.executable, "-c", "import time; time.sleep(10)"], receipt=False
        )
        return self.child


def watchdog(binary: Path, binding: dict[str, Any], work: Path) -> dict[str, Any]:
    owner, admission, watch_meter = ThreadMeter(), ThreadMeter(), ThreadMeter()
    # The fixture's admission context is frozen before owner work begins.
    admission.register()
    admission.freeze()
    owner.register()
    loop = WatchdogLoop(watch_meter)
    loop.start()
    book = ProcessBook()
    supervisor = Supervisor()
    grant = AdmissionGrant(
        time.monotonic,
        CpuLedger(
            {
                "admission": admission.read,
                "owner": owner.read,
                "watchdog": watch_meter.read,
            }
        ),
        wall_ms=2000,
    )
    slot = loop.bind(lambda: None, grant.deadline)
    frozen = FrozenBinary(binary, hashlib.sha256(binary.read_bytes()).hexdigest())
    hooks = QuietHooks(
        book,
        frozen,
        {
            "root": work,
            "revision": "fixture-v1",
            "cache": work,
            "binding": binding,
            "check": frozen.check,
        },
        slot,
    )
    runtime = SupervisedProfile(
        supervisor, grant, hooks, binding, 17, "a" * 64, "watchdog-fixture"
    )
    result: dict[str, Any] = {}
    try:
        runtime.start(1)
        child = cast(OwnedProcess, cast(ProfilePlatform, runtime.platform).child)
        assert child is not None and supervisor.busy()
        original_deadline = grant.deadline
        # No HTTP, owner tick or status call drives this deadline callback.
        assert runtime.job.cancel_event.wait(3.0), "Independent watchdog did not expire"
        until(
            lambda: child.stop_at is not None
            and cast(ProfilePlatform, runtime.platform).reserved.poisoned,
            0.2,
            "Watchdog callback did not complete",
        )
        assert runtime.job.expired
        assert grant.deadline == original_deadline
        assert (
            not cast(ProfilePlatform, runtime.platform).reserved.current()
            and supervisor.busy()
        )
        assert (
            runtime.job.accepted is None
            and not cast(ProfilePlatform, runtime.platform).closed
        )
        result.update(
            child_pid=child.pid,
            expired_without_owner_tick=True,
            original_deadline=original_deadline,
            stop_at=child.stop_at,
            busy_before_reap=True,
            snapshot_charge_before_cleanup=supervisor.charged_snapshot_bytes(),
        )
    finally:

        def finish() -> bool:
            runtime.tick()
            return cast(ProfilePlatform, runtime.platform).closed

        until(finish, 3.0, "Watchdog fixture owner cleanup deadline")
        assert runtime.job.accepted is None
        assert book.settled() and not supervisor.busy()
        runtime.close()
        assert supervisor.charged_snapshot_bytes() == 0
        loop.close()
        owner.freeze()
        assert not cast(Thread, loop.thread).is_alive() and not loop.slots
        result["cleanup"] = {
            "child_reaped": cast(OwnedProcess, hooks.child).reaped,
            "book_settled": book.settled(),
            "watchdog_joined": True,
            "snapshot_bytes": 0,
            "busy": supervisor.busy(),
            "terminal_state": runtime.job.state,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--host-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir()
    require_platform()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (768 * 1024**2, 768 * 1024**2))
    binding = json.loads(args.host_result.read_text())["runs"][0]["pages"][0]["binding"]
    result = {"aggregate": aggregate()}
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    result["watchdog"] = watchdog(args.binary.resolve(), binding, args.output.resolve())
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "PASS", **result}))


if __name__ == "__main__":
    main()
