"""Opt-in live provider qualification with the existing exact synthetic fixture."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import time
from typing import Any, Callable, cast
from threading import Thread
from atlas_host.profile_platform import ProfilePlatform, SupervisedProfile
from atlas_host.profile_os import OwnedProcess
from atlas_host.profile_service import ServiceRecord
from atlas_host.supervisor import AdmissionGrant

from atlas_host.hosted_runtime import HostedApplication
from atlas_host.profile_os import require_platform
from atlas_host.registry import Registry
from profile_worker_primitives import reference_permutation

VALUES = [
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
    0.25,
    -0.25,
    0.125,
    -0.125,
    2.0,
    -2.0,
    1.5,
    -1.5,
    0.75,
    -0.75,
]


def until(probe: Callable[[], Any], seconds: float, description: str) -> Any:
    deadline = time.monotonic() + seconds
    while True:
        result = probe()
        if result:
            return result
        assert time.monotonic() < deadline, description
        time.sleep(0.01)


def reply(
    app: HostedApplication,
    action: str,
    data: dict[str, Any],
    *,
    admission: AdmissionGrant | None = None,
) -> tuple[int, dict[str, Any], dict[str, str]]:
    code, value, headers = app.api.handle(action, data, admission=admission)
    assert isinstance(value, dict)
    return code, value, headers


def exact_pages(
    app: HostedApplication,
    owner: dict[str, Any],
    status: dict[str, Any],
    selected: dict[str, Any],
    visited: int,
) -> list[dict[str, Any]]:
    pages = []
    permutation = cast(Callable[[int, int], list[int]], reference_permutation)(15, 17)
    for axis, length, expected in [("rows", 3, 5), ("columns", 5, 3)]:
        code, page, _ = reply(
            app,
            "page",
            {
                **owner,
                "revision": status["accepted"]["revision"],
                "axis": axis,
                "start": 0,
                "count": length,
            },
        )
        assert code == 200 and page["binding"] == selected
        assert page["visited_values"] == visited and page["total_values"] == 15
        for lane in ["original", "control"]:
            records = []
            for index in range(length):
                entries = [
                    abs(VALUES[i])
                    for i in range(visited)
                    if (
                        (i if lane == "original" else permutation[i]) // 5
                        if axis == "rows"
                        else (i if lane == "original" else permutation[i]) % 5
                    )
                    == index
                ]
                total, count = sum(entries), len(entries)
                records.append(
                    {
                        "index": index,
                        "sum_abs": total,
                        "visited_count": count,
                        "expected_count": expected,
                        "mean_abs": total / count if count else None,
                        "complete": count == expected,
                    }
                )
            assert page[lane] == records, (axis, lane, page[lane], records)
        pages.append(page)
    return pages


def host(binary: Path, fixture: Path, work: Path) -> dict[str, Any]:
    root = work / "fixture"
    root.mkdir()
    shutil.copyfile(fixture / "tiny.safetensors", root / "tiny.safetensors")
    raw = (root / "tiny.safetensors").read_bytes()
    assert (
        len(raw) == 244
        and hashlib.sha256(raw).hexdigest()
        == "c0075bfc55f9e51ccac3c5511ea55a5ca19744b002921e8d2e4ae3f60d321be3"
    )
    manifest = json.loads((fixture / "host-manifest.json").read_text())
    registry = Registry(work / "owner/registry.json")
    registered = registry.register(
        root, manifest, "Tiny lifecycle fixture", max_bytes=244
    )
    registry.set_enabled(registered["model_id"], True)
    cache = work / "cache"
    cache.mkdir()
    app = HostedApplication(
        registry, cache, binary, hashlib.sha256(binary.read_bytes()).hexdigest()
    )
    result: dict[str, Any] = {
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "runs": [],
    }
    try:
        app.start_threads()
        with app.lock:
            lease = app.host.acquire({"model_id": registered["model_id"]})
            assert isinstance(lease["context_id"], str)
            assert app.host.reader is not None
            code, body, _ = app.host.read(
                lease["model_id"], "model", {"context": [lease["context_id"]]}
            )
            assert code == 200
            model = json.loads(body)
            tensor = next(t for t in model["catalog"] if t["name"] == "matrix")
            code, body, _ = app.host.reader.read(
                "/api/binding?tensor=" + str(tensor["id"])
            )
            assert code == 200
            selected = json.loads(body)["source_binding"]
            app.contexts.remember(lease["model_id"], lease["context_id"], selected)
        base = {
            "version": 1,
            "model_id": lease["model_id"],
            "context_id": lease["context_id"],
            "tab_capability": lease["capability"],
        }
        first_revision = None
        for number, visited in enumerate([15, 5, 15]):
            grant, meter = app.profiles.admission()
            try:
                code, started, _ = reply(
                    app,
                    "start",
                    {
                        **base,
                        "binding": selected,
                        "seed": 17,
                        "values": visited,
                        "restart": number > 0,
                    },
                    admission=grant,
                )
                assert code == 202 and started["state"] == "admitting"
                assert app.supervisor.busy()
                assert cast(ServiceRecord, app.profiles.record)["runtime"] is None
                original_deadline = grant.deadline
                owner = {
                    **base,
                    "job_id": started["job_id"],
                    "job_capability": started["job_capability"],
                }
                assert reply(app, "status", owner)[1]["state"] == "admitting"
            finally:
                meter.freeze()
                app.profiles.finish_admission(grant)

            def terminal() -> Any:
                code, status, _ = reply(app, "status", owner)
                assert code == 200 and status["version"] == 1
                assert (
                    status["model_id"] == base["model_id"]
                    and status["context_id"] == base["context_id"]
                )
                assert status["state"] not in ("error", "cancelled", "stopping"), status
                return (
                    status
                    if status["state"] in ("complete", "partial")
                    and not status["cleanup_pending"]
                    else None
                )

            status = until(
                terminal, 5.0, "Profile did not finish inside original grant"
            )
            assert status["state"] == ("complete" if visited == 15 else "partial")
            assert status["accepted"]["visited_values"] == visited
            assert not status["resume_available"] and not app.supervisor.busy()
            assert grant.deadline == original_deadline
            runtime = cast(
                SupervisedProfile, cast(ServiceRecord, app.profiles.record)["runtime"]
            )
            child = cast(OwnedProcess, cast(ProfilePlatform, runtime.platform).child)
            assert child.reaped and child.streams_closed and not child.unexpected
            assert (
                cast(ProfilePlatform, runtime.platform).watch_closed
                and cast(ProfilePlatform, runtime.platform).closed
            )
            assert (
                runtime.job.store.owned_storage_bytes == runtime.job.store.frame_bytes
            )
            assert (
                app.supervisor.charged_snapshot_bytes()
                == 2 * runtime.job.store.frame_bytes
            )
            assert cast(ServiceRecord, app.profiles.record)["watch"].closed
            assert grant.child_cpu == child.cpu
            pages = exact_pages(app, owner, status, selected, visited)
            if number == 0:
                first_revision = status["accepted"]["revision"]
            if number == 2:
                assert status["accepted"]["revision"] == first_revision
            result["runs"].append(
                {
                    "values": visited,
                    "status": status,
                    "pages": pages,
                    "child_pid": child.pid,
                    "child_cpu_seconds": child.cpu,
                    "grant_remaining": grant.remaining(),
                    "reserved_snapshot_bytes": app.supervisor.charged_snapshot_bytes(),
                    "observed": app.book.resources(),
                    "reaped": True,
                }
            )
        reply(app, "cancel", owner)
        until(
            lambda: reply(app, "status", owner)[1]["state"] == "cancelled"
            and not app.supervisor.busy(),
            2.0,
            "Cancel cleanup deadline",
        )
        assert app.supervisor.charged_snapshot_bytes() == 0
        assert not app.profiles.watchdog.slots
        assert not app.lifetime.failed
    finally:
        until(app.close, 3.0, "Provider cleanup deadline")
        assert app.book.settled()
        assert (
            not cast(Thread, app.profiles.thread).is_alive()
            and not cast(Thread, app.profiles.watchdog.thread).is_alive()
        )
        assert not app.profiles.watchdog.slots
        assert (
            not app.supervisor.busy() and app.supervisor.charged_snapshot_bytes() == 0
        )
        result["cleanup"] = {
            "processes_reaped": all(
                c.reaped and c.streams_closed and not c.unexpected
                for c in app.book.owned
            ),
            "threads_joined": True,
            "snapshot_bytes": 0,
            "book_settled": True,
        }
        (work / "host-result.json").write_text(json.dumps(result, indent=2) + "\n")
    assert (root / "tiny.safetensors").read_bytes() == raw
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir()
    require_platform()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (768 * 1024**2, 768 * 1024**2))
    result = host(args.binary.resolve(), args.fixture.resolve(), args.output.resolve())
    print(
        json.dumps(
            {
                "status": "PASS",
                "full_partial_restart": [x["values"] for x in result["runs"]],
                "cleanup": result["cleanup"],
                "scope": "Exact synthetic fixture, actual hosted provider and resident renderer/worker; no HTTP/browser qualification",
            }
        )
    )


if __name__ == "__main__":
    main()
