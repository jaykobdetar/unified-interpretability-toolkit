"""Strict OS ownership clients; function declarations perform no OS work."""

from collections.abc import Callable
from pathlib import Path
import subprocess
from typing import Any, assert_type

from atlas_host.profile_os import (
    BinaryPolicy,
    FrozenBinary,
    LinuxProfileHooks,
    OwnedProcess,
    PlatformFacts,
    ProcessBook,
    ProcessStats,
    ProfileSource,
    ProfileWatchdog,
    ResourceSnapshot,
    SnapshotOwner,
    ThreadMeter,
    platform_capabilities,
    proc_stat,
)
from atlas_host.profile_snapshot import SnapshotStore
from atlas_host.profile_worker import WorkerChild, WorkerRequest


def owned_process(process: subprocess.Popen[bytes]) -> WorkerChild:
    child = OwnedProcess(process)
    assert_type(child.sample(), dict[str, Any])
    assert_type(child.streams_closed, bool)
    return child


def system_observations(book: ProcessBook) -> ResourceSnapshot:
    probe: Callable[[int], ProcessStats] = proc_stat
    assert_type(probe, Callable[[int], ProcessStats])
    assert_type(platform_capabilities(), PlatformFacts)
    result = assert_type(book.resources(), ResourceSnapshot)
    assert_type(result["rss_bytes"], int)
    assert_type(result["all_owned_accounted"], bool)
    return result


def snapshot_owner(store: SnapshotStore) -> SnapshotOwner:
    return store


def provider_client(
    book: ProcessBook,
    binary: FrozenBinary,
    source: ProfileSource,
    watchdog: ProfileWatchdog,
    policy: BinaryPolicy,
    request: WorkerRequest,
    store: SnapshotStore,
) -> OwnedProcess:
    hooks = LinuxProfileHooks(book, binary, source, watchdog, launch_policy=policy)
    assert_type(hooks.start_gate(), tuple[int, int])
    assert_type(hooks.snapshot_closed(store), bool)
    return assert_type(hooks.spawn(request, 19, None), OwnedProcess)


def frozen_executable(path: Path, expected_sha: str) -> FrozenBinary:
    return FrozenBinary(path, expected_sha)


def cpu_clock(meter: ThreadMeter) -> Callable[[], float]:
    return meter.read
