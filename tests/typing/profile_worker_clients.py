"""Strict real worker callers and shared supervisor token compatibility."""

from typing import Any, assert_type

from atlas_host.profile_snapshot import SnapshotPage
from atlas_host.profile_worker import (
    AcceptedProgress,
    ProfileJob,
    WorkerPlatform,
    WorkerStatus,
    WorkerToken,
)
from atlas_host.supervisor import ComputeToken


def actual_shared_token(token: ComputeToken) -> WorkerToken:
    shared: WorkerToken = token
    assert_type(shared.current(), bool)
    return shared


def new_job(platform: WorkerPlatform, selected: dict[str, Any]) -> ProfileJob:
    job = ProfileJob(selected, 17, "t" * 32, "context", platform)
    assert_type(job.store.owned_storage_bytes, int)
    return job


def lifecycle(job: ProfileJob, tab: str, context: str) -> WorkerStatus:
    reply = assert_type(
        job.start(tab, job.job_capability, context, 1, wall_ms=5000, cpu_ms=4000),
        WorkerStatus,
    )
    assert_type(reply["accepted"], AcceptedProgress | None)
    assert_type(job.tick(), bool)
    job.heartbeat(tab, job.job_capability, context)
    job.cancel(tab, job.job_capability, context)
    return assert_type(job.status(tab, job.job_capability, context), WorkerStatus)


def accepted_page(
    job: ProfileJob, tab: str, context: str, revision: str
) -> SnapshotPage:
    return assert_type(
        job.page(tab, job.job_capability, context, revision, "rows", 0, 1),
        SnapshotPage,
    )
