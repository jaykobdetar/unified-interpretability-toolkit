"""Strict platform clients retain actual hooks, grants and primitive interfaces."""

from typing import Any, assert_type

from atlas_host.profile_os import LinuxProfileHooks
from atlas_host.profile_platform import (
    DeferredToken,
    ProfileHooks,
    ProfilePlatform,
    SupervisedProfile,
)
from atlas_host.profile_snapshot import SnapshotPage
from atlas_host.profile_worker import WorkerPlatform, WorkerStatus, WorkerToken
from atlas_host.supervisor import AdmissionGrant, ComputeToken, ProfileOwner, Supervisor


def actual_hooks(hooks: LinuxProfileHooks) -> ProfileHooks:
    return hooks


def primitive_platform(platform: ProfilePlatform) -> WorkerPlatform:
    return platform


def primitive_token(token: DeferredToken) -> WorkerToken:
    return token


def retained_owner(runtime: SupervisedProfile) -> ProfileOwner:
    return runtime


def create_platform(
    supervisor: Supervisor,
    token: ComputeToken,
    grant: AdmissionGrant,
    hooks: LinuxProfileHooks,
) -> ProfilePlatform:
    return ProfilePlatform(supervisor, token, grant, hooks)


def create_session(
    supervisor: Supervisor,
    grant: AdmissionGrant,
    hooks: LinuxProfileHooks,
    selected: dict[str, Any],
    reserved: ComputeToken,
) -> SupervisedProfile:
    return SupervisedProfile(
        supervisor,
        grant,
        hooks,
        selected,
        17,
        "a" * 64,
        "owner-context",
        reserved=reserved,
    )


def control_reads(
    runtime: SupervisedProfile, tab: str, job: str, context: str, revision: str
) -> SnapshotPage:
    assert_type(runtime.status(tab, job, context), WorkerStatus)
    return assert_type(
        runtime.page(tab, job, context, revision, "rows", 0, 8), SnapshotPage
    )
