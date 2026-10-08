"""Strict policy clients use actual registry, binary and resource records."""

from pathlib import Path
from typing import assert_type

from atlas_host.profile_os import BinaryPolicy, FrozenBinary, ResourceSnapshot
from atlas_host.registry import RegistryEntry
from atlas_host.validation_policy import (
    BoundValidationPolicy,
    FileIdentity,
    QualificationBoundary,
    ValidationBinary,
    bind_reviewed_recipe,
    check_memory,
    check_start,
)


def actual_binary(binary: FrozenBinary) -> ValidationBinary:
    return binary


def actual_policy(policy: BoundValidationPolicy) -> BinaryPolicy:
    return policy


def read_policy(
    receipt: Path, approved_sha: str, monitor: Path, harness: Path, binary: Path
) -> BoundValidationPolicy:
    return bind_reviewed_recipe(
        receipt,
        approved_sha,
        monitor_path=monitor,
        harness_path=harness,
        binary_path=binary,
    )


def policy_operations(
    policy: BoundValidationPolicy,
    binary: FrozenBinary,
    entry: RegistryEntry,
    root: Path,
    observed: ResourceSnapshot,
) -> list[str]:
    assert_type(policy.source_inventory, tuple[str, ...])
    assert_type(policy.files, tuple[FileIdentity, ...])
    assert_type(policy.start_bytes, int)
    policy.check_binary(binary)
    policy.check_entry(entry)
    check_start(policy, 5 * 1024**3, 25 * 1024**3)
    check_memory(observed, 0)
    boundary = QualificationBoundary(policy)
    boundary.preflight(5 * 1024**3, 25 * 1024**3)
    boundary.check_sample(observed)
    return assert_type(policy.command("/usr/bin/python3", root), list[str])
