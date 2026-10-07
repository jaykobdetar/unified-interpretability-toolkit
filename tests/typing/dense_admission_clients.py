"""Actual registry, prepared-source and frozen-binary admission clients."""

from pathlib import Path
from typing import Any, assert_type

from atlas_host.dense_static_admission import (
    BoundDenseStaticAdmission,
    PlatformObservation,
    RootIdentity,
    _platform,
    _root_identity,
    bind_dense_policy,
    check_native_model,
    validate_support,
)
from atlas_host.profile_os import FrozenBinary
from atlas_host.registry import Registry, RegistryEntry
from atlas_host.static_models import PreparedStatic


def current_admission(
    bound: BoundDenseStaticAdmission,
    prepared: PreparedStatic,
    binary: FrozenBinary,
    model: dict[str, Any],
) -> RegistryEntry:
    assert_type(bound.allows(prepared.entry["model_id"]), bool)
    assert_type(bound.admit(prepared), bool)
    assert_type(bound.check_binary(binary), None)
    assert_type(check_native_model(prepared, model), None)
    return bound.check()


def bind_owner(
    path: Path, digest: str, registry: Registry, cache: Path, binary: FrozenBinary
) -> BoundDenseStaticAdmission:
    return bind_dense_policy(path, digest, registry, cache, binary.path)


def supporting_identity(value: dict[str, Any]) -> dict[str, Any]:
    return validate_support(value, "sourceRecipe")


def observations(path: Path) -> tuple[RootIdentity, PlatformObservation]:
    return _root_identity(path), _platform()
