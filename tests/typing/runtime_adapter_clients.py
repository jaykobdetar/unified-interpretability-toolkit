"""The adapter uses real registry/policy contracts and separate reader call shapes."""

from pathlib import Path
from typing import assert_type

from atlas_host.registry import Registry
from atlas_host.runtime_adapter import (
    AcquisitionReceipt,
    FixtureFactory,
    FixtureHost,
    HeartbeatReceipt,
    HostError,
    NativeResponse,
    ReleaseReceipt,
    dispatch,
)
from atlas_host.static_operation import OperationHost, StaticOperation


def fixture_client(
    registry: Registry, cache: Path, factory: FixtureFactory
) -> OperationHost:
    host = FixtureHost(registry, cache, factory)
    owner: OperationHost = host
    assert_type(host.static_views_enabled, bool)
    assert_type(host._published_ready(), bool)
    assert_type(host._owns("context", "capability"), bool)
    assert_type(host.acquire({"model_id": "synthetic-fixture"}), AcquisitionReceipt)
    assert_type(
        host.heartbeat("context", {"capability": "capability"}), HeartbeatReceipt
    )
    assert_type(host.release("context", {"capability": "capability"}), ReleaseReceipt)
    assert_type(host.read("model", "model", {"context": ["context"]}), NativeResponse)
    assert_type(dispatch(host, "GET", "/api/models"), NativeResponse)
    assert_type(host.close(), bool)
    return owner


def static_client(host: FixtureHost, operation: StaticOperation) -> NativeResponse:
    assert_type(
        host.acquire({"model_id": "registered-static"}, operation=operation),
        AcquisitionReceipt,
    )
    return host.read("model", "model", {"context": ["context"]}, operation=operation)


def error_client(error: HostError) -> tuple[int, str, str]:
    return error.status, error.code, str(error)
