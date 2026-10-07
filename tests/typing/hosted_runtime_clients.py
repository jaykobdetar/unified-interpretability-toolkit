"""Actual hosted implementations satisfy the existing ownership interfaces."""

from typing import assert_type
from atlas_host.hosted_runtime import (
    HostedApplication,
    HostedContexts,
    HostedRenderer,
    NativeChannel,
)
from atlas_host.dense_static_admission import BoundDenseStaticAdmission
from atlas_host.profile_service import ServiceContext
from atlas_host.runtime_adapter import DenseOperationApp, NativeResponse, StaticRenderer
from atlas_host.static_operation import (
    OperationApp,
    OperationChannel,
    OperationOwner,
    StaticOperation,
)


def application_client(app: HostedApplication) -> OperationApp:
    operation: OperationApp = app
    dense: DenseOperationApp = app
    owner: OperationOwner = app
    assert_type(app.close(), bool)
    assert_type(dense.dense_policy, BoundDenseStaticAdmission | None)
    assert_type(owner.static_operations, list[StaticOperation])
    return operation


def renderer_client(renderer: HostedRenderer, channel: NativeChannel) -> StaticRenderer:
    concrete: StaticRenderer = renderer
    native: OperationChannel = channel
    assert_type(concrete.read("/api/model"), NativeResponse)
    assert_type(native.failed, bool)
    return concrete


def context_client(context: HostedContexts) -> ServiceContext:
    concrete: ServiceContext = context
    return concrete
