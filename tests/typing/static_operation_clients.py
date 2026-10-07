"""Static-operation contracts use the real platform, grant and service objects."""

from typing import Callable, assert_type

from atlas_host.profile_os import FrozenBinary, OwnedProcess, ProcessBook
from atlas_host.profile_service import ProfileService, WatchSlot
from atlas_host.static_models import PreparedStatic
from atlas_host.static_operation import (
    OperationApp,
    OperationChannel,
    OperationReader,
    StaticOperation,
)
from atlas_host.supervisor import AdmissionGrant, Supervisor


def operation_client(
    app: OperationApp,
    grant: AdmissionGrant,
    child: OwnedProcess,
    reader: OperationReader,
    prepared: PreparedStatic,
    write: Callable[[], object],
) -> StaticOperation:
    operation = StaticOperation(app, grant, "metadata")
    assert_type(app.supervisor, Supervisor)
    assert_type(app.profiles, ProfileService)
    assert_type(app.book, ProcessBook)
    assert_type(app.binary, FrozenBinary)
    assert_type(operation.grant, AdmissionGrant)
    assert_type(operation.slot, WatchSlot | None)
    assert_type(reader.channel, OperationChannel)
    operation.prepared = prepared
    operation.context = "selected"
    assert_type(operation.add_child(child, 0.25), None)
    assert_type(operation.add_reader(reader), None)
    assert_type(operation._observe(child), None)
    assert_type(operation.check(), None)
    assert_type(operation.check_publication(), None)
    assert_type(operation.pulse(), None)
    assert_type(operation._fail(), None)
    assert_type(operation._reap(), bool)
    assert_type(operation.abort(), bool)
    assert_type(operation.publish(write), None)
    return operation
