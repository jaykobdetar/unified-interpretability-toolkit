"""Strict lifetime clients use the existing resource and supervisor implementations."""

from collections.abc import Callable
from typing import assert_type

from atlas_host.lifetime_guard import (
    LifetimeGuard,
    LifetimeResources,
    LifetimeSupervisor,
)
from atlas_host.profile_os import ProcessBook
from atlas_host.startup_diagnostics import StartupDiagnostics
from atlas_host.supervisor import Supervisor


def actual_resources(book: ProcessBook) -> LifetimeResources:
    return book


def actual_supervisor(supervisor: Supervisor) -> LifetimeSupervisor:
    return supervisor


def lifetime_owner(
    book: ProcessBook,
    supervisor: Supervisor,
    cleanup: Callable[[], None],
    diagnostics: StartupDiagnostics,
) -> LifetimeGuard:
    guard = LifetimeGuard(book, supervisor, cleanup, diagnostics=diagnostics)
    assert_type(guard.pulse(), bool)
    assert_type(guard.request_shutdown(), None)
    assert_type(guard.failed, bool)
    assert_type(guard.stopping, bool)
    return guard
