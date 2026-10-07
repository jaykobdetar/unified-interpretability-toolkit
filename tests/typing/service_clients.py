"""Strict service clients connect existing routing, OS and worker interfaces."""

from typing import assert_type

from atlas_host.profile_api import ProfileService as RouterService
from atlas_host.profile_os import (
    FrozenBinary,
    LinuxProfileHooks,
    OwnedProcess,
    ProcessBook,
    ProfileSource,
    ProfileWatchdog,
    ThreadMeter,
)
from atlas_host.profile_platform import ProfileHooks, SupervisedProfile
from atlas_host.profile_service import (
    HooksFactory,
    ProfileService,
    RuntimeFactory,
    ServiceContext,
    ServiceMeter,
    ServiceSnapshot,
    StartedSnapshot,
    WaitableChild,
    WatchdogLoop,
    WatchSlot,
)
from atlas_host.startup_diagnostics import DiagnosticCollector
from atlas_host.supervisor import AdmissionGrant, Supervisor


def router_service(service: ProfileService) -> RouterService:
    return service


def slot_registration(slot: WatchSlot) -> ProfileWatchdog:
    return slot


def actual_meter(meter: ThreadMeter) -> ServiceMeter:
    return meter


def actual_waitable(child: OwnedProcess) -> WaitableChild:
    return child


def actual_runtime_factory() -> RuntimeFactory:
    return SupervisedProfile


def diagnostic_meter(diagnostics: DiagnosticCollector) -> ThreadMeter:
    return ThreadMeter(diagnostics=diagnostics)


def actual_hooks_factory(book: ProcessBook, binary: FrozenBinary) -> HooksFactory:
    def build(source: ProfileSource, watch: WatchSlot) -> ProfileHooks:
        return LinuxProfileHooks(book, binary, source, watch)

    return build


def owner_service(
    supervisor: Supervisor,
    context: ServiceContext,
    hooks: HooksFactory,
    meter: ThreadMeter,
    watch: WatchdogLoop,
    diagnostics: DiagnosticCollector,
) -> ProfileService:
    service = ProfileService(
        supervisor,
        context,
        hooks,
        owner_meter=meter,
        watchdog=watch,
        diagnostics=diagnostics,
    )
    assert_type(service.admission(), tuple[AdmissionGrant, ThreadMeter])
    return service


def snapshot_identity(snapshot: ServiceSnapshot, started: StartedSnapshot) -> str:
    assert_type(snapshot["job_id"], str)
    assert_type(snapshot["state"], str)
    return assert_type(started["job_capability"], str)
