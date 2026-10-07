"""Strict callers of the actual private router and supervisor grant."""

from collections.abc import Callable
from typing import Any, assert_type

from atlas_host.profile_api import (
    PrivateProfileAPI,
    ProfileCapabilities,
    ProfileService,
    production_capabilities,
)
from atlas_host.supervisor import AdmissionGrant


def capabilities() -> ProfileCapabilities:
    value = assert_type(production_capabilities(), ProfileCapabilities)
    assert_type(value["profiles_enabled"], bool)
    assert_type(value["resume_available"], bool)
    return value


def private_routes(
    service: ProfileService,
    authorize: Callable[[str, str, str], object],
    data: dict[str, Any],
    admission: AdmissionGrant,
) -> tuple[int, object, dict[str, str]]:
    router = PrivateProfileAPI(service, authorize)
    assert_type(router.service, ProfileService)
    reply = assert_type(
        router.handle("start", data, admission=admission),
        tuple[int, object, dict[str, str]],
    )
    for route in ("reconcile", "status", "page", "cancel", "heartbeat"):
        assert_type(router.handle(route, data), tuple[int, object, dict[str, str]])
    return reply
