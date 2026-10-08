"""Actual standard-library responses and injected memory streams satisfy acquisition."""

from http.client import HTTPResponse
from io import BytesIO
from pathlib import Path
from typing import Any, ContextManager, assert_type
from atlas_host.acquisition import (
    AcquisitionPlan,
    AcquisitionStream,
    Fetch,
    acquire,
    open_public_data,
    plan,
)
from atlas_host.registry import Registry


def response_client(
    response: HTTPResponse, memory: BytesIO
) -> tuple[AcquisitionStream, AcquisitionStream]:
    real: AcquisitionStream = response
    synthetic: AcquisitionStream = memory
    return real, synthetic


def provider_client(url: str, timeout: float) -> ContextManager[AcquisitionStream]:
    provider: Fetch = open_public_data
    return provider(url, timeout)


def acquisition_client(
    registry: Registry, destination: Path, manifest: dict[str, Any], fetch: Fetch
) -> dict[str, Any]:
    reviewed = plan(manifest, "Owner data", max_bytes=100)
    assert_type(reviewed, AcquisitionPlan)
    result = acquire(
        registry,
        destination,
        manifest,
        "Owner data",
        max_bytes=100,
        plan_digest=reviewed["plan_digest"],
        accept_license=True,
        fetch=fetch,
    )
    assert_type(result, dict[str, Any])
    return result
