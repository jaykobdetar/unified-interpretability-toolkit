"""The concrete fixture renderer satisfies the adapter and coordinator contracts."""

from pathlib import Path
from typing import assert_type

from atlas_host.live_inference import TickOwner
from atlas_host.registry import RegistryEntry
from atlas_host.runtime_adapter import FixtureHost, NativeResponse, OwnedRenderer
from host_atlas import HostHandler, Renderer


def renderer_client(
    entry: RegistryEntry, cache: Path, host: FixtureHost
) -> OwnedRenderer:
    owner: TickOwner = host
    renderer = Renderer(entry, cache, 8123, owner)
    concrete: OwnedRenderer = renderer
    assert_type(renderer.initialize(), None)
    assert_type(renderer.alive(), bool)
    assert_type(renderer.ready(), bool)
    assert_type(renderer.read("/api/model"), NativeResponse)
    assert_type(renderer.stop(), bool)
    return concrete


def handler_client(handler: HostHandler) -> NativeResponse:
    assert_type(handler.handle_action(), None)
    return handler.dispatch_host("GET", "/api/models")
