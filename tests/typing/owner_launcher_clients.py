"""Actual owner launcher signatures and shared structural server view."""

from pathlib import Path
from typing import Any, assert_type
import profile_atlas
import static_atlas
from atlas_host.profile_http import HostedServer
from host_atlas import FixtureServer


def server_client(server: HostedServer) -> FixtureServer:
    fixture: FixtureServer = server
    assert_type(server.profile_controls, bool)
    assert_type(server.application.close(), bool)
    return fixture


def launcher_client(config: dict[str, Any], binary: Path, digest: str) -> None:
    assert_type(profile_atlas.run(config, binary, digest), None)
    assert_type(static_atlas.run(config, binary, digest), None)
    assert_type(
        profile_atlas.main(["--config", "owner.json", "--binary-sha256", digest]), None
    )
    assert_type(
        static_atlas.main(["--config", "owner.json", "--binary-sha256", digest]), None
    )
