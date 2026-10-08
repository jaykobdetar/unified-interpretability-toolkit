"""Actual profile handler and hosted application satisfy their declared boundary."""

from typing import assert_type
from atlas_host.hosted_runtime import HostedApplication
from atlas_host.profile_http import HostedHandler, HostedServer
from atlas_host.runtime_adapter import FixtureHost, NativeResponse
from host_atlas import HostHandler


def handler_client(handler: HostedHandler) -> HostHandler:
    parent: HostHandler = handler
    assert_type(handler.handle(), None)
    assert_type(handler.handle_action(), None)
    assert_type(handler.do_GET(), None)
    assert_type(handler.do_POST(), None)
    assert_type(handler.send(200, {"version": 1}), None)
    assert_type(handler.dispatch_host("GET", "/api/models"), NativeResponse)
    return parent


def server_client(server: HostedServer, application: HostedApplication) -> FixtureHost:
    server.application = application
    server.host = application.host
    assert_type(server.application, HostedApplication)
    assert_type(server.host, FixtureHost)
    return server.host
