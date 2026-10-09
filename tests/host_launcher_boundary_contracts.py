"""Fixture launcher calls and response receipts use inert process/server doubles."""

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import unittest

import host_atlas as module
from page_startup_fixture import expected_bundle


class HostLauncherBoundaryTests(unittest.TestCase):
    def ok(self, function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except BaseException as error:
            self.fail(
                f"Valid inert launcher call raised {type(error).__name__}: {error}"
            )

    def renderer(self):
        renderer = object.__new__(module.Renderer)
        renderer.port = 8123
        renderer.session = object()
        renderer.raw = bytearray()
        renderer.announced = False
        renderer.stopping = False
        renderer.process = Mock()
        renderer.process.poll.return_value = None
        renderer.process.stdout.fileno.return_value = 73
        return renderer

    def test_constructor_exact_argv_and_pipe_configuration(self):
        entry = {
            "root": "/inert/source",
            "name": "inert fixture",
            "manifest": {"revision": "inert-revision"},
        }
        session = object()
        with (
            patch.object(module, "available", return_value=100 * module.GIB),
            patch.object(module.Path, "is_file", return_value=True),
            patch.object(module.subprocess, "Popen") as spawn,
        ):
            renderer = self.ok(
                module.Renderer, entry, Path("/inert/cache"), 8123, session
            )
        spawn.assert_called_once_with(
            [
                str(module.ROOT / "target/release/weight-atlas-rust"),
                "serve",
                "--model",
                "/inert/source",
                "--cache",
                "/inert/cache",
                "--name",
                "inert fixture",
                "--revision",
                "inert-revision",
                "--port",
                "8123",
            ],
            stdout=module.subprocess.PIPE,
            stderr=module.subprocess.DEVNULL,
        )
        self.assertIs(renderer.process, spawn.return_value)
        self.assertIs(renderer.session, session)
        self.assertIs(renderer.announced, False)
        self.assertIs(renderer.stopping, False)

    def test_initialize_single_nonblocking_call_and_none_result(self):
        renderer = self.renderer()
        with patch.object(module.os, "set_blocking") as blocking:
            self.assertIsNone(self.ok(renderer.initialize))
        blocking.assert_called_once_with(73, False)

    def test_alive_exact_boolean_and_single_poll(self):
        renderer = self.renderer()
        self.assertIs(self.ok(renderer.alive), True)
        renderer.process.poll.assert_called_once_with()

    def test_ready_exact_read_budget_boolean_and_single_alive_check(self):
        renderer = self.renderer()
        with (
            patch.object(
                module.os,
                "read",
                return_value=b'{"listening":"http://127.0.0.1:8123"}\n',
            ) as read,
            patch.object(renderer, "alive", return_value=True) as alive,
        ):
            self.assertIs(self.ok(renderer.ready), True)
        read.assert_called_once_with(73, 8193)
        alive.assert_called_once_with()
        self.assertIs(renderer.announced, True)

    def test_read_preserves_proxy_arguments_and_response_identity(self):
        renderer = self.renderer()
        response = (200, b"inert", "application/json")
        with (
            patch.object(renderer, "ready", return_value=True),
            patch.object(module, "proxy_request", return_value=response) as proxy,
        ):
            self.assertIs(self.ok(renderer.read, "/api/model"), response)
        proxy.assert_called_once_with(8123, "GET", "/api/model", renderer.session)

    def test_stop_exact_deadline_boolean_and_single_pipe_close(self):
        renderer = self.renderer()
        with patch.object(module, "signal_and_reap", return_value=True) as reap:
            self.assertIs(self.ok(renderer.stop), True)
        reap.assert_called_once_with(renderer.process, terminate=True, timeout=0.2)
        self.assertIs(renderer.stopping, True)
        renderer.process.stdout.close.assert_called_once_with()

    def handler(self, path):
        handler = object.__new__(module.HostHandler)
        handler.command = "GET"
        handler.path = path
        handler.headers = {}
        handler.server = NS(host=object())
        handler.validated = Mock(return_value=(path, 0))
        handler.send = Mock(return_value=None)
        return handler

    def test_dispatch_host_preserves_arguments_and_response_identity(self):
        handler = self.handler("/api/models")
        data = {"inert": True}
        response = (200, b"inert", "application/json")
        with patch.object(module, "dispatch", return_value=response) as dispatch:
            self.assertIs(
                self.ok(handler.dispatch_host, "GET", "/api/models", data), response
            )
        dispatch.assert_called_once_with(
            handler.server.host, "GET", "/api/models", data
        )

    def test_page_response_keeps_exact_file_bytes_and_mime(self):
        handler = self.handler("/")
        self.assertIsNone(self.ok(handler.handle_action))
        handler.send.assert_called_once_with(
            200,
            (module.ROOT / "web/index.html").read_bytes(),
            "text/html; charset=utf-8",
        )

    def test_bundle_response_keeps_original_separator(self):
        handler = self.handler("/viewer.js")
        self.assertIsNone(self.ok(handler.handle_action))
        expected = expected_bundle(module.ROOT, module.BUNDLE)
        handler.send.assert_called_once_with(200, expected, "text/javascript")

    def test_disabled_inference_response_keeps_exact_status_and_receipt(self):
        handler = self.handler("/api/inference")
        self.assertIsNone(self.ok(handler.handle_action))
        handler.send.assert_called_once_with(
            503,
            {
                "api_version": 1,
                "code": "disabled",
                "error": "Inference and analytics are disabled in fixture host mode",
            },
        )

    def main_call(self):
        config = {
            "paths": {"registry": "/inert/registry", "cache": "/inert/cache"},
            "ports": {"renderer": 8123, "coordinator": 8124},
            "bind": "127.0.0.1",
        }
        host = Mock()
        host.close.return_value = True
        server = NS(
            server_port=8124,
            handle_request=Mock(side_effect=KeyboardInterrupt),
            server_close=Mock(),
        )
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(module, "load_config", return_value=config)
            )
            stack.enter_context(
                patch.object(module, "available", return_value=100 * module.GIB)
            )
            stack.enter_context(
                patch.object(module.os, "sched_getaffinity", return_value={0})
            )
            stack.enter_context(patch.object(module.os, "sched_setaffinity"))
            stack.enter_context(patch.object(module, "Registry"))
            stack.enter_context(patch.object(module, "FixtureHost", return_value=host))
            create = stack.enter_context(
                patch.object(module, "HTTPServer", return_value=server)
            )
            signals = stack.enter_context(patch.object(module.signal, "signal"))
            output = stack.enter_context(patch("builtins.print"))
            self.assertIsNone(self.ok(module.main, ["--config", "/inert/config.json"]))
        create.assert_called_once_with(("127.0.0.1", 8124), module.HostHandler)
        server.server_close.assert_called_once_with()
        host.close.assert_called_once_with()
        return server, host, signals, output

    def test_main_exact_timeout_signal_order_and_announcement(self):
        server, host, signals, output = self.main_call()
        self.assertEqual(server.timeout, 0.1)
        self.assertIs(server.host, host)
        self.assertEqual(
            [entry.args[0] for entry in signals.call_args_list],
            [module.signal.SIGTERM, module.signal.SIGINT],
        )
        output.assert_called_once_with(
            "Fixture host: http://127.0.0.1:8124", flush=True
        )

    def test_shutdown_callback_keeps_exact_interrupt_type_and_empty_args(self):
        _, _, signals, _ = self.main_call()
        callback = signals.call_args_list[0].args[1]
        try:
            callback(module.signal.SIGTERM, None)
        except BaseException as error:
            self.assertIs(type(error), KeyboardInterrupt)
            self.assertEqual(error.args, ())
        else:
            self.fail("Shutdown callback must raise the existing interrupt")


if __name__ == "__main__":
    unittest.main()
