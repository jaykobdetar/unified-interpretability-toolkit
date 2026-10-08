"""Inert session, handler and startup contracts; no listener, child or model."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
from email.message import Message
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import live_inference as live
from analytics import service


class SessionContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.clock = self.stack.enter_context(
            mock.patch.object(live.time, "monotonic", return_value=10.0)
        )
        self.memory = self.stack.enter_context(
            mock.patch.object(live, "available", return_value=8 * live.GIB)
        )
        self.session = live.Session("unused-python", self.root)

    def handler(self):
        handler = live.Handler.__new__(live.Handler)
        handler.server = SimpleNamespace(session=self.session, server_port=8796)
        handler.headers = Message()
        handler.headers["Host"] = "127.0.0.1:8796"
        handler.path = "/api/inference?fixture=1"
        handler.command = "GET"
        return handler

    def startup(self, *, analytics=False):
        server = SimpleNamespace(handle_request=mock.Mock(), server_close=mock.Mock())
        renderer = SimpleNamespace(poll=mock.Mock(side_effect=[None, 0]))
        jobs = SimpleNamespace(
            busy=False, tick=mock.Mock(), stop=mock.Mock(return_value=True)
        )
        output = io.StringIO()
        arguments = [
            "live_inference.py",
            "--model",
            str(self.root),
            "--python",
            "unused-python",
        ]
        if analytics:
            arguments.append("--analytics-only")
        with (
            mock.patch.object(sys, "argv", arguments),
            mock.patch.object(live.os, "sched_getaffinity", return_value={7, 8}),
            mock.patch.object(live.os, "sched_setaffinity") as affinity,
            mock.patch.object(
                live, "verified_layout_receipt", return_value=None
            ) as receipt,
            mock.patch.object(live, "HTTPServer", return_value=server) as listen,
            mock.patch.object(live.subprocess, "Popen", return_value=renderer) as spawn,
            mock.patch.object(service, "AnalyticsJobs", return_value=jobs) as factory,
            mock.patch.object(live.signal, "signal") as register,
            mock.patch.object(live, "cleanup_owned", return_value=True) as cleanup,
            redirect_stdout(output),
        ):
            error = None
            try:
                live.main()
            except BaseException as caught:
                error = caught
        return SimpleNamespace(
            server=server,
            renderer=renderer,
            jobs=jobs,
            output=output.getvalue(),
            affinity=affinity,
            receipt=receipt,
            listen=listen,
            spawn=spawn,
            factory=factory,
            register=register,
            cleanup=cleanup,
            error=error,
        )

    def test_contracts(self):
        contracts = live._contracts()
        value = contracts.architecture
        self.assertEqual(value.description, live.ARCH)
        self.assertEqual(value.width, live.WIDTH)
        self.assertEqual(value.layers, live.LAYERS)
        self.assertEqual(
            (value.query_heads, value.kv_heads, value.head_dim, value.vocab_size),
            (live.HEADS, live.KV_HEADS, live.HEAD_DIM, live.VOCAB),
        )
        self.assertEqual(value.capture_sites, live.CAPTURE_SITES)

    def test_init(self):
        session = self.session
        self.assertEqual((session.python, session.model), ("unused-python", self.root))
        self.assertEqual(
            (session.status, session.mode, session.buffer), ("idle", "generation", b"")
        )
        self.assertEqual(
            (
                session.started,
                session.last_seen,
                session.peak_rss,
                session.minimum_available,
            ),
            (10.0, 10.0, 0, 8 * live.GIB),
        )
        self.assertIs(session.inference_enabled, True)
        self.assertEqual(
            (list(session.steps), session.steps.maxlen, session.details), ([], 32, {})
        )
        for name in (
            "process",
            "analytics",
            "head_layout_binding",
            "head_layout_receipt",
            "stop_reason",
            "id",
            "pair_request",
            "sweep_plan",
            "observation",
            "capture_layer",
        ):
            self.assertIsNone(getattr(session, name))

    def test_snapshot(self):
        session = self.session
        session.id, session.status = "fixture-owner", "complete"
        session.steps.append({"fixture": [1, None]})
        session.details = {"fixture": "detail"}
        session.peak_rss = 2.123456 * 1024**2
        session.minimum_available = 6.123456 * live.GIB
        result = session.snapshot()
        self.assertEqual(
            result,
            {
                "session": "fixture-owner",
                "status": "complete",
                "steps": [{"fixture": [1, None]}],
                "details": {"fixture": "detail"},
                "peak_worker_rss_mib": 2.12,
                "minimum_available_gib": 6.123,
                "worker_alive": False,
            },
        )
        self.assertIs(result["details"], session.details)
        self.assertIs(result["steps"][0], session.steps[0])

    def test_metadata(self):
        session = self.session
        result = session.metadata()
        self.assertEqual(result["queue_capacity"], 0)
        self.assertEqual(result["limits"]["prompt_tokens"], 128)
        self.assertIs(result["busy"], False)
        self.assertIsNone(result["busy_owner"])
        session.status = "running"
        session.analytics = SimpleNamespace(busy=True)
        result = session.metadata()
        self.assertIs(result["busy"], True)
        self.assertEqual(result["busy_owner"], "inference session")
        session.status = "idle"
        self.assertEqual(session.metadata()["busy_owner"], "analytics job")

    def test_owns(self):
        self.session.id = "fixture-owner"
        for token, expected in (
            ("fixture-owner", True),
            ("different", False),
            (None, False),
            (7, False),
        ):
            self.assertIs(self.session.owns(token), expected)
        self.session.id = None
        self.assertIs(self.session.owns("fixture-owner"), False)

    def test_stop(self):
        session = self.session
        child = SimpleNamespace(
            stdin=SimpleNamespace(close=mock.Mock()),
            stdout=SimpleNamespace(close=mock.Mock()),
        )
        session.process, session.status, session.buffer = child, "running", b"pending"
        with mock.patch.object(
            live, "signal_and_reap", side_effect=[False, True]
        ) as reap:
            self.assertIs(session.stop(), False)
            self.assertIs(session.process, child)
            self.assertEqual(
                (session.status, session.stop_reason, session.buffer),
                ("stopping", "cancelled", b""),
            )
            self.assertIs(session.details["cleanup_pending"], True)
            child.stdout.close.assert_not_called()
            self.assertIs(session.stop(), True)
        self.assertEqual(reap.call_args_list, [mock.call(child), mock.call(child)])
        self.assertEqual(
            (session.status, session.stop_reason, session.process, session.buffer),
            ("cancelled", None, None, b""),
        )
        self.assertNotIn("cleanup_pending", session.details)
        child.stdin.close.assert_called_once_with()
        child.stdout.close.assert_called_once_with()

    def test_start(self):
        session = self.session
        child = SimpleNamespace(
            stdin=SimpleNamespace(write=mock.Mock(), close=mock.Mock()),
            stdout=SimpleNamespace(fileno=lambda: 123, close=mock.Mock()),
            poll=mock.Mock(return_value=None),
        )
        with (
            mock.patch.object(live.time, "process_time", return_value=3.0),
            mock.patch.object(live, "verify_model") as verify,
            mock.patch.object(live.subprocess, "Popen", return_value=child) as spawn,
            mock.patch.object(live.os, "set_blocking") as blocking,
            mock.patch.object(
                live.secrets, "token_hex", return_value="fixture-owner"
            ) as token,
        ):
            result = session.start({"prompt": "hello", "max_new_tokens": 2, "layer": 0})
        verify.assert_called_once_with(self.root)
        token.assert_called_once_with(16)
        self.assertEqual(
            result,
            {
                "session": "fixture-owner",
                "status": "loading",
                "steps": [],
                "details": {},
                "peak_worker_rss_mib": 0.0,
                "minimum_available_gib": 8.0,
                "worker_alive": True,
            },
        )
        self.assertEqual(
            spawn.call_args.args,
            (
                [
                    "unused-python",
                    "-B",
                    str(live.ROOT / "tools/inference_worker.py"),
                    str(self.root),
                ],
            ),
        )
        self.assertEqual(spawn.call_args.kwargs["bufsize"], 0)
        child.stdin.write.assert_called_once_with(
            b'{"prompt": "hello", "max_new_tokens": 2, "layer": 0, "activation_site": "block"}\n'
        )
        child.stdin.close.assert_called_once_with()
        blocking.assert_called_once_with(123, False)

    def test_tick(self):
        session = self.session
        session.process = SimpleNamespace(
            pid=-999,
            stdout=SimpleNamespace(fileno=lambda: 123),
            poll=mock.Mock(return_value=None),
        )
        session.analytics = SimpleNamespace(tick=mock.Mock())
        session.status = "loading"
        self.memory.return_value = 6 * live.GIB
        with (
            mock.patch.object(Path, "read_text", return_value="VmRSS:\t2048 kB\n"),
            mock.patch.object(
                live.os,
                "read",
                side_effect=[b'{"type":"loaded","seconds":1}\n', BlockingIOError],
            ),
        ):
            session.tick()
        self.assertEqual(
            (session.peak_rss, session.minimum_available, session.details),
            (2 * 1024**2, 6 * live.GIB, {"seconds": 1}),
        )
        session.analytics.tick.assert_called_once_with()
        session.buffer = b"x" * 131072
        with (
            mock.patch.object(Path, "read_text", side_effect=OSError),
            mock.patch.object(live.os, "read", return_value=b"x"),
            mock.patch.object(session, "stop") as stop,
        ):
            session.tick()
        self.assertEqual(session.details["error"], "Worker output exceeded bound")
        stop.assert_called_once_with("error")

    def test_setup(self):
        handler = self.handler()
        handler.connection = SimpleNamespace(settimeout=mock.Mock())
        with mock.patch.object(live.BaseHTTPRequestHandler, "setup") as setup:
            handler.setup()
        setup.assert_called_once_with()
        handler.connection.settimeout.assert_called_once_with(0.5)

    def test_handle(self):
        handler = self.handler()
        wire = b"GET /api/inference HTTP/1.0\r\nHost: 127.0.0.1:8796\r\n\r\n"
        handler.connection = SimpleNamespace(
            settimeout=mock.Mock(), recv=mock.Mock(return_value=wire)
        )
        received = []
        with mock.patch.object(
            live.BaseHTTPRequestHandler,
            "handle",
            autospec=True,
            side_effect=lambda target: received.append(target.rfile.read()),
        ) as handle:
            handler.handle()
        handle.assert_called_once_with(handler)
        self.assertEqual(received, [wire])
        handler.connection.recv.assert_called_once_with(4096)
        self.assertEqual(
            handler.connection.settimeout.call_args_list,
            [mock.call(0.5), mock.call(0.5)],
        )

    def test_log_message(self):
        output, error = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            result = self.handler().log_message("fixture %s", "message")
        self.assertIsNone(result)
        self.assertEqual((output.getvalue(), error.getvalue()), ("", ""))

    def test_send(self):
        handler = self.handler()
        handler.send_response, handler.send_header, handler.end_headers = (
            mock.Mock(),
            mock.Mock(),
            mock.Mock(),
        )
        handler.wfile = io.BytesIO()
        expected = b'{"fixture": [1, null]}'
        self.assertIsNone(handler.send(207, {"fixture": [1, None]}))
        handler.send_response.assert_called_once_with(207)
        self.assertEqual(handler.wfile.getvalue(), expected)
        self.assertIn(
            mock.call("Content-Type", "application/json"),
            handler.send_header.call_args_list,
        )
        self.assertIn(
            mock.call("Content-Length", str(len(expected))),
            handler.send_header.call_args_list,
        )
        self.assertIn(
            mock.call("Cache-Control", "no-store"), handler.send_header.call_args_list
        )
        handler.end_headers.assert_called_once_with()

    def test_validated(self):
        handler = self.handler()
        self.assertEqual(handler.validated(), ("/api/inference", 0))
        handler.headers["Content-Length"] = "3"
        self.assertEqual(handler.validated(), ("/api/inference", 3))

    def test_handle_action(self):
        handler = self.handler()
        session = SimpleNamespace(
            inference_enabled=True,
            tick=mock.Mock(),
            metadata=mock.Mock(return_value={"fixture": 1}),
            start=mock.Mock(return_value={"session": "fixture-owner"}),
            owns=mock.Mock(return_value=False),
        )
        handler.server.session = session
        handler.validated = mock.Mock(return_value=("/api/inference", 0))
        handler.send = mock.Mock(return_value="sent")
        self.assertEqual(handler.handle_action(), "sent")
        handler.send.assert_called_once_with(200, {"fixture": 1})
        session.tick.assert_called_once_with()
        handler.command = "POST"
        handler.headers["X-Atlas-Local"] = "1"
        handler.rfile = io.BytesIO(b'{"prompt":"hello"}')
        handler.validated.return_value = ("/api/inference/start", 18)
        handler.send.reset_mock()
        self.assertEqual(handler.handle_action(), "sent")
        handler.send.assert_called_once_with(202, {"session": "fixture-owner"})
        session.start.assert_called_once_with({"prompt": "hello"})
        handler.rfile = io.BytesIO(b'{"session":"other"}')
        handler.validated.return_value = ("/api/inference/poll", 19)
        handler.send.reset_mock()
        self.assertEqual(handler.handle_action(), "sent")
        handler.send.assert_called_once_with(409, {"error": "Stale session"})

    def test_bind_source(self):
        session = self.session
        source = {
            "source_directory": str(self.root),
            "revision": live.MANIFEST["revision"],
            "inference_source_model": "stale",
        }
        original = dict(source)
        marker = {"fixture": "binding"}
        with (
            mock.patch.object(live, "current_layout_binding", return_value=marker),
            mock.patch.object(
                live, "bind_viewer_head_layout", side_effect=lambda data, binding: data
            ) as bind,
        ):
            result = live.bind_inference_source(source, session)
            self.assertEqual(result["inference_source_model"], live.SOURCE_MODEL)
            self.assertIsNot(result, source)
            self.assertEqual(source, original)
            self.assertIs(bind.call_args.args[1], marker)
            session.inference_enabled = False
            result = live.bind_inference_source(source, session)
            self.assertNotIn("inference_source_model", result)
            self.assertEqual(source, original)

    def test_cleanup(self):
        session = SimpleNamespace(
            stop=mock.Mock(return_value=False),
            analytics=SimpleNamespace(stop=mock.Mock(return_value=True)),
        )
        renderer, server = object(), SimpleNamespace(server_close=mock.Mock())
        with (
            mock.patch.object(
                live, "signal_and_reap", side_effect=[False, True]
            ) as reap,
            mock.patch.object(live.time, "sleep") as sleep,
        ):
            self.assertIs(live.cleanup_owned(session, renderer, server), False)
        self.assertEqual(session.stop.call_count, 8)
        self.assertEqual(session.analytics.stop.call_count, 8)
        self.assertEqual(sleep.call_args_list, [mock.call(0.05)] * 8)
        self.assertEqual(
            reap.call_args_list,
            [
                mock.call(renderer, terminate=True, timeout=0.5),
                mock.call(renderer, timeout=1.5),
            ],
        )
        server.server_close.assert_called_once_with()

    def test_main(self):
        run = self.startup()
        self.assertIsNone(run.error)
        run.affinity.assert_called_once_with(0, {7})
        run.receipt.assert_called_once_with(self.root, required=True)
        run.listen.assert_called_once_with(("127.0.0.1", 8796), live.Handler)
        self.assertEqual((run.server.timeout, run.server.atlas_port), (0.1, 8797))
        self.assertIs(run.server.session.analytics, run.jobs)
        run.server.handle_request.assert_called_once_with()
        run.jobs.tick.assert_called_once_with()
        self.assertEqual(
            run.spawn.call_args.args,
            (
                [
                    str(live.ROOT / "target/release/weight-atlas-rust"),
                    "serve",
                    "--model",
                    str(self.root.resolve()),
                    "--cache",
                    str(live.ROOT / "cache-inference"),
                    "--name",
                    "SmolLM2-135M · fixed weights",
                    "--revision",
                    live.MANIFEST["revision"],
                    "--port",
                    "8797",
                ],
            ),
        )
        run.cleanup.assert_called_once_with(
            run.server.session, run.renderer, run.server
        )
        self.assertEqual(
            run.output,
            "Experimental live inference and bounded analytics: http://127.0.0.1:8796\n",
        )

    def test_analysis_model(self):
        run = self.startup()
        self.assertIsNone(run.error)
        fetch = run.factory.call_args.kwargs["fetch_model"]
        with mock.patch.object(
            live,
            "proxy_request",
            return_value=(200, b'{"fixture":7}', "application/json"),
        ) as proxy:
            self.assertEqual(fetch(), {"fixture": 7})
        proxy.assert_called_once_with(8797, "GET", "/api/model", run.server.session)
        with mock.patch.object(
            live, "proxy_request", return_value=(503, b"{}", "application/json")
        ):
            with self.assertRaises(ValueError) as caught:
                fetch()
        self.assertEqual(
            str(caught.exception),
            "Validated model metadata unavailable; retry after renderer startup",
        )

    def test_shutdown(self):
        run = self.startup()
        self.assertIsNone(run.error)
        self.assertEqual(len(run.register.call_args_list), 2)
        callback = run.register.call_args_list[0].args[1]
        self.assertIs(run.register.call_args_list[1].args[1], callback)
        error = None
        try:
            callback(live.signal.SIGTERM, None)
        except BaseException as caught:
            error = caught
        self.assertIs(type(error), KeyboardInterrupt)
        self.assertEqual(error.args, ())


if __name__ == "__main__":
    unittest.main()
