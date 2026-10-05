#!/usr/bin/env python3
"""Patched helper/control tests only: fake I/O, clocks, and children; no sockets/model."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from email.message import Message

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "live_reliability", ROOT / "tools/live_inference.py"
)
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FakeSocket:
    def __init__(self, wire, clock, timeouts=0, shared=None):
        self.shared = shared or SimpleNamespace(
            wire=bytearray(wire),
            clock=clock,
            timeouts=timeouts,
            sockets=[],
            sent=bytearray(),
        )
        self.shared.sockets.append(self)
        self.closed = False

    def dup(self):
        return FakeSocket(b"", self.shared.clock, shared=self.shared)

    def settimeout(self, value):
        assert 0 < value <= live.IO_TICK

    def send(self, data):
        self.shared.sent.extend(data)
        return len(data)

    def recv_into(self, buffer):
        assert not self.closed
        state = self.shared
        if state.timeouts:
            state.timeouts -= 1
            state.clock.now += 0.1
            raise TimeoutError("Injected short socket wait")
        count = min(len(buffer), len(state.wire), 17)
        buffer[:count] = state.wire[:count]
        del state.wire[:count]
        return count

    def close(self):
        self.closed = True


class ProxyContracts(unittest.TestCase):
    def setUp(self):
        self.mem = patch.object(live, "available", return_value=6 * live.GIB)
        self.mem.start()
        self.addCleanup(self.mem.stop)

    def run_wire(self, status=200, body=b"hello", timeouts=0):
        clock, session = Clock(), SimpleNamespace(tick=Mock())
        wire = (
            f"HTTP/1.1 {status} Result\r\nContent-Length: {len(body)}\r\n"
            "Content-Type: application/json\r\nConnection: close\r\n\r\n"
        ).encode() + body
        fake = FakeSocket(wire, clock, timeouts)
        original = live.http.client.HTTPConnection

        def connection(*args, **kwargs):
            conn = original(*args, **kwargs)
            conn.connect = lambda: setattr(conn, "sock", fake)
            return conn

        with (
            patch.object(live.time, "monotonic", clock),
            patch.object(live.http.client, "HTTPConnection", side_effect=connection),
        ):
            try:
                result = live.proxy_request(8797, "GET", "/api/model", session)
            finally:
                self.assertTrue(all(s.closed for s in fake.shared.sockets))
        return result, session, clock

    def test_valid_response_can_exceed_receive_budget_while_servicing_session(self):
        result, session, clock = self.run_wire(timeouts=8)
        self.assertEqual(result, (200, b"hello", "application/json"))
        self.assertGreater(clock.now, live.REQUEST_DEADLINE)
        self.assertGreaterEqual(session.tick.call_count, 4)

    def test_upstream_status_and_body_are_preserved(self):
        result, _, _ = self.run_wire(503, b'{"error":"statistics not ready"}')
        self.assertEqual(result[:2], (503, b'{"error":"statistics not ready"}'))

    def test_total_operation_deadline_is_504(self):
        with self.assertRaises(live.BackendError) as caught:
            self.run_wire(timeouts=100)
        self.assertEqual(caught.exception.status, 504)
        self.assertEqual(caught.exception.code, "backend_timeout")

    def test_connection_failure_is_503_and_connection_closes(self):
        conn = Mock()
        conn.connect.side_effect = ConnectionRefusedError(
            "Injected unavailable renderer"
        )
        with (
            patch.object(live.http.client, "HTTPConnection", return_value=conn),
            self.assertRaises(live.BackendError) as caught,
        ):
            live.proxy_request(8797, "GET", "/api/model", SimpleNamespace(tick=Mock()))
        self.assertEqual(caught.exception.status, 503)
        self.assertEqual(caught.exception.code, "backend_unavailable")
        conn.close.assert_called_once()

    def test_response_bound_closes_response_and_is_not_client_error(self):
        conn, response = Mock(), Mock()
        response.read.return_value = b"x" * (2 * 1024**2 + 1)
        conn.getresponse.return_value = response
        with (
            patch.object(live.http.client, "HTTPConnection", return_value=conn),
            self.assertRaises(live.BackendError) as caught,
        ):
            live.proxy_request(8797, "GET", "/api/model", SimpleNamespace(tick=Mock()))
        self.assertEqual(caught.exception.status, 502)
        response.read.assert_called_once_with(2 * 1024**2 + 1)
        response.close.assert_called_once()
        conn.close.assert_called_once()

    def test_resource_guard_services_session_before_stopping_operation(self):
        session = SimpleNamespace(tick=Mock())
        deadline = live.OperationDeadline(session)
        with (
            patch.object(live, "available", return_value=3 * live.GIB),
            self.assertRaises(live.BackendError) as caught,
        ):
            deadline.remaining()
        self.assertEqual(caught.exception.code, "resource_limit")
        session.tick.assert_called_once()


class Child:
    def __init__(self):
        self.pid, self.returncode = 999999999, None
        self.stdin, self.stdout = io.BytesIO(), io.BytesIO()
        self.pending = True
        self.kills = self.terminates = 0
        self.signal_error = None

    def poll(self):
        return self.returncode

    def kill(self):
        self.kills += 1
        if self.signal_error:
            raise self.signal_error

    def terminate(self):
        self.terminates += 1

    def wait(self, timeout):
        assert 0 < timeout <= 1.5
        if self.pending:
            raise subprocess.TimeoutExpired("owned-fixture-child", timeout)
        self.returncode = 0
        return 0


class CleanupContracts(unittest.TestCase):
    def setUp(self):
        self.mem = patch.object(live, "available", return_value=6 * live.GIB)
        self.mem.start()
        self.addCleanup(self.mem.stop)
        self.session = live.Session("unused-python", Path("/unused"))
        self.child = Child()
        self.session.process, self.session.status, self.session.id = (
            self.child,
            "running",
            "owner-fixture",
        )

    def handler(self, action):
        handler = live.Handler.__new__(live.Handler)
        handler.server = SimpleNamespace(session=self.session, server_port=8796)
        handler.command, handler.path = "POST", "/api/inference/" + action
        body = json.dumps({"session": self.session.id}).encode()
        handler.rfile, handler.headers = io.BytesIO(body), Message()
        for name, value in {
            "Host": "127.0.0.1:8796",
            "X-Atlas-Local": "1",
            "Content-Length": str(len(body)),
        }.items():
            handler.headers[name] = value
        replies = []
        handler.send = lambda status, body: replies.append((status, body))
        handler.handle_action()
        return replies[0]

    def test_delayed_reap_keeps_handle_busy_and_refuses_replacement_and_reset(self):
        self.assertFalse(self.session.stop())
        self.assertEqual(self.session.status, "stopping")
        self.assertTrue(self.session.metadata()["busy"])
        self.assertTrue(self.session.snapshot()["worker_alive"])
        with (
            patch.object(
                live,
                "verify_model",
                side_effect=AssertionError("Must not load a replacement"),
            ),
            self.assertRaises(ValueError),
        ):
            self.session.start({"prompt": "benign fixture"})
        status, body = self.handler("reset")
        self.assertEqual(status, 503)
        self.assertEqual(body["code"], "cleanup_pending")
        self.assertEqual(self.session.id, "owner-fixture")
        self.assertIs(self.session.process, self.child)
        self.child.pending = False
        self.session.tick()
        self.assertEqual(self.session.status, "cancelled")
        self.assertFalse(self.session.metadata()["busy"])
        self.assertIsNone(self.session.process)
        self.assertTrue(self.child.stdout.closed)
        self.assertEqual(self.handler("reset")[0], 200)
        self.assertIsNone(self.session.id)

    def test_completion_is_not_visible_until_reap_succeeds(self):
        self.session.status = "complete"
        self.assertFalse(self.session.stop())
        self.assertEqual(self.session.status, "stopping")
        self.child.pending = False
        self.session.tick()
        self.assertEqual(self.session.status, "complete")
        self.assertFalse(self.session.snapshot()["worker_alive"])

    def test_signal_failures_retain_ownership_and_lookup_race_can_reap(self):
        self.child.signal_error = PermissionError("Injected signal failure")
        self.assertFalse(self.session.stop())
        self.assertIs(self.session.process, self.child)
        self.child.signal_error = ProcessLookupError("Already exited")
        self.child.pending = False
        self.assertTrue(self.session.stop())
        self.assertIsNone(self.session.process)

    def test_renderer_termination_escalates_and_server_closes(self):
        atlas = Child()
        calls = []

        def wait(timeout):
            calls.append(timeout)
            if atlas.kills == 0:
                raise subprocess.TimeoutExpired("renderer", timeout)
            atlas.returncode = 0
            return 0

        atlas.wait = wait
        server = SimpleNamespace(server_close=Mock())
        with patch.object(live.time, "sleep"):
            self.assertFalse(live.cleanup_owned(self.session, atlas, server))
        self.assertEqual(calls, [0.5, 1.5])
        self.assertEqual(atlas.terminates, 1)
        self.assertEqual(atlas.kills, 1)
        self.assertIs(self.session.process, self.child)
        server.server_close.assert_called_once()

    def test_unexpected_session_cleanup_error_cannot_skip_other_resources(self):
        atlas, server = Child(), SimpleNamespace(server_close=Mock())
        atlas.pending = False
        with (
            patch.object(
                self.session, "stop", side_effect=RuntimeError("Injected cleanup error")
            ),
            self.assertRaises(RuntimeError),
        ):
            live.cleanup_owned(self.session, atlas, server)
        self.assertEqual(atlas.terminates, 1)
        server.server_close.assert_called_once()


class HashContracts(unittest.TestCase):
    def test_bounded_hashing_does_not_depend_on_file_digest(self):
        data = b"official-format hash fixture"
        reads = []

        class Source(io.BytesIO):
            def read(self, size=-1):
                reads.append(size)
                return super().read(size)

        manifest = {"files": {"fixture": hashlib.sha256(data).hexdigest()}}
        with (
            patch.object(live, "MANIFEST", manifest),
            patch.object(Path, "open", return_value=Source(data)),
            patch.object(live.hashlib, "file_digest", None, create=True),
        ):
            live.verify_model(Path("/unused"))
        self.assertEqual(reads, [1024 * 1024, 1024 * 1024])


if __name__ == "__main__":
    unittest.main()
