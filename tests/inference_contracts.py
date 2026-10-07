#!/usr/bin/env python3
"""Bounded coordinator protocol/control tests. Real generation is a separate test."""

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from email.message import Message
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import live_inference as live


class Contracts(unittest.TestCase):
    def test_invalid_requests_do_not_start_work(self):
        session = live.Session("python3", Path("/unused"))
        for data in [
            {},
            {"prompt": ""},
            {"prompt": "x" * 4097},
            {"prompt": "x", "max_new_tokens": 33},
            {"prompt": "x", "max_new_tokens": True},
            {"prompt": "x", "layer": 30},
            {"prompt": "x", "layer": False},
        ]:
            with self.assertRaises(ValueError):
                session.start(data)
            self.assertIsNone(session.process)

    def test_resource_start_refusal(self):
        session = live.Session("python3", Path("/unused"))
        with (
            patch.object(live, "available", return_value=4 * live.GIB),
            self.assertRaises(ValueError),
        ):
            session.start({"prompt": "x"})

    def test_cancel_kills_actual_child_and_reaps(self):
        session = live.Session("python3", Path("/unused"))
        process = subprocess.Popen(
            ["python3", "-c", "import time; time.sleep(30)"], stdout=subprocess.PIPE
        )
        session.process, session.status = process, "loading"
        session.stop()
        self.assertIsNotNone(process.poll())
        self.assertEqual(session.status, "cancelled")
        self.assertIsNone(session.process)
        session.stop()  # idempotent

    def test_low_headroom_kills_only_owned_worker(self):
        session = live.Session("python3", Path("/unused"))
        process = subprocess.Popen(
            ["python3", "-c", "import time; time.sleep(30)"], stdout=subprocess.PIPE
        )
        session.process, session.status = process, "running"
        with patch.object(live, "available", return_value=3 * live.GIB):
            session.tick()
        self.assertEqual(session.status, "resource_limit")
        self.assertIsNotNone(process.poll())

    def test_abandoned_client_expires(self):
        session = live.Session("python3", Path("/unused"))
        process = subprocess.Popen(
            ["python3", "-c", "import time; time.sleep(30)"], stdout=subprocess.PIPE
        )
        session.process, session.status = process, "running"
        session.last_seen = time.monotonic() - 16
        session.tick()
        self.assertEqual(session.status, "client_timeout")
        self.assertIsNotNone(process.poll())

    def test_worker_error_and_truncated_output(self):
        for output, expected in [("bad json\\n", "error"), ("", "error")]:
            session = live.Session("python3", Path("/unused"))
            process = subprocess.Popen(
                ["python3", "-c", f"print({output!r})"], stdout=subprocess.PIPE
            )
            session.process, session.status = process, "running"
            process.wait()
            os = __import__("os")
            os.set_blocking(process.stdout.fileno(), False)
            session.tick()
            self.assertEqual(session.status, expected)
            self.assertIsNone(session.process)

    def test_completion_reaps_worker_before_snapshot(self):
        session = live.Session("python3", Path("/unused"))
        step = {"type": "step", "index": 0, "activation": [0.0] * 576}
        done = {"type": "done", "generated_tokens": 1, "reason": "token_limit"}
        code = f"import time; print({json.dumps(step)!r}, flush=True); print({json.dumps(done)!r}, flush=True); time.sleep(30)"
        process = subprocess.Popen(["python3", "-c", code], stdout=subprocess.PIPE)
        session.process, session.status = process, "running"
        __import__("os").set_blocking(process.stdout.fileno(), False)
        deadline = time.monotonic() + 2
        while session.process is not None and time.monotonic() < deadline:
            session.tick()
            time.sleep(0.01)
        self.assertEqual(session.status, "complete")
        self.assertFalse(session.snapshot()["worker_alive"])
        self.assertIsNotNone(process.poll())
        self.assertEqual(len(session.steps), 1)

    def completed_child(self, wait_for_input=False, done=True):
        session = live.Session("python3", Path("/unused"))
        events = [{"type": "step", "index": 0, "activation": [0.0] * 576}]
        if done:
            events.append(
                {"type": "done", "generated_tokens": 1, "reason": "token_limit"}
            )
        code = "import sys; " + ("sys.stdin.read(1); " if wait_for_input else "")
        code += "; ".join(
            f"print({json.dumps(event)!r}, flush=True)" for event in events
        )
        process = subprocess.Popen(
            ["python3", "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        session.process, session.status = process, "loading"
        os.set_blocking(process.stdout.fileno(), False)
        self.addCleanup(session.stop)
        self.addCleanup(process.stdin.close)
        return session, process

    def test_worker_exit_after_empty_read_keeps_final_records(self):
        session, process = self.completed_child(wait_for_input=True)
        real_read = os.read

        def finish_after_empty_read(fd, size):
            try:
                return real_read(fd, size)
            except BlockingIOError:
                # Controlled ordinary pipe scheduling: EOF arrives after EAGAIN.
                process.stdin.write(b"1")
                process.stdin.flush()
                process.wait(timeout=2)
                raise

        with patch.object(live.os, "read", side_effect=finish_after_empty_read):
            session.tick()
        self.assertIsNotNone(session.process)
        self.assertEqual(session.status, "loading")
        session.tick()
        self.assertEqual(session.status, "complete")
        self.assertEqual(len(session.steps), 1)
        self.assertEqual(session.details["reason"], "token_limit")
        self.assertFalse(session.snapshot()["worker_alive"])

    def test_exited_worker_drain_respects_per_tick_budget(self):
        session, process = self.completed_child()
        process.wait(timeout=2)
        real_read = os.read
        with patch.object(
            live.os, "read", side_effect=lambda fd, size: real_read(fd, min(size, 128))
        ) as read:
            session.tick()
            self.assertEqual(read.call_count, 8)
            self.assertIsNotNone(session.process)
            self.assertEqual(session.status, "loading")
            self.assertLessEqual(len(session.buffer), 8 * 128)
            for _ in range(4):
                before = read.call_count
                session.tick()
                self.assertLessEqual(read.call_count - before, 8)
                if session.process is None:
                    break
        self.assertEqual(session.status, "complete")
        self.assertEqual(len(session.steps), 1)

    def test_eof_without_completion_record_is_error(self):
        session, process = self.completed_child(done=False)
        process.wait(timeout=2)
        session.tick()
        self.assertEqual(session.status, "error")
        self.assertEqual(len(session.steps), 1)
        self.assertIn("before completion", session.details["error"])
        self.assertIsNone(session.process)

    def test_model_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            for name in live.MANIFEST["files"]:
                (Path(d) / name).write_text("wrong")
            with self.assertRaises(ValueError):
                live.verify_model(Path(d))


class SessionBoundary(unittest.TestCase):
    """Benign in-memory handler calls: no server, sockets, or model execution."""

    def setUp(self):
        self.session = live.Session("python3", Path("/unused"))
        self.session.id = "unit-session-capability"
        self.session.status = "running"
        self.session.details = {
            "prompt_ids": [101, 42],
            "error": "session-local message",
        }
        self.session.steps.append(
            {"generated_text": "fixture text", "activation": [0.5] * 576}
        )

    def request(self, action="", data=None):
        handler = live.Handler.__new__(live.Handler)
        handler.server = SimpleNamespace(session=self.session, server_port=8796)
        handler.command = "POST" if action else "GET"
        handler.path = "/api/inference" + ("/" + action if action else "")
        payload = json.dumps(data).encode() if action else b""
        handler.rfile = io.BytesIO(payload)
        handler.headers = Message()
        for key, value in {
            "Host": "127.0.0.1:8796",
            "Origin": "http://127.0.0.1:8796",
            "X-Atlas-Local": "1",
            "Content-Length": str(len(payload)),
        }.items():
            handler.headers[key] = value
        responses = []
        handler.send = lambda status, body: responses.append((status, body))
        handler.handle_action()
        self.assertEqual(len(responses), 1)
        return responses[0]

    def test_metadata_has_only_public_fields_for_every_session_state(self):
        for state in ("idle", "loading", "running", "complete", "error", "cancelled"):
            self.session.status = state
            with patch.object(
                self.session,
                "snapshot",
                side_effect=AssertionError("Private snapshot read"),
            ):
                status, body = self.request()
            self.assertEqual(status, 200)
            self.assertEqual(
                set(body),
                {
                    "model",
                    "revision",
                    "engine",
                    "limits",
                    "busy",
                    "comparison",
                    "observations",
                    "prompt_pair",
                    "sweep",
                    "architecture",
                    "head_layout",
                    "busy_owner",
                    "queue_capacity",
                },
            )
            self.assertEqual(body["busy"], state in ("loading", "running"))
            self.assertEqual(body["model"], live.MANIFEST["repo"])
            self.assertNotIn("unit-session-capability", json.dumps(body))
            self.assertNotIn("fixture text", json.dumps(body))
            self.assertNotIn("prompt_ids", json.dumps(body))

    def test_missing_or_different_capability_cannot_read_or_change_session(self):
        before = self.session.snapshot()
        for action in ("poll", "cancel", "reset"):
            for data in (
                {},
                {"session": None},
                {"session": "different-session"},
                {"session": 3},
                {"session": "é"},
            ):
                status, body = self.request(action, data)
                self.assertEqual(status, 409)
                self.assertEqual(body, {"error": "Stale session"})
                self.assertEqual(self.session.snapshot(), before)

    def test_owner_can_read_cancel_and_clear_its_session(self):
        owner = {"session": self.session.id}
        status, body = self.request("poll", owner)
        self.assertEqual(status, 200)
        self.assertEqual(body["details"]["prompt_ids"], [101, 42])
        self.assertEqual(body["steps"][0]["generated_text"], "fixture text")
        self.assertEqual(self.request("cancel", owner)[1]["status"], "cancelled")
        status, body = self.request("reset", owner)
        self.assertEqual(status, 200)
        self.assertIsNone(body["session"])
        self.assertEqual(body["steps"], [])
        self.assertEqual(body["details"], {})
        self.assertEqual(self.request("poll", owner)[0], 409)

    def test_previous_capability_cannot_access_replacement_session(self):
        old = {"session": self.session.id}
        self.session.id = "replacement-session-capability"
        for action in ("poll", "cancel", "reset"):
            self.assertEqual(self.request(action, old)[0], 409)
        status, body = self.request("poll", {"session": self.session.id})
        self.assertEqual(status, 200)
        self.assertEqual(body["session"], "replacement-session-capability")


if __name__ == "__main__":
    unittest.main()
