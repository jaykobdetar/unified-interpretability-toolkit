"""Observation identity and diagnostic contracts with inert owner/OS doubles."""

from contextlib import ExitStack
import errno
import hashlib
import json
import stat
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from atlas_host.profile_observation import SpawnObservationPending
from atlas_host import startup_diagnostics as diagnostic


class HostLeaves(unittest.TestCase):
    def setUp(self) -> None:
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for target in (
            "subprocess.Popen",
            "socket.socketpair",
            "threading.Thread.start",
            "os.pidfd_open",
            "os.kill",
            "signal.pidfd_send_signal",
            "os.memfd_create",
        ):
            self.stack.enter_context(
                patch(target, side_effect=AssertionError("Live work forbidden"))
            )

    def test_pending_exception_keeps_message_payload_identity_and_mutations(
        self,
    ) -> None:
        observed = {"rss_bytes": 31, "all_owned_accounted": False}
        pending = SpawnObservationPending(observed)
        self.assertIsInstance(pending, RuntimeError)
        self.assertEqual(
            pending.args, ("Owned spawn registration observation pending",)
        )
        self.assertIs(pending.observed, observed)
        observed["available_bytes"] = 47
        self.assertEqual(pending.observed, observed)
        self.assertIsNone(pending.__cause__)

    def test_optional_diagnosis_forwards_exact_objects_and_facts(self) -> None:
        error = ValueError("inert")
        fact = object()
        collector = Mock(spec=["event"])
        owner = SimpleNamespace(diagnostics=collector)
        self.assertIsNone(diagnostic.diagnose(owner, "fixture", error, fact=fact))
        collector.event.assert_called_once_with("fixture", error, fact=fact)
        self.assertIsNone(diagnostic.diagnose(object(), "fixture", error))
        self.assertIsNone(
            diagnostic.diagnose(SimpleNamespace(diagnostics=None), "fixture", error)
        )

    def test_decorator_preserves_callable_metadata_return_and_base_exception(
        self,
    ) -> None:
        class InertInterruption(BaseException):
            pass

        collector = Mock(spec=["event"])
        owner = SimpleNamespace(diagnostics=collector)
        calls = []
        result = object()
        failure = InertInterruption("inert")

        def action(self: object, number: int, *, fail: bool = False) -> object:
            """Fixture callable metadata."""
            calls.append((self, number, fail))
            if fail:
                raise failure
            return result

        action.fixture_marker = result
        wrapped = diagnostic.diagnosed("fixture.action")(action)
        self.assertEqual(wrapped.__name__, "action")
        self.assertEqual(wrapped.__doc__, "Fixture callable metadata.")
        self.assertEqual(
            wrapped.__dict__, {"fixture_marker": result, "__wrapped__": action}
        )
        self.assertIs(wrapped(owner, 7), result)
        self.assertEqual(calls, [(owner, 7, False)])
        collector.event.assert_not_called()
        caught = None
        try:
            wrapped(owner, 9, fail=True)
        except BaseException as error:
            caught = error
        self.assertIs(caught, failure)
        self.assertEqual(calls, [(owner, 7, False), (owner, 9, True)])
        collector.event.assert_called_once_with("fixture.action", failure)

    def test_error_metadata_and_chain_indexes_are_exact(self) -> None:
        class InertError(Exception):
            pass

        self.assertEqual(
            diagnostic.safe_error(InertError()), {"type": "OtherError", "errno": None}
        )
        self.assertEqual(
            diagnostic.safe_error(RuntimeError()),
            {"type": "RuntimeError", "errno": None},
        )
        child = OSError(errno.EIO, "inert")
        self.assertEqual(
            diagnostic.safe_error(child), {"type": "OSError", "errno": errno.EIO}
        )
        root = ValueError("inert")
        root.__cause__ = child
        chain = diagnostic.error_chain(root)
        self.assertEqual(chain["exception_limit"], 8)
        self.assertEqual(chain["total_frame_limit"], 64)
        self.assertFalse(chain["chain_truncated"])
        self.assertEqual(len(chain["exceptions"]), 2)
        self.assertEqual(chain["exceptions"][0]["cause"], 1)
        self.assertIsNone(chain["exceptions"][1]["cause"])

    def test_static_stderr_lines_preserve_separator_and_prefix_limit(self) -> None:
        line = b"ERROR: Owned resources exceeded"
        self.assertEqual(
            diagnostic.sanitized_stderr(line + b"\n" + line),
            "ERROR: Owned resources exceeded\nERROR: Owned resources exceeded",
        )
        many = (line + b"\n") * 600
        self.assertEqual(diagnostic.sanitized_stderr(many), many.decode()[:16384])

    def test_record_initialization_updates_capture_and_snapshot(self) -> None:
        record = diagnostic.StartupRecord(19)
        initial = record.snapshot()
        self.assertTrue(initial["argv_sanitized"])
        self.assertFalse(initial["reaped"])
        self.assertEqual(initial["stderr_bytes_seen"], 0)
        self.assertEqual(initial["stderr_prefix_bytes"], 0)
        self.assertFalse(initial["stderr_truncated"])
        self.assertEqual(initial["capture_errors"], [])
        record.update(pid=42, reaped=True, exit_code=7, wait_status=1792)
        updated = record.snapshot()
        self.assertEqual(
            {
                key: updated[key]
                for key in ("pid", "reaped", "exit_code", "wait_status")
            },
            {"pid": 42, "reaped": True, "exit_code": 7, "wait_status": 1792},
        )
        error = OSError(errno.EIO, "inert")
        record.capture_error("read", error)
        captured = record.snapshot()["capture_errors"]
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["phase"], "read")
        self.assertEqual(captured[0]["type"], "OSError")
        self.assertEqual(captured[0]["errno"], errno.EIO)
        self.assertEqual(captured[0]["error_chain"], diagnostic.error_chain(error))
        captured.clear()
        self.assertEqual(len(record.snapshot()["capture_errors"]), 1)

    def test_stderr_accounting_digest_and_completeness(self) -> None:
        for size in (0, 16384, 16385):
            with self.subTest(size=size):
                record = diagnostic.StartupRecord(19)
                raw = b"xy" * (size // 2) + b"z" * (size % 2)
                record.feed(raw[:3])
                record.feed(raw[3:])
                record.update(stderr_eof=True)
                snap = record.snapshot()
                self.assertEqual(snap["stderr_bytes_seen"], size)
                self.assertEqual(snap["stderr_prefix_bytes"], min(size, 16384))
                self.assertEqual(snap["stderr_truncated"], size > 16384)
                self.assertEqual(
                    snap["stderr_sha256_seen"], hashlib.sha256(raw).hexdigest()
                )
                self.assertTrue(snap["stderr_sha256_complete"])
                record.capture_error("read", OSError(errno.EIO, "inert"))
                self.assertFalse(record.snapshot()["stderr_sha256_complete"])

    def test_collector_initial_state_counters_and_serialized_byte_charge(self) -> None:
        collector = diagnostic.StartupDiagnostics()
        initial = collector.snapshot()
        self.assertEqual(
            initial,
            {
                "version": 1,
                "attempt_limit": 4,
                "stderr_prefix_limit": 16384,
                "attempts": [],
                "events": [],
                "event_limit": 64,
                "event_byte_limit": 524288,
                "event_bytes": 0,
                "events_dropped": 0,
            },
        )
        collector.event("fixture", rss_bytes=31)
        collector.event("fixture", rss_bytes=31)
        collector.event("fixture.second", rss_bytes=47)
        snap = collector.snapshot()
        self.assertEqual(len(snap["events"]), 2)
        first, second = snap["events"]
        self.assertEqual(
            (first["count"], first["first_sequence"], first["last_sequence"]), (2, 1, 2)
        )
        self.assertEqual(
            (second["count"], second["first_sequence"], second["last_sequence"]),
            (1, 3, 3),
        )
        payloads = (
            {"site": "fixture", "thread": "caller", "facts": {"rss_bytes": 31}},
            {"site": "fixture.second", "thread": "caller", "facts": {"rss_bytes": 47}},
        )
        charge = sum(
            len(
                json.dumps(
                    payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
                )
            )
            for payload in payloads
        )
        self.assertEqual(snap["event_bytes"], charge)

    def test_prepare_keeps_inherited_fd_metadata_without_launch(self) -> None:
        argv = [
            "renderer",
            "hosted-renderer",
            "--model",
            "model",
            "--cache",
            "cache",
            "--name",
            "name",
            "--revision",
            "revision",
            "--channel-fd",
            "19",
        ]
        with (
            patch.object(diagnostic.fcntl, "fcntl", return_value=1),
            patch.object(diagnostic.os, "get_inheritable", return_value=False),
            patch.object(
                diagnostic.os,
                "fstat",
                return_value=SimpleNamespace(st_mode=stat.S_IFSOCK),
            ),
        ):
            collector = diagnostic.StartupDiagnostics()
            record = collector.prepare(argv, (19,))
        self.assertEqual(
            record.snapshot()["passed_fds"],
            [
                {
                    "fd": 19,
                    "role": "private renderer channel",
                    "parent_fd_flags": 1,
                    "parent_cloexec": True,
                    "parent_inheritable": False,
                    "parent_is_socket": True,
                }
            ],
        )
        self.assertEqual(collector.snapshot()["attempts"], [record.snapshot()])


if __name__ == "__main__":
    unittest.main()
