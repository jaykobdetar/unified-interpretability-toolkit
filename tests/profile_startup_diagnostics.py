"""Pure startup receipt/failure-cleanup checks; no real process, socket or thread."""

from contextlib import ExitStack
import errno
import hashlib
import json
from pathlib import Path
import signal
import stat
import sys
import types
import unittest
from unittest.mock import patch

from atlas_host import profile_os
from atlas_host.profile_os import OwnedProcess, ProcessBook
from atlas_host.startup_diagnostics import (
    StartupDiagnostics,
    STDERR_LIMIT,
    ATTEMPT_LIMIT,
    error_chain,
    sanitized_stderr,
)

ARGV = [
    "/private/bin",
    "hosted-renderer",
    "--model",
    "/private/token-secret",
    "--cache",
    "/private/cache",
    "--name",
    "capability-secret",
    "--revision",
    "revision-secret",
    "--channel-fd",
    "19",
]


class Pipe:
    def __init__(self):
        self.closed = False

    def fileno(self):
        return 22

    def close(self):
        self.closed = True


class Diagnostics(unittest.TestCase):
    def setUp(self):
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
                patch(target, side_effect=AssertionError("Real OS work forbidden"))
            )
        self.stack.enter_context(
            patch("atlas_host.startup_diagnostics.fcntl.fcntl", return_value=1)
        )
        self.stack.enter_context(patch("os.get_inheritable", return_value=False))
        self.stack.enter_context(
            patch("os.fstat", return_value=types.SimpleNamespace(st_mode=stat.S_IFSOCK))
        )
        self.stack.enter_context(patch("os.set_blocking"))
        self.collector = StartupDiagnostics()

    def record(self):
        return self.collector.prepare(ARGV, (19,))

    def child(self, chunks, *, status=256, reaped=True):
        self.recorded = self.record()
        self.pipe = Pipe()
        process = types.SimpleNamespace(
            pid=42, stdout=None, stderr=self.pipe, returncode=None
        )
        self.chunks = list(chunks)
        self.reaped = reaped
        self.waits = 0

        def read(fd, count):
            self.assertEqual((fd, count), (22, 4096))
            if not self.chunks:
                raise BlockingIOError()
            value = self.chunks.pop(0)
            if isinstance(value, BaseException):
                raise value
            self.assertLessEqual(len(value), count)
            return value

        def wait(pid, flags):
            self.waits += 1
            return (
                (42, status, types.SimpleNamespace(ru_utime=0.1, ru_stime=0.2))
                if self.reaped
                else (0, 0, None)
            )

        return OwnedProcess(
            process,
            diagnostic=self.recorded,
            read=read,
            wait4=wait,
            stats=lambda pid: {"start": 1, "cpu": 0.0, "rss": 1},
            descendants=lambda pid: set(),
        )

    def book(self):
        return ProcessBook(
            stats=lambda pid: {"start": 1, "cpu": 0.0, "rss": 1},
            descendants=lambda pid: set(),
        )

    def test_sanitized_command_fd_metadata_and_expectations(self):
        snap = self.record().snapshot()
        text = json.dumps(snap)
        for secret in (
            "/private",
            "token-secret",
            "capability-secret",
            "revision-secret",
        ):
            self.assertNotIn(secret, text)
        self.assertEqual(snap["argv"][-2:], ["--channel-fd", "19"])
        self.assertEqual(
            snap["passed_fds"][0],
            {
                "fd": 19,
                "role": "private renderer channel",
                "parent_fd_flags": 1,
                "parent_cloexec": True,
                "parent_inheritable": False,
                "parent_is_socket": True,
            },
        )
        self.assertTrue(snap["close_fds"])
        self.assertFalse(snap["child_fd_expectations"]["observed_in_child"])
        self.assertTrue(snap["child_fd_expectations"]["passed_channel_survives_exec"])
        self.assertFalse(
            snap["child_fd_expectations"]["passed_channel_cloexec_after_exec"]
        )

    def test_nonzero_exit_and_safe_stderr_survive_cleanup_and_inventory_prune(self):
        child = self.child([b"ERROR: Cannot duplicate hosted channel\n", b""])
        book = self.book()
        book.owned = [child]
        first = child.sample()
        self.assertFalse(first["reaped"])
        self.assertTrue(child.reaped)
        self.assertFalse(book.settled())  # Wait4 alone is not pipe cleanup.
        self.assertTrue(book.cleanup())
        self.assertTrue(self.pipe.closed)
        snap = self.collector.snapshot()["attempts"][0]
        self.assertEqual((snap["wait_status"], snap["exit_code"]), (256, 1))
        self.assertEqual(snap["stderr"], "ERROR: Cannot duplicate hosted channel")
        with patch.object(
            profile_os.subprocess,
            "Popen",
            side_effect=OSError(errno.EACCES, "secret-path"),
        ):
            with self.assertRaises(OSError):
                book.spawn(
                    ARGV, pass_fds=(19,), receipt=False, diagnostics=self.collector
                )
        self.assertEqual(book.owned, [])
        self.assertEqual(self.collector.snapshot()["attempts"][0], snap)
        self.assertEqual(
            {
                k: v
                for k, v in self.collector.snapshot()["attempts"][1][
                    "launch_error"
                ].items()
                if k != "error_chain"
            },
            {
                "phase": "popen_or_ownership",
                "type": "PermissionError",
                "errno": errno.EACCES,
            },
        )
        self.assertTrue(
            book.spawn_uncertain
        )  # Existing conservative refusal preserved.
        self.assertEqual(self.waits, 1)

    def test_signal_terminal_status_is_not_replaced_by_cleanup(self):
        child = self.child([b""], status=signal.SIGTERM)
        self.assertEqual(child.sample()["exit_code"], -signal.SIGTERM)
        child.stop()
        child.sample()
        snap = self.recorded.snapshot()
        self.assertEqual(snap["exit_code"], -signal.SIGTERM)
        self.assertEqual(snap["wait_status"], signal.SIGTERM)
        self.assertEqual(self.waits, 1)

    def test_nonblocking_drain_and_overflow_are_bounded_without_killing_child(self):
        data = b"x" * 4096
        child = self.child([BlockingIOError()] + [data] * 6 + [b""], reaped=False)
        for _ in range(7):
            self.assertFalse(child.sample()["reaped"])
        self.assertEqual(len(self.recorded.raw), STDERR_LIMIT)
        self.assertIsNone(child.stop_at)
        snap = self.recorded.snapshot()
        self.assertEqual(snap["stderr_bytes_seen"], 6 * 4096)
        self.assertTrue(snap["stderr_truncated"])
        self.assertFalse(snap["stderr_sha256_complete"])
        self.assertEqual(
            snap["stderr_sha256_seen"], hashlib.sha256(data * 6).hexdigest()
        )
        self.reaped = True
        self.assertTrue(child.sample()["reaped"])
        self.assertTrue(self.recorded.snapshot()["stderr_sha256_complete"])

    def test_arbitrary_stderr_secrets_are_withheld_but_errno_and_digest_retained(self):
        child = self.child(
            [
                b"ERROR: /private/token-secret (os error 13)\n"
                b"job_capability=capability-secret\n\xff\n",
                b"",
            ]
        )
        child.sample()
        child.sample()
        snap = self.recorded.snapshot()
        text = json.dumps(snap)
        self.assertNotIn("token-secret", text)
        self.assertNotIn("capability-secret", text)
        self.assertIn("(os error 13)", snap["stderr"])
        self.assertIn("[redacted unrecognized stderr line]", snap["stderr"])
        self.assertTrue(snap["stderr_sha256_complete"])

    def test_initialization_failure_still_retains_stderr_and_terminal_status(self):
        child = self.child([b"ERROR: Connected private channel required\n", b""])
        # pidfd failure is injected before any real syscall; cleanup remains owned.
        with patch("os.pidfd_open", side_effect=OSError(errno.EPERM, "private-secret")):
            with self.assertRaises(OSError):
                child.initialize()
        self.assertTrue(child.sample()["reaped"])
        self.assertEqual(
            self.recorded.snapshot()["initialize_error"]["errno"], errno.EPERM
        )
        self.assertEqual(self.recorded.snapshot()["exit_code"], 1)
        self.assertIn(
            "Connected private channel required", self.recorded.snapshot()["stderr"]
        )

    def test_capture_error_is_raised_and_preserved_while_later_cleanup_reaps(self):
        child = self.child([OSError(errno.EBADF, "secret-fd")])
        with self.assertRaises(OSError):
            child.sample()
        self.assertFalse(child.reaped)
        self.assertTrue(self.pipe.closed)
        self.assertTrue(child.sample()["reaped"])
        snap = self.recorded.snapshot()
        self.assertEqual(
            [
                {k: v for k, v in item.items() if k != "error_chain"}
                for item in snap["capture_errors"]
            ],
            [
                {
                    "phase": "stderr_read_or_setup",
                    "type": "OSError",
                    "errno": errno.EBADF,
                }
            ],
        )
        self.assertFalse(snap["stderr_eof"])
        self.assertFalse(snap["stderr_sha256_complete"])
        self.assertEqual(snap["exit_code"], 1)

    def test_inventory_limit_refuses_before_popen_and_does_not_evict_or_mark_spawn_uncertain(
        self,
    ):
        for _ in range(ATTEMPT_LIMIT):
            self.record()
        before = self.collector.snapshot()
        book = self.book()
        with self.assertRaisesRegex(ValueError, "inventory full"):
            book.spawn(ARGV, pass_fds=(19,), receipt=False, diagnostics=self.collector)
        self.assertEqual(self.collector.snapshot(), before)
        self.assertFalse(book.spawning)
        self.assertFalse(book.spawn_uncertain)

    def test_opt_in_popen_uses_only_explicit_fd_and_stderr_pipe(self):
        book = self.book()
        process = types.SimpleNamespace(
            pid=42, stdout=None, stderr=Pipe(), returncode=None
        )
        with patch.object(
            profile_os.subprocess, "Popen", return_value=process
        ) as launch:
            child = book.spawn(
                ARGV, pass_fds=(19,), receipt=False, diagnostics=self.collector
            )
        self.assertEqual(
            launch.call_args.kwargs,
            {
                "stdin": profile_os.subprocess.DEVNULL,
                "stdout": profile_os.subprocess.DEVNULL,
                "stderr": profile_os.subprocess.PIPE,
                "close_fds": True,
                "pass_fds": (19,),
                "start_new_session": True,
            },
        )
        self.assertIs(book.owned[0], child)
        self.assertFalse(book.settled())
        self.assertEqual(self.collector.snapshot()["attempts"][0]["pid"], 42)

    def test_wrong_command_or_fd_is_refused_without_capture(self):
        for argv, fds in [
            (ARGV[:-1] + ["20"], (19,)),
            (ARGV + ["--secret", "secret"], (19,)),
        ]:
            with self.assertRaises(ValueError):
                self.collector.prepare(argv, fds)
        self.assertFalse(self.collector.snapshot()["attempts"])

    def test_fd_probe_failure_has_safe_receipt_before_any_launch(self):
        book = self.book()
        with patch(
            "atlas_host.startup_diagnostics.fcntl.fcntl",
            side_effect=OSError(errno.EBADF, "secret"),
        ):
            with self.assertRaises(OSError):
                book.spawn(
                    ARGV, pass_fds=(19,), receipt=False, diagnostics=self.collector
                )
        snap = self.collector.snapshot()["attempts"][0]
        self.assertEqual(
            {k: v for k, v in snap["launch_error"].items() if k != "error_chain"},
            {"phase": "fd_metadata", "type": "OSError", "errno": errno.EBADF},
        )
        self.assertFalse(book.spawn_uncertain)
        self.assertFalse(book.spawning)


class ErrorChains(unittest.TestCase):
    def test_cause_and_context_keep_errno_and_sanitized_frame_locations(self):
        try:
            try:
                raise PermissionError(
                    errno.EPERM, "secret-capability", "/private/model"
                )
            except PermissionError as cause:
                raise ValueError("secret-wrapper") from cause
        except ValueError as error:
            receipt = error_chain(error)
        self.assertEqual(len(receipt["exceptions"]), 2)
        outer, inner = receipt["exceptions"]
        self.assertEqual((outer["cause"], outer["context"]), (1, 1))
        self.assertTrue(outer["suppress_context"])
        self.assertEqual(inner["errno"], errno.EPERM)
        self.assertTrue(all(n["frames"] for n in receipt["exceptions"]))
        self.assertTrue(
            all(
                f["file"] == "repo/tests/profile_startup_diagnostics.py"
                for n in receipt["exceptions"]
                for f in n["frames"]
            )
        )
        text = json.dumps(receipt)
        for secret in (
            "secret-capability",
            "secret-wrapper",
            "/private/model",
            str(Path.cwd()),
        ):
            self.assertNotIn(secret, text)
        self.assertFalse(receipt["chain_truncated"])

    def test_exception_graph_cycles_and_long_chains_are_bounded(self):
        root = ValueError("secret")
        current = root
        for _ in range(20):
            current.__cause__ = ValueError("secret")
            current = current.__cause__
        current.__context__ = root
        receipt = error_chain(root)
        self.assertEqual(len(receipt["exceptions"]), 8)
        self.assertTrue(receipt["chain_truncated"])
        root.__cause__ = root
        self.assertEqual(len(error_chain(root)["exceptions"]), 1)
        self.assertEqual(error_chain(root)["exceptions"][0]["cause"], 0)

    def test_frame_budget_and_unknown_paths_and_function_names_are_withheld(self):
        def recurse(n):
            if n:
                recurse(n - 1)
            else:
                raise ValueError("secret")

        try:
            recurse(80)
        except ValueError as error:
            receipt = error_chain(error)
        self.assertEqual(len(receipt["exceptions"][0]["frames"]), 64)
        self.assertTrue(receipt["exceptions"][0]["frames_truncated"])
        try:
            exec(
                compile(
                    'raise ValueError("secret")',
                    "/private/secret-capability.py",
                    "exec",
                )
            )
        except ValueError as error:
            receipt = error_chain(error)
        self.assertEqual(
            receipt["exceptions"][0]["frames"][-1]["file"], "<external-code>"
        )
        self.assertEqual(
            receipt["exceptions"][0]["frames"][-1]["function"], "<withheld>"
        )
        self.assertNotIn("secret", json.dumps(receipt))

    def test_explicit_harness_mapping_keeps_location_without_machine_path(self):
        try:
            exec(
                compile(
                    'raise PermissionError(1,"secret")',
                    "/private/runtime_harness.py",
                    "exec",
                )
            )
        except PermissionError as error:
            receipt = error_chain(
                error,
                locations={
                    "/private/runtime_harness.py": "qualification/runtime_harness.py"
                },
            )
        self.assertEqual(
            receipt["exceptions"][0]["frames"][-1],
            {
                "file": "qualification/runtime_harness.py",
                "function": "<module>",
                "line": 1,
            },
        )
        self.assertNotIn("/private", json.dumps(receipt))

    def test_native_kind_errno_format_is_preserved_and_unknown_kind_withheld(self):
        text = "ERROR: Hosted channel peer_addr failed: kind=PermissionDenied; errno=1"
        self.assertEqual(sanitized_stderr(text.encode()), text)
        for field in (
            "kind=secret_capability; errno=1",
            "kind=MySecret; errno=1",
            "kind=Other; errno=secret",
            "kind=Other; errno=none secret",
        ):
            text = "ERROR: Hosted channel peer_addr failed: " + field
            self.assertEqual(
                sanitized_stderr(text.encode()), "[redacted unrecognized stderr line]"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
