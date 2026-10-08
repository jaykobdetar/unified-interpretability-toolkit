"""OS provider contracts with inert processes, descriptors, signals and clocks."""

from contextlib import ExitStack
from pathlib import Path
import signal
import sys
import threading
import types
import unittest
from unittest.mock import Mock, call, patch

from atlas_host import profile_os as os_boundary


class OSBoundaryContracts(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for target in (
            "subprocess.Popen",
            "threading.Thread.start",
            "os.kill",
            "os.pidfd_open",
            "signal.pidfd_send_signal",
            "os.memfd_create",
        ):
            self.stack.enter_context(
                patch(target, side_effect=AssertionError("Live work forbidden"))
            )

    def ok(self, callback, *args, **kwargs):
        try:
            return callback(*args, **kwargs)
        except Exception as error:
            self.fail(f"Expected valid local call: {type(error).__name__}: {error}")

    def error(self, message, callback, *args, **kwargs):
        try:
            callback(*args, **kwargs)
        except Exception as error:
            self.assertIs(type(error), ValueError)
            self.assertEqual(str(error), message)
        else:
            self.fail("Expected the existing local refusal")

    def owned(self, **kwargs):
        process = types.SimpleNamespace(
            pid=42, stdout=None, stderr=None, returncode=None
        )
        options = {
            "clock": lambda: 2.0,
            "stats": lambda pid: {"start": 17, "cpu": 0.2, "rss": 100},
            "descendants": lambda pid: set(),
            "wait4": lambda pid, flags: (0, 0, None),
            "read": lambda fd, count: b"",
        }
        options.update(kwargs)
        return self.ok(os_boundary.OwnedProcess, process, **options)

    def book(self, **kwargs):
        options = {
            "stats": lambda pid: {"start": 17, "cpu": 0.2, "rss": 100},
            "descendants": lambda pid: set(),
            "available": lambda: 6000,
            "clock": lambda: 2.0,
        }
        options.update(kwargs)
        with patch.object(os_boundary.os, "getpid", return_value=11):
            return self.ok(os_boundary.ProcessBook, **options)

    def hooks(self):
        events = []
        book = types.SimpleNamespace(
            diagnostics=object(),
            spawn_uncertain=False,
            resources=Mock(return_value={"rss_bytes": 123}),
            spawn=Mock(return_value=object()),
        )
        binary = types.SimpleNamespace(
            path="/fixture/bin", check=lambda: events.append("binary")
        )
        source = {
            "root": "/fixture/model",
            "revision": "pinned",
            "cache": "/fixture/cache",
            "binding": {"tensor": "weight", "slice": {"leading_indices": [2, 3]}},
            "check": lambda: events.append("source"),
        }
        watchdog = types.SimpleNamespace(
            register=Mock(return_value=object()), close=Mock()
        )
        policy = types.SimpleNamespace(
            check_binary=lambda value: events.append(("policy", value))
        )
        h = self.ok(
            os_boundary.LinuxProfileHooks,
            book,
            binary,
            source,
            watchdog,
            launch_policy=policy,
        )
        return h, events

    def test_platform_metadata_exact_values_and_export_map(self):
        owner = types.SimpleNamespace(present=True)
        with patch.object(
            os_boundary, "PLATFORM_APIS", {"fixture": (owner, ("present", "absent"))}
        ):
            self.assertEqual(
                self.ok(os_boundary.platform_capabilities),
                {
                    "executable": sys.executable,
                    "version": sys.version,
                    "platform": sys.platform,
                    "apis": {"fixture.present": True, "fixture.absent": False},
                },
            )

    def test_platform_requirement_identity_and_sorted_missing_text(self):
        facts = {"platform": "linux", "apis": {"z": True, "a": True}}
        with patch.object(os_boundary, "platform_capabilities", return_value=facts):
            self.assertIs(self.ok(os_boundary.require_platform), facts)
            facts["apis"] = {"z": False, "a": False}
            self.error(
                "Hosted provider requires an explicit compatible Linux Python; missing APIs: a, z",
                os_boundary.require_platform,
            )

    def test_json_values_and_existing_nonfinite_wording(self):
        self.assertEqual(
            self.ok(os_boundary.strict_json, b'{"a":7,"b":[2,3]}'),
            {"a": 7, "b": [2, 3]},
        )
        self.error("Nonfinite JSON", os_boundary.strict_json, b"NaN")

    def test_proc_stat_exact_fields_and_registered_path(self):
        fields = ["0"] * 22
        for index, value in [(11, 10), (12, 5), (19, 123), (21, 7)]:
            fields[index] = str(value)
        with (
            patch.object(
                os_boundary.Path,
                "read_text",
                return_value="42 (nested (name)) " + " ".join(fields),
            ) as read,
            patch.object(
                os_boundary.os,
                "sysconf",
                side_effect=lambda key: 100 if key == "SC_CLK_TCK" else 4096,
            ),
        ):
            self.assertEqual(
                self.ok(os_boundary.proc_stat, 42),
                {"start": 123, "cpu": 0.15, "rss": 28672},
            )
            read.assert_called_once_with()

    def test_children_exact_set_and_supported_task_and_text_edges(self):
        task = Mock()
        task.__truediv__ = Mock(
            return_value=types.SimpleNamespace(read_text=lambda: "42 42 43")
        )
        with patch.object(os_boundary.Path, "iterdir", return_value=iter([task] * 64)):
            self.assertEqual(self.ok(os_boundary.children, 42), {42, 43})
        task.__truediv__.return_value = types.SimpleNamespace(
            read_text=lambda: "42" + " " * 4094
        )
        with patch.object(os_boundary.Path, "iterdir", return_value=iter([task])):
            self.assertEqual(self.ok(os_boundary.children, 42), {42})
        with patch.object(os_boundary.Path, "iterdir", return_value=iter([task] * 65)):
            self.error("Owned thread inventory exceeds bound", os_boundary.children, 42)

    def test_meter_initial_fields_and_diagnostic_identity(self):
        diagnostic = object()
        m = self.ok(os_boundary.ThreadMeter, diagnostics=diagnostic)
        self.assertIs(m.diagnostics, diagnostic)
        self.assertIsNone(m.clock_id)
        self.assertIsNone(m.final)
        self.assertTrue(m.lock.acquire(False))
        m.lock.release()

    def test_meter_register_exact_thread_clock_and_return(self):
        m = os_boundary.ThreadMeter()
        with (
            patch.object(os_boundary.threading, "get_ident", return_value=42),
            patch.object(
                os_boundary.time, "pthread_getcpuclockid", return_value=77
            ) as clock,
        ):
            self.assertIsNone(self.ok(m.register))
            clock.assert_called_once_with(42)
        self.assertEqual(m.clock_id, 77)
        self.error("CPU context already registered", m.register)

    def test_meter_read_live_and_frozen_values(self):
        m = os_boundary.ThreadMeter()
        self.error("CPU context unavailable", m.read)
        m.clock_id = 77
        with patch.object(os_boundary.time, "clock_gettime", return_value=0.25) as read:
            self.assertEqual(self.ok(m.read), 0.25)
            read.assert_called_once_with(77)
            m.final = 0.5
            self.assertEqual(self.ok(m.read), 0.5)
            self.assertEqual(read.call_count, 1)

    def test_meter_freeze_clears_clock_and_keeps_final_value(self):
        m = os_boundary.ThreadMeter()
        m.clock_id = 77
        with patch.object(os_boundary.time, "clock_gettime", return_value=0.75) as read:
            self.assertIsNone(self.ok(m.freeze))
            read.assert_called_once_with(77)
        self.assertEqual(m.final, 0.75)
        self.assertIsNone(m.clock_id)

    def test_owned_initial_values_and_diagnostic_pid(self):
        diagnostic = types.SimpleNamespace(update=Mock())
        c = self.owned(diagnostic=diagnostic)
        self.assertEqual(c.pid, 42)
        self.assertEqual(c.cpu, 0.0)
        self.assertIsNone(c.receipt)
        self.assertFalse(c.stderr_eof)
        diagnostic.update.assert_called_once_with(pid=42)

    def test_initialize_dispatch_and_exact_error_identity(self):
        c = self.owned()
        c._initialize = Mock()
        self.assertIsNone(self.ok(c.initialize))
        c._initialize.assert_called_once_with()
        c.diagnostic = types.SimpleNamespace(update=Mock())
        error = ValueError("setup failed")
        c._initialize.side_effect = error
        with self.assertRaises(ValueError) as captured:
            c.initialize()
        self.assertIs(captured.exception, error)
        self.assertEqual(c.diagnostic.update.call_count, 1)
        self.assertIn("initialize_error", c.diagnostic.update.call_args.kwargs)

    def test_private_initialize_records_clock_pipe_and_owned_descriptor(self):
        c = self.owned()
        c.process.stdout = types.SimpleNamespace(fileno=lambda: 12)
        c.eof = c.drainable = False
        c._drain_stderr = Mock()
        with (
            patch.object(os_boundary.os, "set_blocking") as blocking,
            patch.object(os_boundary.os, "pidfd_open", return_value=90) as open_fd,
        ):
            self.assertIsNone(self.ok(c._initialize))
            blocking.assert_called_once_with(12, False)
            open_fd.assert_called_once_with(42)
        c._drain_stderr.assert_called_once_with()
        self.assertEqual(c.start, 17)
        self.assertEqual(c.pidfd, 90)
        self.assertTrue(c.drainable)

    def test_observe_descendants_records_once_with_double_only(self):
        c = self.owned(descendants=lambda pid: {43})
        with patch.object(os_boundary.os, "pidfd_open", return_value=91) as open_fd:
            self.assertIsNone(self.ok(c._observe_descendants))
            self.assertIsNone(self.ok(c._observe_descendants))
            open_fd.assert_called_once_with(43)
        self.assertEqual(c.unexpected, {43})
        self.assertEqual(c.descendant_fds, {43: 91})

    def test_signal_dispatch_only_to_fixture_handles(self):
        c = self.owned()
        c.descendant_fds = {43: 91}
        c.pidfd = 90
        with (
            patch.object(os_boundary.signal, "pidfd_send_signal") as send,
            patch.object(os_boundary.os, "kill") as kill,
        ):
            self.assertIsNone(self.ok(c._signal, signal.SIGTERM))
            self.assertEqual(
                send.call_args_list,
                [call(91, signal.SIGTERM), call(90, signal.SIGTERM)],
            )
            kill.assert_not_called()
            c.descendant_fds = {}
            c.pidfd = None
            self.assertIsNone(self.ok(c._signal, signal.SIGTERM))
            kill.assert_called_once_with(42, signal.SIGTERM)

    def test_stop_exact_escalation_bookkeeping_with_fake_clock(self):
        now = [1.0]
        c = self.owned(clock=lambda: now[0])
        c._signal = Mock()
        self.assertIsNone(self.ok(c.stop))
        self.assertEqual(c.stop_at, 1.0)
        self.assertEqual(c._signal.call_args_list, [call(signal.SIGTERM)])
        now[0] = 1.21
        self.assertIsNone(self.ok(c.stop))
        self.assertTrue(c.kill_sent)
        now[0] = 1.4
        self.assertIsNone(self.ok(c.stop))
        self.assertEqual(
            c._signal.call_args_list, [call(signal.SIGTERM), call(signal.SIGKILL)]
        )

    def test_drain_exact_capacity_append_and_eof_close(self):
        c = self.owned()
        stream = types.SimpleNamespace(fileno=lambda: 12, close=Mock())
        c.process.stdout = stream
        c.eof = False
        c.drainable = True
        c.raw = bytearray(b"xy")
        c.read = Mock(side_effect=[b"abc", b""])
        self.assertIsNone(self.ok(c._drain))
        self.assertEqual(c.raw, b"xyabc")
        self.assertIsNone(self.ok(c._drain))
        self.assertTrue(c.eof)
        self.assertEqual(c.read.call_args_list, [call(12, 16383), call(12, 16380)])
        stream.close.assert_called_once_with()

    def test_streams_closed_exact_boolean_result(self):
        c = self.owned()
        for eof, stderr_eof in [
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ]:
            c.eof, c.stderr_eof = eof, stderr_eof
            self.assertIs(c.streams_closed, eof and stderr_eof)

    def test_stderr_single_bounded_read_feed_and_eof_record(self):
        d = types.SimpleNamespace(update=Mock(), feed=Mock(), capture_error=Mock())
        c = self.owned(diagnostic=d)
        d.update.reset_mock()
        c.process.stderr = types.SimpleNamespace(fileno=lambda: 13, close=Mock())
        c.read = Mock(side_effect=[b"diagnostic", b""])
        with patch.object(os_boundary.os, "set_blocking") as blocking:
            self.assertIsNone(self.ok(c._drain_stderr))
            self.assertIsNone(self.ok(c._drain_stderr))
            blocking.assert_called_once_with(13, False)
        self.assertEqual(c.read.call_args_list, [call(13, 4096), call(13, 4096)])
        d.feed.assert_called_once_with(b"diagnostic")
        d.update.assert_called_once_with(stderr_eof=True)
        self.assertTrue(c.stderr_eof)
        c.process.stderr.close.assert_called_once_with()

    def test_sample_final_cpu_returncode_and_receipt(self):
        usage = types.SimpleNamespace(ru_utime=0.4, ru_stime=0.1)
        c = self.owned(wait4=Mock(return_value=(42, 0, usage)))
        c.raw = bytearray(b'{"n":7}')
        c.pidfd = 90
        with patch.object(os_boundary.os, "close") as close:
            result = self.ok(c.sample)
            close.assert_called_once_with(90)
        self.assertEqual(
            result,
            {"cpu_seconds": 0.5, "reaped": True, "exit_code": 0, "receipt": {"n": 7}},
        )
        self.assertIsNone(c.pidfd)
        self.assertEqual(c.process.returncode, 0)
        self.assertEqual(self.ok(c.sample), result)
        c.wait4.assert_called_once_with(42, os_boundary.os.WNOHANG)

    def test_wait_exact_owned_fds_and_short_timeout(self):
        c = self.owned()
        c.pidfd = 90
        c.eof = False
        c.process.stdout = types.SimpleNamespace(fileno=lambda: 12)
        c.stderr_eof = False
        c.stderr_ready = True
        c.process.stderr = types.SimpleNamespace(fileno=lambda: 13)
        with patch.object(os_boundary.select, "select") as select:
            self.assertIsNone(self.ok(c.wait, 0.1))
            self.assertIsNone(self.ok(c.wait, 0.02))
            self.assertEqual(
                select.call_args_list,
                [call([90, 12, 13], [], [], 0.05), call([90, 12, 13], [], [], 0.02)],
            )

    def test_book_initial_fields_and_probe_identity(self):
        stats = Mock(return_value={"start": 17})
        b = self.book(stats=stats)
        self.assertEqual(b.root, 11)
        self.assertEqual(b.root_start, 17)
        self.assertEqual(b.owned, [])
        self.assertIsNone(b.spawn_started)
        self.assertFalse(b.spawning)
        self.assertFalse(b.stop_requested)
        stats.assert_called_once_with(11)

    def test_request_stop_all_calls_all_and_retains_first_error(self):
        b = self.book()
        events = []
        first = ValueError("first")

        def failed():
            events.append("first")
            raise first

        b.owned = [
            types.SimpleNamespace(stop=failed),
            types.SimpleNamespace(stop=lambda: events.append("second")),
        ]
        with self.assertRaises(ValueError) as captured:
            b.request_stop_all()
        self.assertIs(captured.exception, first)
        self.assertTrue(b.stop_requested)
        self.assertEqual(events, ["first", "second"])
        b.owned = []
        self.assertIsNone(self.ok(b.request_stop_all))

    def test_spawn_passes_exact_arguments_and_records_before_initialize(self):
        b = self.book()
        process = object()
        child = object()
        argv = ["/fixture/bin", "worker"]
        with (
            patch.object(
                os_boundary.subprocess, "Popen", return_value=process
            ) as launch,
            patch.object(os_boundary, "OwnedProcess", return_value=child) as wrap,
        ):
            self.assertIs(self.ok(b.spawn, argv, pass_fds=(19,)), child)
            launch.assert_called_once_with(
                argv,
                stdin=os_boundary.subprocess.DEVNULL,
                stdout=os_boundary.subprocess.PIPE,
                stderr=os_boundary.subprocess.DEVNULL,
                close_fds=True,
                pass_fds=(19,),
                start_new_session=True,
            )
            wrap.assert_called_once_with(process, diagnostic=None, diagnostics=None)
        self.assertEqual(b.owned, [child])
        self.assertFalse(b.spawning)
        self.assertEqual(b.spawn_started, 2.0)

    def test_settled_empty_and_completed_inventory_is_true(self):
        b = self.book()
        self.assertIs(self.ok(b.settled), True)
        b.owned = [
            types.SimpleNamespace(reaped=True, streams_closed=True, unexpected=set())
        ]
        self.assertIs(self.ok(b.settled), True)
        b.spawning = True
        self.assertIs(self.ok(b.settled), False)

    def test_cleanup_dispatch_order_filter_and_settled_value(self):
        b = self.book()
        events = []

        def child(name, stop_at):
            return types.SimpleNamespace(
                stop_at=stop_at,
                stop=lambda: events.append((name, "stop")),
                sample=lambda: events.append((name, "sample")),
            )

        b.owned = [child("a", None), child("b", 1.0)]
        b.settled = Mock(return_value=True)
        self.assertIs(self.ok(b.cleanup, stopping_only=True), True)
        self.assertEqual(events, [("b", "stop"), ("b", "sample")])
        b.settled.assert_called_once_with()

    def test_resources_exact_numeric_totals_and_conservative_failure_record(self):
        b = self.book()
        c = self.owned()
        c.start = 17
        b.owned = [c]
        b.descendants = lambda pid: {42} if pid == 11 else set()
        self.assertEqual(
            self.ok(b.resources),
            {
                "rss_bytes": 200,
                "available_bytes": 6000,
                "all_owned_accounted": True,
                "descendants_clear": True,
            },
        )
        b.stats = Mock(side_effect=ValueError("missing local sample"))
        self.assertEqual(
            self.ok(b.resources),
            {
                "rss_bytes": 768 * 1024**2 + 1,
                "available_bytes": 0,
                "all_owned_accounted": False,
                "descendants_clear": False,
            },
        )

    def test_frozen_binary_receipt_hash_algorithm_and_check_call(self):
        stream = Mock()
        stream.fileno.return_value = 19
        path = Mock()
        path.open.return_value.__enter__ = Mock(return_value=stream)
        path.open.return_value.__exit__ = Mock(return_value=False)
        stat = types.SimpleNamespace(st_size=7)
        digest = types.SimpleNamespace(hexdigest=lambda: "expected")
        with (
            patch.object(os_boundary, "Path") as path_type,
            patch.object(os_boundary.os, "fstat", return_value=stat),
            patch.object(
                os_boundary.hashlib, "file_digest", return_value=digest
            ) as file_digest,
            patch.object(os_boundary, "fingerprint", return_value=(7, 8)),
            patch.object(os_boundary.FrozenBinary, "check") as check,
        ):
            path_type.return_value.resolve.return_value = path
            binary = self.ok(os_boundary.FrozenBinary, "fixture", "expected")
            self.assertIs(binary.path, path)
            self.assertEqual(binary.identity, (7, 8))
            file_digest.assert_called_once_with(stream, "sha256")
            check.assert_called_once_with()
            stat.st_size = 32 * 1024**2 + 1
            self.error(
                "Executable exceeds receipt bound",
                os_boundary.FrozenBinary,
                "fixture",
                "expected",
            )

    def test_frozen_binary_check_valid_return_and_changed_text(self):
        b = object.__new__(os_boundary.FrozenBinary)
        b.path = Mock()
        b.identity = (7, 8)
        with patch.object(os_boundary, "fingerprint", return_value=(7, 8)):
            self.assertIsNone(self.ok(b.check))
        with patch.object(os_boundary, "fingerprint", return_value=(9, 8)):
            self.error("Executable changed", b.check)

    def test_hooks_constructor_identity_and_initial_state(self):
        h, _ = self.hooks()
        self.assertIsNone(h.child)
        self.assertFalse(h.attempted)
        self.assertIs(h.diagnostics, h.book.diagnostics)
        self.assertIsNotNone(h.launch_policy)

    def test_hooks_start_gate_exact_memory_and_disk_values(self):
        h, _ = self.hooks()
        with (
            patch.object(os_boundary, "available_bytes", return_value=6000),
            patch.object(
                os_boundary.shutil,
                "disk_usage",
                return_value=types.SimpleNamespace(free=7000),
            ) as disk,
        ):
            self.assertEqual(self.ok(h.start_gate), (6000, 7000))
            disk.assert_called_once_with("/fixture/cache")

    def test_hooks_source_check_order_and_none_return(self):
        h, events = self.hooks()
        self.assertIsNone(self.ok(h.source_check))
        self.assertEqual(events, [("policy", h.binary), "source", "binary"])

    def test_hooks_resource_passthrough_identity_and_count(self):
        h, _ = self.hooks()
        result = self.ok(h.resources)
        self.assertIs(result, h.book.resources.return_value)
        h.book.resources.assert_called_once_with()

    def test_hooks_spawn_exact_option_values_and_retained_child(self):
        h, _ = self.hooks()
        binding = h.source["binding"]
        request = {
            "binding": binding,
            "seed": 17,
            "values": 9,
            "wall_ms": 1000,
            "cpu_ms": 500,
        }
        child = self.ok(h.spawn, request, 19, None)
        self.assertIs(child, h.child)
        self.assertIs(child, h.book.spawn.return_value)
        argv = [
            "/fixture/bin",
            "profile-worker",
            "--model",
            "/fixture/model",
            "--revision",
            "pinned",
            "--tensor",
            "weight",
            "--slice",
            "2,3",
            "--seed",
            "17",
            "--values",
            "9",
            "--wall-ms",
            "1000",
            "--cpu-ms",
            "500",
            "--binding",
            os_boundary.canonical(binding).decode(),
            "--output-fd",
            "19",
        ]
        h.book.spawn.assert_called_once_with(argv, pass_fds=(19,))
        self.assertTrue(h.attempted)

    def test_hooks_watch_callback_deadline_and_identity(self):
        h, _ = self.hooks()
        callback = Mock()
        self.assertIs(self.ok(h.watch, callback, 4.5), h.watchdog.register.return_value)
        h.watchdog.register.assert_called_once_with(callback, 4.5)

    def test_hooks_close_watch_single_callback_and_none(self):
        h, _ = self.hooks()
        self.assertIsNone(self.ok(h.close_admission_watch))
        h.watchdog.close.assert_called_once_with()

    def test_hooks_no_child_true_boolean_without_mutation(self):
        h, _ = self.hooks()
        self.assertIs(self.ok(h.no_child_created), True)
        self.assertIsNone(h.child)
        self.assertFalse(h.book.spawn_uncertain)

    def test_hooks_snapshot_settlement_then_empty_byte_boolean(self):
        h, _ = self.hooks()
        store = types.SimpleNamespace(require_settled=Mock(), owned_storage_bytes=0)
        self.assertIs(self.ok(h.snapshot_closed, store), True)
        store.require_settled.assert_called_once_with()
        store.owned_storage_bytes = 1
        self.assertIs(self.ok(h.snapshot_closed, store), False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
