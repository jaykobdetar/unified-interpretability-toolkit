"""Bounded waits and diagnostic clamps with inert owner and stream doubles."""

import ast
import io
import json
import math
from pathlib import Path
import sys
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from atlas_host import (
    config,
    profile_os,
    profile_service,
    startup_diagnostics as diagnostic,
)


class WaitDiagnosticLimits(unittest.TestCase):
    def test_os_stderr_decimal_errno_one_through_five_digits(self):
        for digits, accepted in ((0, False), (1, True), (5, True), (6, False)):
            text = "ERROR: private text (os error " + "1" * digits + ")"
            value = diagnostic.sanitized_stderr(text.encode())
            self.assertEqual(
                value,
                (
                    "ERROR: [redacted OS error text] (os error " + "1" * digits + ")"
                    if accepted
                    else "[redacted unrecognized stderr line]"
                ),
            )

    def test_exception_message_input_and_output_clamps_are_distinct(self):
        for length in (0, 1, 4096, 4097):
            with patch.object(
                diagnostic, "sanitized_stderr", return_value="inert"
            ) as scrub:
                value = diagnostic.error_chain(ValueError("x" * length))
                self.assertEqual(
                    scrub.call_args.args, (b"ERROR: " + b"x" * min(length, 4096),)
                )
                self.assertEqual(value["exceptions"][0]["message"], "inert")
        for length in (0, 1, 1024, 1025):
            with patch.object(
                diagnostic, "sanitized_stderr", return_value="x" * length
            ):
                value = diagnostic.error_chain(ValueError("inert"))
                self.assertEqual(
                    len(value["exceptions"][0]["message"]), min(length, 1024)
                )
        # Trusted sanitization doubles isolate two independent truncation stages;
        # the static error allowlist does not contain a 1024-character message.

    def test_relative_trace_location_effective_path_length(self):
        class TraceError(ValueError):
            @property
            def __traceback__(self):
                return self.trace

        root = Path(diagnostic.__file__).parents[2]
        for length, accepted in ((1, True), (112, True), (113, False)):
            error = TraceError("inert")
            error.trace = NS(
                tb_next=None,
                tb_lineno=1,
                tb_frame=NS(
                    f_code=NS(co_filename=str(root / ("x" * length)), co_name="inert")
                ),
            )
            frame = diagnostic.error_chain(error)["exceptions"][0]["frames"][0]
            self.assertEqual(
                frame["file"], "repo/" + "x" * length if accepted else "<external-code>"
            )
        # Path's complete relative name cannot be an empty string: the root itself
        # is represented as '.', whose one-character label is accepted.

    def test_watch_cleanup_wait_and_start_join_caps(self):
        for completed in (False, True):
            loop = profile_service.WatchdogLoop(Mock())
            loop.condition = Mock()
            loop.condition.__enter__ = Mock(return_value=None)
            loop.condition.__exit__ = Mock(return_value=False)
            loop.condition.wait_for.return_value = completed
            slot = profile_service.WatchSlot(loop, Mock(), 5)
            loop.slots = [slot]
            if completed:
                slot.close()
                self.assertFalse(loop.slots)
            else:
                with self.assertRaisesRegex(ValueError, "cleanup uncertain"):
                    slot.close()
                self.assertEqual(loop.slots, [slot])
            self.assertEqual(
                loop.condition.wait_for.call_args.kwargs, {"timeout": 0.15}
            )
        for ready in (False, True):
            loop = profile_service.WatchdogLoop(Mock())
            loop.ready = Mock(wait=Mock(return_value=ready))
            with patch.object(profile_service.threading, "Thread") as thread:
                if ready:
                    loop.start()
                else:
                    with self.assertRaisesRegex(ValueError, "startup unavailable"):
                        loop.start()
                thread.return_value.start.assert_called_once_with()
            loop.ready.wait.assert_called_once_with(0.5)
        for alive in (False, True):
            loop = profile_service.WatchdogLoop(Mock())
            loop.thread = Mock(is_alive=Mock(return_value=alive))
            if alive:
                with self.assertRaisesRegex(ValueError, "cleanup pending"):
                    loop.close()
            else:
                loop.close()
            loop.thread.join.assert_called_once_with(0.2)
        # These are fixed call arguments, not configurable elapsed-time domains.
        # Both successful and timed-out dispositions are exercised without threads.

    def service(self):
        service = object.__new__(profile_service.ProfileService)
        service.thread = None
        service.watchdog = Mock()
        service.ready = Mock()
        service.lock = threading.RLock()
        service.record = None
        service.supervisor = NS(busy=lambda: False, charged_snapshot_bytes=lambda: 0)
        service.owner_cleanup = None
        service.stopped = False
        service.wake = Mock()
        return service

    def test_owner_start_and_join_caps_preserve_uncertain_cleanup(self):
        for ready in (False, True):
            service = self.service()
            service.ready.wait.return_value = ready
            with patch.object(profile_service.threading, "Thread") as thread:
                if ready:
                    service.start_threads()
                else:
                    with self.assertRaisesRegex(ValueError, "startup unavailable"):
                        service.start_threads()
                thread.return_value.start.assert_called_once_with()
            service.ready.wait.assert_called_once_with(0.5)
        for alive in (False, True):
            service = self.service()
            service.thread = Mock(is_alive=Mock(return_value=alive))
            if alive:
                with self.assertRaisesRegex(ValueError, "cleanup pending"):
                    service.close()
                service.watchdog.close.assert_not_called()
            else:
                self.assertTrue(service.close())
                service.watchdog.close.assert_called_once_with()
            service.thread.join.assert_called_once_with(0.3)

    def test_watchdog_poll_cap_and_owner_wait_quanta(self):
        for active in (False, True):
            loop = profile_service.WatchdogLoop(Mock())
            loop.condition = Mock()
            loop.condition.__enter__ = Mock(return_value=None)
            loop.condition.__exit__ = Mock(return_value=False)
            loop.slots = [object()] if active else []
            loop.pulse = lambda: setattr(loop, "stopped", True)
            loop.run()
            loop.condition.wait.assert_called_once_with(0.05 if active else None)
            loop.meter.freeze.assert_called_once_with()
        for with_child in (False, True):
            service = self.service()
            service.owner_meter = Mock()
            service.shutdown_requested = NS(is_set=lambda: False)
            service.step = Mock()
            stop = lambda *_: setattr(service, "stopped", True)
            if with_child:
                child = NS(reaped=False, wait=Mock(side_effect=stop))
                service.record = {"runtime": NS(platform=NS(child=child))}
            else:
                service.wake.wait.side_effect = stop
            service.run()
            if with_child:
                child.wait.assert_called_once_with(0.05)
                service.wake.wait.assert_not_called()
            else:
                service.wake.wait.assert_called_once_with(0.25)
            service.owner_meter.freeze.assert_called_once_with()

    def test_owned_descriptor_select_wait_clamps_at_50_ms(self):
        for timeout in (
            0,
            math.nextafter(0.05, 0),
            0.05,
            math.nextafter(0.05, math.inf),
        ):
            child = object.__new__(profile_os.OwnedProcess)
            child.lock = threading.RLock()
            child.pidfd = 17
            child.eof = False
            child.stderr_eof = False
            child.stderr_ready = True
            child.process = NS(
                stdout=NS(fileno=lambda: 18), stderr=NS(fileno=lambda: 19)
            )
            with patch.object(profile_os.select, "select") as select:
                child.wait(timeout)
                select.assert_called_once_with([17, 18, 19], [], [], min(timeout, 0.05))
        # Negative values are outside the real caller contract; this method has
        # no independent negative-time validator and is not claimed to have one.

    def test_actual_config_load_byte_edges_before_closed_schema(self):
        example = (
            Path(__file__).resolve().parents[1] / "config/atlas-host.example.json"
        ).read_bytes()
        json.loads(example)
        for size, accepted in ((len(example), True), (16384, True), (16385, False)):
            raw = example + b" " * (size - len(example))
            with patch.object(Path, "open", return_value=io.BytesIO(raw)):
                if accepted:
                    result = config.load_config("/synthetic/config.json")
                    self.assertEqual(result["profile"], "local-v1")
                else:
                    with self.assertRaisesRegex(ValueError, "byte limit"):
                        config.load_config("/synthetic/config.json")
        for raw in (b"", b" "):
            with patch.object(Path, "open", return_value=io.BytesIO(raw)):
                with self.assertRaises(ValueError):
                    config.load_config("/synthetic/config.json")

    def test_legacy_open_latency_guard_and_http_timeout_without_top_level_work(self):
        path = Path(__file__).resolve().parents[1] / "tools/final_open_latency.py"
        source = ast.parse(path.read_text())
        functions = [
            n
            for n in source.body
            if isinstance(n, ast.FunctionDef) and n.name in {"guard", "get"}
        ]
        self.assertEqual([n.name for n in functions], ["guard", "get"])
        namespace = {
            "Path": Path,
            "minimum_ram": math.inf,
            "shutil": NS(disk_usage=Mock()),
            "a": NS(output=Path("/synthetic")),
            "urllib": NS(request=NS(urlopen=Mock())),
        }
        exec(
            compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
            namespace,
        )
        for ram, disk, accepted in (
            (3 * 1024**3 - 1024, 25 * 1024**3, False),
            (3 * 1024**3, 25 * 1024**3 - 1, False),
            (3 * 1024**3, 25 * 1024**3, True),
        ):
            namespace["shutil"].disk_usage.return_value = NS(free=disk)
            with patch.object(
                Path, "read_text", return_value=f"MemAvailable: {ram // 1024} kB"
            ):
                if accepted:
                    self.assertEqual(namespace["guard"](), ram)
                else:
                    with self.assertRaisesRegex(AssertionError, "reserve below"):
                        namespace["guard"]()
        response = Mock(headers={"inert": "value"})
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = b"inert"
        namespace["urllib"].request.urlopen.return_value = response
        self.assertEqual(
            namespace["get"]("inert:fixture"), (b"inert", {"inert": "value"})
        )
        namespace["urllib"].request.urlopen.assert_called_once_with(
            "inert:fixture", timeout=60
        )
        # Only two unchanged function definitions are compiled. Argument parsing,
        # model/reference hashing, resource changes, sockets and trials never run.


if __name__ == "__main__":
    unittest.main()
