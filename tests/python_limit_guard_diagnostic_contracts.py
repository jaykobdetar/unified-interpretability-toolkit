"""Guard and diagnostic limits with inert process, clock and OS observations."""

import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import live_inference
from atlas_host import startup_diagnostics as diagnostic

GIB = 1024**3


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


build = load("boundary_build_guard", "tools/guarded-build.py")
hash_guard = load("boundary_hash_guard", "tools/check_model_hashes.py")


class GuardLimits(unittest.TestCase):
    def build_case(self, memory, disk):
        with (
            patch.object(sys, "argv", ["guard", "build", "--release"]),
            patch.object(Path, "is_file", return_value=True),
            patch.object(
                Path, "read_text", return_value=f"MemAvailable: {memory // 1024} kB\n"
            ),
            patch.object(build.shutil, "which", return_value="cargo"),
            patch.object(build.shutil, "disk_usage", return_value=NS(free=disk)),
            patch.object(build.os, "sched_getaffinity", return_value={2, 4}),
            patch.object(build.os, "sched_setaffinity") as affinity,
            patch.object(build.os, "nice") as nice,
            patch.object(build.resource, "setrlimit") as limits,
            patch.object(build.subprocess, "call", return_value=0) as call,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            result = build.main()
            self.assertEqual(result, 0)
            affinity.assert_called_once_with(0, {2})
            nice.assert_called_once_with(10)
            limits.assert_called_once_with(build.resource.RLIMIT_AS, (2 * GIB, 2 * GIB))
            call.assert_called_once_with(
                ["cargo", "build", "--offline", "--locked", "-j", "1", "--release"]
            )

    def test_build_admission_ram_disk_and_fixed_os_caps(self):
        self.build_case(5 * GIB, 25 * GIB)
        # /proc meminfo uses KiB, so the closest lower RAM observation is 1 KiB.
        for memory, disk in ((5 * GIB - 1024, 25 * GIB), (5 * GIB, 25 * GIB - 1)):
            with self.assertRaises(SystemExit) as error:
                self.build_case(memory, disk)
            self.assertEqual(error.exception.code, 2)

    def test_hash_read_admission_three_gib(self):
        for memory, accepted in ((3 * GIB, True), (3 * GIB - 1024, False)):
            with patch.object(
                Path, "read_text", return_value=f"MemAvailable: {memory // 1024} kB\n"
            ):
                if accepted:
                    hash_guard.guard()
                else:
                    with self.assertRaisesRegex(RuntimeError, "fewer than 3 GiB"):
                        hash_guard.guard()

    def execute_guard(self, browser, admission, remaining, rss, elapsed):
        process = NS(
            pid=4242,
            returncode=0,
            poll=Mock(side_effect=[None, 0, 0]),
            kill=Mock(),
            terminate=Mock(),
            wait=Mock(return_value=0),
        )
        pages = rss // 4096
        fields = ["S", "1"] + ["0"] * 19 + [str(pages)]
        proc_stat = NS(
            parent=NS(name="4242"), read_text=lambda: "4242 (inert) " + " ".join(fields)
        )
        times = iter([0.0, elapsed])
        clock = lambda: next(times, elapsed)
        with (
            patch.object(sys, "argv", ["guard", "inert-test"]),
            patch.object(
                live_inference, "available", side_effect=[admission, 6 * GIB, remaining]
            ),
            patch("os.sched_getaffinity", return_value={2}),
            patch("os.sched_setaffinity"),
            patch("os.sysconf", return_value=4096),
            patch("subprocess.Popen", return_value=process) as spawn,
            patch("time.monotonic", side_effect=clock),
            patch("time.sleep"),
            patch("os.kill") as kill,
            patch.object(Path, "glob", return_value=[proc_stat]),
            patch.object(Path, "read_text", return_value=f"VmRSS: {rss // 1024} kB\n"),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            try:
                runpy.run_path(
                    str(
                        ROOT
                        / (
                            "tools/guarded-browser-test.py"
                            if browser
                            else "tools/guarded-inference-test.py"
                        )
                    ),
                    run_name="__main__",
                )
            except (SystemExit, RuntimeError) as error:
                return error, process, spawn, kill
        self.fail("guard must exit or refuse")

    def test_legacy_guard_admission_stop_rss_and_wall_edges(self):
        for browser in (False, True):
            admission = 5 * GIB if browser else 19 * GIB // 4
            cap = 768 * 1024**2 if browser else 3 * GIB // 2
            quantum = 4096 if browser else 1024
            for memory, remaining, rss, elapsed, accepted in (
                (admission, 13 * GIB // 4, cap, 120.0, True),
                (admission - 1, 6 * GIB, 0, 0, False),
                (admission, 13 * GIB // 4 - 1, 0, 0, False),
                (admission, 6 * GIB, cap + quantum, 0, False),
                (admission, 6 * GIB, 0, math.nextafter(120.0, math.inf), False),
            ):
                with self.subTest(
                    browser=browser, memory=memory, rss=rss, elapsed=elapsed
                ):
                    error, process, spawn, kill = self.execute_guard(
                        browser, memory, remaining, rss, elapsed
                    )
                    if accepted:
                        self.assertIsInstance(error, SystemExit)
                        self.assertEqual(error.code, 0)
                    elif memory < admission:
                        self.assertIsInstance(error, SystemExit)
                        spawn.assert_not_called()
                        process.wait.assert_not_called()
                    else:
                        self.assertIsInstance(error, RuntimeError)
                        process.wait.assert_called_once_with()
                    kill.assert_not_called()  # No descendant exists in this double.


class DiagnosticLimits(unittest.TestCase):
    def test_event_site_length_and_signed_fact_filter(self):
        for length, accepted in ((0, False), (1, True), (80, True), (81, False)):
            collector = diagnostic.StartupDiagnostics()
            if accepted:
                collector.event("x" * length)
                self.assertEqual(len(collector.events), 1)
            else:
                with self.assertRaises(ValueError):
                    collector.event("x" * length)
        for number, accepted in (
            (-(2**63) - 1, False),
            (-(2**63), True),
            (2**63 - 1, True),
            (2**63, False),
        ):
            collector = diagnostic.StartupDiagnostics()
            collector.event("fixture", rss_bytes=number)
            facts = json.loads(collector.events[0]["json"])["facts"]
            self.assertEqual("rss_bytes" in facts, accepted)

    def test_event_count_deduplication_and_exact_byte_charge(self):
        collector = diagnostic.StartupDiagnostics()
        for number in range(64):
            collector.event(f"site{number}")
        self.assertEqual(len(collector.events), 64)
        collector.event("site63")
        self.assertEqual(collector.events[-1]["count"], 2)
        collector.event("one_past")
        self.assertEqual((len(collector.events), collector.events_dropped), (64, 1))
        probe = diagnostic.StartupDiagnostics()
        probe.event("edge")
        cost = len(probe.events[0]["json"])
        # Isolate the cumulative charge guard: bounded scalar events cannot
        # naturally reach 512 KiB within the separate 64-record cap.
        for left, accepted in ((cost, True), (cost - 1, False)):
            collector = diagnostic.StartupDiagnostics()
            collector.event_bytes = 524288 - left
            collector.event("edge")
            self.assertEqual(len(collector.events), int(accepted))

    def test_stderr_prefix_byte_count_and_exception_inventory(self):
        for size in (0, 16384, 16385):
            record = diagnostic.StartupRecord(19)
            record.feed(b"x" * size)
            value = record.snapshot()
            self.assertEqual(value["stderr_prefix_bytes"], min(size, 16384))
            self.assertEqual(value["stderr_bytes_seen"], size)
            self.assertEqual(value["stderr_truncated"], size > 16384)
        for count in (1, 8, 9):
            nodes = [ValueError("inert") for _ in range(count)]
            for left, right in zip(nodes, nodes[1:]):
                left.__cause__ = right
            chain = diagnostic.error_chain(nodes[0])
            self.assertEqual(len(chain["exceptions"]), min(count, 8))
            self.assertEqual(chain["chain_truncated"], count > 8)


if __name__ == "__main__":
    unittest.main()
