#!/usr/bin/env python3
"""Pure guard control-flow regressions: every process, signal, clock and I/O mocked."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location(
    "core_guard", Path(__file__).resolve().parents[1] / "tools/guarded-core-ui.py"
)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class Output:
    def __init__(self, fail=False):
        self.receipts = []
        self.fail = fail

    def mkdir(self, **_):
        pass

    def __truediv__(self, _):
        return self

    def write_text(self, text):
        if self.fail:
            raise OSError("mock receipt device failure")
        self.receipts.append(json.loads(text))


class Simulation:
    def __init__(self, mode, total_rss=None, available_values=None):
        self.mode = mode
        self.total_rss = total_rss
        self.available_values = available_values
        self.now = 0.0
        self.returncode = None
        self.pid = 2
        self.driver_done = False
        self.descendant_done = False
        self.finish_at = None
        self.reused = False
        self.signals = []
        self.waits = []
        self.memory_reads = 0

    def available(self):
        self.memory_reads += 1
        if self.available_values is not None:
            return self.available_values[
                min(self.memory_reads - 1, len(self.available_values) - 1)
            ]
        if self.mode.startswith("wait_timeout") and self.memory_reads > 1:
            return 2 * 1024**3
        return (4 if self.mode == "launch_refused" else 6) * 1024**3

    def poll(self):
        if not self.mode.startswith("wait_timeout") and self.now >= 0.1:
            self.driver_done = True
        if self.driver_done:
            self.returncode = 0
        return self.returncode

    def wait(self, timeout):
        self.waits.append(timeout)
        if self.mode.startswith("wait_timeout"):
            if self.mode == "wait_timeout_survives" or len(self.waits) == 1:
                self.now += timeout
                raise subprocess.TimeoutExpired("mock direct child", timeout)
            self.driver_done = True
            self.returncode = -9
        return self.returncode

    def sleep(self, duration):
        self.now += duration

    def rows(self):
        result = {1: (0, 10, 1024**2)}
        if not self.driver_done:
            result[2] = (1, 20, 1024**2)
        ended = self.descendant_done or (
            self.finish_at is not None and self.now >= self.finish_at
        )
        if self.mode == "clean" and self.now >= 0.1:
            ended = True
        if self.reused:
            result[3] = (999, 999, 1024**2)  # Unrelated replacement PID.
        elif not ended:
            result[3] = (2, 30, 1024**2)
        if self.total_rss is not None:
            result[1] = (
                0,
                10,
                self.total_rss - sum(row[2] for pid, row in result.items() if pid != 1),
            )
        return result

    def kill(self, pid, kind):
        # Assert at the mock OS boundary that neither guard nor unrelated PID is signaled.
        rows = self.rows()
        assert pid in (2, 3) and rows[pid][1] == {2: 20, 3: 30}[pid]
        self.signals.append((pid, int(kind)))
        if pid == 3:
            if self.mode == "survivor":
                return
            if self.mode == "slow":
                self.finish_at = self.now + 0.25
            elif self.mode == "kill_fallback":
                self.descendant_done = kind == signal.SIGKILL
            elif self.mode == "kill_lag":
                if kind == signal.SIGKILL:
                    self.finish_at = self.now + 0.25
            elif self.mode == "pid_reuse":
                self.reused = True
            else:
                self.descendant_done = True


def exercise(
    mode,
    cleanup_error=False,
    file_error=False,
    total_rss=None,
    available_values=None,
    disk_free=30 * 1024**3,
):
    sim, out, stdout = (
        Simulation(mode, total_rss, available_values),
        Output(file_error),
        io.StringIO(),
    )
    with contextlib.ExitStack() as stack:
        for name, replacement in [("table", sim.rows), ("available", sim.available)]:
            stack.enter_context(patch.object(guard, name, replacement))
        stack.enter_context(patch.object(guard.os, "getpid", return_value=1))
        stack.enter_context(
            patch.object(guard.os, "sched_getaffinity", return_value={0})
        )
        stack.enter_context(patch.object(guard.os, "sched_setaffinity"))
        stack.enter_context(patch.object(guard.os, "nice"))
        stack.enter_context(patch.object(guard.os, "kill", sim.kill))
        stack.enter_context(patch.object(guard.time, "monotonic", lambda: sim.now))
        stack.enter_context(patch.object(guard.time, "sleep", sim.sleep))
        stack.enter_context(
            patch.object(
                guard.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=disk_free),
            )
        )
        spawn = stack.enter_context(
            patch.object(guard.subprocess, "Popen", return_value=sim)
        )
        if cleanup_error:
            stack.enter_context(
                patch.object(
                    guard, "cleanup", side_effect=RuntimeError("mock cleanup exception")
                )
            )
        stack.enter_context(contextlib.redirect_stdout(stdout))
        exit_code = guard.run(["mock-command"], out)
    return SimpleNamespace(
        exit_code=exit_code,
        receipt=json.loads(stdout.getvalue()),
        writes=out.receipts,
        signals=sim.signals,
        waits=sim.waits,
        elapsed=sim.now,
        spawn_count=spawn.call_count,
    )


class GuardContracts(unittest.TestCase):
    def test_twenty_five_gib_disk_boundary_refuses_before_spawn(self):
        refused = exercise("clean", disk_free=25 * 1024**3 - 1)
        self.assertEqual(refused.exit_code, 1)
        self.assertEqual(refused.spawn_count, 0)
        self.assertEqual(refused.signals, [])
        self.assertIn("25 GiB disk reserve", refused.receipt["failure"])
        accepted = exercise("clean", disk_free=25 * 1024**3)
        self.assertEqual(accepted.exit_code, 0)
        self.assertEqual(accepted.spawn_count, 1)

    def test_kill_fallback_allows_bounded_delayed_cleanup(self):
        result = exercise("kill_lag")
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.signals, [(3, signal.SIGTERM), (3, signal.SIGKILL)])
        self.assertTrue(result.receipt["cleanup_verified"])
        self.assertEqual(result.receipt["remaining_owned_pids"], [])
        self.assertGreaterEqual(result.elapsed, 2.25)
        self.assertLess(result.elapsed, 5)

    def test_approved_browser_rss_boundary_is_strictly_above_one_gib(self):
        self.assertEqual(guard.BROWSER_RSS_CAP_BYTES, 1024**3)
        for total in (768 * 1024**2 + 1, 1024**3 - 1, 1024**3):
            with self.subTest(total=total):
                r = exercise("clean", total_rss=total)
                self.assertEqual(r.exit_code, 0)
                self.assertEqual(r.receipt["rss_cap_mib"], 1024)
                self.assertEqual(r.receipt["owned_tree_peak_rss_mib"], total / 1024**2)
                self.assertEqual(
                    r.receipt["browser_resource_policy"], guard.BROWSER_POLICY
                )
        r = exercise("clean", total_rss=1024**3 + 1)
        self.assertEqual(r.exit_code, 1)
        self.assertIn("Qualification guard refused", r.receipt["failure"])
        self.assertTrue(r.receipt["cleanup_verified"])
        self.assertEqual(r.receipt["remaining_owned_pids"], [])

    def test_five_gib_launch_gate_is_unchanged(self):
        r = exercise("clean", available_values=[5 * 1024**3 - 1])
        self.assertEqual(r.exit_code, 1)
        self.assertEqual(r.spawn_count, 0)
        r = exercise("clean", available_values=[5 * 1024**3])
        self.assertEqual(r.exit_code, 0)
        self.assertEqual(r.spawn_count, 1)

    def test_three_point_two_five_gib_stop_reserve_is_unchanged(self):
        boundary = int(3.25 * 1024**3)
        r = exercise("clean", available_values=[6 * 1024**3, boundary])
        self.assertEqual(r.exit_code, 0)
        r = exercise("clean", available_values=[6 * 1024**3, boundary - 1])
        self.assertEqual(r.exit_code, 1)
        self.assertIn("Qualification guard refused", r.receipt["failure"])
        self.assertTrue(r.receipt["cleanup_verified"])
        self.assertEqual(r.receipt["remaining_owned_pids"], [])

    def test_clean_exit_passes_with_verified_empty_remainder(self):
        r = exercise("clean")
        self.assertEqual(r.exit_code, 0)
        self.assertTrue(r.receipt["passed"])
        self.assertTrue(r.receipt["cleanup_verified"])
        self.assertEqual(r.receipt["remaining_owned_pids"], [])
        self.assertEqual(r.signals, [])
        self.assertEqual(r.writes, [r.receipt])

    def test_slow_descendant_gets_bounded_grace_before_success(self):
        r = exercise("slow")
        self.assertEqual(r.exit_code, 0)
        self.assertEqual(r.signals, [(3, signal.SIGTERM)])
        self.assertEqual(r.receipt["remaining_owned_pids"], [])
        self.assertLess(r.elapsed, 2)

    def test_term_ignoring_descendant_gets_identity_checked_kill_fallback(self):
        r = exercise("kill_fallback")
        self.assertEqual(r.exit_code, 0)
        self.assertEqual(r.signals, [(3, signal.SIGTERM), (3, signal.SIGKILL)])
        self.assertEqual(r.receipt["remaining_owned_pids"], [])
        self.assertLess(r.elapsed, 5)

    def test_surviving_descendant_cannot_report_success(self):
        r = exercise("survivor")
        self.assertEqual(r.exit_code, 1)
        self.assertFalse(r.receipt["passed"])
        self.assertEqual(r.receipt["remaining_owned_pids"], [3])
        self.assertIn("Owned processes remain", r.receipt["failure"])
        self.assertEqual(r.writes, [r.receipt])
        self.assertLess(r.elapsed, 5)

    def test_wait_timeout_retains_original_failure_and_always_writes_receipt(self):
        r = exercise("wait_timeout")
        self.assertEqual(r.exit_code, 1)
        self.assertIn("Qualification guard refused", r.receipt["failure"])
        self.assertIn("Direct-child reap failed: TimeoutExpired", r.receipt["failure"])
        self.assertEqual(r.receipt["remaining_owned_pids"], [])
        self.assertEqual(r.waits, [1.0, 1.0])
        self.assertEqual(r.writes, [r.receipt])
        self.assertLess(r.elapsed, 7)

    def test_both_waits_timeout_still_report_owned_survivor_and_failure(self):
        r = exercise("wait_timeout_survives")
        self.assertEqual(r.exit_code, 1)
        self.assertIn("Fallback reap failed: TimeoutExpired", r.receipt["failure"])
        self.assertEqual(r.receipt["remaining_owned_pids"], [2])
        self.assertEqual(r.writes, [r.receipt])
        self.assertLess(r.elapsed, 7)

    def test_reused_pid_is_not_signaled_again(self):
        r = exercise("pid_reuse")
        self.assertEqual(r.exit_code, 0)
        self.assertEqual(r.signals, [(3, signal.SIGTERM)])
        self.assertEqual(r.receipt["remaining_owned_pids"], [])

    def test_unexpected_cleanup_exception_cannot_skip_failure_receipt(self):
        r = exercise("clean", cleanup_error=True)
        self.assertEqual(r.exit_code, 1)
        self.assertFalse(r.receipt["cleanup_verified"])
        self.assertIn("Unexpected cleanup failure", r.receipt["failure"])
        self.assertEqual(r.writes, [r.receipt])

    def test_launch_gate_refusal_writes_receipt_without_spawning(self):
        r = exercise("launch_refused")
        self.assertEqual(r.exit_code, 1)
        self.assertEqual(r.spawn_count, 0)
        self.assertEqual(r.signals, [])
        self.assertIn("launch refused", r.receipt["failure"])
        self.assertEqual(r.writes, [r.receipt])

    def test_receipt_file_error_emits_failure_json_fallback(self):
        r = exercise("clean", file_error=True)
        self.assertEqual(r.exit_code, 1)
        self.assertFalse(r.receipt["passed"])
        self.assertIn("Receipt write failed", r.receipt["failure"])
        self.assertEqual(r.writes, [])


if __name__ == "__main__":
    unittest.main()
