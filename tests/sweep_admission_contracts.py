"""OBS-1 fake-clock/read/signal/child controls only; no timer, worker or model."""

import hashlib
import io
from pathlib import Path
import resource
import signal
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import live_inference as live
import inference_worker as worker
from sweep_contracts import admitted


class BudgetTests(unittest.TestCase):
    def run_admission(
        self,
        *,
        wall_per_read=0,
        cpu_per_read=0,
        interrupt=False,
        corrupt=False,
        low_memory=False,
    ):
        clock = SimpleNamespace(wall=0.0, cpu=0.0)
        reads = []
        closed = []
        handlers = {}
        content = {"one": b"x", "two": b"y"}

        class File:
            def __init__(self, name):
                self.name = name
                self.count = 0

            def __enter__(self):
                return self

            def __exit__(self, *args):
                closed.append(self.name)

            def read(self, bound):
                assert bound == 1024 * 1024
                reads.append((self.name, clock.wall, clock.cpu))
                self.count += 1
                clock.wall += wall_per_read
                clock.cpu += cpu_per_read
                if interrupt:
                    handlers[signal.SIGALRM](signal.SIGALRM, None)
                return content[self.name] if self.count == 1 else b""

        class Directory:
            def __truediv__(self, name):
                return SimpleNamespace(open=lambda mode: File(name))

            def __str__(self):
                return "/synthetic-unused"

        class Stream(io.BytesIO):
            def close(self):
                pass

            def fileno(self):
                return 12345

        child = SimpleNamespace(stdin=Stream(), stdout=Stream(), poll=lambda: None)
        manifest = {
            "files": {
                name: hashlib.sha256(value).hexdigest()
                for name, value in content.items()
            }
        }
        if corrupt:
            manifest["files"]["two"] = "0" * 64
        previous = object()
        error = None
        with (
            patch.object(live, "MANIFEST", manifest),
            patch.object(live.time, "monotonic", side_effect=lambda: clock.wall),
            patch.object(live.time, "process_time", side_effect=lambda: clock.cpu),
            patch.object(
                live,
                "available",
                side_effect=lambda: (3 if low_memory and reads else 6) * live.GIB,
            ),
            patch.object(live.signal, "getitimer", return_value=(0.0, 0.0)),
            patch.object(live.signal, "getsignal", return_value=previous),
            patch.object(
                live.signal,
                "signal",
                side_effect=lambda sig, fn: handlers.update({sig: fn}),
            ) as install,
            patch.object(live.signal, "setitimer") as timers,
            patch.object(live.subprocess, "Popen", return_value=child) as spawn,
            patch.object(live.os, "set_blocking"),
            patch.object(live, "signal_and_reap", return_value=True) as reap,
        ):
            session = live.Session("unused", Directory())
            try:
                session.start(admitted()[0])
            except ValueError as exc:
                error = str(exc)
            argv = spawn.call_args.args[0] if spawn.called else None
            if session.process is not None:
                session.stop()
            self.assertEqual(handlers.get(signal.SIGALRM), previous)
            self.assertEqual(timers.call_args.args, (signal.ITIMER_REAL, 0))
            return SimpleNamespace(
                error=error,
                reads=reads,
                closed=closed,
                argv=argv,
                clock=clock,
                spawned=spawn.call_count,
                reaped=reap.call_count,
            )

    def test_wall_exhaustion_stops_before_another_read_or_file(self):
        result = self.run_admission(wall_per_read=61)
        self.assertIn("wall budget exhausted", result.error)
        self.assertEqual([r[1] for r in result.reads], [0.0, 61.0])
        self.assertEqual(result.closed, ["one"])
        self.assertEqual(result.spawned, 0)

    def test_cpu_exhaustion_spawns_nothing(self):
        result = self.run_admission(cpu_per_read=46)
        self.assertIn("CPU budget exhausted", result.error)
        self.assertEqual(len(result.reads), 2)
        self.assertEqual(result.spawned, 0)

    def test_admission_cpu_reduces_worker_allowance(self):
        result = self.run_admission(wall_per_read=1, cpu_per_read=7.5625)
        self.assertIsNone(result.error)
        self.assertEqual(result.clock.cpu, 30.25)
        self.assertEqual(result.argv[-2:], ["120.0", "59"])
        self.assertEqual(result.closed, ["one", "two"])
        self.assertEqual(result.spawned, 1)
        self.assertEqual(result.reaped, 1)

    def test_simulated_timer_interrupt_unwinds_a_blocked_read(self):
        result = self.run_admission(wall_per_read=121, interrupt=True)
        self.assertIn("wall budget exhausted", result.error)
        self.assertEqual(len(result.reads), 1)
        self.assertEqual(result.closed, ["one"])
        self.assertEqual(result.spawned, 0)

    def test_reserve_is_checked_at_read_boundaries(self):
        result = self.run_admission(low_memory=True)
        self.assertIn("memory reserve", result.error)
        self.assertEqual(len(result.reads), 1)
        self.assertEqual(result.spawned, 0)

    def test_every_pinned_hash_still_required(self):
        result = self.run_admission(corrupt=True)
        self.assertIn("Pinned file hash mismatch: two", result.error)
        self.assertEqual(result.closed, ["one", "two"])
        self.assertEqual(len(result.reads), 4)
        self.assertEqual(result.spawned, 0)

    def test_exhausted_before_verification_and_subsecond_cpu_refuse(self):
        for wall, cpu in [(121, 0), (0, 90)]:
            with (
                patch.object(live.time, "monotonic", return_value=wall),
                patch.object(live.time, "process_time", return_value=cpu),
                patch.object(live, "available", return_value=6 * live.GIB),
                patch.object(live.signal, "signal") as handler,
            ):
                with (
                    self.assertRaises(ValueError),
                    live.SweepAdmissionBudget(120, 90).verification(),
                ):
                    self.fail("must not enter verification")
                handler.assert_not_called()
        with (
            patch.object(live.time, "monotonic", return_value=0),
            patch.object(live.time, "process_time", return_value=89.25),
            patch.object(live, "available", return_value=6 * live.GIB),
        ):
            with self.assertRaisesRegex(ValueError, "Less than one"):
                live.SweepAdmissionBudget(120, 90).remaining_cpu()

    def test_existing_timer_is_never_replaced(self):
        with (
            patch.object(live.SweepAdmissionBudget, "check"),
            patch.object(live.signal, "getitimer", return_value=(1.0, 0.0)),
            patch.object(live.signal, "signal") as handler,
            patch.object(live.signal, "setitimer") as timer,
        ):
            with (
                self.assertRaisesRegex(ValueError, "existing deadline"),
                live.SweepAdmissionBudget(120, 90).verification(),
            ):
                self.fail("must not enter")
            handler.assert_not_called()
            timer.assert_not_called()

    def test_worker_cpu_and_address_limits_never_raise_inherited_bounds(self):
        for inherited, expected in [((-1, -1), 59), ((20, 30), 20), ((10, 10), 10)]:
            with (
                patch.object(worker.resource, "getrlimit", return_value=inherited),
                patch.object(worker.resource, "setrlimit") as limits,
            ):
                worker.configure_worker_limits(59)
                self.assertEqual(
                    limits.call_args.args, (resource.RLIMIT_CPU, (expected, expected))
                )
                for call in limits.call_args_list:
                    self.assertLessEqual(
                        call.args[1][0],
                        3 * 1024**3 if call.args[0] == resource.RLIMIT_AS else 59,
                    )
        for invalid in (True, 0, 91, 1.5):
            with self.assertRaises(ValueError):
                worker.configure_worker_limits(invalid)


if __name__ == "__main__":
    unittest.main()
