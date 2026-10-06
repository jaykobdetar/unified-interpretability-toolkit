"""Request and finalization guards with in-memory headers and inert owners."""

from email.message import Message
import math
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import live_inference as live
import core_guard_contracts as core_fixture
from atlas_host import dense_static_admission as dense
import python_limit_numeric_contracts as numeric_fixture
from analytics import svd_summary as summary


class RequestFinalizationLimits(unittest.TestCase):
    def handler(self, *, path="/", length=None):
        handler = object.__new__(live.Handler)
        handler.path = path
        handler.server = NS(server_port=8775)
        handler.headers = Message()
        handler.headers["Host"] = "127.0.0.1:8775"
        if length is not None:
            handler.headers["Content-Length"] = str(length)
        return handler

    def test_validated_body_path_and_header_accounting_edges(self):
        self.assertEqual((live.MAX_BODY, live.REQUEST_DEADLINE), (8192, 0.5))
        self.assertEqual(self.handler().validated(), ("/", 0))
        for length, accepted in ((-1, False), (0, True), (8192, True), (8193, False)):
            handler = self.handler(length=length)
            if accepted:
                self.assertEqual(handler.validated()[1], length)
            else:
                with self.assertRaisesRegex(ValueError, "body exceeds"):
                    handler.validated()
        for size, accepted in ((0, False), (1, True), (8192, True), (8193, False)):
            handler = self.handler(path="/" + "a" * (size - 1) if size else "")
            if accepted:
                self.assertEqual(len(handler.validated()[0]), size)
            else:
                with self.assertRaisesRegex(ValueError, "local URL"):
                    handler.validated()
        for size, accepted in ((8192, True), (8193, False)):
            handler = self.handler(length=0)
            existing = sum(len(k) + len(v) for k, v in handler.headers.items())
            handler.headers["X-Padding"] = "x" * (size - existing - 9)
            self.assertEqual(
                sum(len(k) + len(v) for k, v in handler.headers.items()), size
            )
            if accepted:
                handler.validated()
            else:
                with self.assertRaisesRegex(ValueError, "Headers exceed"):
                    handler.validated()
        # These call the pure validated() guard. A complete received request
        # adds method, version and headers, so a path of 8192 cannot also fit
        # the independently bounded 8192-byte complete header envelope.

    def test_content_length_cardinality_zero_one_two(self):
        for count, accepted in ((0, True), (1, True), (2, False)):
            handler = self.handler()
            for _ in range(count):
                handler.headers["Content-Length"] = "0"
            if accepted:
                self.assertEqual(handler.validated()[1], 0)
            else:
                with self.assertRaisesRegex(ValueError, "framing"):
                    handler.validated()

    def test_receive_header_body_and_absolute_deadline_with_inert_socket(self):
        def request(header_size, body_size):
            prefix = f"GET / HTTP/1.0\r\nHost: 127.0.0.1:8775\r\nContent-Length: {body_size}\r\nX-Padding: ".encode()
            return (
                prefix
                + b"x" * (header_size - len(prefix) - 4)
                + b"\r\n\r\n"
                + b"x" * body_size
            )

        for header, body, now, accepted in (
            (8192, 0, 0, True),
            (8192, 8192, 0, True),
            (8193, 0, 0, False),
            (256, 8193, 0, False),
            (256, 0, math.nextafter(0.5, 0.0), True),
            (256, 0, 0.5, False),
        ):
            raw = bytearray(request(header, body))
            original = bytes(raw)

            def recv(count):
                self.assertEqual(count, 4096)
                part = bytes(raw[:count])
                del raw[:count]
                return part

            handler = object.__new__(live.Handler)
            handler.connection = NS(settimeout=Mock(), recv=recv)
            first = True

            def clock():
                nonlocal first
                value = 0 if first else now
                first = False
                return value

            with (
                patch.object(live.time, "monotonic", side_effect=clock),
                patch.object(live.BaseHTTPRequestHandler, "handle") as dispatch,
            ):
                handler.handle()
                self.assertEqual(dispatch.call_count, int(accepted))
                if accepted:
                    self.assertEqual(handler.rfile.read(), original)
        # With a conforming recv(4096), accepted header/body caps cannot produce
        # a complete frame over 16384 bytes. That aggregate guard is redundant.

    def test_signal_reap_preserves_bounded_wait_and_uncertain_ownership(self):
        for terminate in (False, True):
            for timeout in (0.2, 0.5, 1.5):
                process = NS(terminate=Mock(), kill=Mock(), wait=Mock())
                with patch.object(live, "process_alive", return_value=True):
                    self.assertTrue(
                        live.signal_and_reap(
                            process, terminate=terminate, timeout=timeout
                        )
                    )
                    process.wait.assert_called_once_with(timeout=timeout)
                    (
                        process.terminate if terminate else process.kill
                    ).assert_called_once_with()
                    process.wait.side_effect = subprocess.TimeoutExpired(
                        "inert", timeout
                    )
                    self.assertFalse(live.signal_and_reap(process, timeout=timeout))

    def test_core_guard_exact_120_second_observation(self):
        original = core_fixture.Simulation
        for elapsed, accepted in (
            (120.0, True),
            (math.nextafter(120.0, math.inf), False),
        ):
            simulation = original("clean")
            observations = 0

            def poll():
                nonlocal observations
                observations += 1
                if observations <= 2:
                    return None
                simulation.driver_done = True
                simulation.returncode = 0
                return 0

            def sleep(duration):
                simulation.now = (
                    elapsed if simulation.now == 0 else simulation.now + duration
                )

            simulation.poll = poll
            simulation.sleep = sleep
            with patch.object(core_fixture, "Simulation", return_value=simulation):
                result = core_fixture.exercise("clean")
            self.assertEqual(result.exit_code == 0, accepted)
            self.assertTrue(result.receipt["cleanup_verified"])
            self.assertFalse(result.receipt["remaining_owned_pids"])

    def test_core_cleanup_term_and_kill_deadline_neighbors(self):
        guard = core_fixture.guard
        for stage in ("term", "kill"):
            deadline = 2.0 if stage == "term" else 4.0
            for end, accepted in (
                (math.nextafter(deadline, 0.0), True),
                (deadline, True),
                (math.nextafter(deadline, math.inf), False),
            ):
                sim = core_fixture.Simulation("survivor")
                sim.driver_done = True
                sim.returncode = 0

                def kill(pid, kind):
                    sim.signals.append((pid, int(kind)))
                    if pid == 3:
                        if (
                            stage == "term"
                            and kind == signal.SIGTERM
                            or stage == "kill"
                            and kind == signal.SIGKILL
                        ):
                            sim.finish_at = end
                        elif stage == "term" and kind == signal.SIGKILL:
                            sim.descendant_done = True

                with (
                    patch.object(guard, "table", sim.rows),
                    patch.object(guard.os, "kill", side_effect=kill),
                    patch.object(guard.time, "monotonic", side_effect=lambda: sim.now),
                    patch.object(guard.time, "sleep", side_effect=sim.sleep),
                ):
                    result = guard.cleanup(sim, {1: 10, 2: 20, 3: 30}, 1)
                self.assertTrue(result["cleanup_verified"])
                self.assertEqual(sim.waits, [1.0])
                if stage == "term":
                    self.assertEqual(
                        any(kind == signal.SIGKILL for _, kind in sim.signals),
                        not accepted,
                    )
                    self.assertFalse(result["remaining_owned_pids"])
                else:
                    self.assertEqual(bool(result["remaining_owned_pids"]), not accepted)

    def test_fixed_dense_topology_numeric_identity_neighbors(self):
        entry = {"synthetic": True}
        policy = object.__new__(dense.BoundDenseStaticAdmission)
        object.__setattr__(policy, "check", Mock(return_value=entry))
        for count, total, accepted in (
            (271, 134515008, False),
            (272, 134515008, True),
            (273, 134515008, False),
            (272, 134515007, False),
            (272, 134515009, False),
        ):
            prepared = NS(
                entry=entry,
                descriptor={"model_type": "llama", "parameter_count": total},
                tensors={str(n): {"dtype": "BF16"} for n in range(count)},
                check=Mock(),
            )
            if accepted:
                self.assertTrue(policy.admit(prepared))
                prepared.check.assert_called_once()
            else:
                with self.assertRaisesRegex(ValueError, "topology required"):
                    policy.admit(prepared)
                prepared.check.assert_not_called()
        # Sealed owner checks are doubled solely to isolate topology claims.
        # This cannot create or enable a real dense admission policy.

    def test_svd_accounting_relative_and_absolute_float_neighbors(self):
        helper = numeric_fixture.NumericLimits(methodName="runTest")
        for ratio, accepted in (
            (1e-12, True),
            (math.nextafter(1e-12, math.inf), False),
        ):
            report, kwargs = helper.report()
            report["results"]["original"]["rank_one_residual_energy_fraction"] = ratio
            if accepted:
                summary.validate(report, **kwargs)
            else:
                with self.assertRaisesRegex(ValueError, "accounting differs"):
                    summary.validate(report, **kwargs)
        lower, upper = 1 - 1e-10, 1 + 1e-10
        for fraction, accepted in (
            (math.nextafter(lower, math.inf), True),
            (lower, False),
            (math.nextafter(upper, -math.inf), True),
            (upper, False),
        ):
            report, kwargs = helper.report()
            report["results"]["original"]["energy_fractions"] = [fraction]
            if accepted:
                summary.validate(report, **kwargs)
            else:
                with self.assertRaises(ValueError):
                    summary.validate(report, **kwargs)
        # Literal decimal tolerance endpoints round slightly outside isclose;
        # these adjacent representable vectors preserve actual float behavior.

    def test_svd_report_body_guard_isolated_from_closed_fields(self):
        helper = numeric_fixture.NumericLimits(methodName="runTest")
        report, kwargs = helper.report()
        dumps = summary.json.dumps
        for size, accepted in ((63488, True), (63489, False)):

            def encode(value, *args, **options):
                return "x" * size if value is report else dumps(value, *args, **options)

            with patch.object(summary.json, "dumps", side_effect=encode):
                if accepted:
                    summary.validate(report, **kwargs)
                else:
                    with self.assertRaisesRegex(ValueError, "body cap"):
                        summary.validate(report, **kwargs)
        # Selective encoding double tests the trusted guard, not a claim that
        # arbitrary padding is legal in the closed SVD report schema.


if __name__ == "__main__":
    unittest.main()
