"""Accepted boundaries and adjacent refusals using synthetic values and doubles."""

from contextlib import contextmanager
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import live_inference as live
import inference_observations as observation
import inference_sweep as sweep
from inference_edits import SOURCE_MODEL
from analytics import core
import observation_contracts
import profile_runtime_doubles
import sweep_contracts


class ReachedPins(Exception):
    """Request validation completed; model access is deliberately stopped."""


class ReviewLimits(unittest.TestCase):
    def generation(self, changes, accepted):
        session = live.Session("unused", Path("/unused"))
        with (
            patch.object(live, "available", return_value=6 * live.GIB),
            patch.object(live, "verify_model", side_effect=ReachedPins) as verify,
            patch.object(
                live.subprocess, "Popen", side_effect=AssertionError("No worker")
            ),
        ):
            try:
                session.start({"prompt": "x", **changes})
            except ReachedPins:
                self.assertTrue(accepted, "Invalid request reached model access")
                verify.assert_called_once_with(Path("/unused"))
            except ValueError:
                self.assertFalse(accepted, "Valid boundary was refused")
                verify.assert_not_called()
            else:
                self.fail("Expected a validation refusal or the stopped model check")
            self.assertIsNone(session.process)

    def test_generation_limits_both_ends_and_source_edit_requirement(self):
        cases = [
            ("prompt", "", False),
            ("prompt", " ", False),
            ("prompt", "x", True),
            ("prompt", "x" * 4096, True),
            ("prompt", "x" * 4097, False),
            ("prompt", "é" * 2048, True),
            ("prompt", "é" * 2048 + "x", False),
            ("max_new_tokens", 0, False),
            ("max_new_tokens", 1, True),
            ("max_new_tokens", 32, True),
            ("max_new_tokens", 33, False),
            ("layer", -1, False),
            ("layer", 0, True),
            ("layer", 29, True),
            ("layer", 30, False),
            ("activation_site", "block", True),
            ("activation_site", "attention", True),
            ("activation_site", "mlp", True),
            ("activation_site", "unknown", False),
        ]
        for field, value, accepted in cases:
            with self.subTest(field=field, value=value):
                self.generation({field: value}, accepted)
        self.generation({"source_model": SOURCE_MODEL}, False)
        self.generation({"source_model": SOURCE_MODEL, "edits": []}, True)
        limits = live.Session("unused", Path("/unused")).metadata()["limits"]
        self.assertEqual(limits["new_tokens"], 32)
        self.assertEqual(limits["worker_rss_mib"], 1536)
        self.assertEqual(limits["vector_width"], 576)
        self.assertEqual(limits["capture_layers"], list(range(30)))

    def test_generation_admission_accepts_minimum_refuses_one_byte_less(self):
        for available in (int(4.75 * live.GIB) - 1, int(4.75 * live.GIB)):
            session = live.Session("unused", Path("/unused"))
            with (
                self.subTest(available=available),
                patch.object(live, "available", return_value=available),
                patch.object(live, "verify_model", side_effect=ReachedPins) as verify,
                patch.object(
                    live.subprocess, "Popen", side_effect=AssertionError("No worker")
                ),
            ):
                if available == int(4.75 * live.GIB):
                    with self.assertRaises(ReachedPins):
                        session.start({"prompt": "x"})
                    verify.assert_called_once()
                else:
                    with self.assertRaisesRegex(ValueError, "Need 4.75 GiB"):
                        session.start({"prompt": "x"})
                    verify.assert_not_called()

    @contextmanager
    def simulated_worker(
        self,
        *,
        now=100,
        started=100,
        last_seen=100,
        available=6 * live.GIB,
        rss_kib=0,
        chunks=(),
    ):
        session = live.Session("unused", Path("/unused"))
        stream = SimpleNamespace(fileno=lambda: 42, close=lambda: None)
        worker = SimpleNamespace(
            pid=424242, stdin=None, stdout=stream, poll=lambda: None
        )
        session.process, session.status = worker, "running"
        session.started, session.last_seen = started, last_seen
        pending = iter(chunks)

        def read(_fd, _size):
            try:
                return next(pending)
            except StopIteration:
                raise BlockingIOError from None

        with (
            patch.object(live, "available", return_value=available),
            patch.object(live.time, "monotonic", return_value=now),
            patch.object(Path, "read_text", return_value=f"VmRSS:\t{rss_kib} kB\n"),
            patch.object(live.os, "read", side_effect=read),
            patch.object(live, "signal_and_reap", return_value=True) as reap,
            patch.object(
                live.subprocess, "Popen", side_effect=AssertionError("No worker")
            ),
        ):
            try:
                yield session, worker, reap
            finally:
                # A simulated PID must never reach real cleanup.
                session.process = None

    def test_runtime_time_memory_and_rss_adjacent_boundaries(self):
        cases = [
            ({"now": 120, "started": 0, "last_seen": 120}, None),
            (
                {"now": math.nextafter(120, math.inf), "started": 0, "last_seen": 120},
                "time_limit",
            ),
            ({"now": 15, "started": 15, "last_seen": 0}, None),
            (
                {"now": math.nextafter(15, math.inf), "started": 15, "last_seen": 0},
                "client_timeout",
            ),
            ({"available": int(3.25 * live.GIB)}, None),
            ({"available": int(3.25 * live.GIB) - 1}, "resource_limit"),
            ({"rss_kib": 1536 * 1024}, None),
            ({"rss_kib": 1536 * 1024 + 1}, "resource_limit"),
        ]
        for parameters, reason in cases:
            with (
                self.subTest(parameters=parameters),
                self.simulated_worker(**parameters) as (session, worker, reap),
            ):
                session.tick()
                if reason is None:
                    self.assertIs(session.process, worker)
                    self.assertEqual(session.status, "running")
                    reap.assert_not_called()
                else:
                    self.assertIsNone(session.process)
                    self.assertEqual(session.status, reason)
                    self.assertEqual(session.details["error"], reason)
                    reap.assert_called_once_with(worker)

    def test_partial_worker_buffer_accepts_128kib_refuses_one_byte_more(self):
        for size in (131072, 131073):
            with (
                self.subTest(size=size),
                self.simulated_worker(chunks=[b"x"]) as (session, worker, reap),
            ):
                session.buffer = b"x" * (size - 1)
                session.tick()
                if size == 131072:
                    self.assertEqual(len(session.buffer), size)
                    self.assertIs(session.process, worker)
                    reap.assert_not_called()
                else:
                    self.assertIsNone(session.process)
                    self.assertEqual(session.status, "error")
                    self.assertEqual(
                        session.details["error"], "Worker output exceeded bound"
                    )
                    reap.assert_called_once_with(worker)

    def test_activation_width_is_exact_not_an_upper_bound(self):
        for width in (575, 576, 577):
            event = {"type": "step", "index": 0, "activation": [0.0] * width}
            packet = json.dumps(event).encode() + b"\n"
            with (
                self.subTest(width=width),
                self.simulated_worker(chunks=[packet]) as (session, worker, reap),
            ):
                session.tick()
                if width == 576:
                    self.assertEqual(len(session.steps), 1)
                    self.assertEqual(len(session.steps[0]["activation"]), 576)
                    self.assertIs(session.process, worker)
                    reap.assert_not_called()
                else:
                    self.assertEqual(list(session.steps), [])
                    self.assertIsNone(session.process)
                    self.assertEqual(session.status, "error")
                    reap.assert_called_once_with(worker)

    def test_attention_group_mapping_all_query_heads(self):
        for head in range(9):
            step = observation_contracts.attention()
            step["attention"].update(query_head=head, kv_head=head // 3)
            with self.subTest(head=head):
                try:
                    observation.validate_record(
                        step, {"kind": "attention", "head": head}, 7
                    )
                except ValueError as error:
                    self.fail(f"Correct grouped-query mapping refused: {error}")
                wrong = deepcopy(step)
                wrong["attention"]["kv_head"] = (head // 3 + 1) % 3
                with self.assertRaisesRegex(ValueError, "geometry"):
                    observation.validate_record(
                        wrong, {"kind": "attention", "head": head}, 7
                    )

    def test_attention_probabilities_and_vocabulary_both_ends(self):
        for probabilities, accepted in (
            ([0, 0, 1], True),
            ([1, 0, 0], True),
            ([math.nextafter(0, -math.inf), 0, 1], False),
            ([math.nextafter(1, math.inf), 0, 0], False),
        ):
            step = observation_contracts.attention()
            step["attention"]["probabilities"] = probabilities
            with self.subTest(probabilities=probabilities):
                if accepted:
                    observation.validate_record(
                        step, {"kind": "attention", "head": 8}, 7
                    )
                else:
                    with self.assertRaisesRegex(ValueError, "probability"):
                        observation.validate_record(
                            step, {"kind": "attention", "head": 8}, 7
                        )
        for token, accepted in ((-1, False), (0, True), (49151, True), (49152, False)):
            step = observation_contracts.attention()
            step["attention"]["key_token_ids"][-1] = step["input_token_id"] = token
            with self.subTest(token=token):
                if accepted:
                    observation.validate_record(
                        step, {"kind": "attention", "head": 8}, 7
                    )
                else:
                    with self.assertRaisesRegex(ValueError, "key address"):
                        observation.validate_record(
                            step, {"kind": "attention", "head": 8}, 7
                        )

    def test_sweep_target_and_prompt_counts_at_both_ends(self):
        for field, counts in (("targets", range(4)), ("prompts", range(4))):
            for count in counts:
                request = sweep_contracts.request()
                request[field] = (
                    [
                        {"kind": "offset", "layer": 0, "heads": [0], "offset": i}
                        for i in range(count)
                    ]
                    if field == "targets"
                    else ["x"] * count
                )
                with self.subTest(field=field, count=count):
                    if count in (1, 2):
                        plan = sweep.build_plan(request, False)
                        self.assertEqual(
                            (
                                len(plan["targets"])
                                if field == "targets"
                                else plan["prompt_count"]
                            ),
                            count,
                        )
                    else:
                        with self.assertRaisesRegex(ValueError, "1–2"):
                            sweep.build_plan(request, False)
        self.assertEqual(sweep.schema()["subset_limits"]["targets"], 2)
        self.assertEqual(sweep.schema()["prompts"], 2)

    def test_analytics_region_value_limit_and_axis_limits(self):
        for rows, cols, accepted in (
            (256, 256, True),
            (257, 256, False),
            (1, 4096, True),
            (1, 4097, False),
            (0, 1, False),
            (1, 0, False),
        ):
            region = {"row": 0, "col": 0, "rows": rows, "cols": cols}
            with self.subTest(rows=rows, cols=cols):
                if accepted:
                    self.assertEqual(
                        core.geometry([8192, 8192], region), (0, 0, rows, cols)
                    )
                else:
                    with self.assertRaises(ValueError):
                        core.geometry([8192, 8192], region)
        # 65537 is prime, so no rectangle within the independent 4096-axis cap
        # contains exactly one value past 65536. Test the value-domain limit too.
        for count, accepted in ((0, True), (65536, True), (-1, False), (65537, False)):
            with self.subTest(count=count):
                if accepted:
                    self.assertEqual(core.integer(count, 0, 65536, "count"), count)
                else:
                    with self.assertRaises(ValueError):
                        core.integer(count, 0, 65536, "count")

    def test_profile_page_lease_expires_at_exact_deadline_without_watchdog_tick(self):
        fixture = profile_runtime_doubles.ServiceTests()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        grant, _ = fixture.start()
        fixture.service.finish_admission(grant)
        fixture.service.step()
        fixture.finish()
        status = fixture.service.status(fixture.owner)
        page = {
            **fixture.owner,
            "revision": status["accepted"]["revision"],
            "axis": "rows",
            "start": 0,
            "count": 2,
        }
        deadline = fixture.service.record["lease"]
        fixture.clock.now = math.nextafter(deadline, -math.inf)
        self.assertEqual(fixture.api.handle("page", page)[1]["end"], 2)
        for now in (deadline, math.nextafter(deadline, math.inf)):
            fixture.clock.now = now
            with (
                self.subTest(now=now),
                self.assertRaisesRegex(ValueError, "Profile unavailable"),
            ):
                fixture.api.handle("page", page)


if __name__ == "__main__":
    unittest.main()
