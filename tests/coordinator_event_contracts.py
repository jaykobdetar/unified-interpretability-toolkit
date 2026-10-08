"""Coordinator event stages using the existing inert worker and cleanup doubles."""

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
import json
from typing import Any
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import live_inference as live
from review_limit_contracts import ReviewLimits


class CoordinatorEventContracts(unittest.TestCase):
    @contextmanager
    def worker(
        self, events: list[dict[str, Any]], mode: str = "generation"
    ) -> Iterator[tuple[live.Session, Any, Mock, SimpleNamespace]]:
        fixture = ReviewLimits(methodName="runTest")
        chunks = [(json.dumps(event) + "\n").encode() for event in events]
        contracts = SimpleNamespace(
            architecture=SimpleNamespace(width=3),
            validate_pair=Mock(),
            validate_sweep_step=Mock(),
            validate_pair_step=Mock(),
            validate_record=Mock(),
            validate_preview=Mock(),
            stops=[],
        )

        def coverage(
            plan: dict[str, Any], count: int, current: object = None
        ) -> dict[str, int]:
            self.assertEqual(plan, {"records": 2})
            self.assertIsNone(current)
            return {"completed": count}

        with (
            fixture.simulated_worker(chunks=chunks, rss_kib=2048) as (
                session,
                process,
                reap,
            ),
            patch.object(live, "_contracts", return_value=contracts),
            patch.object(live.sweep, "coverage", side_effect=coverage),
            patch.object(live.sweep, "record_ids", return_value=["a", "b"]),
        ):
            session.mode = mode
            session.minimum_available = 7 * live.GIB
            session.sweep_plan = {"records": 2} if mode == "sweep" else None
            session.pair_request = {"positions": [0, 1]}
            session.capture_layer = 2
            session.observation = {"kind": "fixture"}
            original_stop = session.stop

            def stop(reason: str = "cancelled") -> bool:
                contracts.stops.append(
                    (session.status, deepcopy(session.details), reason)
                )
                return original_stop(reason)

            with patch.object(session, "stop", side_effect=stop):
                yield session, process, reap, contracts

    def test_step_routing_and_exact_forwarded_values(self) -> None:
        for mode in ("generation", "comparison", "prompt_pair", "sweep"):
            event = {
                "type": "step",
                "index": 0,
                "activation": [1, 2, 3],
                "baseline": {"fixture": 1},
                "edited": {"fixture": 2},
            }
            expected = {key: value for key, value in event.items() if key != "type"}
            with (
                self.subTest(mode=mode),
                self.worker([event], mode) as (
                    session,
                    process,
                    reap,
                    contracts,
                ),
            ):
                session.tick()
                self.assertEqual(list(session.steps), [expected])
                self.assertEqual(session.status, "running")
                self.assertIs(session.process, process)
                reap.assert_not_called()
                self.assertEqual(session.peak_rss, 2 * 1024**2)
                self.assertEqual(session.minimum_available, 6 * live.GIB)
                contracts.validate_pair.assert_called_once_with(expected)
                if mode == "sweep":
                    contracts.validate_sweep_step.assert_called_once_with(
                        expected, session.sweep_plan
                    )
                    contracts.validate_pair_step.assert_not_called()
                    contracts.validate_record.assert_not_called()
                    self.assertEqual(
                        session.details, {"sweep_coverage": {"completed": 1}}
                    )
                elif mode == "prompt_pair":
                    contracts.validate_pair_step.assert_called_once_with(
                        expected, session.pair_request
                    )
                    contracts.validate_sweep_step.assert_not_called()
                    contracts.validate_record.assert_not_called()
                else:
                    contracts.validate_record.assert_called_once_with(
                        expected, session.observation, 2
                    )
                    contracts.validate_sweep_step.assert_not_called()
                    contracts.validate_pair_step.assert_not_called()

    def test_step_refusals_keep_exact_error_and_reap(self) -> None:
        cases = [
            ("generation", {"index": 1, "activation": [1, 2, 3]}),
            ("generation", {"index": 0, "activation": [1, 2]}),
            ("prompt_pair_preview", {"index": 0, "activation": [1, 2, 3]}),
        ]
        for mode, fields in cases:
            with (
                self.subTest(mode=mode, fields=fields),
                self.worker([{"type": "step", **fields}], mode) as (
                    session,
                    process,
                    reap,
                    _contracts,
                ),
            ):
                session.tick()
                self.assertEqual(list(session.steps), [])
                self.assertEqual(session.details, {"error": "Invalid worker output"})
                self.assertEqual(session.status, "error")
                self.assertIsNone(session.process)
                reap.assert_called_once_with(process)

    def test_generation_and_pair_completion(self) -> None:
        for mode, field in (
            ("generation", "generated_tokens"),
            ("comparison", "generated_tokens"),
            ("prompt_pair", "record_count"),
        ):
            event = {"type": "done", field: 2, "reason": "fixture-complete"}
            with (
                self.subTest(mode=mode),
                self.worker([event], mode) as (
                    session,
                    process,
                    reap,
                    _contracts,
                ),
            ):
                session.steps.extend([{"index": 0}, {"index": 1}])
                session.tick()
                self.assertEqual(session.status, "complete")
                self.assertEqual(
                    session.details, {field: 2, "reason": "fixture-complete"}
                )
                self.assertIsNone(session.process)
                self.assertIsNone(session.stop_reason)
                reap.assert_called_once_with(process)

    def test_sweep_complete_and_partial(self) -> None:
        for count, status in ((2, "complete"), (1, "time_limit")):
            event = {
                "type": "sweep_done",
                "status": status,
                "coverage": {"completed": count},
                "reason": "fixture-sweep",
            }
            with (
                self.subTest(status=status),
                self.worker([event], "sweep") as (
                    session,
                    process,
                    reap,
                    contracts,
                ),
            ):
                session.steps.extend({"index": n} for n in range(count))
                session.tick()
                self.assertEqual(session.status, status)
                self.assertEqual(
                    session.details,
                    {
                        "coverage": {"completed": count},
                        "reason": "fixture-sweep",
                        "sweep_coverage": {"completed": count},
                    },
                )
                self.assertIsNone(session.process)
                self.assertEqual(contracts.stops, [(status, session.details, status)])
                reap.assert_called_once_with(process)

    def test_preview_completion(self) -> None:
        preview = {"held": [1, 2, 3]}
        event = {"type": "preview_done", "preview": preview, "reason": "token_preview"}
        with self.worker([event], "prompt_pair_preview") as (
            session,
            process,
            reap,
            contracts,
        ):
            session.tick()
            self.assertEqual(session.status, "complete")
            self.assertEqual(
                session.details, {"preview": preview, "reason": "token_preview"}
            )
            contracts.validate_preview.assert_called_once_with(preview)
            self.assertIsNone(session.process)
            reap.assert_called_once_with(process)

    def test_completion_refusals(self) -> None:
        cases = [
            ("generation", {"type": "done", "generated_tokens": 0}, 0),
            ("generation", {"type": "done", "generated_tokens": True}, 1),
            ("generation", {"type": "done", "generated_tokens": 2}, 1),
            ("prompt_pair", {"type": "done", "record_count": 1}, 1),
            ("sweep", {"type": "done", "generated_tokens": 1}, 1),
            ("prompt_pair_preview", {"type": "done", "generated_tokens": 1}, 1),
            (
                "generation",
                {"type": "preview_done", "preview": {}, "reason": "token_preview"},
                0,
            ),
            (
                "prompt_pair_preview",
                {"type": "preview_done", "preview": {}, "reason": "wrong"},
                0,
            ),
            (
                "sweep",
                {
                    "type": "sweep_done",
                    "status": "complete",
                    "coverage": {"completed": 1},
                },
                1,
            ),
            (
                "sweep",
                {
                    "type": "sweep_done",
                    "status": "time_limit",
                    "coverage": {"completed": 2},
                },
                1,
            ),
        ]
        for mode, event, count in cases:
            with (
                self.subTest(mode=mode, event=event),
                self.worker([deepcopy(event)], mode) as (
                    session,
                    process,
                    reap,
                    _contracts,
                ),
            ):
                session.steps.extend({"index": n} for n in range(count))
                session.tick()
                self.assertEqual(session.details["error"], "Invalid worker output")
                self.assertEqual(session.status, "error")
                self.assertIsNone(session.process)
                reap.assert_called_once_with(process)

    def test_progress_and_error_events(self) -> None:
        for kind in ("prefill", "loaded", "error"):
            with (
                self.subTest(kind=kind),
                self.worker([{"type": kind, "held": [1, 2]}]) as (
                    session,
                    process,
                    reap,
                    _contracts,
                ),
            ):
                session.tick()
                self.assertEqual(session.details, {"held": [1, 2]})
                self.assertEqual(
                    session.status, "error" if kind == "error" else "running"
                )
                if kind == "error":
                    self.assertIsNone(session.process)
                    reap.assert_called_once_with(process)
                else:
                    self.assertIs(session.process, process)
                    reap.assert_not_called()


if __name__ == "__main__":
    unittest.main()
