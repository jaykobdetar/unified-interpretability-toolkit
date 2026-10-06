"""Analytics memory admission outcomes using inert OS and file observations."""

from collections.abc import Callable
import importlib
from pathlib import Path
import resource
from typing import cast
import unittest
from unittest.mock import patch


def configuration_outcome(
    observation: str | Exception,
) -> tuple[type[Exception] | None, str]:
    worker = importlib.import_module("analytics.worker")
    configure = cast(Callable[[], None], worker.configure)
    with (
        patch("signal.signal"),
        patch("signal.alarm"),
        patch.object(worker.os, "sched_getaffinity", return_value={2, 3}),
        patch.object(worker.os, "sched_setaffinity"),
        patch.object(worker.os, "nice"),
        patch.object(
            worker.resource,
            "getrlimit",
            return_value=(resource.RLIM_INFINITY, resource.RLIM_INFINITY),
        ),
        patch.object(worker.resource, "setrlimit"),
        patch.object(
            Path,
            "read_text",
            return_value=observation if isinstance(observation, str) else "",
            side_effect=observation if isinstance(observation, Exception) else None,
        ),
    ):
        try:
            configure()
            return (None, "")
        except Exception as error:
            return (type(error), str(error))


class AnalyticsRuntimeAdmission(unittest.TestCase):
    def test_exact_memory_floor_and_refusal_text(self) -> None:
        floor_kib = 13 * 1024**2 // 4
        for available, expected in [
            (floor_kib - 1, (ValueError, "Memory reserve reached")),
            (floor_kib, (None, "")),
            (floor_kib + 1, (None, "")),
        ]:
            with self.subTest(available_kib=available):
                self.assertEqual(
                    configuration_outcome(f"MemAvailable: {available} kB\n"),
                    expected,
                    "analytics memory admission outcome",
                )

    def test_original_observation_failure_is_not_reclassified_as_admission(
        self,
    ) -> None:
        cases: list[tuple[str | Exception, tuple[type[Exception] | None, str]]] = [
            ("MemFree: 8388608 kB\n", (StopIteration, "")),
            ("MemAvailable:\n", (IndexError, "list index out of range")),
            (
                PermissionError("fixture memory observation denied"),
                (PermissionError, "fixture memory observation denied"),
            ),
        ]
        for observation, expected in cases:
            with self.subTest(observation=observation):
                self.assertEqual(
                    configuration_outcome(observation),
                    expected,
                    "analytics observation error remains unchanged",
                )


if __name__ == "__main__":
    unittest.main()
