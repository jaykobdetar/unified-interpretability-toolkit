"""Inert SVD admission and child-boundary witnesses; no numerical import or spawn."""

from collections.abc import Callable
import importlib
import io
import json
from pathlib import Path
import resource
from types import SimpleNamespace
from typing import cast
import unittest
from unittest.mock import patch


class SvdRuntimeContracts(unittest.TestCase):
    def test_parent_admission_exact_boundaries_request_and_refusal(self) -> None:
        svd = importlib.import_module("analytics.svd")
        run = cast(Callable[..., dict[str, object]], svd.run)
        floor = 15 * 1024**3 // 4
        region = {"row": 0, "col": 0, "rows": 1, "cols": 1}
        for memory, disk, accepted in [
            (floor - 1024, 25 * 1024**3, False),
            (floor, 25 * 1024**3 - 1, False),
            (floor, 25 * 1024**3, True),
            (floor + 1024, 25 * 1024**3 + 1, True),
        ]:
            with (
                self.subTest(memory=memory, disk=disk),
                patch.object(
                    Path,
                    "read_text",
                    return_value=f"MemAvailable: {memory // 1024} kB\n",
                ),
                patch.object(
                    svd.shutil, "disk_usage", return_value=SimpleNamespace(free=disk)
                ),
                patch.object(
                    svd.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=0, stdout='{"fixture":1}'),
                ) as spawn,
            ):
                result = run([0], [1], region, seed=7, python="fixture-python")
                if accepted:
                    self.assertEqual(
                        result,
                        {
                            "available": True,
                            "scope": "full tensor",
                            "region": region,
                            "seed": 7,
                            "control": "exact multiset; each SVD fitted separately",
                            "centered": False,
                            "results": {"fixture": 1},
                        },
                        "SVD accepted result",
                    )
                    spawn.assert_called_once()
                    self.assertEqual(
                        spawn.call_args.args,
                        (["fixture-python", "-B", "-m", "analytics.svd", "--worker"],),
                    )
                    self.assertEqual(
                        spawn.call_args.kwargs["timeout"], 5, "SVD parent wall timer"
                    )
                    self.assertEqual(
                        json.loads(spawn.call_args.kwargs["input"]),
                        {"values": [0], "rows": 1, "cols": 1, "seed": 7},
                    )
                    for key in [
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS",
                        "PYTHONDONTWRITEBYTECODE",
                    ]:
                        self.assertEqual(spawn.call_args.kwargs["env"][key], "1")
                    self.assertTrue(spawn.call_args.kwargs["text"])
                    self.assertTrue(spawn.call_args.kwargs["capture_output"])
                else:
                    self.assertEqual(
                        result,
                        {
                            "available": False,
                            "reason": "Paused by existing memory/disk reserve gates",
                        },
                        "SVD parent reserve outcome",
                    )
                    spawn.assert_not_called()

    def test_memory_observation_errors_stay_original(self) -> None:
        svd = importlib.import_module("analytics.svd")
        run = cast(Callable[..., dict[str, object]], svd.run)
        cases: list[tuple[str | Exception, tuple[type[Exception], str]]] = [
            ("MemFree: 8388608 kB\n", (StopIteration, "")),
            ("MemAvailable:\n", (IndexError, "list index out of range")),
            (
                PermissionError("SVD fixture observation denied"),
                (PermissionError, "SVD fixture observation denied"),
            ),
        ]
        for observation, expected in cases:
            with (
                self.subTest(observation=observation),
                patch.object(
                    Path,
                    "read_text",
                    return_value=observation if isinstance(observation, str) else "",
                    side_effect=(
                        observation if isinstance(observation, Exception) else None
                    ),
                ),
                patch.object(svd.shutil, "disk_usage") as disk,
                patch.object(svd.subprocess, "run") as spawn,
            ):
                outcome: tuple[type[Exception] | None, str] = (None, "")
                try:
                    run([0], [1], {"row": 0, "col": 0, "rows": 1, "cols": 1})
                except Exception as error:
                    outcome = (type(error), str(error))
                self.assertEqual(outcome, expected, "SVD original observation error")
                disk.assert_not_called()
                spawn.assert_not_called()

    def test_worker_affinity_nice_inherited_caps_and_output(self) -> None:
        svd = importlib.import_module("analytics.svd")
        worker = cast(Callable[[], None], svd.worker)
        for inherited in [
            (resource.RLIM_INFINITY, resource.RLIM_INFINITY),
            (1, 2),
            (resource.RLIM_INFINITY, 2),
            (3, resource.RLIM_INFINITY),
        ]:
            with (
                self.subTest(inherited=inherited),
                patch.object(svd.os, "sched_getaffinity", return_value={2, 3}),
                patch.object(svd.os, "sched_setaffinity") as affinity,
                patch.object(svd.os, "nice") as nice,
                patch.object(svd.resource, "getrlimit", return_value=inherited),
                patch.object(svd.resource, "setrlimit") as limits,
                patch("signal.alarm") as alarm,
                patch.object(
                    svd.sys,
                    "stdin",
                    io.StringIO('{"rows":1,"cols":1,"values":[0],"seed":7}'),
                ),
                patch.object(svd.sys, "stdout", io.StringIO()) as output,
                patch.dict(svd.sys.modules, {"numpy": object()}),
                patch.object(
                    svd, "compute_with_numpy", return_value={"fixture": 1}
                ) as compute,
            ):
                worker()
                affinity.assert_called_once_with(0, {2})
                nice.assert_called_once_with(10)
                finite = [x for x in inherited if x != resource.RLIM_INFINITY]
                expected = []
                for kind, ceiling in [
                    (resource.RLIMIT_AS, 768 * 1024**2),
                    (resource.RLIMIT_CPU, 4),
                ]:
                    cap = min([ceiling, *finite])
                    expected.append((kind, (cap, cap)))
                self.assertEqual(
                    [call.args for call in limits.call_args_list],
                    expected,
                    "SVD inherited child caps",
                )
                alarm.assert_not_called()
                compute.assert_called_once()
                self.assertEqual(output.getvalue(), '{"fixture": 1}\n')


if __name__ == "__main__":
    unittest.main()
