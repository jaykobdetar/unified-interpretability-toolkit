"""Inert ownership and presentation contracts for analytics coordination."""

import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent / "analytics"))
from analytics import service
import test_service as service_fixture


class ServiceContracts(unittest.TestCase):
    def setUp(self):
        self.fixture = service_fixture.AdapterTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.jobs = self.fixture.jobs

    def test_init(self):
        f, jobs = self.fixture, self.jobs
        self.assertEqual(jobs.python, "unused-python")
        self.assertEqual(jobs.model, f.root.resolve())
        for name, expected in (
            ("inference_busy", f.busy),
            ("fetch_model", f.fetch),
            ("reap", f.reap),
            ("clock", f.clock),
        ):
            self.assertIs(getattr(jobs, name), expected)
        for name in (
            "process",
            "id",
            "result",
            "error",
            "stop_reason",
            "summary_contract",
        ):
            self.assertIsNone(getattr(jobs, name))
        self.assertEqual(jobs.status, "idle")
        self.assertEqual(jobs.buffer, bytearray())
        self.assertIsInstance(jobs.buffer, bytearray)
        self.assertEqual((jobs.started, jobs.last_seen, jobs.peak_rss), (10.0, 10.0, 0))
        f.clock.assert_called_once_with()

    def test_busy(self):
        for status in (
            "idle",
            "starting",
            "running",
            "stopping",
            "complete",
            "error",
            "expired",
        ):
            with self.subTest(status=status):
                self.jobs.status = status
                self.jobs.process = None
                self.assertIs(
                    self.jobs.busy, status in ("starting", "running", "stopping")
                )
                self.jobs.process = self.fixture.fake
                self.assertIs(self.jobs.busy, True)

    def test_metadata(self):
        self.assertEqual(
            self.jobs.metadata(),
            {
                "available": True,
                "busy": False,
                "limits": {
                    "jobs": 1,
                    "queue": 0,
                    "values": 65536,
                    "axis": 4096,
                    "svd_axis": 64,
                    "svd_values": 4096,
                    "wall_seconds": 5,
                    "svd_summary": {
                        "schema": "weight-atlas.svd-window-summary.v2",
                        "axis": 128,
                        "values": 16384,
                        "preview_axis": 16,
                        "body_bytes": 63488,
                    },
                    "output_bytes": 2095104,
                    "worker_address_space_mib": 768,
                },
                "coverage": "Explicit selected window or bounded catalog prefix; no full-model claim by default",
            },
        )
        self.fixture.busy.return_value = True
        self.assertIs(self.jobs.metadata()["busy"], True)
        self.jobs.status = "running"
        self.fixture.busy.reset_mock()
        self.assertIs(self.jobs.metadata()["busy"], True)
        self.fixture.busy.assert_not_called()

    def test_owns(self):
        self.jobs.id = "fixture-token"
        for token, expected in (
            ("fixture-token", True),
            ("other-token", False),
            (None, False),
            (7, False),
        ):
            self.assertIs(self.jobs.owns(token), expected)
        self.jobs.id = None
        self.assertIs(self.jobs.owns("fixture-token"), False)

    def test_snapshot(self):
        jobs = self.jobs
        marker = {"value": [1, None]}
        jobs.id, jobs.status, jobs.result = "fixture-token", "complete", marker
        jobs.error, jobs.stop_reason = "fixture diagnostic", "complete"
        jobs.process, jobs.peak_rss = self.fixture.fake, 2 * 1024**2
        result = jobs.snapshot()
        self.assertEqual(
            result,
            {
                "job": "fixture-token",
                "status": "complete",
                "result": marker,
                "error": "fixture diagnostic",
                "cleanup_pending": True,
                "worker_alive": True,
                "peak_worker_rss_mib": 2.0,
            },
        )
        self.assertIs(result["result"], marker)
        jobs.status = "stopping"
        self.assertIsNone(jobs.snapshot()["result"])

    def test_model(self):
        f = self.fixture
        f.model.update(model_identity="model-fixture", revision="revision-fixture")
        result = self.jobs._model()
        self.assertEqual(
            result,
            {
                "model_identity": "model-fixture",
                "revision": "revision-fixture",
                "source_identity": "trusted-fixture",
                "catalog": [f.tensor],
            },
        )
        self.assertIs(result["catalog"][0]["shape"], f.tensor["shape"])
        self.assertNotIn("source_directory", result)
        for changes, message in (
            (
                {"comparison_identity": "pair"},
                "Analytics requires one original BF16 source, not a checkpoint comparison",
            ),
            (
                {"catalog": [{**f.tensor, "dtype": "F32"}]},
                "Analytics requires an all-BF16 source catalog; F16/F32 analysis is unsupported",
            ),
            (
                {"source_directory": str(f.root / "different")},
                "Validated renderer model differs from configured analytics source",
            ),
        ):
            f.fetch.return_value = {**f.model, **changes}
            with self.assertRaises(ValueError) as raised:
                self.jobs._model()
            self.assertEqual(str(raised.exception), message)

    def test_start(self):
        f, jobs = self.fixture, self.jobs
        requests = []

        def spawn(*args, **kwargs):
            requests.append((args, kwargs, kwargs["stdin"].read()))
            return f.fake

        with (
            mock.patch.object(service.subprocess, "Popen", side_effect=spawn),
            mock.patch.object(service.os, "set_blocking") as blocking,
            mock.patch.object(
                service.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=30 * service.GIB),
            ),
            mock.patch.object(
                service.secrets, "token_hex", return_value="fixture-token"
            ) as token,
        ):
            result = jobs.start(f.data)
        token.assert_called_once_with(16)
        blocking.assert_called_once_with(123, False)
        self.assertEqual(f.fetch.call_count, 2)
        self.assertEqual((jobs.started, jobs.last_seen), (10.0, 10.0))
        self.assertEqual(
            result,
            {
                "job": "fixture-token",
                "status": "running",
                "result": None,
                "error": None,
                "cleanup_pending": False,
                "worker_alive": True,
                "peak_worker_rss_mib": 0.0,
            },
        )
        self.assertEqual(len(requests), 1)
        args, kwargs, raw = requests[0]
        self.assertEqual(args, (["unused-python", "-B", "-m", "analytics.worker"],))
        self.assertEqual(kwargs["cwd"], service.WORKER_ROOT)
        self.assertEqual(kwargs["bufsize"], 0)
        self.assertEqual(
            (kwargs["stdout"], kwargs["stderr"]),
            (service.subprocess.PIPE, service.subprocess.DEVNULL),
        )
        for name in (
            "PYTHONDONTWRITEBYTECODE",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            self.assertEqual(kwargs["env"][name], "1")
        payload = json.loads(raw)
        self.assertEqual(payload["request"], f.data)
        self.assertEqual(payload["model"]["catalog"], [f.tensor])
        self.assertEqual(
            raw, json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
        )
        self.assertIs(jobs.process, f.fake)

    def test_stop(self):
        f, jobs = self.fixture, self.jobs
        jobs.process, jobs.status = f.fake, "running"
        jobs.buffer.extend(b"pending")
        f.reap.return_value = False
        self.assertIs(jobs.stop(), False)
        self.assertIs(jobs.process, f.fake)
        self.assertEqual((jobs.status, jobs.stop_reason), ("stopping", "cancelled"))
        self.assertEqual(jobs.buffer, b"pending")
        f.fake.stdout.close.assert_not_called()
        f.reap.return_value = True
        self.assertIs(jobs.stop(), True)
        self.assertEqual(
            (jobs.status, jobs.stop_reason, jobs.process), ("cancelled", None, None)
        )
        self.assertEqual(jobs.buffer, b"")
        f.fake.stdout.close.assert_called_once_with()
        self.assertEqual(f.reap.call_args_list, [mock.call(f.fake), mock.call(f.fake)])

    def test_tick(self):
        f, jobs = self.fixture, self.jobs
        jobs.result, jobs.id = {}, "fixture-token"
        f.clock.return_value = 26.0
        jobs.tick()
        self.assertEqual((jobs.status, jobs.id, jobs.result), ("expired", None, None))
        f.clock.return_value = 10.0
        jobs.process, jobs.status = f.fake, "running"
        with (
            mock.patch.object(
                service.Path,
                "read_text",
                return_value="Name: fixture\nVmRSS:\t2048 kB\n",
            ),
            mock.patch.object(service.os, "read", side_effect=BlockingIOError),
        ):
            jobs.tick()
        self.assertEqual(jobs.peak_rss, 2 * 1024**2)
        self.assertEqual(jobs.status, "running")
        jobs.buffer = bytearray(b"x" * (service.MAX_OUTPUT + 1))
        with (
            mock.patch.object(service.Path, "read_text", side_effect=OSError),
            mock.patch.object(service.os, "read", return_value=b"x") as read,
        ):
            jobs.tick()
        self.assertEqual(
            (jobs.status, jobs.error, jobs.process),
            ("error", "Analysis output cap exceeded", None),
        )
        read.assert_called_once_with(123, 65536)
        f.reap.assert_called_once_with(f.fake)

    def test_route(self):
        jobs = mock.Mock()
        jobs.busy = False
        jobs.inference_busy.return_value = False
        jobs.start.return_value = {"job": "fixture-token"}
        handler = SimpleNamespace(
            command="POST",
            rfile=io.BytesIO(b'{"tensor":0}'),
            send=mock.Mock(return_value="sent"),
        )
        self.assertEqual(
            service.route(handler, "/api/analytics/start", 12, jobs), "sent"
        )
        jobs.start.assert_called_once_with({"tensor": 0})
        handler.send.assert_called_once_with(202, {"job": "fixture-token"})
        handler.command = "GET"
        handler.send.reset_mock()
        service.route(handler, "/other", 0, jobs)
        handler.send.assert_called_once_with(404, {"error": "Not found"})
        handler.command, handler.rfile = "POST", io.BytesIO(b'{"job":"other"}')
        jobs.owns.return_value = False
        handler.send.reset_mock()
        service.route(handler, "/api/analytics/poll", 15, jobs)
        handler.send.assert_called_once_with(
            409, {"error": "Stale analysis job", "code": "stale_job"}
        )
        self.assertEqual(jobs.tick.call_count, 3)


if __name__ == "__main__":
    unittest.main()
