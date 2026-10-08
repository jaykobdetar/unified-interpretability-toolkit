"""Declared analytics boundaries using small sources, clocks and process doubles."""

from copy import deepcopy
import importlib.util
import io
import json
import math
from pathlib import Path
import resource
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
from analytics import core, service, source, svd_summary, worker

spec = importlib.util.spec_from_file_location(
    "analytics_limit_fixture", ROOT / "tests/analytics/test_service.py"
)
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)
MIB, GIB = 1024**2, 1024**3


class AnalyticsLimits(unittest.TestCase):
    def edges(self, operation, low, high):
        for number, accepted in (
            (low - 1, False),
            (low, True),
            (high, True),
            (high + 1, False),
        ):
            with self.subTest(number=number):
                if accepted:
                    operation(number)
                else:
                    with self.assertRaises(ValueError):
                        operation(number)

    def fixture(self):
        fixture = fixture_module.AdapterTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        return fixture

    def test_geometry_shape_rank_positions_and_axis_edges(self):
        self.assertEqual(
            (core.MAX_VALUES, core.MAX_AXIS, core.MAX_TOP, core.MAX_TENSORS),
            (65536, 4096, 32, 512),
        )
        unit = {"row": 0, "col": 0, "rows": 1, "cols": 1}
        self.edges(lambda n: core.geometry([n], unit), 1, 200000)
        self.edges(lambda n: core.geometry([n, 1], unit), 1, 200000)
        for shape in ([], [1, 1, 1]):
            with self.assertRaises(ValueError):
                core.geometry(shape, unit)
        for field in ("row", "col"):
            self.edges(lambda n: core.geometry([7, 7], {**unit, field: n}), 0, 6)
        for field in ("rows", "cols"):
            self.edges(
                lambda n: core.geometry([200000, 200000], {**unit, field: n}), 1, 4096
            )
        for field, position in (("rows", "row"), ("cols", "col")):
            self.edges(
                lambda n: core.geometry([7, 7], {**unit, position: 5, field: n}), 1, 2
            )

    def test_geometry_reachable_cell_edges_and_count_domain(self):
        # 65537 is prime and cannot fit the independent 4096-axis caps.
        for rows, cols, accepted in (
            (255, 257, True),
            (256, 256, True),
            (257, 256, False),
        ):
            region = {"row": 0, "col": 0, "rows": rows, "cols": cols}
            if accepted:
                core.geometry([4096, 4096], region)
            else:
                with self.assertRaisesRegex(ValueError, "65536-value"):
                    core.geometry([4096, 4096], region)
        # Permutation has an independent count domain with a reachable one-past.
        self.edges(lambda n: self.assertEqual(len(core.permutation(n, 1)), n), 0, 65536)
        self.edges(lambda seed: core.permutation(3, seed), 0, 2**32 - 1)

    def test_values_top_and_bf16_range(self):
        region = {"row": 0, "col": 0, "rows": 1, "cols": 1}
        for value in (0, -3.39e38, 3.39e38):
            core.checked_values([value], [1], region)
        for value in (
            math.nextafter(-3.39e38, -math.inf),
            math.nextafter(3.39e38, math.inf),
            math.inf,
            math.nan,
            True,
        ):
            with self.assertRaises(ValueError):
                core.checked_values([value], [1], region)
        for values in ([], [0, 0]):
            with self.assertRaisesRegex(ValueError, "count mismatch"):
                core.checked_values(values, [1], region)
        self.edges(lambda top: core.statistics([0], [1], region, top), 1, 32)

    def test_catalog_real_count_name_shape_extent_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "tiny.safetensors"
            # Raw BF16 source adapter; no model or safetensors loader is invoked.
            shard.write_bytes(b"\0" * 400000)
            tensor = source.Tensor("x", (1,), shard.name, 0)
            for count, accepted in ((0, False), (1, True), (512, True), (513, False)):
                tensors = [
                    source.Tensor(str(i), (1,), shard.name, i * 2) for i in range(count)
                ]
                if accepted:
                    self.assertEqual(
                        len(source.Catalog(root, tensors, "fixture").tensors), count
                    )
                else:
                    with self.assertRaises(ValueError):
                        source.Catalog(root, tensors, "fixture")
            self.edges(
                lambda n: source.Catalog(
                    root, [source.Tensor("é" * n, (1,), shard.name, 0)], "fixture"
                ),
                1,
                512,
            )
            self.edges(
                lambda n: source.Catalog(
                    root, [source.Tensor("x", (n,), shard.name, 0)], "fixture"
                ),
                1,
                200000,
            )
            for shape, accepted in (
                ((), False),
                ((1,), True),
                ((1, 1), True),
                ((1, 1, 1), False),
            ):
                if accepted:
                    source.Catalog(
                        root, [source.Tensor("x", shape, shard.name, 0)], "fixture"
                    )
                else:
                    with self.assertRaises(ValueError):
                        source.Catalog(
                            root, [source.Tensor("x", shape, shard.name, 0)], "fixture"
                        )
            self.edges(
                lambda n: source.Catalog(
                    root,
                    [source.Tensor(tensor.name, tensor.shape, shard.name, n)],
                    "fixture",
                ),
                0,
                399998,
            )

    def test_catalog_declared_offset_guard_with_stat_double(self):
        # The signed-63-bit declared offset edge cannot fit a real nonempty shard.
        # This checks metadata admission only, not physical file extent/readability.
        with tempfile.TemporaryDirectory() as directory:
            st = SimpleNamespace(
                st_mode=0o100600,
                st_dev=1,
                st_ino=1,
                st_size=2**63 + 1,
                st_mtime_ns=1,
                st_ctime_ns=1,
            )
            with (
                patch.object(Path, "lstat", return_value=st),
                patch("analytics.source.optional_index_fingerprint", return_value=None),
            ):
                self.edges(
                    lambda n: source.Catalog(
                        directory,
                        [source.Tensor("x", (1,), "tiny.safetensors", n)],
                        "fixture",
                    ),
                    0,
                    2**63 - 1,
                )

    def test_model_ranking_budget_top_seed_and_two_second_cutoff(self):
        fixture = self.fixture()
        catalog = worker.catalog_from_payload(fixture.payload())
        self.edges(lambda n: catalog.model_outliers(top=n), 1, 32)
        self.edges(lambda n: catalog.model_outliers(value_budget=n), 1, 65536)
        self.edges(lambda n: catalog.model_outliers(seed=n), 0, 2**32 - 1)
        with patch(
            "analytics.source.time.monotonic",
            side_effect=[0.0, math.nextafter(2.0, 0.0)],
        ):
            below = catalog.model_outliers()
        with patch("analytics.source.time.monotonic", side_effect=[0.0, 2.0]):
            at = catalog.model_outliers()
        self.assertGreater(below["coverage"]["visited_values"], 0)
        self.assertEqual(at["coverage"]["visited_values"], 0)

    def test_worker_request_tensor_and_seed_edges(self):
        catalog = [{"id": i, "shape": [1]} for i in range(512)]
        data = {"tensor": 0, "region": {"row": 0, "col": 0, "rows": 1, "cols": 1}}
        self.edges(
            lambda n: worker.validate_request({**data, "tensor": n}, catalog), 0, 511
        )
        self.edges(
            lambda n: worker.validate_request({**data, "seed": n}, catalog),
            0,
            2**32 - 1,
        )
        self.edges(
            lambda n: worker.validate_request({"scope": "model", "seed": n}, catalog),
            0,
            2**32 - 1,
        )

    def run_worker(self, raw, result):
        output = io.BytesIO()
        with (
            patch.object(worker, "configure"),
            patch.object(worker, "analyze_request", return_value=result) as analyze,
            patch.object(worker.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(raw))),
            patch.object(worker.sys, "stdout", SimpleNamespace(buffer=output)),
        ):
            worker.main()
        return (
            json.loads(output.getvalue()),
            analyze.call_count,
            len(output.getvalue()) - 1,
        )

    def test_worker_real_input_and_output_encoded_byte_edges(self):
        self.assertEqual((worker.MAX_INPUT, worker.MAX_OUTPUT), (524288, 2095104))
        for size, accepted in ((524288, True), (524289, False)):
            # A real valid JSON document, padded with JSON whitespace.
            raw = b"{}" + b" " * (size - 2)
            response, called, _ = self.run_worker(raw, {})
            self.assertEqual(response["ok"], accepted)
            self.assertEqual(called, int(accepted))
        base = len(
            json.dumps(
                {"ok": True, "result": {"padding": ""}}, separators=(",", ":")
            ).encode()
        )
        for size, accepted in ((2095104, True), (2095105, False)):
            response, called, output_size = self.run_worker(
                b"{}", {"padding": "x" * (size - base)}
            )
            self.assertEqual(response["ok"], accepted)
            self.assertEqual(called, 1)
            if accepted:
                self.assertEqual(output_size, size)
            else:
                self.assertIn("output cap", response["error"])

    def test_worker_configure_caps_inherited_limits_and_reserve(self):
        for available_kib, accepted in (
            (13 * GIB // 4 // 1024 - 1, False),
            (13 * GIB // 4 // 1024, True),
        ):
            for inherited in ((resource.RLIM_INFINITY, resource.RLIM_INFINITY), (1, 2)):
                with (
                    self.subTest(available_kib=available_kib, inherited=inherited),
                    patch("signal.signal"),
                    patch("signal.alarm") as alarm,
                    patch.object(worker.os, "sched_getaffinity", return_value={2, 3}),
                    patch.object(worker.os, "sched_setaffinity") as affinity,
                    patch.object(worker.os, "nice"),
                    patch.object(worker.resource, "getrlimit", return_value=inherited),
                    patch.object(worker.resource, "setrlimit") as set_limit,
                    patch.object(
                        Path,
                        "read_text",
                        return_value=f"MemAvailable: {available_kib} kB\n",
                    ),
                ):
                    if accepted:
                        worker.configure()
                    else:
                        with self.assertRaisesRegex(ValueError, "Memory reserve"):
                            worker.configure()
                    alarm.assert_called_once_with(5)
                    affinity.assert_called_once_with(0, {2})
                    expected = (
                        [
                            (resource.RLIMIT_AS, (768 * MIB, 768 * MIB)),
                            (resource.RLIMIT_CPU, (4, 4)),
                        ]
                        if inherited[0] == resource.RLIM_INFINITY
                        else [
                            (resource.RLIMIT_AS, (1, 1)),
                            (resource.RLIMIT_CPU, (1, 1)),
                        ]
                    )
                    self.assertEqual(
                        [call.args for call in set_limit.call_args_list], expected
                    )

    def test_service_admission_available_and_disk_boundaries(self):
        for memory, disk, accepted in (
            (15 * GIB // 4, 25 * GIB, True),
            (15 * GIB // 4 - 1, 25 * GIB, False),
            (15 * GIB // 4, 25 * GIB - 1, False),
        ):
            fixture = self.fixture()
            fixture.jobs.available = lambda: memory
            with (
                patch(
                    "analytics.service.shutil.disk_usage",
                    return_value=SimpleNamespace(free=disk),
                ),
                patch(
                    "analytics.service.subprocess.Popen", return_value=fixture.fake
                ) as spawn,
                patch("analytics.service.os.set_blocking"),
            ):
                if accepted:
                    fixture.jobs.start(fixture.data)
                    spawn.assert_called_once()
                    fixture.jobs.stop()
                else:
                    with self.assertRaisesRegex(ValueError, "reserve gates"):
                        fixture.jobs.start(fixture.data)
                    spawn.assert_not_called()

    def test_service_running_memory_rss_wall_and_lease_edges(self):
        cases = (
            ("memory", 13 * GIB // 4, False),
            ("memory", 13 * GIB // 4 - 1, True),
            ("rss", 768 * MIB, False),
            ("rss", 768 * MIB + 1, True),
            ("wall", 5, False),
            ("wall", math.nextafter(5, math.inf), True),
            ("lease", 15, False),
            ("lease", math.nextafter(15, math.inf), True),
        )
        for kind, number, stopped in cases:
            with self.subTest(kind=kind, number=number):
                fixture = self.fixture()
                fixture.start()
                if kind == "memory":
                    fixture.jobs.available = lambda: number
                elif kind == "rss":
                    fixture.jobs.peak_rss = number
                elif kind == "wall":
                    fixture.jobs.started = 0.0
                    fixture.jobs.last_seen = number
                    fixture.clock.return_value = number
                else:
                    fixture.jobs.last_seen = 0.0
                    fixture.jobs.started = number
                    fixture.clock.return_value = number
                with (
                    patch("analytics.service.os.read", side_effect=BlockingIOError),
                    patch.object(Path, "read_text", side_effect=FileNotFoundError),
                ):
                    fixture.jobs.tick()
                self.assertEqual(fixture.jobs.process is None, stopped)
                if not stopped:
                    fixture.jobs.stop()

    def test_service_result_lease_and_error_truncation_edges(self):
        for age, expired in ((15.0, False), (math.nextafter(15.0, math.inf), True)):
            fixture = self.fixture()
            fixture.jobs.result = {}
            fixture.jobs.id = "fixture"
            fixture.jobs.status = "complete"
            fixture.jobs.last_seen = 0.0
            fixture.clock.return_value = age
            fixture.jobs.tick()
            self.assertEqual(fixture.jobs.status == "expired", expired)
        for size in (511, 512, 513):
            fixture = self.fixture()
            fixture.start()
            raw = json.dumps({"ok": False, "error": "é" * size}).encode() + b"\n"
            with patch("analytics.service.os.read", return_value=raw):
                fixture.jobs.tick()
            self.assertEqual(fixture.jobs.error, "é" * min(size, 512))
            self.assertIsNone(fixture.jobs.process)

    def test_service_output_buffer_edge_and_four_reads_per_tick(self):
        for size, accepted in ((2095104, True), (2095105, False)):
            fixture = self.fixture()
            fixture.start()
            base = len(
                json.dumps(
                    {"ok": True, "result": {"padding": ""}}, separators=(",", ":")
                ).encode()
            )
            raw = json.dumps(
                {"ok": True, "result": {"padding": "x" * (size - base)}},
                separators=(",", ":"),
            ).encode()
            fixture.jobs.buffer = bytearray(raw)
            with patch("analytics.service.os.read", side_effect=[b"\n"]) as read:
                fixture.jobs.tick()
            self.assertEqual(fixture.jobs.status, "complete" if accepted else "error")
            self.assertIsNone(fixture.jobs.process)
            read.assert_called_once_with(123, 65536)
        fixture = self.fixture()
        fixture.start()
        with patch("analytics.service.os.read", return_value=b"x" * 65536) as read:
            fixture.jobs.tick()
        self.assertEqual(read.call_count, 4)
        self.assertEqual(len(fixture.jobs.buffer), 262144)
        fixture.jobs.stop()

    def test_svd_window_axis_and_derived_cell_caps(self):
        self.assertEqual(
            (
                svd_summary.MAX_AXIS,
                svd_summary.MAX_VALUES,
                svd_summary.PREVIEW_AXIS,
                svd_summary.MAX_BODY,
            ),
            (128, 16384, 16, 63488),
        )
        unit = {"row": 0, "col": 0, "rows": 1, "cols": 1}
        for field in ("rows", "cols"):
            self.edges(
                lambda n: svd_summary.check_window(
                    [200000, 200000], {**unit, field: n}
                ),
                1,
                128,
            )
        # 128 squared reaches the cap; any larger count also violates an axis bound.
        svd_summary.check_window([128, 128], {**unit, "rows": 128, "cols": 128})
        fixture = self.fixture()
        fixture.summary_mode()
        for dimension, preview in ((15, 15), (16, 16), (17, 16)):
            tensor = {**fixture.tensor, "shape": [dimension, dimension]}
            region = {**unit, "rows": dimension, "cols": dimension}
            report = svd_summary.skeleton(fixture.model, tensor, region, 1)
            self.assertEqual(report["preview"]["shape"], [preview, preview])

    def test_svd_binding_utf8_revision_and_name_edges(self):
        fixture = self.fixture()
        fixture.summary_mode()
        for field, high in (("revision", 256), ("name", 1024)):
            for size, accepted in (
                (0, False),
                (1, True),
                (high, True),
                (high + 1, False),
            ):
                model, tensor = deepcopy(fixture.model), deepcopy(fixture.tensor)
                value = "é" * (size // 2) + "x" * (size % 2)
                if field == "revision":
                    model[field] = value
                    model["model_identity"] = svd_summary.digest(
                        ["weight-atlas-model-v1", model["source_identity"], value]
                    )
                else:
                    tensor[field] = value
                if accepted:
                    svd_summary.binding(model, tensor, fixture.data["region"], 1)
                else:
                    with self.assertRaises(ValueError):
                        svd_summary.binding(model, tensor, fixture.data["region"], 1)


if __name__ == "__main__":
    unittest.main()
