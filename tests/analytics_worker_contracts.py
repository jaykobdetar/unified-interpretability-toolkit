"""Trusted tiny-source and inert protocol contracts for the analytics worker."""

import copy
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent / "analytics"))
from analytics import worker
from analytics.source import Catalog
import test_service as service_fixture


class WorkerContracts(unittest.TestCase):
    def setUp(self):
        # Compose the established tiny BF16 fixture without inheriting its tests.
        self.fixture = service_fixture.AdapterTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.fixture.data["seed"] = 7

    def outcome(self, function, *args):
        try:
            return function(*args), None
        except (ValueError, TypeError, OSError) as error:
            return None, (type(error), str(error))

    def test_unique(self):
        marker = []
        result = worker.unique([("a", marker), ("b", 7)])
        self.assertEqual(result, {"a": [], "b": 7})
        self.assertIs(result["a"], marker)
        self.assertEqual(
            self.outcome(worker.unique, [("x", 1), ("x", 2)]),
            (None, (ValueError, "Duplicate JSON key")),
        )

    def test_parse_json(self):
        self.assertEqual(
            worker.parse_json(b'{"a":[1,2],"b":false}'), {"a": [1, 2], "b": False}
        )
        self.assertEqual(
            self.outcome(worker.parse_json, '{"x":1,"x":2}'),
            (None, (ValueError, "Duplicate JSON key")),
        )
        self.assertEqual(
            self.outcome(worker.parse_json, "NaN"),
            (None, (ValueError, "Nonfinite JSON")),
        )

    def test_validate_request(self):
        f = self.fixture
        chosen, error = self.outcome(
            worker.validate_request, f.data, f.model["catalog"]
        )
        self.assertIsNone(error)
        self.assertIs(chosen, f.tensor)
        self.assertEqual(
            self.outcome(
                worker.validate_request,
                {**f.data, "scope": "unsupported"},
                f.model["catalog"],
            ),
            (None, (ValueError, "Unknown analytics scope")),
        )
        self.assertIsNone(
            worker.validate_request({"scope": "model", "seed": 7}, f.model["catalog"])
        )

    def test_catalog_from_payload(self):
        f = self.fixture
        original_verify = Catalog.verify
        with mock.patch.object(
            Catalog, "verify", autospec=True, side_effect=original_verify
        ) as verify:
            catalog = worker.catalog_from_payload(f.payload())
        self.assertIsInstance(catalog, Catalog)
        verify.assert_called_once_with(catalog)
        self.assertEqual(catalog.host_identity, "trusted-fixture")
        self.assertEqual(
            [(t.name, t.shape, t.byte_offset) for t in catalog.tensors],
            [("fixture", (2, 3), f.tensor["byte_offset"])],
        )
        self.assertEqual(
            catalog.read(catalog.tensors[0], f.data["region"]),
            [1.0, -2.0, 0.0, 4.0, 5.0, -6.0],
        )
        invalid = copy.deepcopy(f.payload())
        invalid["model"]["catalog"][0]["id"] = 1
        self.assertEqual(
            self.outcome(worker.catalog_from_payload, invalid),
            (None, (ValueError, "Catalog IDs must be a complete native sequence")),
        )

    def test_analyze_request(self):
        f = self.fixture
        result = worker.analyze_request(f.payload())
        self.assertEqual(result["tensor_id"], 0)
        self.assertEqual(result["host_source_identity"], "trusted-fixture")
        self.assertEqual(result["control"]["seed"], 7)
        self.assertIs(result["region"], f.data["region"])
        self.assertEqual(result["original"]["top_values"][0]["value"], -6.0)
        region = f.data["region"]
        f.data = {"scope": "model", "seed": 7}
        with mock.patch("analytics.source.time.monotonic", return_value=0.0):
            result = worker.analyze_request(f.payload())
        self.assertEqual(result["coverage"]["visited_values"], 6)
        self.assertIs(result["coverage"]["full_model"], True)
        self.assertEqual(result["control"]["seed"], 7)

        # A declared passthrough double checks the optional numerical call boundary.
        f.data = {"tensor": 0, "region": region, "seed": 7, "svd": True}
        backend = object()
        packet = {"fixture": [1.0, None]}
        with (
            mock.patch.dict(sys.modules, {"numpy": backend}),
            mock.patch(
                "analytics.svd.compute_with_numpy", return_value=packet
            ) as compute,
        ):
            result = worker.analyze_request(f.payload())
        self.assertIs(result["svd"]["results"], packet)
        self.assertIs(result["svd"]["available"], True)
        self.assertEqual(result["svd"]["scope"], "full tensor")
        compute.assert_called_once_with(
            backend, [1.0, -2.0, 0.0, 4.0, 5.0, -6.0], 2, 3, 7
        )

        f.summary_mode()
        summary_packet = f.summary_event()["result"]
        with (
            mock.patch.dict(sys.modules, {"numpy": backend}),
            mock.patch(
                "analytics.svd_summary.compute_with_numpy", return_value=summary_packet
            ) as compute,
        ):
            result = worker.analyze_request(f.payload())
        self.assertIs(result, summary_packet)
        compute.assert_called_once_with(
            backend,
            [1.0, -2.0, 0.0, 4.0, 5.0, -6.0],
            model=f.model,
            tensor=f.tensor,
            region=region,
            seed=77,
        )

    def test_main(self):
        incoming = io.BytesIO(b'{"probe":7}')
        output = io.BytesIO()
        with (
            mock.patch.object(worker, "configure") as configure,
            mock.patch.object(worker.sys, "stdin", SimpleNamespace(buffer=incoming)),
            mock.patch.object(incoming, "read", wraps=incoming.read) as read,
            mock.patch.object(worker.sys, "stdout", SimpleNamespace(buffer=output)),
            mock.patch.object(
                worker, "analyze_request", return_value={"fixture": [1, None]}
            ) as analyze,
        ):
            self.assertIsNone(worker.main())
        configure.assert_called_once_with()
        read.assert_called_once_with(worker.MAX_INPUT + 1)
        analyze.assert_called_once_with({"probe": 7})
        self.assertEqual(
            output.getvalue(), b'{"ok":true,"result":{"fixture":[1,null]}}\n'
        )
        for error, expected in (
            (
                ValueError("fixture admission"),
                b'{"ok": false, "error": "fixture admission"}\n',
            ),
            (
                RuntimeError("fixture private"),
                b'{"ok": false, "error": "Analysis failed; source or optional runtime unavailable"}\n',
            ),
        ):
            with self.subTest(error=type(error)):
                output = io.BytesIO()
                with (
                    mock.patch.object(worker, "configure"),
                    mock.patch.object(
                        worker.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"{}"))
                    ),
                    mock.patch.object(
                        worker.sys, "stdout", SimpleNamespace(buffer=output)
                    ),
                    mock.patch.object(worker, "analyze_request", side_effect=error),
                ):
                    self.assertIsNone(worker.main())
                self.assertEqual(output.getvalue(), expected)


if __name__ == "__main__":
    unittest.main()
