"""Exact inert analytics records and independently checked small arithmetic."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from analytics import core

FIXTURE = json.loads(
    (ROOT / "tests/fixtures/analytics-core-contract-v1.json").read_text()
)


class AnalyticsCoreTests(unittest.TestCase):
    def setUp(self):
        self.data = deepcopy(FIXTURE["inputs"])

    def valid(self, function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as exc:
            self.fail("admitted inert analytics operation must complete: " + repr(exc))

    def error(self, function, message, *args):
        try:
            function(*args)
        except ValueError as exc:
            self.assertEqual(str(exc), message)
        except Exception as exc:
            self.fail("expected local ValueError: " + repr(exc))
        else:
            self.fail("expected local ValueError: " + message)

    def packet(self, actual, expected):
        self.assertEqual(actual, expected)
        self.assertEqual(
            json.dumps(actual, sort_keys=True, separators=(",", ":"), allow_nan=False),
            json.dumps(
                expected, sort_keys=True, separators=(",", ":"), allow_nan=False
            ),
        )

    def test_integer(self):
        for value in (1, 8):
            self.assertEqual(self.valid(core.integer, value, 1, 8, "fixture"), value)
        self.error(
            core.integer, "fixture must be an integer in [1, 8]", True, 1, 8, "fixture"
        )

    def test_canonical(self):
        self.assertEqual(
            self.valid(core.canonical, {"z": "é", "a": 1}), '{"a":1,"z":"\\u00e9"}'
        )
        with self.assertRaises(ValueError):
            core.canonical(float("nan"))

    def test_digest(self):
        expected = hashlib.sha256(b'{"a":1,"z":2}').hexdigest()
        self.assertEqual(self.valid(core.digest, {"z": 2, "a": 1}), expected)

    def test_geometry(self):
        actual = self.valid(core.geometry, self.data["shape"], self.data["region"])
        self.assertIsInstance(actual, tuple)
        self.assertEqual(actual, (1, 2, 2, 3))
        self.error(
            core.geometry,
            "Only native vectors and matrices are supported",
            [],
            self.data["region"],
        )

    def test_checked_values(self):
        self.assertIsNone(
            self.valid(
                core.checked_values,
                self.data["values"],
                self.data["shape"],
                self.data["region"],
            )
        )
        self.assertIsNone(
            self.valid(
                core.checked_values,
                [3.39e38],
                [1],
                {"row": 0, "col": 0, "rows": 1, "cols": 1},
            )
        )
        self.error(
            core.checked_values,
            "Region value count mismatch",
            [1],
            self.data["shape"],
            self.data["region"],
        )

    def test_permutation(self):
        for seed, name in ((0, "zero"), (42, "forty_two")):
            actual = self.valid(core.permutation, 8, seed)
            self.packet(actual, FIXTURE["permutations"][name])
            self.assertEqual(sorted(actual), list(range(8)))
        self.assertEqual(self.valid(core.permutation, 0, 0), [])

    def test_ranked(self):
        values = self.data["ranked"]
        before = deepcopy(values)
        result = self.valid(core.ranked, values, 2)
        self.packet(result, FIXTURE["ranked"])
        self.assertIs(result[0], values[1])
        self.assertIs(result[1], values[0])
        self.assertEqual(values, before)

    def test_statistics(self):
        actual = self.valid(
            core.statistics,
            self.data["values"],
            self.data["shape"],
            self.data["region"],
            3,
        )
        self.packet(actual, FIXTURE["statistics"])
        self.assertEqual([row["mean_abs"] for row in actual["rows"]], [1, 5])
        self.assertEqual([row["mean_abs"] for row in actual["columns"]], [2.5, 3.5, 3])

    def test_resolve_heads(self):
        profiles = {
            ("fixture-config", "fixture-layout"): "separate-contiguous-linear-out-in-v1"
        }
        for name, shape in (("q", [4, 4]), ("k", [2, 4]), ("o", [4, 4])):
            actual = self.valid(
                core.resolve_heads,
                f"model.layers.0.self_attn.{name}_proj.weight",
                shape,
                self.data["config"],
                self.data["evidence"],
                profiles,
            )
            self.packet(actual, FIXTURE["heads"][name])
            self.assertIs(actual["evidence"], self.data["evidence"])

    def test_fold(self):
        region = {"row": 0, "col": 0, "rows": 4, "cols": 4}
        for name in ("q", "o"):
            actual = self.valid(
                core.fold, list(range(1, 17)), region, FIXTURE["heads"][name]
            )
            self.packet(actual, FIXTURE["folds"][name])
            self.assertIs(actual["region"], region)
        self.assertIsNone(self.valid(core.fold, [], region, {"available": False}))

    def test_analyze(self):
        before = deepcopy(self.data)
        actual = self.valid(
            core.analyze,
            self.data["values"],
            source_identity="fixture-source-v1",
            tensor="matrix",
            shape=self.data["shape"],
            region=self.data["region"],
            seed=42,
            top=3,
        )
        self.packet(actual, FIXTURE["analyze"])
        self.assertIs(actual["region"], self.data["region"])
        self.assertIs(actual["shape"], self.data["shape"])
        self.assertEqual(self.data, before)


if __name__ == "__main__":
    unittest.main()
