"""Exact small SVD-summary packets using the existing analytic 2x2 double."""

import ast
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests/analytics"))
from analytics import svd_summary as summary
from test_svd_summary import AnalyticTwoByTwo

FIXTURE = json.loads(
    (ROOT / "tests/fixtures/analytics-summary-contract-v1.json").read_text()
)


class SummaryContracts(unittest.TestCase):
    def setUp(self):
        self.data = deepcopy(FIXTURE["inputs"])
        # The existing unsorted binding digest includes caller region order.
        # Restore the declared recording input order after loading sorted JSON.
        self.data["region"] = {
            key: self.data["region"][key] for key in ("row", "col", "rows", "cols")
        }
        self.kw = {key: self.data[key] for key in ("model", "tensor", "region", "seed")}

    def valid(self, function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as exc:
            self.fail("admitted tiny summary must complete: " + repr(exc))

    def packet(self, actual, expected):
        self.assertEqual(actual, expected)
        self.assertEqual(
            json.dumps(actual, sort_keys=True, separators=(",", ":"), allow_nan=False),
            json.dumps(
                expected, sort_keys=True, separators=(",", ":"), allow_nan=False
            ),
        )

    def test_encoded(self):
        self.assertEqual(
            self.valid(summary.encoded, {"z": "é", "a": 1}), b'{"a":1,"z":"\\u00e9"}'
        )
        with self.assertRaises(ValueError):
            summary.encoded(float("nan"))

    def test_digest(self):
        expected = hashlib.sha256('["é",1]'.encode("utf-8")).hexdigest()
        self.assertEqual(self.valid(summary.digest, ["é", 1]), expected)

    def test_check_window(self):
        for rows, cols in ((128, 1), (1, 128), (128, 128)):
            self.assertIsNone(
                self.valid(
                    summary.check_window,
                    [128, 128],
                    {"row": 0, "col": 0, "rows": rows, "cols": cols},
                )
            )
        with self.assertRaises(ValueError) as caught:
            summary.check_window(
                [129, 129], {"row": 0, "col": 0, "rows": 129, "cols": 1}
            )
        self.assertEqual(
            str(caught.exception),
            "SVD summary accepts at most 128 × 128 / 16384 native BF16 values",
        )

    def test_binding(self):
        before = deepcopy(self.data)
        actual = self.valid(summary.binding, **self.kw)
        self.packet(actual, FIXTURE["binding"])
        self.assertIs(actual["region"], self.data["region"])
        self.assertIs(actual["shape"], self.data["tensor"]["shape"])
        self.assertEqual(self.data, before)

    def test_skeleton(self):
        actual = self.valid(summary.skeleton, **self.kw)
        self.packet(actual, FIXTURE["skeleton"])
        self.assertEqual(actual["control"]["preview_position_to_source"], [0, 2, 1, 3])

    def test_compute_with_numpy(self):
        backend = AnalyticTwoByTwo()
        before = deepcopy(self.data)
        actual = self.valid(
            summary.compute_with_numpy, backend, self.data["values"], **self.kw
        )
        self.packet(actual, FIXTURE["computed"])
        self.assertEqual(backend.inputs, [self.data["values"], self.data["values"]])
        self.assertEqual(self.data, before)
        for result in actual["results"].values():
            self.assertEqual(result["singular_values"], [2.0, 1.0])
            self.assertEqual(result["frobenius_energy"], 5.0)
            self.assertEqual(result["rank_one_residual_energy"], 1.0)

    def test_validate(self):
        report = deepcopy(FIXTURE["computed"])
        before = deepcopy(report)
        self.assertIs(self.valid(summary.validate, report, **self.kw), report)
        self.assertEqual(report, before)
        report["results"]["original"]["extra"] = 1
        with self.assertRaises(ValueError) as caught:
            summary.validate(report, **self.kw)
        self.assertEqual(str(caught.exception), "Invalid closed SVD summary result")

    def test_finite(self):
        source = textwrap.dedent(inspect.getsource(summary.validate))
        found = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef) and node.name == "finite"
        ]
        self.assertEqual(len(found), 1)
        namespace = dict(summary.validate.__globals__)
        try:
            exec(
                compile(
                    ast.Module(body=found, type_ignores=[]),
                    summary.validate.__code__.co_filename,
                    "exec",
                ),
                namespace,
            )
        except Exception as exc:
            self.fail("actual finite helper must initialize: " + repr(exc))
        for value, expected in (
            (1, True),
            (0.5, True),
            (True, False),
            (None, False),
            (float("nan"), False),
            (float("inf"), False),
            (10**1000, False),
        ):
            self.assertIs(self.valid(namespace["finite"], value), expected)


if __name__ == "__main__":
    unittest.main()
