"""Pure protocol and analytic 2×2 backend doubles. No NumPy, processes, or models."""

import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import unittest

from analytics import svd_summary as summary
from analytics.core import permutation
from analytics.worker import validate_request

MODEL = {"source_identity": "a" * 64, "revision": "synthetic numeric fixture"}
MODEL["model_identity"] = summary.digest(
    ["weight-atlas-model-v1", MODEL["source_identity"], MODEL["revision"]]
)
TENSOR = {"id": 0, "name": "synthetic", "dtype": "BF16", "shape": [128, 128]}
REGION = {"row": 0, "col": 0, "rows": 128, "cols": 128}


def protocol_fixture():
    """Declared valid zero result exercises wire schema; it is not a numerical fit."""
    report = summary.skeleton(MODEL, TENSOR, REGION, 77)
    side = {
        "singular_values": [0.0] * 128,
        "energy_fractions": [None] * 128,
        "frobenius_energy": 0.0,
        "rank_one_residual_energy": 0.0,
        "rank_one_residual_energy_fraction": None,
        "zero_energy": True,
        "rank_one_residual_preview": [0.0] * 256,
    }
    report["results"] = {
        "original": copy.deepcopy(side),
        "shuffled": copy.deepcopy(side),
    }
    return report


class Array:
    def __init__(self, rows):
        self.rows = rows

    def reshape(self, rows, cols):
        return Array([self.rows[0][i * cols : (i + 1) * cols] for i in range(rows)])

    def ravel(self):
        return [x for row in self.rows for x in row]

    def __getitem__(self, index):
        a, b = index
        return (
            [row[b] for row in self.rows] if isinstance(a, slice) else self.rows[a][b]
        )

    def __rmul__(self, scalar):
        return Array([[scalar * x for x in row] for row in self.rows])

    def __sub__(self, other):
        return Array(
            [[x - y for x, y in zip(a, b)] for a, b in zip(self.rows, other.rows)]
        )


class AnalyticTwoByTwo:
    """Closed-form eigensystem oracle double: deliberately not a LAPACK qualification."""

    float64 = float

    def __init__(self):
        self.inputs = []
        self.linalg = self

    def asarray(self, data, dtype):
        return Array([list(data)])

    def outer(self, u, v):
        return Array([[x * y for y in v] for x in u])

    def svd(self, matrix, full_matrices):
        self.inputs.append(matrix.ravel())
        assert not full_matrices and len(matrix.rows) == len(matrix.rows[0]) == 2
        (a, b), (c, d) = matrix.rows
        aa, dd, off = a * a + b * b, c * c + d * d, a * c + b * d
        delta = math.hypot(aa - dd, 2 * off)
        singular = [
            math.sqrt(max(0, (aa + dd + delta) / 2)),
            math.sqrt(max(0, (aa + dd - delta) / 2)),
        ]
        theta = 0.5 * math.atan2(2 * off, aa - dd)
        u = Array(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
        )
        vt = []
        for i, sigma in enumerate(singular):
            column = u[:, i]
            vt.append(
                [
                    (column[0] * a + column[1] * c) / sigma,
                    (column[0] * b + column[1] * d) / sigma,
                ]
                if sigma
                else [0.0, 0.0]
            )
        return u, singular, Array(vt)


class SummaryTests(unittest.TestCase):
    def test_closed_admission_and_native_bounds(self):
        data = {"scope": "svd_summary", "tensor": 0, "region": REGION, "seed": 77}
        self.assertIs(validate_request(data, [TENSOR]), TENSOR)
        for bad in [
            {**data, "svd": False},
            {**data, "version": 2},
            {k: v for k, v in data.items() if k != "seed"},
            {**data, "seed": True},
            {**data, "region": {**REGION, "rows": 129}},
            {**data, "region": {**REGION, "cols": 0}},
        ]:
            with self.assertRaises(ValueError):
                validate_request(bad, [TENSOR])
        with self.assertRaises(ValueError):
            summary.check_window([2, 128, 128], REGION)

    def test_complete_shuffle_digest_and_partial_coverage(self):
        tensor = {**TENSOR, "shape": [4096, 4096]}
        r = summary.skeleton(MODEL, tensor, REGION, 77)
        order = permutation(16384, 77)
        self.assertEqual(set(order), set(range(16384)))
        self.assertEqual(
            r["control"]["permutation_sha256"],
            hashlib.sha256(struct.pack("<16384I", *order)).hexdigest(),
        )
        self.assertEqual(r["preview"]["positions"][-1], 15 * 128 + 15)
        self.assertEqual(
            r["control"]["preview_position_to_source"],
            [order[i] for i in r["preview"]["positions"]],
        )
        self.assertEqual(r["preview"]["omitted_values"], 16128)
        self.assertEqual(r["coverage"]["tensor_fraction"], 1 / 1024)
        self.assertFalse(r["coverage"]["full_model"])
        self.assertEqual(
            summary.skeleton(MODEL, TENSOR, REGION, 0)["control"]["effective_seed"],
            0x6D2B79F5,
        )
        self.assertNotEqual(
            r["cache_key"], summary.skeleton(MODEL, tensor, REGION, 78)["cache_key"]
        )

    def test_protocol_shape_budget_and_rejected_corruption(self):
        report = protocol_fixture()
        summary.validate(report, model=MODEL, tensor=TENSOR, region=REGION, seed=77)
        self.assertLess(len(json.dumps(report).encode()), summary.MAX_BODY)
        mutations = [
            lambda r: r.update(extra=True),
            lambda r: r.update(seed=True),
            lambda r: r["control"]["preview_position_to_source"].__setitem__(0, 0),
            lambda r: r["results"]["original"].update(rank_one_residual=[0] * 16384),
            lambda r: r["results"]["original"]["singular_values"].append(0),
            lambda r: r["results"]["original"].update(zero_energy=0),
            lambda r: r["results"]["original"]["rank_one_residual_preview"].__setitem__(
                0, float("nan")
            ),
            lambda r: r["results"]["original"].update(frobenius_energy=10**1000),
        ]
        for mutate in mutations:
            bad = copy.deepcopy(report)
            mutate(bad)
            with self.assertRaises(ValueError):
                summary.validate(
                    bad, model=MODEL, tensor=TENSOR, region=REGION, seed=77
                )
        with self.assertRaisesRegex(ValueError, "revision binding"):
            summary.skeleton({**MODEL, "revision": "different"}, TENSOR, REGION, 77)
        with self.assertRaisesRegex(ValueError, "body cap"):
            summary.validate(
                {**report, "padding": "x" * summary.MAX_BODY},
                model=MODEL,
                tensor=TENSOR,
                region=REGION,
                seed=77,
            )

    def test_worst_numeric_body_budget_and_independent_qualification_preparation(self):
        # Declared maximal-range protocol data for serialization, not fitted values.
        report = protocol_fixture()
        energy = 16384 * summary.MAX_BF16**2
        for side in report["results"].values():
            side.update(
                singular_values=[math.sqrt(energy / 128)] * 128,
                energy_fractions=[1 / 128] * 128,
                frobenius_energy=energy,
                rank_one_residual_energy=energy * 127 / 128,
                rank_one_residual_energy_fraction=127 / 128,
                zero_energy=False,
                rank_one_residual_preview=[-1.2345678901234567e38] * 256,
            )
        summary.validate(report, model=MODEL, tensor=TENSOR, region=REGION, seed=77)
        self.assertLess(len(json.dumps(report).encode()), summary.MAX_BODY)
        import importlib.util
        import tempfile

        spec = importlib.util.spec_from_file_location(
            "summary_qualification", Path(__file__).with_name("qualify_svd_summary.py")
        )
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        self.assertEqual(helper.independent_order(16384, 77), permutation(16384, 77))
        with tempfile.TemporaryDirectory() as temp:
            fixture, request = helper.prepare(Path(temp))
            raw = (fixture / "synthetic.safetensors").read_bytes()
            offset = 8 + struct.unpack("<Q", raw[:8])[0]
            self.assertEqual(len(raw) - offset, 32768)
            words = struct.unpack("<16384H", raw[offset:])
            decoded = [
                struct.unpack("<f", struct.pack("<I", word << 16))[0] for word in words
            ]
            self.assertEqual(
                [decoded[i * 128 + i] for i in range(128)], list(range(128, 0, -1))
            )
            self.assertEqual(sum(x * x for x in decoded), 707264)
            self.assertEqual(request["scope"], "svd_summary")
            self.assertEqual(
                json.loads((Path(temp) / "plan.json").read_text())["status"],
                "PREPARED_NOT_RUN",
            )

    def test_independent_known_spectra_noise_zero_and_exact_signed_multiset(self):
        tensor = {**TENSOR, "shape": [2, 2]}
        region = {**REGION, "rows": 2, "cols": 2}
        for values in (
            [4.0, 0.0, 0.0, 3.0],
            [1.0, 2.0, 2.0, 4.0],
            [0.0, -0.0, 0.0, -0.0],
            [1.0, -2.0, 3.0, 5.0],
        ):
            backend = AnalyticTwoByTwo()
            r = summary.compute_with_numpy(
                backend, values, model=MODEL, tensor=tensor, region=region, seed=1
            )
            self.assertEqual(len(backend.inputs), 2)
            self.assertEqual(backend.inputs[1], [values[i] for i in permutation(4, 1)])
            rawbits = lambda xs: sorted(struct.pack("<d", x) for x in xs)
            self.assertEqual(rawbits(backend.inputs[0]), rawbits(backend.inputs[1]))
            a = r["results"]["original"]
            b = r["results"]["shuffled"]
            if values[0] == 4:
                self.assertAlmostEqual(a["singular_values"][0], 4)
                self.assertAlmostEqual(a["singular_values"][1], 3)
                self.assertAlmostEqual(a["rank_one_residual_energy_fraction"], 9 / 25)
                self.assertAlmostEqual(b["singular_values"][0], 5)
                self.assertAlmostEqual(b["rank_one_residual_energy_fraction"], 0)
            elif values[0] == 1 and values[1] == 2:
                self.assertAlmostEqual(a["singular_values"][0], 5)
                self.assertAlmostEqual(a["rank_one_residual_energy"], 0)
            elif values[0] == 0:
                self.assertTrue(a["zero_energy"])
                self.assertIsNone(a["rank_one_residual_energy_fraction"])
            else:
                self.assertAlmostEqual(
                    a["singular_values"][0] ** 2, (39 + math.sqrt(1037)) / 2
                )
                self.assertAlmostEqual(
                    a["rank_one_residual_energy"], (39 - math.sqrt(1037)) / 2
                )


if __name__ == "__main__":
    unittest.main()
