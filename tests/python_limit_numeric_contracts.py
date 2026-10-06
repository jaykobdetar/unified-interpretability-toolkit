"""Numerical boundary contracts with tiny analytic records and in-memory frames."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import profile_worker_primitives as primitive
from analytics import core, source, svd_summary as summary
from atlas_host import profile_snapshot as snapshot


class NumericLimits(unittest.TestCase):
    def heads(self, **changes):
        config = {
            "hidden_size": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "num_hidden_layers": 1,
            "head_dim": 1,
            **changes,
        }
        evidence = {
            "config_sha256": "a" * 64,
            "implementation_sha256": "b" * 64,
            "config_canonical_sha256": core.digest(config),
        }
        profiles = {("a" * 64, "b" * 64): "separate-contiguous-linear-out-in-v1"}
        return core.resolve_heads(
            "model.layers.0.self_attn.q_proj.weight",
            [config["num_attention_heads"] * config["head_dim"], config["hidden_size"]],
            config,
            evidence,
            profiles,
        )

    def test_reviewed_head_dimension_layer_and_gqa_edges(self):
        for field, high in (
            ("hidden_size", 2**20),
            ("num_attention_heads", 4096),
            ("num_key_value_heads", 4096),
            ("num_hidden_layers", 4096),
            ("head_dim", 4096),
        ):
            for n, accepted in ((0, False), (1, True), (high, True), (high + 1, False)):
                changes = {field: n}
                if field == "num_key_value_heads":
                    changes["num_attention_heads"] = n
                with self.subTest(field=field, n=n):
                    self.assertEqual(self.heads(**changes)["available"], accepted)
        for query, kv, accepted in (
            (1, 1, True),
            (4096, 4096, True),
            (4096, 1, True),
            (3, 2, False),
            (1, 2, False),
        ):
            self.assertEqual(
                self.heads(num_attention_heads=query, num_key_value_heads=kv)[
                    "available"
                ],
                accepted,
            )

    def test_folded_matrix_output_edge_in_both_orientations(self):
        for axis in ("row", "column"):
            for dim, other, accepted in (
                (1, 1, True),
                (2, 2048, True),
                (2, 2049, False),
            ):
                heads = {
                    "available": True,
                    "axis": axis,
                    "head_dim": dim,
                    "head_count": 1,
                }
                region = {
                    "row": 0,
                    "col": 0,
                    "rows": 1 if axis == "row" else other,
                    "cols": other if axis == "row" else 1,
                }
                value = core.fold([2.0] * other, region, heads)
                self.assertEqual(value["matrix"]["available"], accepted)
                self.assertEqual(sum(x["count"] for x in value["offsets"]), other)
        # 4097 is prime: legal axis dimensions <=4096 cannot form that
        # rectangle. 4098 is the nearest reachable above-cap product.

    def test_real_layout_evidence_file_size_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            config, implementation = (
                Path(directory) / "config",
                Path(directory) / "implementation",
            )
            config.write_text("{}")
            for size, accepted in ((0, True), (1048576, True), (1048577, False)):
                implementation.write_bytes(b"x" * size)
                if accepted:
                    source.read_layout_evidence(config, implementation)
                else:
                    with self.assertRaisesRegex(ValueError, "at most 1 MiB"):
                        source.read_layout_evidence(config, implementation)
            implementation.write_bytes(b"")
            for size, accepted in ((1048576, True), (1048577, False)):
                config.write_bytes(b"{}" + b" " * (size - 2))
                if accepted:
                    source.read_layout_evidence(config, implementation)
                else:
                    with self.assertRaisesRegex(ValueError, "at most 1 MiB"):
                        source.read_layout_evidence(config, implementation)
            # Empty implementation is valid source evidence; empty config cannot
            # encode a JSON object, independently of its byte-bound admission.

    def snapshot_case(self, records, dtype="F32", visited=1):
        fixture = primitive.Base(methodName="runTest")
        fixture.setUp()
        try:
            selected = primitive.selected(1, 1)
            selected["dtype"] = dtype
            raw, *_ = primitive.frame(selected, visited=visited)
            raw = raw[:-96] + b"".join(struct.pack("<ddQ", *row) for row in records)
            return fixture.validate(raw, selected)
        finally:
            fixture.doCleanups()

    def test_snapshot_finite_sum_and_compensation_edges(self):
        self.snapshot_case([(0.0, 0.0, 0)] * 4, visited=0)
        for dtype, maximum in (
            ("F16", 65504.0),
            ("BF16", float.fromhex("0x1.fep+127")),
            ("F32", float.fromhex("0x1.fffffep+127")),
        ):
            for value, accepted in (
                (0.0, True),
                (maximum, True),
                (math.nextafter(0.0, -math.inf), False),
                (math.nextafter(maximum, math.inf), False),
                (math.inf, False),
                (math.nan, False),
            ):
                if accepted:
                    self.snapshot_case([(value, 0.0, 1)] * 4, dtype)
                else:
                    with self.assertRaises(ValueError):
                        self.snapshot_case([(value, 0.0, 1)] * 4, dtype)
        cap = 64 * 2**-52
        for correction, accepted in (
            (-cap, True),
            (cap, True),
            (math.nextafter(-cap, -math.inf), False),
            (math.nextafter(cap, math.inf), False),
        ):
            if accepted:
                self.snapshot_case([(1.0, correction, 1)] * 4)
            else:
                with self.assertRaisesRegex(ValueError, "compensation"):
                    self.snapshot_case([(1.0, correction, 1)] * 4)

    def test_snapshot_paired_sum_tolerance_representable_neighbors(self):
        low = 1 - 128 * 2**-52
        high = 1 / (1 - 128 * 2**-52)
        for value, accepted in (
            (low, True),
            (high, True),
            (math.nextafter(low, -math.inf), False),
            (math.nextafter(high, math.inf), False),
        ):
            records = [(1.0, 0.0, 1), (value, 0.0, 1), (1.0, 0.0, 1), (1.0, 0.0, 1)]
            if accepted:
                self.snapshot_case(records)
            else:
                with self.assertRaisesRegex(ValueError, "paired sums"):
                    self.snapshot_case(records)

    def report(self, energy=1.0):
        model = {"source_identity": "a" * 64, "revision": "synthetic"}
        model["model_identity"] = summary.digest(
            ["weight-atlas-model-v1", model["source_identity"], model["revision"]]
        )
        tensor = {"id": 0, "name": "synthetic", "dtype": "BF16", "shape": [1, 1]}
        region = {"row": 0, "col": 0, "rows": 1, "cols": 1}
        report = summary.skeleton(model, tensor, region, 17)
        side = {
            "singular_values": [math.sqrt(energy)],
            "energy_fractions": [1.0] if energy else [None],
            "frobenius_energy": energy,
            "rank_one_residual_energy": 0.0,
            "rank_one_residual_energy_fraction": 0.0 if energy else None,
            "zero_energy": energy == 0,
            "rank_one_residual_preview": [0.0],
        }
        report["results"] = {key: deepcopy(side) for key in ("original", "shuffled")}
        return report, {"model": model, "tensor": tensor, "region": region, "seed": 17}

    def test_svd_energy_and_residual_preview_edges(self):
        maximum = float.fromhex("0x1.fep+127")
        for energy, accepted in (
            (0.0, True),
            (maximum**2, True),
            (math.nextafter(0.0, -math.inf), False),
            (math.nextafter(maximum**2, math.inf), False),
        ):
            if energy < 0:
                report, kwargs = self.report()
                report["results"]["original"]["frobenius_energy"] = energy
            else:
                report, kwargs = self.report(energy)
            if accepted:
                summary.validate(report, **kwargs)
            else:
                with self.assertRaises(ValueError):
                    summary.validate(report, **kwargs)
        for preview, accepted in (
            (-2 * maximum, True),
            (2 * maximum, True),
            (math.nextafter(-2 * maximum, -math.inf), False),
            (math.nextafter(2 * maximum, math.inf), False),
        ):
            report, kwargs = self.report()
            report["results"]["original"]["rank_one_residual_preview"] = [preview]
            if accepted:
                summary.validate(report, **kwargs)
            else:
                with self.assertRaisesRegex(ValueError, "numeric arrays"):
                    summary.validate(report, **kwargs)

    def test_svd_residual_ratio_tolerance_and_zero_rules(self):
        cap = 1 + 1e-10
        for ratio, accepted in (
            (0.0, True),
            (cap, True),
            (math.nextafter(0.0, -math.inf), False),
            (math.nextafter(cap, math.inf), False),
        ):
            report, kwargs = self.report()
            for side in report["results"].values():
                side["rank_one_residual_energy"] = ratio
                side["rank_one_residual_energy_fraction"] = ratio
            if accepted:
                summary.validate(report, **kwargs)
            else:
                with self.assertRaises(ValueError):
                    summary.validate(report, **kwargs)
        report, kwargs = self.report(0)
        summary.validate(report, **kwargs)
        for key, value in (
            ("singular_values", [1e-300]),
            ("energy_fractions", [0]),
            ("rank_one_residual_energy", 1e-300),
            ("rank_one_residual_energy_fraction", 0),
            ("rank_one_residual_preview", [1e-300]),
        ):
            bad = deepcopy(report)
            bad["results"]["original"][key] = value
            with self.assertRaises(ValueError):
                summary.validate(bad, **kwargs)


if __name__ == "__main__":
    unittest.main()
