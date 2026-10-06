"""Static metadata guards with inert owner, descriptor and read doubles."""

from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import static_model_contracts as fixture
from atlas_host import dense_static_admission as dense, static_models as static
from atlas_host.common import canonical


class StaticMetadataLimits(unittest.TestCase):
    def prepare(self, *, count=1, metadata=None, index_metadata=None, axis=1):
        shapes = {f"tensor{n}": [1, axis] for n in range(count)}
        header = {
            name: {
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [n * axis * 2, (n + 1) * axis * 2],
            }
            for n, (name, shape) in enumerate(shapes.items())
        }
        if metadata is not None:
            header["__metadata__"] = metadata
        raw = canonical(header)
        files = [
            {"name": "config.json", "bytes": 2},
            {"name": "model.safetensors", "bytes": 8 + len(raw) + count * axis * 2},
        ]
        reads = {"config.json": b"{}", "model.safetensors": raw}
        if index_metadata is not None:
            index = canonical(
                {
                    "weight_map": {name: "model.safetensors" for name in shapes},
                    "metadata": index_metadata,
                }
            )
            reads["model.safetensors.index.json"] = index
            files.append(
                {
                    "name": "model.safetensors.index.json",
                    "bytes": len(index),
                    "sha256": hashlib.sha256(index).hexdigest(),
                }
            )
        entry = {
            "root": "/synthetic",
            "manifest": {
                "revision": "a" * 40,
                "provenance": "owner_expected",
                "files": files,
            },
            "fingerprints": {
                f["name"]: {
                    "bytes": f["bytes"],
                    "device": 1,
                    "inode": 1,
                    "mtime_ns": 1,
                    "ctime_ns": 1,
                }
                for f in files
            },
        }
        registry = NS(owner_receipt=Mock(return_value=entry))
        descriptor = {"parameter_shapes": shapes, "parameter_count": count * axis}
        # Descriptor expansion isolates the published 4096-header guard, whose
        # upper boundary is masked by the smaller canonical architecture inventory.
        with (
            patch.object(static, "_current"),
            patch.object(
                static, "_read", side_effect=lambda entry, name, *a, **k: reads[name]
            ),
            patch.object(static, "pinned_descriptor", return_value=descriptor),
        ):
            return static.StaticPolicy(registry).prepare("synthetic")

    def test_header_count_declared_edges_with_descriptor_double(self):
        for count, accepted in ((0, False), (1, True), (4096, True), (4097, False)):
            with self.subTest(count=count):
                if accepted:
                    self.assertEqual(len(self.prepare(count=count).tensors), count)
                else:
                    with self.assertRaisesRegex(
                        ValueError, "bounded static tensor header"
                    ):
                        self.prepare(count=count)

    def test_header_metadata_fields_key_and_value_lengths(self):
        for field, values in (
            ("count", (0, 64, 65)),
            ("key", (0, 128, 129)),
            ("value", (0, 4096, 4097)),
        ):
            for length in values:
                metadata = (
                    {str(n): "" for n in range(length)}
                    if field == "count"
                    else {
                        "k" * length if field == "key" else "key": (
                            "v" * length if field == "value" else ""
                        )
                    }
                )
                with self.subTest(field=field, length=length):
                    if length == values[-1]:
                        with self.assertRaisesRegex(ValueError, "safetensors metadata"):
                            self.prepare(metadata=metadata)
                    else:
                        self.prepare(metadata=metadata)

    def test_index_metadata_count_key_and_integer_edges(self):
        for metadata, accepted in (
            ({}, True),
            ({str(n): None for n in range(64)}, True),
            ({str(n): None for n in range(65)}, False),
            ({"": True}, True),
            ({"k" * 128: ""}, True),
            ({"k" * 129: ""}, False),
            ({"n": -(2**63)}, True),
            ({"n": -(2**63) - 1}, False),
            ({"n": 2**64 - 1}, True),
            ({"n": 2**64}, False),
            ({"n": 1.0}, False),
        ):
            with self.subTest(metadata=metadata):
                if accepted:
                    self.prepare(index_metadata=metadata)
                else:
                    with self.assertRaisesRegex(ValueError, "index metadata"):
                        self.prepare(index_metadata=metadata)

    def test_display_axis_bound_and_bind_encoded_guard(self):
        self.prepare(axis=1)
        self.prepare(axis=200000)
        with self.assertRaisesRegex(ValueError, "display axis"):
            self.prepare(axis=200001)
        case = fixture.StaticContracts(methodName="runTest")
        case.setUp()
        self.addCleanup(case.doCleanups)
        prepared = case.policy.prepare(case.identifier)
        baseline = case.native_model(prepared)
        for size, accepted in ((2097152, True), (2097153, False)):
            value = deepcopy(baseline)
            value["padding"] = ""
            value["padding"] = "x" * (size - len(canonical(value)))
            self.assertEqual(len(canonical(value)), size)
            if accepted:
                case.policy.bind(prepared, value, reader_ready=True)
            else:
                with self.assertRaisesRegex(ValueError, "bounded Rust model metadata"):
                    case.policy.bind(prepared, value, reader_ready=True)

    def model(self, count=2):
        tensor = {
            "count": count,
            "calibration_complete": True,
            "max_abs": 1.0,
            "median_nonzero_abs": 0.0,
            "q99": 1.0,
            "quantile_order_statistics": "exact",
            "quantile_interpolation": "linear in F64; final floating-point rounding possible",
            "robust_clipped_count": 0,
            "exact_zero_count": 0,
            "unique_bit_patterns": 1,
            "calibration_method": "exact-16-bit-histogram",
            "rule_status": {
                "tensor_signed_percentile": "supported dtype; requires a valid exact histogram"
            },
        }
        return {
            "name": "Owner",
            "fresh_source_hashes": None,
            "coverage": {
                "sha_hashed_shards": 0,
                "sha_expected_matched_shards": 0,
                "sha_missing_expected_shards": 0,
                "sha_verified_shards": 0,
                "source_complete": True,
                "active_tensor": None,
                "all_requested": False,
                "calibration_error": None,
                "materialized_bytes": 0,
                "materialized_tiles": 0,
                "cache_budget_bytes": 2 * 1024**3,
                "fine_tile_file_cap": 1000,
                "calibrated_tensors": 1,
                "values_streamed": count,
                "statistics_complete": True,
            },
            "catalog": [tensor],
            "calibration_complete": True,
            "global_max": 1.0,
        }

    def test_native_cache_calibration_integer_and_statistic_edges(self):
        for key, low, high, target in (
            ("materialized_bytes", 0, 2 * 1024**3, "coverage"),
            ("materialized_tiles", 0, 1000, "coverage"),
            ("unique_bit_patterns", 1, 65536, "tensor"),
            ("robust_clipped_count", 0, 2, "tensor"),
            ("exact_zero_count", 0, 2, "tensor"),
            ("median_nonzero_abs", 0, 1, "tensor"),
            ("q99", 0, 1, "tensor"),
        ):
            for number, accepted in (
                (low - 1, False),
                (low, True),
                (high, True),
                (high + 1, False),
            ):
                value = self.model()
                (value["coverage"] if target == "coverage" else value["catalog"][0])[
                    key
                ] = number
                with self.subTest(key=key, number=number):
                    if accepted:
                        dense.check_native_model(NS(entry={"name": "Owner"}), value)
                    else:
                        with self.assertRaises(ValueError):
                            dense.check_native_model(NS(entry={"name": "Owner"}), value)
        for count, accepted in (
            (0, False),
            (1, True),
            (2**63 - 1, True),
            (2**63, False),
        ):
            value = self.model(count)
            if accepted:
                dense.check_native_model(NS(entry={"name": "Owner"}), value)
            else:
                with self.assertRaises(ValueError):
                    dense.check_native_model(NS(entry={"name": "Owner"}), value)


if __name__ == "__main__":
    unittest.main()
