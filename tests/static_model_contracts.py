"""Small registered dense files and inert Rust metadata; no renderer/ML/process."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from atlas_host.common import canonical
from atlas_host.registry import Registry
from atlas_host.static_models import StaticPolicy, native_source_identity
from inference_model_descriptor import describe_config, parameter_shapes
from model_descriptor_contracts import config


class StaticContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.config = config()
        shapes = parameter_shapes(describe_config(self.config))
        self.header = {}
        self.offset = 0
        for index, (name, shape) in enumerate(shapes.items()):
            dtype = ("BF16", "F16", "F32")[index % 3]
            width = 4 if dtype == "F32" else 2
            end = self.offset + math.prod(shape) * width
            self.header[name] = {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [self.offset, end],
            }
            self.offset = end
        self.raw = canonical(self.header)
        self.shards = {"model.safetensors": (self.raw, self.header)}
        data = {
            "model.safetensors": struct.pack("<Q", len(self.raw))
            + self.raw
            + bytes(self.offset),
            "config.json": canonical(self.config),
            "LICENSE": b"public synthetic license",
        }
        for name, raw in data.items():
            (self.source / name).write_bytes(raw)
        self.manifest = {
            "version": 1,
            "repository": "fixtures/static-qwen3",
            "revision": "a" * 40,
            "provenance": "owner_expected",
            "license": {"id": "Apache-2.0", "accepted": True, "file": "LICENSE"},
            "files": [
                {
                    "name": name,
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
                for name, raw in data.items()
            ],
        }
        self.registry = Registry(
            self.root / "registry.json", free_bytes=lambda _: 100 * 1024**3
        )
        self.identifier = self.registry.register(
            self.source, self.manifest, "Registered Qwen", max_bytes=65536
        )["model_id"]
        self.registry.set_enabled(self.identifier, True)
        self.policy = StaticPolicy(self.registry)

    def native_model(self, prepared):
        tensors = []
        inventory = {
            name: (shard_id, shard, raw, item)
            for shard_id, shard in enumerate(sorted(self.shards))
            for raw, header in [self.shards[shard]]
            for name, item in header.items()
        }
        for tensor_id, name in enumerate(sorted(inventory)):
            shard_id, shard, raw, item = inventory[name]
            shape = item["shape"]
            dtype = item["dtype"]
            rows, cols = (1, shape[0]) if len(shape) == 1 else shape
            tensors.append(
                {
                    "id": tensor_id,
                    "name": name,
                    "shape": deepcopy(shape),
                    "rows": rows,
                    "cols": cols,
                    "count": math.prod(shape),
                    "dtype": dtype,
                    "element_bytes": 4 if dtype == "F32" else 2,
                    "shard": shard,
                    "shard_id": shard_id,
                    "byte_offset": 8 + len(raw) + item["data_offsets"][0],
                    "available": True,
                    "unavailable_reason": None,
                    "min_level": 0,
                    "max_level": (max(rows, cols) - 1).bit_length(),
                    "slice_required": False,
                    "display_axes": [0] if len(shape) == 1 else [0, 1],
                    "private_note": "/private/hidden",
                }
            )
        return {
            "api_version": 1,
            "source_directory": str(self.source),
            "revision": "a" * 40,
            "source_identity": prepared.source_identity,
            "model_identity": prepared.model_identity,
            "source_bytes": sum(
                (self.source / name).stat().st_size for name in self.shards
            ),
            "header_bytes_read": sum(8 + len(raw) for raw, _ in self.shards.values()),
            "parameter_count": sum(
                math.prod(item["shape"]) for item in self.header.values()
            ),
            "catalog": tensors,
            "inference_source_model": {"stale": True},
            "head_layout": {"runtime_verified": True},
            "head_layout_binding": {"stale": True},
            "private_diagnostic": "/private/not-public",
            "coverage": {"calibration_error": "/private/error"},
        }

    def register_current(self):
        manifest = deepcopy(self.manifest)
        manifest["files"] = [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in self.source.iterdir()
        ]
        identifier = self.registry.register(
            self.source, manifest, "Updated source", max_bytes=65536
        )["model_id"]
        self.registry.set_enabled(identifier, True)
        return identifier

    def test_catalog_never_reads_config_headers_or_promotes_candidate_to_ready(self):
        with (
            patch(
                "atlas_host.static_models._read",
                side_effect=AssertionError("No source content on catalog"),
            ),
            patch.object(
                self.policy,
                "prepare",
                side_effect=AssertionError("No activation on catalog"),
            ),
        ):
            item = self.policy.catalog()["models"][0]
        self.assertTrue(item["static_view_candidate"])
        self.assertFalse(item["static_view_ready"])
        self.assertFalse(item["inference_ready"])
        self.assertNotIn("root", canonical(item).decode())

    def test_complete_mixed_dense_catalog_binds_without_inference_or_stale_annotations(
        self,
    ):
        prepared = self.policy.prepare(self.identifier)
        model = self.native_model(prepared)
        binding = self.policy.bind(prepared, model, reader_ready=True)
        result = self.policy.project_model(binding, model, reader_ready=True)
        self.assertEqual(
            result["static_model_descriptor"]["projection_mappings"]["o_proj"]["shape"],
            [8, 12],
        )
        self.assertTrue(result["static_view_ready"])
        self.assertFalse(result["inference_ready"])
        self.assertFalse(result["fit_verified"])
        for key in (
            "source_directory",
            "inference_source_model",
            "head_layout",
            "head_layout_binding",
            "private_diagnostic",
        ):
            self.assertNotIn(key, result)
        self.assertFalse(result["static_binding"]["fresh_payload_hashes_recomputed"])
        self.assertEqual(result["coverage"]["calibration_error"], "calibration_failed")
        self.assertNotIn("/private", canonical(result).decode())
        self.assertTrue(
            self.policy.catalog(active_binding=binding, reader_ready=True)["models"][0][
                "static_view_ready"
            ]
        )
        self.assertFalse(
            self.policy.catalog(active_binding=binding)["models"][0][
                "static_view_ready"
            ]
        )

    def test_renderer_forgery_of_identity_dtype_shape_offset_or_coverage_is_rejected(
        self,
    ):
        prepared = self.policy.prepare(self.identifier)
        model = self.native_model(prepared)
        changes = [
            lambda m: m.update(source_identity="b" * 64),
            lambda m: m.update(revision="c" * 40),
            lambda m: m.update(source_directory="/another/source"),
            lambda m: m.update(parameter_count=m["parameter_count"] + 1),
            lambda m: m["catalog"].pop(),
            lambda m: m["catalog"][0].update(
                dtype="F16" if m["catalog"][0]["dtype"] == "F32" else "F32"
            ),
            lambda m: m["catalog"][0].update(shape=[1, 88]),
            lambda m: m["catalog"][0].update(byte_offset=0),
            lambda m: m["catalog"][1].update(id=0),
            lambda m: m.update(comparison_identity="d" * 64),
            lambda m: m["catalog"][0].update(max_level=0),
            lambda m: m["catalog"][0].update(slice_required=True),
            lambda m: m.update(header_bytes_read=0),
            lambda m: m.update(api_version=True),
            lambda m: m["catalog"].reverse(),
            lambda m: m["catalog"][0].update(display_axes=[False, True]),
        ]
        for index, change in enumerate(changes):
            candidate = deepcopy(model)
            change(candidate)
            with self.subTest(change=index), self.assertRaises(ValueError):
                self.policy.bind(prepared, candidate, reader_ready=True)

    def test_owner_source_changes_disable_old_static_proof_without_catalog_payload_read(
        self,
    ):
        prepared = self.policy.prepare(self.identifier)
        model = self.native_model(prepared)
        binding = self.policy.bind(prepared, model, reader_ready=True)
        path = self.source / "model.safetensors"
        path.write_bytes(path.read_bytes()[:-1] + b"x")
        with patch(
            "atlas_host.static_models._read",
            side_effect=AssertionError("Catalog is metadata-only"),
        ):
            item = self.policy.catalog(active_binding=binding, reader_ready=True)[
                "models"
            ][0]
        self.assertFalse(item["static_view_candidate"])
        self.assertFalse(item["static_view_ready"])
        with self.assertRaises(ValueError):
            self.policy.project_model(binding, model, reader_ready=True)

    def test_unlisted_model_api_or_extra_shard_prevents_activation_and_candidate(self):
        for name in ("model-api.json", "extra.safetensors"):
            path = self.source / name
            path.write_bytes(b"inert unlisted data")
            self.assertFalse(
                self.policy.catalog()["models"][0]["static_view_candidate"]
            )
            with self.assertRaises(ValueError):
                self.policy.prepare(self.identifier)
            path.unlink()

    def test_relocated_owner_directory_is_not_a_current_canonical_source(self):
        prepared = self.policy.prepare(self.identifier)
        binding = self.policy.bind(
            prepared, self.native_model(prepared), reader_ready=True
        )
        moved = self.root / "moved"
        self.source.rename(moved)
        self.source.symlink_to(moved, target_is_directory=True)
        self.assertFalse(
            self.policy.catalog(active_binding=binding, reader_ready=True)["models"][0][
                "static_view_ready"
            ]
        )
        with self.assertRaises(ValueError):
            self.policy.prepare(self.identifier)

    def test_sharded_header_index_and_renderer_correspondence(self):
        (self.source / "model.safetensors").unlink()
        self.shards = {}
        weight_map = {}
        items = list(self.header.items())
        for number, group in enumerate((items[::2], items[1::2])):
            shard = f"model-{number+1}.safetensors"
            header = {}
            cursor = 0
            for name, item in group:
                item = deepcopy(item)
                extent = item["data_offsets"][1] - item["data_offsets"][0]
                item["data_offsets"] = [cursor, cursor + extent]
                cursor += extent
                header[name] = item
                weight_map[name] = shard
            raw = canonical(header)
            self.shards[shard] = (raw, header)
            (self.source / shard).write_bytes(
                struct.pack("<Q", len(raw)) + raw + bytes(cursor)
            )
        index = {"metadata": {"total_size": self.offset}, "weight_map": weight_map}
        path = self.source / "model.safetensors.index.json"
        path.write_bytes(canonical(index))
        identifier = self.register_current()
        prepared = self.policy.prepare(identifier)
        model = self.native_model(prepared)
        binding = self.policy.bind(prepared, model, reader_ready=True)
        self.assertTrue(
            self.policy.project_model(binding, model, reader_ready=True)[
                "static_view_ready"
            ]
        )
        # Independently registered wrong maps and noncanonical numeric metadata
        # never gain readiness merely because their full file hashes were saved.
        for change in ("wrong_map", "float_metadata", "oversized_integer"):
            altered = deepcopy(index)
            if change == "wrong_map":
                altered["weight_map"][next(iter(weight_map))] = "model-2.safetensors"
            elif change == "float_metadata":
                altered["metadata"]["total_size"] = 1.25
            else:
                altered["metadata"]["total_size"] = 2**65
            path.write_bytes(canonical(altered))
            altered_id = self.register_current()
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.policy.prepare(altered_id)

    def test_reader_and_prepared_objects_cannot_be_supplied_as_public_dicts(self):
        prepared = self.policy.prepare(self.identifier)
        model = self.native_model(prepared)
        with self.assertRaises(ValueError):
            self.policy.bind(prepared, model)
        with self.assertRaises(ValueError):
            self.policy.bind({}, model, reader_ready=True)
        with self.assertRaises(ValueError):
            self.policy.catalog(
                active_binding={"static_view_ready": True}, reader_ready=True
            )
        self.registry.set_enabled(self.identifier, False)
        with self.assertRaises(ValueError):
            self.policy.bind(prepared, model, reader_ready=True)
        self.assertEqual(self.policy.catalog()["models"], [])

    def test_header_extent_and_packed_encoding_are_refused_after_owner_registration(
        self,
    ):
        for change in ("packed", "gap", "wrong_shape"):
            header = deepcopy(self.header)
            first = next(iter(header.values()))
            if change == "packed":
                first["dtype"] = "I8"
            elif change == "gap":
                first["data_offsets"][0] = 1
            else:
                first["shape"] = [1, math.prod(first["shape"])]
            raw = canonical(header)
            data = struct.pack("<Q", len(raw)) + raw + bytes(self.offset)
            (self.source / "model.safetensors").write_bytes(data)
            manifest = deepcopy(self.manifest)
            item = next(
                v for v in manifest["files"] if v["name"] == "model.safetensors"
            )
            item.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            identifier = self.registry.register(
                self.source, manifest, "Unsupported", max_bytes=65536
            )["model_id"]
            self.registry.set_enabled(identifier, True)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.policy.prepare(identifier)

    def test_native_identity_uses_literal_sorted_rust_schema_and_nanosecond_parts(self):
        shards = [
            {
                "name": "model.safetensors",
                "fingerprint": {
                    "size": 160,
                    "dev": 1,
                    "inode": 2,
                    "mtime": 3,
                    "mtime_ns": 4,
                    "ctime": 5,
                    "ctime_ns": 6,
                },
                "header_sha256": "d" * 64,
                "data_start": 128,
            }
        ]
        literal = (
            '{"index":null,"index_stat":null,"root":"/synthetic/source","schema":1,"shards":[{"data_start":128,'
            '"fingerprint":{"ctime":5,"ctime_ns":6,"dev":1,"inode":2,"mtime":3,"mtime_ns":4,"size":160},'
            '"header_sha256":"' + "d" * 64 + '","name":"model.safetensors"}]}'
        ).encode()
        self.assertEqual(
            native_source_identity("/synthetic/source", shards, None, None),
            hashlib.sha256(literal).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
