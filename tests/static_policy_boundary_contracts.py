"""Static metadata and identity boundaries over existing tiny synthetic files."""

from collections.abc import Callable
from contextlib import ExitStack
from copy import deepcopy
import hashlib
from typing import TypeVar
import unittest
from unittest.mock import patch

from atlas_host import static_models as static
import static_model_contracts as static_fixture

T = TypeVar("T")


class StaticBoundaries(unittest.TestCase):
    def setUp(self) -> None:
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for target in (
            "subprocess.Popen",
            "socket.socketpair",
            "threading.Thread.start",
            "os.pidfd_open",
            "os.kill",
            "signal.pidfd_send_signal",
        ):
            self.stack.enter_context(
                patch(target, side_effect=AssertionError("Live work forbidden"))
            )

    def success(self, operation: Callable[[], T]) -> T:
        try:
            return operation()
        except BaseException as error:
            self.fail(f"Unexpected {type(error).__name__}: {error}")

    def refuses(self, message: str, operation: Callable[[], object]) -> None:
        caught = None
        try:
            operation()
        except BaseException as error:
            caught = error
        self.assertIs(type(caught), ValueError)
        self.assertEqual(str(caught), message)

    def fixture(self) -> static_fixture.StaticContracts:
        case = static_fixture.StaticContracts(methodName="runTest")
        self.addCleanup(case.doCleanups)
        self.success(case.setUp)
        self.assertIs(case.policy.registry, case.registry)
        self.assertLessEqual(sum(row["bytes"] for row in case.manifest["files"]), 65536)
        return case

    def test_json_keeps_values_and_nonfinite_refusal_text(self) -> None:
        self.assertEqual(
            self.success(lambda: static._json(b'{"number":7,"label":"fixture"}')),
            {"number": 7, "label": "fixture"},
        )
        self.refuses("Nonfinite static metadata", lambda: static._json(b"NaN"))

    def test_inventory_current_receipt_and_catalog_envelope(self) -> None:
        case = self.fixture()
        entry = self.success(lambda: case.registry.owner_receipt(case.identifier))
        self.assertTrue(self.success(lambda: static._inventory(entry)))
        self.success(lambda: static._current(case.registry, entry))
        disabled = deepcopy(entry)
        disabled["enabled"] = False
        self.refuses(
            "Static owner receipt/source changed",
            lambda: static._current(case.registry, disabled),
        )
        snapshot = self.success(case.registry._load)
        catalog = self.success(case.policy.catalog)
        self.assertEqual(catalog["version"], 1)
        self.assertEqual(catalog["registry_revision"], snapshot["revision"])
        self.assertEqual(len(catalog["models"]), 1)
        row = catalog["models"][0]
        self.assertEqual(row["model_id"], case.identifier)
        self.assertEqual(
            row["source_bytes"], sum(item["bytes"] for item in case.manifest["files"])
        )
        self.assertTrue(row["static_view_candidate"])
        self.assertFalse(row["static_view_ready"])
        self.assertFalse(row["inference_ready"])

    def test_metadata_read_charges_header_prefix_and_returns_exact_bytes(self) -> None:
        case = self.fixture()
        entry = self.success(lambda: case.registry.owner_receipt(case.identifier))
        budget = [0]
        config = self.success(lambda: static._read(entry, "config.json", 65536, budget))
        self.assertEqual(config, (case.source / "config.json").read_bytes())
        self.assertEqual(budget, [len(config)])
        header = self.success(
            lambda: static._read(
                entry, "model.safetensors", 2 * 1024**2, budget, header=True
            )
        )
        self.assertEqual(header, case.raw)
        self.assertEqual(budget, [len(config) + 8 + len(case.raw)])

    def test_native_fingerprint_and_ordered_identity_bytes(self) -> None:
        saved = {
            "bytes": 160,
            "device": 11,
            "inode": 22,
            "mtime_ns": 3000000004,
            "ctime_ns": 5000000006,
        }
        self.assertEqual(
            self.success(lambda: static._native_fingerprint(saved)),
            {
                "size": 160,
                "dev": 11,
                "inode": 22,
                "mtime": 3,
                "mtime_ns": 4,
                "ctime": 5,
                "ctime_ns": 6,
            },
        )
        shards = [{"name": "a", "data_start": 8}, {"name": "b", "data_start": 16}]
        literal = (
            b'{"index":null,"index_stat":null,"root":"/synthetic/source","schema":1,'
            b'"shards":[{"data_start":8,"name":"a"},{"data_start":16,"name":"b"}]}'
        )
        self.assertEqual(
            self.success(
                lambda: static.native_source_identity(
                    "/synthetic/source", shards, None, None
                )
            ),
            hashlib.sha256(literal).hexdigest(),
        )

    def test_prepared_copy_domains_delegate_and_binding_identity(self) -> None:
        case = self.fixture()
        entry = self.success(lambda: case.registry.owner_receipt(case.identifier))
        original_name = entry["name"]
        descriptor, tensors = {"probe": 7}, {"probe": {"id": 0}}
        source = "c" * 64
        prepared = self.success(
            lambda: static.PreparedStatic(
                case.registry, entry, descriptor, tensors, source, 128
            )
        )
        expected_model = hashlib.sha256(
            ('["weight-atlas-model-v1","' + "c" * 64 + '","' + "a" * 40 + '"]').encode()
        ).hexdigest()
        expected_descriptor = hashlib.sha256(
            b'["weight-atlas-static-descriptor-v1",{"probe":7}]'
        ).hexdigest()
        self.assertEqual(prepared.model_identity, expected_model)
        self.assertEqual(prepared.descriptor_digest, expected_descriptor)
        entry["name"], descriptor["probe"], tensors["probe"]["id"] = "changed", 99, 88
        self.assertEqual(prepared.entry["name"], original_name)
        self.assertEqual(prepared.descriptor, {"probe": 7})
        self.assertEqual(prepared.tensors, {"probe": {"id": 0}})
        with patch.object(static, "_current") as current:
            self.success(prepared.check)
            current.assert_called_once_with(case.registry, prepared.entry)
        binding = self.success(lambda: static.BoundStatic(prepared))
        self.assertIs(binding.prepared, prepared)
        self.assertEqual(
            self.success(binding.public_binding),
            {
                "schema": "weight-atlas-static-binding-v1",
                "model_id": entry["model_id"],
                "content_digest": entry["content_digest"],
                "descriptor_digest": expected_descriptor,
                "source_identity": source,
                "model_identity": expected_model,
                "evidence": "saved_owner_hash_receipt_current_fingerprints_config_headers_and_bound_catalog",
                "fresh_payload_hashes_recomputed": False,
                "inference_ready": False,
                "fit_verified": False,
            },
        )

    def test_preparation_native_correspondence_and_public_projection(self) -> None:
        case = self.fixture()
        prepared = self.success(lambda: case.policy.prepare(case.identifier))
        self.assertEqual(prepared.header_bytes, 8 + len(case.raw))
        self.assertEqual(list(sorted(prepared.tensors)), list(sorted(case.header)))
        for index, name in enumerate(sorted(case.header)):
            self.assertEqual(prepared.tensors[name]["id"], index)
            self.assertEqual(
                prepared.tensors[name]["byte_offset"],
                8 + len(case.raw) + case.header[name]["data_offsets"][0],
            )
        model = self.success(lambda: case.native_model(prepared))
        binding = self.success(
            lambda: case.policy.bind(prepared, model, reader_ready=True)
        )
        self.assertIs(binding.prepared, prepared)
        projected = self.success(
            lambda: case.policy.project_model(binding, model, reader_ready=True)
        )
        self.assertEqual(projected["name"], "Registered Qwen")
        self.assertTrue(projected["static_view_ready"])
        self.assertEqual(
            projected["coverage"]["calibration_error"], "calibration_failed"
        )
        self.assertFalse(projected["inference_ready"])


if __name__ == "__main__":
    unittest.main()
