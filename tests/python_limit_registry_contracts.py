"""Installed-receipt limits and injected tiny acquisition streams; no network."""

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
import acquisition_contracts
import host_contracts
from atlas_host import acquisition, registry
from atlas_host.common import canonical

GIB = 1024**3


class RegistryLimits(unittest.TestCase):
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

    def manifest(self):
        return deepcopy(host_contracts.MANIFEST)

    def acquisition_fixture(self):
        case = acquisition_contracts.Contracts(methodName="runTest")
        case.setUp()
        self.addCleanup(case.doCleanups)
        case.registry.clock = lambda: 0.0
        return case

    def test_manifest_version_repository_revision_license_and_name_edges(self):
        self.assertEqual(
            (
                registry.MAX_REGISTRY_BYTES,
                registry.MAX_ENTRIES,
                registry.MAX_MANIFEST_BYTES,
            ),
            (524288, 128, 32768),
        )
        self.edges(
            lambda n: registry.validate_manifest({**self.manifest(), "version": n}),
            1,
            1,
        )
        for side in (0, 1):

            def validate(n):
                parts = ["x", "y"]
                parts[side] = "x" * n
                return registry.validate_manifest(
                    {**self.manifest(), "repository": "/".join(parts)}
                )

            self.edges(validate, 1, 96)
        self.edges(
            lambda n: registry.validate_manifest(
                {**self.manifest(), "revision": "a" * n}
            ),
            40,
            40,
        )

        def license_id(n):
            value = self.manifest()
            value["license"]["id"] = "x" * n
            return registry.validate_manifest(value)

        self.edges(license_id, 1, 128)

        def file_name(n):
            value = self.manifest()
            value["files"][0]["name"] = "x" * (n - 12) + ".safetensors"
            return registry.validate_manifest(value)

        # The suffix plus mandatory first character imposes the effective minimum.
        self.edges(file_name, 13, 180)

    def test_manifest_shard_and_auxiliary_sizes_and_digest_lengths(self):
        def shard_bytes(n):
            value = self.manifest()
            value["files"][0]["bytes"] = n
            return registry.validate_manifest(value)

        self.edges(shard_bytes, 1, 2**63 - 1)

        def auxiliary_bytes(n):
            value = self.manifest()
            value["files"].append(
                {"name": "config.json", "bytes": n, "sha256": "a" * 64}
            )
            return registry.validate_manifest(value)

        self.edges(auxiliary_bytes, 1, 8388608)

        def file_digest(n):
            value = self.manifest()
            value["files"][0]["sha256"] = "a" * n
            return registry.validate_manifest(value)

        self.edges(file_digest, 64, 64)

    def test_manifest_shard_count_and_reachable_total_files(self):
        def shards(n):
            value = self.manifest()
            value["files"] = [
                {"name": f"x{i}.safetensors", "bytes": 1, "sha256": "a" * 64}
                for i in range(n)
            ]
            return registry.validate_manifest(value)

        self.edges(shards, 1, 64)
        value = shards(64)
        value["files"] += [
            {"name": name, "bytes": 1, "sha256": "a" * 64}
            for name in sorted(registry.DATA_NAMES)
        ]
        self.assertEqual(len(registry.DATA_NAMES), 11)
        self.assertEqual(len(registry.validate_manifest(value)["files"]), 75)
        # The 80-file guard is masked by 64 shards plus eleven unique auxiliaries.
        # An expanded allowlist double isolates it; this is not production coverage
        # of an accepted 80-file manifest.
        for count, accepted in ((80, True), (81, False)):
            extra = {f"aux{i}.json" for i in range(count - 64)}
            isolated = shards(64)
            isolated["files"] += [
                {"name": name, "bytes": 1, "sha256": "a" * 64} for name in sorted(extra)
            ]
            with patch.object(registry, "DATA_NAMES", registry.DATA_NAMES | extra):
                if accepted:
                    registry.validate_manifest(isolated)
                else:
                    with self.assertRaisesRegex(ValueError, "file count"):
                        registry.validate_manifest(isolated)

    def test_manifest_serialized_guard_isolated_from_field_caps(self):
        # Current bounded fields cannot form a 32768-byte legal manifest.
        # Keep the redundant byte guard covered without claiming reachability.
        for size, accepted in ((32768, True), (32769, False)):
            with patch.object(registry, "canonical", return_value=b"x" * size):
                if accepted:
                    registry.validate_manifest(self.manifest())
                else:
                    with self.assertRaisesRegex(ValueError, "Manifest exceeds"):
                        registry.validate_manifest(self.manifest())

    def test_reservation_exact_free_and_each_declared_integer_bound(self):
        self.assertEqual(
            registry.reservation(25 * GIB + 100, metadata=100)["remaining_free_bytes"],
            25 * GIB,
        )
        with self.assertRaisesRegex(ValueError, "Insufficient disk"):
            registry.reservation(25 * GIB + 99, metadata=100)
        for field in ("remaining_download", "staging", "cache_growth", "metadata"):
            # The necessary reserve limits each field's reachable maximum.
            # Set other reservations to zero so they do not mask that boundary.
            self.edges(
                lambda n: registry.reservation(2**63 - 1, **{"metadata": 0, field: n}),
                0,
                2**63 - 1 - 25 * GIB,
            )
            with self.assertRaisesRegex(ValueError, "Insufficient disk"):
                registry.reservation(2**63 - 1, **{field: 2**63 - 1})
            with self.assertRaisesRegex(ValueError, "Integer outside"):
                registry.reservation(2**63 - 1, **{field: 2**63})
        self.edges(lambda n: registry.reservation(n, metadata=0), 25 * GIB, 2**63 - 1)
        self.edges(
            lambda n: registry.reservation(2**63 - 1, reserve=n, metadata=0),
            25 * GIB,
            2**63 - 1,
        )

    def registry_fixture(self):
        case = host_contracts.RegistryContracts(methodName="runTest")
        case.setUp()
        self.addCleanup(case.doCleanups)
        return case

    def test_registration_effective_bytes_timeout_name_and_clock_edge(self):
        case = self.registry_fixture()
        case.registry.clock = lambda: 0.0
        self.edges(lambda n: case.register(max_bytes=n), 244, 2**63 - 1)
        self.edges(lambda n: case.register(timeout_ms=n), 1, 5000)
        self.edges(
            lambda n: case.registry.register(case.source, self.manifest(), "x" * n),
            1,
            128,
        )
        for tick, accepted in ((0.999999, True), (1.0, False)):
            case = self.registry_fixture()
            calls = iter([0.0])
            case.registry.clock = lambda: next(calls, tick)
            if accepted:
                case.register(timeout_ms=1000)
            else:
                with self.assertRaisesRegex(ValueError, "time allowance"):
                    case.register(timeout_ms=1000)
                self.assertFalse(case.registry.path.exists())

    def receipt_data(self, count):
        models = []
        for number in range(count):
            manifest = self.manifest()
            manifest["revision"] = f"{number:040x}"
            identity = registry.content_digest(manifest)
            models.append(
                {
                    "model_id": "m_" + identity,
                    "content_digest": identity,
                    "name": "fixture",
                    "root": "/synthetic/fixture",
                    "manifest": manifest,
                    "fingerprints": {
                        item["name"]: {
                            "device": 0,
                            "inode": 0,
                            "bytes": item["bytes"],
                            "mtime_ns": 0,
                            "ctime_ns": 0,
                        }
                        for item in manifest["files"]
                    },
                    "verified_at": "fixture",
                    "enabled": False,
                }
            )
        return {"version": 1, "revision": 0, "models": models}

    def test_registry_load_real_bytes_entry_count_and_receipt_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            reader = registry.Registry(path, free_bytes=lambda _: 100 * GIB)
            for count, accepted in ((0, True), (128, True), (129, False)):
                path.write_bytes(canonical(self.receipt_data(count)))
                if accepted:
                    self.assertEqual(len(reader._load()["models"]), count)
                else:
                    with self.assertRaisesRegex(ValueError, "Too many models"):
                        reader._load()
            for size, accepted in ((524288, True), (524289, False)):
                raw = canonical(self.receipt_data(1))
                path.write_bytes(raw + b" " * (size - len(raw)))
                if accepted:
                    reader._load()
                else:
                    with self.assertRaisesRegex(ValueError, "byte limit"):
                        reader._load()
            for field, high in (("name", 128), ("root", 4096), ("verified_at", 64)):

                def load(n):
                    value = self.receipt_data(1)
                    value["models"][0][field] = (
                        "/" + "x" * (n - 1) if field == "root" else "x" * n
                    )
                    if field == "root" and n == 0:
                        value["models"][0][field] = ""
                    path.write_bytes(canonical(value))
                    return reader._load()

                self.edges(load, 1, high)
            for field in ("device", "inode", "bytes", "mtime_ns", "ctime_ns"):

                def load_fingerprint(n):
                    value = self.receipt_data(1)
                    next(iter(value["models"][0]["fingerprints"].values()))[field] = n
                    path.write_bytes(canonical(value))
                    return reader._load()

                self.edges(load_fingerprint, 0, 2**63 - 1)
            for field, low, high in (("version", 1, 1), ("revision", 0, 2**63 - 1)):

                def load_version(n):
                    value = self.receipt_data(0)
                    value[field] = n
                    path.write_bytes(canonical(value))
                    return reader._load()

                self.edges(load_version, low, high)

    def test_registry_save_byte_cap_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            writer = registry.Registry(path, free_bytes=lambda _: 100 * GIB)
            base = len(canonical({"padding": ""}))
            writer._save({"padding": "x" * (524288 - base)})
            before = path.read_bytes()
            self.assertEqual(len(before), 524288)
            with self.assertRaisesRegex(ValueError, "Registry exceeds"):
                writer._save({"padding": "x" * (524289 - base)})
            self.assertEqual(path.read_bytes(), before)
            # This tests the trusted serialization guard; loader schema is covered
            # separately with complete valid receipts above.

    def test_acquisition_plan_effective_payload_cache_and_name_bounds(self):
        case = self.acquisition_fixture()
        total = sum(len(value) for value in case.data.values())
        self.edges(
            lambda n: acquisition.plan(case.manifest, "fixture", max_bytes=n),
            total,
            64 * GIB,
        )
        self.edges(
            lambda n: acquisition.plan(
                case.manifest, "fixture", max_bytes=100, cache_growth=n
            ),
            0,
            8 * GIB,
        )
        self.edges(
            lambda n: acquisition.plan(case.manifest, "x" * n, max_bytes=100), 1, 128
        )

    def test_acquisition_timeout_edges_use_in_memory_streams_only(self):
        for timeout, accepted in (
            (99, False),
            (100, True),
            (600000, True),
            (600001, False),
        ):
            case = self.acquisition_fixture()
            if accepted:
                result = case.run_acquire(timeout_ms=timeout, clock=lambda: 0.0)
                self.assertTrue(result["registered"])
                self.assertFalse(result["enabled"])
            else:
                with self.assertRaises(ValueError):
                    case.run_acquire(timeout_ms=timeout, clock=lambda: 0.0)
                self.assertEqual(case.calls, [])
        for tick, accepted in ((math_next_below_one_tenth(), True), (0.1, False)):
            case = self.acquisition_fixture()
            times = iter([0.0])
            clock = lambda: next(times, tick)
            if accepted:
                case.run_acquire(timeout_ms=100, clock=clock)
            else:
                with self.assertRaises(acquisition.AcquisitionDeadline):
                    case.run_acquire(timeout_ms=100, clock=clock)
                self.assertEqual(case.calls, [])


def math_next_below_one_tenth():
    import math

    return math.nextafter(0.1, 0.0)


if __name__ == "__main__":
    unittest.main()
