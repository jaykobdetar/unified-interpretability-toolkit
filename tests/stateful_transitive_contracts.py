"""Tiny local state/receipt controls; no service, model or network work."""

import contextlib
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


class StatefulTransitiveTests(unittest.TestCase):
    def setUp(self):
        try:
            from atlas_host import progress, registry
        except Exception as exc:
            self.fail("local state helpers must initialize: " + repr(exc))
        self.progress = progress
        self.registry = registry
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.base = Path(directory.name).resolve()
        self.source = self.base / "source"
        self.source.mkdir()
        self.files = {
            "model.safetensors": b"tiny fixture",
            "config.json": b"{}",
            "LICENSE": b"tiny license",
        }
        for name, raw in self.files.items():
            (self.source / name).write_bytes(raw)
        self.manifest = {
            "version": 1,
            "repository": "fixtures/stateful",
            "revision": "a" * 40,
            "provenance": "owner_expected",
            "license": {"id": "Apache-2.0", "accepted": True, "file": "LICENSE"},
            "files": [
                {
                    "name": name,
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
                for name, raw in self.files.items()
            ],
        }
        content = {
            "version": 1,
            "repository": "fixtures/stateful",
            "revision": "a" * 40,
            "license": {"id": "Apache-2.0", "file": "LICENSE"},
            "files": sorted(
                deepcopy(self.manifest["files"]), key=lambda row: row["name"]
            ),
        }
        raw = json.dumps(
            ["weight-atlas-content-v1", content],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        self.content_digest = hashlib.sha256(raw).hexdigest()
        self.identifier = "m_" + self.content_digest

    def valid(self, function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as exc:
            self.fail("valid local state operation must complete: " + repr(exc))

    def error(self, function, message, *args, **kwargs):
        try:
            function(*args, **kwargs)
        except ValueError as exc:
            self.assertEqual(str(exc), message)
        except Exception as exc:
            self.fail("expected local ValueError: " + repr(exc))
        else:
            self.fail("expected local ValueError: " + message)

    def installed(self, **kwargs):
        return self.valid(
            self.registry.Registry,
            self.base / "host/registry.json",
            free_bytes=lambda _: 100 * 1024**3,
            clock=lambda: 0.0,
            **kwargs,
        )

    def entry(self):
        fingerprints = {}
        for name in self.files:
            info = (self.source / name).stat()
            fingerprints[name] = {
                "device": info.st_dev,
                "inode": info.st_ino,
                "bytes": info.st_size,
                "mtime_ns": info.st_mtime_ns,
                "ctime_ns": info.st_ctime_ns,
            }
        manifest = deepcopy(self.manifest)
        manifest["files"].sort(key=lambda row: row["name"])
        return {
            "model_id": self.identifier,
            "content_digest": self.content_digest,
            "name": "Tiny local fixture",
            "root": str(self.source),
            "manifest": manifest,
            "fingerprints": fingerprints,
            "verified_at": "2026-01-01T00:00:00+00:00",
            "enabled": False,
        }

    def record(self):
        return {
            "model_id": self.identifier,
            "kind": "calibration",
            "binding": json.loads(
                (ROOT / "tests/fixtures/host-source-binding-v2.json").read_text()
            ),
            "state": "running",
            "visited_values": 2,
            "total_values": 24,
            "elapsed_active_ms": 7,
            "remaining_authorized_work": {"values": 22, "wall_ms": 100, "cpu_ms": 90},
            "complete": False,
            "error": None,
        }

    def store(self):
        with patch.object(
            self.progress.secrets, "token_hex", side_effect=lambda n: "a" * (2 * n)
        ):
            return self.valid(self.progress.ProgressStore, self.identifier)

    def test_ProgressStore___init(self):
        store = self.store()
        self.assertEqual(store.model_id, self.identifier)
        self.assertEqual(store.epoch, "a" * 32)
        self.assertEqual(store.revision, 0)
        self.assertEqual(store.floor, 0)
        self.assertEqual(store._records, {})

    def test_ProgressStore_publish(self):
        store = self.store()
        key = "b" * 32
        value = self.record()
        before = deepcopy(value)
        self.assertEqual(self.valid(store.publish, key, value), 1)
        self.assertEqual(self.valid(store.publish, key, value), 1)
        self.assertEqual(store.revision, 1)
        later = deepcopy(value)
        later["visited_values"] = 3
        later["elapsed_active_ms"] = 8
        later["remaining_authorized_work"]["values"] = 21
        self.assertEqual(self.valid(store.publish, key, later), 2)
        self.assertEqual(store.revision, 2)
        self.assertEqual(
            store._records[key], {"record": later, "owner": None, "revision": 2}
        )
        self.assertIsNot(store._records[key]["record"], later)
        self.assertEqual(value, before)

    def test_ProgressStore_snapshot(self):
        store = self.store()
        key = "b" * 32
        value = self.record()
        self.valid(store.publish, key, value)
        actual = self.valid(store.snapshot)
        expected = {
            "version": 1,
            "model_id": self.identifier,
            "reset": True,
            "records": [{"id": key, "revision": 1, **value}],
            "has_more": False,
            "cursor": {"epoch": "a" * 32, "revision": 1, "scope": "public"},
        }
        self.assertEqual(actual, expected)
        self.assertIsNot(
            actual["records"][0]["binding"], store._records[key]["record"]["binding"]
        )
        self.assertEqual(
            self.valid(store.snapshot, actual["cursor"]),
            {**expected, "reset": False, "records": []},
        )

    def test_fingerprint(self):
        info = SimpleNamespace(
            st_mode=stat.S_IFREG | 0o600,
            st_dev=5,
            st_ino=7,
            st_size=11,
            st_mtime_ns=17,
            st_ctime_ns=19,
        )
        self.assertEqual(
            self.valid(self.registry.fingerprint, info),
            {"device": 5, "inode": 7, "bytes": 11, "mtime_ns": 17, "ctime_ns": 19},
        )

    def test_reservation(self):
        reserve = 25 * 1024**3
        self.assertEqual(
            self.valid(
                self.registry.reservation,
                reserve + 1000,
                remaining_download=300,
                staging=200,
                cache_growth=100,
                metadata=50,
            ),
            {
                "reserved_bytes": 650,
                "reserve_bytes": reserve,
                "remaining_free_bytes": reserve + 350,
            },
        )

    def test_Registry___init(self):
        path = self.base / "host/registry.json"
        clock = lambda: 7.0
        free = lambda parent: 100 * 1024**3
        target = self.valid(self.registry.Registry, path, free_bytes=free, clock=clock)
        self.assertEqual(target.path, path)
        self.assertIs(target.clock, clock)
        self.assertIs(target._free_bytes, free)
        self.assertEqual(target.clock(), 7.0)

    def test_Registry__disk_guard(self):
        target = self.installed()
        target.path.parent.mkdir()
        with patch.object(target, "_free_bytes", return_value=100 * 1024**3) as free:
            actual = self.valid(target._disk_guard, 29)
        free.assert_called_once_with(target.path.parent)
        self.assertEqual(
            actual,
            {
                "reserved_bytes": 29,
                "reserve_bytes": 25 * 1024**3,
                "remaining_free_bytes": 100 * 1024**3 - 29,
            },
        )

    def test_Registry__load(self):
        target = self.installed()
        self.assertEqual(
            self.valid(target._load), {"version": 1, "revision": 0, "models": []}
        )
        target.path.parent.mkdir()
        value = {"version": 1, "revision": 3, "models": []}
        raw = json.dumps(value).encode("utf-8")
        raw += b" " * (524288 - len(raw))
        target.path.write_bytes(raw)
        self.assertEqual(self.valid(target._load), value)
        self.assertEqual(target.path.read_bytes(), raw)

    def test_Registry__writer(self):
        target = self.installed()
        with patch.object(target, "_disk_guard") as disk:
            try:
                with target._writer() as yielded:
                    self.assertIsNone(yielded)
            except Exception as exc:
                self.fail("valid local writer must complete: " + repr(exc))
        disk.assert_called_once_with(524288 + 65536)
        lock = target.path.with_name(target.path.name + ".lock")
        self.assertTrue(lock.is_file())
        with patch.object(
            self.registry.fcntl,
            "flock",
            side_effect=BlockingIOError("tiny local overlap"),
        ):
            try:
                with target._writer():
                    self.fail("busy writer must refuse")
            except ValueError as exc:
                self.assertEqual(str(exc), "Registry writer busy")
            except Exception as exc:
                self.fail("expected local ValueError: " + repr(exc))

    def test_Registry__save(self):
        target = self.installed()
        target.path.parent.mkdir()
        value = {"version": 1, "revision": 2, "models": []}
        raw = b'{"models":[],"revision":2,"version":1}'
        with patch.object(target, "_disk_guard") as disk:
            self.assertIsNone(self.valid(target._save, value))
        disk.assert_called_once_with(len(raw) + 65536)
        self.assertTrue(target.path.exists())
        self.assertEqual(target.path.read_bytes(), raw)
        self.assertEqual(
            [p.name for p in target.path.parent.iterdir()], ["registry.json"]
        )

    def test_Registry_register(self):
        target = self.installed()
        before = deepcopy(self.manifest)
        self.assertEqual(
            self.valid(
                target.register, self.source, self.manifest, "Tiny local fixture"
            ),
            {
                "model_id": self.identifier,
                "content_digest": self.content_digest,
                "verified_bytes": 26,
                "verification": "matched_owner_expectations",
                "renderer_ready": False,
                "inference_ready": False,
                "enabled": False,
            },
        )
        self.assertEqual(self.valid(target._load)["revision"], 1)
        self.assertEqual(self.manifest, before)

    def test_Registry__unchanged(self):
        entry = self.entry()
        self.assertIs(self.valid(self.registry.Registry._unchanged, entry), True)
        with patch.object(
            self.registry,
            "fingerprint",
            side_effect=OSError("tiny local unreadable fixture"),
        ):
            self.assertIs(self.valid(self.registry.Registry._unchanged, entry), False)

    def test_Registry_set_enabled(self):
        target = self.installed()
        entry = self.entry()
        entry["enabled"] = True
        data = {"version": 1, "revision": 3, "models": [entry]}
        with (
            patch.object(target, "_writer", return_value=contextlib.nullcontext()),
            patch.object(target, "_load", return_value=data),
            patch.object(target, "_save") as save,
        ):
            actual = self.valid(target.set_enabled, self.identifier, False)
        self.assertEqual(
            actual,
            {"model_id": self.identifier, "enabled": False, "registry_revision": 4},
        )
        self.assertIs(entry["enabled"], False)
        self.assertEqual(data["revision"], 4)
        save.assert_called_once_with(data)

    def test_Registry_catalog(self):
        target = self.installed()
        entry = self.entry()
        entry["enabled"] = True
        data = {"version": 1, "revision": 3, "models": [entry]}
        before = deepcopy(data)
        with (
            patch.object(target, "_load", return_value=data),
            patch.object(target, "_unchanged", return_value=True),
        ):
            actual = self.valid(target.catalog)
        self.assertEqual(
            actual,
            {
                "version": 1,
                "registry_revision": 3,
                "models": [
                    {
                        "model_id": self.identifier,
                        "name": "Tiny local fixture",
                        "repository": "fixtures/stateful",
                        "revision": "a" * 40,
                        "license": "Apache-2.0",
                        "source_bytes": 26,
                        "state": "verified_pending_renderer",
                        "verification": "saved_hash_receipt_and_current_fingerprint",
                        "hash_provenance": "owner_expected",
                        "view_ready": False,
                        "inference_ready": False,
                    }
                ],
            },
        )
        self.assertEqual(data, before)

    def test_Registry_owner_receipt(self):
        target = self.installed()
        entry = self.entry()
        before = deepcopy(entry)
        with patch.object(
            target,
            "_load",
            return_value={"version": 1, "revision": 3, "models": [entry]},
        ):
            actual = self.valid(target.owner_receipt, self.identifier)
            self.assertEqual(actual, before)
            self.assertIsNot(actual, entry)
            self.assertIsNot(actual["fingerprints"], entry["fingerprints"])
            self.error(target.owner_receipt, "Unknown installed model", "m_" + "e" * 64)
        actual["fingerprints"]["config.json"]["bytes"] = 99
        self.assertEqual(entry, before)

    def test_Registry_register_checkpoint(self):
        target = self.installed()
        actual = self.valid(
            target.register,
            self.source,
            self.manifest,
            "Tiny local fixture",
            timeout_ms=100,
        )
        self.assertEqual(actual["model_id"], self.identifier)
        ticks = [0.0]

        def clock():
            return ticks.pop() if ticks else 0.1

        target.clock = clock
        self.error(
            target.register,
            "Verification time allowance exhausted",
            self.source,
            self.manifest,
            "Tiny local fixture",
            timeout_ms=100,
        )


if __name__ == "__main__":
    unittest.main()
