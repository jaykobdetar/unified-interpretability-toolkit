"""Tiny ordinary acquisition lifecycle using safe in-memory sources; no network."""

import contextlib
import hashlib
import io
import os
import stat
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from atlas_host.acquisition import acquire, plan, _allowed_url
from atlas_host.registry import Registry
import atlas_host.acquisition as acquisition


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.registry = Registry(
            self.root / "registry.json", free_bytes=lambda _: 100 * 1024**3
        )
        self.data = {
            "model.safetensors": b"synthetic never executed",
            "LICENSE": b"synthetic license",
        }
        self.manifest = {
            "version": 1,
            "repository": "fixtures/tiny",
            "revision": "a" * 40,
            "provenance": "owner_expected",
            "license": {"id": "Apache-2.0", "accepted": True, "file": "LICENSE"},
            "files": [
                {"name": k, "bytes": len(v), "sha256": hashlib.sha256(v).hexdigest()}
                for k, v in self.data.items()
            ],
        }
        self.reviewed = plan(self.manifest, "Tiny", max_bytes=100)
        self.calls = []

    def fetch(self, url, timeout):
        self.calls.append((url, timeout))
        return io.BytesIO(self.data[url.rsplit("/", 1)[1]])

    def run_acquire(self, **kw):
        options = {
            "max_bytes": 100,
            "plan_digest": self.reviewed["plan_digest"],
            "accept_license": True,
            "fetch": self.fetch,
            "free_bytes": lambda _: 100 * 1024**3,
        }
        options.update(kw)
        return acquire(
            self.registry, self.root / "installed", self.manifest, "Tiny", **options
        )

    def test_reviewed_exact_files_are_registered_disabled_and_can_be_explicitly_enabled(
        self,
    ):
        result = self.run_acquire()
        self.assertTrue(result["registered"])
        self.assertFalse(result["enabled"])
        self.assertFalse(result["inference_ready"])
        self.assertEqual(self.registry.catalog()["models"], [])
        receipt = self.registry.owner_receipt(result["model_id"])
        self.assertFalse(receipt["enabled"])
        for name, data in self.data.items():
            self.assertEqual((self.root / "installed" / name).read_bytes(), data)
        self.assertTrue(
            all(
                "/resolve/" + ("a" * 40) + "/" in url and 0 < timeout <= 10
                for url, timeout in self.calls
            )
        )
        self.registry.set_enabled(result["model_id"], True)
        self.assertEqual(len(self.registry.catalog()["models"]), 1)

    def test_no_connection_before_acceptance_plan_and_disk_admission(self):
        for kw in (
            {"accept_license": False},
            {"plan_digest": "b" * 64},
            {"max_bytes": 1},
            {"free_bytes": lambda _: 25 * 1024**3},
        ):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self.run_acquire(**kw)
        self.assertEqual(self.calls, [])
        self.assertFalse((self.root / "installed").exists())

    def test_ordinary_incomplete_transfer_cleans_only_its_stage(self):
        with self.assertRaises(ValueError):
            self.run_acquire(fetch=lambda *_: io.BytesIO(b""))
        self.assertFalse((self.root / "installed").exists())
        self.assertEqual(list(self.root.glob(".atlas-acquire-*")), [])
        self.assertFalse(self.registry.path.exists())

    def test_pending_registration_is_honest_and_never_selectable(self):
        with patch.object(
            self.registry,
            "register",
            side_effect=ValueError("bounded verification timed out"),
        ):
            result = self.run_acquire()
        self.assertTrue(result["installed"])
        self.assertFalse(result["registered"])
        self.assertFalse(result["inference_ready"])
        self.assertEqual(self.registry.catalog()["models"], [])
        self.assertTrue((self.root / "installed/model.safetensors").is_file())

    def test_closed_redirect_hosts_and_https(self):
        _allowed_url(
            "https://huggingface.co/fixtures/tiny/resolve/" + ("a" * 40) + "/LICENSE"
        )
        _allowed_url("https://us.aws.cdn.hf.co/public-data")
        for url in (
            "http://huggingface.co/data",
            "https://localhost/data",
            "https://huggingface.co.example/data",
            "https://user:password@huggingface.co/data",
        ):
            with self.assertRaises(ValueError):
                _allowed_url(url)

    def test_reacquisition_publishes_actual_disabled_state_under_writer_lock(self):
        first = self.run_acquire()
        self.registry.set_enabled(first["model_id"], True)
        second = acquire(
            self.registry,
            self.root / "second",
            self.manifest,
            "Tiny",
            max_bytes=100,
            plan_digest=self.reviewed["plan_digest"],
            accept_license=True,
            fetch=self.fetch,
            free_bytes=lambda _: 100 * 1024**3,
        )
        receipt = self.registry.owner_receipt(second["model_id"])
        self.assertFalse(second["enabled"])
        self.assertFalse(receipt["enabled"])
        self.assertEqual(receipt["root"], str(self.root / "second"))
        self.assertEqual(self.registry.catalog()["models"], [])
        self.assertTrue((self.root / "installed/model.safetensors").is_file())
        # The independent standalone owner operation still preserves enabling.
        self.registry.set_enabled(second["model_id"], True)
        result = self.registry.register(
            self.root / "second", self.manifest, "Tiny", max_bytes=100
        )
        self.assertTrue(result["enabled"])

    def test_deadline_after_install_returns_retained_truthful_outcome(self):
        now = [0.0]
        actual = acquisition._install_new_directory

        def install(source, destination):
            actual(source, destination)
            now[0] = 0.2

        with patch.object(acquisition, "_install_new_directory", side_effect=install):
            result = self.run_acquire(timeout_ms=100, clock=lambda: now[0])
        self.assertTrue(result["installed"])
        self.assertFalse(result["registered"])
        self.assertEqual(result["reason"], "deadline_after_install_data_retained")
        self.assertEqual(
            result["installation_durability"], "parent_directory_fsync_confirmed"
        )
        self.assertEqual(result["registry_publication"], "not_attempted")
        self.assertTrue((self.root / "installed/model.safetensors").is_file())
        self.assertFalse(self.registry.path.exists())

    def test_parent_fsync_error_retains_installation_and_reports_uncertain_durability(
        self,
    ):
        actual = os.fsync

        def fsync(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError("synthetic parent directory sync uncertainty")
            actual(fd)

        with patch.object(acquisition.os, "fsync", side_effect=fsync):
            result = self.run_acquire()
        self.assertTrue(result["installed"])
        self.assertFalse(result["registered"])
        self.assertEqual(result["installation_durability"], "unconfirmed")
        self.assertEqual(result["registry_publication"], "not_attempted")
        self.assertTrue((self.root / "installed/model.safetensors").is_file())

    def test_registry_io_uncertainty_does_not_claim_publication_absent(self):
        with patch.object(
            self.registry,
            "register",
            side_effect=OSError("synthetic uncertain receipt publication"),
        ):
            result = self.run_acquire()
        self.assertTrue(result["installed"])
        self.assertIsNone(result["registered"])
        self.assertIsNone(result["enabled"])
        self.assertEqual(result["registry_publication"], "unconfirmed")

    def test_io_error_after_receipt_replace_is_reported_unconfirmed_not_absent(self):
        actual = os.fsync
        directory_syncs = [0]

        def fsync(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                directory_syncs[0] += 1
                if directory_syncs[0] == 2:
                    raise OSError(
                        "synthetic sync uncertainty after registry replacement"
                    )
            actual(fd)

        with patch.object(acquisition.os, "fsync", side_effect=fsync):
            result = self.run_acquire()
        self.assertTrue(self.registry.path.is_file())
        entry = self.registry._load()["models"][0]
        self.assertEqual(entry["root"], str(self.root / "installed"))
        self.assertFalse(entry["enabled"])
        self.assertTrue(result["installed"])
        self.assertIsNone(result["registered"])
        self.assertEqual(result["registry_publication"], "unconfirmed")

    def test_cooperative_budget_rejects_data_after_a_slow_buffered_read(self):
        now = [0.0]

        class Stream(io.BytesIO):
            def read(self, count):
                now[0] += 20
                return super().read(count)

        with self.assertRaises(acquisition.AcquisitionDeadline):
            self.run_acquire(
                timeout_ms=100,
                clock=lambda: now[0],
                fetch=lambda url, timeout: Stream(self.data[url.rsplit("/", 1)[1]]),
            )
        self.assertFalse((self.root / "installed").exists())
        self.assertEqual(list(self.root.glob(".atlas-acquire-*")), [])
        self.assertIn(
            "external hard supervision", self.reviewed["deadline_enforcement"]
        )


if __name__ == "__main__":
    unittest.main()
