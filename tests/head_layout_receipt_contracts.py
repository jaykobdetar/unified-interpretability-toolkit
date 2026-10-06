"""Fresh pinned-file receipt and native renderer correspondence contracts."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import live_inference as live


class Receipts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.pins = {**live.MANIFEST, "files": {}}
        for name in live.MANIFEST["files"]:
            data = ("held local contract bytes: " + name).encode()
            (self.root / name).write_bytes(data)
            self.pins["files"][name] = hashlib.sha256(data).hexdigest()

    def test_required_and_optional_receipts_follow_full_fresh_hash_verification(self):
        with patch.object(live, "MANIFEST", self.pins):
            for required in (True, False):
                with self.subTest(required=required):
                    expected = live.layout_fingerprints(self.root)
                    receipt = live.verified_layout_receipt(self.root, required=required)
                    self.assertEqual(receipt, expected)
                    for name in self.pins["files"]:
                        with self.subTest(name=name):
                            path = self.root / name
                            before = path.read_bytes()
                            path.write_bytes(before + b" changed")
                            if required:
                                with self.assertRaisesRegex(
                                    ValueError, "Pinned file hash mismatch"
                                ):
                                    live.verified_layout_receipt(
                                        self.root, required=True
                                    )
                            else:
                                self.assertIsNone(
                                    live.verified_layout_receipt(
                                        self.root, required=False
                                    )
                                )
                            path.write_bytes(before)

    def test_optional_unknown_and_oversized_sources_do_not_trigger_full_verifier(self):
        with (
            patch.object(
                live, "verify_model", side_effect=AssertionError("must not hash")
            ),
            patch.object(live, "MANIFEST", self.pins),
        ):
            (self.root / "config.json").write_bytes(b"unknown configuration")
            self.assertIsNone(live.verified_layout_receipt(self.root, required=False))
            with (self.root / "config.json").open("wb") as stream:
                stream.truncate(65537)
            self.assertIsNone(live.verified_layout_receipt(self.root, required=False))
            (self.root / "config.json").write_bytes(
                b"held local contract bytes: config.json"
            )
            with (self.root / "model.safetensors").open("wb") as stream:
                stream.truncate(384 * 1024**2 + 1)
            self.assertIsNone(live.verified_layout_receipt(self.root, required=False))

    def test_missing_and_symlink_files_never_create_optional_receipt(self):
        path = self.root / "model.safetensors"
        path.unlink()
        with patch.object(live, "MANIFEST", self.pins):
            self.assertIsNone(live.verified_layout_receipt(self.root, required=False))
            with self.assertRaises(FileNotFoundError):
                live.verified_layout_receipt(self.root, required=True)
            target = self.root / "other"
            target.write_bytes(b"held local contract bytes: model.safetensors")
            path.symlink_to(target)
            self.assertIsNone(live.verified_layout_receipt(self.root, required=False))
            # Existing required inference verification accepts these exact bytes;
            # annotation is independently refused for a nonregular file receipt.
            self.assertIsNone(live.verified_layout_receipt(self.root, required=True))

    def test_optional_receipt_respects_memory_and_deadline_budgets(self):
        with patch.object(live, "MANIFEST", self.pins):
            with patch.object(live, "available", return_value=3 * live.GIB):
                self.assertIsNone(
                    live.verified_layout_receipt(self.root, required=False)
                )
            with patch.object(live.time, "monotonic", side_effect=[0, 6]):
                self.assertIsNone(
                    live.verified_layout_receipt(self.root, required=False)
                )

    def test_changed_fingerprint_during_verification_does_not_bind(self):
        verify = live.verify_model

        def replacement(directory, check=None):
            verify(directory, check=check)
            path = directory / "model.safetensors"
            new = directory / "replacement"
            new.write_bytes(path.read_bytes())
            new.replace(path)

        with (
            patch.object(live, "MANIFEST", self.pins),
            patch.object(live, "verify_model", side_effect=replacement),
        ):
            for required in (True, False):
                self.assertIsNone(
                    live.verified_layout_receipt(self.root, required=required)
                )

    def session_and_model(self):
        session = live.Session("unused", self.root)
        session.inference_enabled = False
        session.head_layout_receipt = live.layout_fingerprints(self.root)
        model = {
            "source_directory": str(self.root),
            "source_identity": "a" * 64,
            "revision": live.MANIFEST["revision"],
        }
        model["model_identity"] = hashlib.sha256(
            json.dumps(
                ["weight-atlas-model-v1", model["source_identity"], model["revision"]],
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        return session, model

    def test_current_native_correspondence_emits_only_configuration_evidence(self):
        session, model = self.session_and_model()
        with patch.object(
            live, "verify_model", side_effect=AssertionError("no request-time hashing")
        ):
            result = live.bind_inference_source(model, session)
        self.assertEqual(result["head_layout"], live.head_layout_descriptor())
        self.assertEqual(
            result["head_layout_binding"]["source_identity"], model["source_identity"]
        )
        self.assertEqual(
            result["head_layout_binding"]["model_identity"], model["model_identity"]
        )
        self.assertFalse(result["head_layout"]["runtime_verified"])
        self.assertNotIn("inference_source_model", result)
        self.assertIsNone(session.process)

    def test_stale_identity_root_revision_and_comparison_remove_upstream_claims(self):
        session, model = self.session_and_model()
        forged = {
            **model,
            "head_layout": {"runtime_verified": True},
            "head_layout_binding": {},
        }
        for key, value in [
            ("source_directory", str(self.root / "other")),
            ("revision", "other"),
            ("source_identity", "c" * 64),
            ("model_identity", "c" * 64),
            ("comparison_identity", "pair"),
            ("coordinate_space", "checkpoint-comparison-v1"),
        ]:
            with self.subTest(key=key):
                result = live.bind_inference_source({**forged, key: value}, session)
                self.assertNotIn("head_layout", result)
                self.assertNotIn("head_layout_binding", result)

    def test_each_changed_file_invalidates_receipt_without_manual_binding_fallback(
        self,
    ):
        for name in self.pins["files"]:
            with self.subTest(name=name):
                session, model = self.session_and_model()
                session.head_layout_binding = live.current_layout_binding(
                    model, session
                )
                path = self.root / name
                path.write_bytes(path.read_bytes() + b" changed")
                result = live.bind_inference_source(model, session)
                self.assertNotIn("head_layout", result)
                self.assertNotIn("head_layout_binding", result)


if __name__ == "__main__":
    unittest.main()
