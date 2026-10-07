"""Coordinator module identity, repository paths and late defaults without services."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import live_inference as legacy
from atlas_host import live_inference as canonical


class PackageCoordinatorTests(unittest.TestCase):
    def test_module_and_public_objects_are_shared(self):
        self.assertIs(legacy, canonical)
        for name in (
            "BackendError",
            "OperationDeadline",
            "DeadlineReader",
            "DeadlineSocket",
            "Session",
            "Handler",
            "SweepAdmissionBudget",
            "verify_model",
            "proxy_request",
            "signal_and_reap",
            "available",
            "MANIFEST",
            "GIB",
            "Snapshot",
            "LayoutReceipt",
            "_contracts",
            "main",
        ):
            self.assertIs(getattr(legacy, name), getattr(canonical, name))

    def test_repository_paths_and_manifest_are_held(self):
        self.assertEqual(canonical.ROOT, ROOT)
        self.assertEqual(legacy.ROOT, ROOT)
        self.assertEqual(
            canonical.MANIFEST,
            json.loads((ROOT / "docs/models/smollm2-135m.json").read_text()),
        )
        self.assertEqual(Path(canonical.__file__).parent, ROOT / "tools/atlas_host")

    def test_legacy_patches_reach_canonical_globals(self):
        width = canonical.WIDTH
        with patch.object(legacy, "WIDTH", width - 1):
            self.assertEqual(canonical._contracts().architecture.width, width - 1)
        self.assertEqual(canonical._contracts().architecture.width, width)
        with patch.object(legacy, "available", return_value=8 * legacy.GIB):
            session = canonical.Session("unused-python", Path("/unused"))
        self.assertEqual(session.minimum_available, 8 * legacy.GIB)


if __name__ == "__main__":
    unittest.main()
