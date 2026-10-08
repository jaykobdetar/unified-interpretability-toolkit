"""Fixture leaf import and legacy callable addresses; no provider activation."""

import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
from atlas_host import fixture_source, runtime_adapter


class FixtureLeafContracts(unittest.TestCase):
    def test_existing_exports_keep_exact_objects(self):
        for name in ("FIXTURE_SHA", "FIXTURE_FILES", "fixture_entry", "check_fixture"):
            with self.subTest(name=name):
                self.assertIs(
                    getattr(runtime_adapter, name), getattr(fixture_source, name)
                )

    def test_callable_names_and_saved_python_references_keep_legacy_addresses(self):
        for name in ("fixture_entry", "check_fixture"):
            function = getattr(runtime_adapter, name)
            with self.subTest(name=name):
                self.assertEqual(function.__module__, "atlas_host.runtime_adapter")
                self.assertEqual(function.__name__, name)
                self.assertEqual(function.__qualname__, name)
                self.assertIs(pickle.loads(pickle.dumps(function)), function)

    def test_cold_leaf_import_does_not_load_provider_or_runtime_cycle(self):
        forbidden = [
            "atlas_host.runtime_adapter",
            "atlas_host.validation_policy",
            "atlas_host.profile_platform",
            "atlas_host.profile_service",
            "atlas_host.profile_worker",
            "atlas_host.hosted_runtime",
            "atlas_host.dense_static_admission",
            "atlas_host.lifetime_guard",
            "host_atlas",
            "torch",
        ]
        code = (
            "import json,sys; import atlas_host.fixture_source; print(json.dumps([name for name in "
            + repr(forbidden)
            + " if name in sys.modules]))"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=ROOT,
            env={
                **os.environ,
                "PYTHONPATH": str(ROOT / "tools"),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
