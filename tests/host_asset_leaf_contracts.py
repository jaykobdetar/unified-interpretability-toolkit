"""Asset catalogue aliases and cold imports preserve the Python host boundary."""

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import host_atlas
from atlas_host import host_assets


class HostAssetLeafContracts(unittest.TestCase):
    def test_existing_exports_keep_exact_objects(self):
        self.assertIs(host_atlas.ASSETS, host_assets.ASSETS)
        self.assertIs(host_atlas.BUNDLE, host_assets.BUNDLE)

    def test_cold_leaf_import_does_not_load_host_or_runtime(self):
        forbidden = [
            "host_atlas",
            "atlas_host.runtime_adapter",
            "atlas_host.hosted_runtime",
            "atlas_host.dense_static_admission",
            "http.server",
            "torch",
        ]
        root = Path(__file__).resolve().parents[1]
        code = (
            "import json,sys; import atlas_host.host_assets; print(json.dumps([name for name in "
            + repr(forbidden)
            + " if name in sys.modules]))"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=root,
            env={
                **os.environ,
                "PYTHONPATH": str(root / "tools"),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [])


if __name__ == "__main__":
    unittest.main()
