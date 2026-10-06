"""Shared ordinary module identity and preserved launcher entrypoint exports."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import launch
import viewer_resources
from atlas_host import launch as packaged_launch
from atlas_host import viewer_resources as packaged_resources


class PackageLaunchResourceTests(unittest.TestCase):
    def test_shared_resource_state_and_operations(self):
        self.assertIs(viewer_resources, packaged_resources)
        self.assertIs(launch.encode_resources, viewer_resources.encode)
        self.assertIs(viewer_resources.DEFAULTS, packaged_resources.DEFAULTS)
        with patch.dict(viewer_resources.DEFAULTS, {"cpu_count": 4}):
            self.assertEqual(packaged_resources.validate({})["cpu_count"], 4)
        self.assertEqual(viewer_resources.DEFAULTS["cpu_count"], 1)

    def test_launcher_alias_exports_and_repository_paths(self):
        self.assertIs(launch, packaged_launch)
        self.assertIs(launch.main, packaged_launch.main)
        self.assertIs(launch.unique_fields, packaged_launch.unique_fields)
        self.assertEqual(launch.ROOT, ROOT)
        self.assertEqual(launch.BINARY, ROOT / "target/release/weight-atlas-rust")
        self.assertEqual(
            set(launch.__all__),
            {
                "argparse",
                "json",
                "os",
                "Path",
                "shutil",
                "subprocess",
                "sys",
                "encode_resources",
                "ROOT",
                "BINARY",
                "FIELDS",
                "unique_fields",
                "main",
            },
        )


if __name__ == "__main__":
    unittest.main()
