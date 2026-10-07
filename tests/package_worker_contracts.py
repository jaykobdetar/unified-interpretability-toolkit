"""Worker compatibility import, held defaults and canonical late state."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import inference_worker as legacy
from atlas_host import inference_worker as canonical


class PackageWorkerTests(unittest.TestCase):
    def test_shared_module_callbacks_and_held_defaults(self):
        self.assertIs(legacy, canonical)
        for name in (
            "emit",
            "load_engine",
            "_contracts",
            "generate",
            "paired_step",
            "compare",
            "configure_worker_limits",
            "main",
            "ARCH",
            "CAPTURE_SITES",
            "LoaderRuntime",
            "GenerationBindings",
            "ComparisonBindings",
            "execute",
            "MAX_PROMPT",
            "MAX_NEW",
        ):
            self.assertIs(getattr(legacy, name), getattr(canonical, name))
        self.assertIs(canonical.generate.__defaults__[0], canonical.emit)
        self.assertIs(canonical.compare.__defaults__[0], canonical.emit)

    def test_legacy_global_changes_are_seen_by_canonical_operations(self):
        args = tuple(object() for _ in range(6))
        with (
            patch.object(legacy, "MAX_PROMPT", 127),
            patch.object(canonical, "run_generation", return_value=None) as run,
        ):
            canonical.generate(*args)
            run.assert_called_once()
            self.assertEqual(run.call_args.args[7].max_prompt, 127)
        self.assertEqual(canonical.MAX_PROMPT, 128)


if __name__ == "__main__":
    unittest.main()
