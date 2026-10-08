"""Shared architecture defaults, lazy helper state and relocated repository root."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from atlas_host import inference_architecture as packaged
import inference_architecture as legacy
import inference_edits as edits
import inference_observations as observations
import inference_prompt_pair as pair
import inference_sweep as sweep
import inference_worker as worker
import live_inference as live


class PackageContracts(unittest.TestCase):
    def test_module_root_globals_and_imported_defaults_share_identity(self) -> None:
        self.assertIs(legacy, packaged)
        self.assertEqual(packaged.ROOT, Path(__file__).resolve().parents[1])
        for name in ("ARCH", "CONFIG", "MANIFEST", "CAPTURE_SITES"):
            self.assertIs(getattr(legacy, name), getattr(packaged, name))
        for caller in (edits, observations, pair, sweep, worker, live):
            self.assertIs(caller.architecture, packaged.architecture)
        for caller in (edits, observations, pair, sweep):
            self.assertIs(caller.ARCH, packaged.ARCH)
        self.assertIs(worker.verify_attention_layout, packaged.verify_attention_layout)
        self.assertIs(live.head_layout_descriptor, packaged.head_layout_descriptor)
        self.assertIs(live.bind_viewer_head_layout, packaged.bind_viewer_head_layout)

    def test_late_descriptor_and_canonical_state(self) -> None:
        self.assertIs(legacy.geometry, packaged.geometry)
        value, result = object(), {"held": "layout"}
        with (
            patch.object(legacy, "architecture", return_value=value) as current,
            patch.object(
                legacy.geometry, "head_layout_descriptor", return_value=result
            ) as descriptor,
        ):
            self.assertIs(packaged.head_layout_descriptor(), result)
        current.assert_called_once_with()
        descriptor.assert_called_once_with(value)
        self.assertIs(descriptor.call_args.args[0], value)


if __name__ == "__main__":
    unittest.main()
