"""Shared pair/sweep package identities and preserved default helper lookup."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from atlas_host import inference_pair_contract as packaged_pair
from atlas_host import inference_sweep_contract as packaged_sweep
import inference_pair_contract as legacy_pair
import inference_sweep_contract as legacy_sweep
import inference_prompt_pair as pair
import inference_sweep as sweep


class PackageContracts(unittest.TestCase):
    def test_modules_and_binding_classes_share_identity(self) -> None:
        self.assertIs(legacy_pair, packaged_pair)
        self.assertIs(legacy_sweep, packaged_sweep)
        self.assertIs(legacy_pair.PairBindings, packaged_pair.PairBindings)
        self.assertIs(legacy_sweep.SweepBindings, packaged_sweep.SweepBindings)
        self.assertIs(type(pair._bindings()), packaged_pair.PairBindings)
        self.assertIs(type(sweep._bindings()), packaged_sweep.SweepBindings)

    def test_pair_operation_and_late_callback_use_shared_state(self) -> None:
        held = "0" * 64
        prompts, ids = ["A", "B"], [[1, 2, 3], [1, 4]]
        with patch.object(legacy_pair, "digest_for", return_value=held) as digest:
            self.assertIs(pair.digest_for(prompts, ids), held)
        digest.assert_called_once()
        self.assertEqual(digest.call_args.args, (prompts, ids))
        binding = digest.call_args.kwargs["bindings"]
        self.assertIs(binding.source_model, pair.SOURCE_MODEL)
        with patch.object(pair, "digest_for", return_value=held) as callback:
            self.assertIs(binding.digest_for(prompts, ids), held)
        callback.assert_called_once_with(prompts, ids)

    def test_sweep_schema_and_late_target_keep_original_references(self) -> None:
        held = {"held": "schema"}
        with patch.object(legacy_sweep, "schema", return_value=held) as schema:
            self.assertIs(sweep.schema(), held)
        schema.assert_called_once()
        binding = schema.call_args.kwargs["bindings"]
        self.assertIs(binding.source_model, sweep.SOURCE_MODEL)
        self.assertIs(binding.shapes, sweep.SHAPES)
        target, result = object(), {"held": "target"}
        with patch.object(sweep, "_target", return_value=result) as callback:
            self.assertIs(binding.target(target), result)
        callback.assert_called_once_with(target)


if __name__ == "__main__":
    unittest.main()
