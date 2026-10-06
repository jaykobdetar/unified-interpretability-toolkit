"""Canonical pair/sweep modules share legacy state and late callbacks."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from atlas_host import inference_prompt_pair as pair
from atlas_host import inference_sweep as sweep
import inference_prompt_pair as legacy_pair
import inference_sweep as legacy_sweep
import inference_worker as worker
import live_inference as live


class PackageContracts(unittest.TestCase):
    def test_modules_bindings_and_consumers_share_identity(self) -> None:
        self.assertIs(legacy_pair, pair)
        self.assertIs(legacy_sweep, sweep)
        self.assertIs(legacy_pair.PairBindings, pair.PairBindings)
        self.assertIs(legacy_sweep.SweepBindings, sweep.SweepBindings)
        self.assertIs(legacy_sweep.RestorationFailure, sweep.RestorationFailure)
        self.assertIs(live.prompt_pair, pair)
        self.assertIs(live.sweep, sweep)
        for caller in (worker, live):
            graph = caller._contracts()
            self.assertIs(type(graph.pairs), pair.PairBindings)
            self.assertIs(type(graph.sweeps), sweep.SweepBindings)
            self.assertIs(graph.pairs.architecture, graph.architecture)
            self.assertIs(graph.sweeps.architecture, graph.architecture)
            self.assertIs(graph.pairs.source_model, pair.SOURCE_MODEL)
            self.assertIs(graph.sweeps.shapes, sweep.SHAPES)

    def test_current_default_callbacks_keep_late_lookup(self) -> None:
        binding = pair._bindings()
        prompts, ids, result = ["A", "B"], [[1], [2]], "held"
        with patch.object(legacy_pair, "digest_for", return_value=result) as digest:
            self.assertIs(binding.digest_for(prompts, ids), result)
        digest.assert_called_once_with(prompts, ids)
        binding = sweep._bindings()
        plan, result = {"held": 1}, ["held-id"]
        with patch.object(legacy_sweep, "record_ids", return_value=result) as records:
            self.assertIs(binding.record_ids(plan), result)
        records.assert_called_once_with(plan)
        self.assertIs(records.call_args.args[0], plan)


if __name__ == "__main__":
    unittest.main()
