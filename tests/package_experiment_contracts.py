"""Shared experiment registry, tokenizer protocols and worker imports."""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

from atlas_host import inference_experiments as packaged_experiments
from atlas_host import inference_generation as packaged_generation
import inference_experiments as legacy_experiments
import inference_generation as legacy_generation
import inference_worker as worker
import live_inference as live
import experiment_execution_contracts as held


class PackageContracts(unittest.TestCase):
    def test_registry_alias_preserves_objects_and_late_selection(self) -> None:
        self.assertIs(legacy_experiments, packaged_experiments)
        self.assertIs(legacy_experiments.REGISTRY, packaged_experiments.REGISTRY)
        self.assertIs(legacy_experiments.Request, packaged_experiments.Request)
        self.assertIs(legacy_experiments.Record, packaged_experiments.Record)
        request, context, selected = {"held": "request"}, object(), Mock()
        with patch.object(
            legacy_experiments, "select", return_value=selected
        ) as select:
            packaged_experiments.execute(request, context)
        select.assert_called_once_with(request)
        self.assertIs(select.call_args.args[0], request)
        selected.execute.assert_called_once_with(context, request)
        self.assertIs(selected.execute.call_args.args[0], context)
        self.assertIs(selected.execute.call_args.args[1], request)

    def test_entrypoints_retain_imported_types_and_functions(self) -> None:
        self.assertIs(worker.Context, packaged_experiments.Context)
        self.assertIs(worker.execute, packaged_experiments.execute)
        self.assertIs(worker.GenerationBindings, packaged_generation.GenerationBindings)
        self.assertIs(worker.run_generation, packaged_generation.run)
        self.assertIs(live.Kind, packaged_experiments.Kind)
        self.assertIs(live.REGISTRY, packaged_experiments.REGISTRY)
        self.assertIs(live.for_coordinator, packaged_experiments.for_coordinator)

    def test_generation_alias_and_all_six_protocol_interfaces(self) -> None:
        self.assertIs(legacy_generation, packaged_generation)
        self.assertIs(legacy_generation.Tokenizer, packaged_generation.Tokenizer)
        self.assertIs(legacy_generation.Observation, packaged_generation.Observation)
        checks = held.ProtocolContracts()
        for name in unittest.defaultTestLoader.getTestCaseNames(held.ProtocolContracts):
            getattr(checks, name)()
        self.assertEqual(
            len(unittest.defaultTestLoader.getTestCaseNames(held.ProtocolContracts)), 6
        )


if __name__ == "__main__":
    unittest.main()
