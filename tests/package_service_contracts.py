"""Service module/class identity and held per-caller bindings."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from atlas_host import inference_services as packaged
import inference_services as legacy
import inference_worker as worker
import live_inference as live


class PackageContracts(unittest.TestCase):
    def test_alias_and_caller_classes_share_identity(self) -> None:
        self.assertIs(legacy, packaged)
        self.assertIs(legacy.InferenceContracts, packaged.InferenceContracts)
        self.assertIs(worker.InferenceContracts, packaged.InferenceContracts)
        self.assertIs(live.InferenceContracts, packaged.InferenceContracts)
        for caller in (worker, live):
            contracts = caller._contracts()
            self.assertIs(type(contracts), packaged.InferenceContracts)
            for binding in (
                contracts.edits,
                contracts.observations,
                contracts.pairs,
                contracts.sweeps,
            ):
                self.assertIs(binding.architecture, contracts.architecture)

    def test_operation_lookup_keeps_original_shared_module_and_binding(self) -> None:
        contracts = worker._contracts()
        held = {"held": "schema"}
        self.assertIs(legacy.edit, packaged.edit)
        with patch.object(legacy.edit, "schema", return_value=held) as schema:
            self.assertIs(contracts.comparison_schema(), held)
        schema.assert_called_once_with(contracts.edits)
        self.assertIs(schema.call_args.args[0], contracts.edits)


if __name__ == "__main__":
    unittest.main()
