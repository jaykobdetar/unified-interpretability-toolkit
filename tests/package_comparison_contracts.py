"""Comparison package identity, worker imports and held callback records."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from atlas_host import inference_comparison as packaged
import inference_comparison as legacy
import inference_worker as worker
import experiment_interface_contracts as held


class PackageContracts(unittest.TestCase):
    def test_public_imports_share_module_class_and_functions(self) -> None:
        self.assertIs(legacy, packaged)
        self.assertIs(legacy.ComparisonBindings, packaged.ComparisonBindings)
        self.assertIs(worker.ComparisonBindings, packaged.ComparisonBindings)
        self.assertIs(worker.comparison_step, packaged.paired_step)
        self.assertIs(worker.run_comparison, packaged.run)
        self.assertIs(legacy.Event, packaged.Event)
        result = {"held": "comparison"}
        with patch.object(legacy, "paired_step", return_value=result) as callback:
            self.assertIs(packaged.paired_step(0, [], [], [], [], None), result)
        callback.assert_called_once_with(0, [], [], [], [], None)

    def test_original_worker_callback_order_and_exact_records(self) -> None:
        held.InterfaceContracts().test_comparison_packets_and_callback_order()


if __name__ == "__main__":
    unittest.main()
