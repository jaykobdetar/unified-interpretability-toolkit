"""Ordinary package imports share legacy state and preserve held defaults."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from atlas_host import inference_engine as packaged_engine
from atlas_host import inference_geometry as packaged_geometry
import inference_engine as legacy_engine
import inference_geometry as legacy_geometry
import inference_architecture as architecture
import passed_caller_contracts as caller_checks
import architecture_callers_contracts as default_checks


class PackageIdentity(unittest.TestCase):
    def test_geometry_names_share_the_same_module_and_class(self) -> None:
        self.assertIs(legacy_geometry, packaged_geometry)
        self.assertIs(legacy_geometry.Architecture, packaged_geometry.Architecture)
        held = {"held": "description"}
        with patch.object(legacy_geometry, "describe", return_value=held):
            self.assertIs(architecture.describe(architecture.CONFIG), held)

    def test_engine_names_share_the_same_module(self) -> None:
        self.assertIs(legacy_engine, packaged_engine)
        self.assertIs(legacy_engine.LoaderRuntime, packaged_engine.LoaderRuntime)
        self.assertIs(legacy_engine.AttentionRuntime, packaged_engine.AttentionRuntime)

    def test_six_default_builders_retain_held_values_and_late_callbacks(self) -> None:
        caller_checks.PassedCallers().test_worker_and_coordinator_keep_one_value()
        checks = default_checks.Contracts()
        for module in default_checks.MODULES.values():
            checks.factory(module)
        callbacks = [
            name
            for name in unittest.defaultTestLoader.getTestCaseNames(
                default_checks.Contracts
            )
            if name.startswith("test_callback_")
        ]
        self.assertEqual(len(callbacks), 18)
        for name in callbacks:
            getattr(checks, name)()


if __name__ == "__main__":
    unittest.main()
