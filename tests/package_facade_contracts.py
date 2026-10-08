"""Ordinary facade package imports preserve one shared module and late state."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from atlas_host import inference_edits as edits
from atlas_host import inference_observations as observations
import inference_edits as legacy_edits
import inference_observations as legacy_observations
import inference_prompt_pair as pair
import inference_sweep as sweep
import inference_worker as worker
import live_inference as live


class PackageContracts(unittest.TestCase):
    def test_modules_classes_and_imported_values_share_identity(self) -> None:
        self.assertIs(legacy_edits, edits)
        self.assertIs(legacy_observations, observations)
        self.assertIs(legacy_edits.EditBindings, edits.EditBindings)
        self.assertIs(
            legacy_observations.ObservationBindings, observations.ObservationBindings
        )
        for caller in (pair, sweep):
            self.assertIs(caller.edit_facade, edits)
            self.assertIs(caller.SOURCE_MODEL, edits.SOURCE_MODEL)
        self.assertIs(live.SOURCE_MODEL, edits.SOURCE_MODEL)
        self.assertIs(live.schema, edits.schema)
        self.assertIs(live.observation_schema, observations.schema)
        self.assertIs(worker.lens_record, observations.lens_record)
        self.assertIs(worker.validate_record, observations.validate_record)
        self.assertIs(worker.validate_observation, observations.validate_observation)
        binding = edits._bindings()
        self.assertIs(binding.source_model, edits.SOURCE_MODEL)
        self.assertIs(binding.shapes, edits.SHAPES)
        self.assertIs(binding.aliases, edits.ALIASES)

    def test_default_and_explicit_factories_keep_late_lookup(self) -> None:
        values, source, expected = [], object(), [{"held": 1}]
        default = edits._bindings()
        with patch.object(
            legacy_edits, "validate_edits", return_value=expected
        ) as validate:
            self.assertIs(default.validate_edits(values, source), expected)
        validate.assert_called_once_with(values, source)
        value = edits.architecture()
        explicit = edits._bindings(value)
        self.assertIs(explicit.architecture, value)
        with patch.object(
            legacy_edits.contract, "validate_edits", return_value=expected
        ) as validate:
            self.assertIs(explicit.validate_edits(values, source), expected)
        validate.assert_called_once_with(values, source, explicit)
        observed = observations._bindings()
        with patch.object(
            legacy_observations, "_integer", return_value=True
        ) as integer:
            self.assertIs(observed.integer(3, 2, 7), True)
        integer.assert_called_once_with(3, 2, 7)
        for caller in (worker, live):
            graph = caller._contracts()
            for field in ("edits", "observations", "pairs", "sweeps"):
                self.assertIs(getattr(graph, field).architecture, graph.architecture)


if __name__ == "__main__":
    unittest.main()
