"""Shared ordinary-package state and unchanged edit/observation late bindings."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from atlas_host import inference_edit_contract as packaged_edit
from atlas_host import inference_observation_contract as packaged_observation
import inference_edit_contract as legacy_edit
import inference_observation_contract as legacy_observation
import inference_edits as edits
import inference_observations as observations


class PackageContracts(unittest.TestCase):
    def test_module_and_binding_class_identities(self) -> None:
        self.assertIs(legacy_edit, packaged_edit)
        self.assertIs(legacy_observation, packaged_observation)
        self.assertIs(legacy_edit.EditBindings, packaged_edit.EditBindings)
        self.assertIs(
            legacy_observation.ObservationBindings,
            packaged_observation.ObservationBindings,
        )
        self.assertIs(type(edits._bindings()), packaged_edit.EditBindings)
        self.assertIs(
            type(observations._bindings()), packaged_observation.ObservationBindings
        )

    def test_edit_schema_and_late_helper_share_original_state(self) -> None:
        held = {"held": "schema"}
        with patch.object(legacy_edit, "schema", return_value=held) as schema:
            self.assertIs(edits.schema(), held)
        schema.assert_called_once()
        binding = schema.call_args.args[0]
        self.assertIs(binding.source_model, edits.SOURCE_MODEL)
        self.assertIs(binding.shapes, edits.SHAPES)
        self.assertIs(binding.aliases, edits.ALIASES)
        values, source, result = object(), object(), [{"held": "edit"}]
        with patch.object(edits, "validate_edits", return_value=result) as validate:
            self.assertIs(binding.validate_edits(values, source), result)
        validate.assert_called_once_with(values, source)

    def test_observation_delegate_shares_original_bound_values(self) -> None:
        step = {"held": "record"}
        with patch.object(legacy_observation, "validate_record") as validate:
            observations.validate_record(step, None, 0)
        validate.assert_called_once()
        args = validate.call_args.args
        self.assertEqual(args[:3], (step, None, 0))
        self.assertIs(args[3].architecture.description, observations.ARCH)
        self.assertIs(args[3].architecture.capture_sites, observations.CAPTURE_SITES)
        self.assertEqual(args[3].attention_semantics, observations.ATTENTION_SEMANTICS)


if __name__ == "__main__":
    unittest.main()
