"""Passed contracts retain held values and compatibility callback lookup."""

from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import inference_edit_contract as edit_core
import inference_edits as edits
import inference_observation_contract as observation_core
import inference_observations as observations
import observation_contracts as observation_tests


class PassedContracts(unittest.TestCase):
    def test_held_schemas_use_bound_values_after_global_replacement(self) -> None:
        edit_bindings, observation_bindings = (
            edits._bindings(),
            observations._bindings(),
        )
        fixture = json.loads(
            (
                Path(__file__).resolve().parent
                / "fixtures/inference-public-responses.json"
            ).read_text()
        )
        held = json.loads(fixture["responses"]["idle_metadata"]["body_utf8"])
        self.assertIs(edit_bindings.source_model, edits.SOURCE_MODEL)
        self.assertIs(edit_bindings.shapes, edits.SHAPES)
        self.assertIs(edit_bindings.aliases, edits.ALIASES)
        self.assertIs(edit_bindings.architecture.description, edits.ARCH)
        with (
            patch.object(edits, "ARCH", {}),
            patch.object(edits, "SHAPES", {}),
            patch.object(edits, "SOURCE_MODEL", {}),
            patch.object(observations, "HEADS", 1),
            patch.object(observations, "KV_HEADS", 1),
            patch.object(observations, "HEAD_DIM", 1),
        ):
            for actual, expected in [
                (edit_core.schema(edit_bindings), held["comparison"]),
                (observation_core.schema(observation_bindings), held["observations"]),
            ]:
                self.assertEqual(
                    json.dumps(actual, separators=(",", ":")),
                    json.dumps(expected, separators=(",", ":")),
                )

    def test_validation_uses_held_shapes_and_source_identity(self) -> None:
        bindings = edits._bindings()
        source = dict(edits.SOURCE_MODEL)
        name = "model.layers.0.self_attn.q_proj.weight"
        values = [
            {
                "tensor": name,
                "shape": [576, 576],
                "kind": "rows",
                "operation": "zero",
                "start": 0,
                "end": 64,
            }
        ]
        with patch.object(edits, "SOURCE_MODEL", {}), patch.object(edits, "SHAPES", {}):
            self.assertEqual(edit_core.validate_edits(values, source, bindings), values)

    def test_edit_callbacks_resolve_at_invocation(self) -> None:
        bindings = edits._bindings()
        model, values, source = object(), [], edits.SOURCE_MODEL
        validate = Mock(return_value=[])
        parameters = Mock(return_value={})
        torch = SimpleNamespace(no_grad=Mock(side_effect=nullcontext))
        with (
            patch.object(edits, "validate_edits", validate),
            patch.object(edits, "verified_parameters", parameters),
        ):
            self.assertEqual(
                edit_core.apply_edits(torch, model, values, source, bindings), []
            )
        validate.assert_called_once_with(values, source)
        parameters.assert_called_once_with(model)
        torch.no_grad.assert_called_once_with()

    def test_observation_integer_callback_resolves_at_invocation(self) -> None:
        bindings = observations._bindings()
        integer = Mock(return_value=False)
        with (
            patch.object(observations, "_integer", integer),
            self.assertRaisesRegex(ValueError, "^Invalid observation site/position$"),
        ):
            observation_core.validate_record(
                observation_tests.attention(),
                {"kind": "attention", "head": 8},
                7,
                bindings,
            )
        integer.assert_called_once_with(2, 0, 158)

    def test_observation_finite_callback_resolves_at_invocation(self) -> None:
        bindings = observations._bindings()
        finite = Mock(return_value=False)
        with (
            patch.object(observations, "_finite", finite),
            self.assertRaisesRegex(
                ValueError, "^Invalid attention probability distribution$"
            ),
        ):
            observation_core.validate_record(
                observation_tests.attention(),
                {"kind": "attention", "head": 8},
                7,
                bindings,
            )
        finite.assert_called_once_with(0.25)


if __name__ == "__main__":
    unittest.main()
