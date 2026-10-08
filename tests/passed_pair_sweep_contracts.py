"""Held passed geometry/identities and late pair/sweep compatibility callbacks."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import inference_pair_contract as pair_core
import inference_prompt_pair as pair
import inference_sweep as sweep
import inference_sweep_contract as sweep_core
import prompt_pair_contracts as pair_tests
import pair_sweep_bindings_contracts as held_tests


class PassedContracts(unittest.TestCase):
    def test_pair_uses_held_width_vocab_and_source_after_global_replacement(
        self,
    ) -> None:
        bindings = pair._bindings()
        preview = pair.token_preview(pair_tests.Tokenizer(), ["A", "B"])
        a, b = [3.0, 4.0] + [0.0] * 574, [6.0, 8.0] + [0.0] * 574
        self.assertIs(bindings.source_model, pair.SOURCE_MODEL)
        self.assertIs(bindings.architecture.description, pair.ARCH)
        with (
            patch.object(pair, "WIDTH", 1),
            patch.object(pair, "VOCAB", 1),
            patch.object(pair, "SOURCE_MODEL", {}),
        ):
            self.assertEqual(
                pair_core.difference(a, b, bindings=bindings),
                (a, {"a_l2": 5.0, "b_l2": 10.0, "delta_l2": 5.0, "cosine": 1.0}),
            )
            pair_core.validate_preview(preview, bindings=bindings)
            self.assertEqual(
                pair_core.digest_for(
                    ["A", "B"], [[1, 2, 3], [1, 4]], bindings=bindings
                ),
                held_tests.held_digest(),
            )

    def test_pair_callbacks_resolve_at_invocation(self) -> None:
        bindings = pair._bindings()
        digest, validate = Mock(return_value="0" * 64), Mock()
        with (
            patch.object(pair, "digest_for", digest),
            patch.object(pair, "validate_preview", validate),
        ):
            result = pair_core.token_preview(
                pair_tests.Tokenizer(), ["A", "B"], bindings=bindings
            )
        self.assertEqual(result["digest"], "0" * 64)
        digest.assert_called_once_with(["A", "B"], [[1, 2, 3], [1, 4]])
        validate.assert_called_once_with(result)

    def test_sweep_schema_and_geometry_use_held_values(self) -> None:
        bindings = sweep._bindings()
        fixture = json.loads(
            (
                Path(__file__).resolve().parent
                / "fixtures/inference-public-responses.json"
            ).read_text()
        )
        schema = json.loads(fixture["responses"]["idle_metadata"]["body_utf8"])["sweep"]
        self.assertIs(bindings.shapes, sweep.SHAPES)
        self.assertIs(bindings.architecture.description, sweep.ARCH)
        with (
            patch.object(sweep, "HEADS", 1),
            patch.object(sweep, "HEAD_DIM", 1),
            patch.object(sweep, "ARCH", {}),
            patch.object(sweep, "SHAPES", {}),
        ):
            self.assertEqual(
                json.dumps(sweep_core.schema(bindings=bindings), separators=(",", ":")),
                json.dumps(schema, separators=(",", ":")),
            )
            self.assertEqual(
                sweep_core._rows(
                    {"kind": "head", "layer": 0, "head": 2}, bindings=bindings
                ),
                set(range(128, 192)),
            )
            self.assertEqual(
                sweep_core._edits(
                    {"kind": "head", "layer": 0, "head": 2},
                    "zero",
                    None,
                    bindings=bindings,
                ),
                [
                    {
                        "tensor": "model.layers.0.self_attn.o_proj.weight",
                        "shape": [576, 576],
                        "kind": "columns",
                        "operation": "zero",
                        "start": 128,
                        "end": 192,
                    }
                ],
            )

    def test_sweep_integer_callback_resolves_at_invocation(self) -> None:
        bindings = sweep._bindings()
        integer = Mock(return_value=False)
        with (
            patch.object(sweep, "_int", integer),
            self.assertRaisesRegex(ValueError, "^Choose a native target layer 0–29$"),
        ):
            sweep_core._target(
                {"kind": "head", "layer": 0, "head": 2}, bindings=bindings
            )
        integer.assert_called_once_with(0, 0, 29)

    def test_sweep_rows_callback_resolves_at_invocation(self) -> None:
        bindings = sweep._bindings()
        rows = Mock(return_value={3})
        target = {"kind": "offset", "layer": 0, "heads": [0], "offset": 3}
        with patch.object(sweep, "_rows", rows):
            result = sweep_core._edits(target, "zero", None, bindings=bindings)
        rows.assert_called_once_with(target)
        self.assertEqual(result[0]["start"], 3)
        self.assertEqual(result[0]["end"], 4)


if __name__ == "__main__":
    unittest.main()
