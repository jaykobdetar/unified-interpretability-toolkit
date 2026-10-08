"""Held default packets and one architecture identity across bound caller contracts."""

from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from inference_architecture import architecture
import inference_edits as edits
import inference_observations as observations
import inference_prompt_pair as pair
import inference_sweep as sweep
import inference_worker as worker
import live_inference as live
import prompt_pair_contracts as pair_tests


class PassedCallers(unittest.TestCase):
    def test_factories_keep_the_same_held_architecture(self) -> None:
        value = architecture()
        for module in (edits, observations, pair, sweep):
            binding = module._bindings(value)
            self.assertIs(binding.architecture, value)
        self.assertIs(edits._bindings(value).source_model, edits.SOURCE_MODEL)
        self.assertIs(edits._bindings(value).shapes, edits.SHAPES)
        self.assertIs(edits._bindings(value).aliases, edits.ALIASES)
        self.assertIs(sweep._bindings(value).shapes, sweep.SHAPES)

    def test_worker_and_coordinator_keep_one_value(self) -> None:
        for module in (worker, live):
            contracts = module._contracts()
            for binding in (
                contracts.edits,
                contracts.observations,
                contracts.pairs,
                contracts.sweeps,
            ):
                self.assertIs(binding.architecture, contracts.architecture)
            self.assertEqual(contracts.architecture.width, 576)
            self.assertEqual(contracts.architecture.vocab_size, 49152)

    def test_bound_schemas_keep_original_serialization(self) -> None:
        held = json.loads(
            (
                Path(__file__).resolve().parent
                / "fixtures/inference-public-responses.json"
            ).read_text()
        )
        metadata = json.loads(held["responses"]["idle_metadata"]["body_utf8"])
        contracts = live._contracts()
        for name, method in (
            ("comparison", contracts.comparison_schema),
            ("observations", contracts.observations_schema),
            ("sweep", contracts.sweep_schema),
        ):
            self.assertEqual(json.dumps(method()), json.dumps(metadata[name]))

    def test_explicit_pair_callbacks_stay_bound(self) -> None:
        contracts = worker._contracts()
        request = pair_tests.request()
        expected = pair.token_preview(pair_tests.Tokenizer(), ["A", "B"])
        step = pair_tests.step()
        with ExitStack() as stack:
            for module, names in (
                (edits, ("validate_edits",)),
                (pair, ("digest_for", "validate_preview", "difference")),
            ):
                for name in names:
                    stack.enter_context(
                        patch.object(
                            module,
                            name,
                            side_effect=AssertionError("legacy default callback used"),
                        )
                    )
            self.assertEqual(contracts.validate_edits([], edits.SOURCE_MODEL), [])
            self.assertEqual(contracts.validate_request(request), request)
            self.assertEqual(
                contracts.token_preview(pair_tests.Tokenizer(), ["A", "B"]), expected
            )
            contracts.validate_positions(request, expected)
            contracts.validate_pair_step(step, request)

    def test_explicit_sweep_callbacks_keep_original_plan(self) -> None:
        fixture = json.loads(
            (
                Path(__file__).resolve().parent
                / "fixtures/sweep-maximum-acceptance.json"
            ).read_text()
        )
        contracts = live._contracts()
        with ExitStack() as stack:
            for module, names in (
                (edits, ("validate_edits",)),
                (sweep, ("_target", "_rows", "_edits", "schema")),
            ):
                for name in names:
                    stack.enter_context(
                        patch.object(
                            module,
                            name,
                            side_effect=AssertionError("legacy default callback used"),
                        )
                    )
            self.assertEqual(
                contracts.build_plan(fixture["request_without_digest"], False),
                fixture["resolved_plan"],
            )


if __name__ == "__main__":
    unittest.main()
