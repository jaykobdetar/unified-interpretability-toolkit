"""Held schemas and portable correctness gaps for passed edit/observation bindings."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, call

import inference_edits as edits
import inference_observations as obs
import edit_contracts as edit_tests
import observation_contracts as obs_tests


class Scores:
    shape = (49152,)

    def __init__(self, top: list[int], best: int, bias: float) -> None:
        self.top, self.best, self.bias = top, best, bias

    def __getitem__(self, token: int) -> float:
        return token + self.bias


def paired(ids: list[int], candidates: int = 1) -> dict[str, Any]:
    side = {"generated_ids": ids, "token_id": ids[-1], "generated_text": "held"}
    return {
        "index": len(ids) - 1,
        "alignment": "matched_prefix",
        "score_kind": "raw FP32 logits",
        "activation_branch": "edited",
        "baseline": deepcopy(side),
        "edited": deepcopy(side),
        "candidates": [
            {"id": i, "baseline_logit": 1.0, "edited_logit": 2.0, "delta": 1.0}
            for i in range(candidates)
        ],
    }


class BindingContracts(unittest.TestCase):
    def test_held_public_schemas(self) -> None:
        path = (
            Path(__file__).resolve().parent / "fixtures/inference-public-responses.json"
        )
        held = json.loads(path.read_text())["responses"]["idle_metadata"]["body_utf8"]
        metadata = json.loads(held)
        for actual, expected in [
            (edits.schema(), metadata["comparison"]),
            (obs.schema(), metadata["observations"]),
        ]:
            self.assertEqual(
                json.dumps(actual, separators=(",", ":")),
                json.dumps(expected, separators=(",", ":")),
            )

    def test_existing_edit_validation_contracts(self) -> None:
        case = edit_tests.EditContracts()
        case.test_empty_and_whole_head()
        case.test_invalid_inputs()
        case.test_scales_and_element_strict_types()
        case.test_qwen_and_unbound_identity_rejected_before_model_or_process()

    def test_existing_parameter_contracts(self) -> None:
        edit_tests.EditContracts().test_loaded_mapping_checks_shapes_dtype_and_all_aliases()

    def test_existing_edit_arithmetic(self) -> None:
        edit_tests.EditContracts().test_independent_ordered_arithmetic()

    def test_pair_domain_and_size_edges(self) -> None:
        for ids, count in [([1], 1), ([1] * 32, 10), ([49151], 10)]:
            edits.validate_pair(paired(ids, count))
        for ids, count in [([1] * 33, 1), ([49152], 1), ([1], 11)]:
            with (
                self.subTest(ids=len(ids), last=ids[-1], candidates=count),
                self.assertRaises(ValueError),
            ):
                edits.validate_pair(paired(ids, count))

    def test_existing_observation_requests(self) -> None:
        case = obs_tests.Contracts()
        case.test_request_schema_and_pinned_validation_order()
        case.test_bad_requests_stop_before_file_access()

    def test_observation_geometry_and_sum_edges(self) -> None:
        case = obs_tests.Contracts()
        case.test_attention_address_distribution_and_bounds()
        case.test_only_accepted_mode_layer_site_branch()
        case.test_lens_union_both_values_delta_and_semantics()
        step = obs_tests.attention()
        step["attention"]["probabilities"] = [0.5, 0.50002, 0.0]
        with self.assertRaises(ValueError):
            obs.validate_record(step, {"kind": "attention", "head": 8}, 7)

    def test_inert_lens_union_and_argmax_inclusion(self) -> None:
        # Missing top-k boundary maxima exercise the existing replacement rule.
        lens, final = Scores([0, 1, 2, 3, 4], 6, 0.5), Scores([0, 1, 2, 3, 5], 7, 1.5)
        torch = SimpleNamespace(
            isfinite=Mock(return_value=SimpleNamespace(all=lambda: True)),
            topk=Mock(
                side_effect=lambda scores, count: SimpleNamespace(
                    indices=SimpleNamespace(tolist=lambda: scores.top[:count])
                )
            ),
            argmax=Mock(side_effect=lambda scores: scores.best),
        )
        residual, normalized = object(), object()
        model = SimpleNamespace(
            model=SimpleNamespace(norm=Mock(return_value=normalized)),
            lm_head=Mock(return_value=lens),
        )
        tokenizer = SimpleNamespace(
            decode=Mock(side_effect=lambda ids, **_: "token " + str(ids[0]))
        )
        result = obs.lens_record(torch, tokenizer, residual, model, final, 7, 2)
        expected = {
            "layer": 7,
            "position": 2,
            "score_kind": "raw FP32 logits",
            "lens_argmax_id": 6,
            "final_argmax_id": 7,
            "candidates": [
                {
                    "id": token,
                    "piece": "token " + str(token),
                    "lens_logit": token + 0.5,
                    "final_logit": token + 1.5,
                    "delta_lens_minus_final": -1.0,
                }
                for token in [0, 1, 2, 3, 6, 7]
            ],
            "semantics": obs.LENS_SEMANTICS,
        }
        self.assertEqual(result, expected)
        model.model.norm.assert_called_once_with(residual)
        model.lm_head.assert_called_once_with(normalized)
        self.assertEqual(torch.topk.call_args_list, [call(lens, 5), call(final, 5)])
        self.assertEqual(
            torch.argmax.call_args_list,
            [call(lens), call(final), call(lens), call(final)],
        )
        obs.validate_record(
            {
                "index": 0,
                "layer": 7,
                "activation_site": "block",
                "position": 2,
                "token_id": 7,
                "logit_lens": result,
            },
            {"kind": "logit_lens"},
            7,
        )


if __name__ == "__main__":
    unittest.main()
