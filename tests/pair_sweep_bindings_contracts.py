"""Portable held identities, geometry and bounds for passed pair/sweep contracts."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import inference_prompt_pair as pair
import inference_sweep as sweep
import prompt_pair_contracts as pair_tests
import sweep_contracts as sweep_tests
import head_ablation_contracts as head_tests
from inference_edits import SOURCE_MODEL


def held_digest() -> str:
    value = {
        "source_model": SOURCE_MODEL,
        "prompts": ["A", "B"],
        "ids": [[1, 2, 3], [1, 4]],
    }
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


class Contracts(unittest.TestCase):
    def success(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except ValueError as error:
            self.fail(f"Valid held input was rejected: {error}")

    def test_prompt_pair_validate_request(self) -> None:
        request = pair_tests.request()
        self.assertEqual(self.success(pair.validate_request, request), request)
        pair_tests.PairContracts().test_strict_schema_types_and_caps_before_pins()

    def test_prompt_pair_digest_for(self) -> None:
        self.assertEqual(
            pair.digest_for(["A", "B"], [[1, 2, 3], [1, 4]]), held_digest()
        )

    def test_prompt_pair_token_preview(self) -> None:
        actual = self.success(pair.token_preview, pair_tests.Tokenizer(), ["A", "B"])
        expected = {
            "digest": held_digest(),
            "tokens": [
                [
                    {"position": i, "id": token, "piece": "token " + str(token)}
                    for i, token in enumerate(ids)
                ]
                for ids in ([1, 2, 3], [1, 4])
            ],
            "model_loaded": False,
        }
        self.assertEqual(actual, expected)

    def test_prompt_pair_validate_preview(self) -> None:
        preview = pair.token_preview(pair_tests.Tokenizer(), ["A", "B"])
        self.success(pair.validate_preview, preview)
        bad_id, bad_piece, bad_count = (deepcopy(preview) for _ in range(3))
        bad_id["tokens"][0][0]["id"] = 49152
        bad_piece["tokens"][0][0]["piece"] = "x" * 1025
        bad_count["tokens"][0] = [
            {"position": i, "id": 1, "piece": "x"} for i in range(129)
        ]
        for value in (bad_id, bad_piece, bad_count):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pair.validate_preview(value)

    def test_prompt_pair_validate_positions(self) -> None:
        request = pair_tests.request()
        preview = pair.token_preview(pair_tests.Tokenizer(), ["A", "B"])
        self.success(pair.validate_positions, request, preview)
        stale, bound, later = (deepcopy(request) for _ in range(3))
        stale["preview_digest"] = "0" * 64
        bound["positions"] = [{"a": 3, "b": 1}]
        later["positions"] = [{"a": 0, "b": 0}, {"a": 3, "b": 1}]
        for value in (stale, bound, later):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pair.validate_positions(value, preview)

    def test_prompt_pair_difference(self) -> None:
        case = pair_tests.PairContracts()
        self.success(case.test_independent_vector_arithmetic_and_zero_norm)

    def test_prompt_pair_validate_step(self) -> None:
        request, step = pair_tests.request(), pair_tests.step()
        self.success(pair.validate_step, step, request)
        bad_id, bad_piece, bad_metric = (deepcopy(step) for _ in range(3))
        bad_id["prompt_pair"]["a"]["token_id"] = 49152
        bad_piece["prompt_pair"]["a"]["token_piece"] = "x" * 1025
        bad_metric["prompt_pair"]["metrics"]["a_l2"] += 1.0
        for value in (bad_id, bad_piece, bad_metric):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pair.validate_step(value, request)

    def test_sweep_schema(self) -> None:
        fixture = json.loads(
            (
                Path(__file__).resolve().parent
                / "fixtures/inference-public-responses.json"
            ).read_text()
        )
        metadata = json.loads(fixture["responses"]["idle_metadata"]["body_utf8"])
        self.assertEqual(
            json.dumps(sweep.schema(), separators=(",", ":")),
            json.dumps(metadata["sweep"], separators=(",", ":")),
        )

    def test_sweep__target(self) -> None:
        head = {"kind": "head", "layer": 29, "head": 8}
        self.assertEqual(self.success(sweep._target, head), head)
        offset = {"kind": "offset", "layer": 0, "heads": [3, 0], "offset": 63}
        self.assertEqual(
            self.success(sweep._target, offset), {**offset, "heads": [0, 3]}
        )
        for target in (
            {**head, "layer": 30},
            {**head, "head": 9},
            {**offset, "offset": 64},
        ):
            with self.subTest(target=target), self.assertRaises(ValueError):
                sweep._target(target)

    def test_sweep__rows(self) -> None:
        for kind in ("head", "query_head"):
            self.assertEqual(
                sweep._rows({"kind": kind, "layer": 0, "head": 2}), set(range(128, 192))
            )
        self.assertEqual(
            sweep._rows({"kind": "offset", "layer": 0, "heads": [0, 3], "offset": 3}),
            {3, 195},
        )

    def test_sweep__edits(self) -> None:
        for target, projection, axis in (
            ({"kind": "head", "layer": 0, "head": 2}, "o_proj", "columns"),
            ({"kind": "query_head", "layer": 0, "head": 2}, "q_proj", "rows"),
        ):
            self.assertEqual(
                sweep._edits(target, "zero", None),
                [
                    {
                        "tensor": f"model.layers.0.self_attn.{projection}.weight",
                        "shape": [576, 576],
                        "kind": axis,
                        "operation": "zero",
                        "start": 128,
                        "end": 192,
                    }
                ],
            )

    def test_sweep_build_plan(self) -> None:
        fixture = json.loads(
            (
                Path(__file__).resolve().parent
                / "fixtures/sweep-maximum-acceptance.json"
            ).read_text()
        )
        self.assertEqual(
            self.success(sweep.build_plan, fixture["request_without_digest"], False),
            fixture["resolved_plan"],
        )
        self.success(
            head_tests.Contracts().test_all_heads_one_plan_comparison_controls_and_cap
        )

    def test_sweep_validate_step(self) -> None:
        _, plan = sweep_tests.admitted()
        case = plan["cases"][1]
        step = {
            "index": 1,
            "mode": "sweep",
            "layer": 7,
            "activation_site": "block",
            "activation": [0.0] * 576,
            "sweep": {
                "record_id": "target-1/prompt-1",
                "case_id": "target-1",
                "role": "target",
                "prompt_index": 0,
                "selected_cells": case["selected_cells"],
                "changed_cells": 0,
                "parameter_delta_l2": 0.0,
                "restoration_verified": True,
                "metrics": {
                    "logit_delta_rms": 1.0,
                    "logit_delta_max_abs": 1.0,
                    "softmax_total_variation": 0.1,
                    "baseline_argmax_id": 3,
                    "baseline_argmax_logit_delta": 1.0,
                    "edited_argmax_id": 3,
                    "context": "matched fixed original prompt",
                    "semantics": "prompt-set sensitivity; no inferred causal purpose or general head importance",
                },
                "candidates": [
                    {
                        "id": 3,
                        "piece": "held",
                        "baseline_logit": 1.0,
                        "edited_logit": 2.0,
                        "delta": 1.0,
                    }
                ],
            },
        }
        self.success(sweep.validate_step, step, plan)
        bad_width, bad_argmax, bad_delta = (deepcopy(step) for _ in range(3))
        bad_width["activation"] = [0.0] * 575
        bad_argmax["sweep"]["metrics"]["baseline_argmax_id"] = 49152
        bad_delta["sweep"]["candidates"][0]["delta"] = 2.0
        for value in (bad_width, bad_argmax, bad_delta):
            with self.subTest(value=value), self.assertRaises(ValueError):
                sweep.validate_step(value, plan)


if __name__ == "__main__":
    unittest.main()
