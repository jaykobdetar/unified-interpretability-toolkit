#!/usr/bin/env python3
"""No ML/model: strict edit validation, independent arithmetic and trace semantics."""

import copy
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import inference_edits as edits
import inference_worker as worker
import live_inference as live

NAME = "model.layers.0.self_attn.q_proj.weight"
BASE = {
    "tensor": NAME,
    "shape": [576, 576],
    "kind": "rows",
    "operation": "zero",
    "start": 0,
    "end": 64,
}


class EditContracts(unittest.TestCase):
    def validate(self, value):
        return edits.validate_edits(value, edits.SOURCE_MODEL)

    def test_empty_and_whole_head(self):
        self.assertEqual(self.validate([]), [])
        self.assertEqual(self.validate([BASE]), [BASE])
        self.assertEqual(edits.SHAPES[NAME][0] // 9, 64)
        self.assertEqual(len(edits.SHAPES), 211)

    def test_invalid_inputs(self):
        invalid = [None, {}, [None], [BASE] * 9]
        for field, values in {
            "tensor": ["lm_head.weight", "model.norm.weight", "unknown", 1, None],
            "shape": [[576, True], [True, 576], [576, 192], "576,576"],
            "kind": ["head", [], None],
            "operation": ["add", {}, None],
            "start": [True, 0.0, -1, 64, None],
            "end": [False, 64.0, 0, 577, None],
        }.items():
            invalid.extend([{**BASE, field: value}] for value in values)
        invalid += [
            [{**BASE, "extra": 1}],
            [{k: v for k, v in BASE.items() if k != "end"}],
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.validate(value)

    def test_scales_and_element_strict_types(self):
        for scale in [
            True,
            False,
            None,
            "0.5",
            math.inf,
            -math.inf,
            math.nan,
            101,
            -101,
            10**1000,
        ]:
            with self.subTest(scale=str(scale)[:20]), self.assertRaises(ValueError):
                self.validate([{**BASE, "operation": "scale", "scale": scale}])
        for scale in [-100, 0, 0.5, 100]:
            self.validate([{**BASE, "operation": "scale", "scale": scale}])
        scalar = {k: v for k, v in BASE.items() if k not in ("start", "end")}
        scalar.update(kind="element", row=575, col=575)
        self.validate([scalar])
        for key in ("row", "col"):
            for value in (True, False, 0.0, -1, 576):
                with self.assertRaises(ValueError):
                    self.validate([{**scalar, key: value}])
        self.validate([{**BASE, "kind": "columns", "start": 575, "end": 576}])

    def test_qwen_and_unbound_identity_rejected_before_model_or_process(self):
        for source in [
            None,
            {},
            {**edits.SOURCE_MODEL, "repo": "Qwen/Qwen3-8B"},
            {**edits.SOURCE_MODEL, "revision": "stale"},
        ]:
            with self.assertRaises(ValueError):
                edits.validate_edits([], source)
            session = live.Session("python", Path("/unused"))
            with (
                patch.object(
                    live, "verify_model", side_effect=AssertionError("must not read")
                ),
                patch.object(
                    live.subprocess,
                    "Popen",
                    side_effect=AssertionError("must not launch"),
                ),
                self.assertRaises(ValueError),
            ):
                session.start(
                    {"prompt": "synthetic", "edits": [BASE], "source_model": source}
                )

    def test_loaded_mapping_checks_shapes_dtype_and_all_aliases(self):
        from types import SimpleNamespace

        class Parameter:
            def __init__(self, shape, pointer):
                self.shape, self.pointer = shape, pointer
                self.ndim = len(shape)
                self.dtype, self.device = "torch.float32", SimpleNamespace(type="cpu")

            def is_contiguous(self):
                return True

            def storage_offset(self):
                return 0

            def untyped_storage(self):
                return SimpleNamespace(data_ptr=lambda: self.pointer)

        parameters = {
            name: Parameter(shape, index + 1)
            for index, (name, shape) in enumerate(edits.SHAPES.items())
        }
        parameters["lm_head.weight"] = parameters["model.embed_tokens.weight"]
        parameters["model.norm.weight"] = Parameter([576], 1000)
        config = SimpleNamespace(
            model_type="llama",
            hidden_size=576,
            intermediate_size=1536,
            num_hidden_layers=30,
            num_attention_heads=9,
            num_key_value_heads=3,
            vocab_size=49152,
            tie_word_embeddings=True,
        )
        model = SimpleNamespace(
            config=config, named_parameters=lambda **_: parameters.items()
        )
        self.assertEqual(set(edits.verified_parameters(model)), set(parameters))
        query = parameters[NAME]
        for field, value in [
            ("shape", [576, 192]),
            ("dtype", "torch.bfloat16"),
            ("device", SimpleNamespace(type="cuda")),
        ]:
            prior = getattr(query, field)
            setattr(query, field, value)
            with self.assertRaises(ValueError):
                edits.verified_parameters(model)
            setattr(query, field, prior)
        parameters["model.norm.weight"].pointer = query.pointer
        with self.assertRaises(ValueError):
            edits.verified_parameters(model)
        parameters["model.norm.weight"].pointer = 1000
        parameters["lm_head.weight"] = Parameter([49152, 576], 1)
        with self.assertRaises(ValueError):
            edits.verified_parameters(model)

    def test_independent_ordered_arithmetic(self):
        # Small helper matrix; validation still uses the declared pinned shape.
        values = [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0], [7.0, 8.0, 9.0]]

        class Selection:
            def __init__(self, rows, cols):
                self.cells = [(r, c) for r in rows for c in cols]

            def zero_(self):
                for r, c in self.cells:
                    values[r][c] = 0.0

            def mul_(self, scale):
                for r, c in self.cells:
                    values[r][c] *= scale

        class Matrix:
            def __getitem__(self, slices):
                return Selection(
                    range(*slices[0].indices(3)), range(*slices[1].indices(3))
                )

        class Finite:
            def all(self):
                return all(math.isfinite(v) for row in values for v in row)

        from contextlib import nullcontext
        from types import SimpleNamespace

        torch = SimpleNamespace(no_grad=nullcontext, isfinite=lambda _: Finite())
        operations = [
            {**BASE, "end": 2, "operation": "scale", "scale": -2},
            {**BASE, "kind": "columns", "start": 1, "end": 2},
            {
                "tensor": NAME,
                "shape": [576, 576],
                "kind": "element",
                "operation": "scale",
                "scale": 0.5,
                "row": 1,
                "col": 2,
            },
        ]
        with patch.object(edits, "verified_parameters", return_value={NAME: Matrix()}):
            edits.apply_edits(torch, object(), operations, edits.SOURCE_MODEL)
        self.assertEqual(values, [[-2.0, 0.0, -6.0], [-8.0, 0.0, 6.0], [7.0, 0.0, 9.0]])

    def test_score_union_deltas_and_prefixes(self):
        from types import SimpleNamespace

        tokenizer = SimpleNamespace(decode=lambda ids, **_: str(ids[0]))

        def step(i, token, tops):
            return {
                "index": i,
                "token_id": token,
                "token_piece": str(token),
                "generated_text": str(token),
                "eos": False,
                "compute_ms": 1,
                "compute_total_ms": i + 1,
                "top_logits": [{"id": t, "value": 0} for t in tops],
            }

        left = [step(0, 1, [0, 1]), step(1, 3, [1, 3])]
        right = [step(0, 2, [0, 2]), step(1, 3, [2, 3])]
        a, b = [[1.0, 3.0, 2.0, 4.0]] * 2, [[2.0, 1.0, 3.0, 6.0]] * 2
        pair = worker.paired_step(0, left, right, a, b, tokenizer)
        self.assertEqual(pair["alignment"], "matched_prefix")
        self.assertEqual(
            [
                (c["id"], c["baseline_logit"], c["edited_logit"], c["delta"])
                for c in pair["candidates"]
            ],
            [(0, 1.0, 2.0, 1.0), (1, 3.0, 1.0, -2.0), (2, 2.0, 3.0, 1.0)],
        )
        edits.validate_pair(pair)
        second = worker.paired_step(1, left, right, a, b, tokenizer)
        self.assertEqual(second["alignment"], "different_prefix")
        edits.validate_pair(second)
        ended = worker.paired_step(1, left, right[:1], a, b[:1], tokenizer)
        edits.validate_pair(ended)
        self.assertEqual(ended["alignment"], "branch_ended")
        self.assertTrue(
            all(
                c["delta"] is None and c["edited_logit"] is None
                for c in ended["candidates"]
            )
        )
        for field, value in [
            ("alignment", "matched_prefix"),
            ("candidates", second["candidates"] * 4),
        ]:
            bad = {**second, field: value}
            with self.assertRaises(ValueError):
                edits.validate_pair(bad)
        bad = copy.deepcopy(pair)
        bad["candidates"][0]["delta"] = 0.1
        with self.assertRaises(ValueError):
            edits.validate_pair(bad)


if __name__ == "__main__":
    unittest.main()
