"""Experiment request and record edges with synthetic protocol data only."""

from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import inference_edits as edits
import inference_observations as observation
import inference_prompt_pair as pair
import inference_sweep as sweep
import edit_contracts
import observation_contracts
import prompt_pair_contracts
import sweep_contracts


class ExperimentLimits(unittest.TestCase):
    def check(self, operation, accepted):
        if accepted:
            try:
                operation()
            except ValueError as error:
                self.fail(f"Accepted boundary refused: {error}")
        else:
            with self.assertRaises(ValueError):
                operation()

    def test_edit_count_coordinates_ranges_and_float_scales(self):
        for count in (0, 8, 9):
            with self.subTest(edits=count):
                self.check(
                    lambda: edits.validate_edits(
                        [edit_contracts.BASE] * count, edits.SOURCE_MODEL
                    ),
                    count <= 8,
                )
        for axis in ("row", "col"):
            for n, accepted in ((-1, False), (0, True), (575, True), (576, False)):
                value = {
                    "tensor": edit_contracts.NAME,
                    "shape": [576, 576],
                    "kind": "element",
                    "operation": "zero",
                    "row": 0,
                    "col": 0,
                    axis: n,
                }
                with self.subTest(axis=axis, value=n):
                    self.check(
                        lambda: edits.validate_edits([value], edits.SOURCE_MODEL),
                        accepted,
                    )
        for kind in ("rows", "columns"):
            for start, end, accepted in (
                (-1, 1, False),
                (0, 1, True),
                (575, 576, True),
                (0, 576, True),
                (0, 0, False),
                (0, 577, False),
                (576, 576, False),
            ):
                value = {
                    **edit_contracts.BASE,
                    "kind": kind,
                    "start": start,
                    "end": end,
                }
                with self.subTest(kind=kind, start=start, end=end):
                    self.check(
                        lambda: edits.validate_edits([value], edits.SOURCE_MODEL),
                        accepted,
                    )
        for scale, accepted in (
            (math.nextafter(-100, -math.inf), False),
            (-100, True),
            (100, True),
            (math.nextafter(100, math.inf), False),
        ):
            value = {**edit_contracts.BASE, "operation": "scale", "scale": scale}
            with self.subTest(scale=scale):
                self.check(
                    lambda: edits.validate_edits([value], edits.SOURCE_MODEL), accepted
                )

    def test_attention_request_heads_and_record_positions(self):
        for head, accepted in ((-1, False), (0, True), (8, True), (9, False)):
            with self.subTest(head=head):
                self.check(
                    lambda: observation.validate_observation(
                        {"kind": "attention", "head": head}, "attention"
                    ),
                    accepted,
                )
        for position, accepted in ((-1, False), (0, True), (158, True), (159, False)):
            step = observation_contracts.attention()
            n = max(1, position + 1)
            step["position"] = position
            step["attention"].update(
                query_position=position,
                key_positions=list(range(n)),
                key_token_ids=[3] * n,
                probabilities=[0] * (n - 1) + [1],
            )
            with self.subTest(position=position):
                self.check(
                    lambda: observation.validate_record(
                        step, {"kind": "attention", "head": 8}, 7
                    ),
                    accepted,
                )
        # The schema's 160-key ceiling includes capacity beyond the final
        # consumed step; the position guard makes at most 159 keys reachable.
        self.assertEqual(observation.schema()["max_attention_keys"], 160)

    def test_attention_sum_tolerance_closest_representable_edges(self):
        upper = 1 + 1e-5
        lower = 1 - 1e-5
        cases = (
            (math.nextafter(upper, -math.inf), True),
            (upper, False),
            (lower, True),
            (math.nextafter(lower, -math.inf), False),
        )
        for total, accepted in cases:
            step = observation_contracts.attention()
            step["attention"]["probabilities"] = [0.5, total - 0.5, 0.0]
            with self.subTest(total=total):
                self.assertEqual(sum(step["attention"]["probabilities"]), total)
                self.check(
                    lambda: observation.validate_record(
                        step, {"kind": "attention", "head": 8}, 7
                    ),
                    accepted,
                )

    def lens(self, count=1, token=0, piece="x"):
        step = observation_contracts.lens()
        step["token_id"] = token
        value = step["logit_lens"]
        value.update(
            lens_argmax_id=token,
            final_argmax_id=token,
            candidates=[
                {
                    "id": token + i,
                    "piece": piece,
                    "lens_logit": 2,
                    "final_logit": 1,
                    "delta_lens_minus_final": 1,
                }
                for i in range(count)
            ],
        )
        return step

    def test_lens_candidate_count_piece_and_vocab_edges(self):
        for count, accepted in ((0, False), (1, True), (10, True), (11, False)):
            with self.subTest(candidates=count):
                self.check(
                    lambda: observation.validate_record(
                        self.lens(count), {"kind": "logit_lens"}, 29
                    ),
                    accepted,
                )
        for token, accepted in ((-1, False), (0, True), (49151, True), (49152, False)):
            with self.subTest(token=token):
                self.check(
                    lambda: observation.validate_record(
                        self.lens(token=token), {"kind": "logit_lens"}, 29
                    ),
                    accepted,
                )
        for length, accepted in ((0, True), (1024, True), (1025, False)):
            with self.subTest(piece_characters=length):
                self.check(
                    lambda: observation.validate_record(
                        self.lens(piece="é" * length), {"kind": "logit_lens"}, 29
                    ),
                    accepted,
                )

    def test_pair_request_positions_layers_digest_and_utf8_prompt_edges(self):
        for count, accepted in ((0, False), (1, True), (8, True), (9, False)):
            request = prompt_pair_contracts.request()
            request["positions"] = [{"a": i, "b": i} for i in range(count)]
            with self.subTest(position_pairs=count):
                self.check(lambda: pair.validate_request(request), accepted)
        for field in ("a", "b"):
            for position, accepted in (
                (-1, False),
                (0, True),
                (127, True),
                (128, False),
            ):
                request = prompt_pair_contracts.request()
                request["positions"] = [{"a": 0, "b": 0, field: position}]
                with self.subTest(field=field, position=position):
                    self.check(lambda: pair.validate_request(request), accepted)
        for field, values in (
            ("layer", ((-1, False), (0, True), (29, True), (30, False))),
            (
                "preview_digest",
                (("a" * 63, False), ("a" * 64, True), ("a" * 65, False)),
            ),
        ):
            for value, accepted in values:
                request = {**prompt_pair_contracts.request(), field: value}
                with self.subTest(field=field, value=value):
                    self.check(lambda: pair.validate_request(request), accepted)
        for prompt, accepted in (
            ("", False),
            ("x", True),
            ("é" * 1024, True),
            ("é" * 1024 + "x", False),
        ):
            request = {
                "mode": "prompt_pair_preview",
                "source_model": edits.SOURCE_MODEL,
                "prompts": [prompt, "x"],
            }
            with self.subTest(prompt_bytes=len(prompt.encode())):
                self.check(lambda: pair.validate_request(request), accepted)
        for count in (1, 2, 3):
            request = {
                "mode": "prompt_pair_preview",
                "source_model": edits.SOURCE_MODEL,
                "prompts": ["x"] * count,
            }
            with self.subTest(prompts=count):
                self.check(lambda: pair.validate_request(request), count == 2)

    def test_pair_complete_request_actual_json_byte_edge(self):
        base = {
            "mode": "prompt_pair_preview",
            "source_model": edits.SOURCE_MODEL,
            "prompts": ["x" * 2048, "x" * 2048],
        }
        initial = len(json.dumps(base, ensure_ascii=False, allow_nan=False).encode())
        for size in (8192, 8193):
            escaped = size - initial
            self.assertTrue(0 <= escaped <= 4096)
            request = deepcopy(base)
            for i in range(2):
                count = min(2048, escaped)
                request["prompts"][i] = '"' * count + "x" * (2048 - count)
                escaped -= count
            with self.subTest(bytes=size):
                self.assertEqual(
                    len(
                        json.dumps(
                            request, ensure_ascii=False, allow_nan=False
                        ).encode()
                    ),
                    size,
                )
                if size == 8192:
                    pair.validate_request(request)
                else:
                    with self.assertRaisesRegex(
                        ValueError, "Complete prompt-pair request"
                    ):
                        pair.validate_request(request)

    def preview(self, count=1, token=0, piece=""):
        return {
            "digest": "a" * 64,
            "tokens": [
                [{"position": i, "id": token, "piece": piece} for i in range(count)]
                for _ in range(2)
            ],
            "model_loaded": False,
        }

    def test_token_preview_counts_pieces_and_vocabulary(self):
        for count, accepted in ((0, False), (1, True), (128, True), (129, False)):
            with self.subTest(tokens=count):
                self.check(lambda: pair.validate_preview(self.preview(count)), accepted)
                tokenizer = SimpleNamespace(
                    encode=lambda *_args, **_kw: SimpleNamespace(ids=[0] * count),
                    decode=lambda *_args, **_kw: "x",
                )
                self.check(lambda: pair.token_preview(tokenizer, ["A", "B"]), accepted)
        for token, accepted in ((-1, False), (0, True), (49151, True), (49152, False)):
            with self.subTest(token=token):
                self.check(
                    lambda: pair.validate_preview(self.preview(token=token)), accepted
                )
        for piece, accepted in (
            ("", True),
            ("é" * 512, True),
            ("é" * 512 + "x", False),
        ):
            with self.subTest(piece_bytes=len(piece.encode())):
                self.check(
                    lambda: pair.validate_preview(self.preview(piece=piece)), accepted
                )
        for length in (63, 64, 65):
            with self.subTest(digest_characters=length):
                self.check(
                    lambda: pair.validate_preview(
                        {**self.preview(), "digest": "a" * length}
                    ),
                    length == 64,
                )

    def test_token_preview_complete_actual_byte_edge(self):
        for size in (65536, 65537):
            value = self.preview(128)
            remaining = size - len(json.dumps(value, ensure_ascii=False).encode())
            for tokens in value["tokens"]:
                for token in tokens:
                    amount = min(1024, remaining)
                    token["piece"] = "x" * amount
                    remaining -= amount
            with self.subTest(bytes=size):
                self.assertEqual(remaining, 0)
                self.assertEqual(
                    len(json.dumps(value, ensure_ascii=False).encode()), size
                )
                if size == 65536:
                    pair.validate_preview(value)
                else:
                    with self.assertRaisesRegex(ValueError, "64 KiB"):
                        pair.validate_preview(value)

    def test_pair_trace_token_piece_and_vector_edges(self):
        for side in ("a", "b"):
            for token, accepted in (
                (-1, False),
                (0, True),
                (49151, True),
                (49152, False),
            ):
                value = prompt_pair_contracts.step()
                value["prompt_pair"][side]["token_id"] = token
                value["prompt_pair"]["token_equal"] = (
                    value["prompt_pair"]["a"]["token_id"]
                    == value["prompt_pair"]["b"]["token_id"]
                )
                with self.subTest(side=side, token=token):
                    self.check(
                        lambda: pair.validate_step(
                            value, prompt_pair_contracts.request()
                        ),
                        accepted,
                    )
            for piece, accepted in (
                ("", True),
                ("é" * 512, True),
                ("é" * 512 + "x", False),
            ):
                value = prompt_pair_contracts.step()
                value["prompt_pair"][side]["token_piece"] = piece
                with self.subTest(side=side, piece_bytes=len(piece.encode())):
                    self.check(
                        lambda: pair.validate_step(
                            value, prompt_pair_contracts.request()
                        ),
                        accepted,
                    )
        for width in (575, 576, 577):
            with self.subTest(vector_width=width):
                self.check(
                    lambda: pair.difference([0] * width, [0] * 576), width == 576
                )
                self.check(
                    lambda: pair.difference([0] * 576, [0] * width), width == 576
                )

    def test_sweep_declared_plan_dimensions_and_derived_capacity(self):
        fields = {
            "seed": ((-1, False), (0, True), (2**32 - 1, True), (2**32, False)),
            "capture_layer": ((-1, False), (0, True), (29, True), (30, False)),
        }
        for field, cases in fields.items():
            for value, accepted in cases:
                request = {**sweep_contracts.request(), field: value}
                with self.subTest(field=field, value=value):
                    self.check(lambda: sweep.build_plan(request, False), accepted)
        for kind, field, last in (
            ("head", "head", 8),
            ("query_head", "head", 8),
            ("offset", "offset", 63),
        ):
            for value, accepted in (
                (-1, False),
                (0, True),
                (last, True),
                (last + 1, False),
            ):
                request = sweep_contracts.request()
                target = {"kind": kind, "layer": 0, field: value}
                if kind == "offset":
                    target["heads"] = [0]
                request["targets"] = [target]
                with self.subTest(kind=kind, field=field, value=value):
                    self.check(lambda: sweep.build_plan(request, False), accepted)
        for count, accepted in ((0, False), (1, True), (8, True), (9, False)):
            request = sweep_contracts.request()
            request["targets"][0]["heads"] = list(range(count))
            with self.subTest(offset_heads=count):
                self.check(lambda: sweep.build_plan(request, False), accepted)
        for prompt, accepted in (
            ("", False),
            ("x", True),
            ("é" * 1024, True),
            ("é" * 1024 + "x", False),
        ):
            request = {**sweep_contracts.request(), "prompts": [prompt]}
            with self.subTest(prompt_bytes=len(prompt.encode())):
                self.check(lambda: sweep.build_plan(request, False), accepted)
        request = sweep_contracts.request()
        request["prompts"] = ["A", "B"]
        request["targets"] = [
            {"kind": "offset", "layer": 0, "heads": [0], "offset": i} for i in range(2)
        ]
        plan = sweep.build_plan(request, False)
        self.assertEqual(
            (len(plan["cases"]), plan["records"], plan["prefills"]), (5, 10, 20)
        )
        request["targets"] = [{"kind": "layer_heads", "layer": 0}]
        with self.assertRaisesRegex(ValueError, "32-record cap"):
            sweep.build_plan(request, False)
        request["prompts"] = ["A"]
        self.assertEqual(sweep.build_plan(request, False)["records"], 19)


if __name__ == "__main__":
    unittest.main()
