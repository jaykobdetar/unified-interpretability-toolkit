"""Small direct/control tests for the two-prompt contract; no model or HTTP."""

from copy import deepcopy
import json
import io
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import inference_prompt_pair as pair
import live_inference as live
import inference_worker as worker
from inference_edits import SOURCE_MODEL


class Tokenizer:
    def encode(self, text, **_kwargs):
        return SimpleNamespace(
            ids={"A": [1, 2, 3], "B": [1, 4], "empty": [], "long": [1] * 129}[text]
        )

    def decode(self, ids, **_kwargs):
        return "token " + str(ids[0])


def request():
    preview = pair.token_preview(Tokenizer(), ["A", "B"])
    return {
        "mode": "prompt_pair",
        "source_model": SOURCE_MODEL,
        "prompts": ["A", "B"],
        "layer": 7,
        "activation_site": "block",
        "positions": [{"a": 2, "b": 1}],
        "preview_digest": preview["digest"],
    }


def step():
    a, b = [1.0, 0.0] + [0.0] * 574, [0.0, 1.0] + [0.0] * 574
    delta, metrics = pair.difference(a, b)
    return {
        "index": 0,
        "mode": "prompt_pair",
        "layer": 7,
        "activation_site": "block",
        "activation": delta,
        "prompt_pair": {
            "a": {"position": 2, "token_id": 3, "token_piece": "a", "activation": a},
            "b": {"position": 1, "token_id": 4, "token_piece": "b", "activation": b},
            "token_equal": False,
            "prefix_equal": False,
            "metrics": metrics,
        },
    }


class ReachedPins(Exception):
    pass


class PairContracts(unittest.TestCase):
    def test_preview_worker_never_loads_model(self):
        data = {
            "mode": "prompt_pair_preview",
            "source_model": SOURCE_MODEL,
            "prompts": ["A", "B"],
        }
        events = []
        with (
            patch.object(worker.sys, "argv", ["worker", "/unused"]),
            patch.object(
                worker.sys,
                "stdin",
                SimpleNamespace(buffer=io.BytesIO((json.dumps(data) + "\n").encode())),
            ),
            patch.object(worker.os, "sched_setaffinity"),
            patch.object(worker.os, "nice"),
            patch.object(worker.resource, "setrlimit"),
            patch.object(live, "verify_model"),
            patch.object(
                worker,
                "load_engine",
                side_effect=AssertionError("Preview must not load a model"),
            ),
            patch.object(worker, "emit", side_effect=events.append),
            patch.dict(
                sys.modules,
                {
                    "tokenizers": SimpleNamespace(
                        Tokenizer=SimpleNamespace(from_file=lambda _path: Tokenizer())
                    )
                },
            ),
        ):
            worker.main()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "preview_done")
        self.assertFalse(events[0]["preview"]["model_loaded"])

    def test_closed_requests_reach_pins_with_no_model_access(self):
        for mode in pair.MODES:
            data = (
                request()
                if mode == "prompt_pair"
                else {"mode": mode, "source_model": SOURCE_MODEL, "prompts": ["A", "B"]}
            )
            session = live.Session("unused", Path("/unused"))
            with (
                patch.object(live, "available", return_value=6 * live.GIB),
                patch.object(live, "verify_model", side_effect=ReachedPins),
                self.assertRaises(ReachedPins),
            ):
                session.start(data)
            self.assertIsNone(session.process)

    def test_strict_schema_types_and_caps_before_pins(self):
        cases = []
        for field, values in {
            "prompts": [
                [],
                ["A"],
                ["A", "B", "C"],
                [True, "B"],
                ["", "B"],
                ["é" * 1025, "B"],
                ["\x00" * 2048, "B"],
            ],
            "positions": [
                [],
                [{"a": 0, "b": 0}] * 2,
                [{"a": i, "b": i} for i in range(9)],
                [{"a": True, "b": 0}],
                [{"a": 1.0, "b": 0}],
                [{"a": 128, "b": 0}],
                [{"a": -1, "b": 0}],
                [{"a": 0, "b": 0, "c": 0}],
            ],
            "layer": [True, -1, 30, "1"],
            "activation_site": [True, None, [], "q_proj"],
            "preview_digest": [None, True, "0" * 63, "X" * 64],
        }.items():
            for value in values:
                data = request()
                data[field] = value
                cases.append(data)
        for field, value in [
            ("edits", []),
            ("observation", {"kind": "logit_lens"}),
            ("max_new_tokens", 1),
        ]:
            data = request()
            data[field] = value
            cases.append(data)
        data = request()
        data["source_model"] = {**SOURCE_MODEL, "repo": "Qwen/Qwen3-8B"}
        cases.append(data)
        for data in cases:
            with (
                self.subTest(data=data),
                patch.object(
                    live, "verify_model", side_effect=AssertionError("No file access")
                ),
                self.assertRaises(ValueError),
            ):
                live.Session("unused", Path("/unused")).start(data)

    def test_preview_bounds_digest_and_stale_request(self):
        p = pair.token_preview(Tokenizer(), ["A", "B"])
        self.assertEqual([len(x) for x in p["tokens"]], [3, 2])
        self.assertFalse(p["model_loaded"])
        pair.validate_positions(request(), p)
        for prompts in (["empty", "A"], ["A", "long"]):
            with self.assertRaises(ValueError):
                pair.token_preview(Tokenizer(), prompts)
        r = request()
        r["positions"] = [{"a": 3, "b": 1}]
        with self.assertRaises(ValueError):
            pair.validate_positions(r, p)
        with self.assertRaises(ValueError):
            pair.validate_positions(
                request(), pair.token_preview(Tokenizer(), ["B", "A"])
            )
        self.assertNotEqual(
            pair.digest_for(["A", "B"], [[1], [2]]),
            pair.digest_for(["A ", "B"], [[1], [2]]),
        )
        for bad in [True, 1.0, -1, 49152]:
            changed = deepcopy(p)
            changed["tokens"][0][0]["id"] = bad
            with self.assertRaises(ValueError):
                pair.validate_preview(changed)

    def test_sha256_preview_identity_is_independent_of_head_dimension(self):
        preview = pair.token_preview(Tokenizer(), ["A", "B"])
        data = request()
        self.assertEqual(len(preview["digest"]), 64)
        for width in (32, 128):
            with self.subTest(head_dim=width), patch.object(pair, "HEAD_DIM", width):
                pair.validate_preview(preview)
                self.assertEqual(
                    pair.validate_request(data)["preview_digest"], preview["digest"]
                )
                self.assertEqual(
                    pair.token_preview(Tokenizer(), ["A", "B"])["digest"],
                    preview["digest"],
                )
                for bad in ("0" * width, preview["digest"].upper(), "g" * 64):
                    with self.assertRaises(ValueError):
                        pair.validate_preview({**preview, "digest": bad})
                    with self.assertRaises(ValueError):
                        pair.validate_request({**data, "preview_digest": bad})

    def test_independent_vector_arithmetic_and_zero_norm(self):
        a, b = [3.0, 4.0] + [0.0] * 574, [6.0, 8.0] + [0.0] * 574
        delta, metrics = pair.difference(a, b)
        self.assertEqual(delta, a)
        self.assertEqual(
            metrics, {"a_l2": 5.0, "b_l2": 10.0, "delta_l2": 5.0, "cosine": 1.0}
        )
        self.assertEqual(
            pair.difference(a, a),
            ([0.0] * 576, {"a_l2": 5.0, "b_l2": 5.0, "delta_l2": 0.0, "cosine": 1.0}),
        )
        self.assertEqual(pair.difference(a, [-x for x in a])[1]["cosine"], -1.0)
        self.assertIsNone(pair.difference([0.0] * 576, b)[1]["cosine"])
        self.assertEqual(step()["prompt_pair"]["metrics"]["delta_l2"], math.sqrt(2))
        for bad in ([0.0] * 575, [True] + [0.0] * 575, [float("nan")] + [0.0] * 575):
            with self.assertRaises(ValueError):
                pair.difference(a, bad)

    def test_trace_alignment_geometry_metrics_and_coverage(self):
        pair.validate_step(step(), request())
        changes = [
            ("index", 1),
            ("index", True),
            ("layer", True),
            ("mode", "generation"),
            ("activation_site", "mlp"),
            ("activation", [0.0] * 576),
        ]
        for key, value in changes:
            changed = step()
            changed[key] = value
            with self.assertRaises(ValueError):
                pair.validate_step(changed, request())
        for key, value in [
            ("token_equal", True),
            ("prefix_equal", True),
            (
                "metrics",
                {"a_l2": True, "b_l2": 1.0, "delta_l2": math.sqrt(2), "cosine": 0.0},
            ),
        ]:
            changed = step()
            changed["prompt_pair"][key] = value
            with self.assertRaises(ValueError):
                pair.validate_step(changed, request())
        # The maximum selected trace is under the original 32x576 scalar budget.
        self.assertEqual(8 * 3 * 576, 13824)
        self.assertLess(8 * 3 * 576, 32 * 576)
        self.assertLess(len(json.dumps([step()] * 8).encode()), 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
