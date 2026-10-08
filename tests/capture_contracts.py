#!/usr/bin/env python3
"""Strict capture selection and pre-load validation, without numerical imports."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import live_inference as live
import inference_worker as worker


class ReachedPinnedCheck(Exception):
    pass


class CaptureContracts(unittest.TestCase):
    def test_valid_sites_and_any_native_layer_reach_pinned_check(self):
        session = live.Session("unused", Path("/unused"))
        for layer in range(30):
            for site in ("block", "attention", "mlp"):
                with (
                    patch.object(live, "available", return_value=6 * live.GIB),
                    patch.object(live, "verify_model", side_effect=ReachedPinnedCheck),
                    self.assertRaises(ReachedPinnedCheck),
                ):
                    session.start(
                        {"prompt": "fixture", "layer": layer, "activation_site": site}
                    )
        self.assertIsNone(session.process)

    def test_invalid_selection_rejected_before_model_read(self):
        for field, values in {
            "layer": [-1, 30, True, False, 1.5, None, "7"],
            "activation_site": ["q_proj", "pre_residual", "", True, None, {}, []],
        }.items():
            for value in values:
                session = live.Session("unused", Path("/unused"))
                with (
                    self.subTest(field=field, value=value),
                    patch.object(
                        live,
                        "verify_model",
                        side_effect=AssertionError("No model access permitted"),
                    ),
                    self.assertRaises(ValueError),
                ):
                    session.start({"prompt": "fixture", field: value})

    def test_worker_validates_selection_before_hooks(self):
        tokenizer = SimpleNamespace(encode=lambda *_args, **_kwargs: [1])
        for site in ("q_proj", None, True, [], ""):
            with self.assertRaises(ValueError):
                worker.generate(
                    None, tokenizer, None, "fixture", 1, 0, activation_site=site
                )
        for layer in (True, 1.5, 30, -1):
            with self.assertRaises(ValueError):
                worker.generate(None, tokenizer, None, "fixture", 1, layer)
        self.assertEqual(worker.LAYERS, worker.ARCH["layers"])
        self.assertIn(
            "before attention residual addition", worker.CAPTURE_SITES["attention"]
        )
        self.assertIn("before MLP residual addition", worker.CAPTURE_SITES["mlp"])
        self.assertIn(
            "after attention and MLP residual additions", worker.CAPTURE_SITES["block"]
        )


if __name__ == "__main__":
    unittest.main()
