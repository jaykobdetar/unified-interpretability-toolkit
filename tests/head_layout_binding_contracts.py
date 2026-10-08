"""Pure trusted-receipt adapter checks. No files/model/registry/network reads."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from inference_architecture import head_layout_descriptor, bind_viewer_head_layout
import live_inference as live


class Contracts(unittest.TestCase):
    def setUp(self):
        self.descriptor = head_layout_descriptor()
        self.model = {
            "source_identity": "a" * 64,
            "model_identity": "b" * 64,
            "revision": self.descriptor["source_model"]["revision"],
        }
        self.binding = {
            **{k: self.model[k] for k in ("source_identity", "model_identity")},
            **{
                k: self.descriptor["source_model"][k]
                for k in ("weights_sha256", "config_sha256")
            },
        }

    def test_exact_binding_adds_configuration_evidence_without_runtime_claim(self):
        before = deepcopy(self.model)
        result = bind_viewer_head_layout(self.model, self.binding)
        self.assertEqual(result["head_layout"], self.descriptor)
        self.assertEqual(result["head_layout_binding"], self.binding)
        self.assertEqual(self.model, before)
        self.assertEqual(result["head_layout"]["evidence"], "pinned_configuration")
        self.assertFalse(result["head_layout"]["runtime_verified"])
        self.assertNotIn("inference_source_model", result)
        result["head_layout_binding"]["source_identity"] = "c" * 64
        self.assertEqual(self.binding["source_identity"], "a" * 64)

    def test_absent_malformed_or_mismatched_receipts_remove_annotations(self):
        annotated = {
            **self.model,
            "head_layout": {
                **self.descriptor,
                "evidence": "loaded_builtin_layout",
                "runtime_verified": True,
            },
            "head_layout_binding": self.binding,
        }
        values = [
            None,
            {},
            "receipt",
            {**self.binding, "extra": True},
            *[
                {**self.binding, k: v}
                for k in self.binding
                for v in (None, True, "short", "A" * 64, "c" * 64)
            ],
        ]
        for value in values:
            with self.subTest(value=value):
                result = bind_viewer_head_layout(annotated, value)
                self.assertNotIn("head_layout", result)
                self.assertNotIn("head_layout_binding", result)
        self.assertTrue(annotated["head_layout"]["runtime_verified"])

    def test_stale_source_model_revision_or_comparison_never_binds(self):
        for key, value in [
            ("source_identity", "c" * 64),
            ("model_identity", "c" * 64),
            ("revision", "different"),
            ("comparison_identity", "pair"),
            ("coordinate_space", "checkpoint-comparison-v1"),
        ]:
            with self.subTest(key=key):
                self.assertNotIn(
                    "head_layout",
                    bind_viewer_head_layout({**self.model, key: value}, self.binding),
                )

    def test_legacy_startup_does_not_invent_receipt_and_view_support_is_independent(
        self,
    ):
        session = live.Session("unused", Path("/synthetic/pinned"))
        model = {**self.model, "source_directory": str(session.model)}
        with patch.object(
            live,
            "verify_model",
            side_effect=AssertionError("No hashing at metadata projection"),
        ):
            result = live.bind_inference_source(model, session)
            self.assertNotIn("head_layout", result)
            # Host-approved source correspondence permits viewer labels without
            # granting inference access, changing its worker, or loading a model.
            session.head_layout_binding = dict(self.binding)
            session.inference_enabled = False
            result = live.bind_inference_source(model, session)
            self.assertEqual(result["head_layout_binding"], self.binding)
            self.assertNotIn("inference_source_model", result)
            self.assertIsNone(session.process)
            session.head_layout_binding["model_identity"] = "c" * 64
            self.assertNotIn("head_layout", live.bind_inference_source(model, session))


if __name__ == "__main__":
    unittest.main()
