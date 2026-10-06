"""Exact legacy geometry forwarding, bound defaults and trusted receipt records."""

from importlib import import_module
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


class ArchitectureContracts(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.architecture = import_module("inference_architecture")
        except Exception as error:
            self.fail(f"The production architecture module must load: {error!r}")

    def test_describe(self) -> None:
        architecture = self.architecture
        config, result = {"held": "config"}, {"held": "description"}
        with patch.object(
            architecture.geometry, "describe", return_value=result
        ) as describe:
            self.assertIs(architecture.describe(config), result)
        describe.assert_called_once_with(config, architecture.CAPTURE_SITES)
        self.assertIs(describe.call_args.args[0], config)
        self.assertIs(describe.call_args.args[1], architecture.CAPTURE_SITES)

    def test_architecture(self) -> None:
        architecture = self.architecture
        value = architecture.architecture()
        self.assertIs(value.description, architecture.ARCH)
        self.assertIs(value.config, architecture.CONFIG)
        self.assertIs(value.manifest, architecture.MANIFEST)
        self.assertIs(value.capture_sites, architecture.CAPTURE_SITES)
        self.assertEqual(
            (
                value.width,
                value.layers,
                value.query_heads,
                value.kv_heads,
                value.head_dim,
                value.vocab_size,
            ),
            (
                architecture.WIDTH,
                architecture.LAYERS,
                architecture.HEADS,
                architecture.KV_HEADS,
                architecture.HEAD_DIM,
                architecture.VOCAB,
            ),
        )

    def test_shapes(self) -> None:
        architecture = self.architecture
        value, result = {"held": "description"}, {"held": [1, 2]}
        with patch.object(
            architecture.geometry, "shapes", return_value=result
        ) as shapes:
            self.assertIs(architecture.shapes(value), result)
        shapes.assert_called_once_with(value)
        self.assertIs(shapes.call_args.args[0], value)
        with patch.object(
            architecture.geometry, "shapes", return_value=result
        ) as shapes:
            self.assertIs(architecture.shapes(), result)
        shapes.assert_called_once_with(architecture.ARCH)
        self.assertIs(shapes.call_args.args[0], architecture.ARCH)

    def test_head_layout_descriptor(self) -> None:
        architecture = self.architecture
        value, result = object(), {"held": "layout"}
        with (
            patch.object(architecture, "architecture", return_value=value) as current,
            patch.object(
                architecture.geometry, "head_layout_descriptor", return_value=result
            ) as describe,
        ):
            self.assertIs(architecture.head_layout_descriptor(), result)
        current.assert_called_once_with()
        describe.assert_called_once_with(value)
        self.assertIs(describe.call_args.args[0], value)

    def test_verify_attention_layout(self) -> None:
        import package_geometry_contracts as engine

        engine.PackageGeometry().test_delegate()

    def test_bind_viewer_head_layout(self) -> None:
        import head_layout_binding_contracts as receipts

        checks = receipts.Contracts()
        checks.setUp()
        checks.test_exact_binding_adds_configuration_evidence_without_runtime_claim()
        checks.test_absent_malformed_or_mismatched_receipts_remove_annotations()
        checks.test_stale_source_model_revision_or_comparison_never_binds()


if __name__ == "__main__":
    unittest.main()
