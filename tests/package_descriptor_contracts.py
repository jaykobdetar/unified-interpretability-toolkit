"""Descriptor package identity and late lookup with inert configuration records."""

from pathlib import Path
import json
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import inference_model_descriptor as legacy
from atlas_host import inference_model_descriptor as canonical
from atlas_host import static_models
from descriptor_package_contracts import configuration, receipt


class PackageDescriptorTests(unittest.TestCase):
    def test_shared_module_and_original_exports(self):
        self.assertIs(legacy, canonical)
        for name in (
            "hashlib",
            "json",
            "math",
            "content_digest",
            "validate_manifest",
            "FAMILIES",
            "MAX_CONFIG_BYTES",
            "RUNTIME_VERSION",
            "_integer",
            "describe_config",
            "parameter_shapes",
            "projection_mappings",
            "pinned_descriptor",
        ):
            self.assertIs(getattr(legacy, name), getattr(canonical, name))
        self.assertIs(static_models.pinned_descriptor, canonical.pinned_descriptor)
        self.assertEqual(static_models.MAX_CONFIG_BYTES, canonical.MAX_CONFIG_BYTES)

    def test_family_state_and_helpers_remain_late(self):
        config = configuration()
        with patch.dict(legacy.FAMILIES, {"qwen3": "FixtureCausalLM"}):
            config["architectures"] = ["FixtureCausalLM"]
            self.assertEqual(
                canonical.describe_config(config)["architecture"], "FixtureCausalLM"
            )
        self.assertEqual(canonical.FAMILIES["qwen3"], "Qwen3ForCausalLM")
        raw = json.dumps(configuration()).encode()
        manifest = receipt(raw)
        with patch.object(
            legacy, "describe_config", wraps=canonical.describe_config
        ) as describe:
            descriptor = canonical.pinned_descriptor(raw, manifest)
            describe.assert_called_once_with(configuration())
            self.assertEqual(descriptor["parameter_count"], 704)
        with patch.object(legacy, "MAX_CONFIG_BYTES", 0):
            with self.assertRaisesRegex(
                ValueError, "^Configuration exceeds byte bound$"
            ):
                canonical.pinned_descriptor(raw, manifest)
        self.assertEqual(canonical.MAX_CONFIG_BYTES, 65536)


if __name__ == "__main__":
    unittest.main()
