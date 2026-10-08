"""Explicit-value geometry uses held pinned data independently of compatibility globals."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import inference_architecture as legacy
import inference_geometry as geometry


class PassedArchitecture(unittest.TestCase):
    def test_explicit_value_keeps_held_descriptor_and_dimensions(self) -> None:
        value = legacy.architecture()
        expected = json.loads(
            (
                Path(__file__).resolve().parent / "fixtures/smollm2-head-layout-v1.json"
            ).read_text()
        )
        self.assertEqual(
            (
                value.width,
                value.layers,
                value.query_heads,
                value.kv_heads,
                value.head_dim,
                value.vocab_size,
            ),
            (576, 30, 9, 3, 64, 49152),
        )
        self.assertIs(value.description, legacy.ARCH)
        self.assertIs(value.config, legacy.CONFIG)
        self.assertIs(value.manifest, legacy.MANIFEST)
        with (
            patch.object(legacy, "WIDTH", 123),
            patch.object(legacy, "HEADS", 1),
            patch.object(legacy, "MANIFEST", {}),
        ):
            self.assertEqual(geometry.head_layout_descriptor(value), expected)
        with self.assertRaises(FrozenInstanceError):
            setattr(value, "width", 123)

    def test_pure_description_uses_supplied_capture_labels(self) -> None:
        config, labels = dict(legacy.CONFIG), dict(legacy.CAPTURE_SITES)
        expected = dict(legacy.ARCH)
        with patch.object(legacy, "CAPTURE_SITES", {}):
            self.assertEqual(geometry.describe(config, labels), expected)
        self.assertEqual(geometry.shapes(expected), legacy.shapes())


if __name__ == "__main__":
    unittest.main()
