"""Exact small descriptor records and decoder controls; no model work."""

import ast
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 8,
        "intermediate_size": 10,
        "num_hidden_layers": 1,
        "num_attention_heads": 3,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "vocab_size": 11,
        "tie_word_embeddings": False,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
        "max_position_embeddings": 256,
    }


def architecture():
    return {
        "model_type": "qwen3",
        "architecture": "Qwen3ForCausalLM",
        "width": 8,
        "layers": 1,
        "query_heads": 3,
        "kv_heads": 1,
        "head_dim": 4,
        "queries_per_kv": 3,
        "vocab_size": 11,
        "intermediate_size": 10,
        "max_position_embeddings": 256,
        "tie_word_embeddings": False,
        "qkv_bias": False,
        "qk_head_norm": True,
        "layout": "qwen3-eager-head-major-v1",
        "evidence": "configuration_only",
        "runtime_verified": False,
        "fit_verified": False,
        "inference_ready": False,
    }


def shapes():
    return {
        "model.embed_tokens.weight": [11, 8],
        "model.norm.weight": [8],
        "lm_head.weight": [11, 8],
        "model.layers.0.self_attn.q_proj.weight": [12, 8],
        "model.layers.0.self_attn.k_proj.weight": [4, 8],
        "model.layers.0.self_attn.v_proj.weight": [4, 8],
        "model.layers.0.self_attn.o_proj.weight": [8, 12],
        "model.layers.0.mlp.gate_proj.weight": [10, 8],
        "model.layers.0.mlp.up_proj.weight": [10, 8],
        "model.layers.0.mlp.down_proj.weight": [8, 10],
        "model.layers.0.input_layernorm.weight": [8],
        "model.layers.0.post_attention_layernorm.weight": [8],
        "model.layers.0.self_attn.q_norm.weight": [4],
        "model.layers.0.self_attn.k_norm.weight": [4],
    }


def mappings():
    return {
        "q_proj": {"axis": "rows", "heads": 3, "head_dim": 4, "shape": [12, 8]},
        "k_proj": {"axis": "rows", "heads": 1, "head_dim": 4, "shape": [4, 8]},
        "v_proj": {"axis": "rows", "heads": 1, "head_dim": 4, "shape": [4, 8]},
        "o_proj": {"axis": "columns", "heads": 3, "head_dim": 4, "shape": [8, 12]},
    }


def receipt(raw):
    return {
        "version": 1,
        "repository": "fixtures/descriptor",
        "revision": "a" * 40,
        "provenance": "owner_expected",
        "license": {"id": "Apache-2.0", "accepted": True, "file": "LICENSE"},
        "files": [
            {
                "name": name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in (
                ("config.json", raw),
                ("model.safetensors", b"fixture not loaded"),
                ("LICENSE", b"fixture license"),
            )
        ],
    }


class DescriptorPackageTests(unittest.TestCase):
    def setUp(self):
        try:
            import inference_model_descriptor
        except Exception as exc:
            self.fail("pure descriptor must initialize: " + repr(exc))
        self.descriptor = inference_model_descriptor

    def valid(self, function, *args):
        try:
            return function(*args)
        except Exception as exc:
            self.fail("valid local descriptor must complete: " + repr(exc))

    def error(self, function, message, *args):
        try:
            function(*args)
        except ValueError as exc:
            self.assertEqual(str(exc), message)
        except Exception as exc:
            self.fail("expected local ValueError: " + repr(exc))
        else:
            self.fail("expected local ValueError: " + message)

    def test_integer(self):
        for number in (1, 8):
            self.assertEqual(
                self.valid(self.descriptor._integer, number, 1, 8, "width"), number
            )
        self.error(
            self.descriptor._integer,
            "Invalid bounded configuration dimension: width",
            True,
            1,
            8,
            "width",
        )

    def test_describe_config(self):
        value = configuration()
        before = deepcopy(value)
        self.assertEqual(
            self.valid(self.descriptor.describe_config, value), architecture()
        )
        self.assertEqual(value, before)

    def test_parameter_shapes(self):
        value = architecture()
        before = deepcopy(value)
        self.assertEqual(self.valid(self.descriptor.parameter_shapes, value), shapes())
        self.assertEqual(value, before)

    def test_projection_mappings(self):
        value = architecture()
        before = deepcopy(value)
        self.assertEqual(
            self.valid(self.descriptor.projection_mappings, value), mappings()
        )
        self.assertEqual(value, before)

    def test_pinned_descriptor(self):
        raw = json.dumps(configuration(), separators=(",", ":")).encode("utf-8")
        manifest = receipt(raw)
        before = deepcopy(manifest)
        portable = {
            "version": 1,
            "repository": "fixtures/descriptor",
            "revision": "a" * 40,
            "license": {"id": "Apache-2.0", "file": "LICENSE"},
            "files": sorted(deepcopy(manifest["files"]), key=lambda row: row["name"]),
        }
        identity_bytes = json.dumps(
            ["weight-atlas-content-v1", portable],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        expected = {
            "schema": "weight-atlas-model-descriptor-v1",
            **architecture(),
            "content_digest": hashlib.sha256(identity_bytes).hexdigest(),
            "adapter_id": "builtin-qwen3-eager",
            "adapter_version": 1,
            "reviewed_transformers": "4.56.2",
            "source_model": {
                "repo": "fixtures/descriptor",
                "revision": "a" * 40,
                "config_sha256": hashlib.sha256(raw).hexdigest(),
                "weight_files": [deepcopy(manifest["files"][1])],
            },
            "native_weight_layout": ["output_feature", "input_feature"],
            "projection_mappings": mappings(),
            "parameter_shapes": shapes(),
            "parameter_count": 704,
            "aliases": {},
            "qk_semantics": "per-head RMSNorm after projection, before rotary",
            "head_ablation": "o_proj columns",
            "query_intervention": "q_proj rows; preserve declared bias",
            "runtime_support": "requires verified complete dense files, exact built-in runtime, measured fit and owner admission",
        }
        self.assertEqual(
            self.valid(self.descriptor.pinned_descriptor, raw, manifest), expected
        )
        self.assertEqual(manifest, before)

    def test_unique(self):
        # Execute the actual nested function AST without tying the test to file
        # layout or formatting; this decoder has no captured local values.
        source = textwrap.dedent(inspect.getsource(self.descriptor.pinned_descriptor))
        tree = ast.parse(source)
        found = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "unique"
        ]
        self.assertEqual(len(found), 1)
        namespace = dict(self.descriptor.pinned_descriptor.__globals__)
        try:
            exec(
                compile(
                    ast.Module(body=found, type_ignores=[]),
                    self.descriptor.pinned_descriptor.__code__.co_filename,
                    "exec",
                ),
                namespace,
            )
        except Exception as exc:
            self.fail("actual decoder must initialize: " + repr(exc))
        unique = namespace["unique"]
        item = [1, 2]
        actual = self.valid(unique, [("b", item), ("a", None)])
        self.assertEqual(actual, {"b": [1, 2], "a": None})
        self.assertEqual(list(actual), ["b", "a"])
        self.assertIs(actual["b"], item)
        self.error(unique, "Duplicate configuration key", [("a", 1), ("a", 2)])


if __name__ == "__main__":
    unittest.main()
