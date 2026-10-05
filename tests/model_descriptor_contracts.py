"""Independent small topology fixtures; no ML, model files, process or network."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from inference_model_descriptor import (
    describe_config,
    parameter_shapes,
    pinned_descriptor,
    projection_mappings,
)


def config(family="qwen3", tied=False):
    return {
        "model_type": family,
        "architectures": [
            {
                "llama": "LlamaForCausalLM",
                "qwen2": "Qwen2ForCausalLM",
                "qwen3": "Qwen3ForCausalLM",
            }[family]
        ],
        "hidden_size": 8,
        "intermediate_size": 10,
        "num_hidden_layers": 2,
        "num_attention_heads": 3,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "vocab_size": 11,
        "tie_word_embeddings": tied,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
        "max_position_embeddings": 256,
    }


def receipt(raw):
    return {
        "version": 1,
        "repository": "fixtures/small-qwen",
        "revision": "a" * 40,
        "provenance": "owner_expected",
        "license": {"id": "Apache-2.0", "accepted": True, "file": "LICENSE"},
        "files": [
            {
                "name": name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in [
                ("config.json", raw),
                ("model.safetensors", b"fixture not loaded"),
                ("LICENSE", b"fixture license"),
            ]
        ],
    }


class Contracts(unittest.TestCase):
    def test_qwen3_width_is_not_concatenated_head_width_and_norm_is_shared(self):
        value = config()
        before = deepcopy(value)
        arch = describe_config(value)
        shapes = parameter_shapes(arch)
        self.assertEqual(value, before)
        self.assertEqual(shapes["model.layers.1.self_attn.o_proj.weight"], [8, 12])
        self.assertEqual(shapes["model.layers.0.self_attn.k_proj.weight"], [4, 8])
        self.assertEqual(shapes["model.layers.0.self_attn.q_norm.weight"], [4])
        self.assertEqual(shapes["model.layers.0.self_attn.k_norm.weight"], [4])
        self.assertNotIn("model.layers.0.self_attn.q_proj.bias", shapes)
        self.assertEqual(shapes["lm_head.weight"], [11, 8])
        self.assertEqual(len(shapes), 25)
        self.assertEqual(projection_mappings(arch)["o_proj"]["axis"], "columns")
        self.assertEqual([h // arch["queries_per_kv"] for h in range(3)], [0, 0, 0])

    def test_qwen2_has_qkv_biases_and_no_qk_norm(self):
        shapes = parameter_shapes(describe_config(config("qwen2", True)))
        self.assertEqual(shapes["model.layers.1.self_attn.q_proj.bias"], [12])
        self.assertEqual(shapes["model.layers.1.self_attn.k_proj.bias"], [4])
        self.assertEqual(shapes["model.layers.1.self_attn.v_proj.bias"], [4])
        self.assertNotIn("model.layers.1.self_attn.o_proj.bias", shapes)
        self.assertNotIn("model.layers.1.self_attn.q_norm.weight", shapes)
        self.assertNotIn("lm_head.weight", shapes)
        self.assertEqual(len(shapes), 26)

    def test_pin_complete_topology_count_and_no_execution_promotion(self):
        raw = json.dumps(config()).encode()
        manifest = receipt(raw)
        result = pinned_descriptor(raw, manifest)
        # Embedding + untied head + final norm + 2*(projections + MLP + block norms + Q/K norms).
        self.assertEqual(
            result["parameter_count"],
            88 + 88 + 8 + 2 * (96 + 32 + 32 + 96 + 80 + 80 + 80 + 16 + 8),
        )
        self.assertFalse(result["runtime_verified"])
        self.assertFalse(result["fit_verified"])
        self.assertFalse(result["inference_ready"])
        self.assertEqual(result["source_model"]["weight_files"], [manifest["files"][1]])
        self.assertEqual(result["aliases"], {})
        with self.assertRaises(ValueError):
            pinned_descriptor(raw + b" ", manifest)
        duplicate = b'{"hidden_size":8,"hidden_size":8}'
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            pinned_descriptor(duplicate, receipt(duplicate))

    def test_explicit_supported_features_and_finite_dimensions_only(self):
        invalid = [
            {"model_type": "qwen3_moe"},
            {"architectures": ["CustomModel"]},
            {"num_hidden_layers": True},
            {"num_hidden_layers": 257},
            {"head_dim": 3},
            {"num_key_value_heads": 2},
            {"head_dim": None},
            {"auto_map": {}},
            {"quantization_config": {}},
            {"tie_word_embeddings": None},
            {"attention_bias": True},
            {"rope_scaling": {}},
            {"hidden_act": "gelu"},
            {"sliding_window": 64},
            {"layer_types": ["full_attention", "sliding_attention"]},
            {"rms_norm_eps": float("nan")},
            {"attention_dropout": True},
            {"mlp_bias": True},
            {"max_position_embeddings": 0},
        ]
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                describe_config({**config(), **change})
        with self.assertRaises(ValueError):
            describe_config({**config("qwen2"), "attention_bias": False})
        shapes = parameter_shapes(describe_config(config("llama", True)))
        self.assertEqual(len(shapes), 20)
        self.assertFalse(any(".bias" in k or "q_norm" in k for k in shapes))

    def test_official_pinned_small_qwen_config_has_2048_projection_width(self):
        root = Path(__file__).resolve().parents[1]
        raw = (root / "docs/models/qwen3-0.6b-base-config.json").read_bytes()
        candidate = json.loads(
            (root / "docs/models/qwen3-0.6b-base-candidate.json").read_text()
        )
        self.assertEqual(len(raw), candidate["config.json"]["bytes"])
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(), candidate["config.json"]["sha256"]
        )
        arch = describe_config(json.loads(raw))
        shapes = parameter_shapes(arch)
        self.assertEqual(shapes["model.layers.0.self_attn.o_proj.weight"], [1024, 2048])
        self.assertEqual(
            shapes["model.layers.27.self_attn.k_proj.weight"], [1024, 1024]
        )
        self.assertEqual(shapes["model.layers.27.self_attn.k_norm.weight"], [128])
        self.assertEqual(arch["queries_per_kv"], 2)


if __name__ == "__main__":
    unittest.main()
