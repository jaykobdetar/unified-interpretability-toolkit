"""Held layout records and inert engine-loader contracts; no ML execution."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import inference_architecture as architecture
import inference_edits as edits
import inference_worker as worker

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def held_layout() -> dict[str, Any]:
    return json.loads((FIXTURES / "smollm2-head-layout-v1.json").read_text())


class BoundaryContracts(unittest.TestCase):
    def test_held_description_and_shapes(self) -> None:
        expected = json.loads((FIXTURES / "inference-architecture.json").read_text())
        self.assertEqual(architecture.describe(architecture.CONFIG), expected)
        # Order matters to the serialized legacy evidence, including source IDs.
        self.assertEqual(
            json.dumps(architecture.head_layout_descriptor(), separators=(",", ":")),
            json.dumps(held_layout(), separators=(",", ":")),
        )
        config = {
            **architecture.CONFIG,
            "hidden_size": 24,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "num_hidden_layers": 2,
            "intermediate_size": 40,
            "vocab_size": 60,
        }
        desc = architecture.describe(config)
        shapes = architecture.shapes(desc)
        expected_shapes = {"model.embed_tokens.weight": [60, 24]}
        for layer in (0, 1):
            for suffix, shape in [
                ("self_attn.q_proj", [32, 24]),
                ("self_attn.k_proj", [16, 24]),
                ("self_attn.v_proj", [16, 24]),
                ("self_attn.o_proj", [24, 32]),
                ("mlp.gate_proj", [40, 24]),
                ("mlp.up_proj", [40, 24]),
                ("mlp.down_proj", [24, 40]),
            ]:
                expected_shapes[f"model.layers.{layer}.{suffix}.weight"] = shape
        self.assertEqual(shapes, expected_shapes)
        self.assertEqual(desc["queries_per_kv"], 2)

    def test_runtime_geometry_and_exact_descriptor(self) -> None:
        class Model:
            pass

        class Block:
            pass

        class Attention:
            pass

        class Linear:
            def __init__(self, output: int, input_: int) -> None:
                self.out_features, self.in_features = output, input_
                self.bias = None

        model: Any = Model()
        model.training = False
        model.config = SimpleNamespace(_attn_implementation="eager")
        blocks: list[Any] = []
        for _ in range(30):
            block: Any = Block()
            attn: Any = Attention()
            block.self_attn = attn
            attn.head_dim, attn.num_key_value_groups = 64, 3
            for name, output, input_ in [
                ("q_proj", 576, 576),
                ("k_proj", 192, 576),
                ("v_proj", 192, 576),
                ("o_proj", 576, 576),
            ]:
                setattr(attn, name, Linear(output, input_))
            blocks.append(block)
        model.model = SimpleNamespace(layers=blocks)
        modules = {
            "torch": SimpleNamespace(nn=SimpleNamespace(Linear=Linear)),
            "transformers": SimpleNamespace(__version__="4.56.2"),
            "transformers.models.llama.modeling_llama": SimpleNamespace(
                LlamaForCausalLM=Model,
                LlamaAttention=Attention,
                LlamaDecoderLayer=Block,
            ),
        }
        expected = {
            **held_layout(),
            "evidence": "loaded_builtin_layout",
            "runtime_verified": True,
            "runtime": {
                "transformers": "4.56.2",
                "attention_backend": "eager",
                "device": "cpu",
                "dtype": "float32",
            },
        }
        with patch.dict(sys.modules, modules):
            self.assertEqual(architecture.verify_attention_layout(model), expected)
            for obj, key, value, error in [
                (
                    model,
                    "training",
                    True,
                    "Unverified built-in Llama implementation or training mode",
                ),
                (
                    model.config,
                    "_attn_implementation",
                    "sdpa",
                    "Unverified attention implementation or layer count",
                ),
                (
                    blocks[0].self_attn.o_proj,
                    "in_features",
                    192,
                    "Projection is not the verified output-by-input linear layout",
                ),
            ]:
                with self.subTest(key=key):
                    prior = getattr(obj, key)
                    setattr(obj, key, value)
                    with self.assertRaises(ValueError) as failure:
                        architecture.verify_attention_layout(model)
                    self.assertEqual(str(failure.exception), error)
                    setattr(obj, key, prior)

    def test_engine_loader_pins_and_call_order(self) -> None:
        ordered = Mock()
        torch = SimpleNamespace(
            __version__="fake",
            float32=object(),
            set_num_threads=Mock(),
            set_num_interop_threads=Mock(),
            manual_seed=Mock(),
            use_deterministic_algorithms=Mock(),
        )
        for name in (
            "set_num_threads",
            "set_num_interop_threads",
            "manual_seed",
            "use_deterministic_algorithms",
        ):
            ordered.attach_mock(getattr(torch, name), name)
        tokenizer = object()
        model = SimpleNamespace()
        model.eval = Mock(return_value=model)
        tokenizer_factory = Mock(return_value=tokenizer)
        pretrained = Mock(return_value=model)
        ordered.attach_mock(tokenizer_factory, "tokenizer")
        ordered.attach_mock(pretrained, "pretrained")
        ordered.attach_mock(model.eval, "eval")
        transformers = SimpleNamespace(
            LlamaForCausalLM=SimpleNamespace(from_pretrained=pretrained),
            PreTrainedTokenizerFast=tokenizer_factory,
        )
        directory = Path("/unused")
        with (
            patch.dict(sys.modules, {"torch": torch, "transformers": transformers}),
            patch.object(edits, "verified_parameters") as parameters,
            patch.object(
                worker, "verify_attention_layout", return_value={"held": "layout"}
            ) as layout,
        ):
            ordered.attach_mock(parameters, "parameters")
            ordered.attach_mock(layout, "layout")
            result = worker.load_engine(directory)
        self.assertEqual(result, (torch, tokenizer, model))
        self.assertEqual(model._atlas_verified_layout, {"held": "layout"})
        self.assertEqual(
            ordered.mock_calls,
            [
                call.set_num_threads(1),
                call.set_num_interop_threads(1),
                call.manual_seed(0),
                call.use_deterministic_algorithms(True),
                call.tokenizer(tokenizer_file="/unused/tokenizer.json"),
                call.pretrained(
                    "/unused",
                    local_files_only=True,
                    use_safetensors=True,
                    dtype=torch.float32,
                    attn_implementation="eager",
                ),
                call.eval(),
                call.parameters(model),
                call.layout(model),
            ],
        )


if __name__ == "__main__":
    unittest.main()
