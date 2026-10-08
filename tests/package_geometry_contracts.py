"""Existing held engine/geometry checks and exact lazy delegate forwarding."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import inference_architecture as architecture
import inference_engine as engine
import architecture_boundary_contracts as held


class PackageGeometry(unittest.TestCase):
    def test_geometry(self) -> None:
        held.BoundaryContracts().test_held_description_and_shapes()

    def test_verifier(self) -> None:
        held.BoundaryContracts().test_runtime_geometry_and_exact_descriptor()

    def test_loader(self) -> None:
        held.BoundaryContracts().test_engine_loader_pins_and_call_order()

    def test_delegate(self) -> None:
        model, result = object(), {"held": "layout"}
        verify = Mock(return_value=result)
        classes = [type(name, (), {}) for name in ("Model", "Attention", "Block")]
        torch, transformers = object(), object()
        modules = {
            "torch": torch,
            "transformers": transformers,
            "transformers.models.llama.modeling_llama": SimpleNamespace(
                LlamaForCausalLM=classes[0],
                LlamaAttention=classes[1],
                LlamaDecoderLayer=classes[2],
            ),
        }
        with (
            patch.dict(sys.modules, modules),
            patch.object(engine, "verify_attention_layout", verify),
        ):
            self.assertIs(architecture.verify_attention_layout(model), result)
        verify.assert_called_once()
        actual_model, value, runtime = verify.call_args.args
        self.assertIs(actual_model, model)
        self.assertIs(value.description, architecture.ARCH)
        self.assertIs(runtime.descriptor, architecture.head_layout_descriptor)
        self.assertIs(runtime.torch, torch)
        self.assertIs(runtime.transformers, transformers)
        self.assertIs(runtime.model_type, classes[0])
        self.assertIs(runtime.attention_type, classes[1])
        self.assertIs(runtime.decoder_type, classes[2])


if __name__ == "__main__":
    unittest.main()
