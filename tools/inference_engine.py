"""Engine operations with explicit architecture/runtime bindings; no ML imports."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inference_geometry import Architecture


@dataclass(frozen=True)
class AttentionRuntime:
    transformers: Any
    torch: Any
    model_type: type[Any]
    attention_type: type[Any]
    decoder_type: type[Any]
    descriptor: Callable[[], dict[str, Any]]


@dataclass(frozen=True)
class LoaderRuntime:
    torch: Any
    model_factory: Any
    tokenizer_factory: Any
    verified_parameters: Callable[[Any], object]
    verify_attention_layout: Callable[[Any], dict[str, Any]]


def verify_attention_layout(
    model: Any, architecture: Architecture, runtime: AttentionRuntime
) -> dict[str, Any]:
    # This adapter is source-reviewed for this installed implementation only.
    # New versions/architectures require a new review, never a guessed layout.
    transformers, torch = runtime.transformers, runtime.torch
    LlamaForCausalLM, LlamaAttention = runtime.model_type, runtime.attention_type
    LlamaDecoderLayer = runtime.decoder_type
    head_layout_descriptor = runtime.descriptor
    WIDTH, LAYERS = architecture.width, architecture.layers
    HEADS, KV_HEADS = architecture.query_heads, architecture.kv_heads
    HEAD_DIM = architecture.head_dim

    if (
        transformers.__version__ != "4.56.2"
        or type(model) is not LlamaForCausalLM
        or model.training
    ):
        raise ValueError("Unverified built-in Llama implementation or training mode")
    if (
        model.config._attn_implementation != "eager"
        or len(model.model.layers) != LAYERS
    ):
        raise ValueError("Unverified attention implementation or layer count")
    for block in model.model.layers:
        attn = block.self_attn
        if (
            type(block) is not LlamaDecoderLayer
            or type(attn) is not LlamaAttention
            or attn.head_dim != HEAD_DIM
            or attn.num_key_value_groups != HEADS // KV_HEADS
        ):
            raise ValueError("Attention head layout differs from verified GQA mapping")
        for key, out_features, in_features in [
            ("q_proj", HEADS * HEAD_DIM, WIDTH),
            ("k_proj", KV_HEADS * HEAD_DIM, WIDTH),
            ("v_proj", KV_HEADS * HEAD_DIM, WIDTH),
            ("o_proj", WIDTH, HEADS * HEAD_DIM),
        ]:
            linear = getattr(attn, key)
            if (
                type(linear) is not torch.nn.Linear
                or linear.bias is not None
                or linear.out_features != out_features
                or linear.in_features != in_features
            ):
                raise ValueError(
                    "Projection is not the verified output-by-input linear layout"
                )
    return {
        **head_layout_descriptor(),
        "evidence": "loaded_builtin_layout",
        "runtime_verified": True,
        "runtime": {
            "transformers": transformers.__version__,
            "attention_backend": "eager",
            "device": "cpu",
            "dtype": "float32",
        },
    }


def load_engine(directory: Path, runtime: LoaderRuntime) -> tuple[Any, Any, Any]:
    torch = runtime.torch
    LlamaForCausalLM = runtime.model_factory
    PreTrainedTokenizerFast = runtime.tokenizer_factory

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    # Explicit built-in implementation, no AutoModel, remote Python, or pickle.
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(directory / "tokenizer.json")
    )
    model = LlamaForCausalLM.from_pretrained(
        str(directory),
        local_files_only=True,
        use_safetensors=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    verified_parameters = runtime.verified_parameters
    verify_attention_layout = runtime.verify_attention_layout

    verified_parameters(model)
    model._atlas_verified_layout = verify_attention_layout(model)
    return torch, tokenizer, model
