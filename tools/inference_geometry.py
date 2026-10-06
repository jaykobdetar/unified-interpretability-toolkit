"""Pure architecture value and legacy geometry records; no pins, I/O or engines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Architecture:
    # Preserve the existing open configuration/manifest/record boundaries.
    description: Mapping[str, Any]
    config: Mapping[str, Any]
    manifest: Mapping[str, Any]
    capture_sites: Mapping[str, str]
    width: int
    layers: int
    query_heads: int
    kv_heads: int
    head_dim: int
    vocab_size: int


def describe(
    config: Mapping[str, Any], capture_sites: Mapping[str, str]
) -> dict[str, Any]:
    CAPTURE_SITES = capture_sites
    if config.get("model_type") != "llama" or config.get("architectures") != [
        "LlamaForCausalLM"
    ]:
        raise ValueError("No verified attention layout for this architecture")
    keys = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
    )
    if any(type(config.get(k)) is not int or config[k] <= 0 for k in keys):
        raise ValueError("Invalid architecture dimensions")
    width, heads, kv = (
        config[k] for k in ("hidden_size", "num_attention_heads", "num_key_value_heads")
    )
    dim = config.get("head_dim")
    if dim is None:
        if width % heads:
            raise ValueError("Head dimension cannot be inferred exactly")
        dim = width // heads
    if type(dim) is not int or dim <= 0 or heads % kv:
        raise ValueError("Invalid GQA grouping or head dimension")
    if (
        config.get("attention_bias", False)
        or config.get("mlp_bias", False)
        or config.get("tie_word_embeddings") is not True
    ):
        raise ValueError("Only the verified bias-free, tied Llama layout is supported")
    return {
        "model_type": "llama",
        "width": width,
        "layers": config["num_hidden_layers"],
        "query_heads": heads,
        "kv_heads": kv,
        "head_dim": dim,
        "queries_per_kv": heads // kv,
        "vocab_size": config["vocab_size"],
        "intermediate_size": config["intermediate_size"],
        "capture_sites": dict(CAPTURE_SITES),
        "layout": "llama-eager-head-major-v1",
        "head_ablation": "o_proj columns",
        "query_intervention": "q_proj rows",
    }


def shapes(arch: Mapping[str, Any]) -> dict[str, list[int]]:
    w, d, h, kv, inner = (
        arch[k]
        for k in ("width", "head_dim", "query_heads", "kv_heads", "intermediate_size")
    )
    result = {"model.embed_tokens.weight": [arch["vocab_size"], w]}
    for layer in range(arch["layers"]):
        for projection, shape in {
            "self_attn.q_proj": [h * d, w],
            "self_attn.k_proj": [kv * d, w],
            "self_attn.v_proj": [kv * d, w],
            "self_attn.o_proj": [w, h * d],
            "mlp.gate_proj": [inner, w],
            "mlp.up_proj": [inner, w],
            "mlp.down_proj": [w, inner],
        }.items():
            result[f"model.layers.{layer}.{projection}.weight"] = shape
    return result


def head_layout_descriptor(architecture: Architecture) -> dict[str, Any]:
    """Public configuration evidence only; never an execution or fit receipt."""
    MANIFEST, ARCH = architecture.manifest, architecture.description
    WIDTH, HEADS = architecture.width, architecture.query_heads
    KV_HEADS, HEAD_DIM = architecture.kv_heads, architecture.head_dim
    return {
        "schema": "weight-atlas-head-layout-v1",
        "adapter_id": "builtin-llama-eager",
        "adapter_version": 1,
        "source_model": {
            "repo": MANIFEST["repo"],
            "revision": MANIFEST["revision"],
            "weights_sha256": MANIFEST["files"]["model.safetensors"],
            "config_sha256": MANIFEST["files"]["config.json"],
        },
        **ARCH,
        "evidence": "pinned_configuration",
        "runtime_verified": False,
        "inference_support": "requires complete pinned files, supported runtime and resource admission",
        "native_weight_layout": ["output_feature", "input_feature"],
        "projection_mappings": {
            "q_proj": {
                "axis": "rows",
                "head_kind": "query",
                "heads": HEADS,
                "head_dim": HEAD_DIM,
                "shape": [HEADS * HEAD_DIM, WIDTH],
            },
            "k_proj": {
                "axis": "rows",
                "head_kind": "kv",
                "heads": KV_HEADS,
                "head_dim": HEAD_DIM,
                "shape": [KV_HEADS * HEAD_DIM, WIDTH],
            },
            "v_proj": {
                "axis": "rows",
                "head_kind": "kv",
                "heads": KV_HEADS,
                "head_dim": HEAD_DIM,
                "shape": [KV_HEADS * HEAD_DIM, WIDTH],
            },
            "o_proj": {
                "axis": "columns",
                "head_kind": "query_output",
                "heads": HEADS,
                "head_dim": HEAD_DIM,
                "shape": [WIDTH, HEADS * HEAD_DIM],
            },
        },
        "ranges": "zero-based half-open: [head * head_dim, (head + 1) * head_dim)",
        "query_to_kv": "floor(query_head / queries_per_kv)",
        "kv_to_queries": "[kv_head * queries_per_kv, (kv_head + 1) * queries_per_kv)",
        "slice_support": "stored rank-two matrices only; canonical empty slice; no higher-rank inference handoff",
    }
