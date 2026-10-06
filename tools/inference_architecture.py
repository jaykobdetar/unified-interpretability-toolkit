"""Pinned configuration and explicitly supported built-in attention layout; no ML imports."""

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

from atlas_host.inference_geometry import Architecture
from atlas_host import inference_geometry as geometry

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "docs/models/smollm2-135m.json").read_text())
CAPTURE_SITES = {
    "block": "decoder block output after attention and MLP residual additions, before final RMSNorm",
    "attention": "attention output after o_proj, before attention residual addition",
    "mlp": "MLP output after down_proj, before MLP residual addition",
}


def describe(config: Mapping[str, Any]) -> dict[str, Any]:
    return geometry.describe(config, CAPTURE_SITES)


raw = (ROOT / "docs/models/smollm2-135m-config.json").read_bytes()
if hashlib.sha256(raw).hexdigest() != MANIFEST["files"]["config.json"]:
    raise ValueError("Bundled configuration differs from the pinned disk config")
CONFIG = json.loads(raw)
ARCH = describe(CONFIG)
WIDTH, LAYERS, HEADS, KV_HEADS, HEAD_DIM, VOCAB = (
    ARCH[k]
    for k in ("width", "layers", "query_heads", "kv_heads", "head_dim", "vocab_size")
)


def architecture() -> Architecture:
    """Bind current compatibility values once for an explicit internal call."""
    return Architecture(
        description=ARCH,
        config=CONFIG,
        manifest=MANIFEST,
        capture_sites=CAPTURE_SITES,
        width=WIDTH,
        layers=LAYERS,
        query_heads=HEADS,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        vocab_size=VOCAB,
    )


def shapes(arch: Mapping[str, Any] = ARCH) -> dict[str, list[int]]:
    return geometry.shapes(arch)


def head_layout_descriptor() -> dict[str, Any]:
    return geometry.head_layout_descriptor(architecture())


def verify_attention_layout(model: Any) -> dict[str, Any]:
    import transformers
    import torch
    from transformers.models.llama.modeling_llama import (
        LlamaForCausalLM,
        LlamaAttention,
        LlamaDecoderLayer,
    )
    from atlas_host.inference_engine import (
        AttentionRuntime,
        verify_attention_layout as verify,
    )

    return verify(
        model,
        architecture(),
        AttentionRuntime(
            transformers=transformers,
            torch=torch,
            model_type=LlamaForCausalLM,
            attention_type=LlamaAttention,
            decoder_type=LlamaDecoderLayer,
            descriptor=head_layout_descriptor,
        ),
    )


def bind_viewer_head_layout(model_info, trusted_binding=None):
    """Project host-validated receipt correspondence; no validation I/O or admission.

    trusted_binding is an internal host result, NEVER an HTTP/visitor field.
    The host must have checked the installed receipt's current fingerprints and
    correspondence to this renderer before constructing its four-field binding.
    This adapter checks correspondence, not the authenticity of an arbitrary dict.
    It never infers a binding from a path, revision label or copied descriptor.
    """
    result = dict(model_info)
    # Never forward an upstream/stale/runtime-owned annotation without rebinding.
    result.pop("head_layout", None)
    result.pop("head_layout_binding", None)
    fields = {"source_identity", "model_identity", "weights_sha256", "config_sha256"}
    if (
        type(trusted_binding) is not dict
        or set(trusted_binding) != fields
        or any(
            type(v) is not str
            or len(v) != 64
            or any(c not in "0123456789abcdef" for c in v)
            for v in trusted_binding.values()
        )
    ):
        return result
    descriptor = head_layout_descriptor()
    source = descriptor["source_model"]
    if (
        trusted_binding["source_identity"] != model_info.get("source_identity")
        or trusted_binding["model_identity"] != model_info.get("model_identity")
        or trusted_binding["weights_sha256"] != source["weights_sha256"]
        or trusted_binding["config_sha256"] != source["config_sha256"]
        or model_info.get("revision") != source["revision"]
        or "comparison_identity" in model_info
        or "coordinate_space" in model_info
    ):
        return result
    result["head_layout"] = descriptor
    result["head_layout_binding"] = dict(trusted_binding)
    return result
