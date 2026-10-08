"""Bounded config evidence for explicit built-in models; no ML/network imports.

This does not replace the qualified legacy Smol worker or authorize execution.
Layouts are reviewed against Transformers 4.56.2. No default family guessing.
"""

from atlas_host import limits as _limits

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any, TypedDict

from atlas_host.registry import content_digest, validate_manifest

MAX_CONFIG_BYTES = _limits.MODEL_DESCRIPTOR_MAX_CONFIG_BYTES
RUNTIME_VERSION = "4.56.2"
FAMILIES = {
    "llama": "LlamaForCausalLM",
    "qwen2": "Qwen2ForCausalLM",
    "qwen3": "Qwen3ForCausalLM",
}


class ConfigurationEvidence(TypedDict):
    model_type: str
    architecture: str
    width: int
    layers: int
    query_heads: int
    kv_heads: int
    head_dim: int
    queries_per_kv: int
    vocab_size: int
    intermediate_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    qkv_bias: bool
    qk_head_norm: bool
    layout: str
    evidence: str
    runtime_verified: bool
    fit_verified: bool
    inference_ready: bool


class ProjectionMapping(TypedDict):
    axis: str
    heads: int
    head_dim: int
    shape: list[int]


class DescriptorSource(TypedDict):
    repo: str
    revision: str
    config_sha256: str
    weight_files: list[dict[str, Any]]


class PinnedDescriptor(ConfigurationEvidence):
    schema: str
    content_digest: str
    adapter_id: str
    adapter_version: int
    reviewed_transformers: str
    source_model: DescriptorSource
    native_weight_layout: list[str]
    projection_mappings: dict[str, ProjectionMapping]
    parameter_shapes: dict[str, list[int]]
    parameter_count: int
    aliases: dict[str, list[str]]
    qk_semantics: str
    head_ablation: str
    query_intervention: str
    runtime_support: str


def _integer(value: object, low: int, high: int, name: str) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError("Invalid bounded configuration dimension: " + name)
    return value


def describe_config(config: dict[str, Any]) -> ConfigurationEvidence:
    """Coordinates for a narrow dense full-attention topology, never readiness."""
    if type(config) is not dict:
        raise ValueError("Configuration must be an object")
    family = config.get("model_type")
    if (
        type(family) is not str
        or family not in FAMILIES
        or config.get("architectures") != [FAMILIES[family]]
    ):
        raise ValueError("Unsupported explicit built-in architecture")
    limits = {
        "hidden_size": 16384,
        "intermediate_size": 131072,
        "num_hidden_layers": 256,
        "num_attention_heads": 256,
        "num_key_value_heads": 256,
        "vocab_size": 500000,
    }
    dims = {
        key: _integer(config.get(key), 1, bound, key) for key, bound in limits.items()
    }
    width, heads, kv = (
        dims[k] for k in ("hidden_size", "num_attention_heads", "num_key_value_heads")
    )
    dim = config.get("head_dim")
    if dim is None:
        if width % heads or family == "qwen3":
            raise ValueError("Explicit exact head dimension required")
        dim = width // heads
    dim = _integer(dim, 2, 1024, "head_dim")
    if dim % 2 or heads % kv:
        raise ValueError("Unsupported rotary dimension or GQA grouping")
    if type(config.get("tie_word_embeddings")) is not bool:
        raise ValueError("Explicit embedding tie declaration required")
    if config.get("hidden_act") != "silu":
        raise ValueError("Only the reviewed SiLU MLP is described")
    if (
        config.get("auto_map") is not None
        or config.get("quantization_config") is not None
    ):
        raise ValueError("Custom code or quantized topology needs a separate adapter")
    if (
        config.get("rope_scaling") is not None
        or config.get("rope_interleaved", False) is not False
        or type(config.get("partial_rotary_factor", 1)) not in (int, float)
        or config.get("partial_rotary_factor", 1) != 1
        or type(config.get("pretraining_tp", 1)) is not int
        or config.get("pretraining_tp", 1) != 1
    ):
        raise ValueError("Unsupported rotary or tensor-parallel configuration")
    if (
        config.get("use_sliding_window", False) is not False
        or config.get("sliding_window") is not None
        or config.get("layer_types", ["full_attention"] * dims["num_hidden_layers"])
        != ["full_attention"] * dims["num_hidden_layers"]
    ):
        raise ValueError("Only explicit full attention is described")
    for key, default in (
        ("attention_dropout", 0),
        ("rms_norm_eps", None),
        ("rope_theta", None),
    ):
        value: Any = config.get(key, default)
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or (value != 0 if key == "attention_dropout" else value <= 0)
        ):
            raise ValueError("Invalid reviewed numeric configuration: " + key)
    max_positions = _integer(
        config.get("max_position_embeddings"), 1, 1048576, "max_position_embeddings"
    )
    if (
        family in ("llama", "qwen3")
        and config.get("attention_bias", False) is not False
    ):
        raise ValueError("Only bias-free Llama/Qwen3 projections are described")
    if config.get("mlp_bias", False) is not False:
        raise ValueError("Only the reviewed bias-free MLP is described")
    if family == "qwen2" and config.get("attention_bias", True) is not True:
        raise ValueError("Qwen2 built-in Q/K/V projections always have biases")
    return {
        "model_type": family,
        "architecture": FAMILIES[family],
        "width": width,
        "layers": dims["num_hidden_layers"],
        "query_heads": heads,
        "kv_heads": kv,
        "head_dim": dim,
        "queries_per_kv": heads // kv,
        "vocab_size": dims["vocab_size"],
        "intermediate_size": dims["intermediate_size"],
        "max_position_embeddings": max_positions,
        "tie_word_embeddings": config["tie_word_embeddings"],
        "qkv_bias": family == "qwen2",
        "qk_head_norm": family == "qwen3",
        "layout": family + "-eager-head-major-v1",
        "evidence": "configuration_only",
        "runtime_verified": False,
        "fit_verified": False,
        "inference_ready": False,
    }


def parameter_shapes(arch: Mapping[str, Any]) -> dict[str, list[int]]:
    """Stored canonical tensors, including vectors; tied lm_head is an alias."""
    width, dim, heads, kv, inner = (
        arch[k]
        for k in ("width", "head_dim", "query_heads", "kv_heads", "intermediate_size")
    )
    result = {
        "model.embed_tokens.weight": [arch["vocab_size"], width],
        "model.norm.weight": [width],
    }
    if not arch["tie_word_embeddings"]:
        result["lm_head.weight"] = [arch["vocab_size"], width]
    for layer in range(arch["layers"]):
        prefix = f"model.layers.{layer}."
        for projection, shape in {
            "self_attn.q_proj": [heads * dim, width],
            "self_attn.k_proj": [kv * dim, width],
            "self_attn.v_proj": [kv * dim, width],
            "self_attn.o_proj": [width, heads * dim],
            "mlp.gate_proj": [inner, width],
            "mlp.up_proj": [inner, width],
            "mlp.down_proj": [width, inner],
        }.items():
            result[prefix + projection + ".weight"] = shape
        for norm in ("input_layernorm", "post_attention_layernorm"):
            result[prefix + norm + ".weight"] = [width]
        if arch["qkv_bias"]:
            for projection, size in (
                ("q_proj", heads * dim),
                ("k_proj", kv * dim),
                ("v_proj", kv * dim),
            ):
                result[prefix + "self_attn." + projection + ".bias"] = [size]
        if arch["qk_head_norm"]:
            for norm in ("q_norm", "k_norm"):
                result[prefix + "self_attn." + norm + ".weight"] = [dim]
    return result


def projection_mappings(arch: Mapping[str, Any]) -> dict[str, ProjectionMapping]:
    width, dim, heads, kv = (
        arch[k] for k in ("width", "head_dim", "query_heads", "kv_heads")
    )
    return {
        name: {
            "axis": "columns" if name == "o_proj" else "rows",
            "heads": count,
            "head_dim": dim,
            "shape": [width, count * dim] if name == "o_proj" else [count * dim, width],
        }
        for name, count in (
            ("q_proj", heads),
            ("k_proj", kv),
            ("v_proj", kv),
            ("o_proj", heads),
        )
    }


def pinned_descriptor(raw: bytes, manifest: dict[str, Any]) -> PinnedDescriptor:
    """Hash-bind small owner-receipt config data. Weight payloads are not read."""
    manifest = validate_manifest(manifest)
    if type(raw) is not bytes or not 1 <= len(raw) <= MAX_CONFIG_BYTES:
        raise ValueError("Configuration exceeds byte bound")
    files = {file["name"]: file for file in manifest["files"]}
    file = files.get("config.json")
    if (
        file is None
        or len(raw) != file["bytes"]
        or hashlib.sha256(raw).hexdigest() != file["sha256"]
    ):
        raise ValueError("Configuration differs from pinned receipt")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate configuration key")
            result[key] = value
        return result

    try:
        config = json.loads(
            raw,
            object_pairs_hook=unique,
            parse_constant=lambda _: (_ for _ in ()).throw(
                ValueError("Nonfinite configuration")
            ),
        )
        arch = describe_config(config)
    except (UnicodeError, RecursionError) as error:
        raise ValueError("Invalid bounded configuration") from error
    shapes = parameter_shapes(arch)
    return {
        "schema": "weight-atlas-model-descriptor-v1",
        **arch,
        "content_digest": content_digest(manifest),
        "adapter_id": "builtin-" + arch["model_type"] + "-eager",
        "adapter_version": 1,
        "reviewed_transformers": RUNTIME_VERSION,
        "source_model": {
            "repo": manifest["repository"],
            "revision": manifest["revision"],
            "config_sha256": file["sha256"],
            "weight_files": [
                f for f in manifest["files"] if f["name"].endswith(".safetensors")
            ],
        },
        "native_weight_layout": ["output_feature", "input_feature"],
        "projection_mappings": projection_mappings(arch),
        "parameter_shapes": shapes,
        "parameter_count": sum(math.prod(shape) for shape in shapes.values()),
        "aliases": (
            {"model.embed_tokens.weight": ["lm_head.weight"]}
            if arch["tie_word_embeddings"]
            else {}
        ),
        "qk_semantics": (
            "per-head RMSNorm after projection, before rotary"
            if arch["qk_head_norm"]
            else "projection then rotary"
        ),
        "head_ablation": "o_proj columns",
        "query_intervention": "q_proj rows; preserve declared bias",
        "runtime_support": "requires verified complete dense files, exact built-in runtime, measured fit and owner admission",
    }
