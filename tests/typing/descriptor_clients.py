"""Strict clients for real descriptor records and shared legacy imports."""

from typing import Any
import inference_model_descriptor as legacy
from atlas_host import inference_model_descriptor as canonical


def configuration(config: dict[str, Any]) -> tuple[int, str, list[int]]:
    arch = canonical.describe_config(config)
    heads: int = arch["query_heads"]
    layout: str = arch["layout"]
    shape: list[int] = canonical.parameter_shapes(arch)["model.norm.weight"]
    return heads, layout, shape


def pinned(raw: bytes, manifest: dict[str, Any]) -> tuple[int, str, int]:
    record = legacy.pinned_descriptor(raw, manifest)
    count: int = record["parameter_count"]
    digest: str = record["source_model"]["config_sha256"]
    mapping = record["projection_mappings"]["o_proj"]
    return count, digest, mapping["head_dim"]
