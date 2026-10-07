"""Strict callers of bounded static policy and its raw digest validator."""

from collections.abc import Mapping, Sequence
from typing import Any, assert_type

from atlas_host.common import digest
from atlas_host.inference_model_descriptor import PinnedDescriptor
from atlas_host.registry import Registry
from atlas_host.static_models import (
    BoundStatic,
    NativeFingerprint,
    NativeShard,
    PreparedStatic,
    StaticBinding,
    StaticCatalog,
    StaticCatalogModel,
    StaticPolicy,
    StaticTensor,
    native_source_identity,
)


def raw_digest(value: object) -> str:
    return assert_type(digest(value), str)


def source_identity(
    root: str,
    shards: Sequence[Mapping[str, object]],
    index: Mapping[str, object] | None,
    index_stat: Mapping[str, object] | None,
) -> str:
    return assert_type(native_source_identity(root, shards, index, index_stat), str)


def source_records(shard: NativeShard, tensor: StaticTensor) -> None:
    saved = assert_type(shard["fingerprint"], NativeFingerprint)
    assert_type(saved["size"], int)
    assert_type(shard["header_sha256"], str)
    assert_type(tensor["shape"], list[int])
    assert_type(tensor["byte_offset"], int)
    if "id" in tensor:
        assert_type(tensor["id"], int)


def registered_policy(
    registry: Registry, identifier: str, model: dict[str, Any], ready: bool
) -> dict[str, Any]:
    policy = StaticPolicy(registry)
    catalog = assert_type(policy.catalog(), StaticCatalog)
    for row in catalog["models"]:
        assert_type(row, StaticCatalogModel)
        assert_type(row["static_view_ready"], bool)
    prepared = assert_type(policy.prepare(identifier), PreparedStatic)
    assert_type(prepared.descriptor, PinnedDescriptor)
    prepared.check()
    binding = assert_type(policy.bind(prepared, model, reader_ready=ready), BoundStatic)
    public = assert_type(binding.public_binding(), StaticBinding)
    assert_type(public["source_identity"], str)
    assert_type(
        policy.catalog(active_binding=binding, reader_ready=ready), StaticCatalog
    )
    return assert_type(
        policy.project_model(binding, model, reader_ready=ready), dict[str, Any]
    )
