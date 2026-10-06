"""Pinned SmolLM2 native-coordinate edit contract; no ML imports or file writes."""

from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any

from inference_architecture import architecture
from atlas_host.inference_geometry import Architecture
from inference_edit_contract import EditBindings
import inference_edit_contract as contract

from inference_architecture import ARCH, CONFIG, MANIFEST, VOCAB, shapes

SOURCE_MODEL = {
    "repo": MANIFEST["repo"],
    "revision": MANIFEST["revision"],
    "weights_sha256": MANIFEST["files"]["model.safetensors"],
}
MAX_EDITS = 8
# Built-in Llama Linear weights are [output feature, input feature].
SHAPES = shapes()
# lm_head is not a stored/viewer tensor in this checkpoint. Editing the embedding
# intentionally also edits the tied output head; requesting the alias is rejected.
ALIASES = {"model.embed_tokens.weight": ["lm_head.weight"]}


def _bindings(value: Architecture | None = None) -> EditBindings:
    """Capture compatibility data; preserve late callback lookup at invocation."""
    explicit = value is not None

    def validate(values: Any, source: Any) -> list[dict[str, Any]]:
        if not explicit:
            return validate_edits(values, source)
        return contract.validate_edits(values, source, binding)

    def parameters(model: Any) -> dict[str, Any]:
        if not explicit:
            return verified_parameters(model)
        return contract.verified_parameters(model, binding)

    binding = EditBindings(
        architecture=(
            replace(
                architecture(),
                description=ARCH,
                config=CONFIG,
                manifest=MANIFEST,
                vocab_size=VOCAB,
            )
            if value is None
            else value
        ),
        source_model=SOURCE_MODEL,
        shapes=SHAPES,
        aliases=ALIASES,
        max_edits=MAX_EDITS,
        validate_edits=validate,
        verified_parameters=parameters,
    )
    return binding


def schema() -> dict[str, Any]:
    return contract.schema(_bindings())


def validate_edits(edits: Any, source_model: Any) -> list[dict[str, Any]]:
    return contract.validate_edits(edits, source_model, _bindings())


def verified_parameters(model: Any) -> dict[str, Any]:
    return contract.verified_parameters(model, _bindings())


def apply_edits(
    torch: Any, model: Any, edits: Any, source_model: Any
) -> list[dict[str, Any]]:
    return contract.apply_edits(torch, model, edits, source_model, _bindings())


def validate_pair(step: dict[str, Any]) -> None:
    return contract.validate_pair(step, _bindings())
