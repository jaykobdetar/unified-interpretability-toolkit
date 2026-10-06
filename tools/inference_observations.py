"""Closed, bounded observational request/trace contracts; no numerical imports."""

from dataclasses import replace
import math
from typing import Any

from inference_architecture import architecture
from inference_geometry import Architecture
from inference_observation_contract import ObservationBindings
import inference_observation_contract as contract
from inference_architecture import (
    ARCH,
    WIDTH,
    LAYERS,
    HEADS,
    KV_HEADS,
    HEAD_DIM,
    VOCAB,
    CAPTURE_SITES,
)

ATTENTION_SEMANTICS = "selected query head, last consumed query; post-causal-mask softmax probabilities, eval dropout zero"
LENS_SEMANTICS = "selected post-block residual through final RMSNorm and tied head; diagnostic readout, not early-exit inference or causal effect"


def _bindings(value: Architecture | None = None) -> ObservationBindings:
    """Bind compatibility dimensions/semantics and keep helper lookup late."""

    def integer(value: object, low: int, high: int) -> bool:
        return _integer(value, low, high)

    def finite(value: object) -> bool:
        return _finite(value)

    binding = ObservationBindings(
        architecture=(
            replace(
                architecture(),
                description=ARCH,
                width=WIDTH,
                layers=LAYERS,
                query_heads=HEADS,
                kv_heads=KV_HEADS,
                head_dim=HEAD_DIM,
                vocab_size=VOCAB,
                capture_sites=CAPTURE_SITES,
            )
            if value is None
            else value
        ),
        attention_semantics=ATTENTION_SEMANTICS,
        lens_semantics=LENS_SEMANTICS,
        integer=integer,
        finite=finite,
    )
    return binding


def schema() -> dict[str, Any]:
    return contract.schema(_bindings())


def validate_observation(value: Any, site: str) -> dict[str, Any]:
    return contract.validate_observation(value, site, _bindings())


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_record(
    step: dict[str, Any], selected: Any = None, layer: Any = None
) -> None:
    return contract.validate_record(step, selected, layer, _bindings())


def lens_record(
    torch: Any,
    tokenizer: Any,
    residual: Any,
    model: Any,
    final_logits: Any,
    layer: int,
    position: int,
) -> dict[str, Any]:
    return contract.lens_record(
        torch, tokenizer, residual, model, final_logits, layer, position, _bindings()
    )
