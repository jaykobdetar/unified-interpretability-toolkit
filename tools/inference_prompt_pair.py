"""Bounded two-prompt token preview and selected-position activation comparison."""

from dataclasses import replace
from typing import Any
from atlas_host.inference_architecture import architecture
from atlas_host.inference_geometry import Architecture
from atlas_host import inference_edits as edit_facade
from atlas_host import inference_edit_contract as edit_contract
from atlas_host.inference_pair_contract import PairBindings
from atlas_host import inference_pair_contract as contract

import hashlib
import json
import math
from atlas_host.inference_architecture import (
    ARCH,
    WIDTH,
    LAYERS,
    HEADS,
    KV_HEADS,
    HEAD_DIM,
    VOCAB,
    CAPTURE_SITES,
)
from atlas_host.inference_edits import SOURCE_MODEL, validate_edits

MODES = ("prompt_pair_preview", "prompt_pair")
SHA256_HEX_LENGTH = 64


def _bindings(value: Architecture | None = None) -> PairBindings:
    """Retain legacy values and late helper lookup at compatibility entrypoints."""
    explicit = value is not None

    def validate(values: Any, source: Any) -> list[dict[str, Any]]:
        if not explicit:
            return validate_edits(values, source)
        return edit_contract.validate_edits(
            values, source, edit_facade._bindings(binding.architecture)
        )

    def digest(prompts: Any, ids: Any) -> str:
        if not explicit:
            return digest_for(prompts, ids)
        return contract.digest_for(prompts, ids, bindings=binding)

    def preview(value: Any) -> None:
        if not explicit:
            validate_preview(value)
            return
        contract.validate_preview(value, bindings=binding)

    def delta(a: Any, b: Any) -> tuple[list[Any], dict[str, Any]]:
        if not explicit:
            return difference(a, b)
        return contract.difference(a, b, bindings=binding)

    binding = PairBindings(
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
        source_model=SOURCE_MODEL,
        sha256_hex_length=SHA256_HEX_LENGTH,
        validate_edits=validate,
        digest_for=digest,
        validate_preview=preview,
        difference=delta,
    )
    return binding


def validate_request(data: dict[str, Any]) -> dict[str, Any]:
    return contract.validate_request(data, bindings=_bindings())


def digest_for(prompts: Any, ids: Any) -> str:
    return contract.digest_for(prompts, ids, bindings=_bindings())


def token_preview(tokenizer: Any, prompts: Any) -> dict[str, Any]:
    return contract.token_preview(tokenizer, prompts, bindings=_bindings())


def validate_preview(value: Any) -> None:
    return contract.validate_preview(value, bindings=_bindings())


def validate_positions(request: dict[str, Any], preview: dict[str, Any]) -> None:
    return contract.validate_positions(request, preview, bindings=_bindings())


def difference(a: Any, b: Any) -> tuple[list[Any], dict[str, Any]]:
    return contract.difference(a, b, bindings=_bindings())


def validate_step(step: dict[str, Any], request: dict[str, Any]) -> None:
    return contract.validate_step(step, request, bindings=_bindings())


def run(torch, tokenizer, model, request, preview, capture_kinds, record):
    """Exactly two uncached prefills and one selected observational hook."""
    import time

    ids = [[t["id"] for t in tokens] for tokens in preview["tokens"]]
    if ids != [
        tokenizer.encode(p, add_special_tokens=True) for p in request["prompts"]
    ]:
        raise ValueError("Tokenizer wrapper disagrees with reviewed preview")
    layer, site = request["layer"], request["activation_site"]
    block = model.model.layers[layer]
    target = (
        block
        if site == "block"
        else block.self_attn if site == "attention" else block.mlp
    )
    captured = []
    started = time.perf_counter()
    for side, key in enumerate(("a", "b")):
        record(
            {
                "type": "prefill",
                "mode": "prompt_pair",
                "comparison_phase": "prompt " + key.upper(),
            }
        )
        positions = sorted({pair[key] for pair in request["positions"]})
        selected = {}

        def capture(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            for position in positions:
                selected[position] = hidden[0, position].detach().float().tolist()

        hook = target.register_forward_hook(capture)
        try:
            with torch.inference_mode():
                output = model.model(torch.tensor([ids[side]]), use_cache=False)
                del output
        finally:
            hook.remove()
        if set(selected) != set(positions):
            raise ValueError("Selected prompt activations missing")
        captured.append(selected)
    elapsed = (time.perf_counter() - started) * 1000
    for index, positions in enumerate(request["positions"]):
        a, b = (captured[i][positions[key]] for i, key in enumerate(("a", "b")))
        delta, metrics = difference(a, b)
        pair = {
            "metrics": metrics,
            "token_equal": ids[0][positions["a"]] == ids[1][positions["b"]],
            "prefix_equal": ids[0][: positions["a"] + 1]
            == ids[1][: positions["b"] + 1],
        }
        for i, key in enumerate(("a", "b")):
            position = positions[key]
            pair[key] = {
                "position": position,
                "token_id": ids[i][position],
                "token_piece": preview["tokens"][i][position]["piece"],
                "activation": captured[i][position],
            }
        step = {
            "type": "step",
            "index": index,
            "mode": "prompt_pair",
            "layer": layer,
            "activation_site": site,
            "activation_kind": capture_kinds[site]
            + "; B minus A, difference computed in binary64",
            "activation": delta,
            "prompt_pair": pair,
            "compute_ms": elapsed if index == 0 else 0.0,
            "compute_total_ms": elapsed,
        }
        validate_step(step, request)
        record(step)
    record(
        {
            "type": "done",
            "mode": "prompt_pair",
            "record_count": len(request["positions"]),
            "reason": "prompt_pair_complete",
            "compute_total_ms": elapsed,
            "coverage": "exactly the selected layer/site/position pairs; two uncached prefills, no generation",
        }
    )
