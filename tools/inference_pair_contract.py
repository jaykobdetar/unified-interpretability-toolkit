"""Pure prompt-pair contracts with passed architecture, identities and callbacks."""

from __future__ import annotations
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any
from atlas_host.inference_geometry import Architecture


@dataclass(frozen=True)
class PairBindings:
    architecture: Architecture
    source_model: Mapping[str, Any]
    sha256_hex_length: int
    validate_edits: Callable[[Any, Any], list[dict[str, Any]]]
    digest_for: Callable[[Any, Any], str]
    validate_preview: Callable[[Any], None]
    difference: Callable[[Any, Any], tuple[list[Any], dict[str, Any]]]


def validate_request(data: dict[str, Any], *, bindings: PairBindings) -> dict[str, Any]:
    LAYERS = bindings.architecture.layers
    CAPTURE_SITES = bindings.architecture.capture_sites
    SOURCE_MODEL = bindings.source_model
    validate_edits = bindings.validate_edits
    SHA256_HEX_LENGTH = bindings.sha256_hex_length
    mode = data.get("mode")
    fields = {"mode", "prompts", "source_model"}
    if mode == "prompt_pair":
        fields |= {"layer", "activation_site", "positions", "preview_digest"}
    elif mode != "prompt_pair_preview":
        raise ValueError("Unknown prompt-pair mode")
    if set(data) != fields:
        raise ValueError("Prompt-pair requests require exactly their declared fields")
    prompts = data["prompts"]
    if (
        type(prompts) is not list
        or len(prompts) != 2
        or any(
            type(p) is not str or not p.strip() or len(p.encode("utf-8")) > 2048
            for p in prompts
        )
    ):
        raise ValueError("Enter two nonempty prompts of at most 2048 UTF-8 bytes each")
    validate_edits([], data["source_model"])
    result: dict[str, Any] = {
        "mode": mode,
        "prompts": list(prompts),
        "source_model": dict(SOURCE_MODEL),
    }
    if mode == "prompt_pair":
        if (
            type(data["layer"]) is not int
            or not 0 <= data["layer"] < LAYERS
            or type(data["activation_site"]) is not str
            or data["activation_site"] not in CAPTURE_SITES
        ):
            raise ValueError(f"Select one native layer 0–{LAYERS-1} and capture output")
        positions = data["positions"]
        if type(positions) is not list or not 1 <= len(positions) <= 8:
            raise ValueError("Select 1–8 explicit token position pairs")
        seen = set()
        for pair in positions:
            if (
                type(pair) is not dict
                or set(pair) != {"a", "b"}
                or any(
                    type(pair[k]) is not int or not 0 <= pair[k] < 128
                    for k in ("a", "b")
                )
            ):
                raise ValueError("Positions must be strict token integers 0–127")
            key = pair["a"], pair["b"]
            if key in seen:
                raise ValueError("Duplicate position pair")
            seen.add(key)
        digest = data["preview_digest"]
        if (
            type(digest) is not str
            or len(digest) != SHA256_HEX_LENGTH
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("Review the exact token preview before comparing prompts")
        result.update(
            layer=data["layer"],
            activation_site=data["activation_site"],
            positions=[dict(p) for p in positions],
            preview_digest=digest,
        )
    if len(json.dumps(result, ensure_ascii=False, allow_nan=False).encode()) > 8192:
        raise ValueError("Complete prompt-pair request exceeds 8 KiB")
    return result


def digest_for(prompts: Any, ids: Any, *, bindings: PairBindings) -> str:
    SOURCE_MODEL = bindings.source_model
    value = {"source_model": SOURCE_MODEL, "prompts": prompts, "ids": ids}
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def token_preview(
    tokenizer: Any, prompts: Any, *, bindings: PairBindings
) -> dict[str, Any]:
    digest_for = bindings.digest_for
    validate_preview = bindings.validate_preview
    encodings = [tokenizer.encode(p, add_special_tokens=True).ids for p in prompts]
    if any(not 1 <= len(ids) <= 128 for ids in encodings):
        raise ValueError("Each prompt must encode to 1–128 tokens")
    tokens = [
        [
            {
                "position": i,
                "id": token,
                "piece": tokenizer.decode([token], skip_special_tokens=False),
            }
            for i, token in enumerate(ids)
        ]
        for ids in encodings
    ]
    value = {
        "digest": digest_for(prompts, encodings),
        "tokens": tokens,
        "model_loaded": False,
    }
    validate_preview(value)
    return value


def validate_preview(value: Any, *, bindings: PairBindings) -> None:
    VOCAB = bindings.architecture.vocab_size
    SHA256_HEX_LENGTH = bindings.sha256_hex_length
    if (
        type(value) is not dict
        or set(value) != {"digest", "tokens", "model_loaded"}
        or value["model_loaded"] is not False
    ):
        raise ValueError("Invalid tokenizer preview")
    digest = value["digest"]
    if (
        type(digest) is not str
        or len(digest) != SHA256_HEX_LENGTH
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("Invalid preview digest")
    if len(json.dumps(value, ensure_ascii=False).encode()) > 65536:
        raise ValueError("Token preview exceeds 64 KiB")
    arrays = value["tokens"]
    if type(arrays) is not list or len(arrays) != 2:
        raise ValueError("Invalid preview branches")
    for tokens in arrays:
        if type(tokens) is not list or not 1 <= len(tokens) <= 128:
            raise ValueError("Invalid preview token count")
        for i, token in enumerate(tokens):
            if (
                type(token) is not dict
                or set(token) != {"position", "id", "piece"}
                or type(token["position"]) is not int
                or token["position"] != i
                or type(token["id"]) is not int
                or not 0 <= token["id"] < VOCAB
                or type(token["piece"]) is not str
                or len(token["piece"].encode("utf-8")) > 1024
            ):
                raise ValueError("Invalid preview token")


def validate_positions(
    request: dict[str, Any], preview: dict[str, Any], *, bindings: PairBindings
) -> None:
    if request["preview_digest"] != preview["digest"]:
        raise ValueError("Prompts or tokenization changed; review a fresh preview")
    for pair in request["positions"]:
        if any(
            pair[key] >= len(preview["tokens"][i]) for i, key in enumerate(("a", "b"))
        ):
            raise ValueError("Selected position exceeds its encoded prompt")


def difference(
    a: Any, b: Any, *, bindings: PairBindings
) -> tuple[list[Any], dict[str, Any]]:
    WIDTH = bindings.architecture.width
    if any(
        type(v) is not list
        or len(v) != WIDTH
        or any(type(x) not in (int, float) or not math.isfinite(x) for x in v)
        for v in (a, b)
    ):
        raise ValueError(f"Expected two finite {WIDTH}-value vectors")
    delta = [y - x for x, y in zip(a, b)]
    na, nb, nd = math.hypot(*a), math.hypot(*b), math.hypot(*delta)
    cosine = (
        max(-1.0, min(1.0, math.fsum(x * y for x, y in zip(a, b)) / (na * nb)))
        if na and nb
        else None
    )
    if not all(
        math.isfinite(x)
        for x in delta + [na, nb, nd] + ([] if cosine is None else [cosine])
    ):
        raise ValueError("Nonfinite activation difference/metric")
    return delta, {"a_l2": na, "b_l2": nb, "delta_l2": nd, "cosine": cosine}


def validate_step(
    step: dict[str, Any], request: dict[str, Any], *, bindings: PairBindings
) -> None:
    VOCAB = bindings.architecture.vocab_size
    difference = bindings.difference
    index = step.get("index")
    if (
        type(index) is not int
        or not 0 <= index < len(request["positions"])
        or step.get("mode") != "prompt_pair"
    ):
        raise ValueError("Invalid prompt-pair sequence")
    if (
        type(step.get("layer")) is not int
        or step["layer"] != request["layer"]
        or step.get("activation_site") != request["activation_site"]
    ):
        raise ValueError("Prompt-pair capture differs from request")
    pair = step.get("prompt_pair")
    if type(pair) is not dict or set(pair) != {
        "a",
        "b",
        "token_equal",
        "prefix_equal",
        "metrics",
    }:
        raise ValueError("Invalid prompt-pair output")
    for key in ("a", "b"):
        side = pair[key]
        if (
            type(side) is not dict
            or set(side) != {"position", "token_id", "token_piece", "activation"}
            or type(side["position"]) is not int
            or side["position"] != request["positions"][index][key]
            or type(side["token_id"]) is not int
            or not 0 <= side["token_id"] < VOCAB
            or type(side["token_piece"]) is not str
            or len(side["token_piece"].encode()) > 1024
        ):
            raise ValueError("Invalid selected prompt token")
    if (
        type(pair["token_equal"]) is not bool
        or pair["token_equal"] != (pair["a"]["token_id"] == pair["b"]["token_id"])
        or type(pair["prefix_equal"]) is not bool
    ):
        raise ValueError("Invalid prompt alignment label")
    if pair["prefix_equal"] and (
        not pair["token_equal"] or pair["a"]["position"] != pair["b"]["position"]
    ):
        raise ValueError("Contradictory prefix alignment")
    delta, metrics = difference(pair["a"]["activation"], pair["b"]["activation"])
    actual = step.get("activation")
    reported = pair["metrics"]
    if (
        type(actual) is not list
        or any(type(x) not in (int, float) or not math.isfinite(x) for x in actual)
        or type(reported) is not dict
        or set(reported) != set(metrics)
        or any(
            (v is not None and (type(v) not in (int, float) or not math.isfinite(v)))
            for v in reported.values()
        )
        or actual != delta
        or reported != metrics
    ):
        raise ValueError("Prompt difference/metrics disagree with captured values")
