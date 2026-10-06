"""Edit contracts and operations with explicit architecture/data/callback bindings."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from typing import Any

from atlas_host.inference_geometry import Architecture


@dataclass(frozen=True)
class EditBindings:
    architecture: Architecture
    source_model: Mapping[str, Any]
    shapes: Mapping[str, list[int]]
    aliases: Mapping[str, list[str]]
    max_edits: int
    validate_edits: Callable[[Any, Any], list[dict[str, Any]]]
    verified_parameters: Callable[[Any], dict[str, Any]]


def schema(bindings: EditBindings) -> dict[str, Any]:
    SOURCE_MODEL = bindings.source_model
    MAX_EDITS = bindings.max_edits
    SHAPES = bindings.shapes
    ALIASES = bindings.aliases
    ARCH = bindings.architecture.description
    return {
        "source_model": SOURCE_MODEL,
        "max_edits": MAX_EDITS,
        "ranges": "zero-based half-open [start,end)",
        "composition": "request order",
        "scores": "raw FP32 logits, edited minus baseline",
        "architecture": ARCH,
        "tensors": [
            {"name": name, "shape": shape, "aliases": ALIASES.get(name, [])}
            for name, shape in SHAPES.items()
        ],
    }


def validate_edits(
    edits: Any, source_model: Any, bindings: EditBindings
) -> list[dict[str, Any]]:
    SOURCE_MODEL = bindings.source_model
    MAX_EDITS = bindings.max_edits
    SHAPES = bindings.shapes
    if type(source_model) is not dict or source_model != SOURCE_MODEL:
        raise ValueError(
            "Edits require the complete pinned SmolLM2-135M source model; Qwen coordinates are unsupported"
        )
    if type(edits) is not list or len(edits) > MAX_EDITS:
        raise ValueError("Edits must be a list of at most eight operations")
    clean: list[dict[str, Any]] = []
    for edit in edits:
        if type(edit) is not dict:
            raise ValueError("Each edit must be an object")
        name, shape, kind, operation = (
            edit.get(k) for k in ("tensor", "shape", "kind", "operation")
        )
        if type(name) is not str or name not in SHAPES:
            raise ValueError(
                "Tensor is not an allowlisted stored two-dimensional SmolLM2 parameter"
            )
        if (
            type(shape) is not list
            or any(type(n) is not int for n in shape)
            or shape != SHAPES[name]
        ):
            raise ValueError("Native tensor shape does not match the pinned parameter")
        if kind not in ("rows", "columns", "element") or operation not in (
            "zero",
            "scale",
        ):
            raise ValueError("Choose rows, columns, or element and zero or scale")
        keys = {"tensor", "shape", "kind", "operation"} | (
            {"row", "col"} if kind == "element" else {"start", "end"}
        )
        if operation == "scale":
            keys.add("scale")
            factor: Any = edit.get("scale")
            if (
                type(factor) not in (int, float)
                or abs(factor) > 100
                or not math.isfinite(factor)
            ):
                raise ValueError(
                    "Scale must be finite numeric (not boolean), between -100 and 100"
                )
        if set(edit) != keys:
            raise ValueError("Unexpected or missing edit fields")
        if kind == "element":
            if any(
                type(edit[k]) is not int or not 0 <= edit[k] < shape[i]
                for i, k in enumerate(("row", "col"))
            ):
                raise ValueError(
                    "Element coordinates must be native in-bounds integers"
                )
        else:
            start, end = edit["start"], edit["end"]
            bound = shape[0 if kind == "rows" else 1]
            if (
                type(start) is not int
                or type(end) is not int
                or not 0 <= start < end <= bound
            ):
                raise ValueError(
                    "Range must be nonempty native [start,end), end exclusive"
                )
        clean.append({**edit, "shape": list(shape)})
    return clean


def verified_parameters(model: Any, bindings: EditBindings) -> dict[str, Any]:
    """Check the loaded architecture, every stored matrix, and all storage aliases."""
    SHAPES = bindings.shapes
    ALIASES = bindings.aliases
    CONFIG = bindings.architecture.config
    expected = {
        key: CONFIG[key]
        for key in (
            "model_type",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "vocab_size",
            "tie_word_embeddings",
        )
    }
    if any(
        getattr(model.config, key, None) != value for key, value in expected.items()
    ):
        raise ValueError(
            "Loaded model architecture differs from the pinned edit mapping"
        )
    parameters: dict[str, Any] = dict(model.named_parameters(remove_duplicate=False))
    matrices = {name for name, value in parameters.items() if value.ndim == 2}
    if matrices != set(SHAPES) | {"lm_head.weight"}:
        raise ValueError(
            "Loaded model matrix names differ from the pinned edit mapping"
        )
    storages: dict[int, str] = {}
    for name, shape in SHAPES.items():
        parameter = parameters[name]
        if (
            list(parameter.shape) != shape
            or str(parameter.dtype) != "torch.float32"
            or parameter.device.type != "cpu"
            or not parameter.is_contiguous()
            or parameter.storage_offset() != 0
        ):
            raise ValueError(
                "Loaded parameter shape/layout differs from native viewer coordinates"
            )
        pointer = parameter.untyped_storage().data_ptr()
        if pointer in storages:
            raise ValueError("Unexpected parameter storage alias")
        storages[pointer] = name
    # No matrix may silently alias an unrelated vector or another matrix.
    for name, parameter in parameters.items():
        canonical = storages.get(parameter.untyped_storage().data_ptr())
        if canonical is not None and name not in (
            canonical,
            *ALIASES.get(canonical, []),
        ):
            raise ValueError("Unexpected parameter storage alias")
    embed, head = parameters["model.embed_tokens.weight"], parameters["lm_head.weight"]
    if embed is not head:
        raise ValueError("Expected embedding/output-head parameter tie is missing")
    return parameters


def apply_edits(
    torch: Any, model: Any, edits: object, source_model: Any, bindings: EditBindings
) -> list[dict[str, Any]]:
    validate_edits = bindings.validate_edits
    verified_parameters = bindings.verified_parameters
    edits = validate_edits(edits, source_model)
    parameters = verified_parameters(model)
    with torch.no_grad():
        for edit in edits:
            value = parameters[edit["tensor"]]
            if edit["kind"] == "rows":
                selected = value[edit["start"] : edit["end"], :]
            elif edit["kind"] == "columns":
                selected = value[:, edit["start"] : edit["end"]]
            else:
                selected = value[
                    edit["row"] : edit["row"] + 1, edit["col"] : edit["col"] + 1
                ]
            if edit["operation"] == "zero":
                selected.zero_()
            else:
                selected.mul_(edit["scale"])
            if not bool(torch.isfinite(selected).all()):
                raise ValueError("Edit produced nonfinite weights")
    return edits


def validate_pair(step: dict[str, Any], bindings: EditBindings) -> None:
    """Bound paired worker output without trusting sizes or nonfinite JSON values."""
    VOCAB = bindings.architecture.vocab_size
    if step.get("alignment") not in (
        "matched_prefix",
        "different_prefix",
        "branch_ended",
    ):
        raise ValueError("Invalid comparison alignment")
    if step.get("score_kind") != "raw FP32 logits" or step.get(
        "activation_branch"
    ) not in ("baseline", "edited"):
        raise ValueError("Invalid comparison score/activation kind")
    sides: list[Any] = [step.get(key) for key in ("baseline", "edited")]
    for side in sides:
        if side is None:
            continue
        if type(side) is not dict or type(side.get("generated_ids")) is not list:
            raise ValueError("Invalid branch trace")
        ids = side["generated_ids"]
        if (
            len(ids) != step["index"] + 1
            or len(ids) > 32
            or any(type(i) is not int or not 0 <= i < VOCAB for i in ids)
        ):
            raise ValueError("Invalid branch token IDs")
        if (
            side.get("token_id") != ids[-1]
            or type(side.get("generated_text")) is not str
            or len(side["generated_text"]) > 8192
        ):
            raise ValueError("Invalid branch text/token")
    if all(side is None for side in sides):
        raise ValueError("Both branches are absent")
    expected = (
        "branch_ended"
        if any(side is None for side in sides)
        else (
            "matched_prefix"
            if sides[0]["generated_ids"][:-1] == sides[1]["generated_ids"][:-1]
            else "different_prefix"
        )
    )
    if step["alignment"] != expected:
        raise ValueError("Comparison prefix alignment disagrees with tokens")
    candidates = step.get("candidates")
    if type(candidates) is not list or not 1 <= len(candidates) <= 10:
        raise ValueError("Invalid comparison candidate count")
    seen = set()
    for entry in candidates:
        if (
            type(entry) is not dict
            or type(entry.get("id")) is not int
            or not 0 <= entry["id"] < VOCAB
            or entry["id"] in seen
        ):
            raise ValueError("Invalid comparison candidate ID")
        seen.add(entry["id"])
        for side, key in zip(sides, ("baseline_logit", "edited_logit")):
            value: Any = entry.get(key)
            if (side is None and value is not None) or (
                side is not None
                and (type(value) not in (int, float) or not math.isfinite(value))
            ):
                raise ValueError("Invalid comparison logit")
        a, b = entry.get("baseline_logit"), entry.get("edited_logit")
        expected_delta = None if a is None or b is None else b - a
        if entry.get("delta") != expected_delta:
            raise ValueError("Invalid comparison delta")
