"""Pure sweep geometry, planning and output bounds with passed bindings."""

from __future__ import annotations
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any
from atlas_host.inference_geometry import Architecture


@dataclass(frozen=True)
class SweepBindings:
    architecture: Architecture
    source_model: Mapping[str, Any]
    shapes: Mapping[str, list[int]]
    max_cells: int
    max_targets: int
    max_cases: int
    max_prompts: int
    max_records: int
    wall_seconds: int
    cpu_seconds: int
    control_version: str
    max_trace: int
    integer: Callable[[Any, int, int], bool]
    canonical: Callable[[Any], str]
    target: Callable[[Any], dict[str, Any]]
    rows: Callable[[dict[str, Any]], set[int]]
    space: Callable[[dict[str, Any]], tuple[Any, str]]
    edits: Callable[[dict[str, Any], Any, Any], list[dict[str, Any]]]
    choice: Callable[[list[dict[str, Any]], Any, dict[str, Any], int], dict[str, Any]]
    schema: Callable[[], dict[str, Any]]
    validate_edits: Callable[[Any, Any], list[dict[str, Any]]]
    record_ids: Callable[[dict[str, Any]], list[str]]


def schema(*, bindings: SweepBindings) -> dict[str, Any]:
    ARCH = bindings.architecture.description
    HEADS = bindings.architecture.query_heads
    MAX_CELLS = bindings.max_cells
    MAX_TARGETS = bindings.max_targets
    MAX_CASES = bindings.max_cases
    MAX_PROMPTS = bindings.max_prompts
    MAX_RECORDS = bindings.max_records
    WALL_SECONDS = bindings.wall_seconds
    CPU_SECONDS = bindings.cpu_seconds
    CONTROL_VERSION = bindings.control_version
    MAX_TRACE = bindings.max_trace
    return {
        "targets": max(MAX_TARGETS, HEADS),
        "interventions_including_controls": max(MAX_CASES, 1 + 2 * HEADS),
        "prompts": MAX_PROMPTS,
        "subset_limits": {
            "targets": MAX_TARGETS,
            "records": MAX_RECORDS,
            "prefills": 20,
        },
        "probes_per_prompt": 1,
        "records": min(MAX_TRACE, max(MAX_RECORDS, 1 + 2 * HEADS)),
        "prefills": 2 * min(MAX_TRACE, max(MAX_RECORDS, 1 + 2 * HEADS)),
        "edits": 8,
        "selected_cells": MAX_CELLS,
        "wall_seconds": WALL_SECONDS,
        "worker_cpu_seconds": CPU_SECONDS,
        "control_version": CONTROL_VERSION,
        "full_layer_records": 1 + 2 * HEADS,
        "trace_cap": MAX_TRACE,
        "architecture": ARCH,
        "scope": "explicit subset or all output heads in one layer; fixed context; one budget, no resume",
    }


def _target(value: Any, *, bindings: SweepBindings) -> dict[str, Any]:
    LAYERS = bindings.architecture.layers
    HEADS = bindings.architecture.query_heads
    HEAD_DIM = bindings.architecture.head_dim
    _int = bindings.integer
    if type(value) is not dict or not _int(value.get("layer"), 0, LAYERS - 1):
        raise ValueError(f"Choose a native target layer 0–{LAYERS-1}")
    if value.get("kind") in ("head", "query_head"):
        if set(value) != {"kind", "layer", "head"} or not _int(
            value["head"], 0, HEADS - 1
        ):
            raise ValueError(f"Choose exactly one head 0–{HEADS-1}")
        return dict(value)
    if value.get("kind") == "offset":
        heads = value.get("heads")
        if (
            set(value) != {"kind", "layer", "heads", "offset"}
            or not _int(value["offset"], 0, HEAD_DIM - 1)
            or type(heads) is not list
            or not 1 <= len(heads) <= 8
            or any(not _int(h, 0, HEADS - 1) for h in heads)
            or len(set(heads)) != len(heads)
        ):
            raise ValueError(
                "An offset selects a native query-row offset in 1–8 unique query heads"
            )
        return {**value, "heads": sorted(heads)}
    raise ValueError("Choose output head, query_head, or query-row offset targets")


def _rows(target: dict[str, Any], *, bindings: SweepBindings) -> set[int]:
    """Native feature indices, columns for output heads, rows for query interventions."""
    HEAD_DIM = bindings.architecture.head_dim
    return (
        set(range(target["head"] * HEAD_DIM, (target["head"] + 1) * HEAD_DIM))
        if target["kind"] in ("head", "query_head")
        else {h * HEAD_DIM + target["offset"] for h in target["heads"]}
    )


def _edits(
    target: dict[str, Any], operation: Any, scale: Any, *, bindings: SweepBindings
) -> list[dict[str, Any]]:
    HEAD_DIM = bindings.architecture.head_dim
    SHAPES = bindings.shapes
    _rows = bindings.rows
    whole = target["kind"] in ("head", "query_head")
    intervals = (
        [(target["head"] * HEAD_DIM, (target["head"] + 1) * HEAD_DIM)]
        if whole
        else [(r, r + 1) for r in sorted(_rows(target))]
    )
    name = f"model.layers.{target['layer']}.self_attn.{'o_proj' if target['kind']=='head' else 'q_proj'}.weight"
    return [
        {
            "tensor": name,
            "shape": list(SHAPES[name]),
            "kind": "columns" if target["kind"] == "head" else "rows",
            "operation": operation,
            "start": start,
            "end": end,
            **({"scale": scale} if operation == "scale" else {}),
        }
        for start, end in intervals
    ]


def build_plan(
    data: Any, require_digest: bool = True, *, bindings: SweepBindings
) -> dict[str, Any]:
    ARCH = bindings.architecture.description
    WIDTH = bindings.architecture.width
    LAYERS = bindings.architecture.layers
    HEADS = bindings.architecture.query_heads
    HEAD_DIM = bindings.architecture.head_dim
    CAPTURE_SITES = bindings.architecture.capture_sites
    SOURCE_MODEL = bindings.source_model
    validate_edits = bindings.validate_edits
    MAX_CELLS = bindings.max_cells
    MAX_TARGETS = bindings.max_targets
    MAX_PROMPTS = bindings.max_prompts
    CONTROL_VERSION = bindings.control_version
    MAX_TRACE = bindings.max_trace
    _int = bindings.integer
    _canonical = bindings.canonical
    _target = bindings.target
    _rows = bindings.rows
    _space = bindings.space
    _edits = bindings.edits
    _choice = bindings.choice
    schema = bindings.schema
    scale: Any
    if type(data) is not dict:
        raise ValueError("Sweep must be an object")
    fields = {
        "mode",
        "source_model",
        "prompts",
        "targets",
        "operation",
        "seed",
        "capture_layer",
        "activation_site",
    }
    if require_digest:
        fields.add("plan_digest")
    if data.get("operation") == "scale":
        fields.add("scale")
    if type(data) is not dict or set(data) != fields or data.get("mode") != "sweep":
        raise ValueError("Sweep requires exactly its closed plan fields")
    validate_edits([], data["source_model"])
    prompts = data["prompts"]
    if (
        type(prompts) is not list
        or not 1 <= len(prompts) <= MAX_PROMPTS
        or any(
            type(p) is not str or not p.strip() or len(p.encode()) > 2048
            for p in prompts
        )
    ):
        raise ValueError("Select 1–2 prompts of at most 2048 UTF-8 bytes each")
    if (
        type(data["targets"]) is not list
        or not 1 <= len(data["targets"]) <= MAX_TARGETS
    ):
        raise ValueError("Select an explicit subset of 1–2 targets")
    full_layer = (
        len(data["targets"]) == 1
        and isinstance(data["targets"][0], dict)
        and data["targets"][0].get("kind") == "layer_heads"
    )
    if full_layer:
        target = data["targets"][0]
        if set(target) != {"kind", "layer"} or not _int(target["layer"], 0, LAYERS - 1):
            raise ValueError("All-head sweep selects exactly one valid layer")
        if data["operation"] != "zero":
            raise ValueError(
                "All-head ablation requires zero output-projection columns"
            )
        targets = [
            {"kind": "head", "layer": target["layer"], "head": h} for h in range(HEADS)
        ]
        if (1 + 2 * len(targets)) * len(prompts) > MAX_TRACE:
            raise ValueError(
                "All-head plan exceeds the existing 32-record cap; choose one prompt or an explicit subset"
            )
    else:
        targets = [_target(t) for t in data["targets"]]
    if len({_canonical(t) for t in targets}) != len(targets):
        raise ValueError("Duplicate sweep target")
    if (
        not _int(data["seed"], 0, 2**32 - 1)
        or not _int(data["capture_layer"], 0, LAYERS - 1)
        or type(data["activation_site"]) is not str
        or data["activation_site"] not in CAPTURE_SITES
    ):
        raise ValueError("Invalid seed or selected capture layer/output")
    operation, scale = data["operation"], data.get("scale")
    if operation not in ("zero", "scale") or (
        operation == "scale"
        and (
            type(scale) not in (int, float)
            or not math.isfinite(scale)
            or abs(scale) > 100
        )
    ):
        raise ValueError("Use zero or a finite numeric scale between -100 and 100")
    used: dict[tuple[Any, str], set[int]] = {}
    for target in targets:
        used.setdefault(_space(target), set()).update(_rows(target))
    cases: list[dict[str, Any]] = [
        {
            "id": "empty",
            "role": "empty_control",
            "edits": [],
            "selected_cells": 0,
            "target": None,
            "control_geometry": None,
        }
    ]
    for i, target in enumerate(targets):
        choices = []
        for value in range(
            HEADS if target["kind"] in ("head", "query_head") else HEAD_DIM
        ):
            control = {
                **target,
                (
                    "head" if target["kind"] in ("head", "query_head") else "offset"
                ): value,
            }
            excluded = _rows(target) if full_layer else used[_space(target)]
            if not (_rows(control) & excluded):
                choices.append(control)
        control = _choice(choices, data["seed"], target, i)
        used[_space(target)].update(_rows(control))
        for role, descriptor in [("target", target), ("matched_control", control)]:
            edits = validate_edits(_edits(descriptor, operation, scale), SOURCE_MODEL)
            cells = len(_rows(descriptor)) * WIDTH
            if cells > MAX_CELLS:
                raise ValueError("Intervention selected-cell cap exceeded")
            cases.append(
                {
                    "id": f"{role}-{i+1}",
                    "role": role,
                    "target": descriptor,
                    "edits": edits,
                    "selected_cells": cells,
                    "control_geometry": (
                        (
                            f"different head-aligned contiguous {HEAD_DIM}-column output block"
                            if target["kind"] == "head"
                            else (
                                f"different head-aligned contiguous {HEAD_DIM}-row query block"
                                if target["kind"] == "query_head"
                                else "different query offset: one complete row in each same named head"
                            )
                        )
                        if role == "matched_control"
                        else None
                    ),
                }
            )
    plan: dict[str, Any] = {
        "version": "weight-atlas-sweep-plan-v2",
        "seed": data["seed"],
        "control_version": CONTROL_VERSION,
        "source_model": SOURCE_MODEL,
        "targets": targets,
        "cases": cases,
        "prompt_count": len(prompts),
        "records": len(cases) * len(prompts),
        "prefills": 2 * len(cases) * len(prompts),
        "capture_layer": data["capture_layer"],
        "activation_site": data["activation_site"],
        "scope": "layer_heads" if full_layer else "subset",
        "architecture": ARCH,
        "intervention_semantics": "head: o_proj columns; query_head/offset: q_proj rows; zero output block removes only that query head contribution, leaves shared K/V intact",
        "control_semantics": (
            "same geometry and operation; disjoint from paired target; comparison intervention, not a null/no-effect assumption; may equal another target"
            if full_layer
            else "same geometry and operation; disjoint within the same projection; comparison intervention, not a null/no-effect assumption"
        ),
        "coverage": (
            f"all {HEADS} output heads in selected layer planned; completion depends on one total budget"
            if full_layer
            else "explicit subset only; not a complete head sweep"
        ),
        "limits": schema(),
    }
    digest = hashlib.sha256(_canonical([plan, prompts]).encode()).hexdigest()
    if require_digest and (
        type(data["plan_digest"]) is not str or data["plan_digest"] != digest
    ):
        raise ValueError("Plan, prompts or seed changed; review a fresh expanded plan")
    plan["digest"] = digest
    if (
        len(json.dumps({**data, "plan_digest": digest}, ensure_ascii=False).encode())
        > 8192
    ):
        raise ValueError("Sweep request exceeds 8 KiB")
    return plan


def validate_step(
    step: dict[str, Any], plan: dict[str, Any], *, bindings: SweepBindings
) -> None:
    WIDTH = bindings.architecture.width
    VOCAB = bindings.architecture.vocab_size
    _int = bindings.integer
    record_ids = bindings.record_ids
    index: Any = step.get("index")
    count = plan["prompt_count"]
    if not _int(index, 0, plan["records"] - 1) or step.get("mode") != "sweep":
        raise ValueError("Invalid sweep sequence")
    if (
        type(step.get("activation")) is not list
        or len(step["activation"]) != WIDTH
        or any(
            type(v) not in (int, float) or not math.isfinite(v)
            for v in step["activation"]
        )
    ):
        raise ValueError("Invalid sweep activation")
    case, prompt = plan["cases"][index // count], index % count
    value = step.get("sweep")
    if type(value) is not dict or set(value) != {
        "record_id",
        "case_id",
        "role",
        "prompt_index",
        "selected_cells",
        "changed_cells",
        "parameter_delta_l2",
        "restoration_verified",
        "metrics",
        "candidates",
    }:
        raise ValueError("Invalid sweep record")
    if (
        value["record_id"] != record_ids(plan)[index]
        or value["case_id"] != case["id"]
        or value["role"] != case["role"]
        or type(value["prompt_index"]) is not int
        or value["prompt_index"] != prompt
        or type(value["selected_cells"]) is not int
        or value["selected_cells"] != case["selected_cells"]
        or not _int(value["changed_cells"], 0, case["selected_cells"])
        or value["restoration_verified"] is not True
        or type(step.get("layer")) is not int
        or step["layer"] != plan["capture_layer"]
        or step.get("activation_site") != plan["activation_site"]
    ):
        raise ValueError("Sweep record differs from its accepted plan")
    values = value["metrics"]
    fields = {
        "logit_delta_rms",
        "logit_delta_max_abs",
        "softmax_total_variation",
        "baseline_argmax_id",
        "baseline_argmax_logit_delta",
        "edited_argmax_id",
        "context",
        "semantics",
    }
    if (
        type(values) is not dict
        or set(values) != fields
        or values["context"] != "matched fixed original prompt"
        or values["semantics"]
        != "prompt-set sensitivity; no inferred causal purpose or general head importance"
    ):
        raise ValueError("Invalid sweep metric contract")
    for key in ("baseline_argmax_id", "edited_argmax_id"):
        if not _int(values[key], 0, VOCAB - 1):
            raise ValueError("Invalid sweep argmax")
    for key in (
        "logit_delta_rms",
        "logit_delta_max_abs",
        "softmax_total_variation",
        "baseline_argmax_logit_delta",
    ):
        if type(values[key]) not in (int, float) or not math.isfinite(values[key]):
            raise ValueError("Nonfinite sweep metric")
    if (
        values["logit_delta_rms"] < 0
        or values["logit_delta_max_abs"] < 0
        or not 0 <= values["softmax_total_variation"] <= 1 + 1e-12
    ):
        raise ValueError("Invalid sweep metric range")
    if (
        type(value["parameter_delta_l2"]) not in (int, float)
        or not math.isfinite(value["parameter_delta_l2"])
        or value["parameter_delta_l2"] < 0
    ):
        raise ValueError("Invalid parameter perturbation norm")
    candidates = value["candidates"]
    if type(candidates) is not list or not 1 <= len(candidates) <= 10:
        raise ValueError("Invalid sweep candidate cap")
    seen = set()
    for entry in candidates:
        if (
            type(entry) is not dict
            or set(entry) != {"id", "piece", "baseline_logit", "edited_logit", "delta"}
            or not _int(entry["id"], 0, VOCAB - 1)
            or entry["id"] in seen
            or type(entry["piece"]) is not str
            or len(entry["piece"]) > 1024
            or any(
                type(entry[k]) not in (int, float) or not math.isfinite(entry[k])
                for k in ("baseline_logit", "edited_logit", "delta")
            )
            or entry["delta"] != entry["edited_logit"] - entry["baseline_logit"]
        ):
            raise ValueError("Invalid sweep candidate score/delta")
        seen.add(entry["id"])
    if case["role"] == "empty_control" and (
        any(
            values[k] != 0
            for k in (
                "logit_delta_rms",
                "logit_delta_max_abs",
                "softmax_total_variation",
                "baseline_argmax_logit_delta",
            )
        )
        or any(e["delta"] != 0 for e in candidates)
    ):
        raise ValueError("Empty sweep control lost exact parity")
