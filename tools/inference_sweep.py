"""One bounded fixed-context sweep: explicit subset, deterministic controls, undo."""

from contextlib import contextmanager
import hashlib
import json
import math
import time
from inference_edits import (
    SOURCE_MODEL,
    SHAPES,
    apply_edits,
    validate_edits,
    verified_parameters,
)
from inference_architecture import (
    ARCH,
    WIDTH,
    LAYERS,
    HEADS,
    HEAD_DIM,
    VOCAB,
    CAPTURE_SITES,
)

MAX_CELLS = 65536
MAX_TARGETS = 2
MAX_CASES = 5
MAX_PROMPTS = 2
MAX_RECORDS = 10
WALL_SECONDS = 120
CPU_SECONDS = 90
CONTROL_VERSION = "weight-atlas-sweep-control-v2"
MAX_TRACE = 32


def schema():
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


def _int(value, low, high):
    return type(value) is int and low <= value <= high


def _canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _target(value):
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


def _rows(target):
    """Native feature indices, columns for output heads, rows for query interventions."""
    return (
        set(range(target["head"] * HEAD_DIM, (target["head"] + 1) * HEAD_DIM))
        if target["kind"] in ("head", "query_head")
        else {h * HEAD_DIM + target["offset"] for h in target["heads"]}
    )


def _space(target):
    return target["layer"], "output" if target["kind"] == "head" else "query"


def _edits(target, operation, scale):
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


def _choice(pool, seed, target, index):
    if not pool:
        raise ValueError("No disjoint matched control exists for this explicit subset")
    # Canonical JSON list, first SHA256 word interpreted unsigned big-endian.
    # Reject the incomplete modulo bucket; at most 128 draws, never an open loop.
    ceiling = 2**32 - (2**32 % len(pool))
    for counter in range(128):
        payload = [CONTROL_VERSION, seed, target, index, counter]
        word = int.from_bytes(
            hashlib.sha256(_canonical(payload).encode()).digest()[:4], "big"
        )
        if word < ceiling:
            return pool[word % len(pool)]
    raise ValueError("Deterministic control draw limit reached")


def build_plan(data, require_digest=True):
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
    used = {}
    for target in targets:
        used.setdefault(_space(target), set()).update(_rows(target))
    cases = [
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
    plan = {
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


def selected_indices(edits):
    """Deduplicate overlapping native ranges before snapshot allocation."""
    groups = {}
    count = 0
    for edit in edits:
        rows, cols = edit["shape"]
        name = edit["tensor"]
        indices = groups.setdefault(name, set())
        if edit["kind"] == "element":
            positions = iter([edit["row"] * cols + edit["col"]])
        elif edit["kind"] == "rows":
            positions = range(edit["start"] * cols, edit["end"] * cols)
        else:
            positions = (
                row * cols + col
                for row in range(rows)
                for col in range(edit["start"], edit["end"])
            )
        for position in positions:
            if position not in indices:
                indices.add(position)
                count += 1
                if count > MAX_CELLS:
                    raise ValueError("Undo selected-cell cap exceeded")
    return {name: sorted(indices) for name, indices in groups.items()}


class RestorationFailure(RuntimeError):
    pass


@contextmanager
def temporary_edits(torch, model, edits, source_model):
    clean = validate_edits(edits, source_model)
    parameters = verified_parameters(model)
    groups = selected_indices(clean)
    # verified_parameters accepts only the exact tied embedding/head object and
    # rejects other storage aliases. The canonical embedding snapshot restores
    # its head alias too; alias names are not valid edit request names.
    snapshots = []
    with torch.no_grad():
        for name, indices in groups.items():
            flat = parameters[name].reshape(-1)
            index = torch.tensor(indices, dtype=torch.long)
            snapshots.append((flat, index, flat.index_select(0, index).clone()))
    try:
        apply_edits(torch, model, clean, source_model)
        count, changed, squared = 0, 0, 0.0
        with torch.no_grad():
            for flat, index, original in snapshots:
                current = flat.index_select(0, index)
                count += len(index)
                changed += int((current != original).sum())
                squared += float((current.double() - original.double()).square().sum())
        yield {
            "selected_cells": count,
            "changed_cells": changed,
            "parameter_delta_l2": math.sqrt(squared),
        }
    finally:
        try:
            with torch.no_grad():
                for flat, index, original in snapshots:
                    flat.index_copy_(0, index, original)
                for flat, index, original in snapshots:
                    if not torch.equal(
                        flat.index_select(0, index).view(torch.int32),
                        original.view(torch.int32),
                    ):
                        raise RestorationFailure(
                            "Original parameter bit patterns were not restored"
                        )
        except Exception as exc:
            raise RestorationFailure("Undo failed; dispose of this worker") from exc


def record_ids(plan):
    return [
        f"{case['id']}/prompt-{p+1}"
        for case in plan["cases"]
        for p in range(plan["prompt_count"])
    ]


def coverage(plan, completed, current=None):
    ids = record_ids(plan)
    if (
        not _int(completed, 0, len(ids))
        or current is not None
        and current not in ids[completed : completed + 1]
    ):
        raise ValueError("Invalid partial sweep coverage")
    done = set(ids[:completed])
    finished, unfinished = [], []
    for index, target in enumerate(plan["targets"]):
        pair_ids = [
            f"{role}-{index+1}/prompt-{p+1}"
            for role in ("target", "matched_control")
            for p in range(plan["prompt_count"])
        ]
        (finished if all(i in done for i in pair_ids) else unfinished).append(target)
    return {
        "planned_ids": ids,
        "completed_ids": ids[:completed],
        "unrun_ids": ids[completed:],
        "interrupted_id": current,
        "complete": completed == len(ids),
        "finished_targets": finished,
        "unfinished_targets": unfinished,
        "unfinished_heads": [t["head"] for t in unfinished if t["kind"] == "head"],
    }


def metrics(torch, baseline, edited):
    a, b = baseline.double(), edited.double()
    delta = b - a
    selected = int(torch.argmax(baseline))
    return {
        "logit_delta_rms": float(delta.square().mean().sqrt()),
        "logit_delta_max_abs": float(delta.abs().max()),
        "softmax_total_variation": float(
            (torch.softmax(a, dim=-1) - torch.softmax(b, dim=-1)).abs().sum() / 2
        ),
        "baseline_argmax_id": selected,
        "baseline_argmax_logit_delta": float(edited[selected])
        - float(baseline[selected]),
        "edited_argmax_id": int(torch.argmax(edited)),
        "context": "matched fixed original prompt",
        "semantics": "prompt-set sensitivity; no inferred causal purpose or general head importance",
    }


def validate_step(step, plan):
    index = step.get("index")
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


def run(
    torch,
    tokenizer,
    model,
    request,
    plan,
    deadline,
    generate,
    record,
    clock=time.monotonic,
    cpu_deadline=None,
    cpu_clock=time.process_time,
):
    """One model, one total budget; restore before completion, never retry or resume."""
    references = {}
    completed = 0
    total_ms = 0.0

    def exhausted():
        return deadline - clock() < 10 or (
            cpu_deadline is not None and cpu_deadline - cpu_clock() < 5
        )

    def partial():
        record(
            {
                "type": "sweep_done",
                "status": "time_limit",
                "coverage": coverage(plan, completed),
                "reason": "remaining total wall or CPU budget below cleanup margin",
                "compute_total_ms": total_ms,
            }
        )

    def probe(prompt):
        events, scores = [], []
        generate(
            torch,
            tokenizer,
            model,
            prompt,
            1,
            plan["capture_layer"],
            events.append,
            scores.append,
            activation_site=plan["activation_site"],
        )
        steps = [event for event in events if event["type"] == "step"]
        if len(steps) != 1 or len(scores) != 1:
            raise ValueError("Sweep must produce exactly one fixed-context probe")
        return steps[0], scores[0]

    for case in plan["cases"]:
        for prompt_index, prompt in enumerate(request["prompts"]):
            if exhausted():
                partial()
                return
            record_id = record_ids(plan)[completed]
            record(
                {
                    "type": "prefill",
                    "sweep_current": record_id,
                    "comparison_phase": record_id + " baseline",
                }
            )
            baseline, a = probe(prompt)
            if prompt_index in references:
                if not torch.equal(
                    a.view(torch.int32), references[prompt_index].view(torch.int32)
                ):
                    raise RestorationFailure(
                        "Repeated original-model baseline changed; dispose of worker"
                    )
            else:
                references[prompt_index] = a.clone()
            if exhausted():
                partial()
                return
            with temporary_edits(torch, model, case["edits"], SOURCE_MODEL) as changed:
                record({"type": "prefill", "comparison_phase": record_id + " edited"})
                edited, b = probe(prompt)
                values = metrics(torch, a, b)
                ids = sorted(
                    {
                        entry["id"]
                        for step in (baseline, edited)
                        for entry in step["top_logits"]
                    }
                )
                candidates = [
                    {
                        "id": token,
                        "piece": tokenizer.decode([token], skip_special_tokens=False),
                        "baseline_logit": float(a[token]),
                        "edited_logit": float(b[token]),
                        "delta": float(b[token]) - float(a[token]),
                    }
                    for token in ids
                ]
            # Only a verified restore can produce a completed record.
            compute_ms = baseline["compute_ms"] + edited["compute_ms"]
            total_ms += compute_ms
            step = {
                key: edited[key]
                for key in (
                    "activation",
                    "layer",
                    "activation_site",
                    "activation_kind",
                    "position",
                    "input_token_id",
                )
            }
            step.update(
                type="step",
                index=completed,
                mode="sweep",
                phase="fixed_context_probe",
                compute_ms=compute_ms,
                compute_total_ms=total_ms,
                sweep={
                    "record_id": record_id,
                    "case_id": case["id"],
                    "role": case["role"],
                    "prompt_index": prompt_index,
                    **changed,
                    "restoration_verified": True,
                    "metrics": values,
                    "candidates": candidates,
                },
            )
            validate_step(step, plan)
            record(step)
            completed += 1
            del a, b
    record(
        {
            "type": "sweep_done",
            "status": "complete",
            "coverage": coverage(plan, completed),
            "reason": "accepted_plan_complete",
            "compute_total_ms": total_ms,
        }
    )
