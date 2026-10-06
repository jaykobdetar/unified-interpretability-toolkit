"""One bounded fixed-context sweep: explicit subset, deterministic controls, undo."""

from dataclasses import replace
from typing import Any
from inference_architecture import architecture
from atlas_host.inference_geometry import Architecture
import inference_edits as edit_facade
import inference_edit_contract as edit_contract
from inference_sweep_contract import SweepBindings
import inference_sweep_contract as contract

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


def _bindings(value: Architecture | None = None) -> SweepBindings:
    """Retain legacy data/limits and helper lookup without changing runtime work."""
    explicit = value is not None

    def integer(value: Any, low: int, high: int) -> bool:
        return _int(value, low, high)

    def canonical(value: Any) -> str:
        return _canonical(value)

    def target(value: Any) -> dict[str, Any]:
        if not explicit:
            return _target(value)
        return contract._target(value, bindings=binding)

    def rows(value: dict[str, Any]) -> set[int]:
        if not explicit:
            return _rows(value)
        return contract._rows(value, bindings=binding)

    def space(value: dict[str, Any]) -> tuple[Any, str]:
        return _space(value)

    def edits(
        value: dict[str, Any], operation: Any, scale: Any
    ) -> list[dict[str, Any]]:
        if not explicit:
            return _edits(value, operation, scale)
        return contract._edits(value, operation, scale, bindings=binding)

    def choice(
        pool: list[dict[str, Any]], seed: Any, value: dict[str, Any], index: int
    ) -> dict[str, Any]:
        return _choice(pool, seed, value, index)

    def limits() -> dict[str, Any]:
        if not explicit:
            return schema()
        return contract.schema(bindings=binding)

    def validate(values: Any, source: Any) -> list[dict[str, Any]]:
        if not explicit:
            return validate_edits(values, source)
        return edit_contract.validate_edits(
            values, source, edit_facade._bindings(binding.architecture)
        )

    def ids(plan: dict[str, Any]) -> list[str]:
        return record_ids(plan)

    binding = SweepBindings(
        architecture=(
            replace(
                architecture(),
                description=ARCH,
                width=WIDTH,
                layers=LAYERS,
                query_heads=HEADS,
                head_dim=HEAD_DIM,
                vocab_size=VOCAB,
                capture_sites=CAPTURE_SITES,
            )
            if value is None
            else value
        ),
        source_model=SOURCE_MODEL,
        shapes=SHAPES,
        max_cells=MAX_CELLS,
        max_targets=MAX_TARGETS,
        max_cases=MAX_CASES,
        max_prompts=MAX_PROMPTS,
        max_records=MAX_RECORDS,
        wall_seconds=WALL_SECONDS,
        cpu_seconds=CPU_SECONDS,
        control_version=CONTROL_VERSION,
        max_trace=MAX_TRACE,
        integer=integer,
        canonical=canonical,
        target=target,
        rows=rows,
        space=space,
        edits=edits,
        choice=choice,
        schema=limits,
        validate_edits=validate,
        record_ids=ids,
    )
    return binding


def schema() -> dict[str, Any]:
    return contract.schema(bindings=_bindings())


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


def _target(value: Any) -> dict[str, Any]:
    return contract._target(value, bindings=_bindings())


def _rows(target: dict[str, Any]) -> set[int]:
    """Native feature indices, columns for output heads, rows for query interventions."""
    return contract._rows(target, bindings=_bindings())


def _space(target):
    return target["layer"], "output" if target["kind"] == "head" else "query"


def _edits(target: dict[str, Any], operation: Any, scale: Any) -> list[dict[str, Any]]:
    return contract._edits(target, operation, scale, bindings=_bindings())


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


def build_plan(data: Any, require_digest: bool = True) -> dict[str, Any]:
    return contract.build_plan(data, require_digest, bindings=_bindings())


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


def validate_step(step: dict[str, Any], plan: dict[str, Any]) -> None:
    return contract.validate_step(step, plan, bindings=_bindings())


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
