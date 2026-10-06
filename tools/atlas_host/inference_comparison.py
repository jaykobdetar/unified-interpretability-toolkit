"""Comparison experiment with injected worker callbacks and open legacy records."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from atlas_host.inference_experiments import Generate, Record, Request

# Trace envelopes retain arbitrary legacy fields; no new schema or coercion.
Event = dict[str, Any]


@dataclass(frozen=True)
class ComparisonBindings:
    generate: Generate
    paired_step: Callable[
        [int, list[Event], list[Event], list[Any], list[Any], Any], Event
    ]
    validate_edits: Callable[[object, object], object]
    verified_parameters: Callable[[Any], object]
    apply_edits: Callable[[Any, Any, object, object], None]


def paired_step(
    index: int,
    baseline: list[Event],
    edited: list[Event],
    baseline_scores: list[Any],
    edited_scores: list[Any],
    tokenizer: Any,
) -> Event:
    """Union scores have both values; prefix identity is based on consumed IDs."""
    left: Any = baseline[index] if index < len(baseline) else None
    right: Any = edited[index] if index < len(edited) else None
    left_ids = [step["token_id"] for step in baseline[:index]]
    right_ids = [step["token_id"] for step in edited[:index]]
    alignment = (
        "branch_ended"
        if left is None or right is None
        else "matched_prefix" if left_ids == right_ids else "different_prefix"
    )
    candidates = sorted(
        {entry["id"] for step in (left, right) if step for entry in step["top_logits"]}
    )
    scores = []
    for token in candidates:
        a = float(baseline_scores[index][token]) if left else None
        b = float(edited_scores[index][token]) if right else None
        scores.append(
            {
                "id": token,
                "piece": tokenizer.decode([token], skip_special_tokens=False),
                "baseline_logit": a,
                "edited_logit": b,
                "delta": b - a if a is not None and b is not None else None,
            }
        )

    def summary(step: Event | None, prefix: list[int]) -> Event | None:
        return (
            {
                key: step[key]
                for key in (
                    "token_id",
                    "token_piece",
                    "generated_text",
                    "eos",
                    "compute_ms",
                    "compute_total_ms",
                )
            }
            | {"generated_ids": prefix + [step["token_id"]]}
            if step
            else None
        )

    return {
        **(right or left),
        "type": "step",
        "index": index,
        "baseline": summary(left, left_ids),
        "edited": summary(right, right_ids),
        "alignment": alignment,
        "score_kind": "raw FP32 logits",
        "activation_branch": "edited" if right else "baseline",
        "candidates": scores,
    }


def run(
    torch: Any,
    tokenizer: Any,
    model: Any,
    request: Request,
    record: Record,
    bindings: ComparisonBindings,
) -> None:
    # This model belongs exclusively to one disposable worker. No later request
    # can reuse edited parameters. Cancellation/failure destroys this process.
    generate, paired_step = bindings.generate, bindings.paired_step
    apply_edits = bindings.apply_edits
    validate_edits = bindings.validate_edits
    verified_parameters = bindings.verified_parameters
    baseline: list[Event]
    baseline_scores: list[Any]
    edited: list[Event]
    edited_scores: list[Any]

    edits = validate_edits(request["edits"], request["source_model"])
    verified_parameters(model)
    baseline, baseline_scores, edited, edited_scores = [], [], [], []

    def collect(target: list[Event]) -> Record:
        def accept(event: Event) -> None:
            if event["type"] == "step":
                target.append(event)
            elif event["type"] == "prefill":
                record(event)

        return accept

    record({"type": "prefill", "comparison_phase": "baseline", "edits": edits})
    generate(
        torch,
        tokenizer,
        model,
        request["prompt"],
        request["max_new_tokens"],
        request["layer"],
        collect(baseline),
        baseline_scores.append,
        activation_site=request.get("activation_site", "block"),
        observation=request.get("observation"),
    )
    apply_edits(torch, model, edits, request["source_model"])
    record({"type": "prefill", "comparison_phase": "edited"})

    # Fresh generate invocation starts from prompt with no baseline KV cache.
    def accept_edited(event: Event) -> None:
        collect(edited)(event)
        if event["type"] == "step":
            record(
                paired_step(
                    event["index"],
                    baseline,
                    edited,
                    baseline_scores,
                    edited_scores,
                    tokenizer,
                )
            )

    generate(
        torch,
        tokenizer,
        model,
        request["prompt"],
        request["max_new_tokens"],
        request["layer"],
        accept_edited,
        edited_scores.append,
        activation_site=request.get("activation_site", "block"),
        observation=request.get("observation"),
    )
    for index in range(len(edited), len(baseline)):
        record(
            paired_step(
                index, baseline, edited, baseline_scores, edited_scores, tokenizer
            )
        )

    def result(steps: list[Event]) -> Event:
        return {
            "generated_ids": [step["token_id"] for step in steps],
            "generated_text": steps[-1]["generated_text"],
            "reason": "eos" if steps[-1]["eos"] else "token_limit",
        }

    record(
        {
            "type": "done",
            "generated_tokens": max(len(baseline), len(edited)),
            "comparison_phase": "complete",
            "baseline": result(baseline),
            "edited": result(edited),
            "reason": "comparison_complete",
            "compute_total_ms": baseline[-1]["compute_total_ms"]
            + edited[-1]["compute_total_ms"],
        }
    )
