"""Closed worker execution registry; admission and engine loading stay at the caller."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias

Request = dict[str, object]
Record = Callable[[dict[str, object]], None]


class Generate(Protocol):
    def __call__(
        self,
        torch: Any,
        tokenizer: Any,
        model: Any,
        prompt: object,
        limit: object,
        layer: object,
        record: Record = ...,
        capture_scores: Callable[[Any], None] | None = None,
        activation_site: object = "block",
        observation: object = None,
    ) -> None: ...


class Compare(Protocol):
    def __call__(
        self,
        torch: Any,
        tokenizer: Any,
        model: Any,
        request: Request,
        record: Record = ...,
    ) -> None: ...


class PromptPair(Protocol):
    def __call__(
        self,
        torch: Any,
        tokenizer: Any,
        model: Any,
        request: Request,
        preview: object,
        capture_kinds: Mapping[str, str],
        record: Record,
    ) -> None: ...


class Sweep(Protocol):
    def __call__(
        self,
        torch: Any,
        tokenizer: Any,
        model: Any,
        request: Request,
        plan: object,
        deadline: float | None,
        generate: Generate,
        record: Record,
        *,
        cpu_deadline: int,
    ) -> None: ...


@dataclass(frozen=True)
class Context:
    # Engine and tensor values are opaque foreign objects, never imported here.
    torch: Any
    tokenizer: Any
    model: Any
    generate: Generate
    compare: Compare
    prompt_pair: PromptPair
    sweep: Sweep
    record: Record
    capture_sites: Mapping[str, str]
    verified_parameters: Callable[[Any], object] | None
    pair_preview: object
    sweep_plan: object
    sweep_deadline: float | None
    cpu_allowance: int


class Kind(Enum):
    PREVIEW = "prompt_pair_preview"
    SWEEP = "sweep"
    PROMPT_PAIR = "prompt_pair"
    COMPARISON = "comparison"
    GENERATION = "generation"


PreparedRequest: TypeAlias = tuple[
    dict[str, Any] | None, dict[str, Any], dict[str, Any] | None, int | None
]


@dataclass(frozen=True)
class Preparation:
    sweep: Callable[[], PreparedRequest]
    pair: Callable[[], tuple[dict[str, Any], None, int | None]]
    generation: Callable[[], tuple[dict[str, Any], dict[str, Any] | None, int]]


@dataclass(frozen=True)
class Experiment:
    kind: Kind
    execute: Callable[[Context, Request], None]
    prepare_request: Callable[[Preparation], PreparedRequest] | None = None


def run_preview(context: Context, request: Request) -> None:
    context.record(
        {
            "type": "preview_done",
            "preview": context.pair_preview,
            "reason": "token_preview",
        }
    )


def run_sweep(context: Context, request: Request) -> None:
    context.sweep(
        context.torch,
        context.tokenizer,
        context.model,
        request,
        context.sweep_plan,
        context.sweep_deadline,
        context.generate,
        context.record,
        cpu_deadline=context.cpu_allowance,
    )


def run_prompt_pair(context: Context, request: Request) -> None:
    verify = context.verified_parameters
    assert verify is not None
    verify(context.model)
    context.prompt_pair(
        context.torch,
        context.tokenizer,
        context.model,
        request,
        context.pair_preview,
        context.capture_sites,
        context.record,
    )


def run_comparison(context: Context, request: Request) -> None:
    context.compare(context.torch, context.tokenizer, context.model, request)


def run_generation(context: Context, request: Request) -> None:
    context.generate(
        context.torch,
        context.tokenizer,
        context.model,
        request["prompt"],
        request["max_new_tokens"],
        request["layer"],
        activation_site=request.get("activation_site", "block"),
        observation=request.get("observation"),
    )


def prepare_sweep(context: Preparation) -> PreparedRequest:
    return context.sweep()


def prepare_pair(context: Preparation) -> PreparedRequest:
    return (None, *context.pair())


def prepare_generation(context: Preparation) -> PreparedRequest:
    return (None, *context.generation())


REGISTRY: Mapping[Kind, Experiment] = MappingProxyType(
    {
        Kind.PREVIEW: Experiment(Kind.PREVIEW, run_preview, prepare_pair),
        Kind.SWEEP: Experiment(Kind.SWEEP, run_sweep, prepare_sweep),
        Kind.PROMPT_PAIR: Experiment(Kind.PROMPT_PAIR, run_prompt_pair, prepare_pair),
        Kind.COMPARISON: Experiment(
            Kind.COMPARISON, run_comparison, prepare_generation
        ),
        Kind.GENERATION: Experiment(
            Kind.GENERATION, run_generation, prepare_generation
        ),
    }
)


def select(request: Request) -> Experiment:
    if request.get("mode") == Kind.SWEEP.value:
        return REGISTRY[Kind.SWEEP]
    elif request.get("mode") == Kind.PROMPT_PAIR.value:
        return REGISTRY[Kind.PROMPT_PAIR]
    elif request.get("mode") == Kind.PREVIEW.value:
        return REGISTRY[Kind.PREVIEW]
    elif "edits" in request:
        return REGISTRY[Kind.COMPARISON]
    else:
        return REGISTRY[Kind.GENERATION]


def execute(request: Request, context: Context) -> None:
    select(request).execute(context, request)


def for_coordinator(request: Request, mode: object) -> Experiment | None:
    """Use the same registrations with the coordinator's existing closed intake."""
    if mode == Kind.SWEEP.value:
        return REGISTRY[Kind.SWEEP]
    elif mode == Kind.PREVIEW.value:
        return REGISTRY[Kind.PREVIEW]
    elif mode == Kind.PROMPT_PAIR.value:
        return REGISTRY[Kind.PROMPT_PAIR]
    elif mode == Kind.GENERATION.value:
        return REGISTRY[Kind.COMPARISON if "edits" in request else Kind.GENERATION]
    return None
