#!/usr/bin/env python3
"""One bounded offline inference session; JSON lines on stdin/stdout, no HTTP."""

from dataclasses import replace
import json
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

# Set before importing numerical libraries, including when imported by reference tests.
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.update(
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    HF_HUB_DISABLE_TELEMETRY="1",
    TOKENIZERS_PARALLELISM="false",
    CUDA_VISIBLE_DEVICES="",
    PYTHONDONTWRITEBYTECODE="1",
)

MAX_PROMPT = 128
MAX_NEW = 32
from atlas_host.inference_architecture import (
    ARCH,
    WIDTH,
    LAYERS,
    HEADS,
    KV_HEADS,
    HEAD_DIM,
    VOCAB,
    CAPTURE_SITES,
    verify_attention_layout,
)
from atlas_host.inference_observations import (
    ATTENTION_SEMANTICS,
    lens_record,
    validate_observation,
    validate_record,
)

from atlas_host.inference_generation import GenerationBindings, run as run_generation
from atlas_host.inference_engine import LoaderRuntime, load_engine as bound_load_engine
from atlas_host.inference_comparison import (
    ComparisonBindings,
    Event,
    paired_step as comparison_step,
    run as run_comparison,
)
from atlas_host.inference_experiments import Context, Record, Request, execute
from atlas_host.inference_architecture import architecture
from atlas_host.inference_services import InferenceContracts


def emit(record):
    print(json.dumps(record, allow_nan=False, separators=(",", ":")), flush=True)


def load_engine(directory: Path) -> tuple[Any, Any, Any]:
    import torch
    from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

    def verify_parameters(model: Any) -> object:
        from atlas_host.inference_edits import verified_parameters

        return verified_parameters(model)

    def verify_layout(model: Any) -> dict[str, Any]:
        return verify_attention_layout(model)

    return bound_load_engine(
        directory,
        LoaderRuntime(
            torch=torch,
            model_factory=LlamaForCausalLM,
            tokenizer_factory=PreTrainedTokenizerFast,
            verified_parameters=verify_parameters,
            verify_attention_layout=verify_layout,
        ),
    )


def _contracts() -> InferenceContracts:
    """Capture the current compatibility defaults once, without model work."""
    from atlas_host import inference_edits as edits
    from atlas_host import inference_observations as observations
    import inference_prompt_pair as pair
    import inference_sweep as sweep

    value = replace(
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
    return InferenceContracts(
        value,
        edits._bindings(value),
        observations._bindings(value),
        pair._bindings(value),
        sweep._bindings(value),
    )


def generate(
    torch: Any,
    tokenizer: Any,
    model: Any,
    prompt: Any,
    limit: Any,
    layer: Any,
    record: Record = emit,
    capture_scores: Any = None,
    activation_site: str = "block",
    observation: Any = None,
) -> None:
    contracts = _contracts()
    value = contracts.architecture
    return run_generation(
        torch,
        tokenizer,
        model,
        prompt,
        limit,
        layer,
        record,
        GenerationBindings(
            max_prompt=MAX_PROMPT,
            max_new=MAX_NEW,
            width=value.width,
            layers=value.layers,
            heads=value.query_heads,
            kv_heads=value.kv_heads,
            head_dim=value.head_dim,
            vocab=value.vocab_size,
            capture_sites=value.capture_sites,
            attention_semantics=ATTENTION_SEMANTICS,
            validate_observation=contracts.validate_observation,
            validate_record=contracts.validate_record,
            lens_record=contracts.lens_record,
        ),
        capture_scores=capture_scores,
        activation_site=activation_site,
        observation=observation,
    )


def paired_step(
    index: int,
    baseline: list[Event],
    edited: list[Event],
    baseline_scores: list[Any],
    edited_scores: list[Any],
    tokenizer: Any,
) -> Event:
    return comparison_step(
        index, baseline, edited, baseline_scores, edited_scores, tokenizer
    )


def compare(
    torch: Any, tokenizer: Any, model: Any, request: Request, record: Record = emit
) -> None:
    from atlas_host.inference_edits import (
        apply_edits,
        validate_edits,
        verified_parameters,
    )

    return run_comparison(
        torch,
        tokenizer,
        model,
        request,
        record,
        ComparisonBindings(
            generate=generate,
            paired_step=paired_step,
            validate_edits=validate_edits,
            verified_parameters=verified_parameters,
            apply_edits=apply_edits,
        ),
    )


def configure_worker_limits(cpu_seconds=90):
    if type(cpu_seconds) is not int or not 1 <= cpu_seconds <= 90:
        raise ValueError("Invalid remaining worker CPU allowance")
    for kind, limit in (
        (resource.RLIMIT_AS, 3 * 1024**3),
        (resource.RLIMIT_CPU, cpu_seconds),
    ):
        inherited = resource.getrlimit(kind)
        limit = min([limit] + [v for v in inherited if v != resource.RLIM_INFINITY])
        resource.setrlimit(kind, (limit, limit))


def main() -> None:
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    os.nice(10)
    if len(sys.argv) not in (2, 4):
        raise ValueError("Invalid worker budget arguments")
    cpu_allowance = int(sys.argv[3]) if len(sys.argv) == 4 else 90
    configure_worker_limits(cpu_allowance)
    # Entire request is bounded by coordinator and again here.
    request = json.loads(sys.stdin.buffer.readline(8193))
    from live_inference import verify_model, SweepAdmissionBudget

    directory = Path(sys.argv[1])
    import inference_prompt_pair as prompt_pair
    import inference_sweep as sweep

    contracts = _contracts()
    value = contracts.architecture
    sweep_plan, sweep_deadline = None, None
    if request.get("mode") == "sweep":
        sweep_plan = contracts.build_plan(request)
        import math

        if len(sys.argv) != 4:
            raise ValueError("Sweep requires the coordinator total-job deadline")
        sweep_deadline = float(sys.argv[2])
        if (
            not math.isfinite(sweep_deadline)
            or not 0 < sweep_deadline - time.monotonic() <= sweep.WALL_SECONDS
        ):
            raise ValueError("Invalid or exhausted total sweep deadline")
        budget = SweepAdmissionBudget(sweep_deadline, cpu_allowance)
        with budget.verification():
            verify_model(directory, check=budget.check)
        from tokenizers import Tokenizer

        preview_tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        if any(
            not 1
            <= len(preview_tokenizer.encode(p, add_special_tokens=True).ids)
            <= MAX_PROMPT
            for p in request["prompts"]
        ):
            raise ValueError("Each sweep prompt must encode to 1–128 tokens")
        del preview_tokenizer
    else:
        if len(sys.argv) != 2:
            raise ValueError("Budget arguments require a sweep request")
        verify_model(directory)
    pair_preview = None
    if request.get("mode") in prompt_pair.MODES:
        request = contracts.validate_request(request)
        from tokenizers import Tokenizer

        preview_tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        pair_preview = contracts.token_preview(preview_tokenizer, request["prompts"])
        del preview_tokenizer
        if request["mode"] == "prompt_pair_preview":
            execute(
                request,
                Context(
                    torch=None,
                    tokenizer=None,
                    model=None,
                    generate=generate,
                    compare=compare,
                    prompt_pair=prompt_pair.run,
                    sweep=sweep.run,
                    record=emit,
                    capture_sites=value.capture_sites,
                    verified_parameters=None,
                    pair_preview=pair_preview,
                    sweep_plan=sweep_plan,
                    sweep_deadline=sweep_deadline,
                    cpu_allowance=cpu_allowance,
                ),
            )
            return
        contracts.validate_positions(request, pair_preview)
    started = time.perf_counter()
    torch, tokenizer, model = load_engine(directory)
    import platform
    import transformers, tokenizers, safetensors

    emit(
        {
            "type": "loaded",
            "load_ms": (time.perf_counter() - started) * 1000,
            "head_layout": model._atlas_verified_layout,
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "tokenizers": tokenizers.__version__,
                "safetensors": safetensors.__version__,
                "platform": sys.platform,
                "machine": platform.machine(),
                "device": "cpu",
                "dtype": "float32",
                "sampling": (
                    "none"
                    if request.get("mode") in ("prompt_pair", "sweep")
                    else "greedy"
                ),
                "seed": 0,
                "deterministic_algorithms": True,
                "numerical_threads": 1,
                "attention_backend": model.config._attn_implementation,
            },
        }
    )
    verify_pair_parameters = None
    if request.get("mode") == "prompt_pair":
        from atlas_host.inference_edits import verified_parameters

        verify_pair_parameters = verified_parameters
    execute(
        request,
        Context(
            torch=torch,
            tokenizer=tokenizer,
            model=model,
            generate=generate,
            compare=compare,
            prompt_pair=prompt_pair.run,
            sweep=sweep.run,
            record=emit,
            capture_sites=value.capture_sites,
            verified_parameters=verify_pair_parameters,
            pair_preview=pair_preview,
            sweep_plan=sweep_plan,
            sweep_deadline=sweep_deadline,
            cpu_allowance=cpu_allowance,
        ),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Never echo prompt text or model paths in the HTTP-visible error.
        emit(
            {
                "type": "error",
                "error": (
                    str(exc) if isinstance(exc, ValueError) else type(exc).__name__
                ),
            }
        )
        raise SystemExit(1)
