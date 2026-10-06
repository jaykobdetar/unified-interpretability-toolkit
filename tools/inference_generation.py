"""Bounded generation experiment with an injected worker and architecture boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import time
from typing import Any, NotRequired, Protocol, TypedDict


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def decode(self, ids: list[int], *, skip_special_tokens: bool) -> str: ...


class Observation(TypedDict):
    kind: str
    head: NotRequired[int]


@dataclass(frozen=True)
class GenerationBindings:
    max_prompt: int
    max_new: int
    width: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    capture_sites: Mapping[str, str]
    attention_semantics: str
    validate_observation: Callable[[Observation, str], Observation]
    validate_record: Callable[[dict[str, object], Observation | None, int], None]
    lens_record: Callable[[Any, Tokenizer, Any, Any, Any, int, int], dict[str, object]]


def run(
    torch: Any,
    tokenizer: Tokenizer,
    model: Any,
    prompt: str,
    limit: int,
    layer: int,
    record: Callable[[dict[str, object]], None],
    bindings: GenerationBindings,
    capture_scores: Callable[[Any], None] | None = None,
    activation_site: str = "block",
    observation: Observation | None = None,
) -> None:
    # Opaque tensors and engine objects stay at the injected third-party boundary.
    MAX_PROMPT, MAX_NEW = bindings.max_prompt, bindings.max_new
    WIDTH, LAYERS = bindings.width, bindings.layers
    HEADS, KV_HEADS = bindings.heads, bindings.kv_heads
    HEAD_DIM, VOCAB = bindings.head_dim, bindings.vocab
    CAPTURE_SITES = bindings.capture_sites
    ATTENTION_SEMANTICS = bindings.attention_semantics
    validate_observation = bindings.validate_observation
    validate_record = bindings.validate_record
    lens_record = bindings.lens_record
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    if not 1 <= len(ids) <= MAX_PROMPT:
        raise ValueError(f"Prompt must encode to 1–{MAX_PROMPT} tokens; got {len(ids)}")
    if (
        type(limit) is not int
        or not 1 <= limit <= MAX_NEW
        or type(layer) is not int
        or not 0 <= layer < LAYERS
    ):
        raise ValueError("Invalid token limit or activation layer")
    if type(activation_site) is not str or activation_site not in CAPTURE_SITES:
        raise ValueError("Choose activation site block, attention, or mlp")
    if observation is not None:
        observation = validate_observation(observation, activation_site)
        config = model.config
        if (
            config._attn_implementation != "eager"
            or model.training
            or config.num_attention_heads != HEADS
            or config.num_key_value_heads != KV_HEADS
            or config.hidden_size != WIDTH
            or config.vocab_size != VOCAB
            or model.model.layers[layer].self_attn.head_dim != HEAD_DIM
        ):
            raise ValueError(
                "Observation requires the pinned eager SmolLM2 architecture in eval mode"
            )
    record(
        {
            "type": "prefill",
            "prompt_tokens": len(ids),
            "prompt_ids": ids,
            "layer": layer,
            "activation_site": activation_site,
            "seed": 0,
            "sampling": "greedy",
            "dtype": "float32",
        }
    )
    selected: list[float] = []
    observed: dict[str, Any] = {}

    def capture(_module: object, _args: tuple[object, ...], output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        selected[:] = hidden[0, -1, :].detach().float().tolist()
        if observation and observation["kind"] == "attention":
            weights = (
                output[1] if isinstance(output, tuple) and len(output) == 2 else None
            )
            if (
                weights is None
                or weights.ndim != 4
                or weights.shape[0] != 1
                or weights.shape[1] != HEADS
                or not 1 <= weights.shape[3] <= 160
            ):
                raise ValueError("Selected eager attention probabilities unavailable")
            observed["probabilities"] = (
                weights[0, observation["head"], -1, :].detach().float().tolist()
            )
        elif observation:
            observed["residual"] = hidden[0, -1, :].detach().clone()

    block = model.model.layers[layer]
    target = (
        block
        if activation_site == "block"
        else block.self_attn if activation_site == "attention" else block.mlp
    )
    hook = target.register_forward_hook(capture)
    past = None
    tokens: list[int] = []
    inputs = torch.tensor([ids], dtype=torch.long)
    compute_total = 0.0
    try:
        with torch.inference_mode():
            for index in range(limit):
                start = time.perf_counter()
                out = model(
                    input_ids=inputs,
                    past_key_values=past,
                    use_cache=True,
                    logits_to_keep=1,
                )
                past = out.past_key_values
                logits = out.logits[0, -1, :]
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError("Nonfinite inference logits")
                if capture_scores is not None:
                    capture_scores(logits.detach().clone())
                position = len(ids) - 1 + index
                extra: dict[str, object] = {}
                if observation and observation["kind"] == "attention":
                    extra["attention"] = {
                        "layer": layer,
                        "query_head": observation["head"],
                        "kv_head": observation["head"] // (HEADS // KV_HEADS),
                        "head_dim": HEAD_DIM,
                        "query_position": position,
                        "key_positions": list(range(position + 1)),
                        "key_token_ids": ids + tokens,
                        "probabilities": observed["probabilities"],
                        "semantics": ATTENTION_SEMANTICS,
                    }
                elif observation:
                    extra["logit_lens"] = lens_record(
                        torch,
                        tokenizer,
                        observed.pop("residual"),
                        model,
                        logits,
                        layer,
                        position,
                    )
                token = int(torch.argmax(logits))
                top = torch.topk(logits, 5)
                compute_ms = (time.perf_counter() - start) * 1000
                compute_total += compute_ms
                tokens.append(token)
                eos = token == model.config.eos_token_id
                step: dict[str, object] = {
                    "type": "step",
                    "index": index,
                    "phase": "prefill" if index == 0 else "decode",
                    "position": len(ids) - 1 + index,
                    "input_token_id": int(inputs[0, -1]),
                    "token_id": token,
                    "token_piece": tokenizer.decode([token], skip_special_tokens=False),
                    "generated_text": tokenizer.decode(
                        tokens, skip_special_tokens=False
                    ),
                    "compute_ms": compute_ms,
                    "compute_total_ms": compute_total,
                    "layer": layer,
                    "activation": list(selected),
                    "activation_site": activation_site,
                    "activation_kind": CAPTURE_SITES[activation_site]
                    + ", last consumed position",
                    "top_logits": [
                        {"id": int(t), "value": float(v)}
                        for t, v in zip(top.indices, top.values)
                    ],
                    "eos": eos,
                    **extra,
                }
                validate_record(step, observation, layer)
                record(step)
                if eos:
                    break
                inputs = torch.tensor([[token]], dtype=torch.long)
        record(
            {
                "type": "done",
                "reason": "eos" if eos else "token_limit",
                "generated_tokens": len(tokens),
                "compute_total_ms": compute_total,
            }
        )
    finally:
        hook.remove()
