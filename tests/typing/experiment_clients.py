"""Typed consumers of the current public callback and tokenizer protocols."""

from typing import assert_type

from inference_experiments import Compare, Generate, PromptPair, Record, Request, Sweep
from inference_generation import Tokenizer


def callbacks(
    generate: Generate,
    compare: Compare,
    pair: PromptPair,
    sweep: Sweep,
    request: Request,
    record: Record,
    tokenizer: Tokenizer,
) -> None:
    engine, model, prompt = object(), object(), object()
    assert_type(
        generate(engine, tokenizer, model, prompt, object(), object(), record=record),
        None,
    )
    assert_type(compare(engine, tokenizer, model, request, record=record), None)
    assert_type(
        pair(engine, tokenizer, model, request, None, {"block": "held"}, record), None
    )
    assert_type(
        sweep(
            engine,
            tokenizer,
            model,
            request,
            None,
            None,
            generate,
            record,
            cpu_deadline=17,
        ),
        None,
    )
    ids: list[int] = tokenizer.encode("held", add_special_tokens=True)
    text: str = tokenizer.decode(ids, skip_special_tokens=False)
    assert_type(ids, list[int])
    assert_type(text, str)
