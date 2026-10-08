"""Strict clients bind actual worker callbacks and the actual sweep implementation."""

from pathlib import Path
from typing import Any

from inference_worker import (
    configure_worker_limits,
    emit,
    load_engine,
    generate,
    compare,
)
from atlas_host.inference_experiments import Generate, Compare, Record, Sweep
from atlas_host.inference_sweep import run
from atlas_host.inference_comparison import ComparisonBindings
from atlas_host.inference_edits import apply_edits


def callbacks() -> tuple[Generate, Compare, Record, Sweep]:
    return generate, compare, emit, run


def engine(directory: Path) -> tuple[Any, Any, Any]:
    return load_engine(directory)


def budget(value: object) -> None:
    configure_worker_limits(value)


def edit_callback(bindings: ComparisonBindings, torch: Any, model: Any) -> object:
    declared = ComparisonBindings(
        bindings.generate,
        bindings.paired_step,
        bindings.validate_edits,
        bindings.verified_parameters,
        apply_edits,
    )
    return declared.apply_edits(torch, model, [], {})
