"""Both worker import paths expose the actual typed callback implementation."""

from pathlib import Path
from typing import Any
from atlas_host import inference_worker as canonical
from inference_worker import generate, compare, emit, load_engine
from atlas_host.inference_experiments import Generate, Compare, Record


def callbacks() -> tuple[Generate, Generate, Compare, Compare, Record, Record]:
    return (
        canonical.generate,
        generate,
        canonical.compare,
        compare,
        canonical.emit,
        emit,
    )


def engines(directory: Path) -> tuple[tuple[Any, Any, Any], tuple[Any, Any, Any]]:
    return canonical.load_engine(directory), load_engine(directory)
