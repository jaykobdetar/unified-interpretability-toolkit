"""Compatibility alias for the ordinary atlas_host package implementation."""

import sys
from atlas_host import inference_experiments as _implementation
from atlas_host.inference_experiments import (
    Generate as Generate,
    Compare as Compare,
    PromptPair as PromptPair,
    Sweep as Sweep,
    Context as Context,
    Kind as Kind,
    Experiment as Experiment,
    run_preview as run_preview,
    run_sweep as run_sweep,
    run_prompt_pair as run_prompt_pair,
    run_comparison as run_comparison,
    run_generation as run_generation,
    select as select,
    execute as execute,
    for_coordinator as for_coordinator,
    Request as Request,
    Record as Record,
    REGISTRY as REGISTRY,
)

sys.modules[__name__] = _implementation
