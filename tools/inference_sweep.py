"""Compatibility alias for the ordinary atlas_host package implementation."""

import sys
from atlas_host import inference_sweep as _implementation
from atlas_host.inference_sweep import (
    _bindings as _bindings,
    schema as schema,
    _int as _int,
    _canonical as _canonical,
    _target as _target,
    _rows as _rows,
    _space as _space,
    _edits as _edits,
    _choice as _choice,
    build_plan as build_plan,
    selected_indices as selected_indices,
    RestorationFailure as RestorationFailure,
    temporary_edits as temporary_edits,
    record_ids as record_ids,
    coverage as coverage,
    metrics as metrics,
    validate_step as validate_step,
    run as run,
    SweepBindings as SweepBindings,
    Architecture as Architecture,
    MAX_CELLS as MAX_CELLS,
    MAX_TARGETS as MAX_TARGETS,
    MAX_CASES as MAX_CASES,
    MAX_PROMPTS as MAX_PROMPTS,
    MAX_RECORDS as MAX_RECORDS,
    WALL_SECONDS as WALL_SECONDS,
    CPU_SECONDS as CPU_SECONDS,
    CONTROL_VERSION as CONTROL_VERSION,
    MAX_TRACE as MAX_TRACE,
)

sys.modules[__name__] = _implementation
