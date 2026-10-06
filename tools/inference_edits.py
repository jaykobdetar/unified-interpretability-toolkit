"""Compatibility alias for the ordinary atlas_host package implementation."""

import sys
from atlas_host import inference_edits as _implementation
from atlas_host.inference_edits import (
    _bindings as _bindings,
    schema as schema,
    validate_edits as validate_edits,
    verified_parameters as verified_parameters,
    apply_edits as apply_edits,
    validate_pair as validate_pair,
    EditBindings as EditBindings,
    Architecture as Architecture,
    SOURCE_MODEL as SOURCE_MODEL,
    MAX_EDITS as MAX_EDITS,
    SHAPES as SHAPES,
    ALIASES as ALIASES,
)

sys.modules[__name__] = _implementation
