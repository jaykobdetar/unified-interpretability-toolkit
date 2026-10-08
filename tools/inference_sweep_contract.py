"""Compatibility alias for the ordinary atlas_host package implementation."""

import sys
from atlas_host import inference_sweep_contract as _implementation
from atlas_host.inference_sweep_contract import (
    SweepBindings,
    schema,
    _target,
    _rows,
    _edits,
    build_plan,
    validate_step,
)

sys.modules[__name__] = _implementation
