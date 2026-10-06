"""Compatibility alias for the ordinary atlas_host package implementation."""

import sys
from atlas_host import inference_edit_contract as _implementation
from atlas_host.inference_edit_contract import (
    EditBindings,
    schema,
    validate_edits,
    verified_parameters,
    apply_edits,
    validate_pair,
)

sys.modules[__name__] = _implementation
