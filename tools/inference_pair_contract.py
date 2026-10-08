"""Compatibility alias for the ordinary atlas_host package implementation."""

import sys
from atlas_host import inference_pair_contract as _implementation
from atlas_host.inference_pair_contract import (
    PairBindings,
    validate_request,
    digest_for,
    token_preview,
    validate_preview,
    validate_positions,
    difference,
    validate_step,
)

sys.modules[__name__] = _implementation
