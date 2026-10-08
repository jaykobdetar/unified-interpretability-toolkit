"""Compatibility alias for the ordinary atlas_host package implementation."""

import sys
from atlas_host import inference_prompt_pair as _implementation
from atlas_host.inference_prompt_pair import (
    _bindings as _bindings,
    validate_request as validate_request,
    digest_for as digest_for,
    token_preview as token_preview,
    validate_preview as validate_preview,
    validate_positions as validate_positions,
    difference as difference,
    validate_step as validate_step,
    run as run,
    PairBindings as PairBindings,
    Architecture as Architecture,
    MODES as MODES,
    SHA256_HEX_LENGTH as SHA256_HEX_LENGTH,
)

sys.modules[__name__] = _implementation
