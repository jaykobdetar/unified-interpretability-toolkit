"""Compatibility alias for the ordinary atlas_host package implementation."""

import sys
from atlas_host import inference_generation as _implementation
from atlas_host.inference_generation import (
    Tokenizer as Tokenizer,
    Observation as Observation,
    GenerationBindings as GenerationBindings,
    run as run,
)

sys.modules[__name__] = _implementation
