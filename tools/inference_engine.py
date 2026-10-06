"""Compatibility alias for the ordinary atlas_host package implementation."""

import sys
from atlas_host import inference_engine as _implementation
from atlas_host.inference_engine import (
    AttentionRuntime,
    LoaderRuntime,
    verify_attention_layout,
    load_engine,
)

sys.modules[__name__] = _implementation
