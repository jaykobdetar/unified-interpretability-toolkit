"""Compatibility alias for the ordinary atlas_host package implementation."""

import sys
from atlas_host import inference_geometry as _implementation
from atlas_host.inference_geometry import (
    Architecture,
    describe,
    shapes,
    head_layout_descriptor,
)

sys.modules[__name__] = _implementation
