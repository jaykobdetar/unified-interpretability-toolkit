"""Compatibility alias for the ordinary atlas_host package implementation."""

import sys
from atlas_host import viewer_resources as _implementation
from atlas_host.viewer_resources import (
    MIB as MIB,
    GIB as GIB,
    DEFAULTS as DEFAULTS,
    RANGES as RANGES,
    validate as validate,
    encode as encode,
)

sys.modules[__name__] = _implementation
