"""Compatibility alias for the ordinary atlas_host package implementation."""

import sys
from atlas_host import inference_comparison as _implementation
from atlas_host.inference_comparison import ComparisonBindings, paired_step, run, Event

sys.modules[__name__] = _implementation
