"""Compatibility alias for the ordinary atlas_host package implementation."""

import sys
from atlas_host import inference_services as _implementation
from atlas_host.inference_services import InferenceContracts as InferenceContracts

sys.modules[__name__] = _implementation
