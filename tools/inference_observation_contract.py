"""Compatibility alias for the ordinary atlas_host package implementation."""

import sys
from atlas_host import inference_observation_contract as _implementation
from atlas_host.inference_observation_contract import (
    ObservationBindings,
    schema,
    validate_observation,
    validate_record,
    lens_record,
)

sys.modules[__name__] = _implementation
