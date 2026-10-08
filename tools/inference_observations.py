"""Compatibility alias for the ordinary atlas_host package implementation."""

import sys
from atlas_host import inference_observations as _implementation
from atlas_host.inference_observations import (
    _bindings as _bindings,
    schema as schema,
    validate_observation as validate_observation,
    _integer as _integer,
    _finite as _finite,
    validate_record as validate_record,
    lens_record as lens_record,
    ObservationBindings as ObservationBindings,
    Architecture as Architecture,
    ATTENTION_SEMANTICS as ATTENTION_SEMANTICS,
    LENS_SEMANTICS as LENS_SEMANTICS,
)

sys.modules[__name__] = _implementation
