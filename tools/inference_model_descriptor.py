"""Compatibility alias for the ordinary atlas_host package implementation."""

import sys
from atlas_host import inference_model_descriptor as _implementation
from atlas_host.inference_model_descriptor import (
    MAX_CONFIG_BYTES as MAX_CONFIG_BYTES,
    RUNTIME_VERSION as RUNTIME_VERSION,
    FAMILIES as FAMILIES,
    ConfigurationEvidence as ConfigurationEvidence,
    ProjectionMapping as ProjectionMapping,
    DescriptorSource as DescriptorSource,
    PinnedDescriptor as PinnedDescriptor,
    _integer as _integer,
    describe_config as describe_config,
    parameter_shapes as parameter_shapes,
    projection_mappings as projection_mappings,
    pinned_descriptor as pinned_descriptor,
)

sys.modules[__name__] = _implementation
