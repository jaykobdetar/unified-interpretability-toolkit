"""Compatibility alias for the ordinary atlas_host package implementation."""

import sys
from atlas_host import inference_architecture as _implementation
from atlas_host.inference_architecture import (
    describe as describe,
    architecture as architecture,
    shapes as shapes,
    head_layout_descriptor as head_layout_descriptor,
    verify_attention_layout as verify_attention_layout,
    bind_viewer_head_layout as bind_viewer_head_layout,
    Architecture as Architecture,
    ROOT as ROOT,
    MANIFEST as MANIFEST,
    CAPTURE_SITES as CAPTURE_SITES,
    CONFIG as CONFIG,
    ARCH as ARCH,
    WIDTH as WIDTH,
    LAYERS as LAYERS,
    HEADS as HEADS,
    KV_HEADS as KV_HEADS,
    HEAD_DIM as HEAD_DIM,
    VOCAB as VOCAB,
)

sys.modules[__name__] = _implementation
