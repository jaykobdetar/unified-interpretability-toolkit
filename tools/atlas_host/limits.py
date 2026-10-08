"""Shared memory policy facts; callers retain units, timing and refusal text."""

from typing import Final

MEMORY_RESERVE_GIB: Final[float] = 3.25
MODEL_ADMISSION_GIB: Final[float] = 4.75

# analytics/core.py: independent policy bounds.
ANALYTICS_MAX_VALUES: Final[int] = 65536
ANALYTICS_MAX_AXIS: Final[int] = 4096
ANALYTICS_MAX_TOP: Final[int] = 32
ANALYTICS_MAX_TENSORS: Final[int] = 512

# analytics/svd.py: independent policy bounds.
SVD_MAX_AXIS: Final[int] = 64
SVD_MAX_VALUES: Final[int] = 4096
SVD_TIMEOUT_SECONDS: Final[int] = 5

# analytics/svd_summary.py: independent policy bounds.
SVD_SUMMARY_MAX_AXIS: Final[int] = 128
SVD_SUMMARY_MAX_VALUES: Final[int] = 16384
SVD_SUMMARY_MAX_BODY: Final[int] = 63488

# analytics/worker.py: independent policy bounds.
ANALYTICS_MAX_INPUT: Final[int] = 512 * 1024
ANALYTICS_MAX_OUTPUT: Final[int] = 2 * 1024 * 1024 - 2048

# atlas_host/acquisition.py: independent policy bounds.
ACQUISITION_MAX_BYTES: Final[int] = 64 * 1024**3

# atlas_host/inference_edits.py: independent policy bounds.
INFERENCE_MAX_EDITS: Final[int] = 8

# atlas_host/inference_model_descriptor.py: independent policy bounds.
MODEL_DESCRIPTOR_MAX_CONFIG_BYTES: Final[int] = 65536

# atlas_host/inference_sweep.py: independent policy bounds.
SWEEP_MAX_CELLS: Final[int] = 65536
SWEEP_MAX_TARGETS: Final[int] = 2
SWEEP_MAX_CASES: Final[int] = 5
SWEEP_MAX_PROMPTS: Final[int] = 2
SWEEP_MAX_RECORDS: Final[int] = 10
SWEEP_WALL_SECONDS: Final[int] = 120
SWEEP_CPU_SECONDS: Final[int] = 90
SWEEP_MAX_TRACE: Final[int] = 32

# atlas_host/inference_worker.py: independent policy bounds.
INFERENCE_MAX_PROMPT: Final[int] = 128
INFERENCE_MAX_NEW: Final[int] = 32

# atlas_host/live_inference.py: independent policy bounds.
COORDINATOR_MAX_BODY: Final[int] = 8192
COORDINATOR_MAX_TRACE: Final[int] = 32
COORDINATOR_REQUEST_DEADLINE: Final[float] = 0.5
COORDINATOR_UPSTREAM_DEADLINE: Final[float] = 5.0
COORDINATOR_IO_TICK: Final[float] = 0.1

# atlas_host/profile_snapshot.py: independent policy bounds.
PROFILE_MAX_STATE: Final[int] = 32 * 1024**2

# atlas_host/registry.py: independent policy bounds.
REGISTRY_MAX_BYTES: Final[int] = 512 * 1024
REGISTRY_MAX_ENTRIES: Final[int] = 128
REGISTRY_MAX_MANIFEST_BYTES: Final[int] = 32768

# atlas_host/startup_diagnostics.py: independent policy bounds.
STARTUP_STDERR_LIMIT: Final[int] = 16 * 1024
STARTUP_ATTEMPT_LIMIT: Final[int] = 4
STARTUP_ERROR_CHAIN_LIMIT: Final[int] = 8
STARTUP_TRACE_FRAME_LIMIT: Final[int] = 64

# atlas_host/static_models.py: independent policy bounds.
STATIC_MAX_HEADER_BYTES: Final[int] = 2 * 1024**2
STATIC_MAX_ACTIVATION_METADATA: Final[int] = 8 * 1024**2
STATIC_DISPLAY_AXIS_LIMIT: Final[int] = 200000
