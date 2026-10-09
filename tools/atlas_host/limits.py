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

# Hosted owner policy: exact integer bytes, independent of floating worker factors.
HOSTED_DEFAULT_START_BYTES: Final[int] = 5 * 1024**3
HOSTED_STOP_RESERVE_BYTES: Final[int] = 13 * 1024**3 // 4
HOSTED_TREE_CEILING_BYTES: Final[int] = 768 * 1024**2
HOSTED_SNAPSHOT_MAX_BYTES: Final[int] = 32 * 1024**2
HOSTED_DISK_RESERVE_BYTES: Final[int] = 25 * 1024**3

# analytics/svd_summary.py: independent policy bounds.
SVD_SUMMARY_PREVIEW_AXIS: Final[int] = 16

# atlas_host/profile_snapshot.py: independent policy bounds.
PROFILE_FIXED_RESERVE_BYTES: Final[int] = 2 * 1024**2 + 128 * 1024

# atlas_host/profile_worker.py: independent policy bounds.
PROFILE_FINALIZE_RESERVE_SECONDS: Final[float] = 0.8
PROFILE_POLL_SLICE_SECONDS: Final[float] = 0.002

# atlas_host/config.py: independently owned dictionary policy facts.
HOST_CONFIG_CPU_COUNT: Final[int] = 1
HOST_CONFIG_NUMERIC_WORKERS: Final[int] = 1
HOST_CONFIG_HEAVY_JOBS: Final[int] = 1
HOST_CONFIG_RUST_ADDRESS_SPACE_BYTES: Final[int] = 768 * 1024**2
HOST_CONFIG_RUST_AVAILABLE_FLOOR_BYTES: Final[int] = 3 * 1024**3
HOST_CONFIG_BUILD_ADDRESS_SPACE_BYTES: Final[int] = 2 * 1024**3
HOST_CONFIG_BUILD_JOBS: Final[int] = 1
HOST_CONFIG_BUILD_START_AVAILABLE_BYTES: Final[int] = 5 * 1024**3
HOST_CONFIG_BROWSER_TREE_RSS_BYTES: Final[int] = 768 * 1024**2
HOST_CONFIG_BROWSER_START_AVAILABLE_BYTES: Final[int] = 5 * 1024**3
HOST_CONFIG_STOP_AVAILABLE_BYTES: Final[int] = 13 * 1024**3 // 4
HOST_CONFIG_INFERENCE_RSS_BYTES: Final[int] = 3 * 1024**3 // 2
HOST_CONFIG_INFERENCE_ADDRESS_SPACE_BYTES: Final[int] = 3 * 1024**3
HOST_CONFIG_INFERENCE_START_AVAILABLE_BYTES: Final[int] = 19 * 1024**3 // 4
HOST_CONFIG_INFERENCE_WALL_MS: Final[int] = 120000
HOST_CONFIG_INFERENCE_CPU_MS: Final[int] = 90000
HOST_CONFIG_PROMPT_TOKENS: Final[int] = 128
HOST_CONFIG_NEW_TOKENS: Final[int] = 32
HOST_CONFIG_CLIENT_LEASE_MS: Final[int] = 15000
HOST_CONFIG_INFERENCE_QUEUE: Final[int] = 0
HOST_CONFIG_ANALYTICS_ADDRESS_SPACE_BYTES: Final[int] = 768 * 1024**2
HOST_CONFIG_ANALYTICS_RSS_BYTES: Final[int] = 768 * 1024**2
HOST_CONFIG_ANALYTICS_START_AVAILABLE_BYTES: Final[int] = 15 * 1024**3 // 4
HOST_CONFIG_ANALYTICS_WALL_MS: Final[int] = 5000
HOST_CONFIG_ANALYTICS_CPU_MS: Final[int] = 4000
HOST_CONFIG_DISK_RESERVE_BYTES: Final[int] = 25 * 1024**3
HOST_CONFIG_TILE_DISK_BYTES: Final[int] = 2 * 1024**3
HOST_CONFIG_TILE_FILES: Final[int] = 1000
HOST_CONFIG_PENDING_HEADERS: Final[int] = 4
HOST_CONFIG_HEADER_BYTES: Final[int] = 8192
HOST_CONFIG_HEADER_DEADLINE_MS: Final[int] = 500
HOST_CONFIG_DISPATCH_QUEUE: Final[int] = 4
HOST_CONFIG_NUMERIC_QUEUE: Final[int] = 8
HOST_CONFIG_WRITE_DEADLINE_MS: Final[int] = 3000
HOST_CONFIG_COORDINATOR_BODY_BYTES: Final[int] = 8192
HOST_CONFIG_UPSTREAM_RESPONSE_BYTES: Final[int] = 2 * 1024**2

# atlas_host/viewer_resources.py: independently owned dictionary policy facts.
VIEWER_DEFAULT_CPU_COUNT: Final[int] = 1
VIEWER_DEFAULT_ADDRESS_SPACE_BYTES: Final[int] = 768 * 1024**2
VIEWER_DEFAULT_AVAILABLE_FLOOR_BYTES: Final[int] = 3 * 1024**3
VIEWER_DEFAULT_DISK_RESERVE_BYTES: Final[int] = 25 * 1024**3
VIEWER_DEFAULT_WORKSPACE_BYTES: Final[int] = 64 * 1024**2
VIEWER_DEFAULT_TILE_CACHE_BYTES: Final[int] = 2 * 1024**3
VIEWER_DEFAULT_TILE_CACHE_FILES: Final[int] = 1000
VIEWER_MIN_CPU_COUNT: Final[int] = 1
VIEWER_MAX_CPU_COUNT: Final[int] = 8
VIEWER_MIN_ADDRESS_SPACE_BYTES: Final[int] = 256 * 1024**2
VIEWER_MAX_ADDRESS_SPACE_BYTES: Final[int] = 8 * 1024**3
VIEWER_MIN_AVAILABLE_FLOOR_BYTES: Final[int] = 512 * 1024**2
VIEWER_MAX_AVAILABLE_FLOOR_BYTES: Final[int] = 128 * 1024**3
VIEWER_MIN_DISK_RESERVE_BYTES: Final[int] = 1024**3
VIEWER_MAX_DISK_RESERVE_BYTES: Final[int] = 1024 * 1024**3
VIEWER_MIN_WORKSPACE_BYTES: Final[int] = 64 * 1024**2
VIEWER_MAX_WORKSPACE_BYTES: Final[int] = 1024**3
VIEWER_MIN_TILE_CACHE_BYTES: Final[int] = 16 * 1024**2
VIEWER_MAX_TILE_CACHE_BYTES: Final[int] = 64 * 1024**3
VIEWER_MIN_TILE_CACHE_FILES: Final[int] = 64
VIEWER_MAX_TILE_CACHE_FILES: Final[int] = 100000

# Browser archive intake: independent resource policies, not model geometry.
ARCHIVE_MAX_ARRAY_VALUES: Final[int] = 576
ARCHIVE_MAX_ACTIVATION_VALUES: Final[int] = 576
