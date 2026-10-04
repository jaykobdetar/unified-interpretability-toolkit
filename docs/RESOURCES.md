# Standalone viewer resources

The viewer accepts an optional `resources` object in its JSON launcher configuration. Values are integer **bytes** or counts; unknown fields, duplicate fields, booleans, fractional numbers, disabled guards and out-of-range values are refused before launch. This config affects the standalone Rust reader/viewer. Hosted profile, inference, analytics and build/browser guards retain their separate policies.

| Field | Default | Valid range |
| --- | ---: | ---: |
| `version` | 1 | 1 |
| `cpu_count` | 1 | 1–8 |
| `address_space_bytes` | 805306368 (768 MiB) | 256 MiB–8 GiB |
| `available_floor_bytes` | 3221225472 (3 GiB) | 512 MiB–128 GiB |
| `disk_reserve_bytes` | 26843545600 (25 GiB) | 1–1024 GiB |
| `workspace_bytes` | 67108864 (64 MiB) | 64 MiB–1 GiB; at most half address space |
| `tile_cache_bytes` | 2147483648 (2 GiB) | 16 MiB–64 GiB |
| `tile_cache_files` | 1000 | 64–100000 |

For a measured machine with sufficient capacity, this example requests four allowed CPUs and a larger bounded cache:

```json
{
  "model": "/path/to/model",
  "cache": "../cache",
  "resources": {
    "version": 1,
    "cpu_count": 4,
    "address_space_bytes": 2147483648,
    "workspace_bytes": 134217728,
    "tile_cache_bytes": 8589934592,
    "tile_cache_files": 8000
  }
}
```

Launch with `./run-atlas.sh --config config/viewer.local.json`. `--check` validates the schema without applying limits or starting work. Native CLI callers may supply the same object as `--resources '{"cpu_count":4}'`; shell quoting must preserve the JSON. Custom budgets are unavailable for comparison or hosted-renderer commands.

Startup checks allowed CPU affinity, cgroup-v2 CPU quota, inherited address-space hard limit and effective available memory. Effective RAM is host `MemAvailable` capped by each visible cgroup-v2 ancestor's `memory.max - memory.current`. A process budget must fit **above** the configured free-RAM floor. A 768 MiB process budget with the default 3 GiB reserve therefore requires at least 3.75 GiB effective available RAM at admission. This standalone admission is stricter than the original floor-only check. Hosted renderer, comparison and profile worker commands retain the original host-only 3 GiB floor, one CPU and inherited-hard-limit clamp at 768 MiB; they do not activate this standalone policy. The selected address-space limit applies to the entire Rust process and its threads. All threads inherit the selected CPU set. `ATLAS_CPU` chooses the first CPU; additional CPUs come only from the already-allowed set. Owner-selected smaller machines may use lower finite RAM/disk reserves within these bounds, while measured admission and guarded writes still apply. No configuration disables memory or disk protections. Legacy cgroup-v1 memory quotas are not qualified.

One process-global numerical workspace ledger admits one render/calibration operation. Scoped rendering workers split disjoint, pooling-aligned output rows and join before any result or hosted numeric-idle acknowledgment. The workspace reservation accounts for outputs, read bands and thread stacks; insufficient workspace refuses work. Increasing CPU count does not add inference/profile jobs, queues or persistent workers. Calibration remains serial. This is process accounting, not a shared ledger across independent viewers; use separate caches and budget their combined footprint externally.

Narrow adjacent rows reuse reads only within the existing 4× amplification limit. Each raw span is at most 2 MiB. The existing 200000-element display-axis limit remains unchanged; larger axes are unavailable. Numeric order within each pooled pixel is retained. The disk cache tracks all PNG entries in its owned directory, trims before writes, limits each entry to 1 MiB and keeps the configured free-disk reserve. Calibration/histogram files are outside the PNG quota and still use guarded atomic writes.

The listener waits in the OS for socket readiness or the nearest absolute header deadline. Header slots, bounded dispatch/numeric queues, loopback/Host/Origin checks and response deadlines remain in effect. Defaults still use one CPU and the original address-space, RAM-floor, disk-floor and tile-cache values. Multicore performance and container behavior require qualification on the target machine; configuration acceptance alone is not performance evidence.

## Coarse overview preparation and browser tiles

After explicit calibration, `overview` prepares one at-most-256×256 image per selected rule at level `min(max_level, 8)`, using one source pass for up to four rules. It fills the same guarded PNG cache used by normal tile requests. It never calibrates or starts background work. The default explicit source-value budget is 16,777,216; `--max-values` must be 1–67,108,864 and the selected slice must fit. Fine levels remain on demand. For example:

```bash
./target/release/weight-atlas-rust overview --model /path/to/model --cache ./cache --tensor 0 --rules tensor_linear,tensor_asinh --max-values 16777216
```

`/api/view` returns opaque per-side `tile_bindings` for standalone viewer URLs. Their SHA-256 identity includes model revision identity, source identity, explicit slice, renderer version, rule and the exact returned legend. Bound requests are revalidated against the committed calibration before any disk-cache lookup. Matching PNGs use `Cache-Control: private, max-age=86400, immutable`; unbound legacy requests and API/error responses retain `no-store`. No credentials or owner capabilities are placed in the URL. Hosted and comparison views continue without this optional field. Startup JSON reports configured policy separately from measured effective affinity/quota, address-space limit, available memory and workspace accounting.
