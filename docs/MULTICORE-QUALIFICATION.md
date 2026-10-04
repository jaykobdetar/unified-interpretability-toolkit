# Inherited multicore qualification

The standalone renderer supports explicitly configured, bounded multicore rendering. On one fixed 32 MiB BF16 fixture, 1-, 2- and 4-CPU runs produced identical F64 fields and PNGs for global linear/asinh rules, matched an independent numeric and pixel oracle, and showed actual two- and four-worker concurrency with the expected affinities. A separate ordinary TCP proof verified bound-tile reuse and close behavior. The single-fixture timings do not establish a general speedup.

This evidence belongs to resource lane `2d9e115420c0f606a58897d0144a3fcc36eb0256`. The integrated batch retains identical CLI dispatch, renderer, resources, source, slice, state, library, dependencies and build/cleanup guards. Its HTTP transport changed in `server.rs` with two added reuse modules; the tested CLI tile path does not call that server path. Renderer-source correspondence supports retaining this scoped evidence. The combined binary was not rerun with multiple CPUs, so this is inherited qualification, not a current multicore binary test.

Browser cache/reload/source switching, other shapes/dtypes/slices/edges, sustained throughput, idle wakeups, repeated timing statistics, scheduling guarantees, remote/cgroup and FUSE behavior remain separate. Defaults and resource caps are unchanged. No further benchmark is scheduled by this record.
