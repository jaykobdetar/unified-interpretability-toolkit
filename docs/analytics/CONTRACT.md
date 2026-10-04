# Regional analytics contract

## Boundary

`tools/analytics/core.py` provides dependency-free pure bounded calculations.
`tools/analytics/source.py` reads bounded BF16 regions from a **trusted, already
validated catalog** and verifies file identity before/after reading. It is not a
replacement model loader. `web/analytics-panel.js` exports `mountAnalytics(root,
{load, jump})`; `load({seed})` returns the response below and `jump({axis,index})`
moves the existing viewport without reordering it. The caller owns abort/epoch
handling and the existing serialized numeric queue. No new HTTP route or process
is installed. The host must allowlist these assets and adapt its validated
catalog; these are explicit integration steps.

Response: `schema: "weight-atlas.analytics.v1"`, `source_identity`, `tensor`,
`shape`, `region: {row,col,rows,cols}`, `cache_key`, `coverage`, `control`,
`original`, `shuffled`, `heads`, `svd`. Coordinates are native zero-based indices.
Coverage gives visited/total tensors and values, requested region, and explicit
full-tensor/full-model booleans. Full model means every catalog value was visited,
not merely every tensor name. No histogram is used to infer an address.

Original and shuffled hold row/column mean absolute strengths, ranked absolute
values with true source coordinates, and native-order vector values. Null head
labels carry an exclusion reason. Sorting returns index maps and changes only
the strip/list order; geometry stays native. Ties use native indices. Outlier
rankings use raw absolute magnitude (not an inferred significance score), with
row/column scores scoped to exactly the declared region.

Control is a seeded Fisher–Yates permutation of the same region's exact value
multiset, including signed zero. It includes shuffled-position → original-position
mapping. Strength/folded/SVD statistics are recomputed separately for each
control. Display scales are shared between original/control. Any future
transformed color view must declare whether original calibration is reused;
this contract uses raw values, no fitted transform. A shuffle alone does not
establish meaningful structure. Sorting manufactures gradients even for noise.

## Bounds and head provenance

At most 65,536 values and 4,096 entries per axis per region; at most 32 top results.
Model-wide scans accept at most 512 catalog tensors and 65,536 values **total**
per request, recording skipped tensors/regions. No eager tiles, persistent cache,
full Qwen allocation, inference, or new dependencies. Cache keys include source
identity, native region, seed, algorithm version and layout evidence. The host
must still revalidate source identity on cache hits.

Heads require a locally reviewed layout profile tying exact config SHA-256 and
implementation SHA-256 to separate contiguous projection layout, plus actual
tensor name and shape checks. Divisibility alone is insufficient. Q/K/V use row
boundaries; O uses input-column boundaries and is labeled output-projection
columns. Q uses query-head count; K/V use KV-head count. Folding reports mean
absolute value by within-head offset, counts, covered head IDs, and partial/full
head-axis coverage; partial windows never claim a full-head average. Unknown,
fused, transposed, ambiguous, or unreviewed layouts are excluded.

## Optional explicit dense SVD

No Rust linear algebra crate is vendored. Use existing local NumPy LAPACK in a
short-lived, one-CPU worker in an explicitly selected compatible environment. Limit both dimensions to 64, values to 4,096, wall time to 5 seconds,
address space to the existing 768 MiB runtime ceiling; check memory/disk gates.
One original and one exact-multiset shuffled control, each fitted independently.
Return singular energy fractions and leading rank-one residual Frobenius energy
ratio and residual region. Zero energy yields null ratios. Nonfinite input is
rejected. Larger matrices are explicitly excluded unless the caller selects a
bounded native window; window results must not be called full-matrix SVD.

Tradeoffs: optional NumPy availability/version and subprocess overhead; no Rust
dependency or hand-written approximate solver; hard process timeout can report
an exclusion. No SVD run, browser run, substantial scan, or Rust build occurs
without the serialized heavy slot. Existing launch/stop gates remain authoritative.

The separate `svd_summary` request returns a complete selected-window spectrum
up to 128 × 128 with a bounded residual preview and explicit source mapping.
Its [closed contract](SVD-SUMMARY-V2-DESIGN.md) and [qualification scope](SVD-QUALIFICATION.md)
keep the legacy dense response and its 64 × 64 cap unchanged.
