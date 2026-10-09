# Checkpoint comparison

Compare two existing local checkpoints with matching tensor names and shapes. This is a separate viewer: the two panels in the basic weight viewer compare color rules on one tensor.

## Open the comparison viewer

First build the release executable using the [launcher](LAUNCH.md). Stop that viewer if you no longer need it, then run:

```bash
target/release/weight-atlas-rust compare-serve \
  --model /path/to/checkpoint-A \
  --compare-model /path/to/checkpoint-B \
  --cache ./cache-pair --port 8776
```

Open `http://127.0.0.1:8776`, select a compatible tensor, and explicitly calibrate that pair for color rendering. Raw inspection is available before calibration. Choose A/B to inspect originals, delta for `B − A`, or absolute delta for `abs(B − A)`. Pan and zoom stay synchronized. A/B share one original-value scale; derived differences use a separate scale. The mapping selector applies to both panels. Stop with Ctrl-C.

Both complete tensor-name sets and native shapes must match. The viewer does not align different architectures, broadcast shapes, or decode unsupported quantization. Mixed BF16/F16/F32 pairs are supported. Use a dedicated cache outside both sources; another process cannot use that cache while the server owns it.

## Interface and numeric contract

CLI: `compare-metadata`, `compare-calibrate`, `compare-inspect`, `compare-tile`, `compare-serve`. Explicit `--model A --compare-model B`; existing `--cache` external to both sources. Default panel quantities A/B, default mapping linear. `--quantity a|b|delta|abs_delta`, `--mapping linear|asinh|magnitude`; tensor ID addresses an explicitly named compatible pair. `--tensor`, native row/col or level/x/y geometry retain existing conventions.

HTTP: dedicated comparison server/page; GET `/api/comparison/model`, `/api/comparison/view?tensor=ID&left=a&right=b&mapping=linear`, `/api/comparison/inspect?tensor=ID&row=R&col=C`, `/api/comparison/tile?tensor=ID&quantity=delta&mapping=linear&level=L&x=X&y=Y`; POST `/api/comparison/calibrate?tensor=ID` requires existing X-Atlas-Local: 1 and queues only that tensor. No full-model auto scan. Server shares original loopback/Host/Origin/body/admission/deadline gates and serialized bounded numeric work. Single-source/inference routes are unavailable on this comparison server.

### Source identity and compatibility

Identity extension: `comparison_identity` is a digest of ordered existing source identities plus complete named-shape/dtype correspondence and schema. `sources.a.source_identity` and `sources.b.source_identity` remain exact existing identities. They bind canonical paths, fingerprints, header hashes and index metadata; they are not fresh payload SHA claims. Responses include `coordinate_space: checkpoint-comparison-v1` and `inference_editable: false`. View/inspect additionally label whether values are original or derived. Never emit a single-source `tensor` edit-target object for comparison addresses.

Complete equality of name sets and exact native shapes is required before publication; IDs/order never establish compatibility. Mismatch reports bounded name details plus complete mismatch counts. Only BF16/F16/F32; mixed pairs explicitly allowed and decoded to F64 without a storage cast. Source mutation invalidates either direction. No broadcasting, truncation, tensor copying or inference.

### Values, scales, and pooling

Quantities a/b are original values; delta is F64 arithmetic B-A; abs_delta is abs(B-A). Inspector returns exact original strings, raw bytes, dtype, shard/offset and coordinates for both, plus labeled derived F64 difference (not exact symbolic subtraction). Nonfinite originals remain inspectable with null difference/reason. Any nonfinite refuses the whole calibration/render transaction.

One serialized complete paired-tensor scan calculates shared_raw_max=max(abs(A),abs(B)) and difference_max=max(abs(B-A)). These are separate calibration domains. Initial mappings: linear v/bound; magnitude abs(v)/bound; asinh(v/s)/asinh(bound/s), s=bound/100 or1. Bound-zero fields are zero. abs_delta and magnitude use unsigned purple and a one-sided legend; signed quantities use the existing signed palette. Transform before mean pooling; partial edges count actual coordinates. Asinh has an explicitly comparison-specific fixed scale; it does not impersonate median-based tensor_asinh. Comparison robust/percentile/global rules are unsupported, never approximated or silently replaced.

### Resource and cache boundaries

Paired reads total <=2MiB including both input buffers, no full tensor/delta buffers. Existing one-CPU, 768MiB AS, RAM/disk guards remain. Pair cache has existing bounded tile budget and file lock; comparison calibration schema2, tile renderer namespace comparison-v2-exact-json-transform-before-mean. Keys bind ordered identities, name/shape/dtypes, B-A direction, quantity, mapping, calibration and geometry. Cache outside both source trees; stale calibration rejected. Atomic publication occurs only after both source rechecks. Prior failures are preserved in runtime progress/evidence.

### Browser interface

UI: separate comparison.html/comparison.js; no inference script or edit controls. Native coordinate and synchronized geometry shared across two panels; explicit independent quantity selectors plus one common mapping selector. Provenance labels A and B identity/root and full named-shape compatibility. Legends label shared original scale versus derived difference scale. Raw inspector remains available before calibration. Epoch/abort checks clear stale views and inspections; quantity changes never reuse a single-source tile URL. Small screens stack panels with wrapping labels.

## Usage and integration

Build the current source, then use explicit local checkpoint paths:

```bash
target/release/weight-atlas-rust compare-metadata --model /local/A --compare-model /local/B --cache ./cache-pair
target/release/weight-atlas-rust compare-inspect --model /local/A --compare-model /local/B --cache ./cache-pair --tensor 0 --row 0 --col 0
target/release/weight-atlas-rust compare-calibrate --model /local/A --compare-model /local/B --cache ./cache-pair --tensor 0
target/release/weight-atlas-rust compare-tile --model /local/A --compare-model /local/B --cache ./cache-pair --tensor 0 --quantity delta --mapping linear --level 0 --x 0 --y 0 --out ./results/delta
target/release/weight-atlas-rust compare-serve --model /local/A --compare-model /local/B --cache ./cache-pair --port 8776
```

Tile exports are `PREFIX.png` and `PREFIX.f64le`; stdout includes both ordered identities, the exact selected legend and metrics. The prefix must be outside both source trees, with symlink and parent-component resolution checked before output creation. The CLI fields and HTTP tiles use the same comparison field implementation. `source_count` and `max_raw_band_values` count values across both inputs; multiplying half the value count by `element_bytes_A + element_bytes_B` gives the corresponding paired byte count. `source_bytes_read` includes both sources. One HTTP tile has at most 256×256 output cells.

The default cache is `cache-comparison`; the default comparison port is 8776. The cache lock prevents simultaneous use by a CLI command and server. Metadata/header validation is complete before serving; calibration remains an explicit selected-tensor action and never eagerly scans the model. A worker scan is not interrupted by switching views, but obsolete queued tile clients are discarded before computing. Existing cache eviction remains 2 GiB / 1,000 PNG files. Comparison calibration is a separate checksum-wrapped file under the same exclusive cache lock. Reopening with A/B reversed rejects the previous calibration and cannot hit its tile keys.

Data responses use `pair`, `pair_id`, `name`, and native coordinates. They do not emit the single-source `tensor` inspection/edit-target object. JSON data/calibration responses carry the comparison identity extension; PNG responses carry equivalent `X-Atlas-Comparison-Identity`, `X-Atlas-Source-A-Identity`, `X-Atlas-Source-B-Identity`, coordinate-space and non-editable headers. Error responses retain the existing error JSON shape. Static files are ordinary assets. Optional `comparison_identity` query values must match the opened pair; the frontend includes it in every data/tile request after loading metadata.

Integration must retain the dedicated route prefix and comparison coordinate discriminator. Never convert a comparison response into an inference selection, even when a displayed quantity is raw A or raw B. The dedicated comparison server exposes no inference route or single-source API, and its HTML loads no inference controller. Comparison addresses remain separate from the single-source inference contract.
