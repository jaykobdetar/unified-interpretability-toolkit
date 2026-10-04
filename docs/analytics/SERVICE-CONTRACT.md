# Production analytics ownership contract

Analytics attaches to the existing loopback Python coordinator. It is a separate
short-lived analysis worker, never an inference session or a model execution.
No additional listener, HTTP thread, background queue, dependency install, or
Rust dependency. The parent HTTP coordinator owns at most one analysis child.

- `GET /api/analytics`: public capability-free limits/busy metadata only.
- `POST /api/analytics/start`: `{tensor: integer, region: {row,col,rows,cols}, seed,
  svd: false}`, `{scope: "model", seed}` or the closed
  [summary request](SVD-SUMMARY-V2-DESIGN.md); returns a newly generated independent
  `job` capability and queued/running status. No client filesystem paths/catalog.
- `POST /api/analytics/poll`: `{job}`; same-owner status/result only.
- `POST /api/analytics/cancel`: `{job}`; kill/reap only the owned child; retain
  ownership/admission if reap is uncertain. No replacement before successful reap.
- Existing Host, Origin, Sec-Fetch-Site, body/header bounds and X-Atlas-Local checks
  run before analytics routing. Assets use an explicit local allowlist.
- One admission slot, zero pending jobs; busy returns 409/503, never implicit
  cancellation of another tab. Inference admission checks analysis ownership;
  analysis admission checks inference ownership including stopping/cleanup states.
  Pause/replay of inference is not permission to run analytics concurrently.
- Analytics start reads validated Rust `/api/model` metadata on the coordinator's
  trusted fixed upstream; model root comes only from launch arguments. Catalog
  byte extents are resolved from bounded local headers in the worker and checked
  against that validated catalog. No HTTP model-path or source selection.
- Worker input <= 512 KiB; output <= 2 MiB; one region <=65,536 values; model-wide
  <=65,536 values TOTAL with exact skipped/partial coverage. No cache persisted.
  Dense SVD <=64x64 /4,096; explicit `svd_summary` <=128x128 /16,384 with
  <=16x16 residual preview, <=63,488-byte result and <=65,536-byte coordinator
  worker-output buffer. Optional existing CPU Python/NumPy stays within the same worker.
- One CPU, <=768 MiB address space, five-second wall lease per job, bounded pipe
  drains per coordinator tick, existing >=3 GiB reserve (plus launch headroom),
  stop on reserve breach. Client lease 15 seconds; completed results bounded and
  owner-private. All shutdown/error/cancel paths retain child ownership to reap.
- UI mount is a separate ES module. A tiny app.js bridge publishes selected tensor
  metadata and supports native viewport jumps. It never reorders the original
  matrix. Explicit user buttons run bounded region/model analysis or optional SVD;
  selection changes clear old output and cancel only that tab's owned analysis.
- Head profiles pin exact local config and official installed implementation
  hashes, reviewed contiguous Linear [out,in] and attention reshape semantics,
  and actual names/shapes. Unknown/mismatched layouts remain unavailable.

Runtime qualification requires separately assigned capacity. The retained
[summary qualification](SVD-QUALIFICATION.md) covers fixed synthetic inputs and
the ordinary standalone analytics-only route; registry/fixture-host analytics
remains unavailable.
