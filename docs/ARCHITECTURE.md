# Architecture

Unified Interpretability Toolkit has independent local entrypoints. The viewer is the default; optional coordinators add separate ownership and compute lifetimes.

| Layer | Location | Responsibility |
| --- | --- | --- |
| Native reader and renderer | `src/` | Validate safetensors structure; stream bounded source bands; calibrate exact statistics; transform/pool/render tiles; inspect original bytes. |
| Loopback transport | `src/server.rs` | Fixed embedded assets and bounded HTTP queues/deadlines, Host/Origin checks, one numeric worker. |
| Browser viewer | `web/app.js`, `web/atlas-tools.js` | Source-bound selection, synchronized OpenSeadragon panels, legends, exact inspection, exports and navigation state. |
| Checkpoint comparison | `src/comparison.rs`, `web/comparison.js` | Ordered A/B identity, shared original scales, separately labeled differences; no inference edit targets. |
| Optional inference | `tools/live_inference.py`, `tools/inference_worker.py` | Loopback coordinator and one disposable pinned CPU model worker; explicit owner lease, admission, cancellation and cleanup. |
| Optional regional analytics | `tools/analytics/` | Bounded BF16 windows, seeded controls, explicit small SVD; serialized against inference. |
| Experimental local host/profile | `tools/atlas_host/`, `tools/profile_atlas.py` | Owner registry, fixture selection, bounded whole-slice profiles, sealed snapshots, source leases and worker ownership. |

## Source and cache boundary

Model files are opened read-only. Safetensors headers/indexes establish supported extents before payload access. Source identity and explicit slice binding travel with results; stale responses cannot replace a newer selection. Unsupported dtypes/shapes remain unavailable catalog entries. Original scalar bytes stay distinct from derived transforms and differences.

Calibration scans complete original tensors. Opening a checkpoint reads headers, not all payloads. Tensor rules become ready after that tensor's calibration; global rules need the whole supported checkpoint. Source changes invalidate cached results. Cache paths must be outside model trees; cache locking prevents concurrent mutation. Local checksums detect inconsistent cache content but are not authentication. Explicit full hashing is available through the native `verify` command.

## Resource and ownership boundary

Native operations use one CPU, a 768 MiB address-space limit, bounded source bands, bounded output/cache budgets, and memory/disk admission checks. Coordinators preserve their own lower work budgets and stop reserves. A caller never acquires another session's worker; unknown cleanup remains blocking. Only owned children are terminated/reaped. Logging/export requires explicit user actions.

The services bind only to loopback. There is no hosted authentication/deployment architecture or general remote model browser. See [API semantics](API-PROGRESSIVE.md), [data/slice binding](DATA-VIEWER-CONTRACT.md), and [profile worker interface](profile-worker/PROFILE-WORKER-INTERFACE.md).

## Experimental owner fixture workflow

The owner registry uses `config/atlas-host.example.json`; its paths resolve relative to that config. `tools/atlas-model.py` supports local validation, registration, enabling/disabling, and owner receipts. It does not download models. Owner receipts include source paths and should stay local. The HTTP visitor catalog exposes opaque model identities instead.

```bash
python3 tools/atlas-model.py --config config/atlas-host.example.json validate-config
python3 tools/atlas-model.py --config config/atlas-host.example.json plan \
  --manifest fixtures/tiny-bf16/host-manifest.json
```

Registration and owner preparation are explicit operations. `tools/host_atlas.py` serves the fixture picker; `tools/profile_atlas.py --help` describes the profile entrypoint requiring the exact native binary SHA-256. Neither automatically makes arbitrary local models eligible. Default resource gates remain in force. Profile scans are explicit and bounded; partial coverage is labeled and not treated as representative sampling or full completion.

Current regression tests cover the source/ownership contracts using tiny fixtures and doubles. Prior bounded workflows are not proof of long-session, arbitrary-model, physical-mobile, fault-injection, or GPU behavior.
