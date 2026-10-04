# Fixture picker and progress contract

The fixture host connects coordinator handlers and viewer controls. It is limited
to owner-registered copies of the checked-in 244-byte BF16 fixture, identified by
its exact expected SHA-256. No arbitrary model activation, inference, analytics,
profiles, downloads, resident worker, keepalive or increased caps are enabled.
The new optional launcher is `tools/host_atlas.py`; existing launchers retain
their behavior. See architecture for the separate fixture-profile entrypoint.

| Route | Contract |
|---|---|
| `GET /api/models` | Read-only saved catalog plus fixture eligibility/reader status. Does not start a process, hash tensor payloads or grant authority. |
| `POST /api/view-contexts` | `{model_id}`; optionally current `{context_id,capability}` for an atomic owned switch. One reader, <=4 tab leases, 15-second lease. Same-model tabs share reader. Different model returns 409 `reader_busy` while another tab holds a lease. Starting a reader returns 202; no automatic restart on failure. |
| `POST /api/view-contexts/{context}/heartbeat` | `{capability}`; renew only a current unexpired lease. No computation grant. |
| `POST /api/view-contexts/{context}/release` | `{capability}`; release only caller's lease. Last release starts owned renderer cleanup; uncertain reap retains the slot. |
| `GET /api/models/{model_id}/{model,view,inspect,tile,progress,tensor-status}?context=...` | Explicit current model/reader generation. Closed read allowlist, bounded query fields. Context is routing identity, not a secret. Other methods/routes are refused. No arbitrary backend forwarding. |

Lease capabilities are only in POST bodies and tab memory, never query URLs,
storage, bookmarks or visitor status. Catalog/selection cannot write registry,
choose a filesystem path/upstream port, enable inference, or request calibration.
All responses retain no-store. HTTP ownership covers tabs in the existing local
trust model, not public hosting or hostile same-user local processes.

Activation resolves the owner registry internally, checks enabled status, exact
fixture manifest and current fingerprints, then hashes just the known 244 bytes.
The root's complete renderer-interpreted safetensors/index inventory must match.
Only then can the coordinator start its one owned Rust renderer with existing
guards. Initial Rust metadata must match root, pinned revision, tensor inventory
and source bytes before a visitor receives a path-free projection. Context/source
identity changes invalidate stale requests. No ready flag is copied into registry
receipts. No head/inference descriptor is invented for this synthetic fixture.

Rust `/api/progress` and `/api/tensor-status?tensor=N` call the same small state
projection, never `State::model()`. Response is <=16 KiB and contains source/model
identities, global readiness, aggregate coverage and optionally one tensor's
calibration statistics. It reports only completed scan counts, not estimated
in-flight progress. Native status schemas remain distinct from private profile
ProgressStore records. The basic picker has no profile routes; the separate profile-host entrypoint supplies them.

The viewer uses context URLs for model/view/inspect/tile/export requests. It
rejects responses from an old context and replaces whole-catalog polling with
small status/selected-tensor updates. In fixture host mode, selection does not
auto-calibrate and calibration controls are hidden. Owner-only CLI preparation
must populate the fixture cache before opening if color views are wanted; raw
inspection remains available without it. Registry/cache paths and ports remain
owner configuration, not browser input. A busy switch leaves the previous view
correctly identified; successful switch clears stale selections and requests.

Pure tests cover the route handlers, activation/reap/lease boundaries with fake
processes, source correspondence, no catalog side-effect, no payload paths,
cross-tab busy, stale context and small status forwarding, plus the actual viewer
client's model URL and stale-response behavior. These pure checks do not substitute for guarded runtime or browser qualification.
