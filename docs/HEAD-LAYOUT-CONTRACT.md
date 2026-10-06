# Head-layout descriptor v1

Source implementation: `tools/inference_architecture.py`. `head_layout_descriptor()` returns the pinned, configuration-derived JSON shown in `docs/models/smollm2-head-layout-v1.json`. `describe(config)` validates dimensions for this adapter without reading model files or granting inference support. Neither function imports numerical libraries. Other architectures and ambiguous head dimensions fail closed.

`GET /api/inference` adds `head_layout` (configuration evidence only) and `architecture` (compact dimensions/capture sites). The worker's owner-private `loaded` event includes a separate `head_layout` with `evidence: loaded_builtin_layout` and `runtime_verified: true` only after loaded parameter, alias, class and layout checks. Public metadata must never promote a prior session's evidence to the next model or session. Runtime layout validation does not prove disk immutability, resource fit, numerical correctness or pristine restoration by itself.

Descriptor fields:

- `schema: weight-atlas-head-layout-v1`, `adapter_id: builtin-llama-eager`, `adapter_version: 1`.
- `source_model`: exact repository/revision, weight digest and configuration digest. Inference edit requests keep their existing **three-field** source identity; the descriptor's additional config hash is not an edit-request field.
- `width`, `layers`, `query_heads`, `kv_heads`, `head_dim`, `queries_per_kv`, `vocab_size`, `intermediate_size`, `capture_sites`.
- `native_weight_layout: [output_feature,input_feature]`; each projection mapping gives exact shape, selected axis, head count/kind and dimension. The native interval is `[h*d,(h+1)*d)`. No transpose or inferred reshape is allowed.
- Q rows and O columns index query heads; K/V rows index KV heads. Query `h` uses KV `floor(h/queries_per_kv)`. KV `k` serves queries `[k*groups,(k+1)*groups)`. Zeroing O columns removes one query-head output contribution and leaves shared K/V intact. Zeroing Q rows changes queries; it generally leaves a nonzero value mixture.
- `evidence`, `runtime_verified` and `inference_support` explicitly separate configuration layout claims from runtime validation/admission. The only execution allowlist remains the existing fully pinned SmolLM2 model, CPU FP32, built-in Transformers 4.56.2 Llama eager. A viewer supporting another dtype/model does not imply inference support.

Viewer/host consumers must bind this descriptor to the same source receipt and native tensor name/shape, and only annotate the listed exact rank-two projection matrices at native layer indices. Tensor names resembling these projections are insufficient. Unknown source, architecture, version, matrix shape, axis layout, noninteger GQA grouping or missing head dimensions mean **unavailable**, not a best-guess head mapping. An explicit `head_dim` may differ from hidden width/query-head count: the O input extent is `query_heads * head_dim`.

The viewer slice binding fixes leading axes and presents trailing matrix axes. This descriptor authorizes only canonical empty slices for rank-two stored weights; higher-rank slice bindings remain view-only and cannot enter current edit requests. Host and viewer own receipt/slice binding v2; this lane adds no competing source-binding schema, model registry route, downloader or progress transport. A future host adapter can wrap the existing closed validated request with its `{model_id,receipt_digest,adapter_id/version,job_budget}` envelope; the current worker still accepts the legacy pinned request and coordinator-owned deadline only. Do not send the future envelope directly to this worker or accept visitor-provided model paths, devices or budgets.

## Separate viewer receipt adapter revision

`bind_viewer_head_layout(model_info, trusted_binding=None)` now constructs the viewer consumer fields on the `/api/model` object:

```
model.head_layout = <weight-atlas-head-layout-v1 pinned_configuration descriptor>
model.head_layout_binding = {
  source_identity, model_identity, weights_sha256, config_sha256
}
```

The four binding fields must be lowercase SHA-256 strings. Source/model identities must exactly equal the current renderer metadata; weight/config hashes and revision must exactly match the pinned descriptor. Comparison coordinate spaces are refused. Unknown, absent, mismatched or stale bindings remove both annotation fields; an upstream descriptor is never forwarded as authority. The returned binding is detached from caller-owned data. This produces configuration evidence only and never forwards a prior job's loaded-runtime evidence.

The **host**, not this pure projection, validates the installed receipt, current file fingerprints and correspondence of the receipt root/content to the renderer's source/model identities. Passing an arbitrary dictionary does not authenticate a receipt. The callback is internal only: no HTTP body, metadata supplied by a visitor, path/revision shortcut, current inference capability, or source filename can create a trusted binding. The host `configuration_evidence` service is correspondence evidence only; its `runtime_verified`, `fit_verified` and `inference_ready` remain false. No duplicate registry verifier is added here.

The single-model coordinator keeps internal `Session.head_layout_binding` and `Session.head_layout_receipt` fields, both default **None**. Direct Session construction does no file I/O. Standalone startup reuses the complete pinned-file verifier and retains a private canonical-root/regular-file fingerprint receipt only when the before/after fingerprints match. For analytics-only startup, an exact bounded pinned configuration gates optional verification: at most 384 MiB of manifest files, 1 MiB hash chunks, five seconds and the existing 3.25 GiB memory reserve. Unknown, oversized, missing, symlinked or unverifiable sources remain usable for analytics without annotation. Required inference startup retains its original full pin failures. No receipt is accepted from HTTP.

At `/api/model` projection the receipt must still match every current file fingerprint, the owned native renderer must report the receipt root and pinned revision, and its model identity must equal the existing hash of source identity/revision. Only then are the pinned configuration/weight hashes bound to that renderer metadata. No request-time weight hashing or model loading occurs. Changed files, comparison spaces and mismatched identities remain unbound. An internal host adapter can still install an independently validated four-field binding when no standalone receipt is present; an invalid standalone receipt cannot fall back to that binding. Inference-source binding continues separately, so labels with `inference_enabled` false do not grant inference or edit access. No registration or activation policy is changed. Other architectures remain unavailable to this Llama-only consumer.

The canonical `source_binding` v2 remains unchanged: `version`, `model_identity`, `source_identity`, `tensor`, `name`, `dtype`, complete native `shape`, `rows`, `cols`, and `slice:{leading_indices,display_axes}`. Rank-two annotations require `leading_indices:[]`, `display_axes:[0,1]`; descriptor mappings still authorize only exact stored rank-two projection matrices. No `head_layout` fields enter worker edit requests, and no extra source/config hash is appended to the legacy three-field edit identity.

Tests: `tests/head_layout_binding_contracts.py` (four pure tests) checks exact bindings, strict hashes/keys, stale identities/revisions, removal of forged/runtime annotations, comparison refusal, detached outputs, no read/model work during projection, and viewer/inference capability independence. Retained combined and session-boundary suites also pass. The standalone receipt contracts in `tests/head_layout_receipt_contracts.py` additionally exercise full hash failures, fingerprint replacement, resource budgets, optional-source refusal and native metadata correspondence. Real pinned CPU/browser qualification is an explicit gate for changes to this coordinator; held fixtures are never re-recorded to accommodate a change.
