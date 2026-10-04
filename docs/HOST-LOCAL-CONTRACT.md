# Owner registry contract

This reference describes local-owner registry and manifest services. Runtime integration and supported fixture scope are summarized in [architecture](ARCHITECTURE.md).

## Ownership and routes

`tools/atlas_host/` owns configuration, registry, receipt identities, bounded progress, and response-cache policy. Source/slice math lives in Rust and profile primitives; inference adapters retain their separate pinned model boundary. Neither a local-action header nor an inference capability grants registry administration.

The fixture-host HTTP routes expose opaque model catalogs, view contexts, bounded progress and source-bound views. The separate profile host adds owner-scoped JSON POST actions at `/api/profiles/{action}`. See `profile_http.py` and `profile_api.py` for closed action schemas. Registry registration/enabling and owner receipts stay in the CLI. There is no public filesystem browsing, downloader, or general remote model administration.

## Schemas and behavior

Canonical JSON is UTF-8, sorted keys, compact separators, finite numbers only;
duplicate object keys are rejected by file loaders. Version is integer `1`.
Unknown fields fail closed. The executable validators live in
`tools/atlas_host/{config,registry,progress,cache}.py`; the portable example is
`config/atlas-host.example.json`. Relative machine paths resolve relative to the
config file, not the caller's working directory. Paths never enter visitor DTOs.
Configuration loading does not create directories or change process limits.

Owner registry storage is a bounded JSON file with atomic replacement and an
exclusive local writer lock. CLI `register` reads a supplied exact manifest and
existing files only. It does not resolve repository names on a network. Manifest:
`{version,repository,revision,license:{id,accepted,file},provenance,files:[{name,bytes,sha256}]}`.
`revision` must be a full 40-hex commit or, only for synthetic fixture provenance,
`fixture-v1`. Names are flat safe names; allowed data is safetensors, its standard
index, bounded config/tokenizer JSON, README/LICENSE/NOTICE. No code or pickle.
Hashes are streamed within an explicit total byte/time allowance; source changes
during verification refuse registration. Registry storage must be outside the
source directory. A digest receipt records expected and measured matching hashes;
provenance distinguishes owner-supplied expectations from synthetic fixture data.
This is not proof of upstream authenticity, license approval or tensor validity.
Owner-expected manifests must include a hashed LICENSE/LICENSE.txt named by
`license.file`; synthetic fixtures alone may use null and the repository license.
License acceptance must already be explicit in the owner manifest. Downloads and
license acceptance are never inferred by the CLI.

Registry `model_id` is `m_` plus a domain-separated canonical manifest digest.
Content identities exclude local root/inode and acceptance time, but include repo,
revision, sorted file hashes/sizes and license ID. Owner-visible receipt retains
source fingerprints for cheap stale-file refusal on listing; it does not claim a
fresh rehash on each list. Catalog status is `verified_pending_renderer`, never
ready for inference. Publishing/enabling a registry entry only affects local
picker visibility; it does not publish a website or repository. Unregister/delete
of model data is absent from this stage.

Progress input: `{model_id,kind,binding,state,visited_values,total_values,
elapsed_active_ms,remaining_authorized_work:{values,wall_ms,cpu_ms},complete,error}`. Kind is
`calibration|profile|overview`; state is `queued|running|partial|complete|cancelled|error`.
Public errors are fixed reason codes only. Binding uses the exact versioned
`SOURCE-BINDING-V2.json`, retained as `tests/fixtures/host-source-binding-v2.json`:
`{version:2,model_identity,source_identity,tensor,name,dtype,shape,rows,cols,
slice:{leading_indices,display_axes}}`. Shape is original/native; display uses
trailing axes only. Rank one has empty leading indices and axes `[0]`; rank two
has empty leading indices and axes `[0,1]`. Higher ranks require every leading
index. No byte origins/counts/permutation hashes belong in this binding. The host
validates the exact schema rather than passing its own projection to siblings.
Calibration progress counts the complete original tensor regardless of the
view's selected slice; profile/overview progress counts the selected slice. A
binding supplied to these trusted services is not authorization to read a source.

Each store has one random epoch, monotonic revision and at most 64 latest records;
it is a bounded latest-state delta, not a complete event log. Cursor is
`{epoch,revision,scope}`; wrong epoch/scope, too old or future revision requests reset.
Snapshots are <=16 KiB and <=16 records, with `has_more` and a cursor that never
skips an undelivered matching record. Authoritative records remain server-owned;
callers receive copies. Profile progress is private by default: only explicit
capability authorization returns it. Public model readiness never reveals private
job IDs or records. Active private records cannot be evicted by public progress;
capacity refuses if all slots are reserved. One fixed store binds to one
`model_id` so tabs cannot mix models. This service reports progress; it never
admits, advances or extends work.

Task15 core pages use `axis=rows|columns,start,count<=1024` and return paired
original/control values. The future HTTP adapter preserves paired values; no
separate field request may cause different coverage snapshots. Its
`swap-or-not-8-v1` control is an exact bijection/multiset control, not a claim of
uniform sampling or a significance null. `WorkGrant` accounts one total wall
allowance across waits/chunks and requires an explicit CPU clock covering all
owned work; no parent-only CPU measurement may omit Rust/child work. It passes an
absolute deadline into each chunk and refuses continuation after uncertain
outcomes. The 5-second wall/4-second CPU ceilings are conservative local adapter
ceilings, not changes to inference limits. It is a cooperative ledger, not an OS
interrupt or allocation guard: runtime integration must retain independent
enforcement, global admission and lease ownership. No extension API is enabled.

Task16 compatibility lives in `tools/atlas_host/inference.py`.
`configuration_evidence()` checks exact weights/config hashes and full revision,
but never promotes them to runtime verification/fit. `head_layout_metadata()`
returns `head_layout` plus
`head_layout_binding={source_identity,model_identity,weights_sha256,config_sha256}`
only for a trusted same-source descriptor and matching rank-two native attention
projection with canonical empty slice and validated GQA dimensions. Missing or
mismatched descriptors mean unavailable; no Qwen/hardcoded fallback. Loaded
runtime evidence is session-owned and cannot enter registry metadata. The current
worker's closed request and three-field edit source identity remain unchanged;
neither host envelope nor descriptor config hash is forwarded as a worker field.

Derivation identity binds the content digest, renderer/algorithm version,
calibration digest, exact native name/shape/dtype/slice, rule and finite parameters,
level/x/y, control algorithm/seed and output encoding. Local fingerprint-based
source/model identities are validated but excluded from the portable derivation
hash; its content digest supplies that binding. Original whole-tensor
calibration and selected-slice artifacts remain distinct. No profiles persist yet.
Immutable response policy is `private, max-age=31536000, immutable` plus strong
ETag; a requested digest must match payload bytes. All other response classes use
`no-store`. No automatic `public` cache policy and no immutable job/result data.
Browser retention cannot guarantee revocation; hosted access policy is later work.
