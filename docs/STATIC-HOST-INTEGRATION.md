# Registered static host integration

The static hooks extend the existing fixture host while preserving its ownership and lease protocol. Default dense admission is closed. The explicit owner launcher may supply the separately bound, narrow policy described in [DENSE-STATIC-ADMISSION.md](DENSE-STATIC-ADMISSION.md). The accepted evidence and remaining UI limits are described in [STATIC-QUALIFICATION.md](STATIC-QUALIFICATION.md).

## Private wiring and public API

`FixtureHost` accepts private `static_policy` and `static_admission` hooks. The policy must be the exact `StaticPolicy` type for the same registry. Admission requires that policy and a trusted callable returning exactly `True`. Both hooks default to `None`; public acquire fields cannot supply them.

The default hosted application exposes candidate metadata but has no dense admission callback. Its independent renderer factory remains fixture-only without a bound dense policy. Supplying a mock predicate establishes neither resource reservation nor real-model qualification.

API version 1 adds `static_views_enabled`, `static_view_candidate`, `static_activation_allowed`, `static_view_ready` and the compatibility `view_ready` state. Catalog remains metadata/stat-only. `static_activation_allowed` is a row hint; the client also requires the global permission. Trusted registry classification, rather than that public hint, determines whether an acquire needs the dense operation.

Acquire remains exactly `{model_id, context_id?, capability?}`, with complete current-lease pairs. A lease adds `view_kind: "fixture" | "static"`; static leases have `profiles_enabled:false`. No browser path, download, registration, enablement or calibration command is added. A static model response includes its verified binding and current `host_context` before values become available.

## Lifecycle

1. Validate request shape, current capability, stopping state and reader/tab capacities. Another tab's reader cannot be replaced. A sole current owner may request a switch.
2. Prepare and admit the proposed static source before stopping a valid old reader. Refused prospective candidates preserve that selection.
3. Give the trusted factory the private receipt and external cache. Install the owned handle before initialization. Failures revoke readiness; uncertain stop retains the stopping slot.
4. Observe health through the serial coordinator and bind the complete private native catalog before publishing readiness. Child readiness alone is insufficient. The context must remain selected and have an unexpired lease.
5. Require the binding and current context before numeric/status/tile reads, then recheck after I/O and during final publication. Response sizes, MIME types and native route identities remain bounded and validated.
6. Revoke identities, proofs and cached readiness on close, switch, expiry, disable, source change, observed health loss or failure. Confirmed stop clears ownership; a new context starts with fresh preparation and full binding.
7. Catalog consumes coordinator-published health without calling reader methods, ticking, starting, sampling, hashing or calibrating. Expired leases make readiness false; lifecycle cleanup remains the coordinator's work.

Late responses cannot stop a replacement context. Channel failures and interrupts attempt cleanup of the exact owned reader. Uncertain cleanup blocks replacement. Static reacquisition, including same-reader reuse, repeats the owner admission under the original private metadata operation before issuing a lease.

Static profile-binding requests are refused before native I/O. Static views are not placed in the profile selection cache. The client disables profiles, verifies static metadata before numeric URLs, clears verification after an invalid response, and preserves generation/capability privacy and stale-response rejection. Fixtures keep their existing automatic selection; registered static candidates require explicit selection.

## UI behavior and retained verification

For an uncalibrated hosted tensor, the application exposes the row/column raw-inspection form and explains that color scales need separately reviewed owner preparation. Opening the model does not start a full scan. Raw readiness does not establish tile, color or calibration readiness.

Retained pure Python and fake-DOM/transport tests cover closed defaults, private hooks, correspondence, source changes, lease expiry, refused switches, uncertain cleanup, selected publication and client gating. They use deterministic tiny data and inert reader/transport doubles. Native correspondence and ordinary lifecycle evidence supplement those checks within their exact recorded scopes. An actual browser test of this picker and raw-inspection path remains outstanding.
