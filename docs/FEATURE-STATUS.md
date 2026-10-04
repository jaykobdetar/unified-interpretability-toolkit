# Feature status

This table describes implemented scope. Passing fixture/contract CI is not exhaustive real-model or browser qualification.

| Area | Available | Limits |
| --- | --- | --- |
| Weight viewer | BF16/F16/F32, exact original inspection, local/global rules, explicit slices, bounded exports | Linux; unsupported storage stays unavailable; complete calibration required for the corresponding rules. |
| Checkpoint comparison | Matching names/shapes, A/B and derived differences, linear/magnitude/fixed-asinh mappings | No inference edit handoff; no paired global/percentile/robust mapping. |
| CPU inference | Pinned complete SmolLM2-135M, RAM edits, paired generation, activation captures | Separate runtime; one bounded disposable model worker; no Qwen inference/GPU/general-model support. |
| Sweeps and prompt pairs | Explicit plans, coverage/restoration labels, bounded records and redacted export | Partial/time-limited results do not imply complete coverage; changed-prefix logits are not matched-context causal effects. |
| Regional analytics | BF16 bounded means, extremes, seeded shuffle and explicit small SVD | At most 65,536 values per job, SVD at most 64×64; not a full-model statistical interpretation. |
| Fixture profiles | Explicit source/slice/seed-bound Start, paged paired row/column results, owner reconciliation/restart/reset | Separate owner-prepared fixture host; partial coverage stays labeled; no automatic scan or general model acquisition. |
| Logging | Opt-in bounded records, separate prompt consent, explicit export/import, browser persistence and optional selected file journal | Browser storage can be evicted; filesystem journal requires browser support and explicit file selection; no automatic restore/execution; review archives before sharing. |
| Standalone resources | Validated finite resource configuration, bounded scoped rendering, explicit coarse overview, source-bound tile URLs | Defaults and hosted/comparison policies retain separate guards. Native numerical/cache checks passed on the accepted source lane; browser cache retention/source switching, target multicore performance and remote-filesystem behavior remain unqualified. |
| Model descriptors and owner acquisition | Configuration-bound Qwen candidate description; explicit owner plan, full pins, byte/hash admission, no-replace installation and disabled registry publication | Acquisition uses cooperative deadlines; strict limits require an external supervisor. Accepted source contracts and the separate bounded Qwen acquisition trial do not establish general picker eligibility, topology/runtime fit or inference readiness. |

The integrated export/archive source lane has bounded synthetic browser qualification for typed exports and detached journal reload/import mechanics. Active journal reload, native file permissions and actual inference/archive integration remain unqualified. The worked Qwen key-normalization example verifies a bounded source read; the proposed 128×128 SVD summary is documentation only. The implemented SVD remains limited to 64×64.

Qwen BF16 runtime qualification failed its baseline oracle in a separate trial. GPU/resident runtime, diagnostics, connection reuse and larger SVD implementations are excluded from this integration; no GPU readiness is claimed. Lane-level native/browser evidence is inherited qualification, not a rerun against the combined tree. Integrated native, smoke and browser checks require separate qualification.

Default CI covers offline Rust build/test/clippy, syntax, deterministic contracts, and synthetic CLI smoke. Optional real-model, numerical-reference, and browser harnesses remain separate. Long uninterrupted workflows, physical mobile devices, arbitrary checkpoints, low-memory endurance, GPU support and public internet service operation are not claimed. Historical machine-specific qualification receipts are kept privately outside the release tree.
