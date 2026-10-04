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
| Logging | Opt-in tab-memory records, separate prompt consent, explicit download | No automatic durable history; review any export before sharing. |

Default CI covers offline Rust build/test/clippy, syntax, deterministic contracts, and synthetic CLI smoke. Optional real-model, numerical-reference, and browser harnesses remain separate. Long uninterrupted workflows, physical mobile devices, arbitrary checkpoints, low-memory endurance, GPU support and public internet service operation are not claimed. Historical machine-specific qualification receipts are kept privately outside the release tree.
