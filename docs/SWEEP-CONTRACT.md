# Explicit sweep subset contract

[Head sweeps](HEAD-SWEEPS.md) defines current v2 output-column ablation, query-row interventions, and the single-layer extension. The subset bounds below apply to explicit one/two-target plans.

## One small, explicit plan

A sweep is one owned request to the existing inference coordinator, one disposable CPU worker and one pinned SmolLM2 model. The complete job, including verification/loading, all cases and cleanup initiation, shares a **120-second wall budget and a 90-second admission-plus-worker CPU allowance**. The same 4.75 GiB launch gate, 3.25 GiB stop reserve, 1.5 GiB worker RSS cap, same-origin/local-header checks, session capability, 15-second client lease and bounded queue remain. Work stops at a limit; it is never resubmitted automatically.

The initial supported subset is deliberately small:

| Dimension | Hard maximum / semantics |
| --- | --- |
| Named user targets | 2 explicit query-head or within-head-offset interventions |
| Interventions including controls | 5: one empty edit, then target 1/control 1, optionally target 2/control 2 |
| Prompts | 2 explicitly reviewed strings, each 1–128 tokens and ≤2,048 UTF-8 bytes |
| Prediction positions | 1 next-token score probe per prompt, at its last consumed position; no free-running continuation |
| Cartesian product | 5 interventions × 2 prompts × 1 probe = **10 case/prompt records** maximum |
| Model prefills | One baseline and one edited prefill per record: **20 prefills** maximum; no decode loop |
| Native edits per intervention | Existing maximum 8; zero or finite scale using the existing ordered, half-open native edit contract |
| Distinct selected cells per intervention | 65,536 maximum; capped before allocation or loading |
| Output | At most 10 records, each with the existing one 576-value activation and at most 10 top-union candidates; summaries remain bounded |
| Input / private logs | Existing 8-KiB request and 8-record / 1-MiB private log envelope; one sweep is one log entry |

Only allowlisted `model.layers.L.self_attn.q_proj.weight` matrices are initially eligible. A whole query head is the architecture-verified 64-row native interval `[64h,64(h+1))` within a `[576,576]` matrix. A within-head offset selects one native row, or the same explicit offset in an explicitly named subset of at most eight heads (one row edit per head). The UI must label this as at most **8 of 9 heads**, never as complete head coverage. Two whole heads in a single intervention exceed the selected-cell cap and are refused; the UI can offer them as the two separate target cases. No embedding alias, other projection family, arbitrary tensor, Qwen inference source, or implicit all-layer expansion is admitted in this first mode.

The plan is fully expanded and reviewable before Run: source/revision/hash, named tensor/shape, native ranges, operation/scale, control role, prompt order, probe position, selected-cell count and capture layer/site. The capture layer/site is fixed for the entire job and is explicitly independent of the layers being edited. Edits and extra attention/lens diagnostics cannot be silently combined. Unknown fields, booleans in numeric fields, duplicate targets, out-of-bounds shapes/positions, invalid scales and oversized plans are rejected. A rejected plan is not silently truncated.

## Matched controls and reproducibility

Each target receives one explicit seeded control with the **same selected-cell count, same tensor, operation and scale**. Whole-head controls use a different randomly selected head-aligned 64-row interval in that same layer. Offset controls use a different randomly selected within-head offset in exactly the same named head subset. This is a random matched-location control, not a shuffled-weight-value operation or a claim of uniform arbitrary-cell sampling. Control overlap with another target/control is rejected and resampled from the finite candidate set; exhaustion refuses the plan. The empty-edit control must report exact baseline parity.

Use a strict integer seed 0–2³²−1 and a versioned deterministic SHA-256-based selector with rejection sampling, not runtime-dependent PRNG state. The selector input includes the version, seed, exact tensor name, target descriptor and control index; canonical input is sorted-key compact UTF-8 JSON for `[version, seed, normalized_target, zero_based_target_index, counter]`; the first four SHA-256 bytes are an unsigned big-endian word. Reject the incomplete modulo bucket and stop after at most 128 draws. Three committed test vectors fix the resulting offsets and full plan digests. The expanded plan, selector version and resolved native edits are recorded so later users can reproduce it without trusting the RNG. Case order is fixed: empty, target 1, its control, target 2, its control; prompt order is retained inside each case. No adaptive search or data-dependent additional cases are allowed.

Matching region size does not match weight magnitudes or guarantee the same number of numerically changed values. Record selected-cell count, actual numerically changed-cell count and Frobenius norm of the applied parameter delta. These quantities make control differences inspectable; they do not establish that controls are interchangeable.

## Original-model restoration is a prerequisite

The initial edit feature isolates sessions by destroying the model worker. A sweep would intentionally reuse its model **inside one owned job**, so it needs a separately verified restoration mechanism before shipping. This is a new correctness obligation, not permission to weaken session isolation.

Before each intervention, snapshot only its deduplicated selected native cells, at most 65,536 FP32 values (256 KiB plus bounded descriptors). Verify native mapping and absence of parameter/storage aliases first. Generate that prompt's baseline on the original model, apply ordered edits, then score the exact same original prompt with a fresh uncached prefill. Restore in `finally` and verify original **bit patterns**, including signed zero, before any subsequent baseline. Generated head/offset cases contain distinct rows. The generic undo helper also deduplicates overlapping row/column/element selections before snapshotting and preserves their existing ordered composition; the known tied embedding/head object restores through its canonical parameter, while unexpected storage aliases are rejected. The sweep planner itself still excludes embeddings. Any restoration failure ends the entire job; it cannot continue with a tainted model. Cancellation/time/resource/error destroys the worker, and no later session can reuse it.

Keep a bounded original next-token logit reference per prompt (at most two vocabulary vectors) and require every later baseline for that prompt to match it exactly under the pinned deterministic runtime. A mismatch aborts before further cases. Restore the last case too and report restoration verified only when it actually completed. Disk files are never saved or changed. This mechanism must pass controlled success/failure/cancellation/fresh-session and independent arithmetic tests before the sweep feature can claim acceptance.

## Fixed-context scores and honest rankings

All case comparisons consume exactly the same original prompt IDs for their baseline and edited scores. Every record is labeled **matched fixed prompt context**, with no later generated-prefix divergence. It returns the top-five union with both raw FP32 logits, their edited-minus-baseline deltas and both argmax IDs. No text continuation is generated; token IDs/pieces describe the first-token predictions only.

Primary summaries, calculated from the full vocabulary transiently before discarding it:

- RMS next-token logit delta over the vocabulary, plus maximum absolute logit delta.
- Total variation distance between the two next-token softmax distributions, computed in binary64 from the FP32 logits; range 0–1. These are model distributions, not calibrated truth probabilities.
- Delta for the **baseline argmax token**, retaining the same token on both sides, and whether the edited argmax changed.

No full-vocabulary score arrays are exported. The top-union table is visibly a candidate subset; full-vocabulary metrics are labeled separately. An optional ranking sorts target cases by the mean RMS delta over only the explicit tested prompts, with every per-prompt value and the matched-control result still available. Missing or interrupted cases are not scored as zero and cannot enter a complete ranking. The label is “sensitivity on this prompt set,” not general head importance, learned semantics, or a whole-model causal map. A small ablation's causal score effect under these fixed inputs does not establish the functional meaning of the head.

## Ownership, partial coverage and no runaway resume

One accepted plan owns one session capability. Start is disabled while active; duplicate start and stale responses cannot create additional plans. Poll and playback remain separate; pausing playback does not pause compute. Cancel/reset affect only that capability. The worker emits one completed record after each case/prompt, and progress states the exact intervention and prompt currently being processed. It never emits a successful record for an unfinished case.

The completion envelope includes ordered intended case IDs, completed IDs, unrun IDs and any interrupted current ID, with explicit `complete`, `cancelled`, `time_limit`, `resource_limit` or `error` status. If the remaining global budget is too small to start another case with a fixed cleanup margin, end with partial coverage. No retry follows a resource refusal, no automatic background continuation occurs, and the UI preserves the refusal reason.

“Run remaining subset” would require a new explicit reviewed request, reuse the exact pinned source and resolved edits, and clearly report that it is a separate job/runtime budget. It must never be an automatic chain of nominally 120-second jobs. Exported checkpoints remain passive data; no imported plan executes itself. If the user requests a larger sweep, offer this bounded subset and disclose its fraction of the requested heads/layers/prompts before execution.

## API and UI

The existing same-origin/local-header guarded `POST /api/inference/sweep-plan` computes a bounded plan without a worker, model load, private session lookup or server-side persistence. The request is a closed object with mode `sweep`, pinned `source_model`, explicit `prompts`, `targets`, `operation` (and `scale` only for scale), integer `seed`, `capture_layer` and `activation_site`. Run uses the existing `/start` action with exactly those fields plus `plan_digest`. The digest binds the fully resolved plan and exact prompt strings; edits, seed, prompt, layer or output changes require a fresh review. The digest is a freshness binding, not an ownership credential.

The UI can replace the explicit subset with the supported inspector's head intervention; Qwen/foreign selections are disabled. Manual target syntax includes `head L H`, `query_head L H`, `offset L H[,H...] O`, or the separate `layer L` extension; see the v2 head contract. Both prompts and seed are visible before review. The expanded plan identifies every target/control native interval and its geometry. The controls are explicitly another contiguous head-aligned range or another offset in the same named heads, not arbitrary scattered-cell edits. The result table lists completed cases independently of playback, while the selected step shows the edited activation, both candidate logits and fixed-context metrics. Unrun cases stay visible and never become zero scores. There is no Run Remaining button, chaining or automatic retry.

Sweep admission starts its wall and process-CPU clocks before validation/verification. Bounded verification checks wall/CPU exhaustion and the unchanged 3.25 GiB reserve before each file/read and after each read/hash chunk. A temporary POSIX real-time alarm checks the same budgets at most every 0.1 seconds, with its last interval shortened to the wall deadline; raising from its handler interrupts a blocked open/read instead of allowing an automatic syscall retry. The timer and previous signal handler are restored on all exits; an existing timer fails closed rather than being replaced. Every pinned hash is still mandatory. The owned worker receives the absolute wall deadline and only `floor(90 − admission process CPU seconds)` of CPU allowance; less than one second refuses spawning. Its RLIMIT_CPU includes its own startup and verification and never raises a lower inherited limit. Worker verification uses the same bounded checkpoints/alarm. The coordinator's unchanged `started > 120` stop remains authoritative. These are mocked-control checks, not a claim that kernel uninterruptible I/O or actual runtime fit has been tested. Between cases the worker stops with partial coverage when fewer than ten seconds remain. A native operation still obeys the outer worker CPU/RSS/wall guards; this does not guarantee every maximum-size plan can finish.

Private logs include the resolved native plan and partial coverage, but omit exact prompts, the prompt-bound plan digest and **all** per-probe source input-token IDs unless prompt inclusion is explicitly selected. Every sweep record is a fresh prefill, so redaction covers every index, not just index zero. Turning consent off removes these fields from earlier history exports as well. No private runtime logs or model values were committed.
