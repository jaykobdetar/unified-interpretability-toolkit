# Selected capture and explicit local experiment export

## Capture API

`POST /api/inference/start` adds optional string `activation_site` with exactly `block` (default), `attention`, or `mlp`. Integer `layer` accepts 0–29. Boolean/fractional layers, unknown sites, arrays, and null are rejected. Exactly one site at one layer is selected per request, with the existing 576-value vector × at most 32 output records. No additional forward pass or hook changes computation.

- `block`: decoder block output after both attention and MLP residual additions. For layer 29 this is before the model's final RMSNorm.
- `attention`: self-attention module output after `o_proj`, before the attention residual addition. It is not an attention probability matrix.
- `mlp`: feed-forward module output after `down_proj`, before the MLP residual addition.

Each vector remains the selected site's output at the last consumed input position, predicting the next token. `activation_site`, `activation_kind`, and `layer` are included in trace records and replay metadata; default legacy requests retain block semantics. The output adapter only observes detached tensors. Hooks are always removed in `finally`.

`GET /api/inference.limits` advertises allowed capture layers/sites. Worker loaded metadata records actual torch/Transformers/tokenizers/safetensors/Python versions plus platform/CPU execution/dtype/determinism provenance; it contains no filesystem paths. Comparisons use the same capture site for both branches, displaying the edited branch's vector as before. Existing pinned model, one worker/model, numerical determinism, token/trace/deadline/RAM caps and ownership controls are unchanged.

Independent model qualification will compare attention/MLP captures against differences across their residual additions, block output against hidden-state references, and generated tokens/logits across capture sites. The full post-addition output and the standalone component are not interchangeable; any subtraction oracle has a floating-point tolerance.

## Private session logging and export contract

An unchecked **Enable private experiment logging** control opts into collecting each subsequently accepted run in this tab's memory. Each run retains the logging/prompt-consent choices at acceptance. Completion, cancellation, resource/time limits and errors become distinct log statuses; duplicate/stale responses cannot append duplicate records. Reset during a run records a cancelled result with only the last received tab snapshot, explicitly labeled. Lost transport without confirmed cleanup is recorded as `connection_lost` with `worker_cleanup_confirmed: false`; it is never labeled complete.

An unchecked **Include prompt text and prompt token IDs** checkbox separately controls prompt inclusion. Without it, request prompt/IDs and the prefill input-token ID are omitted and `request_replayable` is false. Generated output may still reveal prompt content and is explicitly private. Turning the checkbox off also redacts prompt fields from file downloads; stored memory records retain their original consent until explicitly cleared.

The session envelope is `weight-atlas-session-log-v1`, capped at **8 runs and 1 MiB of UTF-8 JSON**. Append is atomic and never drops older records. A record that cannot fit remains visibly pending (one additional at-most-1-MiB record), and new runs are disabled. The user can export the pending run and saved log, then explicitly use **Clear log + discard pending record**. Export never silently clears records. A nonfinite/invalid/oversized single run produces an explicit error and no file.

**Download current/pending experiment JSON** and **Download session log JSON** create browser Blob downloads only on explicit clicks. Opt-in browser persistence and an explicitly selected, supported browser file journal are described in [Experiment archives](EXPERIMENT-ARCHIVES.md). Defaults remain in-memory; no server log endpoint, URL state or upload is added. Reloading never automatically restores archives, opts in or resumes computation. Browser storage is distinct from filesystem durability; file handles are memory-only. Manual imports append validated historical records without execution or worker ownership.

Per-run schema `weight-atlas-experiment-v1` includes pinned model identity, exact ordered edits, optionally prompt text/IDs, generated tokens/text, bounded per-step top-union logits/deltas/alignment, selected capture/site semantics and vectors, seed/greedy/FP32 settings, actual runtime versions, status/completeness, confirmed-cleanup flag, source limits and timestamp. Session capability, internal file paths, process IDs and unrelated metadata are excluded by explicit field selection. Incomplete records retain available provenance. The 1-MiB cap applies to each file export; the existing trace remains 32 × 576 values and at most ten score candidates per step.

When prompt inclusion was selected, a file contains enough request inputs for manual reconstruction, subject to the same pinned files/runtime and numerical portability limits. The application does not import/execute log files. No real private logs or local evidence are committed; only code and named synthetic fixtures may enter source control.
