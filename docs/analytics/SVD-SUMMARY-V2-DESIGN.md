# Proposed 128 × 128 SVD summary contract

Status: source-only design. The shipped dense SVD remains capped at 64 × 64. No 128-window decomposition, model analysis or runtime qualification has been performed. This proposal bounds output structurally; the unchanged four CPU seconds, five wall seconds and 768 MiB address-space limits still need an actual granted attempt.

## Admission and lifecycle

Add one explicit request scope, `svd_summary`, to the existing `/api/analytics/start` lifecycle. Keep its sole owned job, no queue, cancellation/reaping, expiry, source verification and coordinator exclusion rules. Do not create a separate worker pool, retry, automatic continuation, model scan or seed search. Existing `scope: region` with `svd: true` retains the legacy dense 64-window contract.

The proposed request is closed: `scope`, native `tensor`, `region` (`row`, `col`, `rows`, `cols`) and unsigned 32-bit `seed`. Each axis is 1–128 and the selected population is at most 16,384. Existing native vector/matrix and original BF16 admission applies; this does not add higher-rank or dtype admission. Preserve the existing source-header/fingerprint checks before and after reads. Source reads remain at most 32,768 BF16 bytes for the window, below the existing region cap. The request does not accept client paths, raw arrays, layout profiles, runtime choices or control data.

Compute this standalone summary directly from the admitted window. Do not first construct the general analytics report, folded matrices or dense JSON residuals. Keep the current algorithm until separately versioned/qualified: uncentered F64 NumPy/LAPACK, exact selected-window SVD, then a separate fit of its shuffled control. Import NumPy only inside the currently guarded child. Array-size estimates do not establish startup/workspace/CPU fit.

## Reduced response

Use schema `weight-atlas.svd-window-summary.v2` with explicit native shape/region, source and model/revision binding, algorithm version and coverage. Reject oversized identifiers; do not send roots or paths.

| Field | Bound and meaning |
| --- | --- |
| `coverage` | Original tensor count; exactly visited window count; covered fraction; full-tensor boolean. A partial window has no whole-matrix spectrum claim. |
| `results.original`, `results.shuffled` | At most 128 singular values and 128 energy fractions each; full-window Frobenius energy, rank-one residual energy/fraction and zero-energy flag. Undefined zero-energy ratios remain null. |
| `rank_one_residual_preview` on each result | Fixed top-left window-position rectangle with axes at most 16, at most 256 finite residual values, its explicit position origin/shape, and displayed/omitted counts. Energy is computed over the full selected window, never only this preview. |
| `control` | Exact same-window multiset permutation, declared seed, effective zero-seed alias, algorithm version, population and SHA-256 of the complete permutation encoded as unsigned little-endian 32-bit indices. |
| `control.preview_position_to_source` | At most 256 source-local indices for the displayed shuffled preview, preserving its source-coordinate mapping. Omitted mapping count is explicit. The complete order is reproducible from algorithm/population/seed and its digest, rather than silently truncated. |

Use the existing xorshift32, rejection-sampled Fisher–Yates permutation with a versioned definition. Declared seed zero uses effective seed `0x6D2B79F5`; report this documented alias instead of implying distinct permutations for those two seeds. Reorder the exact raw-value multiset, including signed zero, before fitting the control. Residual arithmetic need not preserve source signed-zero bits. Never independently fit the control on a different sample, reuse the original fitted basis, or treat shuffle as a no-effect or significance null.

For a 128 × 128 window, the residual preview shows 256 of 16,384 window positions (1.5625%), with 16,128 omitted per side. Its singular spectrum and residual energy still describe the entire selected window. On a 4,096 × 4,096 source matrix, that selected window covers only 0.09765625% of source values. Several windows remain separate fits; they cannot be combined into a whole-matrix spectrum or whole-matrix shuffle.

## Byte and resource gates

The largest response has at most 1,056 finite F64 scalar tokens, 256 preview mapping integers, fixed geometry/counts, bounded identities and fixed labels. A conservative 26 bytes per float token, seven per mapping integer and 16 KiB for all remaining structure/metadata stays below 48 KiB. Enforce a summary-specific serialized-body limit of `64 KiB - 2 KiB`, leaving the existing envelope reserve; the current worker's roughly 2 MiB outer cap is unchanged. Refuse a larger serialization instead of truncating or falling back silently. A separately retained pure worst-case JSON sizing check is design evidence, not numerical or runtime qualification.

The hosted worker input remains source-bound metadata under its existing 512 KiB limit. Keep the standalone legacy runner capped at 64: its 200,000-character decimal-array input cannot safely admit every 128-window. Any future standalone summary adapter would need its own bounded raw-byte encoding and independent validation; raising its input cap is not part of this proposal.

A reduced response controls JSON/Python-list overhead, not LAPACK computation. Dense F64 matrix, residual and factor arrays are still needed inside the child. Current memory/disk reserve checks, one CPU/BLAS thread, four CPU seconds, five wall seconds, 768 MiB AS, output caps and stop/reap rules remain authoritative. A refusal is not a zero spectrum or partial success.

## Required coherent change and first qualification

Before enabling the scope, coordinate ownership of `worker.py`, `svd.py`, `service.py` and the analytics UI with the parent. Version the new response/cache identity; publish independent legacy dense and new summary limits. The UI must show full-window spectrum versus preview-only residual coverage, label shuffled positions and mapped native addresses, and retain original/control side-by-side fitting and honest refusal messages. Do not have the legacy dense residual plot assume the preview is the full window.

A proposed complete qualification contains at most five independent requests: zero, known-spectrum diagonal, rank-one, signed noise and one public source window, each with predeclared seed 77 and its original guards. The first granted request should be the known-spectrum synthetic case; subsequent cases require their own assigned capacity. There is no automatic retry or implicit sequence. A possible public window is the already documented Qwen `q_proj.weight`, rows `[128,256)`, columns `[0,128)`, bound to the current pinned source before starting. Each run needs independent source-byte reconstruction, permutation bijection/digest/multiset checks, singular/Frobenius/residual-energy invariants, exact preview address mapping, original/control separate fitting, finite serialized output, time/CPU/RSS/AS receipts and confirmed cleanup. Degenerate bases/signs use invariant comparisons. Stop at the first gate failure; source/output feasibility does not authorize a heavier job or a relaxed budget.
