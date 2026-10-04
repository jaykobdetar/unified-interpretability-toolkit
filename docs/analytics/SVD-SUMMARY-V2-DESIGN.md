# 128 × 128 SVD summary contract and qualification boundary

Status: implemented with portable fixtures; [retained qualification](SVD-QUALIFICATION.md) covers four fixed synthetic 128-window numeric cases and one ordinary standalone HTTP/browser workflow at the exact recorded runtime. The dense 64 × 64 implementation remains unchanged. Four CPU seconds, five wall seconds and 768 MiB address-space limits remain authoritative. Public-model and broader runtime claims require separate source bindings, review and assigned capacity.

## Admission and lifecycle

The explicit request scope `svd_summary` uses the existing `/api/analytics/start` lifecycle. Keep its sole owned job, no queue, cancellation/reaping, expiry, source verification and coordinator exclusion rules. Do not create a separate worker pool, retry, automatic continuation, model scan or seed search. Existing `scope: region` with `svd: true` retains the legacy dense 64-window contract.

The request is closed: `scope`, native `tensor`, `region` (`row`, `col`, `rows`, `cols`) and unsigned 32-bit `seed`. Each axis is 1–128 and the selected population is at most 16,384. Existing native vector/matrix and original BF16 admission applies; this does not add higher-rank or dtype admission. Preserve the existing source-header/fingerprint checks before and after reads. Source reads remain at most 32,768 BF16 bytes for the window, below the existing region cap. The request does not accept client paths, raw arrays, layout profiles, runtime choices or control data.

Compute this standalone summary directly from the admitted window. Do not first construct the general analytics report, folded matrices or dense JSON residuals. Keep the current algorithm until separately versioned/qualified: uncentered F64 NumPy/LAPACK, exact selected-window SVD, then a separate fit of its shuffled control. Import NumPy only inside the currently guarded child. Array-size estimates do not establish startup/workspace/CPU fit.

## Reduced response

Use schema `weight-atlas.svd-window-summary.v2` with explicit native shape/region, source and model/revision binding, algorithm version and coverage. Its `source_binding` matches the viewer v2 binding shape: source/model identities, native tensor/name/dtype/shape/rows/columns, empty leading slice indices and native display axes. Higher-rank slices are refused rather than silently substituted. The renderer model identity must match the domain-separated source/revision digest; a bounded revision assertion is checked but not returned as text. Reject oversized identifiers; do not send roots or paths.

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

The largest response has at most 1,056 finite F64 scalar tokens, 256 preview mapping integers plus 256 explicit window positions, fixed geometry/counts, bounded identities and fixed labels. A conservative 26 bytes per float token, seven per mapping/window-position integer and 16 KiB for all remaining structure/metadata stays below 48 KiB. Enforce a summary-specific serialized-body limit of `64 KiB - 2 KiB`, leaving the existing envelope reserve; the current worker's roughly 2 MiB outer cap is unchanged. Refuse a larger serialization instead of truncating or falling back silently. A separately retained pure worst-case JSON sizing check is design evidence, not numerical or runtime qualification.

The production worker input remains source-bound metadata under its existing 512 KiB limit. Keep the legacy CLI runner capped at 64: its 200,000-character decimal-array input cannot safely admit every 128-window. The standalone HTTP coordinator uses source-bound worker metadata. Any future CLI summary adapter would need its own bounded raw-byte encoding and independent validation; raising its input cap is outside this contract.

A reduced response controls JSON/Python-list overhead, not LAPACK computation. Dense F64 matrix, residual and factor arrays are still needed inside the child. Current memory/disk reserve checks, one CPU/BLAS thread, four CPU seconds, five wall seconds, 768 MiB AS, output caps and stop/reap rules remain authoritative. A refusal is not a zero spectrum or partial success.

## Coherent implementation and remaining qualification

The implementation versions the new response/cache identity and publishes independent legacy dense and new summary limits. The UI shows full-window spectrum versus preview-only residual coverage, labels shuffled positions and mapped native addresses, and retains original/control separate fitting and honest refusal messages. Do not have the legacy dense residual plot assume the preview is the full window. Further edits to worker/service/UI contracts still require coordinated ownership.

The retained known-spectrum diagonal, zero, rank-one and dense signed offset cases used seed 77 and their original guards. A public source window remains a separate qualification. A possible window is the already documented Qwen `q_proj.weight`, rows `[128,256)`, columns `[0,128)`, bound to the current pinned source before starting. Each further run needs assigned capacity, independent source-byte reconstruction, permutation bijection/digest/multiset checks, singular/Frobenius/residual-energy invariants, exact preview address mapping, original/control separate fitting, finite serialized output, time/CPU/RSS/AS receipts and confirmed cleanup. Degenerate bases/signs use invariant comparisons. Stop at the first gate failure; source/output feasibility does not authorize a heavier job or a relaxed budget. There is no automatic retry or implicit sequence.

## Frozen source and first bounded qualification

Portable CI includes pure closed-schema/body-budget, source/slice/revision binding, independently spelled permutation, signed-zero multiset, analytic diagonal/rank-one/noise/zero doubles, verified raw-source reads, owned admission/reaping, stale-result rejection, and separate UI-mode fixtures. These do not establish NumPy/LAPACK startup, runtime fit, browser rendering, or public-model spectrum correctness.

`tests/analytics/qualify_svd_summary.py --prepare-only --evidence-dir EVIDENCE_DIR` prepares one synthetic BF16 diagonal matrix `diag(128,…,1)` and seed 77 outside the checkout. It imports no NumPy and launches no process. Run mode requires an assigned parent slot and explicit existing `--renderer` and `--python`; it admits one actual source-bound job through `AnalyticsJobs`, observes the kernel/BLAS limits and source identity, checks the closed coordinator output/binding and cleanup, then launches one separately capped oracle. The oracle checks the original spectrum analytically and the shuffled spectrum/residual preview through a symmetric Gram eigensystem with an independently spelled shuffle. Each numeric child stays at one CPU, 768 MiB address space, four CPU seconds and five wall seconds; there are exactly two sequential numeric children, no retry.

The existing Rust CLI provides native source metadata for this synthetic fixture. Its revision binding is explicitly constructed for the synthetic revision because CLI metadata does not emit a renderer state revision. This helper qualifies the actual owned analytics pipeline if successful; it does not itself qualify live HTTP metadata or browser UI. The separately retained [standalone HTTP/browser attempt](SVD-QUALIFICATION.md) checks live renderer revision identity and the actual UI for the dense offset fixture. Higher-rank slicing, all 128-window inputs, registry/fixture-host admission and the proposed public Qwen window remain outside that qualification.
