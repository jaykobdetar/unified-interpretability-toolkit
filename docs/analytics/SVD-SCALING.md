# Larger SVD windows: assessment and qualification limits

The legacy dense implementation supports an explicitly selected native window of at most 64 × 64 / 4,096 values. A separate `svd_summary` scope supports at most 128 × 128 / 16,384 values with capped residual previews. Both compute an uncentered exact NumPy LAPACK SVD for the original window and a seeded exact-multiset shuffled window, fitting them separately. [Retained qualification](SVD-QUALIFICATION.md) covers four fixed synthetic summary cases and one standalone HTTP/browser workflow. Public-model and arbitrary-input qualification remain separate.

A larger selected-window SVD is a practical first candidate before a streaming approximate whole-matrix solver. A 128 × 128 window has 16,384 values. For a 4,096 × 4,096 matrix, this covers 1/1,024 of the source values (0.09765625%); it remains a window result. A 256 × 256 window covers 65,536 values (the existing maximum region read), but two dense rank-one residual arrays plus the source-index control map can exceed the analytics worker's 2 MiB response cap. Raising just the SVD axis constants would not establish a usable workflow.

The unchanged worker constraints also cover source checking, bounded reads, analysis, NumPy startup, both decompositions and JSON serialization: one allowed CPU, 768 MiB address space, four CPU seconds, five wall seconds, finite request/output bounds and source identity rechecking. Larger CPU/RAM on a host does not override those budgets. Arithmetic array sizes alone omit NumPy/BLAS mappings, decomposition workspace, Python float objects, JSON encoding and simultaneous coordinator buffers.

## Bounded summary and further qualification

For any additional 128 × 128 exact window qualification, keep the current guards. Use an explicit region/seed selected before inspecting numerical outcomes. Retain source geometry and visited/total counts. Preserve a Fisher–Yates permutation of that window's exact raw multiset, including signed zero, and fit the control independently. Shared visualization scales do not imply a shared fitted basis. Use one declared seed initially; do not search seeds until the control appears weak.

The retained zero, diagonal, rank-one and dense signed offset cases passed their predeclared gates. A future source-bound public model window still needs wall/CPU/RSS and serialized-byte receipts. An independent numerical oracle must reconstruct native source bytes, verify the permutation as a bijection, compare singular values, conserve Frobenius energy, and check residual energy against `1 - leading_energy_fraction`. Degenerate singular values do not guarantee identical singular-vector signs/bases; compare invariant quantities and the reconstruction/residual appropriately. Zero energy keeps undefined ratios null. Refusal or time/output limits are outcomes, not numerical zero or success.

The separate summary schema coherently updates the production analytics worker, advertised limits, UI, cache/algorithm version and fixtures. Its capped residual preview has native addresses and explicit omitted coverage. It does not raise the legacy dense runner cap or enable registry/fixture-host analytics. Do not silently truncate a dense residual while presenting it as the whole window.

The [reduced summary contract](SVD-SUMMARY-V2-DESIGN.md) defines the explicit 128-window scope with a capped residual preview and reproducible control mapping. Its [qualification record](SVD-QUALIFICATION.md) separates numerical cases from actual standalone HTTP/browser behavior. The dense contract is unchanged.

## Whole-matrix and sampled scope

Several disjoint windows do not produce the singular spectrum of their parent matrix: cross-window row/column interactions are missing. Report each window independently, the explicit selection policy, overlap/deduplicated visited counts and covered fraction. A whole-matrix shuffled null would require a permutation of the whole matrix's values; shuffling each sampled window tests only its local multiset. Neither is a significance test or evidence of learned function.

A future randomized/streaming solver would need explicit rank, oversampling, iteration/pass counts, seeds, read/time/memory/output budgets, failure status and a residual/error qualification. Its approximate spectrum and estimated coverage must be labeled. Repeated passes may exceed full-model I/O allowances even when RAM fits. This lane adds no implicit scan, automatic continuation, new Rust linear algebra dependency, GPU worker or source acquisition.

Further public-model or vector/rectangular runtime qualification requires its own frozen source binding and assigned capacity, with no retry or cap relaxation after failure. No additional job follows automatically from the completed synthetic and standalone workflow qualification.
