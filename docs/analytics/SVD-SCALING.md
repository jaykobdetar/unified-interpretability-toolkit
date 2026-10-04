# Larger SVD windows: assessment and pending qualification

The current implementation supports an explicitly selected native window of at most 64 × 64 / 4,096 values. It computes an uncentered exact NumPy LAPACK SVD for the original window and a seeded exact-multiset shuffled window, fitting them separately. This assessment does not increase the shipped cap or qualify larger windows.

A larger selected-window SVD is a practical first candidate before a streaming approximate whole-matrix solver. A 128 × 128 window has 16,384 values. For a 4,096 × 4,096 matrix, this covers 1/1,024 of the source values (0.09765625%); it remains a window result. A 256 × 256 window covers 65,536 values (the existing maximum region read), but two dense rank-one residual arrays plus the source-index control map can exceed the analytics worker's 2 MiB response cap. Raising just the SVD axis constants would not establish a usable workflow.

The unchanged worker constraints also cover source checking, bounded reads, analysis, NumPy startup, both decompositions and JSON serialization: one allowed CPU, 768 MiB address space, four CPU seconds, five wall seconds, finite request/output bounds and source identity rechecking. Larger CPU/RAM on a host does not override those budgets. Arithmetic array sizes alone omit NumPy/BLAS mappings, decomposition workspace, Python float objects, JSON encoding and simultaneous coordinator buffers.

## Bounded candidate

First qualify 128 × 128 exact window SVD, keeping the current guards. Use an explicit region/seed selected before inspecting numerical outcomes. Retain source geometry and visited/total counts. Preserve a Fisher–Yates permutation of that window's exact raw multiset, including signed zero, and fit the control independently. Shared visualization scales do not imply a shared fitted basis. Use one declared seed initially; do not search seeds until the control appears weak.

Measure wall/CPU/RSS and serialized byte counts on zero, diagonal/known-spectrum, rank-one, signed noise and a source-bound public model window. An independent numerical oracle must reconstruct native source bytes, verify the permutation as a bijection, compare singular values, conserve Frobenius energy, and check residual energy against `1 - leading_energy_fraction`. Degenerate singular values do not guarantee identical singular-vector signs/bases; compare invariant quantities and the reconstruction/residual appropriately. Zero energy keeps undefined ratios null. Refusal or time/output limits are outcomes, not numerical zero or success.

Only after those results fit the guards should a coherent implementation update the runner, hosted analytics worker, advertised limits, exclusion messages, UI contract, cache/algorithm version and fixtures together. A capped residual preview with native addresses and explicit omitted coverage could allow a larger decomposition without returning every residual cell, but it would change the response contract. Do not silently truncate a dense residual while presenting it as the whole window.

The [reduced summary contract proposal](SVD-SUMMARY-V2-DESIGN.md) defines a separate explicit 128-window scope with a capped residual preview and reproducible control mapping. It is source-only: output feasibility is checked, while numerical, runtime and UI qualification remain pending. The shipped dense contract is unchanged.

## Whole-matrix and sampled scope

Several disjoint windows do not produce the singular spectrum of their parent matrix: cross-window row/column interactions are missing. Report each window independently, the explicit selection policy, overlap/deduplicated visited counts and covered fraction. A whole-matrix shuffled null would require a permutation of the whole matrix's values; shuffling each sampled window tests only its local multiset. Neither is a significance test or evidence of learned function.

A future randomized/streaming solver would need explicit rank, oversampling, iteration/pass counts, seeds, read/time/memory/output budgets, failure status and a residual/error qualification. Its approximate spectrum and estimated coverage must be labeled. Repeated passes may exceed full-model I/O allowances even when RAM fits. This lane adds no implicit scan, automatic continuation, new Rust linear algebra dependency, GPU worker or source acquisition.

The next granted qualification is one bounded 128 × 128 synthetic/reference attempt and one preselected public-model window, with no retry or cap relaxation after failure. The current 64 × 64 contract stays authoritative until the larger implementation and actual workflow pass together.
