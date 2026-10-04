# Bounded SVD summary qualification

The runtime at commit `80c252de62827668621846a6569820d258b453a5` (tree `b6cffb5da7862eb61cc28a9bb13d1d77e71b8821`) passed four fixed synthetic numerical cases and one ordinary standalone HTTP/browser workflow. Later documentation changes do not change that runtime. The [summary contract](SVD-SUMMARY-V2-DESIGN.md) adds explicit `svd_summary` windows up to 128 × 128; the legacy dense mode remains capped at 64 × 64.

## Numerical cases

Each case used native BF16 bytes, seed 77, one owned production analytics worker and a separately bounded independent oracle. Original and shuffled windows were fitted separately using NumPy 2.4.2. Analytic invariants and a symmetric Gram eigensystem checked complete spectra, Frobenius and residual energies, and all 256 residual-preview values per side. Independent source reconstruction, raw-bit multiset/permutation checks, exact preview mappings, closed source/model/slice/region bindings, body caps and cleanup passed. Zero-energy fractions stayed null; comparisons did not require arbitrary singular-vector signs or degenerate bases to match.

| Fixed source/window | Tensor coverage | Result body bytes |
| --- | --- | --- |
| `diag(128,…,1)`, complete 128 × 128 tensor | 100% | 21,602 |
| Alternating positive/negative zero, complete 128 × 128 tensor | 100% | 10,763 |
| Signed rank-one matrix, complete 128 × 128 tensor | 100% | 25,470 |
| Fixed dense signed 256 × 256 tensor; rows [64,192), columns [32,160) | 25% | 25,989 |

These are four separately assigned first attempts, with no retries or seed search. Each numeric child kept one CPU/BLAS thread, 768 MiB address space, four CPU seconds and the existing five-second wall policy. Retained guards report complete owned-process cleanup. They establish these cases, not arbitrary inputs, universal performance or deadline-expiry behavior.

## Standalone HTTP and browser workflow

One guarded production `--analytics-only` coordinator and real Rust renderer served the dense offset fixture above. Live HTTP source/model/revision and native slice binding, current loaded module/viewer/index/CSS hashes, and the literal serialized result body were checked. Both full spectra and every residual-preview value matched the previously qualified dense result exactly; no oracle rerun was needed. The unchanged embedded viewer binary was reused because all Rust and embedded bundle parts were unchanged; the two changed analytics modules were served and hash-checked through the production coordinator.

Chromium 153.0.8010.12 with Playwright 1.62.1 exercised exactly two starts: one normal completion and one ordinary active owned cancel after admission. Terminal cleanup recorded no live worker or pending cleanup, ordinary reselection cleared the report, both ports closed and all observed owned processes were gone. There was no completion race, third start, inference mutation, adapter or guard relaxation. The entire guarded startup/browser/cleanup sequence took 7.94 seconds and reached 915.60 MiB sampled summed RSS under the 1 GiB guard; these are workflow measurements, not isolated SVD timings. The result body was 25,989 bytes under 63,488, and its envelope was 26,166 bytes under 65,536.

Desktop and 390 × 844 width-emulation screenshots were inspected. Coverage/cautions, paired spectra, signed residual previews and visible mappings were readable; DOM checks covered all 128 spectrum bars and all 256 mappings and found no horizontal overflow. The long mobile select label truncates. Two desktop captures are identical and show only the first seven expanded mapping entries. Viewports do not show every vertically scrollable chart or entry simultaneously. The short cancelled worker's PID/resource limits were not separately sampled; its live admission and reaped terminal were observed. No physical-device, fault or transient resource-maximum claim follows.

## Evidence and remaining scope

Private receipts and raw owned-job capabilities remain outside the checkout. The retained evidence manifests are identified by SHA-256:

- Diagonal numeric case: `ce394c242f5ab955e37b7b56fc04fdeabc85b0d2055c98ca68cfd88edb1ad410`.
- Three additional numeric cases: `12192aab03833a7bb67747468959369d0fb3e01abd6eeff5b2091af9b7e04dcc`.
- Standalone HTTP/browser: `d394ed08bcf75f75616a325062fa417c4d5500fb8c5ec7581d8dab60540d1361`.

Registry/fixture-host analytics admission remains unavailable. Public-model windows, vector/rectangular runtime numerics, whole-model spectra, statistical significance, failure/network/resource probes and physical devices remain unqualified. A 128-window spectrum covers the complete selected window; its 16 × 16 residual preview covers only 1.5625% of that window, with 16,128 omitted positions per side. Independent window fits do not combine into a whole-matrix spectrum. Any further runtime needs its own bounded binding and assigned capacity; this qualification schedules no continuation.
