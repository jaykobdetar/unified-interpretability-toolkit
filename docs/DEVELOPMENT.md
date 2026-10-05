# Development and testing

Use Linux, Rust 1.92.0 with clippy, Python 3.12 or newer, and Node 22. The viewer and portable checks need no pip/npm packages. Rust crates are pinned in `Cargo.lock` and vendored under `vendor/`; keep offline builds and original dependency licenses/checksums intact. No new Rust dependency is required for this release pass.

## CI recipe

Run these sequentially from the checkout:

```bash
python3 tools/check.py lint
python3 tools/check.py contracts
python3 tools/guarded-build.py clippy --all-targets -- -D warnings
python3 tools/guarded-build.py test --release -- --test-threads=1
python3 tools/guarded-build.py build --release
python3 tools/smoke.py
./run-atlas.sh --demo --check
```

`lint` AST-parses first-party Python, syntax-checks first-party JavaScript with Node, and checks the launcher shell syntax. It is not a Python style or type checker. Clippy treats Rust warnings as errors. `contracts` lists its selected Python/Node files and runs every declared scenario for the asynchronous UI tests. Tests use mocks and tiny synthetic fixtures; they do not launch browsers or load trained models. The smoke test independently checks 26 exact source values and native/pooled PNG outputs across seven rules; other rules and dtype/slice behavior have dedicated Rust/contract coverage.

The inference public-response snapshot drives complete ordinary HTTP requests through the production handler with an in-memory transport. It pins idle metadata, polling without an active owner, and the 32-token acceptance boundary followed by an unavailable-model refusal, including exact body bytes, status and all application headers. A fixed admission-capacity double makes the boundary deterministic; live guards are unchanged and the portable test forbids sockets and model children. Each scenario is repeated and must match exactly. Only the standard library's wall-clock `Date` and Python-runtime `Server` headers are excluded from exact snapshots; their UTC-date and fixed server/runtime format are still checked. Application values and wording are preserved. This complements real-model acceptance rather than qualifying model execution or every service route.

The GitHub workflow uses pinned official action revisions, read-only repository permissions, serial steps, and no deployment, secrets, or `pull_request_target`. Toolchain setup may access official tool distribution services; Cargo builds remain offline. On disposable GitHub-hosted runners, CI removes the unused preinstalled Android and .NET SDKs to recover build disk space. This step does not run on self-hosted machines. A runner that still fails memory/disk guards fails CI rather than weakening the guards.

## Optional qualification

Numerical references such as `tests/parity.py`, `tests/data_view_reference.py`, and `tests/comparison_reference.py` need separately installed reference packages and an explicitly selected interpreter. Browser harnesses require a separately installed Playwright and sandbox-enabled Chromium; set `NODE_PATH` and `ATLAS_CHROMIUM` to your local installation. Real inference additionally needs the complete pinned model and compatible CPU packages. These are outside default CI.

The core browser harness uses the viewer's A/B selector at mobile widths. It checks that each selected panel is visible and paints, that the inactive panel is hidden, and captures both panels sequentially without changing the production layout.

Combined API acceptance checks the exact eight renderer rule IDs in their public order, including `tensor_magnitude_asinh`. Its pure contract cases reject missing, extra, duplicated, renamed or reordered IDs; historical seven-rule acceptance does not describe the current renderer contract.

Run one heavy build/browser/model workload at a time. Existing guards retain one CPU, build 2 GiB address space, browser-specific aggregate caps, workload time limits, memory stop reserves, and the 25 GiB free-disk reserve. Do not extend a job or replace an uncertain owner automatically. Every harness must stop/reap its own processes and preserve failed/partial outcomes honestly. [Acceptance harness instructions](../tests/acceptance/README.md) describe optional model cases and their limits.

Local model tests use `ATLAS_SMOL_MODEL`, `ATLAS_QWEN_MODEL`, and `ATLAS_CPU_PYTHON`; no machine paths are built in. The numeric analytics harness always includes the synthetic fixture and explicitly reports which optional models were not selected. The ownership harness requires `ATLAS_SMOL_MODEL`. Historical head-ablation phase-B reuse requires an explicitly supplied `ATLAS_PHASE_A_EVIDENCE` archive with its original file hashes; it is not needed by the ordinary combined workflow or CI.

## Contributor privacy

Keep model files, local config, prompts, caches, credentials, and machine receipts out of commits. Example paths should be generic placeholders. Inspect actual screenshot pixels and metadata before adding images. The README captures use only `fixtures/tiny-bf16`; the two PNGs should be recaptured after visible UI changes. Never substitute a mockup for a screenshot of the running app.

Historical qualification records are maintained in a verified private archive outside the release checkout. Removing them from a tree does not sanitize Git history. Public publication should use the reviewed clean snapshot in the separate repository, with a deliberate public author identity; the original repository remains private. Do not copy `.git`, old refs, private archives, or local receipts into the release.
