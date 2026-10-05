# Development and testing

Use Linux, Rust 1.92.0 with clippy, Python 3.12 or newer, and Node 22. The viewer and portable checks need no pip/npm packages. Rust crates are pinned in `Cargo.lock` and vendored under `vendor/`; keep offline builds and original dependency licenses/checksums intact. No new Rust dependency is required for this release pass.

## CI recipe

After installing the pinned optional development tools below, run the same command locally and in CI:

```bash
python3 tools/check.py all
```

`all` requires the exact qualified versions in `dev/versions.json`, checks formatter output, and runs every original syntax/contract/Rust/smoke/launcher check plus the dense regression suite. It also runs ordinary debug `cargo test` and release tests with four libtest threads, while the existing build guard retains one CPU and one Cargo build job. Required tools, wrong versions and failed checks are fatal; nothing is skipped. Existing `python3 tools/check.py lint` and `python3 tools/check.py contracts` remain available without installing formatter or browser packages. `format` checks only formatter output; `--dev-python` selects the isolated pinned Black interpreter.

`lint` AST-parses first-party Python, syntax-checks first-party JavaScript with Node, and checks the launcher shell syntax. It is not a Python style or type checker. Clippy treats Rust warnings as errors. `contracts` lists its selected Python/Node files and runs every declared scenario for the asynchronous UI tests. Tests use mocks and tiny synthetic fixtures; they do not launch browsers or load trained models. The smoke test independently checks 26 exact source values and native/pooled PNG outputs across seven rules; other rules and dtype/slice behavior have dedicated Rust/contract coverage.

The inference public-response snapshot drives complete ordinary HTTP requests through the production handler with an in-memory transport. It pins idle metadata, polling without an active owner, and the 32-token acceptance boundary followed by an unavailable-model refusal, including exact body bytes, status and all application headers. A fixed admission-capacity double makes the boundary deterministic; live guards are unchanged and the portable test forbids sockets and model children. Each scenario is repeated and must match exactly. Only the standard library's wall-clock `Date` and Python-runtime `Server` headers are excluded from exact snapshots; their UTC-date and fixed server/runtime format are still checked. Application values and wording are preserved. This complements real-model acceptance rather than qualifying model execution or every service route.

The analytics public-response snapshot covers idle and inference-busy metadata, busy admission, stale ownership, accepted running work, completed results, replacement after cleanup, and cancellation through the same HTTP handler. Its six-value BF16 file runs the real source adapter and arithmetic. Process/pipe ownership, admission capacity, job entropy and filesystem stat provenance are declared doubles; the stat double retains actual byte length. Source and cache identities are computed normally and pinned, with no response normalization. Exact body bytes, application headers, status and repeated-run equality are required, with the same two validated standard-library header exclusions. It opens no listener or real worker and does not qualify worker peak memory or filesystem identity change detection; existing source and owned-worker tests cover those separately.

The fixture-host public-response snapshot uses a temporary registry with the shipped 26-value fixture. It pins empty, disabled, enabled and active catalog projections; disabled compute routes; unavailable routes; acquisition; selected-model metadata; heartbeat; release; and stale ownership. Registry validation, fixture hashing and the host projection run normally. The renderer handle and its metadata, clock and lease entropy are declared doubles; no renderer process or listener starts. The test never uses an installed owner registry or activation policy. Complete response status, application headers and exact bytes must repeat and match, using the same validated `Date`/`Server` exclusions.

The profile public-response snapshot drives the actual hosted handler, private router, service and snapshot pages through sixteen HTTP scenarios. It covers disabled and owner catalogs, disabled admission, full start/admitting/completion, original/control row and column pages, heartbeat, partial restart, stale ownership, partial rows, pending cancellation, cancelled status and reconciliation. The existing worker/kernel/clock doubles supply a deterministic 12-value frame; the supervision-context admission bridge and entropy are explicit doubles. Status requests never execute work, and final cleanup must leave zero owned fake descriptors and zero charged snapshot bytes. It opens no listener, child or execution thread and activates no installed model/policy. Exact response bytes, status, application headers and repeat equality use the same validated two standard-library header exclusions. Actual worker execution and resource qualification remain separate optional cases.

Four unit tests use the process-global numeric workspace ledger: exclusive admission/release, bounded row rendering, bound overview pixels, and calibration publication barriers. A test-only fixture mutex isolates those tests from each other; it does not wrap production operations or replace the ledger. All original assertions, refusal paths, renderer thread scenarios and publication threads remain. Other tests can run concurrently. The fixture guard recovers its own poisoned lock after an assertion failure without resetting resource accounting. Integration test binaries use a matching local fixture guard for their fourteen calibration/rendering cases. Partial reservations additionally assert exact used-byte/active-slot accounting and exclusive admission while spare capacity remains. Both ordinary Cargo tests with parallel libtest scheduling and explicit multi-thread release tests must pass; the build guard still uses one CPU and one Cargo build job.

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

The optional workspace browser harness restarts its owned fixture with identical files and a different revision. It first requires polling to reject the new identity while keeping the prior revision and notes, then uses the existing Refresh status button before checking the new revision, old-bookmark rejection and separate note scopes. Polling does not adopt a different model identity.

The deterministic frontend tests import fresh factories from `tests/support/ui-fixture.cjs` and `tests/support/inference-fixture.cjs`. Each invocation owns its DOM, VM, transport queues and timers. Scenarios do not extract another test’s setup by source-text delimiters; their assertions remain independent of that test’s formatting. The app factory accepts an explicit source path for disposable correctness-mutation trials.

`tests/ui-direct-validation.cjs` checks both invalid legend identities and fractional row/column addresses directly. These assertions queue no viewer or transport wait, so a pending async operation cannot produce a false successful process exit. The frontend correctness-fault set retains the eight original catches and adds coverage for these two former misses.

The optional exact CPU protocol net uses ten fixed public synthetic scenarios in `tests/fixtures/cpu-protocols`: block/attention/MLP capture, attention observation, logit lens, tokenizer preview, empty/changed edits, prompt-pair activations and a bounded three-record sweep. Their held complete JSONL records come from unchanged starting production; the test has no recording mode. Each scenario runs in two fresh verified offline workers through the existing inference guard. Runtime versions, model/source identity, events, key order, numerical spelling, tokens, activations, scores and metrics must match exactly. Only explicitly listed elapsed-clock paths vary; their presence, numeric type, finiteness and nonnegativity remain required. Wire whitespace and duplicate keys are checked rather than normalized. The portable integrity contracts run without model packages.

Run the opt-in net from a Linux x86_64 checkout using the pinned CPU runtime recorded in the fixture manifest:

```bash
python3 -B tests/cpu_protocol_reference.py --python "$ATLAS_CPU_PYTHON" \
  --model "$ATLAS_SMOL_MODEL" --output "$EVIDENCE/cpu-protocols"
```

The evidence directory must be new and outside the model directory. `--case` selects one named scenario for diagnosis; the default runs all ten. Retain requests, raw stdout/stderr/exits, commands, exact-result hashes and lifecycle receipts. The existing direct-worker 1.5 GiB RSS cap, one CPU, 120-second wall limit, worker 3 GiB address-space/90-second CPU caps and memory admission/stop reserves remain. Sweep exec retains the guard-observed worker PID and original total-job deadline. Interrupt cleanup reuses the existing bounded PID/start-time guard cleanup and records verified survivors/errors. No inference server, model registration or activation policy is needed.

The profile bundle export, host-client function boundaries and comparison mobile-media guard use whitespace-tolerant syntax matching. The original assertions remain, including disabled profile scope, exact native status, absent comparison inference routes and the 720px breakpoint. Ten benign disposable correctness mutations were caught by both the original and rewritten tests; all three rewritten tests also pass on pinned Prettier output.

Optional browser gap checks use the existing resource guard. `tests/held-link-browser.cjs` cold-loads the held pre-format v2 fixture and checks native geometry, both rules, region, exact re-save spelling and reload. The fixture preserves the historical zero and observed starting re-save signed zero separately. `tests/archive_browser_driver.py --model MODEL --python CPU_PYTHON` runs `tests/archive-browser.cjs` through the actual coordinator: file import, lazy codec bytes, default redaction, save/reload/explicit restore, unchanged inputs and zero model starts. Model registration is unchanged; head-layout binding and native file-dialog permissions are separate qualifications.

## Pinned optional development tools

The formatter and browser development dependencies are isolated from runtime dependencies. `dev/versions.json` records the qualified tool versions. Install the hash-pinned Black wheels in a local CPython 3.12 virtual environment, and install the npm lockfile without lifecycle scripts or browser downloads:

```bash
python3 -m venv qualification/dev/black-26.1.0
qualification/dev/black-26.1.0/bin/python -m pip install --require-hashes -r dev/requirements-format.txt
PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 npm --prefix dev ci --ignore-scripts
```

The supplied Black wheel hashes target Linux x86_64 with CPython 3.12. The npm packages use official registry tarballs and lockfile integrity hashes. Playwright still requires an explicitly selected, separately installed Chromium for browser qualification. Neither installation changes the offline vendored Rust build or the existing no-install syntax and contract checks.

Mechanical formatting covers first-party Python under `tools/` and `tests/`, Rust selected by `cargo fmt --all`, and first-party JavaScript/CSS under `web/` and `tests/`. The immutable supplied behaviour lock, vendored code, JSON snapshots and HTML are outside that formatter scope. These exclusions preserve supplied and held bytes; they do not disable any existing project check.

Existing syntax checks, contracts and Clippy warning enforcement remain required. Additional Python lint rules, unused-import removal, type-checking rules and structural cleanup are deferred to the separately reviewed Part 2; no new lint/type rule has been run and then suppressed in this pass.

Prettier 3.9.6 reaches stable output after two passes for three chain-heavy test files. The initial safety trial also reported an AST serialization-order difference for `inference.js`; parsed structure, values and array order remained identical. Qualification retained that first failure, checked semantic AST equality in disposable copies, and required stable formatter output. The normal `format` check reports any later formatting drift.

The 28 portable CJS frontend/profile/host/workspace async drivers require their original main promise to complete. A tiny test-only helper checks Node's existing `beforeExit` event: an unfinished main fails even when a pending Promise has no active event-loop handle. It tracks only that main, preserves every original expression/assertion, and adds no timer or production behavior. Ten benign dropped-settlement cases demonstrate the prior premature successful exit and the new failure; resolved, deferred and rejected controls retain their outcomes. ESM analytics tests keep top-level await, whose unfinished module evaluation already fails. Existing per-test timeout and resource guards remain.

Pure guard simulations also check admission at the exact 25 GiB disk boundary and a bounded delayed cleanup after the kill fallback. Every process, signal, clock, memory/disk query and receipt I/O is mocked. Ten disposable policy/reporting faults were probed; these two cases close the prior misses while preserving all original test methods and every live guard limit. They perform no real process signaling or resource pressure.
