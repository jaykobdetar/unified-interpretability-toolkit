# Pending bounded observation acceptance

These are **unrun real-workload harnesses** added on `tests/observation-acceptance` from reviewed runtime `dbb2f314f392178332f72e2a25a0d05e1b3df08e`. Runtime files are unchanged. Only syntax checks, pure support fixtures and fake owner-client tests were run during preparation; core owns the heavy slot.

## Cases

- `prompt-pair-browser.cjs`: two valid tokenizer-only previews (the second after ordinary prompt editing invalidates the first), one layer-7 attention-output comparison at two explicit positions, actual redacted download, independent binary64 difference/norm/cosine checks, token/prefix labels, step/replay/reset and mobile emulation. Both prompts are public synthetic fixtures. No text-generation claim. Pin hashes are checked before/after.
- `sweep-maximum-browser.cjs`: ordinary reviewed plan/start via the real coordinator UI. Exactly the fixed fixture's two targets, two prompts, five interventions, ten records and twenty prefills maximum. It records observed effects, exact coverage/restoration flags, top-union arithmetic, the empty-control parity and redacted download. `PASS` requires all ten records and terminal cleanup within the existing budget. Partial/time/resource/cancel coverage returns exit 3; server errors/assertion failures return exit 1. No retry/resume or completion padding. A guard interruption leaves an atomic bounded `progress.json`; that last observation is not a complete qualification. The short-prompt maximum-record fixture does not qualify 128-token prompts or full model/head coverage.
- `sweep-owner-acceptance.py`: three **explicit sequential** ordinary jobs: a one-token empty-edit control, the fixed sweep followed promptly by normal owner cancellation, then a fresh one-token empty-edit control. Replacement is forbidden until terminal non-pending cleanup. Both controls require exact IDs/text/union logits/activation equality. If cancellation races with another terminal result, output is `PARTIAL_CANCELLATION_NOT_OBSERVED`, exit 3, with no retry. It reports the phase actually cancelled; it does not manufacture or claim a deterministic post-edit cancellation. Pin hashes are rechecked after cleanup, including failure paths. Unknown admission ownership is reported unconfirmed; the owning supervisor must stop/reap its coordinator.

All evidence directories must be fresh. Output omits owner capabilities; private source prompts are never taken from an existing tab. These harnesses use only declared synthetic prompts. No malformed request, fault response, server crash, vulnerability reproduction or unrelated process termination occurs.

## Exact pending commands

Only after the parent grants the sole heavy slot and unchanged memory/disk gates pass, run from this candidate worktree. Build sequentially under the existing guard; do not copy a prior embedded-asset binary as qualification:

```bash
python3 tools/guarded-build.py test --release -- --test-threads=1
python3 tools/guarded-build.py clippy --all-targets -- -D warnings
python3 tools/guarded-build.py build --release
```

Start **one owned coordinator**, using fresh free ports. Keep it in an owned execution session; do not reuse the user's viewer or an unrelated coordinator:

```bash
python3 -B tools/live_inference.py \
  --model /path/to/smollm2-135m \
  --python /path/to/cpu-environment/bin/python \
  --port 8816 --atlas-port 8817
```

After that owned process confirms readiness, run the following tests **one at a time**, keeping the coordinator idle between them. Stop on a failed/refused/partial phase and preserve its stdout, exit code and artifacts. Each browser uses the installed sandbox-enabled Chromium, one renderer and the unchanged 768 MiB/120-second browser guard. The application still enforces its existing per-job wall/CPU/RSS/ownership limits. The helper waits at most 105 seconds for terminal state, leaving outer-guard cleanup margin; it never extends a job's deadline.

```bash
export NODE_PATH=/path/to/node_modules
export ATLAS_CHROMIUM=/path/to/chromium
export ATLAS_TEST_URL=http://127.0.0.1:8816
export ATLAS_MODEL_DIR=/path/to/smollm2-135m
mkdir -p results/observation-acceptance-attempt1

ATLAS_EVIDENCE_DIR=results/observation-acceptance-attempt1/prompt-pair \
  python3 tools/guarded-browser-test.py node tests/prompt-pair-browser.cjs \
  > results/observation-acceptance-attempt1/prompt-pair.log 2>&1

ATLAS_EVIDENCE_DIR=results/observation-acceptance-attempt1/sweep-maximum \
  python3 tools/guarded-browser-test.py node tests/sweep-maximum-browser.cjs \
  > results/observation-acceptance-attempt1/sweep-maximum.log 2>&1

python3 tools/guarded-inference-test.py python3 -B tests/sweep-owner-acceptance.py \
  --base http://127.0.0.1:8816 \
  --model /path/to/smollm2-135m \
  --out results/observation-acceptance-attempt1/sweep-owner \
  > results/observation-acceptance-attempt1/sweep-owner.log 2>&1
```

The owner-client guard measures the client itself; the coordinator remains responsible for its owned model's RSS/CPU/deadline enforcement. Record peak model RSS from returned snapshots separately. Do not interpret the client's small RSS as model RSS. A harness exit/guard failure is not proof of worker cleanup: after every phase, and especially any uncertain admission or transport failure, stop the **owned** coordinator through its normal SIGINT/SIGTERM path, wait for its owned children to reap, and verify its ports close before releasing the slot. Never terminate unrelated PIDs. A fresh coordinator can be started explicitly for the next approved phase.

The three existing model-reference commands in the parent qualification plan remain required; browser field checks and exported top-union deltas are not a substitute for independent full-vocabulary attention/lens/prompt-pair/sweep numerical oracles. Static OBS-1 tests verify CPU deduction; real acceptance records the ordinary guarded job's observed outcome, not an independently instrumented coordinator CPU total. Permanent transport loss and kernel-uninterruptible I/O retain their documented limitations.

## Lightweight preparation checks

```bash
node --check tests/prompt-pair-browser.cjs
node --check tests/sweep-maximum-browser.cjs
node --check tests/acceptance/support.cjs
node tests/acceptance-support.cjs
python3 -B tests/acceptance_owner_contracts.py
```

Python harnesses are also AST-parsed without executing their main functions. These checks import no numerical runtime, start no browser/model/listener/child, and take no capacity reading. Synthetic support cases ensure partial coverage cannot become PASS, cleanup-pending false/alive states remain authoritative, metric/privacy failures are detected, replacement is blocked until cleanup and uncertain admission never claims cleanup.
