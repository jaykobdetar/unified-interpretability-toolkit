# Experimental local inference

See [head sweeps](HEAD-SWEEPS.md) for output-column ablation and query-row intervention semantics.

Weight Atlas includes bounded SmolLM2-135M generation and activation playback. The viewer is a Rust weight renderer; the inference path uses an isolated CPU-only PyTorch/Transformers process behind a Python standard-library loopback coordinator. No Rust inference crate was added: the project builds offline from its existing vendored dependencies, and the CPU runtime is installed separately.

## Run

The [launch guide](LAUNCH.md) is the primary mode/setup reference, including the static-launcher/Generate distinction and exact qualification boundaries.

Build the viewer as usual, then explicitly select an existing Python environment and model directory:

```bash
python3 tools/guarded-build.py build --release
python3 tools/live_inference.py \
  --model /absolute/path/to/smollm2-135m \
  --python /absolute/path/to/cpu-environment/bin/python
# Open http://127.0.0.1:8796
# Ctrl-C stops the coordinator and both owned children.
```

The coordinator starts its own Rust renderer on port 8797. Both bind only 127.0.0.1. Ports 8774, 8775, and 8785 are reserved and rejected; occupied ports fail rather than replacing an existing process. Use `--port` and `--atlas-port` for other fresh ports. No inference starts until **Run comparison locally** is pressed. The ordinary Rust launcher still works; it identifies the session as viewer-only and hides unavailable inference controls.

Tested runtime: Python 3.12, `torch==2.8.0+cpu`, `transformers==4.56.2`, `tokenizers==0.22.2`, `safetensors==0.8.0`, `numpy==2.5.3`. The supplied requirement constraints record these direct versions. This prototype does not install packages, create an environment, or claim an independently reproduced transitive dependency lock. The caller must select a trusted, compatible existing CPU environment.

The `python3` launching the coordinator is separate from the ML interpreter selected by `--python`. The coordinator uses only the standard library. Pinned-file verification streams SHA-256 in 1 MiB blocks and does not depend on Python 3.11's `hashlib.file_digest`; every file is still checked in full. Python 3.12 is the tested coordinator version; no older-interpreter execution is claimed.

The model directory must contain regular local files `model.safetensors`, `config.json`, `tokenizer.json`, and `tokenizer_config.json` from **HuggingFaceTB/SmolLM2-135M**, revision **93efa2f097d58c2a74874c7e644dbc9b0cee75a2**, Apache-2.0. See the [pinned official tree](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/93efa2f097d58c2a74874c7e644dbc9b0cee75a2) and [expected hashes](models/smollm2-135m.json). All four SHA-256 values are checked before startup and each session. The full weights are about 269 MB on disk. No files are downloaded by the runtime. Do not use the earlier selected-tensor visualization subset; it is not a complete model. Do not point this inference launcher at an 8B model.

The model is loaded through the built-in Llama implementation with local-only safetensors, explicit FP32 CPU execution, eager attention, greedy argmax, seed 0, and deterministic algorithms. There is no remote Python, pickle loading, `trust_remote_code`, accelerator execution, paid API, training or telemetry. Private experiment collection is off by default and requires explicit browser-session opt-in; durable files require an explicit download. Prompt text travels in a bounded localhost POST body and a local stdin pipe, never a URL or command argument.

## What the controls mean

- **Run comparison locally** computes baseline and edited branches sequentially, each up to the selected token limit, within the same original total session deadline. A fresh process reloads the model each time; startup is separate from forward-pass timing.
- **Pause/Resume playback**, **Single step**, and the rate control change how computed token records are displayed. They do not slow or pause inference computation. Step mode starts with no record displayed and advances one record per click.
- **Cancel compute** signals only this session's worker. Cancellation is complete only after reaping; if cleanup is delayed, `stopping` remains active and blocks a new session while polling continues. Already received records remain available as a recorded session.
- **Rewind recorded trace** replays the bounded in-memory records without running the model. Its label explicitly says “Recorded replay.”
- **Reset / clear** cancels work and clears the trace after reaping. A delayed cleanup returns an explicit retry error and retains the owner capability and records. The tab stays in `stopping`, blocks Start, and continues status polling even through transient polling errors; Reset can be retried. A rejected new generation request also preserves the previously accepted session and trace. The editable prompt stays available for another run after cleanup. Reloading the page drops its local association; an abandoned active worker expires after 15 seconds without owner polling.

The base model has no chat template and is not instruction tuned. Correct execution does not imply useful or factual completions. The default example continues “the capital of the…” rather than answering a geography question.

Each record observes one chosen site at the **last consumed input position**, for one chosen layer (0–29). Block capture is after both residual additions; attention capture is after `o_proj` but before its residual addition; MLP capture is after `down_proj` but before its residual addition. Attention output here is a 576-value projected vector, not an attention probability matrix. Step zero is the prompt prefill: its activation predicts the first generated token. Later records consume the previous generated token. Position is zero based in the full prompt-plus-generated sequence. The predicted token's own activation is only available when it is consumed on the following step. The layer-29 capture is before final RMSNorm.

The 576 values are laid out row-major in an 18×32 grid solely for display. It is a vector, not a native activation matrix. Click a cell or enter an index to read the FP32 value. Each displayed step has its own linear ±max-absolute scale, printed next to the grid; colors across steps do not represent a common absolute magnitude. The original BF16 weight panels and exact weight inspector remain separate and fixed.

## Bounds and lifecycle

| Resource | Bound |
| --- | --- |
| Inference sessions | One active worker, no waiting session queue |
| Prompt | 4096 UTF-8 bytes and 1–128 tokenizer tokens |
| Generation | 1–32 tokens per branch, each ends earlier on EOS; sequential branches share the total deadline |
| Activation history | One displayed branch × one layer × 576 scalars × at most 32 paired records |
| KV cache | One branch at a time, at most 160 total token positions; approximately 7.1 MiB in FP32 |
| CPU | One affinity-selected CPU; numerical thread counts all one |
| Inference worker | 1.5 GiB sampled RSS ceiling, 3 GiB virtual address cap, 90 CPU seconds |
| Headroom | At least 4.75 GiB available before model load; stop at less than 3.25 GiB to preserve the 3 GiB reserve |
| Session wall time / client lease | 120 seconds / 15 seconds |
| HTTP / IPC | 8 KiB body and headers; absolute 0.5 s receive deadline; 128 KiB partial IPC record buffer |

A 32-record deque bounds server history; a full stdout pipe backpressures generation. Each browser keeps at most the same 32 records. The coordinator checks memory and leases every 100 ms when idle and between requests. Client receipt retains its absolute 0.5-second deadline. The fixed local upstream has a separate 5-second total operation budget, with connection establishment capped at 0.5 seconds and short response I/O waits that continue servicing memory/lease checks. No request is automatically replayed. Other HTTP actions can wait behind that operation; this is a bounded single-coordinator design, not concurrent inference. The 5-second budget is a policy limit, not a measured worst-case CPU-contention guarantee.

Model-file verification, process reaping, and OS scheduling can add servicing latency. These controls are not a host-wide memory reservation against unrelated programs. The Rust renderer retains its 768 MiB address-space and cache limits, one numeric worker, and eight queued numeric jobs. Its four pending header readers share one thread, an 8 KiB per-request cap, and absolute 0.5-second receive deadlines. One fixed I/O dispatcher (2 MiB stack, four queued completed requests) separates header intake from metadata and response work; response partial writes share one absolute 3-second deadline. It adds no numeric or inference worker and runs within the same CPU/process caps. Recoverable accept failures back off to at most 250 ms; fatal listener errors still surface.

Completion, errors, cancellation, and lease expiry retain process ownership until reaping. Each cleanup attempt is bounded; a delayed child stays busy and is retried on later ticks. Shutdown also attempts renderer cleanup and closes the listener even if worker cleanup fails. If bounded shutdown cannot confirm reaping, it exits with an explicit incomplete-cleanup error rather than claiming success.

## API and validation

The extra API is `/api/inference`; actions require same-origin requests, `X-Atlas-Local: 1`, and JSON. General `GET /api/inference` returns only model, revision, engine, limits, a public `comparison` edit schema, and a boolean `busy` flag. It never returns a session ID, prompt IDs, trace records, or session-specific errors/metrics. `POST /start` accepts legacy `{prompt, max_new_tokens, layer}` or the comparison extension `{prompt, max_new_tokens, layer, source_model, edits}` described in the [edit contract](EDIT-CONTRACT.md) and returns a random ephemeral session capability to its initiating caller. `POST /poll`, `/cancel`, and `/reset` require that capability in their `{session}` body; missing, different, or stale capabilities return 409 without reading or changing the session. Invalid input or an already active session returns 400. The capability is held only in memory; no login, persistent credential, or new access mechanism is added. No endpoint accepts model paths or executable paths. The coordinator marks only its own pinned renderer metadata with `inference_source_model`, allowing the inspector to pass a supported native selection. Static Qwen viewers have no such marker. Owner poll responses contain bounded records and independent compute timing; browser playback is entirely local.

Upstream response status/body are preserved. Coordinator-generated failures use 503 with `code: backend_unavailable` or `resource_limit`, 504 with `code: backend_timeout`, and 502 for an oversized upstream response. They are distinct from client validation errors (400) and the renderer's existing calibration-readiness 503. A reset still awaiting reaping returns 503 with `code: cleanup_pending`; its session is not cleared. The frontend keeps backend failures separate from calibration messages.

Run the no-model controls and retained baseline checks:

```bash
python3 tests/inference_contracts.py
python3 tests/edit_contracts.py
node tests/ui-edits.cjs
python3 tests/reliability_contracts.py
node tests/ui-inference-races.cjs
node tests/ui-contract-races.cjs
python3 tools/smoke.py
python3 tools/guarded-build.py test --release -- --test-threads=1
python3 tools/guarded-build.py clippy --all-targets -- -D warnings
```

Real-model reference comparison (no network):

```bash
python3 tools/guarded-inference-test.py /path/to/cpu/python \
  tests/inference_reference.py /path/to/smollm2-135m
```

It compares tokenizer IDs with official AutoTokenizer, generated IDs with Transformers `generate`, and cached decoding activations/logits with uncached full-prefix forwards. Layer 29 is compared after applying final RMSNorm to match the reference hidden-state convention. EOS uses a controlled test-only EOS-ID override, not a claimed natural-EOS example.

For an installed Playwright and Chromium, run `tests/inference-browser.cjs` through `tools/guarded-browser-test.py`, supplying `NODE_PATH` and `ATLAS_CHROMIUM` for that installation. Tests use a fresh browser context, Chromium sandbox enabled, one CPU, a 768 MiB browser-tree RSS cap for this qualification, and localhost-only page requests. Evidence is written under ignored `results/`; trace archives can contain local model paths from the metadata endpoint and must not be committed.

See [feature status](FEATURE-STATUS.md) and [development checks](DEVELOPMENT.md) for current scope.


## Native edits and score comparison

Inspect a stored matrix below the inference panel. Add a zero/scale operation on the inspected element, or enter a native row/column range with an **exclusive end**. The query-head shortcut proposes the selected one of nine 64-row query heads; it does not claim that a head has a particular semantic purpose. Review the ordered draft before running. At most eight operations are accepted; scale factors must be finite in [-100,100]. Runtime dtype is FP32 and any nonfinite edited values or output logits fail the run. Operations on overlapping addresses compose in list order. Vectors and aliases are excluded from the stored matrix allowlist. Editing the stored embedding explicitly affects its tied output projection too.

An empty edit list performs two real deterministic runs and is expected to produce exactly matching generated IDs, text and scores in the same environment. The model loads once; baseline generation finishes before edits are applied. Each branch begins with a fresh prompt/KV cache. The process accepts one request and exits/reaps, so no edited model is retained for future sessions. No model save API is called. Pins are verified before coordinator startup, before every session, and again by the worker before loading.

The table shows the union of the two top-five candidate sets (at most ten tokens). Both raw logits are available for each candidate, including candidates only in the other branch's top five. Delta is edited minus baseline; the logits originate as FP32 and subtraction uses the host's Python float. Raw logits are not probabilities and a common logit shift is not itself a probability shift. Step zero uses the identical prompt. Later rows explicitly distinguish identical versus different consumed generated prefixes. Free-running differences after divergence combine changed weights and changed context; they are not matched-context causal effects. If one branch ends early, missing scores/deltas are null and the UI shows a dash. Full generated token IDs and text for both branches are returned; no full-vocabulary score vectors leave the worker.

The activation grid displays the edited branch while available, otherwise baseline after the edited branch ends. It identifies that branch. The accepted run inputs remain separate from the editable draft and prompt. Reset clears the accepted run/trace after ownership-safe cleanup and retains the draft for review or a future run; Clear draft edits removes it explicitly.

Only the complete pinned SmolLM2-135M model is supported for inference. `run-atlas.sh` launches a static viewer, so inference remains unavailable and its message persists during viewer actions. Run `tools/live_inference.py` as shown above to open a separate SmolLM2 viewer on 8796/8797; do not reuse the Qwen viewer's coordinates or port 8775.

Additional opt-in guarded checks (run one at a time after obtaining the shared heavy-test slot):

```bash
python3 tools/guarded-inference-test.py /path/to/cpu/python tests/edit_reference.py /path/to/smollm2-135m
python3 tools/guarded-inference-test.py python3 tests/edit_lifecycle.py /path/to/smollm2-135m /path/to/cpu/python
```

`edit_reference.py` independently checks selected scalar arithmetic, overlaps, tied storage, empty-edit equality, and whole-query-head score changes with an uncached full-prefix forward. Effects are measured on an explicit public synthetic prompt fixture, not forced by a minimum-effect assertion. `edit_lifecycle.py` uses a test-only helper to stop or fail after edits are applied, verifies reaping and all four disk hashes, then checks a fresh empty-edit session. Neither helper is reachable through the HTTP API. Current tested/unrun scope is recorded in the feature ledger and comparison evidence document when available.


## Capture and opt-in logging

See [capture/logging contract and current qualification](CAPTURE-LOGS-CONTRACT.md). `activation_site` defaults to `block`; `attention` and `mlp` observe one selected component without altering computation. Layer selection is strictly integer 0–29. One hook, one 576-value vector and the existing 32-step trace cap remain.

Private logging is off by default. Enabling it collects future accepted runs, including failed/cancelled status, in this tab's memory, capped at 8 runs / 1 MiB. Prompt inclusion requires a separate checkbox. The log never silently evicts: overflow holds one pending record, visibly blocks new runs, and requires explicit export/clear/discard. Downloading is manual; unsaved records disappear on reload/close. This is partial support for reproducible experiment logs, not automatic durable file recording. Runtime-version provenance is recorded only when returned by the loaded worker. All file exports omit capabilities and internal paths. Generated outputs remain private even when prompt fields are omitted.

## Selected observations

The optional attention mode selects one query head 0–8 at the chosen layer and displays its genuine eager last-query softmax distribution, with native consumed-token positions/IDs and probabilities. It does not expose all prompt queries or establish causal attribution. Logit-lens mode applies the actual final RMSNorm and tied output head to one selected post-block residual, reporting a bounded union of top candidates with both lens/final logits and their difference. This is an observational readout of an intermediate representation, not an early-exit prediction. Only one mode is selected, and the corresponding activation site is explicit. Both use ordinary playback and the displayed activation branch. Export omits all attention key token IDs unless separate prompt inclusion is enabled. See [the contract and pending validation scope](OBSERVATION-CONTRACT.md).

## Explicit prompt-pair captures

Choose the two-prompt mode, preview and review exact tokenizer positions, then select up to eight explicit A,B pairs at one layer/output. A fresh worker runs exactly two uncached prefills with no text generation or edits. The UI steps through vectors and B−A differences, with token/prefix equality and zero-norm-aware metrics; no semantic or causal alignment is inferred. Preview and comparison retain existing ownership, resource and time limits. Prompt and source-token export uses separate opt-in consent. See [PROMPT-PAIR.md](PROMPT-PAIR.md) for exact closed schemas, privacy and tested/unrun scope.

## Bounded head sweeps

Review explicit output-head/query-row targets and their matched controls, or a single-layer output-head plan. A pinned SmolLM2 layer expands to nine targets, nine controls and one empty case: 19 records / at most 38 prefills for one prompt. Explicit subsets allow at most two targets and two prompts, ten records / twenty prefills. Every plan shares the original 120-second wall / 90-CPU-second allowance, with no per-head reset, split jobs, or automatic retry. Restoration verifies selected original bits and repeated baseline logits before continuing. Controls are comparison interventions, not assumed null effects. Partial results stay explicit. See [HEAD-SWEEPS.md](HEAD-SWEEPS.md) and [SWEEP-CONTRACT.md](SWEEP-CONTRACT.md).
