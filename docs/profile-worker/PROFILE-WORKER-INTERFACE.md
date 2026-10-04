# Profile worker and snapshot interface

Session-only sealed memfd snapshots preserve a latest accepted frame across explicit disposable-worker grants. The fixture host integrates these primitives; see [architecture](../ARCHITECTURE.md).

## Implemented native interface

`strength::snapshot::layout(rows, cols, binding_bytes) -> Result<Layout>` returns
`frame_bytes` and `live_bytes`, refusing before accumulator/frame allocation.
`StrengthProfile::write_snapshot(source, Write, Instant) -> Result<String>` writes
one frame incrementally and returns its SHA-256. Partial writes are discarded.
`StrengthProfile::restore_snapshot(source, TensorSlice, model_identity, seed,
Read, encoded_len, expected_revision, Instant) -> Result<StrengthProfile>` restores
an exhausted profile: zero authorized remaining values and zero time allowance.
`visited_values()` is a scalar accessor. Existing `advance_until` is unchanged.

Schema is the exact 168-byte header and 24-byte record layout in DESIGN-CONTRACT.
Algorithm preimage is UTF-8:
`weight-atlas-strength-snapshot-v1:kahan-f64-abs:swap-or-not-8-v1`.
Canonical binding bytes are sorted-key compact UTF-8 JSON, matching serde_json's
current map serialization and Python canonical(); snapshot binding is compared
byte-for-byte against the owner-reconstructed binding, not freely deserialized.
Header digest offsets are 72 (binding), 104 (algorithm), 136 (profile identity).
Records are original rows, original columns, control rows, control columns.
Neither clocks, grants, capabilities nor filesystem paths are stored.

Validation checks finite nonnegative sums, finite compensation with magnitude
<=64*f64-epsilon*sum, empty-count zeros, counts, and exact source-dtype maximum bounds (F16 65504, BF16 `0x1.fep+127`,
F32 `0x1.fffffep+127`).
Paired totals must agree within 128*f64-epsilon*max(total pair). Native and Python
validators use the same specified invariants, but independent implementations.
Control counts reconstruct the visited bijective prefix in bounded steps; full
coverage uses known exact axis counts. A valid large partial snapshot can exhaust
validation time and be refused. No arbitrary visited-value cutoff replaces that
bound; Restart remains the public option until safe resume is qualified.

Private CLI entry: `profile-worker`. Required option pairs:
`--model ROOT --revision REV --tensor ID --slice INDICES --seed UINT32 --values N
--wall-ms REMAINING --cpu-ms REMAINING --binding CANONICAL_JSON --output-fd FD`.
Optional restore pair: `--input-fd FD --input-sha EXPECTED_REVISION`.
Only the trusted adapter constructs these, including owner-resolved ROOT and REV.
Unknown/duplicate options and oversized metadata refuse. Main dispatches before
State/cache/server initialization. No model execution, calibration or subprocess.

The worker installs one-CPU/768-MiB-AS limits, a coarse child CPU rlimit and exact
frame-sized RLIMIT_FSIZE. Child CPU seconds round upward; this is NOT enforcement
of a fractional aggregate allowance. The host's independent watchdog is required.
The parent supplies an elapsed-adjusted duration and retains its original hard
wall deadline. Worker-local Instant starts at entry; source/restore/setup consume
that duration. Scan ends 200 ms before its supplied deadline for serialization.

The worker duplicates designated descriptors, verifies restored input seals/size,
streams output, adds WRITE/GROW/SHRINK/SEAL seals, rechecks source, then emits one
small stdout receipt, schema `weight-atlas.profile-candidate.v1`, with
`revision, visited_values, new_values, frame_bytes, live_bytes`. This is a candidate,
never accepted progress. Errors exit nonzero. No raw path/error is public progress.
Only the adapter's designated FDs may be inherited; all others must be closed.

## Snapshot host interface and ownership

`profile_snapshot.layout(binding) -> (frame_bytes, live_bytes, canonical_bytes)`.
For A=rows+cols, B=canonical binding length, F=168+B+48*A:
`live_bytes = 56*A + 2*F + 2 MiB + 128 KiB <= 32 MiB`.
56*A reserves native Kahan accumulators plus control-count verification arrays;
2*F reserves old and candidate frames; fixed space covers buffers/metadata and
one bounded page. Worker allocations fit this same reservation. Native source
metadata and process machinery remain within the separate 768 MiB AS/RSS envelope.
Conservatively add allocated memfd storage to owned-process-tree RSS even if a
mapping makes it counted twice; never subtract shared mappings to obtain a pass.

`SnapshotStore(binding, seed)` owns all descriptors. `begin() -> output_fd`
creates exactly one pending anonymous sealable memfd after the checked estimate.
`new_memfd()` fails explicitly if Linux memfd/seals are unavailable; no disk path.
Parent retains FD ownership throughout the child lifetime. The child inherits an
FD reference, not ownership of the parent's handle. On successful exit the parent
validates seals, exact length, same-handle checksum and complete frame invariants.
Required seal mask is WRITE|GROW|SHRINK|SEAL. No external FD/path is accepted.

`publication_steps(revision, minimum_visited, maximum_visited, deadline,
source_check, final_check, clock)` is a generator. Each step performs at most
256 count/record items, with deadline checks. It returns immutable SnapshotInfo
(revision, visited, total, frame_bytes, live_bytes). It compares trusted expected
binding/seed and receipt cursor bounds, calls fresh source_check before/after,
and requires final_check before swapping handles. A rejected/closed generator
closes its candidate while retaining an otherwise valid prior snapshot before
swap. The Job independently gates reads by accepted revision/publication serial;
failed post-swap finalization retires the replacement. The
synchronous `publish()`/`validate()` wrappers are for controlled qualification;
do not call them in HTTP start/status/page handlers.

`page(revision, axis, start, count<=1024, source_check)` reads original/control
records by bounded offsets, never replaying prefix/checksum scans. Output includes
one revision, binding and paired counts/means, capped at 2 MiB. A short lock pins
one page reader; no new admission or publication can occur while it holds the
old handle. `page_handle()` exists for trusted bounded readers only; a handle must
not escape its context. No dup/pinned third revision is permitted. Cancellation
cleanup waits until this bounded reader exits. Source is checked before/after.

`owned_storage_bytes` reports every owned allocation without a store-lock wait,
including retiring/uncertain handles until confirmed close. Close uncertainty
retains the charge and poisons the shared slot; no FD retry is allowed without
trusted disposition reconciliation. `require_settled()` gates cleanup/admission.

`abort_candidate()` closes the candidate. `close()` closes both generations and
clears the accepted snapshot; it refuses while a page reader is pinned. Session
expiry/end must invoke this after safe worker cleanup, including terminal jobs.

## Lifecycle adapter: mandatory injected provider

`ProfileJob(binding, seed, tab_capability, context_id, platform)` supplies a new
random job capability. Store it privately with its owning tab lease. A shared
renderer context ID alone grants no access. Every control/status/page call checks
all three values. Capabilities never enter receipts, frame data, URLs or progress.

- `start(tab, job, context, values, wall_ms<=5000, cpu_ms<=4000, resume=False)` is
  one explicit action in a dedicated serial job-owning execution context. It
  charges clocks before admission, acquires the common slot without queuing,
  checks reserves/source, creates a candidate and spawns one child. It preserves
  the previous accepted snapshot until a new candidate succeeds. Restart scans
  from zero; resume=True explicitly refuses until separately qualified.
- `tick()` runs in that same owner context independently of HTTP arrivals. It
  handles reap and at most four validation-generator steps / cooperative 2 ms.
  No replacement child or new allowance is created. `status()` is a small read.
- `cancel()` is capability-checked, sets an Event and sends a nonblocking stop;
  owner ticks perform reap/cleanup. `heartbeat()` only extends a live owner lease;
  it never alters the original wall deadline, CPU budget or authorized values.
- `guard()` is called by an independently scheduled watchdog, even if the serial
  HTTP server or job tick is blocked. It checks original wall/CPU/lease/slot and
  resources and signals stop on uncertainty. Unreaped child ownership retains and
  poisons the shared token. No terminal result is published before reap, final
  CPU rusage, same-handle validation, source correspondence and cleanup checks.
- `page()` authorizes owner/revision and delegates to the bounded pinned read.

The fixture host supplies the concrete platform/provider in `tools/atlas_host/profile_os.py` and `hosted_runtime.py`. It enforces the following ownership boundaries within that host; separately launched inference/viewer processes do not share a global exclusion token.

| Provider method | Required behavior |
|---|---|
| clock() | Monotonic wall clock, one domain for host ledger |
| owner_cpu() | Job-local cumulative CPU meter covering dedicated owner work plus watchdog/adapter work; queryable correctly from watchdog context, never that caller thread's unrelated clock or server lifetime CPU |
| acquire() | Nonblocking common exclusion; return owned token or refuse busy; calibration/inference/tile work must honor it |
| token.current/release/poison | Check exact ownership; release only after reap and finalization; poison keeps uncertain ownership unavailable |
| start_gate() | Unchanged 5 GiB available and 25 GiB disk gate, checked once per explicit action |
| source_check() | Bounded current registry/renderer/native correspondence under token; no fresh whole-model hash |
| spawn(request, output_fd, input_fd) | Resolve owner registry source; spawn trusted frozen Rust binary with only designated FDs; return owned handle before fallible post-spawn initialization |
| child.initialize() | Bounded setup after ownership, e.g. nonblocking output; failure leaves handle owned |
| child.sample() | Bounded thread-safe sample of same PID/start or pidfd; cumulative owned child CPU, reaped flag, exit code and bounded strict receipt; cache final wait4 rusage; no double reaping/counting; unexpected descendants require stop/reap and failure |
| child.stop() | Nonblocking signal to exact owned process identity; no PID reuse race; safe before initialize completes |
| resources_ok(frame_bytes) | Whole owned-tree RSS + explicit memfd bytes <=768 MiB, >=3.25 GiB available; account all owned processes, no exclusions |
| arm_watchdog(job) | Independent scheduler covering admission through cleanup; return bounded close() registration; cannot depend on status GETs or job-owner liveness |

Child CPU is added to owner deltas from before admission. Finalization reserves
800 ms of the same host wall grant before dispatch; the child's additional 200 ms
serialization reserve is within its shorter duration. Slow validation can still
exhaust the grant and must fail. There is no guarantee every shape fits five
seconds. Time checks after final handle/slot cleanup prevent late success; a late
already-swapped candidate is retired rather than exposed as accepted output.
Public terminal errors are fixed admission_failed/worker_error/resource_limit;
internal validation messages are not exposed. Cancellation yields cancelled only
when it is an explicit cancellation, not a resource-watchdog expiry.
