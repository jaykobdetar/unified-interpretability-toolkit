# Bounded data workflow qualification plan

Status: export/import/storage contracts, independent numeric fixtures, and bounded synthetic browser export/archive mechanics are checked. The [browser qualification scope](DATA-BROWSER-QUALIFICATION.md) records exact production assets and the remaining permission/integration gaps. Separate [128-window SVD qualification](analytics/SVD-QUALIFICATION.md) covers four fixed synthetic numeric cases and an ordinary standalone HTTP/browser workflow. Full-slice profile host workflows, higher-rank bookmark/note scope and real-model profile qualification remain separately assigned work. The steps below are remaining qualification plans; keep receipts, temporary models and logs outside the checkout.

## Build and retained checks

From the exact candidate commit, run the retained CI sequence under existing guards: `tools/check.py lint`, `tools/check.py contracts`, guarded offline Clippy, guarded release Rust tests, guarded release build, fixture smoke and launcher preflight. One job at a time unless the parent explicitly assigns separate capacity. Do not reuse an earlier embedded-asset binary to qualify changed JavaScript. No new dependency installation or model download is required for the source/contracts.

The independent export oracle is `tests/workspace-export-numpy.py`, using already installed NumPy. It checks original BF16/F16/F32 bytes, signed zero, minimum subnormal and maximum finite values through CSV and `np.load(..., allow_pickle=False)`, including explicit rank-three indices. It qualifies the codec, not the actual HTTP/browser flow.

## Actual viewer and archive workflow

Use an owned loopback fixture viewer on fresh ports. Open small owner-created BF16/F16/F32 tensors with shape `[2,3,4]` and distinct leading slices; retain source hashes and independently known values before/after. Through the ordinary UI, select leading index 1, inspect `[1,2,3]`, compare signed/magnitude views, export CSV and NumPy plus metadata, then independently decode the downloaded values/bytes and source/slice binding. Switch leading index explicitly, verify tile/inspection/bookmark/note/export scope follows it and no inference edit handoff appears. Repeat only the predeclared small regions; no model-wide scan.

Use a synthetic prompt-inclusive v1 archive to qualify the actual import dialog and downloads without private prompts or model execution. Check unchecked import consent redacts prompts, imported records leave run inputs unchanged, and no `/start` request occurs. The trusted fixed importer asset loads only on the first explicit import/restore. Ordinary browser initialization is checked; deterministic load failure, explicit retry, timeout, cancellation and reset remain pure-double coverage. Do not inject live fault responses without separate authorization. With explicit browser-storage consent, save, reload, observe empty memory/off toggles, then explicitly restore. Open a second tab for archive discovery and stale-snapshot recovery; retain each tab's archive without worker adoption. Cover the per-log cap and explicit deletion/recovery with local safe fixtures.

On a browser supporting File System Access, manually select one fresh task-owned JSON file through its real picker; verify a logged terminal outcome reaches the file after the saved status. Check a separate explicitly captured-prompt retention choice and a cleanup-confirmation update. Record exact browser/runtime and supported/unsupported status. Safe mocked write failures qualify the recovery logic only; they do not qualify the real browser permission flow. Stop journal updates, close the tab, and confirm reload does not reuse the handle. Physical mobile devices remain separate from emulation.

## Full-slice profiles

The current explicit profile host only activates the exact registered 244-byte synthetic BF16 fixture, with tensors `[3,5]`, `[7]` and `[4]`. It does not admit an arbitrary rank-three fixture or real model through that route. Keep this boundary intact; do not broaden its allowlist solely to make a test pass.

In that owned fixture host, use the ordinary **Start full slice** action once on the 15-value matrix. Verify native binding/seed, visited=total, complete state and cleanup, and page both original/control row and column aggregates. Independently reconstruct the exact source multiset and declared control, sums, visited counts and means. Explicitly reset and confirm snapshot release before another job. A predeclared partial allowance can qualify partial/unvisited labels and an explicit restart from zero; there is no Resume/automatic continuation. Stop and retain a partial outcome if any existing resource/time gate refuses it.

`tests/profile_worker_oracle.py --binary /path/to/qualified/binary` provides a separate tiny native-worker/frame numerical oracle for BF16/F16/F32 rank-three slice 1, full and explicit partial/resumed frame equivalence. It does not establish host admission, global accounting/watchdog or actual browser behavior. The HTTP fixture host does not enable resume merely because that native oracle succeeds.

## SVD and completion evidence

The legacy dense SVD cap stays 64 × 64. The explicit 128 × 128 summary has the [retained synthetic and standalone workflow scope](analytics/SVD-QUALIFICATION.md); registry/fixture-host admission, public-model windows and vector/rectangular runtime cases remain unqualified. Follow [SVD scaling assessment](analytics/SVD-SCALING.md) for further work. Stop on failure/refusal; do not silently retry with looser gates or another seed.

For each granted phase, report exact candidate/binary digest, fixture source identity, executed bounded cases, numerical oracle, wall/CPU/RSS/output sizes, observed complete/partial status and confirmed owned-process cleanup. A successful source-only test is not a successful runtime workflow. Release capacity only after the owned coordinator and its children are stopped/reaped and its ports close; leave unrelated applications and pods alone.
