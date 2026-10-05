# Synthetic data workflow browser qualification

The integrated local batch passed 80 Rust tests and a successful exact-source release build, then fresh synthetic exports/archive browser checks and independent decoding of all 18 downloads. Its current binary SHA-256 is `3e501dd150f646fffc0c1321d3bb2d74128f5c3bc2bf08cf24e880a4a62169e7`. Source, build, harness and asset bindings were preserved before and after; independent final qualification is accepted. Remote publication and exact-commit CI remain separate.

Current exports took 6.61 seconds with 855.68 MiB sampled owned RSS; archives took 4.76 seconds with 866.96 MiB. Both retained the one-CPU/1-GiB/120-second guard, verified cleanup and closed ports, with no page/proxy errors or inference mutations. NumPy 2.4.2 decoded all 18 files across six BF16/F16/F32 slice/vector cases. These are observed bounded synthetic runs, not general performance or permission guarantees.

## Inherited source qualification

Earlier production assets at `f9d8a56514a210314af82872040c6c03e90d7f86` passed 60 release tests and release build; that binary SHA-256 was `54d13bc37c43dd805444588a90841dcf0089051ecb35503595d3ecaff4a434fe`. The earlier measurements below remain historical. The combined batch above received a fresh build and runtime qualification rather than reusing that binary.

The deterministic source fixture has 84 finite BF16/F16/F32 values in six tensors: three `[2,3,4]` tensors with distinct leading slices and three four-value vectors. Its 640-byte safetensors SHA-256 is `88a2a3c670327ac387ec6133f8b4773b7e0f6ad72b70c1178d44c7781d0846d5`. It includes signed zero, minimum subnormals and maximum finite values. No trained weights or private prompts are used.

### Earlier executed checks

Chromium `153.0.8010.12`, through installed Playwright `1.62.1`, qualified two separate ordinary browser lifetimes:

| Phase | Passed scope | Peak owned RSS | Wall time |
| --- | --- | --- | --- |
| Exports | Actual rebuilt bundle/CSP; explicit rank-three slice selection and switching; native scalar indices/bytes; BF16/F16/F32 CSV, numeric NPY and metadata downloads for slices and vectors | 818.39 MiB | 7.64 s |
| Archives | Actual file input and lazy codec MIME/hash/CSP; default redaction; unchanged run inputs; explicit save/reload/restore; copied-owner Web Lock FIFO contention in both save/delete orders with visible stale-save refusal; real OPFS writable transactions, prompt retention/redaction, detach followed by reload with no reattachment | 865.48 MiB | 4.44 s |

Each lifetime used the existing one-CPU, 1 GiB summed-RSS, 3.25 GiB available-memory reserve and 120-second guard. Available memory stayed above 8.3 GiB. Guard and fixture receipts verified owned-process cleanup, both closed ports, no unexpected server mutations and no proxy failures. The initial restricted-sandbox socket refusal was retained as a separate failed attempt; the supported approved loopback execution kept the same limits.

Installed NumPy `2.4.2` independently decoded all 18 actual export files in six tensor cases. CSV values, original bytes, native indices, source/revision bindings and NPY bits matched the fixture, including signed zero/subnormal/extrema. NPY loading used `allow_pickle=False` and verified numeric dtype, two-dimensional C-order shape and sidecar bytes. Viewport screenshots of export and archive controls were inspected.

## Scope and remaining interactions

The archive phase used a clearly labeled metadata-only fixture coordinator. It forwarded the actual rebuilt Rust assets and their CSP, exposed only benign availability metadata, and refused every mutating server request. It proves the archive UI mechanics with synthetic data. It does not prove integration with an actual inference coordinator or model worker.

The OPFS test used a real browser-owned `FileSystemFileHandle`, but replaced the picker with that task-owned handle. It qualifies write/close and consent/redaction mechanics, not native file-picker permission. Reload with an active journal was not exercised; initialization forgets the handle in source, while this browser sequence detached it first. Actual native picker choice/cancellation and permission prompts need a headed interaction selecting one declared fresh temporary journal, without personal files. Headless file-input selection exercised the upload control but did not qualify an operating-system file dialog. Downloads used `acceptDownloads:true`; multiple-download permission UI remains unqualified.

Deterministic load/write failure, timeout, cancellation, reset and stale initialization remain covered by the retained pure doubles. No live failure responses, denial probes or permission revocation were injected. Bookmark/note scope across higher-rank slices, full-slice profile host workflow and physical mobile devices remain separate qualifications. The later [128-window SVD qualification](analytics/SVD-QUALIFICATION.md) records its own exact runtime and standalone HTTP/browser scope; it does not broaden the export/archive integration or permission claims above.

## Reusable harness

`tests/data_workflow_fixture.py` creates the declared fixture only in a fresh explicit directory outside the checkout. `tests/data_workflow_fixture_contracts.py` checks its immutable bytes and bounds without filesystem writes or runtime jobs.

Only after receiving a runtime slot, rebuild the exact accepted production assets with the existing offline build guard, provide the installed `NODE_PATH` and `ATLAS_CHROMIUM`, and set `ATLAS_EVIDENCE_DIR` to a fresh directory outside the checkout. Run each phase separately:

For a later source snapshot, explicitly set `SOURCE_COMMIT` to its full reviewed commit and `BINARY_SHA256` to its recorded successful release-build digest. The driver retains its historical candidate as the default; selecting another immutable commit requires the binary pin and still rejects changes in `src/` or `web/`. Receipts distinguish the production source commit from the test checkout HEAD and record any supplied behaviour lock's hash. The measurements above describe the historical accepted batch, not a new snapshot.

```sh
python3 tools/guarded-core-ui.py python3 -B tests/data_browser_driver.py exports --binary target/release/weight-atlas-rust --source-commit "$SOURCE_COMMIT" --binary-sha256 "$BINARY_SHA256"
python3 tools/guarded-core-ui.py python3 -B tests/data_browser_driver.py archives --binary target/release/weight-atlas-rust --source-commit "$SOURCE_COMMIT" --binary-sha256 "$BINARY_SHA256"
```

Use different evidence directories for the phases. The driver binds the production commit, binary, fixture and asset digests, owns its loopback servers, and reaps them before returning. It refuses changed production sources, lacks worker routes, and fails on unexpected mutations. Run `tests/data_browser_download_numpy.py` on the exports directory using already installed NumPy. Retain receipts, downloads, screenshots and temporary fixtures outside the checkout. Never reuse an earlier embedded binary for changed JavaScript or relax limits after a refusal.
