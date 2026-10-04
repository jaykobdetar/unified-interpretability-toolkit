# Workspace tools contract

`web/atlas-tools.js` provides pure source-binding/bookmark/note/export helpers. `web/workspace-tools.js` integrates them with the viewer; neither owns inference sessions. A selection remains native source coordinates, including an explicit slice when required. Every asynchronous result must match the current source/model/tensor/slice and view epoch.

Bookmarks encode only version, opaque source/model identity, numeric tensor ID, bounded region/viewport, allowlisted rules and explicit slice indices. They never include prompts, session capabilities, source paths, tensor names, free-form revision strings or note text. Existing 2D bookmarks remain v2; higher-rank mappings use v3.

Notes use bounded browser localStorage scoped to exact source/model/tensor/slice identity. They are never sent to a server or embedded in a bookmark. Local storage persists on the same browser origin; review and clear private notes before sharing that browser profile.

CSV export uses bounded, sequential original-address inspection, validating binding and scalar dtype/bytes before every append and before download. Selection/source changes cancel export; an obsolete request cannot restore newer inspection or highlights. Exports preserve original signed values even when a magnitude rule is displayed. CSV export currently supports BF16 only and retains its original bytes and exact decimal values; other dtypes are refused before reading. The viewer inspector separately supports BF16/F16/F32. Native NPY export is not implemented.

All downloads require a user action. Retry applies only to bounded, cancellable read-only requests; it never automatically retries a generation, edit, calibration, ownership-changing request or refused job. Progress and cleanup status remain visible. See the retained workspace/UI contract tests for boundary cases, and [capture/log privacy](CAPTURE-LOGS-CONTRACT.md) for inference records.
