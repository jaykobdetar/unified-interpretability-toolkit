# Native interface declarations

`src/command.rs` declares the fifteen exact native command IDs, startup class,
twenty-three static fallback sites, help text and audited command option names.
`src/api.rs` declares viewer/comparison paths, private renderer IDs and parameter
sets. Both modules are pure; transport, source/resource checks and status/error
mapping stay at their existing callers. These declarations preserve current
behavior. Ordinary CLI and native HTTP intake still ignore unknown names. The
private renderer and profile worker retain their existing strict admission.

All ordinary commands retain the documented common names `model`, `cache`,
`name`, `revision` and `resources`. Early metadata ignores cache/name/revision;
comparison ignores name/revision. A known resource name does not override the
existing restriction to standalone commands. Profile-worker uses its own set.

| Command | Additional names |
| --- | --- |
| metadata, verify | None |
| serve | port, verify-sha |
| calibrate | tensor |
| overview | tensor, slice, rules, max-values |
| tile | tensor, slice, rules, level, x, y, out |
| inspect | tensor, slice, row, col, left, right |
| bench | tensor, repeats |
| compare-metadata, compare-calibrate | compare-model, tensor |
| compare-serve | compare-model, tensor, port |
| compare-tile | compare-model, tensor, quantity, mapping, level, x, y, out |
| compare-inspect | compare-model, tensor, row, col |
| hosted-renderer | channel-fd |

Comparison metadata/serve still parse tensor eagerly. Unknown `compare-` IDs
still enter comparison construction before their exact refusal. Private worker
required names are model, revision, tensor, slice, seed, values, wall-ms, cpu-ms,
binding and output-fd; optional input-fd/input-sha remain paired. Its unknown or
missing-name, metadata-size, restore-pair, duration and finalization checks retain
their original order and messages.

| Viewer path | Method | Native query names |
| --- | --- | --- |
| /api/model | GET | None |
| /api/progress, /api/tensor-status | GET | tensor |
| /api/view | GET | tensor, slice, left, right |
| /api/inspect | GET | tensor, slice, row, col, left, right |
| /tile | GET | tensor, slice, rule, level, x, y, binding |
| /api/calibrate | POST | tensor, all |

The broker-only `/api/binding` path maps to private `binding` with tensor/slice.
It is not served by native public HTTP. Private IDs are model, progress,
tensor-status, view, inspect, tile, calibration and binding. Their sets match the
table except private tile excludes binding and private calibration excludes all.
Private tensor-status preserves its existing contains-key behavior, which differs
from the public selected-tensor requirement. Request/envelope unknown fields are
still refused by the existing serde declarations.

| Comparison path suffix | Method | Names in addition to comparison_identity |
| --- | --- | --- |
| model | GET | None |
| view | GET | tensor, left, right, mapping |
| inspect | GET | tensor, row, col |
| tile | GET | tensor, quantity, mapping, level, x, y |
| calibrate | POST | tensor, all |

Every comparison path begins `/api/comparison/`. Identity is checked before
method/route handling, including calibration. The all name stays declared so
all=1 retains the tensor-scoped refusal. Native parsing, authority/origin checks,
headroom, local-action header checks, readiness, queue behavior and error status
selection remain unchanged. Public parameter sets and lazy typed fields share their declared names.
Unknown-name rejection remains a separately scoped change.

The caller audit covers repository launchers, wrappers, browser builders,
reference/acceptance harnesses, the behaviour lock and documented recipes.
`run-atlas.sh` delegates to launch.py; launch.py and live_inference.py launch serve;
prepare_fixture.py launches calibrate; hosted_runtime.py builds the private argv
and eight route envelopes; profile_os.py builds worker options and descriptor
arguments. RuntimeAdapter strips broker context before its six native reads.
Host-client adds context to broker URLs, not the native renderer request. Viewer
settings supply tensor/slice/left/right; scalar reads add row/col; exports use the
same inspection set. Tile builders carry binding only on public native tiles.
Comparison builders attach identity to every request once model state exists.
The live-inference proxy forwards native method/path/query without rebuilding
parameter names. Host/profile/static launcher wrappers delegate to these same
builders. Startup diagnostics and validation policy inspect their argv but add
no native arguments. Smoke, numerical references, acceptance drivers and the
lock also use slice, left/right, output prefixes and worker restore names that
are easy to miss in the short help text. Arbitrary callers outside the repository
are not audited.

Unknown-name rejection requires a separate production/tests/changelog commit.
Its deliberate refusal differences must be reported separately. Current ignored
names, unknown-command routing and validation priorities remain held here.

## Lazy CLI field parsing

`command::argument` declares the existing option names, Rust result types, static
fallback references, required-value messages and parsing functions. The common
`parameter::Parameter` reader parses only the requested field. Ordinary CLI
callers evaluate these fields at their original points: model admission precedes
command-specific parsing, optional calibration tensor omission still means all
available tensors, and comparison commands still parse tensor eagerly. Dynamic
tile levels are obtained from the selected tensor/pair before the field read,
even when an explicit level is present. Empty values remain present values.

The command help and static fallback text still come from the existing macro;
the typed fields reference those fallback constants. Resource configuration and
numerical/domain checks retain their existing owner and order. Unknown options
remain in the parsed map and remain ignored by ordinary command execution.
The private worker and HTTP field declarations use the same dependency-leaf
reader at their original validation points.

The pure slice-index parser lives in the dependency-leaf parameter module.
`slice::parse_indices` re-exports the same function for existing callers; the
parser body, length guard and error text are unchanged.

## Route and private fields

`api::argument` declares borrowed text defaults and typed numeric/slice parsers.
Viewer/private route sets and comparison route sets derive their names from those
fields. The route macro also supplies the method constants used by dispatch.
`Query::field` borrows text; `Query::read` parses one declared value. Existing
public `Query::get` and `Query::int` remain available. No request-wide eager
validation is added, and unknown public query names remain ignored.

Comparison calibration keeps its empty missing-tensor value and resulting integer
parse error. Viewer calibration still defaults to tensor zero. Public status reads
still distinguish an absent/empty tensor from a selected tensor; private status
reads retain their contains-key distinction. Levels still parse as `usize` before
checked conversion to `u32`, preserving the original error family and text.

`command::argument` also supplies private worker field types and generates its
existing required/restore name sets. Presence, unknown-field, metadata-size,
restore-pair, grant, seed, descriptor, seal and ownership checks retain their
original owner, sequence, values and messages. Number parsing stays at the same
points between those checks. No guard or resource allowance changes.

Command-specific audited name sets now reference the same typed fields used by
their consumers; inspect shares the viewer fields it already delegates to. Help
bytes remain generated by the original command/default template. These changes
complete field parsing and declared-name/method integration, while domain and
resource validation remains with the owner that has the required state.
