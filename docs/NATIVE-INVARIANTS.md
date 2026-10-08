# Native unwrap, lock and deletion audit

This review describes the native source at commit
`b48a29af0d723df487baaab9bb90e4726725db3b`. It changes no production code,
tests, panic behavior or lock recovery. The counts cover all 23 `src/**/*.rs`
files, excluding 20 reviewed `cfg(test)` items. They count explicit source
occurrences, including multiple operations on one line; implicit indexing,
allocation failures, FFI and all possible execution paths are outside this count.

There are 62 `.unwrap()` calls, two `.expect()` calls and two `unreachable!`
sites. Forty-two unwraps consume mutex acquisition results. The other 24 sites
are the 20 value unwraps, two expects and two unreachable branches below. No
production `panic!`, `todo!`, `unimplemented!` or `allow(dead_code)` matches were
found by this bounded inventory. Absence of these spellings does not prove that
execution cannot panic or that every item is used.

## Value invariants and callers

| Location | Sites | Existing prerequisite and review boundary |
| --- | ---: | --- |
| `comparison::Comparison::fields` | 1 | Calls its own fallible `legend` first; that constructor supplies the numeric bound. |
| `hosted_renderer::execute` | 1 | Private route admission uses the same eight IDs as dispatch. The fallback assumes those declarations and arms agree. |
| `render::rank_bucket` | 1 | Private F32 calibration derives ranks within a complete finite histogram from a supported nonempty tensor. Refined ranks remain inside the selected bucket. This relies on the validated tensor and scan counts. |
| `render::legend`, statistics | 4 | Tensor-scoped rules first require selected-tensor statistics. Quantile, tensor-asinh and histogram facts in the ordered rule catalog determine which accesses follow. Global asinh takes its separate checkpoint branch. |
| `render::legend`, objects | 2 | Rule metadata and the added legend fields are constructed as JSON objects locally. |
| `render::mapping` | 1 | Repository render paths pass a legend containing numeric `max`. The function is public and does not validate this field for arbitrary library callers. |
| `render::lookup` | 2 | The public BF16 convenience helper requires a valid non-percentile rule and legend. BF16 mapping then has a table. Repository calls are in contract tests; this is insufficient evidence for deletion. |
| `server::collect_headers` | 1 | The preceding match handles `Ok(None)` separately. The mapped successful result therefore contains a completed header. The reader returns errors through the existing result path. |
| `server::reuse_transport`, returned admission | 1 | The same branch requires `deadline.is_some_and(...)` before taking the stored deadline. The local option is not changed between the check and access. |
| `source::Dtype::bits` | 2 | Repository scalar, rendering, comparison and profile callers provide exactly `dtype.bytes()` bytes. This public helper relies on that width contract for arbitrary library calls. |
| `state::State::{mapping,tensor_mapping}` | 2 | A position in the deque is found and removed while the same mutex guard remains held, with no intervening deque mutation. |
| `state::State::tensor_mapping`, histogram words | 1 | Length and checksum admission precede `chunks_exact(8)`. Each converted word is exactly eight bytes. |
| `state::State::model`, tensor serialization | 1 | The concrete tensor record derives serialization over its existing primitive/string/array fields. No custom fallible serializer is present in that record. This conclusion must be revisited if the record changes. |
| `state::State::model`, rule metadata | 1 | Iterates the same ordered catalog accepted by `rule_info`. The catalog and metadata lookup must stay consistent. |
| `state::TileCache::trim` | 1 | The loop first requires a nonempty entry map, then selects its minimum without intervening map mutation. |
| `state::TileCache::get_with` | 1 | The entry exists before the owned read closure; the closure has no mutable access to the map. Its successful read branch updates the same entry. |
| `strength::snapshot::number` | 1 | Private restore callers use fixed header offsets through 64 in the fixed header array, or offset 16 in a 24-byte sum record. Bounded exact reads precede decoding. |

These are caller and representation arguments, not proofs for arbitrary public
library inputs. In particular, public value-based mapping helpers and public
tensor/state fields leave responsibilities with library callers that differ
from the admitted CLI/HTTP paths. Changing these contracts, replacing panics
with errors or removing helpers would change behavior and requires its own
qualified change. No malformed-input reproduction is needed for this review.

## Lock behavior and ownership

The inventory has 50 `.lock()` sites. Their current poison handling is distinct:

| Behavior | Sites | Current locations |
| --- | ---: | --- |
| Unwrap acquisition result | 42 | Viewer/comparison state and progress, numeric serialization, lookup/tile caches, and the verification result publication. |
| Map poison to the existing error | 3 | Resource snapshot/reservation: `Resource accounting unavailable`; returned queue pop: `Reuse queue unavailable`. |
| Refuse admission through `None` | 2 | Reuse ledger admission and bounded queue insertion. |
| Recover only for owned release/removal | 3 | Resource permit drop, reuse lease drop and bounded queue removal. |

Poison cannot be declared unreachable from these source arguments. Several
state mutex fields are public, and the private locks can be held while other
code runs. The 42 unwrap sites retain their panic behavior on poison. Recovery
in the three cleanup paths does not establish that recovery is appropriate for
new admission or numerical publication.

Existing numerical writers acquire `compute` before their work. Viewer and
comparison calibration clone the current calibration, publish the replacement
file, then commit the replacement in memory. Metadata reads retain calibration
and progress while constructing their response; viewer metadata also reads the
tile cache and fresh hash result. Queue, ledger and cleanup guards protect their
own bounded accounting and release ownership. The returned queue removal drops
the removed item after releasing the queue guard, avoiding a nested lease drop
under that guard.

The reviewed paths preserve those orders and scopes. This is not a formal
deadlock or thread-schedule proof, and arbitrary external use of public state
locks has not been audited. Existing parallel resource/lease tests, calibration
publication tests and saved-cache upgrade checks remain the executable evidence
for their guarded scenarios.

## Reachability and unused-code decisions

The native crate exposes a library as well as the CLI. Public functions can be
used outside this repository, so a repository call search cannot prove them
unused. `render::lookup` is a concrete example: only contract tests call it here,
but it remains part of the library surface. Typed response construction is
referenced by scalar inspection and slice binding callers; command/route/rule
facts are referenced by their declared consumers. Feature-independent private
entrypoints and their callbacks also require indirect caller review.

The current offline build and test commands compile the existing surfaces;
compiler diagnostics and lexical references are useful inventory evidence.
They do not prove behavioral equivalence after deletion. No symbol has been
declared safe to delete, no code has been deleted, and unused-code proof remains
unfinished. A future deletion needs explicit visibility/caller evidence,
pre-change correctness qualification, an isolated commit and the complete
unchanged compatibility checks. Dependencies and all tests remain in scope.
