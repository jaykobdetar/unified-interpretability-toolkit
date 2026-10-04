# Data and viewer contract

## Selection and availability

Frozen binding example: [SOURCE-BINDING-V2.json](SOURCE-BINDING-V2.json). The exact keys are version, model_identity, source_identity, tensor, name, dtype, shape, rows, cols and slice. Slice has exactly leading_indices and display_axes. Shape is always the complete original native shape. Rank-one canonical slice is `{ "leading_indices": [], "display_axes": [0] }`; rank-two is `{ "leading_indices": [], "display_axes": [0,1] }`. No paths, source byte offsets or human revision text enter this binding. Source/model identities bind those separately. Rich slice metadata (hashed identity, element start, count, byte offset) is separate from the canonical binding. Consumers compare both slice arrays and every native field; JSON object key order is immaterial. Every new inspect/view/profile response uses binding v2, including rank one/two; legacy 2D request URLs still work. Existing 2D bookmarks remain version 2; higher-rank bookmarks use version 3 with required `i` leading indices. Legacy 2D note keys remain unchanged.

The catalog retains every structurally valid tensor, including unavailable storage. Supported numeric decoding remains exactly BF16, F16 and F32. Known storage widths validate extents but do not authorize numeric interpretation or dequantization. Unknown encodings retain their declared extents, explicitly marked unvalidated storage semantics; overlapping, gapped, out-of-file and overflowing extents still reject the file. Scalar, empty and oversized display axes are unavailable rather than silently reshaped.

Higher ranks use the final two axes as rows and columns. `slice=2,1` fixes the leading axes to those exact indices. Every leading index is required; no implicit zero slice. Arbitrary axis selection/transposition is unsupported. Rank one/two retain their current URLs and one-row vector convention. Their slice is canonically empty.

Original tensor shape/count remain native. A selected slice has its own leading indices, element start, 2D count and byte origin. All tile, inspect, export, bookmark, note and profile bindings include slice identity. Inspector indices concatenate leading indices and row/column. Raw dtype/bytes/decimal and original shard offset remain authoritative. Source mutation invalidates everything. Inference edit handoff and legacy window analytics are disabled for higher-rank mappings until explicitly supported.

Calibration remains over the complete original tensor, not a silently selected slice. Legends say so. Calibration cache stays tensor-scoped; pixel/profile caches and navigation state are slice-scoped. Unsupported tensors cannot be calibrated; global calibration is unavailable when any tensor is unavailable. Supported tensors remain individually usable.

## Whole-slice strength profiles

Core API: construct one source/slice/seed-bound accumulator, then explicitly advance bounded chunks. A chunk accepts a remaining value budget and a wall-time deadline. No arbitrary total matrix window truncation and no automatic indefinite continuation. A caller-authorized total scan/time allowance spans all chunks; reaching it returns partial data and requires a new explicit user action to extend it. The separate fixture-profile coordinator integrates these primitives under explicit owner grants.

Progress is a small record: binding, state, visited/total values, elapsed active time, remaining authorized work, error and completion. Row/column output is paged (at most 1,024 entries per response). Each original/control entry includes native axis index, sum of absolute finite values, visited count, expected count, mean (null if unvisited) and complete flag. Partial results describe observed cells only; prefix coverage is not representative sampling. No sorting/reordering in this stage.

Original and control consume exactly the same visited source values. Control uses a seeded bijective permutation of flattened destination coordinates, with its algorithm and limitations declared. Partial original/control axes may have different coverage; final counts must equal the full corresponding axis length. No significance, concept or causal claims. Raw source coordinates never change.

One accumulator, O(rows+columns) sums/counts, checked allocation estimate and a fixed 32 MiB profile-state ceiling within the existing process limit. Read buffers remain below 2 MiB; bounded contiguous input batches. Progress never serializes the whole profile. Chunk limits are scheduling limits, not claims of full coverage. No persistent profile cache initially; disk-cache policy requires separate review.

## Optional typical magnitude rule

`tensor_magnitude_asinh`: `min(1, asinh(abs(x)/s) / asinh(D/s))`, where `s` is the exact median of nonzero absolute values, or 1 for an all-zero tensor, and `D` is exact Q99 of all absolute values (including zeros), or 1 when Q99 is zero. Quantiles use the existing linear interpolation of exact order statistics, including ties. The existing Q99 fallback clipping count applies. Both signs of zero map to zero; an all-zero tensor is uniformly zero; nonfinite input refuses calibration/rendering. Transform each scalar before pooling. Pooled units are mean transformed magnitude, not mean raw magnitude. Colors are sequential purple. The max-normalized `tensor_magnitude` rule remains available unchanged. This pointwise rule is permutation-equivariant and uses the same scale for any matched shuffled field.

## Quantization decision

No quantized weights are decoded in this stage. INT8/UINT8, packed INT32 GPTQ/AWQ, bitsandbytes codebooks, scaled FP8 and sub-byte float storage need exact encoding, packing, grouping, zero-point and scale semantics, plus tests for that exact declared schema. Storage dtype alone is insufficient. Related scale/codebook tensors are never inferred by name. Unsupported entries explain this limitation instead of being presented as original floating weights.
