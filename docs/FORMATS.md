# Formats and color semantics

Supported numerical storage is BF16, F16, and F32 safetensors. Checkpoints may contain 1–64 shards and an optional matching `model.safetensors.index.json`. Headers are limited to 8 MiB each / 32 MiB combined, at most 10,000 tensors, and display dimensions from 1 to 200,000. Duplicate names/keys, invalid or overlapping/gapped extents, mismatched indexes, and shard/index symlinks are rejected.

Vectors use one native row. Matrices preserve rows and columns. Rank 3–32 tensors require explicit leading indices and display their final two axes; arbitrary axis selection is unsupported. Scalar, empty, oversized, quantized, F64, and unknown storage entries remain unavailable instead of being silently reshaped or approximated. Nonfinite values are inspectable as original bits, but calibration/rendering refuses them.

The inspector reports dtype, original little-endian bytes, shard, absolute file offset, native indices, and the full mathematical decimal value, preserving signed zero. A pooled image click selects one original address. Neutral color is not proof of exact zero.

## Eight rules

| Rule | Meaning |
| --- | --- |
| Global linear / asinh | Signed transform using complete-checkpoint bounds; unavailable until full calibration. |
| Tensor linear / asinh | Signed transform using complete-tensor bounds; independently calibrated tensors have different scales. |
| Tensor magnitude | `abs(x)/M`, where `M` is the complete tensor's exact maximum magnitude; all zero when `M=0`. |
| Tensor typical magnitude | `min(1, asinh(abs(x)/s)/asinh(D/s))`, with exact median nonzero magnitude `s` and Q99 `D`; zero fallbacks are 1. |
| Tensor robust 99% | Clip `x/D` to ±1, with exact Q99 magnitude `D` or 1 when Q99 is zero. |
| Tensor signed percentile | Signed mid-CDF magnitude rank, equal ranks for ties, both zeros map to zero; BF16/F16 only. F32 is unavailable. |

The actual legend states bound, scale, scope, clipping, units and availability. Magnitude removes sign only from color; inspection still reports the original signed value. Quantiles interpolate exact original order statistics, including zeros where specified. F32 order statistics use bounded two-pass radix selection; F32 unique-pattern count is unavailable.

For pyramid level `L`, pooling factor is `2^(max_level-L)`. Transform every original scalar first, then average over the aligned block. Partial edge blocks count only actual source addresses. Consequently magnitude pooling is `mean(abs(x))/M`, never `abs(mean(x))/M`. Floating-point accumulation and library rounding can produce small differences from independent reference implementations.

These are views of numerical data. A stripe, extreme, head boundary, or shuffled control is not by itself evidence of a learned concept or causal importance.
