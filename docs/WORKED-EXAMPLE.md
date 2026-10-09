# Reading a real Qwen weight window

This walkthrough uses the committed [Qwen observation fixture](../tests/fixtures/ui-polish-qwen-observations.json), recorded through bounded original BF16 reads without model execution. It pins Qwen/Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218` and source identity `cbf36b69cb8aab0ff2ec797fbd76cf756a0e72e151cdfac9c675d8bc9430e8f6`. A locally opened checkpoint must match the viewer's exact source/revision binding; matching a model name or shape is insufficient.

With that already available source selected, open **Start with a verified finding** or select the tensor and inclusive native bounds manually. Start with `model.layers.0.self_attn.q_proj.weight`, shape `[4096,4096]`, rows 128–135 and columns 0–7. Inspect individual cells to read their original decimals and BF16 bytes. Compare a signed view with tensor magnitude: sign distinguishes positive from negative values; magnitude shows absolute size. The source values do not change.

| Recorded native region | Observed values | Coverage |
| --- | --- | --- |
| `q_proj.weight`, rows 128–135, columns 0–7 | 28 positive, 36 negative; −0.0439453125 to 0.0556640625 | 64 of 16,777,216 values |
| `k_proj.weight`, rows 0–7, columns 0–7 | 36 positive, 28 negative; −0.05712890625 to 0.06103515625 | 64 of 4,194,304 values |
| `model.layers.0.input_layernorm.weight`, native indices 0–31 | All 32 positive; 0.00860595703125 to 0.0125732421875 | 32 of 4,096 values |

These observations establish native tensor names, shapes, coordinates, and stored values. They do not establish trusted attention-head labels. The plain static server has no bound head-layout descriptor, and the Qwen producer/consumer qualification remains incomplete; do not interpret these windows as verified head selections. Head labels require reviewed configuration/implementation evidence bound to the opened source, not color, tensor divisibility, or a matching model name. See the [head-layout contract](HEAD-LAYOUT-CONTRACT.md). The normalization tensor stays a one-dimensional vector displayed as one native row; the positive first-32 window does not establish signs for the rest of the tensor.

Use **Export original values** for the chosen region. CSV retains source bytes and coordinates; Numeric NumPy + metadata retains the numeric array and its provenance in two files. The same selected values should match the inspector after independent decoding. Bookmarking stores the source-bound view, not prompts, notes or a new claim that the checkpoint was authenticated upstream.

A large-magnitude cell is an outlier observation in its declared scope. It does not identify a learned concept, prove a head's importance or establish an effect on model behavior. Regional strength, shuffled controls and SVD describe the selected weights. A proper causal experiment needs the exact supported model/runtime, fixed prompt and matched contexts, a precisely defined edit, appropriate controls, restoration verification and bounded complete coverage. The current pinned SmolLM2 inference experiment is separate; editing a SmolLM2 coordinate cannot establish a cause in this Qwen checkpoint. Qwen viewer coordinates have no supported inference edit handoff in that runtime.

## One key-normalization vector

A separate bounded read of the already available checkpoint verified `model.layers.0.self_attn.k_norm.weight`: BF16, shape `[128]`, native index `[51]` (viewer row 0, column 51) is exactly **34**, with original little-endian bytes **`0842`**. It was read from `model-00001-of-00005.safetensors`; the 256-byte vector begins at absolute byte offset 1,546,675,320, and index 51 is at 1,546,675,422. The selected safetensors header contains data offsets `[1546665984,1546666240]`, following its eight-byte length prefix and 9,328-byte header. This binds the scalar to its stored tensor address, not a remembered value or synthetic display.

For an existing `MODEL_DIR` at the pinned revision, select this stored tensor and inspect native index 51. Compare the positive sign with the absolute magnitude, then export the original vector. The fresh full-vector read found 126 positive values, two negative values and no zeros; index 51 was the unique largest absolute value. Its minimum was −0.024169921875. This covers all 128 values of this vector, not the other normalization tensors or the model. It does not establish what the parameter does or imply that editing it would improve a prompt.

The vector-byte SHA-256 was `aaf5042c20082b5c13daed62ad9627f426cda5edc38eca48bd3f956da4eb05cd`. The bounded index/header read, header address, decoded bytes and stable before/after file identity are fresh observations. A cached provider manifest identifies `Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`; its expected shard size matched the local file. Its full-shard SHA and the earlier viewer source identity are historical references, not a fresh whole-shard hash, recomputed global viewer binding or upstream authentication. Check the current viewer's exact source/revision binding when reproducing the view.

The committed observation fixture still contains only the three windows above. The key-normalization result uses its own separately retained bounded byte evidence. Full-slice profile and broader workflow qualification remain separate.
