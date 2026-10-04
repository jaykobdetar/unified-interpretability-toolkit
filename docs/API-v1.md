# Frozen frontend contract, v1

All endpoints are loopback GETs. IDs are numeric catalog IDs. Errors return JSON `{error: string}` with HTTP 400; missing completed statistics returns 503. Every global legend is available only after the complete inventory pass. No model inference occurs.

`GET /api/model` returns:
```
{
  "api_version": 1,
  "name": "Qwen3-8B",
  "revision": "b968826d9c46dd6066d109eabc6255188de91218",
  "representation": "Original BF16 source values; no model execution",
  "parameter_count": 8190735360,
  "global_max": 0,
  "calibration_complete": true,
  "source_directory": ".../source",
  "source_bytes": 16381516776,
  "catalog": [{"id":0,"name":"...","shape":[4096,4096],"rows":4096,"cols":4096,"count":16777216,"max_abs":0,"max_level":12,"min_level":0}],
  "rules": [{"id":"global_linear","title":"Global linear","formula":"clip(x/G, -1, 1)"}],
  "coverage": {
    "source_complete": true,
    "sha_verified_shards": 5,
    "statistics_complete": true,
    "values_streamed": 8190735360,
    "rendering_policy": "Bounded numeric pyramid; fine tiles generated on demand",
    "materialized_tiles": 0,
    "materialized_bytes": 0,
    "all_pixels_materialized": false
  },
  "render_semantics": "At pyramid level L, each pixel averages the selected transformed field over aligned factor-by-factor source blocks; factor = 2^(max_level-L). Edges count actual source addresses only."
}
```
The sample's `global_max` and `max_abs` zeroes and first tensor shape/name are schema placeholders, not measured results. Actual JSON contains all measured values and all catalog tensors. Rule IDs: `global_linear`, `global_asinh`, `tensor_linear`, `tensor_asinh`.

`GET /api/view?tensor=0&left=global_linear&right=global_asinh` returns:
```
{
 "api_version":1,
 "tensor":{"id":0,"name":"...","shape":[4096,4096],"rows":4096,"cols":4096,"count":16777216,"max_level":12,"min_level":0},
 "legends":{
   "left":{"id":"global_linear","title":"Global linear","formula":"clip(x/G, -1, 1)","min":-1,"zero":0,"max":1,"s":null,"scope":"complete checkpoint","units":"raw weight"},
   "right":{"id":"global_asinh","title":"Global asinh","formula":"clip(asinh(x/s)/asinh(G/s), -1, 1); s = G/100","min":-1,"zero":0,"max":1,"s":0.01,"scope":"complete checkpoint","units":"raw weight on nonlinear scale"}
 },
 "tile_size":256,"overlap":0,"source_values_unchanged":true
}
```
Legend numbers above are schema examples. Actual bounds and s must be displayed from the response without fixed decimal truncation.

`GET /tile?tensor=0&rule=global_linear&level=12&x=0&y=0` returns a PNG (edge tiles can be smaller). `max_level=ceil(log2(max(rows,cols)))`; factor `2^(max_level-level)`. Native scalar level is `max_level`. Image level dimensions use ceiling division. A level is a numeric mean of the transformed field, never a resized scalar PNG. Colors use the same blue-neutral-red palette as the small lab.

`GET /api/inspect?tensor=0&row=0&col=0&left=global_linear&right=global_asinh` returns:
```
{"api_version":1,"tensor":0,"row":0,"col":0,"raw_exact":"0.125","bf16_hex_le":"003e","shard":"model-00001-of-00005.safetensors","byte_offset":1234,"native_indices":[0,0],"transformed":{"left":0.1,"right":0.2}}
```
These are schema-only example values/offsets; the actual response reads the original two bytes and gives real numeric fields. Both panels inspect this single unchanged source address. A vector has shape [N], rows=1, cols=N and native_indices=[column]. The viewer must not invent a second source or a trained-checkpoint difference.

An image clicked in a pooled view may select a scalar coordinate via native image coordinates. The inspector labels it as one exact address rather than the pooled mean. Requested pyramid levels and display magnification are separate: fit can hide cells even at scalar resolution. Do not infer all model pixels are rendered from complete source/statistics coverage.
