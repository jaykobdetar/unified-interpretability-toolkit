# Model preparation

No model download is needed for the included synthetic fixture. For your own checkpoint, obtain local safetensors files from a source you trust and retain its license. The viewer reads supported numeric tensors; it does not import model repository code or load pickle. Supported formats and bounds are in [FORMATS.md](FORMATS.md).

```bash
./run-atlas.sh /path/to/your/safetensors-model
```

The retained provenance manifests record exact prior source revisions, expected file sizes and SHA-256 hashes:

- [Qwen3-8B manifest](models/qwen3-8b.json): view-only source, revision `b968826d9c46dd6066d109eabc6255188de91218`.
- [SmolLM2-135M manifest](models/smollm2-135m.json): the only supported experimental inference source, revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.

These manifests are functional inputs to source verification and inference architecture binding. Keep them and `smollm2-135m-config.json` with the code. A matching local hash establishes correspondence to the recorded bytes, not a new upstream authenticity or license review. A tensor subset is insufficient for inference, which verifies all required pinned files.

For existing Qwen files, `python3 tools/check_model_hashes.py /path/to/qwen3-8b` verifies recorded sizes and hashes with bounded reads. This is a full-file scan and can take time. Neither that script nor the launcher downloads weights. Allow enough storage for the checkpoint plus the unchanged 25 GiB free-disk reserve.

Model files are ignored by Git. Source paths, local cache receipts, and model-specific metadata should remain local. The repository's license does not grant rights to a separately obtained model; check its own terms before use or redistribution.
