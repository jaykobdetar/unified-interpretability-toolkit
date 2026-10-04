# Third-party notices

## OpenSeadragon

`web/vendor/openseadragon.min.js` is OpenSeadragon 6.1.1, copied unchanged from the validated Python Weight Atlas frontend. Its original license is retained at `web/vendor/OpenSeadragon-LICENSE.txt`. The surrounding application frontend derives from that user-authorized reference and adds progressive calibration and immediate exact scalar inspection.

## Rust crates

`Cargo.lock` pins all 26 registry dependencies. Their original source, Cargo checksums, licenses and notices are retained under `vendor/`. They were copied from the installed crates.io cache and built offline. Build scripts inspect compiler/platform capabilities, generate Rust source or compile feature probes; no model code is executed.

## Model provenance

Model weights are excluded from this repository. The tested Qwen3-8B revision and expected hashes are documented in `docs/MODELS.md` and `docs/models/qwen3-8b.json`. Qwen3-8B is Apache-2.0 licensed. The project license does not replace licenses accompanying other models selected by users.

## Synthetic fixture

`fixtures/tiny-bf16/tiny.safetensors` is a small deterministic fixture generated for this project. It contains no downloaded or trained model weights. Its generation script and expected values are included.

## Optional experimental inference

HuggingFaceTB/SmolLM2-135M is Apache-2.0. Model/tokenizer files are not redistributed in this repository; provenance and SHA-256 values are in `docs/models/smollm2-135m.json`. The selected official source revision is `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.

The optional runtime uses separately installed PyTorch (BSD-style license and bundled third-party notices), Hugging Face Transformers (Apache-2.0), Tokenizers (Apache-2.0), Safetensors (Apache-2.0), and NumPy (BSD-3-Clause and bundled notices). These packages are not vendored or redistributed here; retain their installed license files. Direct tested version constraints are recorded in `inference-requirements.txt`. Playwright and Chromium are optional test tools, not runtime dependencies, and are not redistributed.

## Retained Qwen license notice

`docs/models/Qwen3-LICENSE.txt` preserves the original Qwen3-8B license artifact, including its Alibaba Cloud attribution, byte for byte. That model-specific artifact previously occupied the project LICENSE. The project retains its already-declared Apache-2.0 terms with the standard unfilled appendix example; this does not assert a new copyright owner. No model weights are distributed by this notice.
