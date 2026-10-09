# Screenshot sources

The `smol-*.png` images are direct browser captures of the application at commit `a6053703b680e7707cd8c180c608599dc1903e50`, taken on October 9, 2026 in local Chrome 155.0.8059.39. The browser used the real Rust viewer and existing local model files. No inference, synthetic replacement values, injected UI content, generated imagery, or pixel editing was used. Source-path details stayed collapsed; all published images were visually inspected.

## Real checkpoint

- Model: `HuggingFaceTB/SmolLM2-135M`.
- Revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
- File: `model.safetensors`, 269,060,552 bytes.
- File SHA-256: `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`.
- Tensor: `model.layers.0.self_attn.q_proj.weight`, BF16, `[576,576]`.
- Left rule: **Tensor asinh** (`tensor_asinh`). Right rule: **Tensor typical magnitude** (`tensor_magnitude_asinh`).

The original file hash matches the project's retained model pin. This identifies the pictured source, not a claim that static viewing qualifies arbitrary inference or head-layout interpretation. Weights are not distributed with these screenshots.

| Image | View |
| --- | --- |
| [smol-overview.png](smol-overview.png) | Full tensor, rows/columns 0–575; image cells pool 2×2 source blocks. The small navigator separately uses 4×4 blocks. |
| [smol-detail.png](smol-detail.png) | Workspace capture after focusing rows/columns 0–15. Aspect-ratio fitting displays rows 0–16 and columns 0–21; badges confirm scalar-level cells. |
| [smol-inspector.png](smol-inspector.png) | Inspector capture at native `[0,0]`: exactly −0.08935546875, BF16 bytes `b7bd`, absolute file offset 62,849,096. |

To reproduce with your existing copy, [launch the viewer](../LAUNCH.md), select the named tensor, choose the two rules, and wait until neither resolution badge says loading. Use **Fit tensor** for the overview. Under **Save, annotate & export a region**, enter the bounds above and choose **Focus region** for detail; enter row 0 and column 0 under **Exact scalar** for inspection. Viewport dimensions affect the visible bounds. The original captures use a 1600×1400 browser viewport; detail and inspector are direct element screenshots.

Only the selected tensors were calibrated. The visible incomplete-global-calibration notice is expected: tensor-local rules work without scanning the entire checkpoint. Both panels are the same checkpoint under different rules, not a before/after checkpoint pair.

## Synthetic demo captures

`viewer-demo.png` and `viewer-inspector.png` are retained older captures of the 26-value BF16 fixture. They show the previous project display name and are not trained-model results or the current README illustrations.
