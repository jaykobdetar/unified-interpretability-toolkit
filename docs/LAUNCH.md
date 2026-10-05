<a name="launching-weight-atlas"></a>

# Launching Unified Interpretability Toolkit

Run commands from the repository root. Linux is required by the current CPU, process, and memory guards. The launcher uses Python's standard library; Rust dependencies are vendored and builds use `--offline --locked -j 1`.

## Viewer: shortest path

```bash
./run-atlas.sh --demo --build
# Later, without rebuilding:
./run-atlas.sh --demo
```

Open `http://127.0.0.1:8775`. The tiny fixture has a 3×5 matrix, a vector, and zeros. Local tensor calibration occurs when selected; whole-model calibration is an explicit action. This command starts only the weight viewer. Ctrl-C stops the foreground process.

```bash
./run-atlas.sh /path/to/model --port 8776 --cache ./cache-other
./run-atlas.sh --demo --check
./run-atlas.sh --help
```

`--check` validates paths, port, platform, and binary/build dependencies without starting a process or reading model payloads. Runtime source, memory, disk, cache-lock, and port checks still run at launch. `--build` explicitly rebuilds; without it a missing binary produces instructions. No dependency installer or model downloader runs.

## Portable configuration

Copy `config/viewer.example.json` to ignored `config/viewer.local.json` and set your model/cache paths, port, display name, and revision. Then:

```bash
./run-atlas.sh --config config/viewer.local.json
```

Optional standalone `resources` budgets are described in [resource configuration](RESOURCES.md). Defaults preserve the original finite limits; startup admission also checks process budget plus the effective RAM floor.

Precedence is CLI options, `ATLAS_MODEL/ATLAS_CACHE/ATLAS_PORT/ATLAS_NAME/ATLAS_REVISION`, then JSON values. `ATLAS_CONFIG` selects a config file. `--demo` explicitly selects the checked-in fixture. Paths in JSON resolve relative to the config file; CLI/environment paths resolve relative to your current directory. The default cache is the checkout's ignored `cache/` directory. Configuration is parsed as JSON and never executed. Cache paths inside the selected model are refused. Use separate caches for concurrent viewers.

## Inference is a separate process

Only the complete pinned SmolLM2-135M source is supported. Obtain it separately under its model license, and select an existing compatible CPU Python environment. Versions and file pins are in [INFERENCE.md](INFERENCE.md), `inference-requirements.txt`, and `docs/models/`.

```bash
python3 tools/live_inference.py \
  --model /path/to/smollm2-135m \
  --python /path/to/cpu-environment/bin/python \
  --port 8796 --atlas-port 8797
```

Open the **coordinator on port 8796**. Port 8797 is its internal weight viewer. Both ports must be free. The runtime verifies pinned files and uses built-in model code; it does not execute downloaded model code. Edits are in RAM; original files remain unchanged. Baseline and edited runs are sequential. Playback replays completed records; it does not pause compute.

The coordinator admits work at 4.75 GiB available RAM and stops owned work below 3.25 GiB. Inference retains one CPU/model worker, 1.5 GiB sampled worker RSS, 128 prompt tokens, at most 32 generated tokens per branch, 120 seconds wall time, and 90 CPU seconds. These bounds do not guarantee that every workload completes. Cancellation/reset must confirm owned cleanup before replacement. Ctrl-C stops only this coordinator's owned children.

For bounded BF16 analytics without inference, add `--analytics-only`, select your BF16 source, and provide an explicit name/revision. Analytics has separate source/extent checks and bounded work; it does not support F16/F32 or checkpoint-pair sources. See [the analytics contract](analytics/CONTRACT.md).

The owner registry and experimental fixture-profile launchers are documented in [ARCHITECTURE.md](ARCHITECTURE.md). They are not part of the basic viewer quickstart.

## Troubleshooting

| Message or symptom | Action |
| --- | --- |
| Missing Python or Cargo | Install the prerequisite, then retry explicitly. Rust 1.92.0 is the tested toolchain. |
| Missing viewer binary | Run `./run-atlas.sh --demo --build`. |
| Memory/disk guard refusal | Free resources or use a machine with sufficient capacity. The build needs 5 GiB available RAM and 25 GiB free disk; native default admission needs 3.75 GiB effective available RAM (3 GiB reserve plus a 768 MiB process budget) and the same disk reserve. Optional standalone budgets are described in RESOURCES.md. |
| Port occupied | Choose a free `--port`; do not stop an unrelated service. |
| Cache locked | Stop the viewer that owns it, or choose another cache directory. |
| No color view yet | Wait for selected-tensor calibration. Raw scalar inspection is available first. Global rules require complete calibration. |
| Inference controls absent | You opened the viewer-only launcher. Use the pinned inference coordinator for generation. |
| Unsupported tensor | Check its dtype/shape and unavailable reason. No approximate decoding or silent model substitution occurs. |

Never expose these loopback services through a public proxy without a separate security design. Local metadata can contain source paths; review exported records and screenshots before sharing.
