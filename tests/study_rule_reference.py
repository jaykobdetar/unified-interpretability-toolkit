#!/usr/bin/env python3
"""Compare robust99/rank to the original supplied color-study module.
Requires an existing NumPy environment and an explicitly supplied ATLAS_STUDY_ZIP.
Run only after the lead grants the heavy slot and this source has been built.
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
import hashlib, json, math, resource, struct, subprocess, sys, zipfile
from pathlib import Path

resource.setrlimit(resource.RLIMIT_AS, (768 * 1024**2, 768 * 1024**2))
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ZIP = Path(os.environ["ATLAS_STUDY_ZIP"]).resolve()
PREFIX = "weight-atlas-color-lab-code-only"
sys.path.insert(0, str(ZIP) + "/" + PREFIX)
from lab.rules import fit, apply

with zipfile.ZipFile(ZIP) as z:
    reference_sha = hashlib.sha256(z.read(PREFIX + "/lab/rules.py")).hexdigest()
OUT = ROOT / "results/study-rule-reference"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = OUT / "model"
MODEL.mkdir(exist_ok=True)
CACHE = OUT / "cache"
BIN = ROOT / "target/release/weight-atlas-rust"
records = []


def run(command, *args, ok=True):
    p = subprocess.run(
        [
            str(BIN),
            command,
            "--model",
            str(MODEL),
            "--cache",
            str(CACHE),
            *map(str, args),
        ],
        capture_output=True,
        text=True,
    )
    with (OUT / "commands.jsonl").open("a") as f:
        f.write(
            json.dumps(
                dict(
                    command=command,
                    args=args,
                    code=p.returncode,
                    stdout=p.stdout,
                    stderr=p.stderr,
                ),
                default=str,
            )
            + "\n"
        )
    if ok:
        assert p.returncode == 0, p.stderr
    return json.loads(p.stdout) if p.returncode == 0 else p


rng = np.random.default_rng(990032)
base = {
    "ties": np.array([[-4.0, -1.0, -0.0, 0.0, 1.0, 1.0, 2.0, 8.0]]),
    "odd": rng.choice([-4.0, -2.0, 0.0, 1.0, 2.0, 8.0], (17, 19)),
    "zero": np.zeros((3, 7)),
    "cancel": np.array([[-1.0, 1.0]]),
    "sparse_small": np.r_[np.zeros(1000), 0.5].reshape(1, -1),
    "sparse_large": np.r_[np.zeros(1000), 2.0].reshape(1, -1),
}
arrays = {}
header = {}
raw = bytearray()
for dtype in ["BF16", "F16", "F32"]:
    for name, x in base.items():
        name = dtype + "_" + name
        if dtype == "BF16":
            bits = (x.astype("<f4").view("<u4") >> 16).astype("<u2")
            values = (bits.astype("<u4") << 16).view("<f4").astype(float)
        elif dtype == "F16":
            bits = x.astype("<f2").view("<u2")
            values = bits.view("<f2").astype(float)
        else:
            bits = x.astype("<f4").view("<u4")
            values = bits.view("<f4").astype(float)
        arrays[name] = values
        start = len(raw)
        raw.extend(bits.tobytes())
        header[name] = dict(
            dtype=dtype, shape=list(bits.shape), data_offsets=[start, len(raw)]
        )
h = json.dumps(header).encode()
(MODEL / "fixture.safetensors").write_bytes(struct.pack("<Q", len(h)) + h + raw)
meta = run("metadata")
first_float = next(t for t in meta["catalog"] if t["dtype"] == "F32")
refused = run(
    "tile",
    "--tensor",
    first_float["id"],
    "--rules",
    "tensor_signed_percentile",
    "--out",
    OUT / "unsupported",
    ok=False,
)
assert refused.returncode != 0 and "unsupported for F32" in refused.stderr
run("calibrate")
stats = json.loads(json.loads((CACHE / "calibration.json").read_text())["payload"])[
    "tensors"
]
for t in meta["catalog"]:
    x = arrays[t["name"]]
    calibration = fit(x, 8.0)
    mappings = [("tensor_robust99", "signed_robust")]
    if t["dtype"] != "F32":
        mappings.append(("tensor_signed_percentile", "signed_rank"))
    else:
        inspected = run(
            "inspect",
            "--tensor",
            t["id"],
            "--left",
            "tensor_signed_percentile",
            "--right",
            "tensor_robust99",
        )
        assert (
            inspected["transformed"]["left"] is None
            and "exact absolute-value rank index"
            in inspected["transform_errors"]["left"]
        )
    for rule, reference_rule in mappings:
        transformed, info = apply(reference_rule, x, calibration)
        if rule == "tensor_robust99":
            np.testing.assert_allclose(
                stats[str(t["id"])]["q99"], calibration["Q99"], rtol=3e-16, atol=0
            )
            assert stats[str(t["id"])]["robust_clipped_count"] == round(
                info["clipped_fraction"] * x.size
            )
        for factor in sorted({1, 2, min(16, 2 ** t["max_level"]), 2 ** t["max_level"]}):
            for tx, ty in sorted(
                {
                    (0, 0),
                    (
                        (t["cols"] - 1) // (256 * factor),
                        (t["rows"] - 1) // (256 * factor),
                    ),
                }
            ):
                prefix = OUT / f"{t['name']}-{rule}-{factor}-{tx}-{ty}"
                result = run(
                    "tile",
                    "--tensor",
                    t["id"],
                    "--rules",
                    rule,
                    "--level",
                    t["max_level"] - int(math.log2(factor)),
                    "--x",
                    tx,
                    "--y",
                    ty,
                    "--out",
                    prefix,
                )
                region = transformed[
                    ty * 256 * factor : min(t["rows"], (ty + 1) * 256 * factor),
                    tx * 256 * factor : min(t["cols"], (tx + 1) * 256 * factor),
                ]
                want = np.array(
                    [
                        [
                            region[r : r + factor, c : c + factor].mean()
                            for c in range(0, region.shape[1], factor)
                        ]
                        for r in range(0, region.shape[0], factor)
                    ]
                )
                got = np.fromfile(str(prefix) + "-" + rule + ".f64le", "<f8").reshape(
                    want.shape
                )
                np.testing.assert_allclose(got, want, atol=2e-13, rtol=2e-13)
                records.append(
                    dict(
                        dtype=t["dtype"],
                        tensor=t["name"],
                        rule=rule,
                        factor=factor,
                        tile=[tx, ty],
                        error=float(np.abs(got - want).max()),
                    )
                )
report = dict(
    passed=True,
    study_module_sha256=reference_sha,
    cases=len(records),
    max_error=max(r["error"] for r in records),
    f32_percentile="explicitly unsupported; no approximate rank substitution",
    records=records,
)
(OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
