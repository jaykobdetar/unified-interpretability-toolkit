#!/usr/bin/env python3
"""Compare original Python numeric implementation with Rust; write only this project."""

import os, sys, json, struct, subprocess, zlib, math, time, hashlib, shutil
from pathlib import Path

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
ROOT = Path(__file__).resolve().parents[1]
REF = Path(os.environ["ATLAS_PYTHON_REFERENCE"]).resolve()
sys.path.insert(0, str(REF))
from atlas8.limits import configure

configure()
import numpy as np
from PIL import Image
from unittest.mock import patch
from atlas8 import render, source

BIN = ROOT / "target/release/weight-atlas-rust"
OUT = ROOT / "results/parity"
OUT.mkdir(exist_ok=True)
MODEL = OUT / "model"
MODEL.mkdir(exist_ok=True)
CACHE = OUT / "cache"
records = []
failures = []


def run(cmd, model=MODEL, cache=CACHE, check=True):
    p = subprocess.run(
        [str(BIN), cmd[0], "--model", str(model), "--cache", str(cache), *cmd[1:]],
        capture_output=True,
        text=True,
    )
    with (OUT / "commands.log").open("a") as f:
        f.write(
            json.dumps(
                {
                    "cmd": cmd,
                    "returncode": p.returncode,
                    "stdout": p.stdout,
                    "stderr": p.stderr,
                }
            )
            + "\n"
        )
    if check:
        assert p.returncode == 0, p.stderr
    return json.loads(p.stdout) if p.returncode == 0 else p


def write_model(path, ts):
    h = {}
    raw = bytearray()
    for name, bits in ts.items():
        a = len(raw)
        raw.extend(bits.astype("<u2").tobytes())
        h[name] = {
            "dtype": "BF16",
            "shape": list(bits.shape),
            "data_offsets": [a, len(raw)],
        }
    header = json.dumps(h, separators=(",", ":")).encode()
    header += b" " * ((-len(header)) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + raw)


rng = np.random.default_rng(817)
arrays = {}
for name, shape in [
    ("matrix", (577, 579)),
    ("odd", (17, 19)),
    ("band", (299, 8192)),
    ("vector", (579,)),
    ("zero", (9,)),
    ("single", (1,)),
]:
    x = (
        (rng.integers(-255, 256, shape).astype(np.float32) / 64)
        if name not in ["zero", "single"]
        else np.zeros(shape, np.float32)
    )
    arrays[name] = (x.view(np.uint32) >> 16).astype("<u2")
arrays["extreme"] = np.array(
    [0, 32768, 1, 32769, 0x7F7F, 0xFF7F, 0x3F80, 0xBF80], dtype="<u2"
)
write_model(
    MODEL / "first.safetensors", {k: arrays[k] for k in ["matrix", "odd", "band"]}
)
write_model(
    MODEL / "second.safetensors",
    {k: arrays[k] for k in ["vector", "zero", "single", "extreme"]},
)
index = {
    "weight_map": {
        k: (
            "first.safetensors"
            if k in ["matrix", "odd", "band"]
            else "second.safetensors"
        )
        for k in arrays
    }
}
(MODEL / "model.safetensors.index.json").write_text(json.dumps(index))
meta = run(["metadata"])
cal = run(["calibrate"])
payload = json.loads(json.loads((CACHE / "calibration.json").read_text())["payload"])
G = cal["model"]["global_max"]
for t in meta["catalog"]:
    st = payload["tensors"][str(t["id"])]
    bits = arrays[t["name"]]
    x = source.values_from_bits(bits)
    hist = np.frombuffer(
        zlib.decompress(
            (
                CACHE
                / "histograms"
                / f"{meta['source_identity']}-{t['id']:04}.u64le.zlib"
            ).read_bytes()
        ),
        dtype="<u8",
    )
    np.testing.assert_array_equal(hist, np.bincount(bits.ravel(), minlength=65536))
    assert st["max_abs"] == float(np.abs(x).max())
    nonzero = np.abs(x[x != 0]).astype(np.float64)
    assert st["median_nonzero_abs"] == (
        float(np.median(nonzero)) if len(nonzero) else 0.0
    )
    pt = {**t, **st}
    pt["shape"] = t["shape"]
    factors = sorted(set([1, 2, 4, min(32, 2 ** t["max_level"]), 2 ** t["max_level"]]))
    for factor in [f for f in factors if f <= 2 ** t["max_level"]]:
        level = t["max_level"] - int(math.log2(factor))
        coords = [(0, 0)]
        if factor == 1 and t["rows"] > 512 and t["cols"] > 512:
            coords.append((2, 2))
        for tx, ty in coords:
            prefix = OUT / f"{t['name']}-f{factor}-{tx}-{ty}"
            r = run(
                [
                    "tile",
                    "--tensor",
                    str(t["id"]),
                    "--rules",
                    ",".join(render.RULES),
                    "--level",
                    str(level),
                    "--x",
                    str(tx),
                    "--y",
                    str(ty),
                    "--out",
                    str(prefix),
                ]
            )
            with patch.object(render, "SOURCE", MODEL):
                fields, _ = render.tile_fields(pt, list(render.RULES), G, level, tx, ty)
            for rule in render.RULES:
                a = np.fromfile(
                    str(prefix) + "-" + rule + ".f64le", dtype="<f8"
                ).reshape(fields[rule].shape)
                b = fields[rule]
                err = float(np.max(np.abs(a - b)))
                np.testing.assert_allclose(a, b, atol=2e-13, rtol=2e-13)
                got = np.array(Image.open(str(prefix) + "-" + rule + ".png"))
                want = render.rgba(b)
                channel = int(np.max(np.abs(got.astype(int) - want.astype(int))))
                assert channel <= 1
                records.append(
                    {
                        "tensor": t["name"],
                        "shape": t["shape"],
                        "factor": factor,
                        "tile": [tx, ty],
                        "rule": rule,
                        "max_abs_field_error": err,
                        "max_channel_difference": channel,
                        "differing_channels": int(np.count_nonzero(got != want)),
                        **r["metrics"],
                    }
                )
    # Exact BF16 checks at first, last, seeded internal addresses, including zero/sign/subnormal/extrema.
    positions = set([0, t["count"] - 1, *rng.integers(0, t["count"], size=12).tolist()])
    for pos in positions:
        row, col = divmod(pos, t["cols"])
        r = run(
            ["inspect", "--tensor", str(t["id"]), "--row", str(row), "--col", str(col)]
        )
        with patch.object(source, "SOURCE", MODEL):
            ref = source.scalar(pt, row, col)
        assert (
            r["bf16_hex_le"] == ref["hex_le"]
            and r["byte_offset"] == ref["byte_offset"]
            and float(r["raw_exact"]) == ref["value"]
        )
        assert r["native_indices"] == ref["source_indices"]
# Persistent reuse preserves calibrated values; mutation with restored mtime invalidates.
assert run(["calibrate"])["model"]["calibration_complete"]
p = MODEL / "first.safetensors"
old = p.stat()
data = p.read_bytes()
changed = bytearray(data)
changed[-2] ^= 1
p.write_bytes(changed)
os.utime(p, ns=(old.st_atime_ns, old.st_mtime_ns))
assert run(["inspect", "--tensor", "0"])["transforms_ready"] is False
p.write_bytes(data)
# Malformed fixture cases, no source file access outside explicit chosen directory.
bad = OUT / "malformed"
bad.mkdir(exist_ok=True)


def reject(name, body, index_text=None):
    for f in bad.iterdir():
        f.unlink()
    (bad / "bad.safetensors").write_bytes(body)
    if index_text:
        (bad / "model.safetensors.index.json").write_text(index_text)
    r = run(["metadata"], model=bad, check=False)
    assert hasattr(r, "returncode") and r.returncode != 0, name
    failures.append({"case": name, "expected_rejection": r.stderr.strip()})


def packed(h, raw=b"\0\0"):
    return struct.pack("<Q", len(h)) + h + raw


reject(
    "duplicate keys",
    packed(
        b'{"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}'
    ),
)
reject(
    "overlap",
    packed(
        json.dumps(
            {
                k: {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}
                for k in ["a", "b"]
            }
        ).encode()
    ),
)
reject("dtype", packed(b'{"a":{"dtype":"F64","shape":[1],"data_offsets":[0,2]}}'))
reject("truncated", packed(b'{"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}'))
reject(
    "path outside root",
    packed(b"{}"),
    json.dumps({"weight_map": {"a": "../outside.safetensors"}}),
)
reject(
    "dimension overflow",
    packed(
        b'{"a":{"dtype":"BF16","shape":[18446744073709551615,2],"data_offsets":[0,2]}}'
    ),
)
# Nonfinite source may open metadata/inspect but cannot claim completed calibration.
for f in bad.iterdir():
    f.unlink()
write_model(bad / "bad.safetensors", {"nan": np.array([0x7FC0], "<u2")})
r = run(["calibrate"], model=bad, cache=OUT / "nonfinite-cache", check=False)
assert r.returncode != 0
result = {
    "passed": True,
    "field_cases": len(records),
    "max_abs_field_error": max(r["max_abs_field_error"] for r in records),
    "max_channel_difference": max(r["max_channel_difference"] for r in records),
    "differing_channels_total": sum(r["differing_channels"] for r in records),
    "records": records,
    "expected_rejections": failures,
    "scope": "Deterministic fixtures, sharded index, exact histograms/scalars, all four rules, vectors/odd/edge/zero/extreme; F64 pooling and libm differences allowed <=2e-13.",
}
(OUT / "report.json").write_text(json.dumps(result, indent=2))
print(
    json.dumps(
        {
            k: v
            for k, v in result.items()
            if k not in ["records", "expected_rejections"]
        },
        indent=2,
    )
)
