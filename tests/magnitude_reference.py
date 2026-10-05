#!/usr/bin/env python3
"""Independent NumPy oracle; fixtures or one immutable real tensor. No model execution.
Run after receiving the shared heavy-test slot. Uses bounded reads and the normal Rust guards.
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
import argparse
import hashlib
import json
from pathlib import Path
import resource
import shutil
import socket
import struct
import subprocess
import time
import urllib.request

resource.setrlimit(resource.RLIMIT_AS, (768 * 1024**2, 768 * 1024**2))
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", type=Path)
parser.add_argument(
    "--port",
    type=int,
    default=0,
    help="Owned loopback test port; 0 selects an ephemeral port",
)
parser.add_argument("--tensor-name", default="model.layers.0.self_attn.q_proj.weight")
parser.add_argument("--out", type=Path, default=ROOT / "results/magnitude-reference")
args = parser.parse_args()
OUT = args.out.resolve()
OUT.mkdir(parents=True, exist_ok=True)
BIN = ROOT / "target/release/weight-atlas-rust"
CACHE = OUT / "cache"
records = []


def guard():
    available = (
        int(
            next(
                x
                for x in Path("/proc/meminfo").read_text().splitlines()
                if x.startswith("MemAvailable:")
            ).split()[1]
        )
        * 1024
    )
    assert available >= 3 * 1024**3
    assert shutil.disk_usage(OUT).free >= 25 * 1024**3


def run(*cmd):
    guard()
    p = subprocess.run(
        [str(BIN), *map(str, cmd), "--model", str(MODEL), "--cache", str(CACHE)],
        capture_output=True,
        text=True,
    )
    with (OUT / "commands.jsonl").open("a") as f:
        f.write(
            json.dumps(
                dict(
                    command=cmd,
                    returncode=p.returncode,
                    stdout=p.stdout,
                    stderr=p.stderr,
                ),
                default=str,
            )
            + "\n"
        )
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


def decode(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def write_fixture():
    rng = np.random.default_rng(52937)
    arrays = {
        name: (rng.integers(-511, 512, size=shape).astype(np.float32) / 128)
        for name, shape in [
            ("random", (577, 579)),
            ("edge", (17, 19)),
            ("band", (129, 8193)),
            ("vector", (579,)),
        ]
    }
    arrays.update(
        cancellation=np.array([[1, -1], [-1, 1]], np.float32),
        zero=np.zeros((3, 7), np.float32),
        zero_vector=np.zeros(9, np.float32),
    )
    header, raw = {}, bytearray()
    for name, a in arrays.items():
        start = len(raw)
        raw.extend((a.view(np.uint32) >> 16).astype("<u2").tobytes())
        header[name] = dict(
            dtype="BF16", shape=list(a.shape), data_offsets=[start, len(raw)]
        )
    h = json.dumps(header).encode()
    (MODEL / "fixture.safetensors").write_bytes(struct.pack("<Q", len(h)) + h + raw)


def rows(t, start, end, c0=0, c1=None):
    guard()
    c1 = t["cols"] if c1 is None else c1
    assert (end - start) * (c1 - c0) <= 1048576
    with (MODEL / t["shard"]).open("rb") as f:
        a = np.empty((end - start, c1 - c0), np.float64)
        for r in range(start, end):
            f.seek(t["byte_offset"] + 2 * (r * t["cols"] + c0))
            raw = f.read(2 * (c1 - c0))
            assert len(raw) == 2 * (c1 - c0)
            a[r - start] = decode(np.frombuffer(raw, "<u2"))
    return a


def exact_max(t):
    maximum = 0.0
    step = max(1, 1048576 // t["cols"])
    for start in range(0, t["rows"], step):
        maximum = max(
            maximum, float(np.abs(rows(t, start, min(start + step, t["rows"]))).max())
        )
    return maximum


def reference(t, factor, tx, ty, maximum, magnitude):
    r0, c0 = ty * 256 * factor, tx * 256 * factor
    r1, c1 = min(t["rows"], r0 + 256 * factor), min(t["cols"], c0 + 256 * factor)
    field = np.zeros(
        ((r1 - r0 + factor - 1) // factor, (c1 - c0 + factor - 1) // factor)
    )
    # One row at a time: no complete real tensor, padded addresses, or mean-of-means.
    for r in range(r0, r1):
        x = rows(t, r, r + 1, c0, c1)[0]
        if magnitude:
            x = np.abs(x)
        sums = np.add.reduceat(x, np.arange(0, len(x), factor))
        field[(r - r0) // factor] += sums
    count_r = np.minimum(factor, r1 - r0 - np.arange(field.shape[0]) * factor)
    count_c = np.minimum(factor, c1 - c0 - np.arange(field.shape[1]) * factor)
    return (
        field / count_r[:, None] / count_c[None, :] / maximum if maximum else field * 0
    )


MODEL = args.model.resolve() if args.model else OUT / "model"
if not args.model:
    MODEL.mkdir(exist_ok=True)
    write_fixture()
meta = run("metadata")
selected = [
    t for t in meta["catalog"] if not args.model or t["name"] == args.tensor_name
]
assert selected
if args.model:
    assert selected[0]["shape"] == [
        4096,
        4096,
    ], "Real-matrix acceptance requires native 4096×4096"
for t in selected:
    run("calibrate", "--tensor", t["id"])
    maximum = exact_max(t)
    factors = (
        [1, 8, 16, 32, 64, 4096]
        if args.model
        else sorted({1, 2, 4, 16, 2 ** t["max_level"]})
    )
    for factor in factors:
        if factor > 2 ** t["max_level"]:
            continue
        coords = {
            (0, 0),
            ((t["cols"] - 1) // (256 * factor), (t["rows"] - 1) // (256 * factor)),
        }
        if args.model and factor == 8:
            coords = {(x, y) for x in range(2) for y in range(2)}
        for tx, ty in sorted(coords):
            prefix = OUT / f"tensor-{t['id']}-f{factor}-{tx}-{ty}"
            result = run(
                "tile",
                "--tensor",
                t["id"],
                "--rules",
                "tensor_magnitude,tensor_linear",
                "--level",
                t["max_level"] - int(np.log2(factor)),
                "--x",
                tx,
                "--y",
                ty,
                "--out",
                prefix,
            )
            assert result["metrics"]["max_raw_band_values"] <= 1048576
            for rule in ["tensor_magnitude", "tensor_linear"]:
                want = reference(t, factor, tx, ty, maximum, rule == "tensor_magnitude")
                got = np.fromfile(str(prefix) + "-" + rule + ".f64le", "<f8").reshape(
                    want.shape
                )
                np.testing.assert_allclose(got, want, atol=2e-13, rtol=2e-13)
                if t["name"] == "cancellation" and factor >= 2:
                    assert got.item() == (1 if rule == "tensor_magnitude" else 0)
                records.append(
                    dict(
                        tensor=t["name"],
                        shape=t["shape"],
                        factor=factor,
                        tile=[tx, ty],
                        rule=rule,
                        maximum=maximum,
                        error=float(np.max(np.abs(got - want))),
                        min=float(got.min()),
                        max=float(got.max()),
                        mean=float(got.mean()),
                        std=float(got.std()),
                        row_means=got.mean(axis=1).tolist(),
                        column_means=got.mean(axis=0).tolist(),
                        metrics=result["metrics"],
                    )
                )
    # Inspector preserves bytes, address, native coordinate and signed exact source value.
    for row, col in [(0, 0), (t["rows"] - 1, t["cols"] - 1)]:
        got = run(
            "inspect",
            "--tensor",
            t["id"],
            "--row",
            row,
            "--col",
            col,
            "--left",
            "tensor_magnitude",
            "--right",
            "tensor_linear",
        )
        x = rows(t, row, row + 1, col, col + 1).item()
        assert float(got["raw_exact"]) == x
        with (MODEL / t["shard"]).open("rb") as source:
            source.seek(t["byte_offset"] + 2 * (row * t["cols"] + col))
            assert got["bf16_hex_le"] == source.read(2).hex()
        assert got["byte_offset"] == t["byte_offset"] + 2 * (row * t["cols"] + col)
        assert got["native_indices"] == ([col] if len(t["shape"]) == 1 else [row, col])
        assert got["transformed"]["left"] == (abs(x) / maximum if maximum else 0)
# Only this harness's generated tile cache; make repeat runs exercise a real miss.
for tile in (CACHE / "tiles").glob("*.png"):
    tile.unlink()
# Normal loopback API and repeated tile requests, same standalone Rust process/cache.
with socket.socket() as s:
    s.bind(("127.0.0.1", args.port))
    port = s.getsockname()[1]
with (OUT / "server.log").open("w") as log:
    server = subprocess.Popen(
        [
            str(BIN),
            "serve",
            "--model",
            str(MODEL),
            "--cache",
            str(CACHE),
            "--port",
            str(port),
        ],
        stdout=log,
        stderr=log,
    )
    try:

        def get(path):
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}" + path, timeout=60
            ) as r:
                return r.read(), dict(r.headers)

        for _ in range(100):
            assert server.poll() is None
            try:
                body, _ = get("/api/model")
                break
            except OSError:
                time.sleep(0.05)
        else:
            raise AssertionError("Server unavailable")
        assert "tensor_magnitude" in [r["id"] for r in json.loads(body)["rules"]]
        t = selected[0]
        for left, right in [
            ("tensor_magnitude", "tensor_linear"),
            ("tensor_linear", "tensor_magnitude"),
        ]:
            body, _ = get(f"/api/view?tensor={t['id']}&left={left}&right={right}")
            view = json.loads(body)
            l = view["legends"]["left" if left == "tensor_magnitude" else "right"]
            assert (
                l["min"] == 0
                and l["max"] == exact_max(t)
                and l["palette"] == "sequential-purple-v1"
            )
        factor = 16 if args.model else 2 ** t["max_level"]
        level = t["max_level"] - int(np.log2(factor))
        url = f"/tile?tensor={t['id']}&rule=tensor_magnitude&level={level}&x=0&y=0"
        first, h1 = get(url)
        second, h2 = get(url)
        assert h1["X-Atlas-Cache"] == "miss" and h2["X-Atlas-Cache"] == "hit"
        assert int(h2["X-Atlas-Source-Bytes"]) == 0
        assert (
            first
            == second
            == (
                OUT / f"tensor-{t['id']}-f{factor}-0-0-tensor_magnitude.png"
            ).read_bytes()
        )
    finally:
        server.terminate()
        server.wait(timeout=10)
report = dict(
    passed=True,
    scope="single real tensor" if args.model else "synthetic fixtures",
    source_identity=meta["source_identity"],
    binary_sha256=hashlib.sha256(BIN.read_bytes()).hexdigest(),
    cases=len(records),
    max_error=max(r["error"] for r in records),
    records=records,
)
(OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
