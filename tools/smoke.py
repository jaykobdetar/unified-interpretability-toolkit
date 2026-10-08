#!/usr/bin/env python3
"""Portable smoke checks using only tiny synthetic data and Python's standard library."""

import json, os, subprocess, tempfile, struct, zlib, math, decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(
    os.environ.get("ATLAS_BIN", ROOT / "target/release/weight-atlas-rust")
).resolve()
MODEL = ROOT / "fixtures/tiny-bf16"
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
assert BIN.is_file(), "Build the Rust binary first, or set ATLAS_BIN explicitly."
RULES = [
    "global_linear",
    "global_asinh",
    "tensor_linear",
    "tensor_asinh",
    "tensor_magnitude",
    "tensor_robust99",
    "tensor_signed_percentile",
]
expected = json.loads((MODEL / "expected.json").read_text())
with tempfile.TemporaryDirectory(prefix="weight-atlas-smoke-") as d:
    cache = Path(d) / "cache"

    def run(command: str, *args: object) -> Any:
        p = subprocess.run(
            [
                str(BIN),
                command,
                "--model",
                str(MODEL),
                "--cache",
                str(cache),
                *map(str, args),
            ],
            capture_output=True,
            text=True,
        )
        assert p.returncode == 0, p.stderr
        return json.loads(p.stdout)

    metadata = run("metadata")
    assert metadata["tensor_count"] == 3 and metadata["parameter_count"] == 26
    before = run("inspect", "--tensor", 1, "--row", 0, "--col", 4)
    assert (
        before["raw_exact"] == "34"
        and before["bf16_hex_le"] == "0842"
        and not before["transforms_ready"]
    )
    calibrated = run("calibrate")["model"]
    assert calibrated["calibration_complete"] and calibrated["global_max"] == 34
    count = 0
    fields = 0
    for t in metadata["catalog"]:
        ex = expected[t["name"]]
        for pos, value in enumerate(ex["values"]):
            row, col = divmod(pos, t["cols"])
            r = run(
                "inspect",
                "--tensor",
                t["id"],
                "--row",
                row,
                "--col",
                col,
                "--left",
                "tensor_linear",
                "--right",
                "tensor_asinh",
            )
            assert decimal.Decimal(r["raw_exact"]) == decimal.Decimal.from_float(value)
            assert r["bf16_hex_le"] == struct.pack("<f", value)[2:].hex()
            assert r["byte_offset"] == t["byte_offset"] + 2 * pos
            count += 1
        for factor in [1, 2 ** t["max_level"]]:
            level = t["max_level"] - int(math.log2(factor))
            prefix = Path(d) / f"{t['id']}-{factor}"
            for batch in [RULES[:4], RULES[4:]]:
                result = run(
                    "tile",
                    "--tensor",
                    t["id"],
                    "--rules",
                    ",".join(batch),
                    "--level",
                    level,
                    "--out",
                    prefix,
                )
                for rule in batch:
                    png = Path(str(prefix) + "-" + rule + ".png").read_bytes()
                    assert png[:8] == b"\x89PNG\r\n\x1a\n"
                    w, h = struct.unpack(">II", png[16:24])
                    assert (w, h) == (
                        result["metrics"]["width"],
                        result["metrics"]["height"],
                    )
                    fields += 1
    result = {
        "passed": True,
        "synthetic_values": count,
        "native_and_pooled_rule_pngs": fields,
        "calibration_global_max": 34,
        "source_weights_downloaded": False,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/smoke.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
