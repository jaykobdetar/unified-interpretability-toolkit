#!/usr/bin/env python3
"""Generate the tiny synthetic BF16 fixture; no downloaded weights."""

import json, struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALUES = {
    "matrix": {
        "shape": [3, 5],
        "values": [
            -1.0,
            -0.5,
            0.0,
            0.5,
            1.0,
            0.25,
            -0.25,
            0.125,
            -0.125,
            2.0,
            -2.0,
            1.5,
            -1.5,
            0.75,
            -0.75,
        ],
    },
    "vector": {"shape": [7], "values": [0.0, -0.0, 1.0, -1.0, 34.0, 0.5, -0.5]},
    "zeros": {"shape": [4], "values": [0.0, 0.0, 0.0, 0.0]},
}


def generate(destination):
    destination.mkdir(parents=True, exist_ok=True)
    header = {}
    payload = bytearray()
    for name, tensor in VALUES.items():
        begin = len(payload)
        for value in tensor["values"]:
            raw = struct.pack("<f", value)
            assert raw[:2] == b"\0\0", "Fixture must be exactly BF16-representable"
            payload.extend(raw[2:])
        header[name] = {
            "dtype": "BF16",
            "shape": tensor["shape"],
            "data_offsets": [begin, len(payload)],
        }
    data = json.dumps(header, separators=(",", ":")).encode()
    data += b" " * ((-len(data)) % 8)
    (destination / "tiny.safetensors").write_bytes(
        struct.pack("<Q", len(data)) + data + payload
    )
    (destination / "expected.json").write_text(json.dumps(VALUES, indent=2))


if __name__ == "__main__":
    generate(ROOT / "fixtures/tiny-bf16")
