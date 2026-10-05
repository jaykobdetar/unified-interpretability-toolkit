"""Generate only a declared tiny BF16/F16/F32 browser fixture; stdlib, no models."""

import argparse
import hashlib
import json
from pathlib import Path
import struct

REVISION = "data-browser-fixture-v1"
PATTERNS = {
    "BF16": ["0000", "0080", "0100", "0180", "803f", "80bf", "7f7f", "7fff"],
    "F16": ["0000", "0080", "0100", "0180", "003c", "00bc", "ff7b", "fffb"],
    "F32": [
        "00000000",
        "00000080",
        "01000000",
        "01000080",
        "0000803f",
        "000080bf",
        "ffff7f7f",
        "ffff7fff",
    ],
}


def build():
    header, tensors, payload = {}, {}, bytearray()
    for dtype, pattern in PATTERNS.items():
        for kind, shape, words in [
            ("rank3", [2, 3, 4], (pattern * 3)),
            ("vector", [4], [pattern[i] for i in (1, 2, 6, 4)]),
        ]:
            name = dtype.lower() + "_" + kind
            start = len(payload)
            payload.extend(b"".join(bytes.fromhex(word) for word in words))
            header[name] = {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [start, len(payload)],
            }
            tensors[name] = {**header[name], "source_hex_le": words}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    raw = struct.pack("<Q", len(encoded)) + encoded + payload
    manifest = {
        "schema": "weight-atlas-data-browser-fixture-v1",
        "revision": REVISION,
        "file": "data.safetensors",
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "tensors": tensors,
        "scope": "84 synthetic finite values; no trained weights, private prompts or workers",
    }
    return raw, manifest


def write(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    raw, manifest = build()
    (destination / manifest["file"]).write_bytes(raw)
    (destination / "fixture-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "destination",
        type=Path,
        help="Fresh explicit task-owned temporary directory outside the checkout",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.destination.resolve().is_relative_to(repo):
        parser.error("Fixture/evidence must stay outside the checkout")
    print(json.dumps(write(args.destination), indent=2))
