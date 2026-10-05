"""Pure fixture integrity/caps; no filesystem writes, Rust, browser or model."""

import hashlib
import json
import struct
from data_workflow_fixture import build

raw, manifest = build()
header_size = struct.unpack("<Q", raw[:8])[0]
header = json.loads(raw[8 : 8 + header_size])
payload = raw[8 + header_size :]
assert len(raw) < 2048 and len(header) == 6 and len(payload) == 224
assert (
    manifest["bytes"] == len(raw)
    and manifest["sha256"] == hashlib.sha256(raw).hexdigest()
)
count = 0
for name, tensor in manifest["tensors"].items():
    assert header[name] == {k: tensor[k] for k in ("dtype", "shape", "data_offsets")}
    width = 4 if tensor["dtype"] == "F32" else 2
    start, end = tensor["data_offsets"]
    words = tensor["source_hex_le"]
    assert end - start == width * len(words)
    assert payload[start:end] == b"".join(bytes.fromhex(w) for w in words)
    for word in words:
        bits = int.from_bytes(bytes.fromhex(word), "little")
        exponent = (
            bits >> (23 if width == 4 else 10 if tensor["dtype"] == "F16" else 7)
        ) & (255 if tensor["dtype"] != "F16" else 31)
        assert exponent != (31 if tensor["dtype"] == "F16" else 255)
    if name.endswith("rank3"):
        assert tensor["shape"] == [2, 3, 4] and words[:12] != words[12:]
    else:
        assert tensor["shape"] == [4] and int.from_bytes(
            bytes.fromhex(words[0]), "little"
        ) == (1 << (width * 8 - 1))
    count += len(words)
assert count == 84
print(
    json.dumps(
        {
            "status": "PASS",
            "values": count,
            "bytes": len(raw),
            "fixture_sha256": manifest["sha256"],
            "scope": "Pure deterministic finite BF16/F16/F32 bits, distinct native slices, signed zero/subnormal/max finite, no jobs",
        }
    )
)
