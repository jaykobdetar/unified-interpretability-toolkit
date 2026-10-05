#!/usr/bin/env python3
r"""Behaviour lock for the Weight Atlas native binary (version 2).

Records what the binary outputs for fixed synthetic checkpoints, so that a
refactor can be shown to leave behaviour unchanged. Standard library only
(Python 3.10 or newer); it builds its own tiny checkpoints, so it needs no
model, no browser and no GPU.

    # Before changing any code: keep a copy of the binary, and record from it.
    cp target/release/weight-atlas-rust ../atlas-before
    python3 behaviour_lock.py --binary ../atlas-before --record ../before.json

    # After each step, on the rebuilt binary.
    python3 behaviour_lock.py --binary target/release/weight-atlas-rust \
        --compare ../before.json --old-binary ../atlas-before

Exit status 0 means nothing strict differs and every upgrade check passed.
The binary's own memory and free-disk guards apply to the working files, which
go in the system temporary directory unless `--work DIR` says otherwise; if the
basic steps are refused the script says so and records nothing.
Record and compare on the same machine: the last digits of some transforms
depend on the platform's maths library. During a refactor this file is the
referee: change the code, not the lock, and do not record a new baseline.

Every observation is one entry with a category:

    numeric  hashes of the exact F64 fields behind each tile      (strict)
    image    hashes of PNG bytes                                  (strict)
    data     numbers, flags and identifiers in JSON outputs       (strict)
    status   exit codes and HTTP status codes                     (strict)
    header   selected HTTP response headers                       (strict)
    text     human-readable wording: messages, titles, help       (strict)
    asset    hashes of the served page files                      (reported)

A strict entry that is changed, missing or new fails the comparison. Wording is
strict because a check that has been removed often shows only as a different
refusal message from the check behind it. `--allow PREFIX` lists the
differences under one key prefix without failing on them, for a change that
was agreed beforehand; everything it excuses is printed.

Each run makes two passes in two different directories. Anything that differs
between the passes (paths, timings, file identities) is treated as volatile and
left out, so what remains is what the binary decides on its own.

Identities, bindings and cache keys depend on file paths, so they are volatile
and the comparison above cannot see them. `--old-binary` adds an upgrade check
for exactly that: the old binary prepares caches, then the new binary must
report the same identities and bindings, reuse the saved calibration and find
the cached tiles, for the viewer and for the comparison viewer.

Covered: every command except the two below, including requests the command
line must refuse and checkpoints it must refuse to open; the viewer server and
the comparison server: every route, tiles requested the way the page requests
them, calibration started through the server, answers before calibration is
ready, and the requests they must refuse. Not covered: the hosted-renderer and
profile-worker commands; non-default --resources settings; behaviour under
simultaneous requests; speed and memory use; the Python tools; and what the
page does in a browser.

Version 2 adds, over version 1: the refusal and boundary cases, broken
checkpoints, caches that are stale or damaged, the "not ready" answers,
calibration through the server, tiles with bindings, neighbouring tiles (cache
keys), slices over HTTP, the naming and hashing options, the comparison
server's rules and files, and five more upgrade checks; and it treats wording
as strict. A version 1 baseline cannot be compared; record again from the same
preserved starting binary.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import random
import re
import select
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
from http import HTTPStatus
from pathlib import Path
from urllib.parse import urlencode

LOCK_VERSION = 2
SHOWN_PER_GROUP = 12  # differences printed per group unless --full is given
STRICT = ("numeric", "image", "data", "status", "header", "text")
ADVISORY = ("asset",)
COMMAND_TIMEOUT = 120
STARTUP_SECONDS = 30
SETTLE_SECONDS = 30  # longest wait for calibration started through a server
TILE_SIZE = 256
QUIET_ROW_EVERY, QUIET_ROW_FIRST = 7, 3  # which synthetic rows are scaled down

KNOWN_RULES = [
    "global_linear",
    "global_asinh",
    "tensor_linear",
    "tensor_asinh",
    "tensor_magnitude",
    "tensor_magnitude_asinh",
    "tensor_robust99",
    "tensor_signed_percentile",
]
INSPECT_RULE_PAIRS = [("tensor_linear", "tensor_asinh"), ("global_linear", "tensor_robust99")]
GRID_RULES = ["tensor_linear", "tensor_asinh"]
COMPARE_QUANTITIES = ["a", "b", "delta", "abs_delta"]
COMPARE_MAPPINGS = ["linear", "asinh", "magnitude"]
COMPARE_VIEWS = [("a", "delta", "linear"), ("b", "abs_delta", "asinh"), ("a", "b", "magnitude")]

# (name, dtype, shape). Rank one and two only, so every version can open it.
CORE_TENSORS = [
    ("blocks.0.attn.q.weight", "BF16", (300, 517)),
    ("blocks.0.attn.k.weight", "F16", (260, 130)),
    ("blocks.0.mlp.up.weight", "F32", (129, 700)),
    ("blocks.0.norm.weight", "BF16", (513,)),
    ("blocks.0.mlp.bias", "F32", (900,)),
]
# Newer features: a rank-four tensor and a storage type that is listed but not decoded.
EXTRA_TENSORS = [
    ("conv.weight", "F32", (2, 3, 60, 70)),
    ("plain.weight", "BF16", (40, 24)),
    ("quant.codes", "I8", (16, 16)),
]
EXTRA_SLICES = {"conv.weight": ["0,0", "1,2"]}
SMALL_TENSORS = [("w.weight", "F32", (4, 6)), ("b.bias", "BF16", (16,))]

# JSON keys whose values depend on the machine or the moment, never on behaviour.
VOLATILE_KEY = re.compile(r"(^|_)(seconds|ms|mib|gib|rss|cpu|elapsed|wall|pid|port)($|_)")
VOLATILE_NAMES = {
    "source_directory",
    "source_identity",
    "model_identity",
    "comparison_identity",
    "root",
    "listening",
    "dev",
    "inode",
    "mtime",
    "mtime_ns",
    "ctime",
    "ctime_ns",
}
IDENTITY_NAMES = ("source_identity", "model_identity")
SERVED_FILES = [
    "/",
    "/index.html",
    "/viewer.js",
    "/app.js",
    "/atlas-tools.js",
    "/workspace-tools.js",
    "/inference.js",
    "/inference-import.js",
    "/style.css",
    "/vendor/openseadragon.min.js",
    "/vendor/OpenSeadragon-LICENSE.txt",
]
COMPARISON_FILES = [
    "/",
    "/comparison.html",
    "/comparison.js",
    "/comparison.css",
    "/vendor/openseadragon.min.js",
]
# Timing headers are deliberately absent: two passes could agree on one by chance.
RECORDED_HEADERS = [
    "content-type",
    "cache-control",
    "connection",
    "x-content-type-options",
    "content-security-policy",
    "x-atlas-factor",
    "x-atlas-cache",
    "x-atlas-source-bytes",
    "x-atlas-coordinate-space",
    "x-atlas-inference-editable",
]
# Steps that must succeed for a recording to mean anything.
ESSENTIAL = {
    "core/metadata/exit": 0,
    "core/calibrate/exit": 0,
    "http/start": "listening",
    "pairhttp/start": "listening",
    "fresh/calibrate-all/settled": True,
}
EVIL = "http://evil.example"
LOCAL = {"X-Atlas-Local": "1"}
# label -> (method, path, extra headers): requests the viewer must refuse, and a few it accepts.
REQUEST_RULES = {
    "cross-origin": ("GET", "/api/model", {"Origin": EVIL}),
    "cross-site": ("GET", "/api/model", {"Sec-Fetch-Site": "cross-site"}),
    "traversal": ("GET", "/../Cargo.toml", {}),
    "unknown-route": ("GET", "/no-such-route", {}),
    "unknown-tensor": ("GET", "/api/view?tensor=9999", {}),
    "bad-level": ("GET", "/tile?tensor=0&rule=tensor_linear&level=99&x=0&y=0", {}),
    "bad-rule": ("GET", "/tile?tensor=0&rule=no_such_rule&level=0&x=0&y=0", {}),
    "outside": ("GET", "/api/inspect?tensor=0&row=999999&col=0", {}),
    "duplicate-parameter": ("GET", "/api/view?tensor=0&tensor=1", {}),
    "bad-escape": ("GET", "/api/view?tensor=%zz", {}),
    "wrong-method": ("PUT", "/api/model", {}),
    "post-without-local-header": ("POST", "/api/calibrate?tensor=0", {}),
    "post-with-local-header": ("POST", "/api/calibrate?tensor=0", LOCAL),
    "post-cross-origin": ("POST", "/api/calibrate?tensor=0", {**LOCAL, "Origin": EVIL}),
}
MORE_REQUEST_RULES = {
    "same-origin": ("GET", "/api/model", {"Sec-Fetch-Site": "same-origin"}),
    "head": ("HEAD", "/api/model", {}),
    "options": ("OPTIONS", "/api/model", {}),
    "delete": ("DELETE", "/api/model", {}),
    "get-on-post-route": ("GET", "/api/calibrate?tensor=0", {}),
    "post-on-get-route": ("POST", "/api/model", LOCAL),
    "post-unknown-tensor": ("POST", "/api/calibrate?tensor=9999", LOCAL),
    "post-no-selection": ("POST", "/api/calibrate", LOCAL),
    "post-everything": ("POST", "/api/calibrate?all=1", LOCAL),
    "declares-a-body": ("GET", "/api/model", {"Content-Length": "5"}),
    "declares-chunked": ("GET", "/api/model", {"Transfer-Encoding": "chunked"}),
    "traversal-encoded": ("GET", "/%2e%2e/Cargo.toml", {}),
    "file-outside-list": ("GET", "/vendor/../index.html", {}),
}
# label -> path: how the viewer treats missing, malformed and default parameters.
PARAMETER_PROBES = {
    "tensor-not-a-number": "/api/view?tensor=abc",
    "tensor-negative": "/api/view?tensor=-1",
    "tensor-empty": "/api/view?tensor=",
    "tensor-missing": "/api/view",
    "view-defaults": "/api/view?tensor=0",
    "view-unknown-rule": "/api/view?tensor=0&left=nope&right=tensor_linear",
    "view-empty-rule": "/api/view?tensor=0&left=&right=tensor_linear",
    "view-encoded-name": "/api/view?tensor=0&left=tensor%5Flinear&right=tensor_asinh",
    "slice-on-a-matrix": "/api/view?tensor=0&slice=0",
    "status-without-tensor": "/api/tensor-status",
    "status-unknown-tensor": "/api/tensor-status?tensor=9999",
    "progress-with-tensor": "/api/progress?tensor=1",
    "progress-unknown-tensor": "/api/progress?tensor=9999",
    "inspect-defaults": "/api/inspect?tensor=0",
    "inspect-with-rules": "/api/inspect?tensor=0&row=0&col=5&left=tensor_linear&right=tensor_asinh",
    "inspect-fractional-row": "/api/inspect?tensor=0&row=1.5&col=0",
    "inspect-negative-row": "/api/inspect?tensor=0&row=-1&col=0",
    "tile-defaults": "/tile?tensor=0",
    "tile-level-not-a-number": "/tile?tensor=0&rule=tensor_linear&level=x&x=0&y=0",
    "tile-negative-x": "/tile?tensor=0&rule=tensor_linear&level=0&x=-1&y=0",
    "tile-empty-rule": "/tile?tensor=0&rule=&level=0&x=0&y=0",
}
# What a server with an empty cache is asked before, during and after calibration.
FRESH_VIEW = "/api/view?tensor=0&left=tensor_linear&right=tensor_asinh"
FRESH_PROBES = {
    "model": "/api/model",
    "progress": "/api/progress",
    "status": "/api/tensor-status?tensor=0",
    "status-other": "/api/tensor-status?tensor=1",
    "view": FRESH_VIEW,
    "view-global": "/api/view?tensor=0&left=global_linear&right=global_asinh",
    "view-other": "/api/view?tensor=1&left=tensor_linear&right=tensor_asinh",
    "inspect": "/api/inspect?tensor=0&row=0&col=5",
}
# The same idea as the request rules above, for the comparison server.
COMPARISON_RULES = {
    "cross-origin": ("GET", "/api/comparison/model", {"Origin": EVIL}),
    "cross-site": ("GET", "/api/comparison/model", {"Sec-Fetch-Site": "cross-site"}),
    "unknown-route": ("GET", "/api/comparison/nothing", {}),
    "single-source-view": ("GET", "/api/view?tensor=0", {}),
    "single-source-tile": ("GET", "/tile?tensor=0&rule=tensor_linear&level=0&x=0&y=0", {}),
    "single-source-file": ("GET", "/app.js", {}),
    "wrong-method": ("PUT", "/api/comparison/model", {}),
    "head": ("HEAD", "/api/comparison/model", {}),
    "duplicate-parameter": ("GET", "/api/comparison/view?tensor=0&tensor=1", {}),
    "bad-escape": ("GET", "/api/comparison/view?tensor=%zz", {}),
    "unknown-pair": ("GET", "/api/comparison/view?tensor=9999", {}),
    "view-defaults": ("GET", "/api/comparison/view?tensor=0", {}),
    "view-unknown-quantity": ("GET", "/api/comparison/view?tensor=0&left=nope", {}),
    "view-unknown-mapping": ("GET", "/api/comparison/view?tensor=0&mapping=nope", {}),
    "inspect-defaults": ("GET", "/api/comparison/inspect?tensor=0", {}),
    "inspect-outside": ("GET", "/api/comparison/inspect?tensor=0&row=999999&col=0", {}),
    "tile-defaults": ("GET", "/api/comparison/tile?tensor=0", {}),
    "tile-bad-level": (
        "GET",
        "/api/comparison/tile?tensor=0&quantity=delta&mapping=linear&level=99&x=0&y=0",
        {},
    ),
    "identity-changed": ("GET", "/api/comparison/view?tensor=0&comparison_identity=0000", {}),
    "post-without-local-header": ("POST", "/api/comparison/calibrate?tensor=0", {}),
    "post-with-local-header": ("POST", "/api/comparison/calibrate?tensor=0", LOCAL),
    "post-cross-origin": (
        "POST",
        "/api/comparison/calibrate?tensor=0",
        {**LOCAL, "Origin": EVIL},
    ),
    "post-everything": ("POST", "/api/comparison/calibrate?all=1", LOCAL),
    "post-unknown-pair": ("POST", "/api/comparison/calibrate?tensor=9999", LOCAL),
    "post-no-selection": ("POST", "/api/comparison/calibrate", LOCAL),
}


# --------------------------------------------------------------------------
# Synthetic checkpoints
# --------------------------------------------------------------------------


def synthetic_values(name: str, shape: tuple[int, ...], variant: str) -> list[float]:
    """Deterministic values with structure worth rendering.

    Uses only `random.random()` and exact arithmetic, so the same numbers come
    out on every platform. Includes quiet rows, exact zeros, a negative zero, an
    outlier and a subnormal.
    """
    rng = random.Random(f"weight-atlas-lock:{name}")
    count = 1
    for dim in shape:
        count *= dim
    cols = shape[-1]
    values = []
    for index in range(count):
        value = (rng.random() + rng.random() + rng.random() + rng.random() - 2.0) * 0.05
        if (index // cols) % QUIET_ROW_EVERY == QUIET_ROW_FIRST:
            value *= 0.25
        if index % 53 == 0:
            value = 0.0
        values.append(value)
    for position, planted in ((5, 3.25), (7, -0.0), (11, 1e-41)):
        if position < count:
            values[position] = planted
    if variant == "b":
        drift = random.Random(f"weight-atlas-lock:{name}:b")
        for index in range(0, count, 3):
            values[index] += (drift.random() - 0.5) * 0.02
    return values


def encode(dtype: str, values: list[float]) -> bytes:
    """Little-endian payload bytes for one tensor."""
    if dtype == "F32":
        return struct.pack(f"<{len(values)}f", *values)
    if dtype == "F16":
        return struct.pack(f"<{len(values)}e", *values)
    if dtype == "BF16":
        raw = struct.pack(f"<{len(values)}f", *values)
        return bytes(byte for pair in zip(raw[2::4], raw[3::4], strict=True) for byte in pair)
    if dtype == "I8":
        codes = [max(-127, min(127, round(value * 400))) for value in values]
        return struct.pack(f"<{len(codes)}b", *codes)
    raise ValueError(f"No encoder for {dtype}")


def shard_bytes(header: bytes, payload: bytes) -> bytes:
    """A safetensors file from a ready-made header: length, padded header, payload."""
    padded = header + b" " * (-len(header) % 8)
    return struct.pack("<Q", len(padded)) + padded + payload


def header_and_payload(tensors: list[tuple[str, str, tuple[int, ...], bytes]]):
    """The header dictionary and joined payload for contiguous tensors."""
    header, payloads, offset = {}, [], 0
    for name, dtype, shape, payload in tensors:
        end = offset + len(payload)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, end]}
        offset = end
        payloads.append(payload)
    return header, b"".join(payloads)


def write_shard(path: Path, tensors: list[tuple[str, str, tuple[int, ...], bytes]]) -> None:
    """Write one safetensors file with contiguous payloads."""
    header, payload = header_and_payload(tensors)
    raw = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(shard_bytes(raw, payload))


def encoded(tensors, variant: str = "a") -> list[tuple[str, str, tuple[int, ...], bytes]]:
    return [
        (name, dtype, shape, encode(dtype, synthetic_values(name, shape, variant)))
        for name, dtype, shape in tensors
    ]


def build_checkpoint(directory: Path, tensors, variant: str, shards: int = 1) -> None:
    """Write a checkpoint directory, optionally split into shards with an index."""
    directory.mkdir(parents=True)
    ready = encoded(tensors, variant)
    if shards == 1:
        write_shard(directory / "model.safetensors", ready)
        return
    weight_map = {}
    for number in range(shards):
        filename = f"model-{number + 1:05d}-of-{shards:05d}.safetensors"
        part = ready[number::shards]
        write_shard(directory / filename, part)
        weight_map.update({name: filename for name, _, _, _ in part})
    index = {"metadata": {}, "weight_map": weight_map}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index))


def broken_headers() -> dict[str, tuple[bytes, bytes]]:
    """label -> (header bytes, payload): single files the reader must judge."""
    good = encoded(SMALL_TENSORS)
    header, payload = header_and_payload(good)

    def changed(**replacement) -> bytes:
        return json.dumps({**header, **replacement}, separators=(",", ":")).encode()

    first, second = header["w.weight"], header["b.bias"]
    end_of_first = first["data_offsets"][1]
    length_of_second = second["data_offsets"][1] - second["data_offsets"][0]
    overlapping = [end_of_first - length_of_second, end_of_first]
    plain = changed()
    shared = json.dumps({"w.weight": first, "v.weight": first}, separators=(",", ":")).encode()
    return {
        "ok-plain": (plain, payload),
        "ok-with-metadata-entry": (changed(__metadata__={"format": "pt"}), payload),
        "no-tensors": (b"{}", b""),
        "only-a-metadata-entry": (b'{"__metadata__":{"format":"pt"}}', b""),
        "header-is-not-json": (b"this is not json", payload),
        "header-is-a-list": (b"[1,2,3]", payload),
        "repeated-name-in-header": (plain[:-1] + b"," + plain[1:], payload),
        "data-past-the-end": (plain, payload[:-8]),
        "bytes-after-the-data": (plain, payload + b"\0" * 16),
        "size-does-not-match-shape": (changed(**{"w.weight": {**first, "shape": [5, 6]}}), payload),
        "unknown-number-format": (changed(**{"w.weight": {**first, "dtype": "F64"}}), payload),
        "empty-dimension": (changed(**{"w.weight": {**first, "shape": [0, 6]}}), payload),
        "negative-dimension": (changed(**{"w.weight": {**first, "shape": [-4, 6]}}), payload),
        "offsets-reversed": (
            changed(**{"w.weight": {**first, "data_offsets": [end_of_first, 0]}}),
            payload,
        ),
        "overlapping-data": (
            changed(**{"b.bias": {**second, "data_offsets": overlapping}}),
            payload,
        ),
        "entry-without-shape": (changed(**{"w.weight": {"dtype": "F32"}}), payload),
        "two-tensors-share-bytes": (shared, payload[:end_of_first]),
        "gap-between-tensors": (
            changed(
                **{"b.bias": {**second, "data_offsets": [x + 4 for x in second["data_offsets"]]}}
            ),
            payload[:end_of_first] + b"\0" * 4 + payload[end_of_first:],
        ),
    }


def broken_checkpoints(root: Path) -> dict[str, Path]:
    """Checkpoint directories the reader must refuse, with two it must accept."""
    cases = {}

    def new(label: str) -> Path:
        cases[label] = root / label
        cases[label].mkdir(parents=True)
        return cases[label]

    for label, (header, payload) in broken_headers().items():
        (new(label) / "model.safetensors").write_bytes(shard_bytes(header, payload))
    good = encoded(SMALL_TENSORS)
    new("empty-directory")
    (new("header-longer-than-file") / "model.safetensors").write_bytes(
        struct.pack("<Q", 1 << 20) + b"{}"
    )
    (new("file-shorter-than-its-length-field") / "model.safetensors").write_bytes(b"\x10\0\0")
    write_shard(root / "elsewhere.safetensors", good)
    (new("linked-file") / "model.safetensors").symlink_to(root / "elsewhere.safetensors")
    twice, many = new("same-name-in-two-files"), new("sixty-five-files")
    for number in range(2):
        write_shard(twice / f"part{number}.safetensors", good)
    for number in range(65):
        one = encoded([(f"t{number:02d}.weight", "BF16", (2, 2))])
        write_shard(many / f"s{number:02d}.safetensors", one)
    indexed = {
        "index-names-a-missing-file": {"weight_map": {"w.weight": "absent.safetensors"}},
        "index-omits-a-tensor": {"weight_map": {"w.weight": "model.safetensors"}},
        "index-names-an-absent-tensor": {
            "weight_map": {name: "model.safetensors" for name in ("w.weight", "b.bias", "ghost")}
        },
        "index-names-a-file-in-a-folder": {"weight_map": {"w.weight": "sub/model.safetensors"}},
        "index-names-a-file-above": {"weight_map": {"w.weight": "../elsewhere.safetensors"}},
        "index-without-a-weight-map": {"metadata": {}},
    }
    for label, index in indexed.items():
        write_shard(new(label) / "model.safetensors", good)
        (cases[label] / "model.safetensors.index.json").write_text(json.dumps(index))
    write_shard(new("index-is-not-json") / "model.safetensors", good)
    (cases["index-is-not-json"] / "model.safetensors.index.json").write_text("not json")
    write_shard(new("linked-index") / "model.safetensors", good)
    full = {"weight_map": {name: "model.safetensors" for name, _, _, _ in good}}
    (root / "elsewhere.index.json").write_text(json.dumps(full))
    (cases["linked-index"] / "model.safetensors.index.json").symlink_to(
        root / "elsewhere.index.json"
    )
    return cases


# --------------------------------------------------------------------------
# Talking to the binary
# --------------------------------------------------------------------------


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def flags(**options) -> list[str]:
    """Turn keyword options into command-line flags: level=3 becomes --level 3."""
    arguments = []
    for name, value in options.items():
        arguments += ["--" + name.replace("_", "-"), str(value)]
    return arguments


def run_command(binary: str, *arguments: str) -> subprocess.CompletedProcess | None:
    """Run one command to completion; None means it timed out."""
    command = [binary, *arguments]
    try:
        return subprocess.run(
            command, capture_output=True, text=True, timeout=COMMAND_TIMEOUT, check=False
        )
    except subprocess.TimeoutExpired:
        return None


def json_output(done: subprocess.CompletedProcess | None):
    """Parsed JSON from a successful command, otherwise None."""
    if done is None or done.returncode != 0:
        return None
    try:
        return json.loads(done.stdout)
    except ValueError:
        return None


def announced(process: subprocess.Popen) -> bool:
    """Wait for the line in which a server says where it is listening."""
    output = process.stdout
    assert output is not None  # requested with stdout=PIPE by the caller
    deadline = time.monotonic() + STARTUP_SECONDS
    while time.monotonic() < deadline:
        ready, _, _ = select.select([output], [], [], max(0.0, deadline - time.monotonic()))
        line = output.readline() if ready else b""
        if not line:
            return False
        if b'"listening"' in line:
            return True
    return False


def start_server(binary: str, *arguments: str):
    """Start a server on a free port; return (process, port) once it is listening."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    command = [binary, *arguments, *flags(port=port)]
    # Unbuffered, so that waiting on the pipe never misses a line already read.
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0
    )
    if not announced(process):
        stop_server(process)
        return None, port
    return process, port


def stop_server(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def http_request(port: int, path: str, method: str = "GET", headers: dict | None = None):
    """One request on a fresh connection: (status, headers, body) or an error name."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        connection.request(method, path, headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    except (OSError, http.client.HTTPException) as error:
        return type(error).__name__, {}, b""
    finally:
        connection.close()


def http_json(port: int, path: str):
    """(status, parsed JSON or None) for one request."""
    status, _, body = http_request(port, path)
    try:
        return status, json.loads(body)
    except ValueError:
        return status, None


def tile_plan(tensor: dict) -> list[tuple[int, int, int]]:
    """(level, x, y) for the first and last tile at three zoom levels."""
    top = tensor["max_level"]
    plan = set()
    for level in {top, max(0, top - 2), 0}:
        span = TILE_SIZE * (1 << (top - level))
        plan.add((level, 0, 0))
        plan.add((level, (tensor["cols"] - 1) // span, (tensor["rows"] - 1) // span))
    return sorted(plan)


def tiles_across(tensor: dict, level: int) -> tuple[int, int]:
    """How many tiles wide and high the tensor is at one zoom level."""
    span = TILE_SIZE * (1 << (tensor["max_level"] - level))
    return -(-tensor["cols"] // span), -(-tensor["rows"] // span)


def tile_grid(tensor: dict) -> list[tuple[int, int, int]]:
    """(level, x, y) for up to four neighbouring tiles at the two finest levels."""
    top = tensor["max_level"]
    grid = []
    for level in sorted({top, max(0, top - 1)}):
        wide, high = tiles_across(tensor, level)
        grid += [(level, x, y) for y in range(min(high, 2)) for x in range(min(wide, 2))]
    return grid


def tile_edges(tensor: dict) -> dict[str, dict]:
    """Tile addresses just past each edge of the tensor: all must be refused."""
    top = tensor["max_level"]
    wide, high = tiles_across(tensor, top)
    return {
        "right-edge": {"level": top, "x": wide, "y": 0},
        "bottom-edge": {"level": top, "x": 0, "y": high},
        "above-top-level": {"level": top + 1, "x": 0, "y": 0},
    }


def inspect_points(tensor: dict) -> list[tuple[int, int]]:
    """A few addresses: the origin, the planted special values and the last cell."""
    rows, cols = tensor["rows"], tensor["cols"]
    return sorted({(0, 0), (0, min(5, cols - 1)), (0, min(7, cols - 1)), (rows - 1, cols - 1)})


def calibration_complete(_status, body) -> bool:
    """Whether a progress answer says the whole checkpoint is calibrated."""
    return bool(body and body.get("calibration_complete"))


def another_binding(binding: str) -> str:
    """A well-formed binding that differs from the given one in its last digit."""
    return binding[:-1] + ("0" if binding[-1] != "0" else "1")


def command_line_probes(work: Path, core: Path, other: Path, extras: Path) -> dict[str, list[str]]:
    """label -> arguments: how the command line treats missing, malformed and odd requests."""
    source = flags(model=core, cache=work / "cache-core")
    sliced = flags(model=extras, cache=work / "cache-extras")
    pair = flags(model=core, compare_model=other, cache=work / "cache-pair")
    out = flags(out=work / "probe")
    tile = [*flags(rules="tensor_linear"), *out]
    return {
        "no-command": [],
        "unknown-command": ["frobnicate", *source],
        "no-model": ["metadata"],
        "option-without-value": ["metadata", "--model"],
        "value-without-option": ["metadata", "model", str(core)],
        "repeated-option": ["metadata", *flags(model=core), *flags(model=core)],
        "model-is-a-file": ["metadata", *flags(model=core / "model.safetensors")],
        "model-is-missing": ["metadata", *flags(model=core / "absent")],
        "tensor-not-a-number": ["inspect", *source, *flags(tensor="abc")],
        "tensor-negative": ["inspect", *source, *flags(tensor=-1)],
        "tensor-unknown": ["inspect", *source, *flags(tensor=9999)],
        "inspect-defaults": ["inspect", *source],
        "row-not-a-number": ["inspect", *source, *flags(tensor=0, row="x")],
        "column-outside": ["inspect", *source, *flags(tensor=0, row=0, col=999999)],
        "inspect-unknown-rule": ["inspect", *source, *flags(tensor=0, left="nope")],
        "tile-defaults": ["tile", *source, *out],
        "tile-two-rules": ["tile", *source, *out, *flags(rules="tensor_linear,tensor_asinh")],
        "tile-repeated-rule": ["tile", *source, *out, *flags(rules="tensor_linear,tensor_linear")],
        "tile-empty-rule-list": ["tile", *source, *out, *flags(rules="")],
        "tile-level-not-a-number": ["tile", *source, *tile, *flags(level="x")],
        "tile-negative-x": ["tile", *source, *tile, *flags(x=-1)],
        "slice-on-a-matrix": ["tile", *source, *tile, *flags(slice="0")],
        "slice-outside": ["tile", *sliced, *tile, *flags(tensor=0, slice="9,9")],
        "slice-too-short": ["tile", *sliced, *tile, *flags(tensor=0, slice="1")],
        "slice-not-numbers": ["tile", *sliced, *tile, *flags(tensor=0, slice="a,b")],
        "overview-without-tensor": ["overview", *source],
        "overview-unknown-rule": ["overview", *source, *flags(tensor=0, rules="nope")],
        "calibrate-unknown-tensor": ["calibrate", *source, *flags(tensor=9999)],
        "bench-unknown-tensor": ["bench", *source, *flags(tensor=9999, repeats=1)],
        "bench-no-repeats": ["bench", *source, *flags(tensor=0, repeats=0)],
        "compare-without-second-model": ["compare-metadata", *source],
        "compare-with-itself": [
            "compare-metadata",
            *flags(model=core, compare_model=core, cache=work / "cache-self"),
        ],
        "compare-calibrate-without-tensor": ["compare-calibrate", *pair],
        "compare-tile-without-out": ["compare-tile", *pair, *flags(tensor=0)],
        "compare-tile-unknown-quantity": ["compare-tile", *pair, *out, *flags(quantity="nope")],
        "compare-tile-unknown-mapping": ["compare-tile", *pair, *out, *flags(mapping="nope")],
        "compare-unknown-pair": ["compare-inspect", *pair, *flags(tensor=9999)],
        "compare-inspect-outside": ["compare-inspect", *pair, *flags(tensor=0, row=999999)],
        "resources-not-json": ["metadata", *flags(model=core, resources="not json")],
        "resources-empty-object": ["calibrate", *source, *flags(resources="{}")],
    }


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------


class Recorder:
    """Runs the binary against the synthetic checkpoints and collects entries."""

    def __init__(self, binary: Path, work: Path):
        self.binary = str(binary)
        self.work = work
        self.port = 0  # the server currently being probed
        self.entries: dict[str, list] = {}

    # ---- entry helpers ----

    def put(self, key: str, category: str, value) -> None:
        self.entries[key] = [category, value]

    def put_json(self, key: str, value) -> None:
        """Flatten a JSON value into one entry per leaf, skipping volatile keys."""
        if isinstance(value, dict):
            for name in sorted(value):
                if not (VOLATILE_KEY.search(name) or name in VOLATILE_NAMES):
                    self.put_json(f"{key}/{name}", value[name])
        elif isinstance(value, list):
            self.put(f"{key}#len", "data", len(value))
            for position, item in enumerate(value):
                self.put_json(f"{key}[{position}]", item)
        else:
            wording = isinstance(value, str) and " " in value
            self.put(key, "text" if wording else "data", value)

    # ---- command line ----

    def run(self, key: str, *arguments: str):
        """Run one command; record its exit code and JSON output or error text."""
        done = run_command(self.binary, *arguments)
        if done is None:
            self.put(f"{key}/exit", "status", "timeout")
            return None
        self.put(f"{key}/exit", "status", done.returncode)
        if done.returncode != 0:
            lines = done.stderr.strip().splitlines()
            self.put(f"{key}/stderr", "text", lines[-1] if lines else "")
            return None
        parsed = json_output(done)
        if parsed is None:
            self.put(f"{key}/stdout", "text", done.stdout.strip())
            return None
        self.put_json(f"{key}/out", parsed)
        return parsed

    def record_written_files(self, key: str, prefix: str) -> None:
        """Hash the field and image a tile command wrote, then remove them."""
        for extension, category in ((".f64le", "numeric"), (".png", "image")):
            path = Path(prefix + extension)
            found = path.exists()
            self.put(key + extension, category, sha256(path.read_bytes()) if found else "absent")
            if found:
                path.unlink()

    def record_single_source(self, label: str, model: Path, slices: dict[str, list[str]]) -> dict:
        """Every single-checkpoint command, for each tensor and chosen slice."""
        source = flags(model=model, cache=self.work / f"cache-{label}")
        metadata = self.run(f"{label}/metadata", "metadata", *flags(model=model)) or {}
        self.run(f"{label}/calibrate", "calibrate", *source)
        catalog = {tensor["name"]: tensor for tensor in metadata.get("catalog", [])}
        for name, tensor in catalog.items():
            for chosen in slices.get(name, [""]):
                where = (
                    source + flags(tensor=tensor["id"]) + (flags(slice=chosen) if chosen else [])
                )
                tag = f"{label}/{name}" + (f"@{chosen}" if chosen else "")
                self.record_inspections(tag, where, tensor)
                self.record_tiles(tag, where, tensor)
                self.run(f"{tag}/overview", "overview", *where)
        refused = flags(tensor=0, max_values=1000)
        self.run(f"{label}/overview[refused]", "overview", *source, *refused)
        self.run(f"{label}/bench", "bench", *source, *flags(tensor=0, repeats=1))
        self.run(f"{label}/verify", "verify", *source)
        return catalog

    def record_inspections(self, tag: str, where: list[str], tensor: dict) -> None:
        for row, col in inspect_points(tensor):
            for left, right in INSPECT_RULE_PAIRS:
                key = f"{tag}/inspect[{row},{col}]/{left}+{right}"
                self.run(key, "inspect", *where, *flags(row=row, col=col, left=left, right=right))
        outside = flags(row=tensor["rows"], col=0)
        self.run(f"{tag}/inspect[outside]", "inspect", *where, *outside)
        beside = flags(row=0, col=tensor["cols"])
        self.run(f"{tag}/inspect[outside-column]", "inspect", *where, *beside)

    def record_tiles(self, tag: str, where: list[str], tensor: dict) -> None:
        prefix = str(self.work / "tile")
        for level, x, y in tile_plan(tensor):
            for rule in KNOWN_RULES:
                key = f"{tag}/tile[L{level},{x},{y}]/{rule}"
                self.run(key, "tile", *where, *flags(rules=rule, level=level, x=x, y=y, out=prefix))
                self.record_written_files(key, f"{prefix}-{rule}")
        top = tensor["max_level"]
        refusals = {
            "bad-level": flags(rules="tensor_linear", level=99),
            "bad-rule": flags(rules="no_such_rule", level=top),
            "outside": flags(rules="tensor_linear", level=top, x=9999),
        }
        for label, edge in tile_edges(tensor).items():
            refusals[label] = flags(rules="tensor_linear", **edge)
        for label, options in refusals.items():
            self.run(f"{tag}/tile[{label}]", "tile", *where, *options, *flags(out=prefix))
            self.record_written_files(f"{tag}/tile[{label}]/written", f"{prefix}-tensor_linear")

    def record_comparison(self, model_a: Path, model_b: Path) -> dict:
        """Metadata, calibration, inspection and tiles for an ordered pair."""
        source = flags(model=model_a, compare_model=model_b, cache=self.work / "cache-pair")
        metadata = self.run("pair/metadata", "compare-metadata", *source) or {}
        pairs = {pair["name"]: pair for pair in metadata.get("catalog", [])}
        for name, pair in pairs.items():
            where = source + flags(tensor=pair["id"])
            self.run(f"pair/{name}/calibrate", "compare-calibrate", *where)
            for row, col in ((0, 0), (0, 5), (pair["rows"] - 1, pair["cols"] - 1)):
                key = f"pair/{name}/inspect[{row},{col}]"
                self.run(key, "compare-inspect", *where, *flags(row=row, col=col))
            self.record_comparison_tiles(name, where, pair["max_level"])
        return pairs

    def record_comparison_tiles(self, name: str, where: list[str], top: int) -> None:
        prefix = str(self.work / "pairtile")
        for level in sorted({top, max(0, top - 3)}):
            for quantity in COMPARE_QUANTITIES:
                for mapping in COMPARE_MAPPINGS:
                    key = f"pair/{name}/tile[L{level}]/{quantity}+{mapping}"
                    options = flags(quantity=quantity, mapping=mapping, level=level, x=0, y=0)
                    self.run(key, "compare-tile", *where, *options, *flags(out=prefix))
                    self.record_written_files(key, prefix)

    def record_refusals(self, core: Path, extras: Path, extra_catalog: dict) -> None:
        """Command-line requests that must be refused."""
        conv = extra_catalog.get("conv.weight", {}).get("id", 0)
        no_slice = flags(model=extras, cache=self.work / "cache-extras", tensor=conv)
        options = flags(rules="tensor_linear", out=self.work / "tile")
        self.run("extras/conv.weight/tile[no-slice]", "tile", *no_slice, *options)
        mismatch = flags(model=core, compare_model=extras, cache=self.work / "cache-mismatch")
        self.run("pair/mismatched", "compare-metadata", *mismatch)
        self.run("cache-inside-model", "calibrate", *flags(model=core, cache=core / "cache"))

    def record_command_line_probes(self, core: Path, other: Path, extras: Path) -> None:
        """Missing, malformed and unusual command lines, each with its exact outcome."""
        kinds = {".f64le": "numeric", ".png": "image"}
        for label, arguments in command_line_probes(self.work, core, other, extras).items():
            self.run(f"probe/{label}", *arguments)
            for written in sorted(self.work.glob("probe*")):
                kind = kinds.get(written.suffix, "data")
                suffix = written.name.removeprefix("probe")
                self.put(f"probe/{label}/wrote{suffix}", kind, sha256(written.read_bytes()))
                written.unlink()

    def record_broken_checkpoints(self) -> None:
        """Checkpoints the reader must refuse to open, and two plain ones it must accept."""
        for label, directory in broken_checkpoints(self.work / "broken").items():
            self.run(f"broken/{label}", "metadata", *flags(model=directory))

    def record_stepwise_calibration(self, model: Path) -> None:
        """One tensor first, then the rest, then nothing left to do."""
        source = flags(model=model, cache=self.work / "cache-steps")
        self.run("steps/one-tensor", "calibrate", *source, *flags(tensor=1))
        self.run("steps/the-rest", "calibrate", *source)
        self.run("steps/nothing-left", "calibrate", *source)
        self.run("steps/one-again", "calibrate", *source, *flags(tensor=1))

    def record_stale_caches(self, core: Path, other: Path) -> None:
        """A cache made for one checkpoint or pair, then offered to a different one."""
        cache = self.work / "cache-stale"
        first, second = flags(model=core, cache=cache), flags(model=other, cache=cache)
        self.run("stale/first-checkpoint", "calibrate", *first)
        self.run("stale/second-checkpoint", "calibrate", *second)
        self.run("stale/second-checkpoint/inspect", "inspect", *second, *flags(row=0, col=6))
        prefix = str(self.work / "stale")
        tile = flags(rules="tensor_linear", out=prefix)
        self.run("stale/second-checkpoint/tile", "tile", *second, *tile)
        self.record_written_files("stale/second-checkpoint/tile", f"{prefix}-tensor_linear")
        self.run("stale/first-checkpoint-again", "calibrate", *first)
        pair_cache = self.work / "cache-stale-pair"
        forward = flags(model=core, compare_model=other, cache=pair_cache, tensor=0)
        backward = flags(model=other, compare_model=core, cache=pair_cache, tensor=0)
        self.run("stale/pair", "compare-calibrate", *forward)
        self.run("stale/pair-reversed", "compare-inspect", *backward, *flags(row=0, col=6))
        self.run("stale/pair-reversed/calibrate", "compare-calibrate", *backward)
        options = flags(quantity="delta", mapping="linear", out=prefix)
        self.run("stale/pair-reversed/tile", "compare-tile", *backward, *options)
        self.record_written_files("stale/pair-reversed/tile", prefix)
        itself = flags(model=core, compare_model=core, cache=pair_cache, tensor=0)
        self.run("stale/pair-with-itself/calibrate", "compare-calibrate", *itself)
        self.run("stale/pair-with-itself/tile", "compare-tile", *itself, *options)
        self.record_written_files("stale/pair-with-itself/tile", prefix)

    def record_damaged_caches(self, core: Path, other: Path) -> None:
        """Saved calibration that was edited, or is unreadable, must not be believed."""
        cache, pair_cache = self.work / "cache-damaged", self.work / "cache-damaged-pair"
        source = flags(model=core, cache=cache)
        pair = flags(model=core, compare_model=other, cache=pair_cache)
        self.run("damaged/prepare", "calibrate", *source)
        self.run("damaged/pair/prepare", "compare-calibrate", *pair, *flags(tensor=0))
        steps = (
            ("damaged", cache / "calibration.json", "exact_zero_count", ["calibrate", *source]),
            (
                "damaged/pair",
                pair_cache / "comparison-calibration.json",
                "nonzero_difference_count",
                ["compare-metadata", *pair],
            ),
        )
        for tag, saved, field, command in steps:
            text = saved.read_text() if saved.is_file() else ""
            edited = text.replace(field + '\\":', field + '\\":1', 1)
            self.put(f"{tag}/edited", "status", edited != text)
            if edited != text:
                saved.write_text(edited)
                self.run(f"{tag}/after-edit", *command)
                saved.write_text("not a calibration file")
                self.run(f"{tag}/after-garbage", *command)

    # ---- servers ----

    def fetch(self, key: str, path: str, method="GET", headers=None, kind="json"):
        """One HTTP request to the current server. Records status, headers, then the body."""
        status, response_headers, body = http_request(self.port, path, method, headers)
        self.put(f"{key}/status", "status", status)
        lowered = {name.lower(): value for name, value in response_headers.items()}
        for name in RECORDED_HEADERS:
            if name in lowered:
                self.put(f"{key}/header/{name}", "header", lowered[name])
        if kind == "asset":
            self.put(f"{key}/sha256", "asset", sha256(body))
            return None
        if kind == "image" and status == HTTPStatus.OK:
            self.put(f"{key}/png", "image", sha256(body))
            return None
        try:
            parsed = json.loads(body)
        except ValueError:
            self.put(f"{key}/body", "text", body[:200].decode("latin-1"))
            return None
        self.put_json(f"{key}/body", parsed)
        return parsed

    def fetch_tile(self, key: str, route: str, query: dict) -> None:
        self.fetch(key, f"{route}?{urlencode(query)}", kind="image")

    def raw_request(self, key: str, request: bytes) -> None:
        """Send bytes exactly as given and record the status line."""
        try:
            with socket.create_connection(("127.0.0.1", self.port), timeout=10) as raw:
                raw.sendall(request)
                reply = raw.recv(256)
        except OSError as error:
            self.put(f"{key}/status", "status", type(error).__name__)
            return
        self.put(f"{key}/status", "status", reply.split(b"\r\n")[0].decode("latin-1"))

    def serve(self, tag: str, command: str, source: list[str]) -> subprocess.Popen | None:
        """Start a server and make it the current one; record whether it came up."""
        process, self.port = start_server(self.binary, command, *source)
        self.put(f"{tag}/start", "status", "listening" if process else "failed")
        return process

    def settle(self, key: str, path: str, ready) -> None:
        """Poll the current server until ready(status, json) holds; record whether it did."""
        deadline = time.monotonic() + SETTLE_SECONDS
        settled = False
        while not settled and time.monotonic() < deadline:
            settled = bool(ready(*http_json(self.port, path)))
            if not settled:
                time.sleep(0.05)
        self.put(f"{key}/settled", "status", settled)

    def record_viewer_server(self, model: Path, catalog: dict) -> None:
        process = self.serve("http", "serve", flags(model=model, cache=self.work / "cache-core"))
        if process is None:
            return
        try:
            for path in SERVED_FILES:
                self.fetch(f"http/file{path}", path, kind="asset")
            described = self.fetch("http/api/model", "/api/model") or {}
            rules = [rule["id"] for rule in described.get("rules", [])] or KNOWN_RULES
            for name, tensor in catalog.items():
                self.record_viewer_tensor(name, tensor, rules)
            self.fetch("http/api/progress", "/api/progress")
            self.record_request_rules()
            for name, tensor in catalog.items():
                self.record_viewer_extras(name, tensor, rules)
            self.record_parameter_probes()
        finally:
            stop_server(process)

    def record_viewer_tensor(self, name: str, tensor: dict, rules: list[str]) -> None:
        tid, top = tensor["id"], tensor["max_level"]
        self.fetch(f"http/{name}/status", "/api/tensor-status?" + urlencode({"tensor": tid}))
        for left, right in zip(rules[0::2], rules[1::2], strict=False):
            query = urlencode({"tensor": tid, "left": left, "right": right})
            self.fetch(f"http/{name}/view/{left}+{right}", "/api/view?" + query)
        for row, col in ((0, min(5, tensor["cols"] - 1)), (tensor["rows"] - 1, tensor["cols"] - 1)):
            query = urlencode({"tensor": tid, "row": row, "col": col})
            self.fetch(f"http/{name}/inspect[{row},{col}]", "/api/inspect?" + query)
        for rule in rules:
            for level in sorted({top, 0}):
                query = urlencode({"tensor": tid, "rule": rule, "level": level, "x": 0, "y": 0})
                self.fetch(f"http/{name}/tile[L{level}]/{rule}", "/tile?" + query, kind="image")

    def record_viewer_extras(self, name: str, tensor: dict, rules: list[str]) -> None:
        """Tiles as the page asks for them, neighbouring tiles, and the tensor's edges."""
        where = {"tensor": tensor["id"]}
        for left, right in zip(rules[0::2], rules[1::2], strict=False):
            view = http_json(
                self.port, "/api/view?" + urlencode({**where, "left": left, "right": right})
            )
            bindings = (view[1] or {}).get("tile_bindings") or {}
            self.record_bound_tiles(
                f"http/{name}/bound/{left}+{right}", where, (left, right), bindings
            )
        for turn in ("first", "again"):
            for rule in GRID_RULES:
                for level, x, y in tile_grid(tensor):
                    key = f"http/{name}/grid/{turn}/{rule}[L{level},{x},{y}]"
                    self.fetch_tile(
                        key, "/tile", {**where, "rule": rule, "level": level, "x": x, "y": y}
                    )
        for label, edge in tile_edges(tensor).items():
            self.fetch_tile(
                f"http/{name}/tile[{label}]", "/tile", {**where, "rule": "tensor_linear", **edge}
            )
        beside = urlencode({**where, "row": 0, "col": tensor["cols"]})
        self.fetch(f"http/{name}/inspect[outside-column]", "/api/inspect?" + beside)

    def record_bound_tiles(self, tag: str, where: dict, rules: tuple, bindings: dict) -> None:
        """Tiles requested with the binding a view handed out, and with bindings it did not."""
        origin = {**where, "level": 0, "x": 0, "y": 0}
        for side, rule in zip(("left", "right"), rules, strict=True):
            binding = bindings.get(side)
            self.put(f"{tag}/{side}/offered", "status", isinstance(binding, str))
            if isinstance(binding, str):
                self.fetch_tile(
                    f"{tag}/{side}", "/tile", {**origin, "rule": rule, "binding": binding}
                )
        offered = bindings.get("left")
        if not isinstance(offered, str):
            return
        left, right = rules
        changed = another_binding(offered)
        self.fetch_tile(f"{tag}/changed", "/tile", {**origin, "rule": left, "binding": changed})
        self.fetch_tile(f"{tag}/other-rule", "/tile", {**origin, "rule": right, "binding": offered})
        self.fetch_tile(
            f"{tag}/other-level", "/tile", {**origin, "rule": left, "level": 1, "binding": offered}
        )
        self.fetch_tile(f"{tag}/malformed", "/tile", {**origin, "rule": left, "binding": "zz"})

    def record_request_rules(self) -> None:
        """Requests the server must refuse (and one it must accept), with exact outcomes."""
        wrong_host = {"Host": f"evil.example:{self.port}"}
        self.fetch("http/rules/wrong-host", "/api/model", headers=wrong_host)
        for label, (method, path, headers) in REQUEST_RULES.items():
            self.fetch(f"http/rules/{label}", path, method=method, headers=headers)
        host = f"Host: 127.0.0.1:{self.port}\r\n".encode()
        raw_cases = {
            "raw-bad-escape": b"GET /api/model?a=%a\xc3\xa9 HTTP/1.1\r\n" + host + b"\r\n",
            "raw-http-2": b"GET /api/model HTTP/2.0\r\n" + host + b"\r\n",
            "raw-no-host": b"GET /api/model HTTP/1.1\r\n\r\n",
        }
        for label, request in raw_cases.items():
            self.raw_request(f"http/rules/{label}", request)

    def record_more_rules(self, tag: str, rules: dict, model_path: str) -> None:
        """A table of requests, then a few that have to be sent as raw bytes."""
        localhost = {"Host": f"localhost:{self.port}"}
        self.fetch(f"{tag}/localhost-name", model_path, headers=localhost)
        self.fetch(f"{tag}/wrong-host", model_path, headers={"Host": f"evil.example:{self.port}"})
        same = {"Origin": f"http://127.0.0.1:{self.port}"}
        self.fetch(f"{tag}/same-origin-header", model_path, headers=same)
        for label, (method, path, headers) in rules.items():
            kind = "image" if "/tile" in path else "json"
            self.fetch(f"{tag}/{label}", path, method=method, headers=headers, kind=kind)
        host = f"Host: 127.0.0.1:{self.port}\r\n".encode()
        line = b"GET " + model_path.encode()
        raw_cases = {
            "raw-repeated-header": line + b" HTTP/1.1\r\n" + host + host + b"\r\n",
            "raw-http-1-0": line + b" HTTP/1.0\r\n" + host + b"\r\n",
            "raw-header-without-colon": line + b" HTTP/1.1\r\n" + host + b"nonsense\r\n\r\n",
            "raw-no-version": line + b"\r\n" + host + b"\r\n",
        }
        for label, request in raw_cases.items():
            self.raw_request(f"{tag}/{label}", request)

    def record_parameter_probes(self) -> None:
        self.record_more_rules("http/more", MORE_REQUEST_RULES, "/api/model")
        for label, path in PARAMETER_PROBES.items():
            kind = "image" if path.startswith("/tile") else "json"
            self.fetch(f"http/parameters/{label}", path, kind=kind)
        self.fetch("http/more/model-afterwards", "/api/model")

    def record_sliced_server(self, model: Path, catalog: dict) -> None:
        """Slices and an undecodable tensor over HTTP; this checkpoint can never be complete."""
        source = flags(model=model, cache=self.work / "cache-extras")
        process = self.serve("httpx", "serve", source)
        if process is None:
            return
        try:
            self.fetch("httpx/model", "/api/model")
            for name, tensor in catalog.items():
                for chosen in EXTRA_SLICES.get(name, [""]):
                    where = {"tensor": tensor["id"], **({"slice": chosen} if chosen else {})}
                    tag = f"httpx/{name}" + (f"@{chosen}" if chosen else "")
                    self.record_sliced_tensor(tag, where, tensor)
            conv = catalog.get("conv.weight", {}).get("id", 0)
            probes = {
                "slice-missing": f"/api/view?tensor={conv}",
                "slice-outside": f"/api/view?tensor={conv}&slice=9,9",
                "slice-too-short": f"/api/view?tensor={conv}&slice=1",
                "slice-not-numbers": f"/api/view?tensor={conv}&slice=a,b",
                "inspect-without-slice": f"/api/inspect?tensor={conv}&row=0&col=0",
            }
            for label, path in probes.items():
                self.fetch(f"httpx/rules/{label}", path)
            self.fetch_tile(
                "httpx/rules/tile-without-slice", "/tile", {"tensor": conv, "rule": "tensor_linear"}
            )
            self.fetch("httpx/progress", "/api/progress")
        finally:
            stop_server(process)

    def record_sliced_tensor(self, tag: str, where: dict, tensor: dict) -> None:
        self.fetch(f"{tag}/status", f"/api/tensor-status?tensor={tensor['id']}")
        rules = {"left": "tensor_linear", "right": "tensor_magnitude"}
        view = self.fetch(f"{tag}/view", "/api/view?" + urlencode({**where, **rules})) or {}
        everything = {"left": "global_linear", "right": "global_asinh"}
        self.fetch(f"{tag}/view-global", "/api/view?" + urlencode({**where, **everything}))
        self.fetch(f"{tag}/inspect", "/api/inspect?" + urlencode({**where, "row": 1, "col": 2}))
        for rule in ("tensor_linear", "global_linear"):
            tile = {**where, "rule": rule, "level": tensor["max_level"], "x": 0, "y": 0}
            self.fetch_tile(f"{tag}/tile/{rule}", "/tile", tile)
        bindings = view.get("tile_bindings") or {}
        self.record_bound_tiles(f"{tag}/bound", where, tuple(rules.values()), bindings)

    def record_fresh_viewer(self, model: Path) -> None:
        """A server that must calibrate on request: refusals first, then the same requests."""
        source = flags(model=model, cache=self.work / "cache-fresh")
        process = self.serve("fresh", "serve", source)
        if process is None:
            return
        try:
            self.record_fresh_stage("before")
            one = "/api/calibrate?tensor=0"
            self.fetch("fresh/calibrate-one", one, method="POST", headers=LOCAL)
            self.settle(
                "fresh/calibrate-one", FRESH_VIEW, lambda status, _: status == HTTPStatus.OK
            )
            self.record_fresh_stage("one")
            everything = "/api/calibrate?all=1"
            self.fetch("fresh/calibrate-all", everything, method="POST", headers=LOCAL)
            self.settle("fresh/calibrate-all", "/api/progress", calibration_complete)
            self.record_fresh_stage("all")
        finally:
            stop_server(process)

    def record_fresh_stage(self, stage: str) -> None:
        for label, path in FRESH_PROBES.items():
            self.fetch(f"fresh/{stage}/{label}", path)
        for rule in ("tensor_linear", "global_linear"):
            tile = {"tensor": 0, "rule": rule, "level": 0, "x": 0, "y": 0}
            self.fetch_tile(f"fresh/{stage}/tile/{rule}", "/tile", tile)

    def record_named_server(self, model: Path) -> None:
        """The naming and full-hash options, on a cache that is already calibrated."""
        source = flags(model=model, cache=self.work / "cache-core")
        options = flags(name="Lock Model", revision="rev-7", verify_sha="true")
        process = self.serve("named", "serve", source + options)
        if process is None:
            return
        try:
            self.fetch("named/model", "/api/model")
            self.fetch("named/view", "/api/view?tensor=0&left=tensor_linear&right=tensor_asinh")
            self.fetch("named/progress", "/api/progress")
        finally:
            stop_server(process)

    def record_comparison_server(self, model_a: Path, model_b: Path, pairs: dict) -> None:
        source = flags(model=model_a, compare_model=model_b, cache=self.work / "cache-pair")
        process = self.serve("pairhttp", "compare-serve", source)
        if process is None:
            return
        try:
            described = self.fetch("pairhttp/model", "/api/comparison/model") or {}
            for name, pair in pairs.items():
                self.record_comparison_pair(name, pair)
            self.fetch("pairhttp/single-source-route", "/api/model")
            for path in COMPARISON_FILES:
                self.fetch(f"pairhttp/file{path}", path, kind="asset")
            identity = described.get("comparison_identity")
            for name, pair in pairs.items():
                self.record_comparison_extras(name, pair, identity)
            self.record_more_rules("pairhttp/rules", COMPARISON_RULES, "/api/comparison/model")
        finally:
            stop_server(process)

    def record_comparison_pair(self, name: str, pair: dict) -> None:
        base = "/api/comparison/"
        view = urlencode({"tensor": pair["id"], "left": "a", "right": "delta", "mapping": "linear"})
        self.fetch(f"pairhttp/{name}/view", f"{base}view?{view}")
        where = urlencode({"tensor": pair["id"], "row": 0, "col": 5})
        self.fetch(f"pairhttp/{name}/inspect", f"{base}inspect?{where}")
        for quantity in COMPARE_QUANTITIES:
            tile = {"tensor": pair["id"], "quantity": quantity, "mapping": "linear"}
            tile.update(level=pair["max_level"], x=0, y=0)
            key = f"pairhttp/{name}/tile/{quantity}"
            self.fetch(key, f"{base}tile?{urlencode(tile)}", kind="image")

    def record_comparison_extras(self, name: str, pair: dict, identity) -> None:
        """More views and mappings, neighbouring tiles, edges and the identity parameter."""
        base, where = "/api/comparison/", {"tensor": pair["id"]}
        for left, right, mapping in COMPARE_VIEWS[1:]:
            query = urlencode({**where, "left": left, "right": right, "mapping": mapping})
            self.fetch(f"pairhttp/{name}/view/{left}+{right}+{mapping}", f"{base}view?{query}")
        last = urlencode({**where, "row": pair["rows"] - 1, "col": pair["cols"] - 1})
        self.fetch(f"pairhttp/{name}/inspect[last]", f"{base}inspect?{last}")
        for mapping in COMPARE_MAPPINGS[1:]:
            tile = {**where, "quantity": "delta", "mapping": mapping, "level": 0, "x": 0, "y": 0}
            self.fetch_tile(f"pairhttp/{name}/tile/delta+{mapping}", base + "tile", tile)
        for turn in ("first", "again"):
            for level, x, y in tile_grid(pair):
                tile = {**where, "quantity": "abs_delta", "mapping": "linear", "level": level}
                key = f"pairhttp/{name}/grid/{turn}[L{level},{x},{y}]"
                self.fetch_tile(key, base + "tile", {**tile, "x": x, "y": y})
        for label, edge in tile_edges(pair).items():
            tile = {**where, "quantity": "delta", "mapping": "linear", **edge}
            self.fetch_tile(f"pairhttp/{name}/tile[{label}]", base + "tile", tile)
        origin = {**where, "quantity": "a", "mapping": "linear", "level": 0, "x": 0, "y": 0}
        self.put(f"pairhttp/{name}/identity/offered", "status", isinstance(identity, str))
        if isinstance(identity, str):
            same = {**origin, "comparison_identity": identity}
            self.fetch_tile(f"pairhttp/{name}/identity/same", base + "tile", same)
            other = {**origin, "comparison_identity": another_binding(identity)}
            self.fetch_tile(f"pairhttp/{name}/identity/changed", base + "tile", other)

    def record_fresh_comparison(self, model_a: Path, model_b: Path) -> None:
        """A comparison server with nothing calibrated: refusals, one pair, then again."""
        source = flags(model=model_a, compare_model=model_b, cache=self.work / "cache-pair-fresh")
        process = self.serve("pairfresh", "compare-serve", source)
        if process is None:
            return
        base = "/api/comparison/"
        tile = "tile?quantity=delta&mapping=linear&level=0&x=0&y=0&tensor="
        probes = {
            "model": "model",
            "view": "view?tensor=0&left=a&right=delta&mapping=linear",
            "view-other": "view?tensor=1&left=a&right=delta&mapping=linear",
            "inspect": "inspect?tensor=0&row=0&col=5",
        }
        try:
            for stage in ("before", "after"):
                if stage == "after":
                    calibrate = base + "calibrate?tensor=0"
                    self.fetch("pairfresh/calibrate", calibrate, method="POST", headers=LOCAL)
                    ready = base + probes["view"]
                    self.settle(
                        "pairfresh/calibrate", ready, lambda status, _: status == HTTPStatus.OK
                    )
                for label, path in probes.items():
                    self.fetch(f"pairfresh/{stage}/{label}", base + path)
                for pair in (0, 1):
                    key = f"pairfresh/{stage}/tile[{pair}]"
                    self.fetch(key, f"{base}{tile}{pair}", kind="image")
        finally:
            stop_server(process)

    def record_help(self) -> None:
        shown = run_command(self.binary, "--help")
        self.put("help/exit", "status", "timeout" if shown is None else shown.returncode)
        self.put("help/text", "text", "" if shown is None else shown.stdout.strip())

    def record_everything(self) -> dict[str, list]:
        core_a, core_b, extras = (self.work / name for name in ("core-a", "core-b", "extras"))
        build_checkpoint(core_a, CORE_TENSORS, "a")
        build_checkpoint(core_b, CORE_TENSORS, "b", shards=2)
        build_checkpoint(extras, EXTRA_TENSORS, "a")
        self.record_help()
        catalog = self.record_single_source("core", core_a, {})
        self.record_single_source("sharded", core_b, {})
        extra_catalog = self.record_single_source("extras", extras, EXTRA_SLICES)
        pairs = self.record_comparison(core_a, core_b)
        self.record_refusals(core_a, extras, extra_catalog)
        self.record_viewer_server(core_a, catalog)
        self.record_comparison_server(core_a, core_b, pairs)
        self.record_command_line_probes(core_a, core_b, extras)
        self.record_broken_checkpoints()
        self.record_stepwise_calibration(core_a)
        self.record_stale_caches(core_a, core_b)
        self.record_damaged_caches(core_a, core_b)
        self.record_sliced_server(extras, extra_catalog)
        self.record_fresh_viewer(core_a)
        self.record_named_server(core_a)
        self.record_fresh_comparison(core_a, core_b)
        return self.entries


class Workspace:
    """Where working files are made, and whether they are kept afterwards."""

    def __init__(self, root: Path | None, keep: bool):
        self.root = root
        self.keep = keep
        if root:
            root.mkdir(parents=True, exist_ok=True)

    def make(self, name: str) -> Path:
        return Path(tempfile.mkdtemp(prefix=f"atlas-lock-{name}-", dir=self.root))

    def clear(self, *directories: Path) -> None:
        if not self.keep:
            for directory in directories:
                shutil.rmtree(directory, ignore_errors=True)


def stable_entries(binary: Path, space: Workspace) -> tuple[dict[str, list], list[str]]:
    """Two passes in different directories; keep only what agrees between them.

    Both directories exist until the end, so the second pass cannot be handed
    the same file identities as the first.
    """
    works = [space.make(f"pass{n}") for n in (1, 2)]
    try:
        first, second = (Recorder(binary, work).record_everything() for work in works)
    finally:
        space.clear(*works)
    stable = {key: value for key, value in first.items() if second.get(key) == value}
    volatile = sorted((set(first) | set(second)) - set(stable))
    return stable, volatile


def unusable(entries: dict[str, list]) -> list[str]:
    """Essential steps that did not succeed. A lock recorded without them guards nothing."""
    problems = []
    for key, expected in ESSENTIAL.items():
        found = entries.get(key, ["", "nothing the same on both passes"])[1]
        if found != expected:
            said = entries.get(key.removesuffix("/exit") + "/stderr", ["", ""])[1]
            problems.append(f"  {key}: expected {expected!r}, found {found!r}. {said}".rstrip())
    return problems


# --------------------------------------------------------------------------
# Upgrade check: what the old binary saved must still be honoured
# --------------------------------------------------------------------------

Check = tuple[str, bool, str]


def tile_results(port: int, paths: list[str]) -> dict[str, tuple]:
    """path -> (status, cache header, PNG hash) for each tile."""
    fetched = {}
    for path in paths:
        status, headers, png = http_request(port, path)
        lowered = {name.lower(): value for name, value in headers.items()}
        fetched[path] = (status, lowered.get("x-atlas-cache"), sha256(png))
    return fetched


def viewer_state(binary: str, source: list[str], tiles: list[str], views: list[str]) -> dict:
    """Identities, calibration state, bindings and tile results from one viewer run."""
    process, port = start_server(binary, "serve", *source)
    if process is None:
        return {"started": False}
    try:
        model = http_json(port, "/api/model")[1] or {}
        bindings = {}
        for path in views:
            view = http_json(port, path)[1] or {}
            bindings[path] = (view.get("tile_bindings"), view.get("source_binding"))
        return {
            "started": True,
            "tiles": tile_results(port, tiles),
            "views": bindings,
            "calibrated": model.get("coverage", {}).get("calibrated_tensors"),
            "complete": model.get("calibration_complete"),
            **{name: model.get(name) for name in IDENTITY_NAMES},
        }
    finally:
        stop_server(process)


def comparison_state(binary: str, source: list[str], tiles: list[str]) -> dict:
    """Identity, calibration state and tile results from one comparison-server run."""
    process, port = start_server(binary, "compare-serve", *source)
    if process is None:
        return {"started": False}
    try:
        model = http_json(port, "/api/comparison/model")[1] or {}
        ready = [pair.get("calibration_complete") for pair in model.get("catalog", [])]
        identity = model.get("comparison_identity")
        return {
            "started": True,
            "identity": identity,
            "ready": ready,
            "tiles": tile_results(port, tiles),
        }
    finally:
        stop_server(process)


def viewer_upgrade(old: str, new: str, work: Path) -> list[Check]:
    """The old binary calibrates and serves two checkpoints; the new one must agree."""
    core, extras = work / "model", work / "extras"
    build_checkpoint(core, CORE_TENSORS, "a")
    build_checkpoint(extras, EXTRA_TENSORS, "a")
    described = [json_output(run_command(b, "metadata", *flags(model=core))) for b in (old, new)]
    catalog = (described[0] or {}).get("catalog", [])
    rules = ("tensor_linear", "tensor_asinh", "tensor_magnitude")
    tiles = [
        "/tile?"
        + urlencode({"tensor": t["id"], "rule": rule, "level": t["max_level"], "x": 0, "y": 0})
        for t in catalog
        for rule in rules
    ]
    pair = {"left": "tensor_linear", "right": "tensor_asinh"}
    views = ["/api/view?" + urlencode({"tensor": t["id"], **pair}) for t in catalog]
    sliced = ["/api/view?" + urlencode({"tensor": 0, "slice": "1,2", **pair})]
    sources = [flags(model=core, cache=work / "cache"), flags(model=extras, cache=work / "cache-x")]
    for source in sources:
        run_command(old, "calibrate", *source)
    before = viewer_state(old, sources[0], tiles, views)
    after = viewer_state(new, sources[0], tiles, views)
    before_x = viewer_state(old, sources[1], [], sliced)
    after_x = viewer_state(new, sources[1], [], sliced)
    states = (before, after, before_x, after_x)
    if not (all(described) and all(state.get("started") for state in states)):
        return [("both binaries open and serve the checkpoints", False, "one of them failed")]
    results = viewer_results(described, before, after, len(catalog))
    views_before = {**before["views"], **before_x["views"]}
    views_after = {**after["views"], **after_x["views"]}
    offered = sum(1 for bindings, _ in views_before.values() if bindings)
    same = sum(1 for path, seen in views_after.items() if seen == views_before[path] and seen[0])
    detail = f"{same} of {len(views_before)}"
    ok = same == len(views_before) == offered
    return [*results, ("views hand out the same tile and slice bindings", ok, detail)]


def viewer_results(described, before, after, tensors: int) -> list[Check]:
    same_source = described[0].get("source_identity") == described[1].get("source_identity")
    results = [("metadata reports the same source identity", same_source, "")]
    for name in IDENTITY_NAMES:
        label = f"server reports the same {name.replace('_', ' ')}"
        results.append((label, before[name] == after[name] and before[name] is not None, ""))
    reused = after["calibrated"] == tensors and after["complete"] is True
    detail = f"{after['calibrated']} of {tensors} tensors ready without recalibrating"
    results.append(("saved calibration is reused", reused, detail))
    return [*results, *tile_reuse("cached tiles", before["tiles"], after["tiles"])]


def tile_reuse(what: str, before: dict, after: dict) -> list[Check]:
    hits = sum(1 for result in after.values() if result[1] == "hit")
    same = sum(1 for path, result in after.items() if result[2] == before[path][2])
    total = len(after)
    return [
        (f"{what} are found again", hits == total, f"{hits} of {total}"),
        (f"{what} are byte-identical", same == total, f"{same} of {total}"),
    ]


def comparison_upgrade(old: str, new: str, work: Path) -> list[Check]:
    """The old binary calibrates and serves one pair; the new one must agree and reuse it."""
    first, second = work / "pair-a", work / "pair-b"
    build_checkpoint(first, CORE_TENSORS, "a")
    build_checkpoint(second, CORE_TENSORS, "b", shards=2)
    source = flags(model=first, compare_model=second, cache=work / "cache-pair")
    run_command(old, "compare-calibrate", *source, *flags(tensor=0))
    tile = {"tensor": 0, "mapping": "linear", "level": 0, "x": 0, "y": 0}
    tiles = [
        "/api/comparison/tile?" + urlencode({**tile, "quantity": quantity})
        for quantity in COMPARE_QUANTITIES
    ]
    before = comparison_state(old, source, tiles)
    after = comparison_state(new, source, tiles)
    if not (before.get("started") and after.get("started")):
        return [("both binaries serve the comparison", False, "one of them failed")]
    same = before["identity"] == after["identity"] and before["identity"] is not None
    reused = (
        bool(after["ready"]) and after["ready"] == before["ready"] and after["ready"][0] is True
    )
    return [
        ("comparison server reports the same comparison identity", same, ""),
        ("saved comparison calibration is reused", reused, ""),
        *tile_reuse("cached comparison tiles", before["tiles"], after["tiles"]),
    ]


def upgrade_checks(old: Path, new: Path, space: Workspace) -> list[Check]:
    """The old binary prepares caches; the new binary must agree with them and reuse them."""
    work = space.make("upgrade")
    try:
        return [
            *viewer_upgrade(str(old), str(new), work),
            *comparison_upgrade(str(old), str(new), work),
        ]
    finally:
        space.clear(work)


# --------------------------------------------------------------------------
# Comparing
# --------------------------------------------------------------------------


def count_by_category(entries: dict[str, list]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for category, _ in entries.values():
        counts[category] = counts.get(category, 0) + 1
    return counts


def difference(before: dict, current: dict, ignored: set[str]):
    """Group changed, missing and new keys by category."""
    changed: dict[str, list] = {}
    missing: dict[str, list] = {}
    added: dict[str, list] = {}
    for key, (category, value) in before.items():
        if key in ignored:
            continue
        if key not in current:
            missing.setdefault(category, []).append(key)
        elif current[key][1] != value:
            was_now = f"{key}\n      was {value!r}\n      now {current[key][1]!r}"
            changed.setdefault(category, []).append(was_now)
    for key, (category, _) in current.items():
        if key not in before and key not in ignored:
            added.setdefault(category, []).append(key)
    return changed, missing, added


def split_allowed(groups, allowed: list[str]) -> list[str]:
    """Move differences under an allowed key prefix out of the groups; return them."""
    moved = []
    for group in groups:
        for category, items in group.items():
            kept = [item for item in items if not item.startswith(tuple(allowed))]
            moved += [item for item in items if item.startswith(tuple(allowed))]
            group[category] = kept
    return moved


def print_group(title: str, items: list[str], limit: int | None) -> None:
    """Print one group of differences; a limit of None prints all of them."""
    if not items:
        return
    shown = items if limit is None else items[:limit]
    print(f"\n{title} ({len(items)}):")
    for item in shown:
        print("  " + item)
    if len(shown) < len(items):
        print(f"  ... and {len(items) - len(shown)} more; --full lists them all")


def compare(baseline: dict, current: dict, volatile: list[str], allowed: list[str], limit) -> int:
    """Print differences by category; return the number of strict differences."""
    before = baseline["entries"]
    ignored = set(baseline["volatile"]) | set(volatile)
    changed, missing, added = difference(before, current, ignored)
    excused = split_allowed((changed, missing, added), allowed) if allowed else []
    totals = count_by_category(before)
    print(f"{'category':10} {'recorded':>9} {'changed':>8} {'missing':>8} {'new':>6}")
    for category in (*STRICT, *ADVISORY):
        counts = [len(group.get(category, [])) for group in (changed, missing, added)]
        recorded = totals.get(category, 0)
        print(f"{category:10} {recorded:>9} {counts[0]:>8} {counts[1]:>8} {counts[2]:>6}")
    for category in (*STRICT, *ADVISORY):
        print_group(f"CHANGED {category}", changed.get(category, []), limit)
        print_group(f"MISSING {category}", missing.get(category, []), limit)
        print_group(f"NEW {category}", added.get(category, []), limit)
    print_group(f"ALLOWED by --allow {' '.join(allowed)}", excused, None)

    def total(categories) -> int:
        groups = (changed, missing, added)
        return sum(len(group.get(c, [])) for group in groups for c in categories)

    strict, advisory = total(STRICT), total(ADVISORY)
    print(f"\nStrict differences: {strict}. Page-file differences: {advisory}.")
    return strict


def print_upgrade(results: list[Check]) -> int:
    """Print the upgrade checks; return how many failed."""
    print("\nUpgrade check (old binary's caches opened by the new binary):")
    for label, passed, detail in results:
        print(f"  {'ok  ' if passed else 'FAIL'}  {label}" + (f" ({detail})" if detail else ""))
    return sum(1 for _, passed, _ in results if not passed)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--binary", type=Path, required=True, help="path to weight-atlas-rust")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--record", type=Path, metavar="FILE", help="write a baseline")
    action.add_argument("--compare", type=Path, metavar="FILE", help="compare against a baseline")
    parser.add_argument(
        "--old-binary", type=Path, help="with --compare: also run the upgrade check"
    )
    parser.add_argument(
        "--allow",
        action="append",
        default=[],
        metavar="PREFIX",
        help="with --compare: list differences under this key prefix without failing",
    )
    parser.add_argument("--full", action="store_true", help="print every difference")
    parser.add_argument("--work", type=Path, metavar="DIR", help="make working files here")
    parser.add_argument("--keep", action="store_true", help="do not delete the working files")
    args = parser.parse_args()
    if not args.binary.is_file():
        parser.error(f"No binary at {args.binary}")
    if args.old_binary and (args.record or not args.old_binary.is_file()):
        parser.error("--old-binary needs --compare and an existing file")
    if args.compare and json.loads(args.compare.read_text()).get("lock_version") != LOCK_VERSION:
        parser.error(
            f"That baseline is not from version {LOCK_VERSION} of this script; "
            "record a new one from the same preserved starting binary"
        )
    return args


def record(path: Path, binary: Path, entries: dict[str, list], volatile: list[str]) -> None:
    baseline = {
        "lock_version": LOCK_VERSION,
        "binary_sha256": sha256(binary.read_bytes()),
        "entries": entries,
        "volatile": volatile,
    }
    path.write_text(json.dumps(baseline, indent=1, sort_keys=True))
    print(f"Recorded {len(entries)} stable observations: {count_by_category(entries)}")
    print(f"Left out {len(volatile)} that differed between the two passes.")


def main() -> int:
    args = parse_arguments()
    binary = args.binary.resolve()
    space = Workspace(args.work, args.keep)
    entries, volatile = stable_entries(binary, space)
    problems = unusable(entries)
    if problems:
        print("The binary did not complete the basic steps, so this run proves nothing:")
        print("\n".join(problems))
        print("Its memory and free-disk guards apply to the working directory;")
        print("--work DIR puts the working files somewhere else.")
    if args.record:
        if not problems:
            record(args.record, binary, entries, volatile)
        return 2 if problems else 0
    baseline = json.loads(args.compare.read_text())
    limit = None if args.full else SHOWN_PER_GROUP
    failures = len(problems) + compare(baseline, entries, volatile, args.allow, limit)
    if args.old_binary:
        if sha256(args.old_binary.read_bytes()) != baseline.get("binary_sha256"):
            print("\nNote: --old-binary is not the binary this baseline was recorded from.")
        failures += print_upgrade(upgrade_checks(args.old_binary.resolve(), binary, space))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
