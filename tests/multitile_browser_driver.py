"""Own a synthetic multi-tile source and native server under guarded-core-ui.py."""

import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ["ATLAS_EVIDENCE_DIR"]).resolve()
BIN = Path(os.environ.get("ATLAS_BINARY", ROOT / "target/release/weight-atlas-rust"))


def main():
    assert not OUT.is_relative_to(ROOT), "Evidence must stay outside the checkout"
    OUT.mkdir(parents=True, exist_ok=True)
    model = OUT / "model"
    model.mkdir(exist_ok=False)
    rows, cols = 513, 769
    words = [
        struct.pack("<I", struct.unpack("<I", struct.pack("<f", n))[0])[2:]
        for n in range(-15, 16)
    ]
    payload = b"".join(
        words[(row * 3 + col * 5) % 31] for row in range(rows) for col in range(cols)
    )
    header = json.dumps(
        {
            "multitile": {
                "dtype": "BF16",
                "shape": [rows, cols],
                "data_offsets": [0, len(payload)],
            }
        },
        separators=(",", ":"),
    ).encode()
    header += b" " * (-len(header) % 8)
    raw = struct.pack("<Q", len(header)) + header + payload
    (model / "multitile.safetensors").write_bytes(raw)
    (OUT / "fixture.json").write_text(
        json.dumps(
            {
                "rows": rows,
                "cols": cols,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "formula": "((row*3+col*5)%31)-15",
                "binary_sha256": hashlib.sha256(BIN.read_bytes()).hexdigest(),
                "trained_weights": False,
            },
            indent=2,
        )
        + "\n"
    )
    cache = OUT / "cache"
    with (
        (OUT / "calibrate.stdout").open("w") as stdout,
        (OUT / "calibrate.stderr").open("w") as stderr,
    ):
        subprocess.run(
            [str(BIN), "calibrate", "--model", str(model), "--cache", str(cache)],
            stdout=stdout,
            stderr=stderr,
            timeout=30,
            check=True,
        )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = None
    try:
        with (OUT / "server.log").open("w") as log:
            server = subprocess.Popen(
                [
                    str(BIN),
                    "serve",
                    "--model",
                    str(model),
                    "--cache",
                    str(cache),
                    "--port",
                    str(port),
                ],
                stdout=log,
                stderr=log,
            )
            base = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 10
            while True:
                assert server.poll() is None, "Owned native server exited"
                try:
                    with urllib.request.urlopen(
                        base + "/api/model", timeout=0.5
                    ) as response:
                        metadata = json.load(response)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Owned native server startup deadline")
                    time.sleep(0.05)
            assert (
                metadata["catalog"][0]["rows"] == rows
                and metadata["catalog"][0]["cols"] == cols
            )
            (OUT / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")
            browser_out = OUT / "browser"
            env = {
                **os.environ,
                "ATLAS_TEST_URL": base,
                "ATLAS_EVIDENCE_DIR": str(browser_out),
                "NODE_PATH": str(ROOT / "dev/node_modules"),
            }
            with (
                (OUT / "browser.stdout").open("w") as stdout,
                (OUT / "browser.stderr").open("w") as stderr,
            ):
                result = subprocess.run(
                    ["node", "tests/multitile-browser.cjs"],
                    cwd=ROOT,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=90,
                )
            assert result.returncode == 0, (OUT / "browser.stderr").read_text()
    finally:
        if server is not None:
            if server.poll() is None:
                server.terminate()
            server.wait(timeout=5)
        (OUT / "server-cleanup.json").write_text(
            json.dumps(
                {
                    "owned_pid": None if server is None else server.pid,
                    "reaped": server is None or server.poll() is not None,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
