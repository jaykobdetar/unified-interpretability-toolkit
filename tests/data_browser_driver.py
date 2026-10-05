"""Owned loopback synthetic data-browser lifecycle; invoke under guarded-core-ui.py."""

import argparse
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
from data_workflow_fixture import write, REVISION

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "f9d8a56514a210314af82872040c6c03e90d7f86"


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_source_binding(root, binary, source_commit, binary_sha256):
    assert len(source_commit) == 40 and all(
        c in "0123456789abcdef" for c in source_commit
    ), "Use a full immutable source commit"
    resolved = subprocess.check_output(
        ["git", "rev-parse", source_commit + "^{commit}"], cwd=root, text=True
    ).strip()
    assert resolved == source_commit, "Source commit did not resolve exactly"
    assert not subprocess.check_output(
        ["git", "diff", "--name-only", source_commit, "--", "src", "web"],
        cwd=root,
        text=True,
    ).strip(), "Production sources differ from selected candidate"
    assert binary.is_file(), "Missing release binary"
    actual_sha = sha(binary)
    if source_commit != CANDIDATE:
        assert (
            binary_sha256 is not None
        ), "An explicit source commit requires its recorded binary SHA-256"
    if binary_sha256 is not None:
        assert len(binary_sha256) == 64 and all(
            c in "0123456789abcdef" for c in binary_sha256
        ), "Use a full binary SHA-256"
        assert actual_sha == binary_sha256, "Binary differs from selected build"
    return {
        "production_commit": source_commit,
        "production_tree": subprocess.check_output(
            ["git", "rev-parse", source_commit + "^{tree}"], cwd=root, text=True
        ).strip(),
        "test_checkout_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "historical_candidate": CANDIDATE,
        "binary_sha256": actual_sha,
        "binary_pin_verified": binary_sha256 is not None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("exports", "archives"))
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--source-commit", default=CANDIDATE)
    parser.add_argument("--binary-sha256")
    args = parser.parse_args()
    out = Path(os.environ["ATLAS_EVIDENCE_DIR"]).resolve()
    assert not out.is_relative_to(ROOT), "Evidence stays outside checkout"
    assert os.environ.get("ATLAS_CHROMIUM") and os.environ.get(
        "NODE_PATH"
    ), "Explicit installed browser runtime required"
    binary = args.binary.resolve()
    source_binding = validate_source_binding(
        ROOT, binary, args.source_commit, args.binary_sha256
    )
    out.mkdir(parents=True, exist_ok=True)
    fixture = out / "fixture"
    manifest = write(fixture)
    cache = out / "cache"
    with (out / "calibrate.log").open("w") as log:
        subprocess.run(
            [
                str(binary),
                "calibrate",
                "--model",
                str(fixture),
                "--cache",
                str(cache),
                "--revision",
                REVISION,
                "--name",
                "Synthetic data workflow fixture",
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=10,
            check=True,
        )
    port = free_port()
    log = (out / "renderer.log").open("w")
    child = subprocess.Popen(
        [
            str(binary),
            "serve",
            "--model",
            str(fixture),
            "--cache",
            str(cache),
            "--revision",
            REVISION,
            "--name",
            "Synthetic data workflow fixture",
            "--port",
            str(port),
        ],
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    mutated, proxy_errors, server = [], [], None

    class Adapter(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            mutated.append({"method": "POST", "route": self.path})
            self.send_response(405)
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            self.end_headers()

        do_PUT = do_POST
        do_PATCH = do_POST
        do_DELETE = do_POST

        def do_GET(self):
            if self.path == "/api/inference":
                data = json.dumps(
                    {
                        "model": "Synthetic archive fixture — no worker",
                        "engine": "metadata-only qualification adapter; model execution disabled",
                        "fixture_only": True,
                        "busy": False,
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(data)
                return
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
            try:
                assert len(self.path) <= 8192 and self.path.startswith("/")
                conn.request("GET", self.path)
                response = conn.getresponse()
                data = response.read(2 * 1024 * 1024 + 1)
                assert len(data) <= 2 * 1024 * 1024
                self.send_response(response.status)
                for key, value in response.getheaders():
                    if key.lower() not in (
                        "content-length",
                        "connection",
                        "transfer-encoding",
                        "server",
                        "date",
                    ):
                        self.send_header(key, value)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(data)
            except (AssertionError, OSError, http.client.HTTPException) as error:
                proxy_errors.append(type(error).__name__)
                self.close_connection = True
            finally:
                conn.close()

    ready_deadline = time.monotonic() + 5
    try:
        while True:
            assert child.poll() is None, "Owned renderer exited before readiness"
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                assert time.monotonic() < ready_deadline, "Renderer readiness timeout"
                time.sleep(0.05)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Adapter)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = "http://127.0.0.1:" + str(server.server_port)
        lock = ROOT / "tools/behaviour_lock.py"
        binding = {
            **source_binding,
            "behaviour_lock_sha256": sha(lock) if lock.is_file() else None,
            "harness_sha256": sha(__file__),
            "fixture_sha256": manifest["sha256"],
            "fixture_bytes": manifest["bytes"],
            "phase": args.phase,
            "adapter": "metadata-only fixture; actual rebuilt Rust assets/CSP; no inference integration claim",
            "assets": {
                name: sha(ROOT / "web" / name)
                for name in (
                    "vendor/openseadragon.min.js",
                    "atlas-tools.js",
                    "app.js",
                    "workspace-tools.js",
                    "inference.js",
                    "inference-import.js",
                )
            },
        }
        (out / "source-binding.json").write_text(json.dumps(binding, indent=2) + "\n")
        result = subprocess.run(
            ["node", "tests/data-workflows-browser.cjs"],
            cwd=ROOT,
            env={**os.environ, "ATLAS_TEST_URL": base, "ATLAS_DATA_PHASE": args.phase},
            timeout=90,
        )
        assert result.returncode == 0, f"Browser phase exited {result.returncode}"
        assert (
            not mutated and not proxy_errors
        ), "Unexpected mutation or actual proxy failure"
    finally:
        if server:
            server.shutdown()
            server.server_close()
        if child.poll() is None:
            child.terminate()
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=2)
        log.close()
        closed = {}
        for label, value in [
            ("renderer", port),
            ("adapter", server.server_port if server else None),
        ]:
            if value is None:
                continue
            try:
                with socket.create_connection(("127.0.0.1", value), timeout=0.1):
                    closed[label] = False
            except OSError:
                closed[label] = True
        receipt = {
            "renderer_reaped": child.poll() is not None,
            "ports_closed": closed,
            "unexpected_mutations": mutated,
            "proxy_errors": proxy_errors,
        }
        (out / "fixture-cleanup.json").write_text(json.dumps(receipt, indent=2) + "\n")
        assert receipt["renderer_reaped"] and all(
            closed.values()
        ), "Owned fixture cleanup failed"


if __name__ == "__main__":
    main()
