"""Actual coordinator archive acceptance. Invoke under guarded-core-ui.py."""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as value:
        value.bind(("127.0.0.1", 0))
        return value.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    args = parser.parse_args()
    out = Path(os.environ["ATLAS_EVIDENCE_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    ports = [free_port(), free_port()]
    assert len(set(ports)) == 2 and not set(ports) & {8774, 8775, 8785}
    coordinator = None
    try:
        with (
            (out / "coordinator.stdout").open("w") as stdout,
            (out / "coordinator.stderr").open("w") as stderr,
        ):
            coordinator = subprocess.Popen(
                [
                    "/usr/bin/python3",
                    "-B",
                    "tools/live_inference.py",
                    "--model",
                    str(args.model),
                    "--python",
                    str(args.python),
                    "--port",
                    str(ports[0]),
                    "--atlas-port",
                    str(ports[1]),
                ],
                cwd=ROOT,
                stdout=stdout,
                stderr=stderr,
            )
            base = "http://127.0.0.1:" + str(ports[0])
            deadline = time.monotonic() + 12
            while True:
                assert (
                    coordinator.poll() is None
                ), "Owned coordinator exited during readiness"
                try:
                    with urllib.request.urlopen(
                        base + "/api/model", timeout=0.3
                    ) as response:
                        model = json.load(response)
                    break
                except OSError:
                    assert time.monotonic() < deadline
                    time.sleep(0.05)
            assert model["revision"] == "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
            environment = {**os.environ, "ATLAS_TEST_URL": base}
            subprocess.run(
                ["node", "tests/archive-browser.cjs"],
                cwd=ROOT,
                env=environment,
                timeout=90,
                check=True,
            )
    finally:
        if coordinator is not None:
            if coordinator.poll() is None:
                coordinator.terminate()
            try:
                coordinator.wait(timeout=5)
            except subprocess.TimeoutExpired:
                coordinator.kill()
                coordinator.wait(timeout=3)
        closed = []
        for port in ports:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    closed.append(False)
            except OSError:
                closed.append(True)
        receipt = {
            "coordinator_reaped": coordinator is None or coordinator.poll() is not None,
            "ports_closed": closed,
            "registry_used": False,
            "model_jobs_started": 0,
        }
        (out / "driver-cleanup.json").write_text(json.dumps(receipt, indent=2) + "\n")
        assert receipt["coordinator_reaped"] and all(closed)


if __name__ == "__main__":
    main()
