"""Own the existing native/pinned CPU polish servers; no registry or downloads."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "target/release/weight-atlas-rust"
spec = importlib.util.spec_from_file_location(
    "polish_guard", ROOT / "tools/guarded-core-ui.py"
)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=["desktop", "mobile", "qwen", "inference"], required=True
    )
    parser.add_argument(
        "--python",
        type=Path,
        help="Existing qualified CPU interpreter; required for Smol desktop/mobile/inference",
    )
    parser.add_argument(
        "--section",
        choices=["all", "pointer", "navigation", "light", "dark"],
        default="all",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check file/package prerequisites without starting a process or reading weights",
    )
    args = parser.parse_args()
    if not args.model.is_dir() or not BIN.is_file():
        parser.error("Existing model directory and built release binary are required")
    if args.phase != "qwen" and (args.python is None or not args.python.is_file()):
        parser.error(
            "Smol cases require --python with the existing qualified CPU environment"
        )
    chromium = Path(os.environ.get("ATLAS_CHROMIUM", ""))
    if not chromium.is_file():
        parser.error(
            "Set ATLAS_CHROMIUM to an existing sandbox-compatible Chromium executable"
        )
    for name in ["playwright", "prettier"]:
        package = ROOT / "dev/node_modules" / name / "package.json"
        pins = json.loads((ROOT / "dev/versions.json").read_text())
        if (
            not package.is_file()
            or json.loads(package.read_text())["version"] != pins[name]
        ):
            parser.error(f"Install the documented exact development {name} pin first")
    if args.check:
        print(
            "Prerequisite files and exact browser packages present; no server, worker or browser started. Source pins, identities and runtime/resource gates remain launch requirements."
        )
        return
    out = Path(os.environ["ATLAS_EVIDENCE_DIR"]).resolve() / "polish"
    if out.is_relative_to(ROOT) or out.exists():
        parser.error(
            "ATLAS_EVIDENCE_DIR must be a fresh directory outside the checkout"
        )
    out.mkdir(parents=True)
    assert guard.available() >= 5 * 1024**3, "Need 5 GiB available RAM"
    assert shutil.disk_usage(out).free >= 25 * 1024**3, "Need 25 GiB disk reserve"
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    observed = json.loads(
        (
            ROOT
            / "tests/fixtures"
            / (
                "ui-polish-qwen-observations.json"
                if args.phase == "qwen"
                else "ui-polish-observations.json"
            )
        ).read_text()
    )
    ports = [free_port(), free_port()]
    assert len(set(ports)) == 2 and not set(ports) & {8774, 8775, 8785}
    owner = ROOT
    if args.phase != "qwen":
        # The existing coordinator's relative cache is isolated outside the checkout.
        owner = out / "owner"
        for folder in ["tools", "web", "docs/models", "tests/fixtures"]:
            shutil.copytree(
                ROOT / folder,
                owner / folder,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        (owner / "target/release").mkdir(parents=True)
        shutil.copy2(BIN, owner / "target/release/weight-atlas-rust")
    cache = owner / "cache-inference" if args.phase != "qwen" else out / "cache"
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "NODE_PATH": str(ROOT / "dev/node_modules"),
        "ATLAS_POLISH_PHASE": args.phase,
        "ATLAS_DESKTOP_SECTION": args.section,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    calibrate = [
        str(BIN),
        "calibrate",
        "--model",
        str(args.model.resolve()),
        "--cache",
        str(cache),
        "--revision",
        observed["revision"],
    ]
    if args.phase == "inference":
        env["ATLAS_EVIDENCE_DIR"] = str(out / "calibration-guard")
        calibrate = [
            sys.executable,
            "-B",
            str(ROOT / "tools/guarded-core-ui.py"),
            *calibrate,
        ]
    with (
        (out / "calibrate.stdout").open("w") as stdout,
        (out / "calibrate.stderr").open("w") as stderr,
    ):
        subprocess.run(
            calibrate,
            cwd=ROOT,
            env=env,
            stdout=stdout,
            stderr=stderr,
            timeout=120,
            check=True,
        )
    command = [
        str(BIN),
        "serve",
        "--model",
        str(args.model.resolve()),
        "--cache",
        str(cache),
        "--revision",
        observed["revision"],
        "--port",
        str(ports[0]),
    ]
    if args.phase != "qwen":
        command = [
            sys.executable,
            "-B",
            str(owner / "tools/live_inference.py"),
            "--model",
            str(args.model.resolve()),
            "--python",
            str(args.python.absolute()),
            "--revision",
            observed["revision"],
            "--port",
            str(ports[0]),
            "--atlas-port",
            str(ports[1]),
        ]
        if args.phase != "inference":
            command.append("--analytics-only")
    server = None
    owned = {}
    try:
        with (
            (out / "server.stdout").open("w") as stdout,
            (out / "server.stderr").open("w") as stderr,
        ):
            server = subprocess.Popen(
                command, cwd=owner, env=env, stdout=stdout, stderr=stderr
            )
            owned[server.pid] = guard.table()[server.pid][1]
            base = f"http://127.0.0.1:{ports[0]}"
            deadline = time.monotonic() + 12
            while True:
                assert server.poll() is None, "Owned server exited during readiness"
                try:
                    with urllib.request.urlopen(
                        base + "/api/model", timeout=0.5
                    ) as response:
                        model = json.load(response)
                    break
                except OSError:
                    assert time.monotonic() < deadline, "Owned server startup deadline"
                    time.sleep(0.05)
            assert (
                model["source_identity"] == observed["source_identity"]
            ), "Selected source must match held observations; path/inode/timestamp identities are LOCAL"
            (out / "launch.json").write_text(
                json.dumps(
                    {
                        "model": model,
                        "binary_sha256": hashlib.sha256(BIN.read_bytes()).hexdigest(),
                        "registry_used": False,
                        "phase": args.phase,
                        "separate_model_owner": args.phase != "qwen",
                    },
                    indent=2,
                )
                + "\n"
            )
            env.update(ATLAS_TEST_URL=base, ATLAS_EVIDENCE_DIR=str(out / "browser"))
            browser = ["node", "tests/ui-polish-browser.cjs"]
            if args.phase == "inference":
                browser = [sys.executable, "-B", "tools/guarded-core-ui.py", *browser]
            with (
                (out / "browser.stdout").open("w") as stdout,
                (out / "browser.stderr").open("w") as stderr,
            ):
                child = subprocess.Popen(
                    browser, cwd=ROOT, env=env, stdout=stdout, stderr=stderr
                )
                owned[child.pid] = guard.table()[child.pid][1]
                stop = time.monotonic() + 120
                while child.poll() is None:
                    guard.discover(guard.table(), owned)
                    assert time.monotonic() < stop, "Browser deadline"
                    time.sleep(0.05)
                assert child.returncode == 0, "Browser failed; retained raw stderr"
    finally:
        cleanup = guard.cleanup(server, owned, os.getpid())
        (out / "driver-cleanup.json").write_text(
            json.dumps(
                {
                    "owned_pid": None if server is None else server.pid,
                    "reaped": server is None or server.poll() is not None,
                    "registry_used": False,
                    **cleanup,
                },
                indent=2,
            )
            + "\n"
        )
        assert (
            cleanup["cleanup_verified"]
            and not cleanup["remaining_owned_pids"]
            and not cleanup["cleanup_errors"]
        ), "Owned cleanup failed"


if __name__ == "__main__":
    main()
