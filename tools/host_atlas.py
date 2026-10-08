#!/usr/bin/env python3
"""Optional loopback fixture picker. Requires a separately qualified Rust binary."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
from typing import Any, BinaryIO, NoReturn, Protocol, Sequence, cast
from http.server import HTTPServer

from atlas_host import limits as _limits
from atlas_host.common import canonical, require
from atlas_host.host_assets import ASSETS as ASSETS, BUNDLE as BUNDLE
from atlas_host.config import load_config
from atlas_host.registry import Registry, RegistryEntry
from atlas_host.runtime_adapter import FixtureHost, HostError, NativeResponse, dispatch
from live_inference import (
    Handler as GuardedHandler,
    BackendError,
    TickOwner,
    proxy_request,
    signal_and_reap,
    available,
    GIB,
)

ROOT = Path(__file__).resolve().parents[1]


class FixtureServer(Protocol):
    host: FixtureHost
    server_port: int
    timeout: float | None

    def handle_request(self) -> None: ...
    def server_close(self) -> None: ...


class Renderer:
    def __init__(
        self, entry: RegistryEntry, cache: str | Path, port: int, session: TickOwner
    ) -> None:
        require(
            available() >= _limits.MODEL_ADMISSION_GIB * GIB,
            "Existing launch memory guard refused",
        )
        binary = ROOT / "target/release/weight-atlas-rust"
        require(binary.is_file(), "Qualified Rust binary required")
        self.port, self.session = port, session
        self.raw = bytearray()
        self.announced = False
        self.stopping = False
        # Return the owned handle before any fallible post-spawn pipe setup.
        # FixtureHost installs it in the slot before calling initialize().
        self.process = subprocess.Popen(
            [
                str(binary),
                "serve",
                "--model",
                entry["root"],
                "--cache",
                str(cache),
                "--name",
                entry["name"],
                "--revision",
                entry["manifest"]["revision"],
                "--port",
                str(port),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def initialize(self) -> None:
        os.set_blocking(cast(BinaryIO, self.process.stdout).fileno(), False)

    def alive(self) -> bool:
        return self.process.poll() is None

    def ready(self) -> bool:
        if self.announced:
            return self.alive()
        try:
            data = os.read(
                cast(BinaryIO, self.process.stdout).fileno(), 8193 - len(self.raw)
            )
        except BlockingIOError:
            return False
        self.raw.extend(data)
        require(len(self.raw) <= 8192, "Renderer startup message too large")
        if b"\n" not in self.raw:
            return False
        notice = json.loads(bytes(self.raw).split(b"\n", 1)[0])
        require(
            notice.get("listening") == f"http://127.0.0.1:{self.port}",
            "Renderer startup mismatch",
        )
        self.announced = True
        return self.alive()

    def read(self, path: str) -> NativeResponse:
        require(self.ready(), "Renderer startup not acknowledged")
        return proxy_request(self.port, "GET", path, self.session)

    def stop(self) -> bool:
        done = signal_and_reap(self.process, terminate=not self.stopping, timeout=0.2)
        self.stopping = True
        if done:
            cast(BinaryIO, self.process.stdout).close()
        return done


class HostHandler(GuardedHandler):
    """Inherit existing absolute receive/Host/Origin/body bounds and no-store send."""

    def dispatch_host(
        self, method: str, raw_path: str, data: dict[str, Any] | None = None
    ) -> NativeResponse:
        return dispatch(cast(FixtureServer, self.server).host, method, raw_path, data)

    def handle_action(self) -> None:
        try:
            path, length = self.validated()
            if self.command == "POST" and self.headers.get("X-Atlas-Local") != "1":
                raise ValueError("Local action header required")
            if self.command == "GET" and path in ("/", "/index.html"):
                require(not length, "GET bodies are unsupported")
                return self.send(
                    200,
                    (ROOT / "web/index.html").read_bytes(),
                    "text/html; charset=utf-8",
                )
            if self.command == "GET" and path == "/viewer.js":
                require(not length, "GET bodies are unsupported")
                return self.send(
                    200,
                    b"\n;\n".join(
                        (ROOT / "web" / name).read_bytes() for name in BUNDLE
                    ),
                    "text/javascript",
                )
            if self.command == "GET" and path in ASSETS:
                require(not length, "GET bodies are unsupported")
                name, mime = ASSETS[path]
                return self.send(200, (ROOT / "web" / name).read_bytes(), mime)
            if path.startswith("/api/inference") or path.startswith("/api/analytics"):
                return self.send(
                    503,
                    {
                        "api_version": 1,
                        "code": "disabled",
                        "error": "Inference and analytics are disabled in fixture host mode",
                    },
                )
            require(self.command == "POST" or not length, "GET bodies are unsupported")
            data = (
                json.loads(self.rfile.read(length)) if self.command == "POST" else None
            )
            status, body, mime = self.dispatch_host(self.command, self.path, data)
            self.send(status, body, mime)
        except HostError as error:
            self.send(
                error.status,
                {"api_version": 1, "code": error.code, "error": str(error)},
            )
        except Exception as error:
            # Public errors never include owner paths, raw backend payloads or credentials.
            if isinstance(error, BackendError):
                self.send(
                    error.status,
                    {
                        "api_version": 1,
                        "code": error.code,
                        "error": "Fixture backend unavailable; retry status",
                    },
                )
            elif isinstance(error, (ValueError, OSError, TypeError, KeyError)):
                self.send(
                    400,
                    {
                        "api_version": 1,
                        "code": "invalid_request",
                        "error": "Invalid or unavailable fixture request",
                    },
                )
            else:
                raise

    do_GET = handle_action
    do_POST = handle_action


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    if available() < _limits.MODEL_ADMISSION_GIB * GIB:
        raise SystemExit("Need 4.75 GiB available RAM")
    host = FixtureHost(
        Registry(config["paths"]["registry"]),
        config["paths"]["cache"],
        lambda entry, cache: Renderer(entry, cache, config["ports"]["renderer"], host),
    )
    server = cast(
        FixtureServer,
        HTTPServer((config["bind"], config["ports"]["coordinator"]), HostHandler),
    )
    server.host = host
    server.timeout = 0.1

    def shutdown(*_: object) -> NoReturn:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    print(f"Fixture host: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        while available() >= _limits.MEMORY_RESERVE_GIB * GIB:
            server.handle_request()
            host.tick()
    except KeyboardInterrupt:
        pass
    finally:
        done = False
        for _ in range(8):
            if host.close():
                done = True
                break
        server.server_close()
        if not done:
            raise SystemExit("Shutdown incomplete: owned renderer cleanup pending")


if __name__ == "__main__":
    main()
