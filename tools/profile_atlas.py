#!/usr/bin/env python3
"""Explicit local fixture-profile launcher. Source candidate; acceptance pending.

The owner supplies a qualified native binary digest. No calibrated validation
recipe is accepted here: HTTP uses the existing default 5 GiB admission gate.
"""

import argparse
import signal
import time
from http.server import HTTPServer
from pathlib import Path
from typing import Any, Sequence, cast

from atlas_host.common import digest, require
from atlas_host.config import load_config
from atlas_host.hosted_runtime import HostedApplication
from atlas_host.profile_http import HostedHandler, HostedServer
from atlas_host.registry import Registry

ROOT = Path(__file__).resolve().parents[1]


def run(config: dict[str, Any], binary: str | Path, expected_sha: str) -> None:
    digest(expected_sha)
    require(config["bind"] == "127.0.0.1", "Local profile host requires loopback")
    app = HostedApplication(
        Registry(config["paths"]["registry"]),
        config["paths"]["cache"],
        binary,
        expected_sha,
    )  # No launch_policy: default gate.
    server: HostedServer | None = None

    def shutdown(*_: object) -> None:
        raise KeyboardInterrupt

    try:
        app.start_threads()
        server = cast(
            HostedServer,
            HTTPServer((config["bind"], config["ports"]["coordinator"]), HostedHandler),
        )
        server.application = app
        server.host = app.host
        server.profile_controls = True  # Trusted owner entrypoint, never request input.
        server.timeout = 0.1
        signal.signal(signal.SIGTERM, shutdown)
        signal.signal(signal.SIGINT, shutdown)
        print(
            f"Local fixture profiles: http://127.0.0.1:{server.server_port}", flush=True
        )
        while not app.lifetime.stopping:
            server.handle_request()
            app.tick()
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        # Keep ownership reachable, with watchdog alive, until confirmed cleanup.
        # A stuck/uncertain close never becomes a successful launcher exit.
        while True:
            try:
                if app.close():
                    break
            except (ValueError, OSError):
                pass
            time.sleep(0.05)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "target/release/weight-atlas-rust"
    )
    parser.add_argument("--binary-sha256", required=True)
    args = parser.parse_args(argv)
    run(load_config(args.config), args.binary, args.binary_sha256)


if __name__ == "__main__":
    main()
