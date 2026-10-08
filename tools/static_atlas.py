#!/usr/bin/env python3
"""Explicit loopback static host. Dense admission stays closed without an owner bundle.

The optional policy/digest pair is trusted owner input for its separately reviewed
scope. It does not register a model, accept a license or perform qualification.
"""

import argparse
from http.server import HTTPServer
from pathlib import Path
from typing import Any, Sequence, cast
import signal
import time

from atlas_host.common import digest, require
from atlas_host.config import load_config
from atlas_host.dense_static_admission import bind_dense_policy
from atlas_host.hosted_runtime import HostedApplication
from atlas_host.profile_http import HostedHandler, HostedServer
from atlas_host.registry import Registry

ROOT = Path(__file__).resolve().parents[1]


def run(
    config: dict[str, Any],
    binary: str | Path,
    expected_sha: str,
    *,
    policy_path: str | Path | None = None,
    approved_sha: str | None = None,
) -> None:
    digest(expected_sha)
    require(
        (policy_path is None) == (approved_sha is None),
        "Supply both owner policy/digest arguments",
    )
    require(config["bind"] == "127.0.0.1", "Static host requires loopback")
    registry = Registry(config["paths"]["registry"])
    policy = (
        bind_dense_policy(
            policy_path,
            cast(str, approved_sha),
            registry,
            config["paths"]["cache"],
            binary,
        )
        if policy_path is not None
        else None
    )
    require(
        policy is None or policy.binary_sha == expected_sha,
        "Static executable digest differs",
    )
    app = HostedApplication(
        registry, config["paths"]["cache"], binary, expected_sha, dense_policy=policy
    )
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
        server.profile_controls = False
        server.timeout = 0.1
        signal.signal(signal.SIGTERM, shutdown)
        signal.signal(signal.SIGINT, shutdown)
        print(f"Local static host: http://127.0.0.1:{server.server_port}", flush=True)
        while not app.lifetime.stopping:
            server.handle_request()
            app.tick()
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        while True:
            try:
                if app.close():
                    break
            except (ValueError, OSError):
                pass
            time.sleep(0.05)  # Ownership remains reachable until confirmed cleanup.


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "target/release/weight-atlas-rust"
    )
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--dense-static-policy", type=Path)
    parser.add_argument("--approved-dense-static-sha256")
    args = parser.parse_args(argv)
    require(
        (args.dense_static_policy is None)
        == (args.approved_dense_static_sha256 is None),
        "Supply both owner policy/digest arguments",
    )
    run(
        load_config(args.config),
        args.binary,
        args.binary_sha256,
        policy_path=args.dense_static_policy,
        approved_sha=args.approved_dense_static_sha256,
    )


if __name__ == "__main__":
    main()
