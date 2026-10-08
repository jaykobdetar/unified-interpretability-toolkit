#!/usr/bin/env python3
"""Portable configuration and preflight for the Linux, loopback-only viewer."""

import argparse
from collections.abc import Iterable
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Literal, overload

from atlas_host.viewer_resources import encode as encode_resources

ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / "target/release/weight-atlas-rust"
FIELDS = {"model", "cache", "port", "name", "revision", "resources"}
__all__ = [
    "argparse",
    "json",
    "os",
    "Path",
    "shutil",
    "subprocess",
    "sys",
    "encode_resources",
    "ROOT",
    "BINARY",
    "FIELDS",
    "unique_fields",
    "main",
]


def unique_fields(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate configuration field: " + key)
        value[key] = item
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Weight Atlas viewer (no inference or model downloads).",
        epilog="Inference: python3 tools/live_inference.py --help. CLI flags override ATLAS_* variables, then JSON config. Paths in JSON are relative to that file; other paths are relative to your current directory.",
    )
    parser.add_argument("model", nargs="?", help="local safetensors directory")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="use the included 244-byte synthetic fixture",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="build offline with existing resource guards before launch",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="check configuration and dependencies without building or starting a server",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("ATLAS_CONFIG"),
        help="JSON configuration file (never executed as shell code)",
    )
    for field in ("cache", "port", "name", "revision"):
        parser.add_argument("--" + field)
    args = parser.parse_args(argv)
    if args.demo and args.model:
        parser.error("choose --demo or a model directory")
    config: dict[str, Any] = {}
    base = Path.cwd()
    if args.config:
        source = Path(args.config).expanduser().resolve()
        try:
            config = json.loads(source.read_text(), object_pairs_hook=unique_fields)
        except (OSError, ValueError) as exc:
            parser.error(f"cannot read JSON configuration: {exc}")
        if not isinstance(config, dict) or set(config) - FIELDS:
            parser.error(
                "configuration must be an object containing only model, cache, port, name, revision, resources"
            )
        if any(
            not isinstance(v, (str, int)) or isinstance(v, bool)
            for k, v in config.items()
            if k != "resources"
        ):
            parser.error(
                "configuration values must be strings (port may also be an integer)"
            )
        base = source.parent
    try:
        resources = encode_resources(config.get("resources", {}))
    except ValueError as exc:
        parser.error(str(exc))

    @overload
    def value(field: Literal["cache"], default: str) -> Path: ...

    @overload
    def value(field: Literal["model"], default: None = None) -> Path | None: ...

    @overload
    def value(field: Literal["port"], default: int) -> str | int: ...

    @overload
    def value(field: Literal["name", "revision"], default: str) -> str | int: ...

    def value(field: str, default: str | int | None = None) -> str | int | Path | None:
        explicit: str | None = getattr(args, field, None) or os.environ.get(
            "ATLAS_" + field.upper()
        )
        item: str | int | None = (
            explicit if explicit is not None else config.get(field, default)
        )
        if field in ("model", "cache") and item is not None:
            path = Path(str(item)).expanduser()
            return (
                path
                if path.is_absolute()
                else (Path.cwd() if explicit is not None else base) / path
            ).resolve()
        return item

    model = ROOT / "fixtures/tiny-bf16" if args.demo else value("model")
    if model is None:
        parser.error("select a model directory or use --demo")
    cache = value("cache", str(ROOT / "cache"))
    try:
        port = int(value("port", 8775))
    except (ValueError, TypeError):
        parser.error("port must be an integer from 1 to 65535")
    if not 1 <= port <= 65535:
        parser.error("port must be an integer from 1 to 65535")
    if sys.platform != "linux":
        parser.error(
            "Linux is required for the existing CPU, memory, and process guards"
        )
    if not model.is_dir():
        parser.error("model directory does not exist")
    if cache == model or model in cache.parents:
        parser.error("cache must be outside the model directory")
    if args.build and shutil.which("cargo") is None:
        parser.error(
            "Cargo is missing; install Rust with the clippy component, then retry --build"
        )
    if not args.build and not os.access(BINARY, os.X_OK):
        parser.error("viewer binary is missing; run ./run-atlas.sh --demo --build")
    if args.check:
        print(
            "Configuration and dependencies OK; no model payload read, build, cache write, or server started. Runtime resource and source checks run at launch."
        )
        return 0
    if args.build:
        result = subprocess.call(
            [
                sys.executable,
                str(ROOT / "tools/guarded-build.py"),
                "build",
                "--release",
            ],
            cwd=ROOT,
        )
        if result:
            return result
    command = [
        str(BINARY),
        "serve",
        "--model",
        str(model),
        "--cache",
        str(cache),
        "--port",
        str(port),
        "--name",
        str(
            value(
                "name",
                "Synthetic BF16 fixture" if args.demo else "Local safetensors model",
            )
        ),
        "--revision",
        str(
            value(
                "revision",
                (
                    "synthetic fixture"
                    if args.demo
                    else "local selection; revision not asserted"
                ),
            )
        ),
    ]
    if "resources" in config:
        command.extend(["--resources", resources])
    print(
        f"Viewer only: http://127.0.0.1:{port} — Ctrl-C to stop. Startup checks may refuse launch.",
        flush=True,
    )
    os.execv(str(BINARY), command)


if __name__ == "__main__":
    raise SystemExit(main())
