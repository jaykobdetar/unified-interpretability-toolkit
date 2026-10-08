#!/usr/bin/env python3
"""Compile release test targets serially; the full checker must still run."""

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def target_commands(metadata: dict[str, Any]) -> list[list[str]]:
    commands = []
    for package in sorted(metadata["packages"], key=lambda value: value["name"]):
        if package["id"] not in metadata["workspace_members"]:
            continue
        for target in sorted(package["targets"], key=lambda value: value["name"]):
            if not target["test"]:
                continue
            kind = target["kind"]
            if kind == ["lib"]:
                selector = ["--lib"]
            elif kind in (["bin"], ["test"]):
                selector = ["--" + kind[0], target["name"]]
            else:
                continue
            commands.append(
                [
                    "timeout",
                    "--signal=TERM",
                    "--kill-after=5s",
                    "140s",
                    sys.executable,
                    "-B",
                    "tools/guarded-build.py",
                    "test",
                    "--release",
                    "--no-run",
                    "--package",
                    package["name"],
                    *selector,
                ]
            )
    return commands


def main() -> None:
    metadata = json.loads(
        subprocess.check_output(
            [
                "cargo",
                "metadata",
                "--offline",
                "--locked",
                "--no-deps",
                "--format-version",
                "1",
            ],
            cwd=ROOT,
            text=True,
            timeout=30,
        )
    )
    commands = target_commands(metadata)
    if not commands:
        raise RuntimeError("No release test targets found")
    for command in commands:
        print("+ " + " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    print(
        "Release test compilation prepared; full checks and test execution still required."
    )


if __name__ == "__main__":
    main()
