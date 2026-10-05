#!/usr/bin/env python3
"""Portable CI checks; no weights, browser, GPU, or numerical Python packages."""

import argparse
import ast
import os
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(command, *, extra_env=None, timeout=120):
    print("+ " + " ".join(map(str, command)), flush=True)
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join([str(ROOT / "tools"), str(ROOT / "tests")]),
    }
    env.update(extra_env or {})
    subprocess.run(command, cwd=ROOT, env=env, check=True, timeout=timeout)


def lint():
    for base in ("tools", "tests"):
        for file in sorted((ROOT / base).rglob("*.py")):
            ast.parse(file.read_text(), filename=str(file.relative_to(ROOT)))
    for base in ("web", "tests"):
        for file in sorted((ROOT / base).rglob("*")):
            if file.suffix in (".js", ".cjs", ".mjs") and "vendor" not in file.parts:
                run(["node", "--check", str(file.relative_to(ROOT))])
    run(["bash", "-n", "run-atlas.sh"])


def contracts():
    python = sorted((ROOT / "tests").glob("*contracts.py"))
    python += [
        ROOT / "tests" / name
        for name in (
            "host_profile_adapter.py",
            "ui_polish_readiness.py",
            "profile_worker_primitives.py",
            "profile_review_regressions.py",
            "profile_host_supervisor.py",
            "profile_runtime_doubles.py",
            "profile_review_fixes.py",
            "profile_platform_preflight.py",
            "profile_http_integration.py",
            "profile_http_ownership.py",
            "profile_internal_diagnostics.py",
            "profile_lifetime_policy.py",
            "profile_spawn_registration.py",
            "profile_startup_diagnostics.py",
            "analytics/test_core.py",
            "analytics/test_service.py",
            "analytics/test_svd_summary.py",
        )
    ]
    javascript = sorted((ROOT / "tests").glob("ui-*.cjs"))
    javascript = [p for p in javascript if "browser" not in p.name]
    javascript += sorted((ROOT / "tests").glob("profile_*.js"))
    javascript += [
        ROOT / "tests" / name
        for name in (
            "acceptance-support.cjs",
            "acceptance-pin-buffer.cjs",
            "data-view-contracts.cjs",
            "host-client.cjs",
            "inference-availability.cjs",
            "experiment_logs.cjs",
            "experiment-import.cjs",
            "durable-log.cjs",
            "workspace-tools.cjs",
            "region-export.cjs",
            "workspace-ui.cjs",
            "analytics/mount_contract.mjs",
            "analytics/ui_contract.mjs",
        )
    ]
    for file in python:
        run([sys.executable, "-B", str(file.relative_to(ROOT))])
    variants = {
        "ui-combined-context.cjs": ["slice-change", "model-refresh", "host-reopen"],
        "ui-pending-inspection-ownership.cjs": [
            "restore",
            "prior-pin",
            "newer-complete",
            "newer-pending",
            "cancel-before",
            "cancel-opening",
            "tensor-change",
            "binding-mismatch",
        ],
        "ui-readiness-ordering.cjs": ["busy", "failure", "finally", "session"],
        "ui-recolor-selection.cjs": ["successful", "pending"],
        "ui-head-selection-calibration.cjs": ["", "pending-first-read"],
    }
    for file in javascript:
        for variant in variants.get(file.name, [""]):
            run(["node", str(file.relative_to(ROOT)), *([variant] if variant else [])])
    print(f"PASS: {len(python)} Python and {len(javascript)} Node contract files")


def require_version(name, command, expected):
    actual = subprocess.check_output(command, cwd=ROOT, text=True, timeout=30).strip()
    if actual != expected:
        raise RuntimeError(f"{name} version {actual!r}; required {expected!r}")


def format_checks(dev_python):
    pins = json.loads((ROOT / "dev/versions.json").read_text())
    if not dev_python.is_file():
        raise RuntimeError(
            "required Black interpreter is missing; install dev/requirements-format.txt in the documented local environment"
        )
    prettier = ROOT / "dev/node_modules/prettier/bin/prettier.cjs"
    if not prettier.is_file():
        raise RuntimeError(
            "required Prettier is missing; run the documented npm ci command"
        )
    commands = [
        (
            "Python",
            [sys.executable, "-c", "import platform; print(platform.python_version())"],
            pins["python"],
        ),
        ("Node", ["node", "--version"], "v" + pins["node"]),
        (
            "Black",
            [str(dev_python), "-c", "import black; print(black.__version__)"],
            pins["black"],
        ),
        ("Prettier", ["node", str(prettier), "--version"], pins["prettier"]),
    ]
    for name, command, expected in commands:
        require_version(name, command, expected)
    for name in ("rustc", "cargo", "rustfmt"):
        actual = subprocess.check_output(
            [name, "--version"], cwd=ROOT, text=True, timeout=30
        ).strip()
        if not actual.startswith(f"{name} {pins[name]} "):
            raise RuntimeError(f"{name} version {actual!r}; required {pins[name]!r}")
    tracked = (
        subprocess.check_output(
            ["git", "ls-files", "-z", "tools", "tests", "web"], cwd=ROOT
        )
        .decode()
        .split("\0")
    )
    python = [
        p for p in tracked if p.endswith(".py") and p != "tools/behaviour_lock.py"
    ]
    javascript = [
        p
        for p in tracked
        if Path(p).suffix in (".js", ".cjs", ".mjs", ".css")
        and "vendor" not in Path(p).parts
    ]
    if not python or not javascript:
        raise RuntimeError("formatter scope is unexpectedly empty")
    run([str(dev_python), "-m", "black", "--workers", "1", "--check", *python])
    run(["node", str(prettier), "--check", *javascript])
    run(["cargo", "fmt", "--all", "--check"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=["lint", "contracts", "format", "all"])
    parser.add_argument(
        "--dev-python",
        type=Path,
        default=ROOT / "qualification/dev/black-26.1.0/bin/python",
        help="interpreter with the pinned optional Black development tool",
    )
    args = parser.parse_args()
    if args.suite in ("format", "all"):
        format_checks(args.dev_python.absolute())
    if args.suite in ("lint", "all"):
        lint()
    if args.suite in ("contracts", "all"):
        contracts()
    if args.suite != "all":
        return
    run([sys.executable, "-B", "tests/dense_static_review_regressions.py"])
    guard = [sys.executable, "-B", "tools/guarded-build.py"]
    run([*guard, "clippy", "--all-targets", "--", "-D", "warnings"], timeout=140)
    run([*guard, "test"], extra_env={"RUST_TEST_THREADS": "4"}, timeout=140)
    run(
        [*guard, "test", "--release"], extra_env={"RUST_TEST_THREADS": "4"}, timeout=140
    )
    run([*guard, "build", "--release"], timeout=140)
    run([sys.executable, "-B", "tools/smoke.py"])
    run(["./run-atlas.sh", "--demo", "--check"])
    print("PASS: unified pinned development and CI checks")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"CHECK FAILED: {error}", file=sys.stderr)
        sys.exit(1)
