#!/usr/bin/env python3
"""Portable CI checks; no weights, browser, GPU, or numerical Python packages."""

import argparse
import ast
from collections.abc import Mapping, Sequence
import os
import json
import re
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
_MYPY_FILES: tuple[str, ...] = tuple(
    (ROOT / "dev/mypy-files.txt").read_text().splitlines()
)


def run(
    command: Sequence[str | Path],
    *,
    extra_env: Mapping[str, str] | None = None,
    timeout: int = 120,
) -> None:
    print("+ " + " ".join(map(str, command)), flush=True)
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join([str(ROOT / "tools"), str(ROOT / "tests")]),
    }
    env.update(extra_env or {})
    subprocess.run(command, cwd=ROOT, env=env, check=True, timeout=timeout)


def lint() -> None:
    for base in ("tools", "tests"):
        for file in sorted((ROOT / base).rglob("*.py")):
            ast.parse(file.read_text(), filename=str(file.relative_to(ROOT)))
    for base in ("web", "tests"):
        for file in sorted((ROOT / base).rglob("*")):
            if file.suffix in (".js", ".cjs", ".mjs") and "vendor" not in file.parts:
                run(["node", "--check", str(file.relative_to(ROOT))])
    run(["bash", "-n", "run-atlas.sh"])
    run([sys.executable, "-B", "tools/generate_experiment_schema.py", "--check"])
    run([sys.executable, "-B", "tools/generate_rule_catalog.py", "--check"])


def contracts() -> None:
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


def require_version(name: str, command: Sequence[str], expected: str) -> None:
    actual = subprocess.check_output(command, cwd=ROOT, text=True, timeout=30).strip()
    if actual != expected:
        raise RuntimeError(f"{name} version {actual!r}; required {expected!r}")


def require_minimum_version(name: str, command: Sequence[str], minimum: str) -> None:
    actual = subprocess.check_output(command, cwd=ROOT, text=True, timeout=30).strip()
    version = re.search(r"(?<!\d)(\d+)\.(\d+)\.(\d+)(?!\d)", actual)
    if version is None or tuple(map(int, version.groups())) < tuple(
        map(int, minimum.split("."))
    ):
        raise RuntimeError(f"{name} version {actual!r}; minimum {minimum!r}")


def formatter_files() -> list[str]:
    if (ROOT / ".git").exists():
        files = (
            subprocess.check_output(
                ["git", "ls-files", "-z", "tools", "tests", "web"], cwd=ROOT
            )
            .decode()
            .split("\0")
        )
    else:
        files = [
            str(path.relative_to(ROOT))
            for base in ("tools", "tests", "web")
            for path in (ROOT / base).rglob("*")
            if path.is_file()
        ]
    return sorted(
        path
        for path in files
        if path
        and not {"vendor", "node_modules", "__pycache__"}.intersection(Path(path).parts)
    )


def format_checks(dev_python: Path) -> None:
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
    minimums = pins["runtime_minimums"]
    for name, command, minimum in [
        (
            "Python",
            [sys.executable, "-c", "import platform; print(platform.python_version())"],
            minimums["python"],
        ),
        ("Node", ["node", "--version"], minimums["node"]),
        ("rustc", ["rustc", "--version"], minimums["rustc"]),
        ("cargo", ["cargo", "--version"], minimums["cargo"]),
    ]:
        require_minimum_version(name, command, minimum)
    commands = [
        (
            "Black",
            [str(dev_python), "-c", "import black; print(black.__version__)"],
            pins["black"],
        ),
        ("Prettier", ["node", str(prettier), "--version"], pins["prettier"]),
    ]
    for name, command, expected in commands:
        require_version(name, command, expected)
    for name in ("rustfmt",):
        actual = subprocess.check_output(
            [name, "--version"], cwd=ROOT, text=True, timeout=30
        ).strip()
        if not actual.startswith(f"{name} {pins[name]} "):
            raise RuntimeError(f"{name} version {actual!r}; required {pins[name]!r}")
    tracked = formatter_files()
    python = [
        p for p in tracked if p.endswith(".py") and p != "tools/behaviour_lock.py"
    ]
    javascript = [
        p
        for p in tracked
        if Path(p).suffix in (".js", ".cjs", ".mjs", ".css", ".html")
        and "vendor" not in Path(p).parts
    ]
    if not python or not javascript:
        raise RuntimeError("formatter scope is unexpectedly empty")
    run([str(dev_python), "-m", "black", "--workers", "1", "--check", *python])
    run(["node", str(prettier), "--check", *javascript])
    run(["cargo", "fmt", "--all", "--check"])


def static_checks(dev_python: Path, ruff: Path) -> None:
    pins = json.loads((ROOT / "dev/versions.json").read_text())
    eslint = ROOT / "dev/node_modules/eslint/bin/eslint.js"
    if not dev_python.is_file() or not ruff.is_file() or not eslint.is_file():
        raise RuntimeError(
            "required pinned Ruff, ESLint and mypy development tools are missing; "
            "use the documented development setup"
        )
    for name, command, expected in [
        ("Ruff", [str(ruff), "--version"], f"ruff {pins['ruff']}"),
        ("ESLint", ["node", str(eslint), "--version"], f"v{pins['eslint']}"),
        (
            "mypy",
            [
                str(dev_python),
                "-c",
                "import mypy.version; print(mypy.version.__version__)",
            ],
            pins["mypy"],
        ),
    ]:
        require_version(name, command, expected)
    run([str(ruff), "check", "--no-cache", "tools", "tests"])
    javascript = [
        path
        for path in formatter_files()
        if Path(path).suffix in (".js", ".cjs", ".mjs")
    ]
    if not javascript:
        raise RuntimeError("ESLint scope is unexpectedly empty")
    run(
        [
            "node",
            str(eslint),
            "--no-config-lookup",
            "--config",
            "dev/eslint.config.cjs",
            "--max-warnings",
            "0",
            *javascript,
        ]
    )
    run(
        [
            str(dev_python),
            "-m",
            "mypy",
            "--config-file",
            "dev/mypy.ini",
            "--cache-dir",
            "/dev/null",
            *_MYPY_FILES,
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "suite", choices=["lint", "contracts", "format", "static", "all"]
    )
    parser.add_argument(
        "--dev-python",
        type=Path,
        default=ROOT / "qualification/dev/black-26.1.0/bin/python",
        help="interpreter with the pinned optional Black development tool",
    )
    parser.add_argument(
        "--ruff",
        type=Path,
        help="existing exact Ruff executable; defaults beside --dev-python",
    )
    args = parser.parse_args()
    if args.suite in ("format", "all"):
        format_checks(args.dev_python.absolute())
    if args.suite in ("static", "all"):
        static_checks(
            args.dev_python.absolute(),
            (
                args.ruff.absolute()
                if args.ruff
                else args.dev_python.absolute().parent / "ruff"
            ),
        )
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
    run([*guard, "build", "--release"], timeout=140)
    run(
        [*guard, "test", "--release"], extra_env={"RUST_TEST_THREADS": "4"}, timeout=140
    )
    run([sys.executable, "-B", "tools/smoke.py"])
    run(["./run-atlas.sh", "--demo", "--check"])
    print("PASS: unified pinned development and CI checks")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"CHECK FAILED: {error}", file=sys.stderr)
        sys.exit(1)
