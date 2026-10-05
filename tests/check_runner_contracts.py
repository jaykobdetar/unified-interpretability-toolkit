#!/usr/bin/env python3
"""Required check failures must never report success or continue the all suite."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "tools/check.py"
spec = importlib.util.spec_from_file_location("check_runner", path)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)

with tempfile.TemporaryDirectory() as temporary:
    missing = Path(temporary) / "missing-black-python"
    for suite in ("format", "all"):
        result = subprocess.run(
            [sys.executable, "-B", str(path), suite, "--dev-python", str(missing)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 1, result
        assert "required Black interpreter is missing" in result.stderr
        assert (
            "+ " not in result.stdout
        ), "missing required tool must stop before launching checks"
    selected = Path(temporary) / "venv-python"
    selected.symlink_to(sys.executable)
    original_argv, original_format = sys.argv, check.format_checks
    captured = []
    try:
        sys.argv = [str(path), "format", "--dev-python", str(selected)]
        check.format_checks = captured.append
        check.main()
    finally:
        sys.argv, check.format_checks = original_argv, original_format
    assert (
        captured == [selected.absolute()] and captured[0].is_symlink()
    ), "the selected virtual-environment launcher must retain its path"
    try:
        check.require_version(
            "synthetic tool", [sys.executable, "-c", "print('wrong')"], "pinned"
        )
    except RuntimeError as error:
        assert "required 'pinned'" in str(error)
    else:
        raise AssertionError("a wrong tool version was accepted")
    try:
        check.run([sys.executable, "-c", "raise SystemExit(7)"])
    except subprocess.CalledProcessError as error:
        assert error.returncode == 7
    else:
        raise AssertionError("a failing required check was accepted")
print("PASS: missing tools, incorrect pins and child failure remain fatal")
