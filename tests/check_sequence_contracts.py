"""Required checker order, exact lint rules and the hosted CI entry point."""

from contextlib import ExitStack
import io
from pathlib import Path
import re
import sys
import tomllib
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import check

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "tools/check.py",
    "tools/behaviour_lock.py",
    "tests/example.py",
    "web/app.js",
    "tests/example.cjs",
    "web/style.css",
]


class Sequence(unittest.TestCase):
    def test_all_stages_and_commands_run_in_exact_order(self):
        calls = []
        python = Path("/synthetic-development-python")
        ruff = Path("/synthetic-ruff")
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    sys,
                    "argv",
                    [
                        "check.py",
                        "all",
                        "--dev-python",
                        str(python),
                        "--ruff",
                        str(ruff),
                    ],
                )
            )
            for name in ("format_checks", "static_checks", "lint", "contracts"):
                stack.enter_context(
                    patch.object(
                        check,
                        name,
                        side_effect=lambda *args, name=name: calls.append(
                            (name, tuple(map(str, args)))
                        ),
                    )
                )
            stack.enter_context(
                patch.object(
                    check,
                    "run",
                    side_effect=lambda command, **options: calls.append(
                        (list(map(str, command)), options)
                    ),
                )
            )
            stack.enter_context(patch.object(sys, "stdout", io.StringIO()))
            check.main()
        executable = sys.executable
        guard = [executable, "-B", "tools/guarded-build.py"]
        self.assertEqual(
            calls,
            [
                ("format_checks", (str(python),)),
                ("static_checks", (str(python), str(ruff))),
                ("lint", ()),
                ("contracts", ()),
                ([executable, "-B", "tests/dense_static_review_regressions.py"], {}),
                (
                    [*guard, "clippy", "--all-targets", "--", "-D", "warnings"],
                    {"timeout": 140},
                ),
                (
                    [*guard, "test"],
                    {"extra_env": {"RUST_TEST_THREADS": "4"}, "timeout": 140},
                ),
                ([*guard, "build", "--release"], {"timeout": 140}),
                (
                    [*guard, "test", "--release"],
                    {"extra_env": {"RUST_TEST_THREADS": "4"}, "timeout": 140},
                ),
                ([executable, "-B", "tools/smoke.py"], {}),
                (["./run-atlas.sh", "--demo", "--check"], {}),
            ],
        )

    def formatter_context(self, stack, rustfmt="rustfmt 1.8.0-stable (synthetic)"):
        versions, commands = [], []
        stack.enter_context(patch.object(Path, "is_file", return_value=True))
        stack.enter_context(
            patch.object(
                check,
                "require_minimum_version",
                side_effect=lambda name, command, minimum: versions.append(
                    (name, list(map(str, command)), minimum)
                ),
            )
        )
        stack.enter_context(patch.object(check, "require_version"))
        stack.enter_context(
            patch.object(check.subprocess, "check_output", return_value=rustfmt)
        )
        stack.enter_context(patch.object(check, "formatter_files", return_value=FILES))
        stack.enter_context(
            patch.object(
                check,
                "run",
                side_effect=lambda command: commands.append(list(map(str, command))),
            )
        )
        return versions, commands

    def test_format_commands_are_read_only_and_runtime_minimums_are_independent(self):
        python = Path("/synthetic-development-python")
        with ExitStack() as stack:
            versions, commands = self.formatter_context(stack)
            check.format_checks(python)
        self.assertEqual(
            versions,
            [
                (
                    "Python",
                    [
                        sys.executable,
                        "-c",
                        "import platform; print(platform.python_version())",
                    ],
                    "3.12.0",
                ),
                ("Node", ["node", "--version"], "22.0.0"),
                ("rustc", ["rustc", "--version"], "1.92.0"),
                ("cargo", ["cargo", "--version"], "1.92.0"),
            ],
        )
        self.assertEqual(
            commands,
            [
                [
                    str(python),
                    "-m",
                    "black",
                    "--workers",
                    "1",
                    "--check",
                    "tools/check.py",
                    "tests/example.py",
                ],
                [
                    "node",
                    str(ROOT / "dev/node_modules/prettier/bin/prettier.cjs"),
                    "--check",
                    "web/app.js",
                    "tests/example.cjs",
                    "web/style.css",
                ],
                ["cargo", "fmt", "--all", "--check"],
            ],
        )

    def test_rustfmt_pin_refuses_adjacent_and_malformed_versions(self):
        for version in (
            "rustfmt 1.7.0-stable (synthetic)",
            "rustfmt 1.9.0-stable (synthetic)",
            "rustfmt 1.8.0-stable-extra",
        ):
            with self.subTest(version=version), ExitStack() as stack:
                self.formatter_context(stack, version)
                with self.assertRaisesRegex(RuntimeError, "required"):
                    check.format_checks(Path("/synthetic-development-python"))

    def test_static_commands_preserve_exact_scopes_and_strict_settings(self):
        commands = []
        python, ruff = Path("/synthetic-development-python"), Path("/synthetic-ruff")
        with (
            patch.object(check, "_MYPY_FILES", ("tools/check.py",), create=True),
            patch.object(Path, "is_file", return_value=True),
            patch.object(check, "require_version"),
            patch.object(check, "formatter_files", return_value=FILES),
            patch.object(
                check,
                "run",
                side_effect=lambda command: commands.append(list(map(str, command))),
            ),
        ):
            check.static_checks(python, ruff)
        self.assertEqual(
            commands,
            [
                [str(ruff), "check", "--no-cache", "tools", "tests"],
                [
                    "node",
                    str(ROOT / "dev/node_modules/eslint/bin/eslint.js"),
                    "--no-config-lookup",
                    "--config",
                    "dev/eslint.config.cjs",
                    "--max-warnings",
                    "0",
                    "web/app.js",
                    "tests/example.cjs",
                ],
                [
                    str(python),
                    "-m",
                    "mypy",
                    "--config-file",
                    "dev/mypy.ini",
                    "--cache-dir",
                    "/dev/null",
                    "tools/check.py",
                ],
            ],
        )

    def test_lint_rule_lists_and_documented_oracle_exceptions_are_exact(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertEqual(
            project["tool"]["ruff"]["lint"]["select"], ["E9", "F63", "F7", "F82"]
        )
        self.assertEqual(
            project["tool"]["ruff"]["lint"]["per-file-ignores"],
            {
                "tests/observation_reference.py": ["F821"],
                "tests/prompt_pair_reference.py": ["F821"],
            },
        )
        eslint = (ROOT / "dev/eslint.config.cjs").read_text()
        rule_list = re.search(r"const\s+rules\s*=\s*\[(.*?)\];", eslint, re.S)
        self.assertIsNotNone(rule_list)
        self.assertEqual(
            re.findall(r'"([a-z-]+)"', rule_list[1]),
            [
                "constructor-super",
                "for-direction",
                "getter-return",
                "no-async-promise-executor",
                "no-compare-neg-zero",
                "no-cond-assign",
                "no-const-assign",
                "no-dupe-args",
                "no-dupe-class-members",
                "no-dupe-else-if",
                "no-dupe-keys",
                "no-duplicate-case",
                "no-empty-character-class",
                "no-ex-assign",
                "no-func-assign",
                "no-import-assign",
                "no-invalid-regexp",
                "no-irregular-whitespace",
                "no-loss-of-precision",
                "no-obj-calls",
                "no-setter-return",
                "no-this-before-super",
                "no-unexpected-multiline",
                "no-unreachable",
                "no-unsafe-finally",
                "no-unsafe-negation",
                "use-isnan",
                "valid-typeof",
            ],
        )
        self.assertIn('rules.map((name) => [name, "error"])', eslint)
        mypy = (ROOT / "dev/mypy.ini").read_text()
        self.assertRegex(mypy, r"(?m)^strict\s*=\s*True$")

    def test_hosted_ci_uses_full_suite_and_target_branch_referee(self):
        workflow = (ROOT / ".github/workflows/ci.yml").read_text()
        self.assertEqual(
            re.findall(r"(?m)^\s+run:\s*(python3 tools/check\.py \S+)\s*$", workflow),
            ["python3 tools/check.py all"],
        )
        self.assertIn("ref: ${{ github.event.pull_request.base.sha }}", workflow)
        self.assertIn("ref: ${{ github.event.pull_request.head.sha }}", workflow)
        self.assertEqual(
            workflow.count("python3 -B referee-base/tools/behaviour_lock.py"), 2
        )
        self.assertIn(
            "--old-binary referee-base/target/release/weight-atlas-rust", workflow
        )
        self.assertNotIn("python3 -B referee-pr/tools/behaviour_lock.py", workflow)
        self.assertNotIn("--allow", workflow)


if __name__ == "__main__":
    unittest.main()
