"""Static checker admission and version calls use inert tool doubles only."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import check


class StaticAdmission(unittest.TestCase):
    def test_missing_each_tool_precedes_versions_and_checks(self):
        python = Path("/synthetic-development-python")
        ruff = Path("/synthetic-ruff")
        eslint = check.ROOT / "dev/node_modules/eslint/bin/eslint.js"
        for missing in (python, ruff, eslint):
            with self.subTest(missing=missing):
                with (
                    patch.object(Path, "is_file", lambda path: path != missing),
                    patch.object(check, "require_version") as versions,
                    patch.object(check, "run") as commands,
                    patch.object(
                        check, "formatter_files", return_value=["web/app.js"]
                    ) as files,
                    self.assertRaises(RuntimeError) as failure,
                ):
                    check.static_checks(python, ruff)
                self.assertEqual(
                    str(failure.exception),
                    "required pinned Ruff, ESLint and mypy development tools are missing; "
                    "use the documented development setup",
                )
                versions.assert_not_called()
                commands.assert_not_called()
                files.assert_not_called()

    def test_exact_versions_precede_all_check_launches(self):
        python = Path("/synthetic-development-python")
        ruff = Path("/synthetic-ruff")
        eslint = check.ROOT / "dev/node_modules/eslint/bin/eslint.js"
        calls = []
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(
                check,
                "require_version",
                side_effect=lambda name, command, expected: calls.append(
                    ("version", name, command, expected)
                ),
            ),
            patch.object(check, "formatter_files", return_value=["web/app.js"]),
            patch.object(
                check, "run", side_effect=lambda command: calls.append(("run", command))
            ),
        ):
            check.static_checks(python, ruff)
        self.assertEqual(
            calls[:3],
            [
                ("version", "Ruff", [str(ruff), "--version"], "ruff 0.16.8"),
                ("version", "ESLint", ["node", str(eslint), "--version"], "v9.15.0"),
                (
                    "version",
                    "mypy",
                    [
                        str(python),
                        "-c",
                        "import mypy.version; print(mypy.version.__version__)",
                    ],
                    "1.18.2",
                ),
            ],
        )
        self.assertEqual(len(calls), 6)
        self.assertEqual([kind for kind, *_ in calls[3:]], ["run", "run", "run"])
        self.assertEqual(
            [command[0] for _, command in calls[3:]], [str(ruff), "node", str(python)]
        )

    def test_empty_javascript_stops_after_ruff(self):
        python = Path("/synthetic-development-python")
        ruff = Path("/synthetic-ruff")
        commands = []
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(check, "require_version") as versions,
            patch.object(
                check,
                "formatter_files",
                return_value=["web/style.css", "tests/example.py"],
            ),
            patch.object(check, "run", side_effect=commands.append),
            self.assertRaises(RuntimeError) as failure,
        ):
            check.static_checks(python, ruff)
        self.assertEqual(str(failure.exception), "ESLint scope is unexpectedly empty")
        self.assertEqual(versions.call_count, 3)
        self.assertEqual(
            commands, [[str(ruff), "check", "--no-cache", "tools", "tests"]]
        )


if __name__ == "__main__":
    unittest.main()
