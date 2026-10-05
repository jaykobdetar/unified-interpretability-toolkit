"""Development runner runtime bounds and equivalent archive formatter scope."""

import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import check


class DevelopmentChecks(unittest.TestCase):
    def test_runtime_versions_accept_minimum_and_newer_refuse_older_or_unknown(self):
        cases = [
            ("3.12.0", "3.12.0", True),
            ("3.14.7", "3.12.0", True),
            ("3.11.14", "3.12.0", False),
            ("v22.0.0", "22.0.0", True),
            ("v24.1.2", "22.0.0", True),
            ("v21.99.99", "22.0.0", False),
            ("rustc 1.92.0 (release)", "1.92.0", True),
            ("cargo 1.93.1 (release)", "1.92.0", True),
            ("cargo 1.91.9 (release)", "1.92.0", False),
            ("unrecognised version", "3.12.0", False),
        ]
        for actual, minimum, accepted in cases:
            with (
                self.subTest(actual=actual),
                patch.object(check.subprocess, "check_output", return_value=actual),
            ):
                if accepted:
                    check.require_minimum_version("runtime", ["unused"], minimum)
                else:
                    with self.assertRaisesRegex(RuntimeError, "minimum"):
                        check.require_minimum_version("runtime", ["unused"], minimum)

    def test_formatter_pins_remain_exact(self):
        with patch.object(check.subprocess, "check_output", return_value="26.1.0\n"):
            check.require_version("Black", ["unused"], "26.1.0")
            with self.assertRaisesRegex(RuntimeError, "required"):
                check.require_version("Black", ["unused"], "26.1.1")

    def test_archive_scope_matches_git_scope_and_never_calls_git_without_dotgit(self):
        files = [
            "tools/check.py",
            "tools/behaviour_lock.py",
            "tests/contract.py",
            "tests/analytics/contract.mjs",
            "web/app.js",
            "web/style.css",
            "web/vendor/library.js",
            "tests/__pycache__/ignored.py",
            "tools/node_modules/ignored.js",
        ]
        expected = sorted(files[:6])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in files:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("")
            with (
                patch.object(check, "ROOT", root),
                patch.object(
                    check.subprocess,
                    "check_output",
                    side_effect=AssertionError("No Git"),
                ),
            ):
                self.assertEqual(check.formatter_files(), expected)
            (root / ".git").mkdir()
            with (
                patch.object(check, "ROOT", root),
                patch.object(
                    check.subprocess,
                    "check_output",
                    return_value=("\0".join(files) + "\0").encode(),
                ) as git,
            ):
                self.assertEqual(check.formatter_files(), expected)
                git.assert_called_once()


if __name__ == "__main__":
    unittest.main()
