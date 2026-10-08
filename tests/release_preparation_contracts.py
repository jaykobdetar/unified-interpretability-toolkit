"""Release compilation preparation cannot replace or bypass acceptance."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import prepare_release_tests as preparation


def metadata():
    return {
        "workspace_members": ["root"],
        "packages": [
            {
                "id": "root",
                "name": "sample",
                "targets": [
                    {"name": "z_test", "kind": ["test"], "test": True},
                    {"name": "library", "kind": ["lib"], "test": True},
                    {"name": "app", "kind": ["bin"], "test": True},
                    {"name": "disabled", "kind": ["test"], "test": False},
                    {"name": "build", "kind": ["custom-build"], "test": False},
                ],
            },
            {
                "id": "dependency",
                "name": "dependency",
                "targets": [{"name": "foreign", "kind": ["test"], "test": True}],
            },
        ],
    }


class ReleasePreparation(unittest.TestCase):
    def test_discovers_targets_with_exact_guard_and_deadline(self):
        prefix = [
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
            "sample",
        ]
        self.assertEqual(
            preparation.target_commands(metadata()),
            [
                prefix + ["--bin", "app"],
                prefix + ["--lib"],
                prefix + ["--test", "z_test"],
            ],
        )

    def test_runs_serially_and_reports_compilation_only(self):
        with (
            patch.object(
                preparation.subprocess,
                "check_output",
                return_value=json.dumps(metadata()),
            ) as discover,
            patch.object(preparation.subprocess, "run") as run,
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            preparation.main()
        discover.assert_called_once_with(
            [
                "cargo",
                "metadata",
                "--offline",
                "--locked",
                "--no-deps",
                "--format-version",
                "1",
            ],
            cwd=preparation.ROOT,
            text=True,
            timeout=30,
        )
        self.assertEqual(
            [call.args[0] for call in run.call_args_list],
            preparation.target_commands(metadata()),
        )
        self.assertTrue(
            all(
                call.kwargs == {"cwd": preparation.ROOT, "check": True}
                for call in run.call_args_list
            )
        )
        self.assertIn(
            "full checks and test execution still required", output.getvalue()
        )

    def test_failure_or_timeout_stops_without_success_or_later_targets(self):
        for status in (1, 124, 137):
            with (
                self.subTest(status=status),
                patch.object(
                    preparation.subprocess,
                    "check_output",
                    return_value=json.dumps(metadata()),
                ),
                patch.object(
                    preparation.subprocess,
                    "run",
                    side_effect=[
                        None,
                        subprocess.CalledProcessError(status, "compile"),
                    ],
                ) as run,
                contextlib.redirect_stdout(io.StringIO()) as output,
                self.assertRaises(subprocess.CalledProcessError) as failure,
            ):
                preparation.main()
            self.assertEqual(failure.exception.returncode, status)
            self.assertEqual(run.call_count, 2)
            self.assertNotIn("compilation prepared", output.getvalue())

    def test_failed_discovery_or_empty_targets_do_not_compile(self):
        with (
            patch.object(
                preparation.subprocess,
                "check_output",
                side_effect=subprocess.TimeoutExpired("metadata", 30),
            ),
            patch.object(preparation.subprocess, "run") as run,
            self.assertRaises(subprocess.TimeoutExpired),
        ):
            preparation.main()
        run.assert_not_called()
        with (
            patch.object(
                preparation.subprocess,
                "check_output",
                return_value='{"packages":[],"workspace_members":[]}',
            ),
            patch.object(preparation.subprocess, "run") as run,
            self.assertRaisesRegex(RuntimeError, "No release test targets"),
        ):
            preparation.main()
        run.assert_not_called()

    def test_ci_keeps_job_bound_full_check_and_dependent_referee(self):
        workflow = (
            Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml"
        ).read_text()
        prepare = "timeout-minutes: 10\n        run: python3 -B tools/prepare_release_tests.py"
        full = "run: python3 tools/check.py all"
        self.assertEqual(workflow.count(prepare), 1)
        self.assertEqual(workflow.count(full), 1)
        self.assertLess(workflow.index(prepare), workflow.index(full))
        self.assertEqual(workflow.count("timeout-minutes: 25"), 2)
        self.assertIn("needs: check", workflow)
        self.assertNotIn("continue-on-error", workflow)


if __name__ == "__main__":
    unittest.main()
