"""Exact inert resource/launcher controls before ordinary package relocation."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
import viewer_resources as budgets

spec = importlib.util.spec_from_file_location("launch", ROOT / "tools/launch.py")
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class LaunchResourcePackageTests(unittest.TestCase):
    def command(self, args):
        with (
            patch.object(launch.os, "access", return_value=True),
            patch.object(launch.subprocess, "call") as build,
            patch.object(launch.os, "execv") as start,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            try:
                result = launch.main(args)
            except (Exception, SystemExit) as exc:
                self.fail("valid local preflight must complete: " + repr(exc))
            self.assertIsNone(result)
            build.assert_not_called()
            start.assert_called_once()
            binary, argv = start.call_args.args
            self.assertEqual(binary, str(launch.BINARY))
            self.assertEqual(argv[:2], [str(launch.BINARY), "serve"])
            self.assertEqual(len(argv), len(set(argv[2::2])) * 2 + 2)
            return dict(zip(argv[2::2], argv[3::2], strict=True))

    def test_validated_mapping_identity_and_half_boundary(self):
        explicit = {
            "cpu_count": 4,
            "address_space_bytes": 512 * budgets.MIB,
            "workspace_bytes": 256 * budgets.MIB,
            "tile_cache_files": 8000,
        }
        expected = {**budgets.DEFAULTS, **explicit}
        try:
            actual = budgets.validate(explicit)
        except Exception as exc:
            self.fail("valid half-boundary mapping must complete: " + repr(exc))
        self.assertEqual(actual, expected)
        self.assertIsNot(actual, budgets.DEFAULTS)
        self.assertIsNot(actual, explicit)
        actual["cpu_count"] = 7
        self.assertEqual(explicit["cpu_count"], 4)
        self.assertEqual(budgets.DEFAULTS["cpu_count"], 1)

    def test_exact_canonical_encoding(self):
        selected = {"cpu_count": 4, "tile_cache_files": 8000}
        expected = json.dumps(
            {**budgets.DEFAULTS, **selected}, separators=(",", ":"), sort_keys=True
        )
        self.assertEqual(budgets.encode(selected), expected)
        self.assertEqual(
            budgets.encode(dict(reversed(list(selected.items())))), expected
        )
        self.assertEqual(selected, {"cpu_count": 4, "tile_cache_files": 8000})

    def test_local_unique_fields_and_exact_diagnostic(self):
        pairs = [("port", 8111), ("name", "local fixture"), ("resources", {})]
        self.assertEqual(launch.unique_fields(pairs), dict(pairs))
        self.assertEqual(launch.unique_fields([]), {})
        with self.assertRaises(ValueError) as exc:
            launch.unique_fields([("port", 8111), ("port", 8222)])
        self.assertEqual(str(exc.exception), "duplicate configuration field: port")
        self.assertEqual(pairs[0], ("port", 8111))

    def test_exact_demo_command_and_no_process_work(self):
        with patch.dict(os.environ, {}, clear=True):
            actual = self.command(["--demo"])
        self.assertEqual(
            actual,
            {
                "--model": str(launch.ROOT / "fixtures/tiny-bf16"),
                "--cache": str((launch.ROOT / "cache").resolve()),
                "--port": "8775",
                "--name": "Synthetic BF16 fixture",
                "--revision": "synthetic fixture",
            },
        )

    def test_explicit_environment_config_path_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            config_base = base / "config-dir"
            outside = base / "outside"
            config_base.mkdir()
            outside.mkdir()
            (config_base / "model").mkdir()
            (outside / "cli-model").mkdir()
            config = config_base / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "model": "model",
                        "cache": "config-cache",
                        "port": 8111,
                        "name": "config name",
                        "revision": "config revision",
                    }
                )
            )
            with (
                patch.object(launch.Path, "cwd", return_value=outside),
                patch.dict(os.environ, {}, clear=True),
            ):
                actual = self.command(["--config", str(config)])
                self.assertEqual(actual["--model"], str(config_base / "model"))
                self.assertEqual(actual["--cache"], str(config_base / "config-cache"))
                self.assertEqual(actual["--port"], "8111")
                self.assertEqual(actual["--name"], "config name")
                self.assertEqual(actual["--revision"], "config revision")
                with patch.dict(os.environ, {"ATLAS_PORT": "8222"}):
                    actual = self.command(
                        [
                            "cli-model",
                            "--config",
                            str(config),
                            "--cache",
                            "cli-cache",
                            "--port",
                            "8333",
                        ]
                    )
                self.assertEqual(actual["--model"], str(outside / "cli-model"))
                self.assertEqual(actual["--cache"], str(outside / "cli-cache"))
                self.assertEqual(actual["--port"], "8333")
                self.assertEqual(actual["--name"], "config name")


if __name__ == "__main__":
    unittest.main()
