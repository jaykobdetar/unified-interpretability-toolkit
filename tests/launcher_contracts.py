"""Launcher preflight: no native execution, server, build, or model loading."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('launch', ROOT / 'tools/launch.py')
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def run_quiet(self, args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return launch.main(args)

    def test_help_needs_no_binary_or_model(self):
        with self.assertRaises(SystemExit) as exc:
            self.run_quiet(['--help'])
        self.assertEqual(exc.exception.code, 0)

    def test_demo_check_does_not_build_or_spawn(self):
        with patch.object(launch.os, 'access', return_value=True), patch.object(launch.subprocess, 'call') as build, patch.object(launch.os, 'execv') as start:
            self.assertEqual(self.run_quiet(['--demo', '--check']), 0)
            build.assert_not_called()
            start.assert_not_called()

    def test_invalid_port_and_missing_binary(self):
        for args in (['--demo', '--port', '0'], ['--demo', '--port', 'oops'], ['--demo', '--port', '65536'], ['--demo', 'other']):
            with self.assertRaises(SystemExit):
                self.run_quiet(args)
        with patch.object(launch.os, 'access', return_value=False), self.assertRaises(SystemExit):
            self.run_quiet(['--demo', '--check'])

    def test_config_relative_paths_and_cli_precedence(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            (base / 'model').mkdir()
            config = base / 'config.json'
            config.write_text(json.dumps({'model': 'model', 'cache': 'cache', 'port': 8111, 'name': 'literal $(text)'}))
            with patch.object(launch.os, 'access', return_value=True), patch.object(launch.os, 'execv') as start, patch.dict(os.environ, {'ATLAS_PORT': '8222'}):
                self.run_quiet(['--config', str(config), '--port', '8333'])
                argv = start.call_args.args[1]
                self.assertEqual(argv[argv.index('--model') + 1], str(base / 'model'))
                self.assertEqual(argv[argv.index('--cache') + 1], str(base / 'cache'))
                self.assertEqual(argv[argv.index('--port') + 1], '8333')
                self.assertEqual(argv[argv.index('--name') + 1], 'literal $(text)')
            config.write_text(json.dumps({'model': 'model', 'cache': 'model/cache'}))
            with self.assertRaises(SystemExit):
                self.run_quiet(['--config', str(config), '--check'])
            config.write_text(json.dumps({'bind': '0.0.0.0'}))
            with self.assertRaises(SystemExit):
                self.run_quiet(['--config', str(config), '--demo', '--check'])

    def test_resource_configuration_is_validated_before_spawn(self):
        with tempfile.TemporaryDirectory() as d:
            config = Path(d) / 'config.json'
            config.write_text(json.dumps({'resources': {'cpu_count': 4, 'tile_cache_files': 8000}}))
            with patch.object(launch.os, 'access', return_value=True), patch.object(launch.os, 'execv') as start:
                self.run_quiet(['--demo', '--config', str(config)])
                argv = start.call_args.args[1]
                resources = json.loads(argv[argv.index('--resources') + 1])
                self.assertEqual(resources['cpu_count'], 4)
                self.assertEqual(resources['disk_reserve_bytes'], 25*1024**3)
            for raw in ['{"resources":{"disk_reserve_bytes":0}}',
                        '{"resources":{"cpu_count":true}}',
                        '{"resources":{"cpu_count":1,"cpu_count":2}}',
                        '{"port":8775,"port":8776}']:
                config.write_text(raw)
                with patch.object(launch.os, 'execv') as start, self.assertRaises(SystemExit):
                    self.run_quiet(['--demo', '--config', str(config), '--check'])
                start.assert_not_called()


if __name__ == '__main__':
    unittest.main()
