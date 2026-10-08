"""Nominal operation identity and legacy addresses survive leaf extraction."""

import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import unittest

from atlas_host import hosted_runtime, runtime_adapter, static_operation


class StaticOperationLeafContracts(unittest.TestCase):
    def test_old_and_new_exports_are_the_same_nominal_class(self):
        self.assertIs(hosted_runtime.StaticOperation, static_operation.StaticOperation)
        self.assertIs(runtime_adapter.StaticOperation, static_operation.StaticOperation)

    def test_class_keeps_legacy_address_and_reference_round_trip(self):
        cls = static_operation.StaticOperation
        self.assertEqual(cls.__module__, "atlas_host.hosted_runtime")
        self.assertEqual(cls.__qualname__, "StaticOperation")
        self.assertIs(pickle.loads(pickle.dumps(cls)), cls)

    def test_all_original_methods_keep_legacy_addresses(self):
        for name in [
            "__init__",
            "add_child",
            "add_reader",
            "_observe",
            "check",
            "check_publication",
            "_fail",
            "pulse",
            "_reap",
            "abort",
            "publish",
        ]:
            with self.subTest(method=name):
                method = getattr(static_operation.StaticOperation, name)
                self.assertEqual(method.__module__, "atlas_host.hosted_runtime")
                self.assertEqual(method.__qualname__, "StaticOperation." + name)
                self.assertIs(pickle.loads(pickle.dumps(method)), method)

    def test_cold_leaf_does_not_load_runtime_adapter_or_host(self):
        forbidden = [
            "host_atlas",
            "atlas_host.runtime_adapter",
            "atlas_host.hosted_runtime",
            "http.server",
            "torch",
        ]
        root = Path(__file__).resolve().parents[1]
        code = (
            "import json,sys; import atlas_host.static_operation; print(json.dumps([name for name in "
            + repr(forbidden)
            + " if name in sys.modules]))"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=root,
            env={
                **os.environ,
                "PYTHONPATH": str(root / "tools"),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [])


if __name__ == "__main__":
    unittest.main()
