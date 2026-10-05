"""Portable integrity/precision checks for the opt-in held CPU protocol net."""

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import cpu_protocol_reference as reference


class Contracts(unittest.TestCase):
    def setUp(self):
        self.data = reference.manifest()

    def raw(self, name):
        case = self.data["cases"][name]
        events = [
            json.loads(line)
            for line in (reference.FIXTURES / case["golden"]).read_text().splitlines()
        ]
        for event, paths in zip(events, case["clock_paths_by_event"]):
            for path in paths:
                owner = event
                keys = path.split(".")
                for key in keys[:-1]:
                    owner = owner[key]
                owner[keys[-1]] = 0.0
        return events

    def test_all_ten_held_application_protocols_match(self):
        for name in self.data["cases"]:
            self.assertEqual(
                reference.compare(reference.encoding(self.raw(name)), name),
                self.data["cases"][name]["golden_sha256"],
            )

    def test_only_clock_values_can_vary_and_presence_is_required(self):
        events = self.raw("block")
        events[0]["load_ms"] = 123.45
        reference.compare(reference.encoding(events), "block")
        for value in (-1, True, "0", None, float("inf")):
            invalid = copy.deepcopy(events)
            invalid[0]["load_ms"] = value
            with self.assertRaises((AssertionError, ValueError)):
                reference.compare(reference.encoding(invalid), "block")
        del events[0]["load_ms"]
        with self.assertRaises(KeyError):
            reference.compare(reference.encoding(events), "block")

    def test_runtime_source_numeric_event_and_key_identity_are_exact(self):
        for change in (
            "runtime",
            "source",
            "numeric",
            "extra",
            "event_order",
            "key_order",
        ):
            events = self.raw("block")
            if change == "runtime":
                events[0]["runtime"]["seed"] = 1
            elif change == "source":
                events[0]["head_layout"]["source_model"]["revision"] = "changed"
            elif change == "numeric":
                events[2]["activation"][0] += 1e-12
            elif change == "extra":
                events[0]["unexpected_elapsed"] = 0
            elif change == "event_order":
                events[2], events[3] = events[3], events[2]
            else:
                events[0] = dict(reversed(list(events[0].items())))
            with self.assertRaises((AssertionError, KeyError)):
                reference.compare(reference.encoding(events), "block")

    def test_signed_zero_and_numeric_spelling_are_preserved(self):
        self.assertNotEqual(
            reference.encoding([{"value": -0.0}]), reference.encoding([{"value": 0.0}])
        )
        self.assertNotEqual(
            reference.encoding([{"value": 0.0}]), reference.encoding([{"value": 0}])
        )

    def test_duplicate_fields_and_wire_whitespace_are_rejected(self):
        raw = reference.encoding(self.raw("block"))
        with self.assertRaises(AssertionError):
            reference.compare(
                raw.replace('"type":"loaded"', '"type":"loaded","type":"loaded"', 1),
                "block",
            )
        with self.assertRaises(AssertionError):
            reference.compare(raw.replace('"type":', '"type": ', 1), "block")

    def test_outer_interruption_reuses_owned_cleanup_and_preserves_receipt(self):
        # Actual disposable dummy guard/worker only; no model, listener or ML.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = root / "tools"
            tools.mkdir()
            shutil.copyfile(
                reference.ROOT / "tools/guarded-core-ui.py",
                tools / "guarded-core-ui.py",
            )
            ready = root / "ready"
            (tools / "guarded-inference-test.py").write_text(
                "import subprocess,sys\np=subprocess.Popen(sys.argv[1:],stdin=sys.stdin);p.wait()\n"
            )
            (tools / "inference_worker.py").write_text(
                "import os,time\nfrom pathlib import Path\nPath("
                + repr(str(ready))
                + ").write_text(str(os.getpid()))\ntime.sleep(30)\n"
            )
            actual = subprocess.Popen

            class Interrupted(actual):
                def communicate(self, input=None, timeout=None):
                    self.stdin.write(input)
                    self.stdin.flush()
                    deadline = time.monotonic() + 2
                    while not ready.exists():
                        assert time.monotonic() < deadline, "Dummy worker did not start"
                        time.sleep(0.01)
                    raise KeyboardInterrupt("Declared local interruption")

            output = root / "evidence"
            output.mkdir()
            with (
                patch.object(reference, "ROOT", root),
                patch.object(subprocess, "Popen", Interrupted),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    reference.execute_case(
                        "dummy", {"request": {}}, Path(sys.executable), root, output
                    )
            receipt = json.loads((output / "lifecycle.json").read_text())
            self.assertTrue(receipt["interrupted"])
            self.assertTrue(receipt["guard_reaped"])
            self.assertTrue(receipt["cleanup_verified"])
            self.assertEqual(receipt["remaining_owned_pids"], [])
            self.assertEqual(receipt["cleanup_errors"], [])


if __name__ == "__main__":
    unittest.main()
