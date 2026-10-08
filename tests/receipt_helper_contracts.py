"""Small local receipt/JSON correctness controls; no server or model work."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def manifest():
    return {
        "version": 1,
        "repository": "fixtures/receipt",
        "revision": "a" * 40,
        "provenance": "owner_expected",
        "license": {"id": "Apache-2.0", "accepted": True, "file": "LICENSE"},
        "files": [
            {"name": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            for name, raw in (
                ("model.safetensors", b"small local receipt fixture"),
                ("config.json", b"{}"),
                ("LICENSE", b"fixture license"),
            )
        ],
    }


class ReceiptHelperTests(unittest.TestCase):
    def setUp(self):
        try:
            from atlas_host import common, registry
        except Exception as exc:
            self.fail("pure receipt helpers must initialize: " + repr(exc))
        self.common = common
        self.registry = registry

    def valid(self, function, *args):
        try:
            return function(*args)
        except Exception as exc:
            self.fail("valid local fixture must complete: " + repr(exc))

    def error(self, function, message, *args):
        try:
            function(*args)
        except ValueError as exc:
            self.assertEqual(str(exc), message)
        except Exception as exc:
            self.fail("expected local ValueError: " + repr(exc))
        else:
            self.fail("expected local ValueError: " + message)

    def test_require(self):
        self.assertIsNone(self.valid(self.common.require, True, "local condition"))
        self.error(self.common.require, "local condition", False, "local condition")

    def test_fields(self):
        value = {"name": "fixture", "other": 1}
        self.assertIsNone(self.valid(self.common.fields, value, ("name",), ("other",)))
        self.assertEqual(value, {"name": "fixture", "other": 1})
        self.error(
            self.common.fields,
            "Missing or unknown fields",
            {"other": 1},
            ("name",),
            ("other",),
        )

    def test_integer(self):
        self.assertEqual(self.valid(self.common.integer, 0), 0)
        self.assertEqual(self.valid(self.common.integer, 99, 0, 99), 99)
        self.error(self.common.integer, "Integer outside allowed range", True)

    def test_label(self):
        for value in (" with spaces ", "x" * 128):
            self.assertEqual(self.valid(self.common.label, value), value)
        self.error(self.common.label, "Invalid label", "")

    def test_digest(self):
        value = "ab" * 32
        self.assertEqual(self.valid(self.common.digest, value), value)
        self.error(self.common.digest, "Expected lowercase SHA-256 digest", "X" * 64)

    def test_canonical(self):
        value = {"é": [1, None], "a": " value "}
        self.assertEqual(
            self.valid(self.common.canonical, value),
            '{"a":" value ","é":[1,null]}'.encode("utf-8"),
        )
        with self.assertRaises(ValueError):
            self.common.canonical({"x": float("nan")})

    def test_identity(self):
        raw = '["fixture-domain",{"é":1}]'.encode("utf-8")
        expected = hashlib.sha256(raw).hexdigest()
        self.assertEqual(
            self.valid(self.common.identity, "fixture-domain", {"é": 1}), expected
        )

    def test_unique(self):
        item = [1, 2]
        actual = self.valid(self.common._unique, [("a", item), ("b", None)])
        self.assertEqual(actual, {"a": [1, 2], "b": None})
        self.assertIs(actual["a"], item)
        self.error(self.common._unique, "Duplicate JSON key", [("a", 1), ("a", 2)])

    def test_read_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.json"
            raw = b'{"a":[1,2],"z":null}'
            path.write_bytes(raw)
            self.assertEqual(
                self.valid(self.common.read_json, path, len(raw)),
                {"a": [1, 2], "z": None},
            )
            self.assertEqual(path.read_bytes(), raw)
            path.write_bytes(b"{}  ")
            self.error(self.common.read_json, "JSON file exceeds byte limit", path, 2)
            path.write_bytes(b'{"a":1,"a":2}')
            self.error(self.common.read_json, "Duplicate JSON key", path, 32)

    def test_manifest(self):
        value = manifest()
        before = deepcopy(value)
        expected = deepcopy(value)
        expected["files"].sort(key=lambda entry: entry["name"])
        actual = self.valid(self.registry.validate_manifest, value)
        self.assertEqual(actual, expected)
        self.assertEqual(value, before)
        self.assertIsNot(actual, value)
        self.assertIsNot(actual["license"], value["license"])
        self.assertIsNot(actual["files"], value["files"])

    def test_content_digest(self):
        value = manifest()
        before = deepcopy(value)
        expected = {
            "version": 1,
            "repository": "fixtures/receipt",
            "revision": "a" * 40,
            "license": {"id": "Apache-2.0", "file": "LICENSE"},
            "files": sorted(deepcopy(value["files"]), key=lambda entry: entry["name"]),
        }
        raw = json.dumps(
            ["weight-atlas-content-v1", expected],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        self.assertEqual(
            self.valid(self.registry.content_digest, value),
            hashlib.sha256(raw).hexdigest(),
        )
        self.assertEqual(value, before)


if __name__ == "__main__":
    unittest.main()
