"""Pure supporting-schema edges; these dictionaries are not owner evidence."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import dense_static_contracts as fixture
from atlas_host import dense_static_admission as admission

# Independent, reviewed bounds. Schema traversal finds all callers, while these
# literals determine the accepted/rejected values rather than the implementation.
BOUNDS = {
    ("file", "bytes"): (0, 524288),
    ("fingerprint", "bytes"): (0, 65536),
    ("fingerprint", "ctime_ns"): (-(2**63), 2**63 - 1),
    ("fingerprint", "mtime_ns"): (-(2**63), 2**63 - 1),
    ("fingerprint", "device"): (0, 2**63 - 1),
    ("fingerprint", "inode"): (1, 2**63 - 1),
    ("openIdentity", "bytes_read"): (1, 65536),
    ("pureEvidence", "test_count"): (1, 4096),
    ("root", "device"): (0, 2**63 - 1),
    ("root", "directory_inode"): (1, 2**63 - 1),
    ("sourceRecipe", "native_binary.bytes"): (1, 33554432),
}
LENGTHS = {
    ("file", "path"): (1, 180),
    ("root", "canonical_root"): (1, 4096),
    ("filesystemReceipt", "platform.kernel_release"): (1, 128),
    ("filesystemReceipt", "platform.native_arch"): (1, 64),
    ("filesystemReceipt", "platform.python_version"): (5, 32),
}


def leaves(schema, path=(), definition=None, local=()):
    if "$ref" in schema:
        name = schema["$ref"].split("/")[-1]
        yield from leaves(admission.SCHEMAS[name], path, name, ())
    elif schema.get("type") == "object":
        for key, child in schema["properties"].items():
            yield from leaves(child, path + (key,), definition, local + (key,))
    elif schema.get("type") == "array":
        yield path, definition, ".".join(local), schema
        yield from leaves(schema["items"], path + (0,), definition, local + ("[]",))
    else:
        yield path, definition, ".".join(local), schema


def assign(value, path, new):
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = new


class DenseSchemaLimits(unittest.TestCase):
    def fixtures(self):
        for kind in ("filesystemReceipt", "sourceRecipe"):
            value = fixture.skeleton(admission.SCHEMAS[kind])
            if kind == "filesystemReceipt":
                value["platform"]["python_version"] = "3.12.3"
                for path, definition, key, _ in leaves(
                    admission.SCHEMAS[kind], definition=kind
                ):
                    if (definition, key) == ("root", "canonical_root"):
                        assign(value, path, "/fixture")
            admission.validate_support(value, kind)
            yield kind, value

    def check_edges(self, kind, baseline, path, low, high, make=lambda x: x):
        for bound, accepted in (
            (low - 1, False),
            (low, True),
            (high, True),
            (high + 1, False),
        ):
            with self.subTest(kind=kind, path=path, bound=bound):
                value = deepcopy(baseline)
                assign(value, path, make(bound))
                if accepted:
                    self.assertEqual(admission.validate_support(value, kind), value)
                else:
                    with self.assertRaises(ValueError):
                        admission.validate_support(value, kind)

    def test_every_reachable_integer_definition_and_reference_edge(self):
        found = set()
        for kind, value in self.fixtures():
            for path, definition, key, schema in leaves(
                admission.SCHEMAS[kind], definition=kind
            ):
                if schema.get("type") == "integer":
                    bounds = BOUNDS[(definition, key)]
                    found.add((definition, key))
                    self.assertEqual(
                        (schema.get("minimum", 0), schema.get("maximum", 2**63 - 1)),
                        bounds,
                    )
                    self.check_edges(kind, value, path, *bounds)
                    for wrong in (True, 1.0, "1"):
                        bad = deepcopy(value)
                        assign(bad, path, wrong)
                        with self.assertRaises(ValueError):
                            admission.validate_support(bad, kind)
        self.assertEqual(found, set(BOUNDS))

    def test_every_declared_string_length_and_pattern_edge(self):
        found = set()
        for kind, value in self.fixtures():
            for path, definition, key, schema in leaves(
                admission.SCHEMAS[kind], definition=kind
            ):
                if schema.get("type") != "string":
                    continue
                identity = (definition, key)
                if identity in LENGTHS:
                    low, high = LENGTHS[identity]
                    found.add(identity)
                    self.assertEqual(schema["maxLength"], high)
                    if key == "canonical_root":
                        make = lambda n: "/" + "x" * (n - 1) if n else ""
                    elif key == "platform.python_version":
                        make = lambda n: (
                            "3." + "1" * (n - 4) + ".0" if n >= 5 else "3.0"
                        )
                    else:
                        make = lambda n: "x" * n
                    self.check_edges(kind, value, path, low, high, make)
                else:
                    # A digest's exact encoding masks the default 2**31 string
                    # ceiling: there is no valid complete digest at that ceiling.
                    pattern = schema["pattern"]
                    width = {"^[0-9a-f]{40}$": 40, "^[0-9a-f]{64}$": 64}[pattern]
                    for raw, accepted in (
                        ("a" * (width - 1), False),
                        ("a" * width, True),
                        ("a" * (width + 1), False),
                        ("z" * width, False),
                    ):
                        bad = deepcopy(value)
                        assign(bad, path, raw)
                        if accepted:
                            admission.validate_support(bad, kind)
                        else:
                            with self.assertRaises(ValueError):
                                admission.validate_support(bad, kind)
        self.assertEqual(found, set(LENGTHS))

    def test_inventory_one_and_128_items_and_numeric_constants(self):
        for kind, value in self.fixtures():
            for path, _, _, schema in leaves(admission.SCHEMAS[kind], definition=kind):
                if schema.get("type") == "array":
                    self.assertEqual((schema["minItems"], schema["maxItems"]), (1, 128))
                    item = fixture.skeleton(schema["items"])
                    self.check_edges(
                        kind,
                        value,
                        path,
                        1,
                        128,
                        lambda n: [deepcopy(item) for _ in range(n)],
                    )
                elif type(schema.get("const")) is int:
                    bound = schema["const"]
                    self.check_edges(kind, value, path, bound, bound)


if __name__ == "__main__":
    unittest.main()
