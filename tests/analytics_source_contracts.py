"""Tiny trusted catalog contracts; no model discovery or execution."""

import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from analytics import source
from analytics.core import digest


class SourceContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "fixture.safetensors"
        values = (1.0, -2.0, 0.0, 4.0, 5.0, -6.0, 7.0, 8.0)
        self.path.write_bytes(
            b"".join(
                struct.pack("<H", struct.unpack("<I", struct.pack("<f", x))[0] >> 16)
                for x in values
            )
        )
        self.tensors = (
            source.Tensor("matrix", (2, 3), self.path.name, 0),
            source.Tensor("vector", (2,), self.path.name, 12),
        )
        self.catalog = source.Catalog(self.root, self.tensors, "trusted-fixture")
        self.region = {"row": 1, "col": 1, "rows": 1, "cols": 2}
        self.config = self.root / "config.json"
        self.impl = self.root / "layout.py"
        self.config.write_bytes(b'{"hidden_size": 4, "axes": [2, 3]}\n')
        self.impl.write_bytes(b"# inert reviewed layout fixture\n")

    def outcome(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs), None
        except (ValueError, TypeError, OSError) as exc:
            return None, (type(exc), str(exc))

    def test_fingerprint(self):
        metadata = SimpleNamespace(
            st_dev=11, st_ino=23, st_size=37, st_mtime_ns=41, st_ctime_ns=53
        )
        self.assertEqual(source.fingerprint(metadata), (11, 23, 37, 41, 53))

    def test_optional_index(self):
        self.assertIsNone(source.optional_index_fingerprint(self.root))
        index = self.root / "model.safetensors.index.json"
        index.write_bytes(b"{}\n")
        st = index.lstat()
        expected = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        self.assertEqual(source.optional_index_fingerprint(self.root), expected)

    def test_constructor(self):
        self.assertEqual(self.catalog.host_identity, "trusted-fixture")
        self.assertEqual(self.catalog.tensors, self.tensors)
        st = self.path.lstat()
        fingerprint = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        files = {self.path.name: fingerprint}
        self.assertEqual(self.catalog.files, files)
        expected = {
            "host_identity": "trusted-fixture",
            "files": files,
            "index": None,
            "tensors": [
                {
                    "name": "matrix",
                    "shape": (2, 3),
                    "shard": self.path.name,
                    "byte_offset": 0,
                },
                {
                    "name": "vector",
                    "shape": (2,),
                    "shard": self.path.name,
                    "byte_offset": 12,
                },
            ],
        }
        self.assertEqual(self.catalog.identity, digest(expected))

    def test_verify(self):
        with mock.patch.object(
            source, "fingerprint", wraps=source.fingerprint
        ) as fingerprint:
            self.assertEqual(self.outcome(self.catalog.verify), (None, None))
            self.assertEqual(fingerprint.call_count, 1)
        with mock.patch.object(
            source, "optional_index_fingerprint", return_value=("changed",)
        ):
            self.assertEqual(
                self.outcome(self.catalog.verify),
                (
                    None,
                    (
                        ValueError,
                        "Source index identity changed; discard analytics and caches",
                    ),
                ),
            )
        with mock.patch.object(source, "fingerprint", return_value=("changed",)):
            self.assertEqual(
                self.outcome(self.catalog.verify),
                (
                    None,
                    (
                        ValueError,
                        "Source identity changed; discard analytics and caches",
                    ),
                ),
            )

    def test_read(self):
        self.assertEqual(self.catalog.read(self.tensors[0], self.region), [5.0, -6.0])
        self.assertEqual(
            self.catalog.read(
                self.tensors[1], {"row": 0, "col": 0, "rows": 1, "cols": 2}
            ),
            [7.0, 8.0],
        )

    def test_region(self):
        values = [5.0, -6.0]
        sentinel = object()
        with (
            mock.patch.object(self.catalog, "read", return_value=values) as read,
            mock.patch.object(source, "analyze", return_value=sentinel) as analyze,
        ):
            result = self.catalog.region("matrix", self.region, seed=7, top=2)
        self.assertIs(result, sentinel)
        read.assert_called_once_with(self.tensors[0], self.region)
        analyze.assert_called_once_with(
            values,
            source_identity=self.catalog.identity,
            tensor="matrix",
            shape=[2, 3],
            region=self.region,
            seed=7,
            top=2,
        )
        self.assertIs(analyze.call_args.kwargs["region"], self.region)

    def test_validate_cached(self):
        report = {"source_identity": self.catalog.identity, "sentinel": object()}
        with mock.patch.object(self.catalog, "verify") as verify:
            self.assertIs(self.catalog.validate_cached(report), report)
            verify.assert_called_once_with()
        self.assertEqual(
            self.outcome(self.catalog.validate_cached, {"source_identity": "other"}),
            (None, (ValueError, "Cached analytics belong to another source")),
        )

    def test_model_outliers(self):
        with mock.patch.object(source.time, "monotonic", return_value=0.0):
            full = self.catalog.model_outliers(seed=7)
            partial = self.catalog.model_outliers(value_budget=2, seed=7)
        self.assertIs(full["coverage"]["full_model"], True)
        self.assertIs(partial["coverage"]["full_model"], False)
        self.assertEqual(full["control"]["seed"], 7)
        self.assertEqual(full["coverage"]["visited_values"], 8)
        self.assertEqual(partial["coverage"]["visited_values"], 2)
        self.assertEqual(
            [row["value"] for row in full["rankings"]["original"]["values"]],
            [8.0, 7.0, -6.0, 5.0, 4.0, -2.0, 1.0, 0.0],
        )
        self.assertEqual(
            [row["mean_abs"] for row in full["rankings"]["original"]["rows"]],
            [7.5, 5.0, 1.0],
        )
        self.assertEqual(
            partial["coverage"]["tensors"][1]["excluded_reason"],
            "value budget exhausted",
        )
        self.assertEqual(full["source_identity"], self.catalog.identity)

    def test_layout_evidence(self):
        config, evidence = source.read_layout_evidence(self.config, self.impl)
        self.assertEqual(config, {"hidden_size": 4, "axes": [2, 3]})
        self.assertEqual(
            evidence,
            {
                "config_sha256": hashlib.sha256(self.config.read_bytes()).hexdigest(),
                "config_canonical_sha256": digest(config),
                "implementation_sha256": hashlib.sha256(
                    self.impl.read_bytes()
                ).hexdigest(),
                "review_requirement": "Exact digest pair must match application-owned reviewed layout profile",
            },
        )

    def test_layout_read(self):
        with mock.patch.object(source.os, "read", wraps=source.os.read) as read:
            result, error = self.outcome(
                source.read_layout_evidence, self.config, self.impl
            )
        self.assertIsNone(error)
        self.assertEqual(
            [call.args[1] for call in read.call_args_list], [1048577, 1048577]
        )
        self.assertEqual(
            result[1]["config_sha256"],
            hashlib.sha256(self.config.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            result[1]["implementation_sha256"],
            hashlib.sha256(self.impl.read_bytes()).hexdigest(),
        )

    def test_layout_unique(self):
        result, error = self.outcome(
            source.read_layout_evidence, self.config, self.impl
        )
        self.assertIsNone(error)
        self.assertEqual(result[0], json.loads(self.config.read_bytes()))
        self.config.write_bytes(b'{"x": 1, "x": 2}')
        self.assertEqual(
            self.outcome(source.read_layout_evidence, self.config, self.impl),
            (None, (ValueError, "Duplicate configuration key")),
        )


if __name__ == "__main__":
    unittest.main()
