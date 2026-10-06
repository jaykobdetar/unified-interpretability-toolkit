"""Inert exact-profile metadata and refusal contracts; no package imports or reads."""

import copy
from importlib.machinery import ModuleSpec
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from analytics import profiles


class ProfileContracts(unittest.TestCase):
    def setUp(self):
        self.root = Path("fixture-model")
        self.config_path = self.root / "config.json"
        self.config_sha = "a" * 64
        self.implementation_sha = "b" * 64
        self.config = {"model_type": "fixture_type", "hidden_size": 4}
        self.profile = {
            "model_type": "fixture_type",
            "implementation": self.implementation_sha,
            "model": "fixture/model",
            "revision": "fixture-revision",
        }
        self.evidence = {
            "config_sha256": self.config_sha,
            "config_canonical_sha256": "c" * 64,
            "implementation_sha256": self.implementation_sha,
            "review_requirement": "fixture-only reviewed evidence",
        }
        self.transformers = ModuleSpec(
            "transformers",
            loader=None,
            origin="fixture-packages/transformers/__init__.py",
        )
        self.torch = ModuleSpec(
            "torch", loader=None, origin="fixture-packages/torch/__init__.py"
        )

    def reads(self):
        return [
            (self.config, {**self.evidence, "implementation_sha256": self.config_sha}),
            (self.config, copy.deepcopy(self.evidence)),
            (
                self.config,
                {**self.evidence, "implementation_sha256": profiles.LINEAR_SHA},
            ),
        ]

    def invoke(self, reads=None, specs=None):
        with (
            mock.patch.object(profiles, "PROFILES", {self.config_sha: self.profile}),
            mock.patch.object(
                profiles,
                "read_layout_evidence",
                side_effect=self.reads() if reads is None else reads,
            ) as read,
            mock.patch.object(
                profiles.importlib.util,
                "find_spec",
                side_effect=[self.transformers, self.torch] if specs is None else specs,
            ) as find,
        ):
            result = profiles.resolve_local(self.root)
        return result, read.call_args_list, find.call_args_list

    def test_resolve(self):
        reads = self.reads()
        (options, reason), calls, lookups = self.invoke(reads=reads)
        expected_evidence = {
            **self.evidence,
            "model": "fixture/model",
            "revision": "fixture-revision",
            "linear_implementation_sha256": profiles.LINEAR_SHA,
            "implementation_package": "transformers 4.56.2; source files only, no model imported",
        }
        self.assertIsNone(reason)
        self.assertEqual(
            options,
            {
                "config": {"model_type": "fixture_type", "hidden_size": 4},
                "evidence": expected_evidence,
                "reviewed_profiles": {
                    (self.config_sha, self.implementation_sha): profiles.LAYOUT,
                },
            },
        )
        self.assertIs(options["config"], self.config)
        self.assertIs(options["evidence"], reads[1][1])
        self.assertEqual(
            calls,
            [
                mock.call(self.config_path, self.config_path),
                mock.call(
                    self.config_path,
                    Path(
                        "fixture-packages/transformers/models/fixture_type/modeling_fixture_type.py"
                    ),
                ),
                mock.call(
                    self.config_path,
                    Path("fixture-packages/torch/nn/modules/linear.py"),
                ),
            ],
        )
        self.assertEqual(lookups, [mock.call("transformers"), mock.call("torch")])

        reads = self.reads()
        reads[0][1]["config_sha256"] = "d" * 64
        result, calls, lookups = self.invoke(reads=reads)
        self.assertEqual(result, ({}, "No reviewed exact configuration profile"))
        self.assertEqual(len(calls), 1)
        self.assertEqual(lookups, [])

        for specs in ([None, self.torch], [self.transformers, None]):
            with self.subTest(missing_spec=specs):
                result, calls, _ = self.invoke(specs=specs)
                self.assertEqual(
                    result, ({}, "Reviewed installed implementation source unavailable")
                )
                self.assertEqual(len(calls), 1)

        for key in ("config_sha256", "implementation_sha256"):
            with self.subTest(changed_implementation_field=key):
                reads = self.reads()
                reads[1][1][key] = "d" * 64
                result, calls, _ = self.invoke(reads=reads)
                self.assertEqual(
                    result,
                    ({}, "Configuration or official implementation digest changed"),
                )
                self.assertEqual(len(calls), 2)

        for key in ("config_sha256", "implementation_sha256"):
            with self.subTest(changed_linear_field=key):
                reads = self.reads()
                reads[2][1][key] = "d" * 64
                result, calls, _ = self.invoke(reads=reads)
                self.assertEqual(result, ({}, "PyTorch Linear layout evidence changed"))
                self.assertEqual(len(calls), 3)

        for specs, count in (
            ([ModuleSpec("transformers", loader=None, origin=None), self.torch], 1),
            ([self.transformers, ModuleSpec("torch", loader=None, origin=None)], 2),
        ):
            with self.subTest(missing_origin=specs):
                result, calls, _ = self.invoke(specs=specs)
                self.assertEqual(
                    result,
                    (
                        {},
                        "Missing or invalid local layout evidence; head labels disabled",
                    ),
                )
                self.assertEqual(len(calls), count)

        result, calls, lookups = self.invoke(
            reads=[ValueError("inert evidence refusal")]
        )
        self.assertEqual(
            result,
            ({}, "Missing or invalid local layout evidence; head labels disabled"),
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(lookups, [])


if __name__ == "__main__":
    unittest.main()
