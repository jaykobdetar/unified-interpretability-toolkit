"""Policy and tiny-fixture records without provider activation or live workers."""

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from atlas_host import runtime_adapter as adapter
from atlas_host import validation_policy as policy
from atlas_host.registry import fingerprint
import profile_lifetime_policy as existing


class PolicyBoundaryContracts(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for target in (
            "subprocess.Popen",
            "threading.Thread.start",
            "socket.socketpair",
            "os.pidfd_open",
            "os.kill",
            "signal.pidfd_send_signal",
            "os.memfd_create",
        ):
            self.stack.enter_context(
                patch(target, side_effect=AssertionError("Live work forbidden"))
            )

    def ok(self, callback, *args, **kwargs):
        try:
            return callback(*args, **kwargs)
        except Exception as error:
            self.fail(f"Expected valid local call: {type(error).__name__}: {error}")

    def error(self, message, callback, *args, **kwargs):
        try:
            callback(*args, **kwargs)
        except Exception as error:
            self.assertIs(type(error), ValueError)
            self.assertEqual(str(error), message)
        else:
            self.fail("Expected existing local refusal")

    def example(self):
        case = existing.PolicyTests("test_default_internal_gates_remain_five_gib")
        self.ok(case.setUp)
        self.addCleanup(case.doCleanups)
        bound = self.ok(case.bind)
        return case, bound

    def fixture(self):
        root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        source = (
            Path(adapter.__file__).resolve().parents[2]
            / "fixtures/tiny-bf16/tiny.safetensors"
        )
        shutil.copyfile(source, root / "tiny.safetensors")
        entry = {
            "root": str(root),
            "enabled": True,
            "manifest": {
                "provenance": "synthetic_fixture",
                "files": deepcopy(adapter.FIXTURE_FILES),
            },
            "fingerprints": {
                "tiny.safetensors": fingerprint((root / "tiny.safetensors").lstat())
            },
        }
        return root, entry

    def test_package_root_matches_module_location_and_path_type(self):
        actual = self.ok(policy.package_root)
        self.assertIsInstance(actual, Path)
        self.assertEqual(actual, Path(policy.__file__).resolve().parents[2])

    def test_source_names_exact_required_sorted_inventory_and_upper_edge(self):
        root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        (root / "tools/atlas_host").mkdir(parents=True)
        (root / "src").mkdir()
        (root / "tools/atlas_host/z.py").write_text("")
        (root / "tools/atlas_host/a.py").write_text("")
        expected = sorted(
            [
                "Cargo.toml",
                "Cargo.lock",
                "fixtures/tiny-bf16/tiny.safetensors",
                "fixtures/tiny-bf16/host-manifest.json",
                "tools/make_fixture.py",
                "tools/atlas_host/z.py",
                "tools/atlas_host/a.py",
            ]
        )
        self.assertEqual(self.ok(policy.source_names, root), expected)
        for i in range(121):
            (root / "src" / f"source_{i:03}.rs").write_text("")
        self.assertEqual(len(self.ok(policy.source_names, root)), 128)

    def test_verified_file_exact_sorted_identity_and_existing_refusals(self):
        root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        path = root / "small"
        path.write_bytes(b"abc")
        sha = hashlib.sha256(b"abc").hexdigest()
        expected = (str(path), tuple(sorted(fingerprint(path.stat()).items())))
        self.assertEqual(self.ok(policy._verified_file, path, sha, 3), expected)
        self.error("Recipe file exceeds bound", policy._verified_file, path, sha, 2)
        self.error(
            "Reviewed recipe file mismatch", policy._verified_file, path, "0" * 64, 3
        )

    def test_bound_start_bytes_exact_integer_threshold(self):
        _, bound = self.example()
        self.assertIs(type(bound.start_bytes), int)
        self.assertEqual(bound.start_bytes, 17 * 1024**3 // 4)

    def test_bound_check_none_result_and_exact_seal_or_inventory_refusals(self):
        case, bound = self.example()
        self.assertIsNone(self.ok(bound.check))
        self.error(
            "Owner-reviewed recipe required", replace(bound, _seal=object()).check
        )
        (case.root / "tools/atlas_host/new.py").write_text("")
        self.error("Bound source inventory changed", bound.check)

    def test_bound_binary_checks_once_and_preserves_existing_path_message(self):
        case, bound = self.example()
        binary = types.SimpleNamespace(path=Path(bound.binary_path), check=Mock())
        self.assertIsNone(self.ok(bound.check_binary, binary))
        binary.check.assert_called_once_with()
        binary.path = case.root / "other"
        self.error("Recipe binary path mismatch", bound.check_binary, binary)

    def test_bound_entry_rechecks_policy_and_reads_one_exact_tiny_fixture(self):
        _, bound = self.example()
        root, entry = self.fixture()
        checked = Mock()
        original = policy.BoundValidationPolicy.check

        def check(instance):
            checked(instance)
            return original(instance)

        with (
            patch.object(policy.BoundValidationPolicy, "check", check),
            patch.object(os, "open", wraps=os.open) as opened,
        ):
            self.assertIsNone(self.ok(bound.check_entry, entry))
            checked.assert_called_once_with(bound)
            opened.assert_called_once_with(
                root / "tiny.safetensors", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            )

    def test_bound_command_exact_owner_arguments_and_refusal_text(self):
        case, bound = self.example()
        self.assertEqual(
            self.ok(bound.command, "/usr/bin/python3", case.root),
            [
                "/usr/bin/python3",
                "-B",
                str(case.root / "runtime_harness.py"),
                str(case.root),
                bound.receipt_sha256,
            ],
        )
        self.error(
            "Recipe command unavailable", bound.command, "fixture-python", case.root
        )

    def test_bound_receipt_outputs_exact_paths_and_digest(self):
        case, bound = self.example()
        self.assertEqual(
            bound.receipt_sha256, hashlib.sha256(case.receipt.read_bytes()).hexdigest()
        )
        self.assertEqual(bound.monitor_path, str(case.root / "run.py"))
        self.assertEqual(bound.harness_path, str(case.root / "runtime_harness.py"))
        self.assertEqual(
            bound.binary_path, str(case.root / "target/release/weight-atlas-rust")
        )
        self.assertEqual(
            bound.source_inventory, tuple(sorted(case.data["source_sha256"]))
        )

    def test_start_gate_exact_threshold_return_and_error(self):
        self.assertIsNone(self.ok(policy.check_start, None, 5 * 1024**3, 25 * 1024**3))
        self.error(
            "Hosted launch gate refused",
            policy.check_start,
            None,
            5 * 1024**3 - 1,
            25 * 1024**3,
        )

    def test_memory_error_words_for_each_independent_refusal(self):
        good = {
            "rss_bytes": 0,
            "available_bytes": policy.STOP_RESERVE,
            "all_owned_accounted": True,
            "descendants_clear": True,
        }
        self.assertIsNone(self.ok(policy.check_memory, good, 0))
        self.error(
            "Hosted resource observation uncertain",
            policy.check_memory,
            {**good, "all_owned_accounted": False},
            0,
        )
        self.error(
            "Hosted charged memory ceiling exceeded",
            policy.check_memory,
            {**good, "rss_bytes": policy.TREE_CEILING + 1},
            0,
        )
        self.error(
            "Hosted stop reserve breached",
            policy.check_memory,
            {**good, "available_bytes": policy.STOP_RESERVE - 1},
            0,
        )

    def test_qualification_boundary_identity_and_single_check(self):
        _, bound = self.example()
        with patch.object(policy.BoundValidationPolicy, "check") as check:
            boundary = self.ok(policy.QualificationBoundary, bound)
            self.assertIs(boundary.policy, bound)
            check.assert_called_once_with()
        self.error(
            "Owner-reviewed recipe required", policy.QualificationBoundary, object()
        )

    def test_qualification_preflight_exact_arguments_and_none(self):
        _, bound = self.example()
        boundary = self.ok(policy.QualificationBoundary, bound)
        with patch.object(policy, "check_start") as check:
            self.assertIsNone(self.ok(boundary.preflight, 7000, 8000))
            check.assert_called_once_with(bound, 7000, 8000)

    def test_qualification_sample_exact_full_snapshot_charge_and_none(self):
        _, bound = self.example()
        boundary = self.ok(policy.QualificationBoundary, bound)
        observed = {"rss_bytes": 1}
        with patch.object(policy, "check_memory") as check:
            self.assertIsNone(self.ok(boundary.check_sample, observed))
            check.assert_called_once_with(observed, 32 * 1024**2)

    def test_fixture_entry_exact_true_boolean_on_owned_tiny_receipt(self):
        _, entry = self.fixture()
        self.assertIs(self.ok(adapter.fixture_entry, entry), True)

    def test_fixture_check_existing_inventory_fingerprint_and_hash_words(self):
        root, entry = self.fixture()
        self.assertIsNone(self.ok(adapter.check_fixture, entry, hash_bytes=True))
        extra = root / "extra.safetensors"
        extra.write_bytes(b"x")
        self.error("Fixture source inventory changed", adapter.check_fixture, entry)
        extra.unlink()
        changed = deepcopy(entry)
        changed["fingerprints"]["tiny.safetensors"]["bytes"] += 1
        self.error("Fixture fingerprint changed", adapter.check_fixture, changed)
        path = root / "tiny.safetensors"
        path.write_bytes(b"x" * 244)
        entry["fingerprints"]["tiny.safetensors"] = fingerprint(path.lstat())
        self.error(
            "Fixture hash differs from the allowed synthetic source",
            adapter.check_fixture,
            entry,
            hash_bytes=True,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
