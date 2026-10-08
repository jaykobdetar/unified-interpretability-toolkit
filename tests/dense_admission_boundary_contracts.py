"""Admission return values and observations use synthetic receipts and inert data."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, fields
import hashlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from atlas_host import dense_static_admission as module
from atlas_host.common import canonical
from atlas_host.registry import fingerprint
import dense_static_contracts as existing


class DenseAdmissionBoundaryTests(unittest.TestCase):
    def ok(self, function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except BaseException as error:
            self.fail(f"Valid inert call raised {type(error).__name__}: {error}")

    def error(self, message, function, *args, **kwargs):
        try:
            function(*args, **kwargs)
        except BaseException as error:
            self.assertIs(type(error), ValueError)
            self.assertEqual(str(error), message)
        else:
            self.fail("Expected the existing refusal")

    def case(self):
        case = existing.PolicyTests(methodName="runTest")
        self.addCleanup(case.doCleanups)
        self.ok(case.setUp)
        return case

    def bound(self):
        case = self.case()
        file, owner, binary = self.ok(case.bundle)
        bound = self.ok(
            module.bind_dense_policy,
            file,
            existing.H(owner),
            case.registry,
            case.fs["cache"]["canonical_root"],
            binary,
        )
        self.assertIsInstance(bound, module.BoundDenseStaticAdmission)
        return case, bound, binary

    def fs(self, case, value=None):
        return self.ok(
            module.filesystem_checks,
            case.fs if value is None else value,
            case.fs["source"]["canonical_root"],
            case.fs["cache"]["canonical_root"],
        )

    def test_supporting_values_keep_identity_and_both_scopes(self):
        case = self.case()
        for kind, value in [
            ("filesystemReceipt", case.fs),
            ("sourceRecipe", case.recipe),
        ]:
            self.assertIs(self.ok(module.validate_support, value, kind), value)
        self.error("Supporting scope unavailable", module.validate_support, {}, "other")

    def test_finite_schema_defaults_reference_and_array_boundary(self):
        schema = {
            "type": "object",
            "required": ["number", "label", "items"],
            "properties": {
                "number": {"$ref": "#/number"},
                "label": {"type": "string"},
                "items": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 1,
                    "items": {"type": "integer"},
                },
            },
        }
        value = {"number": 0, "label": "ok", "items": [0]}
        with patch.dict(
            module.SCHEMAS, {"filesystemReceipt": schema, "number": {"type": "integer"}}
        ):
            self.assertIs(
                self.ok(module.validate_support, value, "filesystemReceipt"), value
            )

    def test_verified_binary_default_and_canonical_json_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inert-data"
            for raw, kwargs, expected in [
                (b"ordinary-data", {}, None),
                (canonical({"value": 2}), {"json_artifact": True}, {"value": 2}),
            ]:
                path.write_bytes(raw)
                before = fingerprint(path.stat())
                result = self.ok(
                    module._verified,
                    path,
                    hashlib.sha256(raw).hexdigest(),
                    len(raw),
                    **kwargs,
                )
                self.assertEqual(
                    result, (expected, (str(path), tuple(sorted(before.items()))))
                )

    def test_git_result_order_and_exact_bounded_command(self):
        root = Path("/synthetic/source")
        with patch.object(
            module.subprocess, "run", return_value=NS(stdout=b"head\ntree\n")
        ) as run:
            self.assertEqual(self.ok(module._git_version, root), ["head", "tree"])
        run.assert_called_once_with(
            ["git", "rev-parse", "HEAD", "HEAD^{tree}"],
            cwd=root,
            check=True,
            stdout=module.subprocess.PIPE,
            stderr=module.subprocess.DEVNULL,
            timeout=1,
        )

    def test_root_identity_fields_and_longest_mount(self):
        root = Path("/synthetic/source")
        raw = b"1 0 1:1 / / rw - ext4 none rw\n2 1 1:2 / /synthetic rw - xfs none rw\n"
        with (
            patch.object(Path, "is_dir", return_value=True),
            patch.object(Path, "is_symlink", return_value=False),
            patch.object(Path, "resolve", return_value=root),
            patch.object(Path, "stat", return_value=NS(st_dev=11, st_ino=19)),
            patch.object(Path, "open", return_value=io.BytesIO(raw)),
        ):
            result = self.ok(module._root_identity, root)
        self.assertEqual(
            result,
            {
                "canonical_root": str(root),
                "device": 11,
                "directory_inode": 19,
                "filesystem_type": "xfs",
            },
        )

    def test_platform_fields_and_required_api_observations(self):
        with (
            patch.object(module.sys, "platform", "unit-system"),
            patch.object(module.platform, "release", return_value="unit-kernel"),
            patch.object(module.platform, "machine", return_value="unit-arch"),
            patch.object(
                module.platform, "python_version", return_value="unit-version"
            ),
            patch.object(module.os, "stat", return_value=NS(st_mtime_ns=1)),
        ):
            result = self.ok(module._platform)
        self.assertEqual(
            result,
            {
                "system": "unit-system",
                "kernel_release": "unit-kernel",
                "native_arch": "unit-arch",
                "python_implementation": module.sys.implementation.name,
                "python_version": "unit-version",
                "required_apis": {
                    "stat_ns": True,
                    "fstat": True,
                    "pread": True,
                    "o_nofollow": True,
                    "flock_lock_ex": True,
                },
            },
        )

    def test_filesystem_success_retains_none_and_binds_both_roots(self):
        case = self.case()
        self.assertIsNone(self.fs(case))

    def test_filesystem_binding_refusal_text(self):
        case = self.case()
        bad = deepcopy(case.fs)
        bad["platform"]["kernel_release"] = "other"
        self.error(
            "Filesystem/platform binding differs",
            module.filesystem_checks,
            bad,
            case.fs["source"]["canonical_root"],
            case.fs["cache"]["canonical_root"],
        )

    def test_publication_receipt_refusal_text(self):
        case = self.case()
        bad = deepcopy(case.fs)
        item = bad["observations"]["source"]["ordinary_replace"]
        item["after"]["inode"] = item["before"]["inode"]
        self.error(
            "Ordinary publication evidence differs",
            module.filesystem_checks,
            bad,
            case.fs["source"]["canonical_root"],
            case.fs["cache"]["canonical_root"],
        )

    def test_nested_fingerprint_device_refusal_text(self):
        case = self.case()
        bad = deepcopy(case.fs)
        opened = bad["observations"]["source"]["open_identity"]
        for name in ["before", "opened", "after_fd", "after_path"]:
            opened[name] = dict(opened[name], device=99)
        self.error(
            "Probe fingerprint device differs",
            module.filesystem_checks,
            bad,
            case.fs["source"]["canonical_root"],
            case.fs["cache"]["canonical_root"],
        )

    def test_cache_preflight_none_results_and_closed_owned_descriptor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIsNone(self.ok(module.check_cache_available, root / "absent"))
            (root / "atlas.lock").write_bytes(b"inert lock")
            close = module.os.close
            with patch.object(module.os, "close", wraps=close) as closed:
                self.assertIsNone(self.ok(module.check_cache_available, root))
            self.assertEqual(closed.call_count, 1)

    def test_cache_busy_refusal_keeps_exact_wording(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "atlas.lock").write_bytes(b"inert lock")
            with patch.object(
                module.fcntl, "flock", side_effect=BlockingIOError("busy")
            ):
                self.error("Cache owner busy", module.check_cache_available, root)

    def test_target_entry_none_and_exact_owner_refusal(self):
        case = self.case()
        self.assertIsNone(self.ok(module.target_entry, case.entry))
        value = deepcopy(case.entry)
        value["enabled"] = False
        self.error("Fixed accepted owner target required", module.target_entry, value)

    def test_bound_check_returns_actual_current_receipt_object(self):
        case, bound, _ = self.bound()
        case.registry.owner_receipt.side_effect = None
        case.registry.owner_receipt.return_value = case.entry
        self.assertIs(self.ok(bound.check), case.entry)

    def test_bound_filesystem_refusal_text(self):
        case, bound, _ = self.bound()
        with patch.object(module, "_platform", return_value={"changed": True}):
            self.error("Bound filesystem moved/changed", bound.check)

    def test_allows_returns_exact_booleans_for_each_outcome(self):
        case, bound, _ = self.bound()
        self.assertIs(self.ok(bound.allows, case.entry["model_id"]), True)
        self.assertIs(self.ok(bound.allows, "different"), False)
        case.registry.owner_receipt.side_effect = ValueError(
            "inert receipt unavailable"
        )
        self.assertIs(self.ok(bound.allows, case.entry["model_id"]), False)

    def test_admit_exact_true_and_single_prepared_check(self):
        case, bound, _ = self.bound()
        prepared = NS(
            entry=case.entry,
            descriptor={"model_type": "llama", "parameter_count": 134515008},
            tensors={str(i): {"dtype": "BF16"} for i in range(272)},
            check=Mock(),
        )
        self.assertIs(self.ok(bound.admit, prepared), True)
        prepared.check.assert_called_once_with()

    def test_binary_check_count_none_and_exact_path_refusal(self):
        _, bound, path = self.bound()
        binary = NS(path=path, check=Mock())
        self.assertIsNone(self.ok(bound.check_binary, binary))
        binary.check.assert_called_once_with()
        binary.path = path.parent / "other-inert-data"
        self.error("Owner binary path differs", bound.check_binary, binary)

    def test_binding_identity_file_order_frozen_fields_and_one_final_check(self):
        case = self.case()
        file, owner, binary = self.ok(case.bundle)
        check = module.BoundDenseStaticAdmission.check
        with patch.object(
            module.BoundDenseStaticAdmission, "check", autospec=True, side_effect=check
        ) as checked:
            bound = self.ok(
                module.bind_dense_policy,
                file,
                existing.H(owner),
                case.registry,
                case.fs["cache"]["canonical_root"],
                binary,
            )
        self.assertIsInstance(bound, module.BoundDenseStaticAdmission)
        checked.assert_called_once_with(bound)
        self.assertIs(bound.registry, case.registry)
        self.assertEqual(bound.model_id, case.entry["model_id"])
        self.assertEqual(bound.owner_sha, existing.H(case.entry))
        self.assertEqual(bound.inventory, ("only.py",))
        self.assertEqual(
            [row[0] for row in bound.files],
            [
                str(file),
                str(
                    file.parent
                    / "receipts"
                    / (owner["filesystem_receipt_sha256"] + ".json")
                ),
                str(
                    file.parent / "receipts" / (owner["source_recipe_sha256"] + ".json")
                ),
                str(file.parent / "only.py"),
                str(binary),
            ],
        )
        self.assertEqual(
            [field.name for field in fields(bound)],
            [
                "registry",
                "model_id",
                "owner_sha",
                "binary_sha",
                "binary_path",
                "files",
                "inventory",
                "source_root",
                "cache_root",
                "filesystem",
                "_seal",
            ],
        )
        with self.assertRaises(FrozenInstanceError):
            bound.model_id = "changed"

    def test_saved_native_statistics_counts_and_none_result(self):
        tensor = {
            "calibration_complete": True,
            "count": 5,
            "max_abs": 1.0,
            "median_nonzero_abs": 0.5,
            "q99": 0.9,
            "robust_clipped_count": 1,
            "exact_zero_count": 0,
            "unique_bit_patterns": 2,
            "calibration_method": "exact-16-bit-histogram",
            "quantile_order_statistics": "exact",
            "quantile_interpolation": "linear in F64; final floating-point rounding possible",
            "rule_status": {
                "tensor_signed_percentile": "supported dtype; requires a valid exact histogram"
            },
        }
        coverage = {
            "source_complete": True,
            "active_tensor": None,
            "all_requested": False,
            "calibration_error": None,
            "materialized_bytes": 0,
            "materialized_tiles": 0,
            "cache_budget_bytes": 2 * 1024**3,
            "fine_tile_file_cap": 1000,
            "calibrated_tensors": 1,
            "values_streamed": 5,
            "statistics_complete": True,
            "sha_hashed_shards": 0,
            "sha_expected_matched_shards": 0,
            "sha_missing_expected_shards": 0,
            "sha_verified_shards": 0,
        }
        model = {
            "name": "Owner",
            "fresh_source_hashes": None,
            "global_max": 1.0,
            "calibration_complete": True,
            "coverage": coverage,
            "catalog": [tensor],
        }
        self.assertIsNone(
            self.ok(module.check_native_model, NS(entry={"name": "Owner"}), model)
        )


if __name__ == "__main__":
    unittest.main()
