"""Owner adapters use ordinary fixtures and inert CLI/provider calls."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
from typing import Any, Callable, ParamSpec, TypeVar
import unittest
from unittest.mock import Mock, patch

from atlas_host import acquisition, cli, inference, prepare_fixture
from atlas_host.registry import content_digest
import host_contracts as fixtures
import host_picker_contracts as picker

P = ParamSpec("P")
T = TypeVar("T")


class OwnerAdapterBoundaryTests(unittest.TestCase):
    def ok(self, function: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return function(*args, **kwargs)
        except BaseException as error:
            self.fail(f"Valid inert call raised {type(error).__name__}: {error}")

    def test_cli_register_keeps_defaults_arguments_stdout_and_exit(self) -> None:
        config = {"paths": {"registry": "/inert/registry.json"}}
        manifest, receipt = {"version": 1}, {"version": 1, "enabled": False}
        registry = Mock()
        registry.register.return_value = receipt
        out, err = io.StringIO(), io.StringIO()
        with (
            patch.object(cli, "load_config", return_value=config) as load,
            patch.object(cli, "Registry", return_value=registry) as registry_type,
            patch.object(cli, "read_json", return_value=manifest) as read,
            redirect_stdout(out),
            redirect_stderr(err),
        ):
            result = self.ok(
                cli.main,
                [
                    "--config",
                    "owner.json",
                    "register",
                    "--manifest",
                    "manifest.json",
                    "--source",
                    "source",
                    "--name",
                    "Fixture",
                ],
            )
        self.assertEqual(result, 0)
        self.assertEqual(out.getvalue(), '{"enabled": false, "version": 1}\n')
        self.assertEqual(err.getvalue(), "")
        load.assert_called_once_with(Path("owner.json"))
        registry_type.assert_called_once_with(config["paths"]["registry"])
        read.assert_called_once_with(Path("manifest.json"), cli.MAX_MANIFEST_BYTES)
        registry.register.assert_called_once_with(
            Path("source"), manifest, "Fixture", max_bytes=1024**2, timeout_ms=1000
        )

    def test_cli_ordinary_config_error_keeps_stderr_and_exit(self) -> None:
        out, err = io.StringIO(), io.StringIO()
        with (
            patch.object(
                cli, "load_config", side_effect=OSError("inert config unavailable")
            ),
            redirect_stdout(out),
            redirect_stderr(err),
        ):
            self.assertEqual(self.ok(cli.main, ["--config", "owner.json", "list"]), 2)
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(err.getvalue(), "atlas-model: inert config unavailable\n")

    def test_cli_acquisition_pending_keeps_arguments_and_nonzero_exit(self) -> None:
        config = {"paths": {"registry": "/inert/registry.json"}}
        manifest = {"version": 1}
        registry = object()
        acquire = Mock(return_value={"installed": True, "registered": False})
        out, err = io.StringIO(), io.StringIO()
        with (
            patch.object(cli, "load_config", return_value=config),
            patch.object(cli, "Registry", return_value=registry),
            patch.object(cli, "read_json", return_value=manifest),
            patch.object(acquisition, "acquire", acquire),
            patch.object(cli, "acquire", acquire, create=True),
            redirect_stdout(out),
            redirect_stderr(err),
        ):
            result = self.ok(
                cli.main,
                [
                    "--config",
                    "owner.json",
                    "acquire",
                    "--manifest",
                    "manifest.json",
                    "--name",
                    "Fixture",
                    "--max-bytes",
                    "17",
                    "--destination",
                    "destination",
                    "--plan-digest",
                    "a" * 64,
                    "--accept-license",
                ],
            )
        self.assertEqual(result, 2)
        self.assertEqual(out.getvalue(), '{"installed": true, "registered": false}\n')
        self.assertEqual(err.getvalue(), "")
        acquire.assert_called_once_with(
            registry,
            Path("destination"),
            manifest,
            "Fixture",
            plan_digest="a" * 64,
            accept_license=True,
            timeout_ms=120000,
            max_bytes=17,
            cache_growth=0,
        )

    def test_configuration_evidence_keeps_exact_receipt_and_input_values(self) -> None:
        case = fixtures.InferenceBoundary(methodName="runTest")
        manifest, descriptor = case.inputs()
        selected = fixtures.binding_value([3, 4])
        before = deepcopy((manifest, descriptor, selected))
        result = self.ok(
            inference.configuration_evidence, manifest, descriptor, selected
        )
        self.assertEqual(
            result,
            {
                "content_digest": content_digest(manifest),
                "adapter_id": "builtin-llama-eager",
                "adapter_version": 1,
                "evidence": "pinned_configuration",
                "source_correspondence": True,
                "runtime_verified": False,
                "fit_verified": False,
                "inference_ready": False,
                "reason": "host_worker_adapter_not_active",
            },
        )
        self.assertIs(result["source_correspondence"], True)
        for field in ("runtime_verified", "fit_verified", "inference_ready"):
            self.assertIs(result[field], False)
        self.assertEqual((manifest, descriptor, selected), before)

    def layout(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        descriptor = json.loads(
            (fixtures.ROOT / "tests/fixtures/host-head-layout-v1.json").read_text()
        )
        source = descriptor["source_model"]
        manifest = {
            **deepcopy(fixtures.MANIFEST),
            "repository": source["repo"],
            "revision": source["revision"],
            "files": [
                {
                    "name": "model.safetensors",
                    "bytes": 1,
                    "sha256": source["weights_sha256"],
                },
                {"name": "config.json", "bytes": 1, "sha256": source["config_sha256"]},
            ],
        }
        selected = fixtures.binding_value([576, 576])
        selected["name"] = "model.layers.0.self_attn.o_proj.weight"
        return manifest, descriptor, selected

    def test_head_metadata_keeps_distinct_identities_and_independent_copy(self) -> None:
        manifest, descriptor, selected = self.layout()
        before = deepcopy((manifest, descriptor, selected))
        self.assertNotEqual(selected["source_identity"], selected["model_identity"])
        result = self.ok(inference.head_layout_metadata, manifest, descriptor, selected)
        self.assertEqual(
            result["head_layout_binding"],
            {
                "source_identity": selected["source_identity"],
                "model_identity": selected["model_identity"],
                "weights_sha256": descriptor["source_model"]["weights_sha256"],
                "config_sha256": descriptor["source_model"]["config_sha256"],
            },
        )
        self.assertEqual(result["head_layout"], descriptor)
        self.assertEqual(
            result["host_inference_evidence"],
            self.ok(inference.configuration_evidence, manifest, descriptor, selected),
        )
        result["head_layout"]["layers"] = 999
        self.assertEqual((manifest, descriptor, selected), before)

    def test_fixture_preparation_keeps_complete_command_and_receipt(self) -> None:
        case = picker.HostFixtureTests(methodName="runTest")
        self.addCleanup(case.doCleanups)
        self.ok(case.setUp)
        identifier = case.ids[0]
        entry = case.registry.owner_receipt(identifier)
        config = {
            "paths": {
                "registry": str(case.registry.path),
                "cache": str(case.root / "cache"),
            }
        }
        run = Mock(return_value=Mock(returncode=0))
        result = self.ok(prepare_fixture.prepare, config, identifier, run=run)
        self.assertEqual(
            result,
            {
                "model_id": identifier,
                "fixture_calibrated": True,
                "inference_ready": False,
                "download_enabled": False,
            },
        )
        self.assertIs(result["fixture_calibrated"], True)
        binary = fixtures.ROOT / "target/release/weight-atlas-rust"
        run.assert_called_once_with(
            [
                str(binary),
                "calibrate",
                "--model",
                entry["root"],
                "--cache",
                str(Path(config["paths"]["cache"]) / identifier),
                "--name",
                entry["name"],
                "--revision",
                entry["manifest"]["revision"],
            ],
            timeout=5,
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
