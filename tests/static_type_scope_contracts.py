"""Pin the qualified default type scope and its inert checker invocation."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import check

# Independent witness of the qualified runner, modules and clients.
EXPECTED = (
    "tools/check.py",
    "tools/atlas_host/inference_geometry.py",
    "tools/atlas_host/inference_engine.py",
    "tools/atlas_host/inference_edit_contract.py",
    "tools/atlas_host/inference_observation_contract.py",
    "tools/atlas_host/inference_pair_contract.py",
    "tools/atlas_host/inference_sweep_contract.py",
    "tools/inference_geometry.py",
    "tools/inference_engine.py",
    "tools/inference_edit_contract.py",
    "tools/inference_observation_contract.py",
    "tools/inference_pair_contract.py",
    "tools/inference_sweep_contract.py",
    "tools/inference_experiments.py",
    "tools/inference_comparison.py",
    "tools/inference_generation.py",
    "tools/inference_services.py",
    "tools/atlas_host/inference_comparison.py",
    "tools/atlas_host/inference_generation.py",
    "tools/atlas_host/inference_experiments.py",
    "tests/typing/experiment_clients.py",
    "tools/atlas_host/inference_services.py",
    "tools/atlas_host/inference_architecture.py",
    "tools/inference_architecture.py",
    "tools/atlas_host/inference_edits.py",
    "tools/atlas_host/inference_observations.py",
    "tools/inference_edits.py",
    "tools/inference_observations.py",
    "tools/atlas_host/inference_prompt_pair.py",
    "tools/atlas_host/inference_sweep.py",
    "tools/inference_prompt_pair.py",
    "tools/inference_sweep.py",
    "tools/atlas_host/viewer_resources.py",
    "tools/atlas_host/launch.py",
    "tools/viewer_resources.py",
    "tools/launch.py",
    "tools/atlas_host/common.py",
    "tests/typing/receipt_clients.py",
    "tools/atlas_host/inference_model_descriptor.py",
    "tools/inference_model_descriptor.py",
    "tests/typing/descriptor_clients.py",
    "tools/analytics/core.py",
    "tests/typing/analytics_core_clients.py",
    "tools/analytics/svd_summary.py",
    "tests/typing/analytics_summary_clients.py",
    "tools/analytics/source.py",
    "tests/typing/analytics_source_clients.py",
    "tools/analytics/svd.py",
    "tests/typing/analytics_svd_clients.py",
    "tools/analytics/profiles.py",
    "tests/typing/analytics_profile_clients.py",
    "tools/analytics/worker.py",
    "tests/typing/analytics_worker_clients.py",
    "tools/analytics/service.py",
    "tests/typing/analytics_service_clients.py",
    "tools/live_inference.py",
    "tests/typing/coordinator_clients.py",
    "tools/inference_worker.py",
    "tests/typing/worker_clients.py",
    "tools/atlas_host/inference_worker.py",
    "tests/typing/package_worker_clients.py",
    "tools/atlas_host/live_inference.py",
    "tests/typing/package_coordinator_clients.py",
    "tools/atlas_host/profile_observation.py",
    "tools/atlas_host/startup_diagnostics.py",
    "tests/typing/host_leaf_clients.py",
    "tools/atlas_host/supervisor.py",
    "tests/typing/supervisor_clients.py",
    "tools/atlas_host/static_models.py",
    "tests/typing/static_policy_clients.py",
    "tools/atlas_host/profile_api.py",
    "tests/typing/profile_api_clients.py",
    "tools/atlas_host/profile_snapshot.py",
    "tests/typing/snapshot_clients.py",
    "tools/atlas_host/profile_worker.py",
    "tests/typing/profile_worker_clients.py",
    "tools/atlas_host/profile_os.py",
    "tests/typing/profile_os_clients.py",
    "tools/atlas_host/fixture_source.py",
    "tests/typing/fixture_leaf_clients.py",
    "tools/atlas_host/validation_policy.py",
    "tests/typing/policy_clients.py",
    "tools/atlas_host/lifetime_guard.py",
    "tests/typing/lifetime_clients.py",
    "tools/atlas_host/profile_platform.py",
    "tests/typing/platform_clients.py",
    "tools/atlas_host/profile_service.py",
    "tests/typing/service_clients.py",
    "tools/atlas_host/host_assets.py",
    "tests/typing/host_asset_clients.py",
    "tools/atlas_host/dense_static_admission.py",
    "tests/typing/dense_admission_clients.py",
)


class DefaultTypeScope(unittest.TestCase):
    def test_default_declaration_keeps_all_qualified_targets(self):
        self.assertEqual(check._MYPY_FILES, EXPECTED)
        self.assertEqual(len(EXPECTED), 92)
        self.assertEqual(len(set(EXPECTED)), len(EXPECTED))
        for relative in EXPECTED:
            with self.subTest(path=relative):
                self.assertFalse(Path(relative).is_absolute())
                self.assertTrue((check.ROOT / relative).is_file())

    def test_default_scope_reaches_actual_mypy_command(self):
        python = Path("/synthetic-development-python")
        ruff = Path("/synthetic-ruff")
        commands = []
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(check, "require_version"),
            patch.object(check, "formatter_files", return_value=["web/app.js"]),
            patch.object(check, "run", side_effect=commands.append),
        ):
            check.static_checks(python, ruff)
        self.assertEqual(len(commands), 3)
        self.assertEqual(
            commands[-1],
            [
                str(python),
                "-m",
                "mypy",
                "--config-file",
                "dev/mypy.ini",
                "--cache-dir",
                "/dev/null",
                *EXPECTED,
            ],
        )


if __name__ == "__main__":
    unittest.main()
