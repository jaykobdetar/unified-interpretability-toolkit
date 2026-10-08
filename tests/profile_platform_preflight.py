"""Preflight checks exported APIs before any hosted launch side effect."""

from pathlib import Path
import sys, unittest
from unittest.mock import patch

from atlas_host import profile_os, hosted_runtime


class Preflight(unittest.TestCase):
    def test_all_required_exported_apis_are_recorded(self):
        facts = profile_os.require_platform()
        self.assertEqual(facts["executable"], sys.executable)
        self.assertTrue(all(facts["apis"].values()))
        self.assertIn("signal.pidfd_send_signal", facts["apis"])
        self.assertIn("fcntl.F_SEAL_SEAL", facts["apis"])

    def test_each_missing_api_refuses_without_constructing_process_book(self):
        original = profile_os.platform_capabilities()
        for name in original["apis"]:
            facts = {**original, "apis": {**original["apis"], name: False}}
            with (
                self.subTest(api=name),
                patch.object(profile_os, "platform_capabilities", return_value=facts),
                patch.object(hosted_runtime, "ProcessBook") as book,
            ):
                with self.assertRaisesRegex(
                    ValueError, "explicit compatible Linux Python"
                ):
                    hosted_runtime.HostedApplication(None, None, None, None)
                book.assert_not_called()

    def test_non_linux_metadata_refuses(self):
        facts = {**profile_os.platform_capabilities(), "platform": "unsupported"}
        with patch.object(profile_os, "platform_capabilities", return_value=facts):
            with self.assertRaises(ValueError):
                profile_os.require_platform()

    def test_start_rechecks_preflight_before_gate_or_threads(self):
        app = object.__new__(hosted_runtime.HostedApplication)
        with (
            patch.object(
                hosted_runtime,
                "require_platform",
                side_effect=ValueError("missing API"),
            ),
            patch.object(hosted_runtime, "available_bytes") as available,
        ):
            with self.assertRaisesRegex(ValueError, "missing API"):
                app.start_threads()
            available.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
