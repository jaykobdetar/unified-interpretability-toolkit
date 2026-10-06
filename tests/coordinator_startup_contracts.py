"""Exercise coordinator startup with tiny pinned files and no listener or child."""

from contextlib import ExitStack, redirect_stdout
import hashlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import live_inference as live
from analytics import service
import head_layout_receipt_contracts


class Startup(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.pins = {**live.MANIFEST, "files": {}}
        for name in live.MANIFEST["files"]:
            raw = ("synthetic pinned startup bytes " + name).encode()
            (self.root / name).write_bytes(raw)
            self.pins["files"][name] = hashlib.sha256(raw).hexdigest()

    def run_startup(self, analytics=False):
        events = []
        server = SimpleNamespace()
        renderer = SimpleNamespace(poll=lambda: 0)

        def listen(*_args):
            events.append("listen")
            return server

        def spawn(*_args, **_options):
            events.append("renderer")
            return renderer

        args = [
            "live_inference.py",
            "--model",
            str(self.root),
            "--python",
            "unused",
            "--revision",
            self.pins["revision"],
        ]
        if analytics:
            args.append("--analytics-only")
        with ExitStack() as stack:
            for context in (
                patch.object(sys, "argv", args),
                patch.object(live, "MANIFEST", self.pins),
                patch.object(live, "available", return_value=6 * live.GIB),
                patch.object(live.os, "sched_setaffinity"),
                patch.object(live.signal, "signal"),
                patch.object(live, "HTTPServer", side_effect=listen),
                patch.object(live.subprocess, "Popen", side_effect=spawn),
                patch.object(service, "AnalyticsJobs", return_value=Mock(busy=False)),
                patch.object(live, "cleanup_owned", return_value=True),
                redirect_stdout(io.StringIO()),
            ):
                stack.enter_context(context)
            try:
                live.main()
            except Exception as error:
                return server, events, error
        return server, events, None

    def test_bad_inference_pins_refuse_before_listening_or_spawning(self):
        (self.root / "config.json").write_bytes(b"changed bytes")
        _, events, error = self.run_startup()
        self.assertIs(type(error), ValueError)
        self.assertIn("Pinned file hash mismatch", str(error))
        self.assertEqual(events, [])

    def test_analytics_only_accepts_unpinned_source_without_receipt(self):
        (self.root / "config.json").write_bytes(b"unpinned local source")
        server, events, error = self.run_startup(analytics=True)
        self.assertIsNone(error, f"Analytics-only startup refused: {error}")
        self.assertEqual(events, ["listen", "renderer"])
        self.assertFalse(server.session.inference_enabled)
        self.assertIsNone(server.session.head_layout_receipt)

    def test_verified_receipt_reaches_session_in_both_startup_modes(self):
        expected = live.layout_fingerprints(self.root)
        for analytics in (False, True):
            with self.subTest(analytics=analytics):
                server, events, error = self.run_startup(analytics=analytics)
                self.assertIsNone(error, f"Verified startup refused: {error}")
                self.assertEqual(events, ["listen", "renderer"])
                self.assertEqual(server.session.head_layout_receipt, expected)
                self.assertEqual(server.session.inference_enabled, not analytics)
                self.assertIsNone(server.session.process)

    def test_verified_native_binding_publishes_exact_identity_and_file_digests(self):
        fixture = head_layout_receipt_contracts.Receipts()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        session, model = fixture.session_and_model()
        with patch.object(
            live, "verify_model", side_effect=AssertionError("No request-time hashing")
        ):
            result = live.bind_inference_source(model, session)
        self.assertIn("head_layout", result)
        self.assertIn("head_layout_binding", result)
        self.assertEqual(result["head_layout"], live.head_layout_descriptor())
        self.assertEqual(
            result["head_layout_binding"],
            {
                "source_identity": model["source_identity"],
                "model_identity": model["model_identity"],
                "weights_sha256": live.MANIFEST["files"]["model.safetensors"],
                "config_sha256": live.MANIFEST["files"]["config.json"],
            },
        )
        self.assertIsNone(session.process)


if __name__ == "__main__":
    unittest.main()
