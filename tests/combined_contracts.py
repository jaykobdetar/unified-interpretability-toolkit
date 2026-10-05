"""Combined admission/source boundaries with fake objects only; no runtime/model."""

import sys, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import live_inference as live
from analytics.service import AnalyticsJobs


class Combined(unittest.TestCase):
    def test_analysis_excludes_every_inference_mode_before_pin_or_preview(self):
        session = live.Session("unused", Path("/synthetic/smol"))
        session.analytics = SimpleNamespace(busy=True, tick=lambda: None)
        with (
            patch.object(
                live, "verify_model", side_effect=AssertionError("must not verify")
            ),
            patch.object(
                live.subprocess, "Popen", side_effect=AssertionError("must not spawn")
            ),
        ):
            for mode in ["generation", "prompt_pair_preview", "prompt_pair", "sweep"]:
                with (
                    self.subTest(mode=mode),
                    self.assertRaisesRegex(ValueError, "Analysis owns"),
                ):
                    session.start({"mode": mode})

    def test_inference_cleanup_excludes_analysis_before_metadata(self):
        session = live.Session("unused", Path("/synthetic/smol"))
        jobs = AnalyticsJobs(
            "unused",
            session.model,
            inference_busy=lambda: session.process is not None
            or session.status in ("loading", "running", "stopping"),
            fetch_model=lambda: (_ for _ in ()).throw(AssertionError("no source read")),
            available=lambda: 6 * live.GIB,
            reap=lambda p: False,
        )
        for state in ["loading", "running", "stopping"]:
            session.status = state
            with (
                self.subTest(state=state),
                self.assertRaisesRegex(ValueError, "compute busy"),
            ):
                jobs.start({})

    def test_single_pinned_source_binding_only(self):
        session = live.Session("unused", Path("/synthetic/smol"))
        valid = {
            "source_directory": str(session.model),
            "revision": live.MANIFEST["revision"],
        }
        self.assertEqual(
            live.bind_inference_source(valid, session)["inference_source_model"],
            live.SOURCE_MODEL,
        )
        for change in [
            {"source_directory": "/synthetic/qwen"},
            {"revision": "different"},
            {"comparison_identity": "pair"},
            {"coordinate_space": "checkpoint-comparison-v1"},
            {"inference_editable": False},
        ]:
            with self.subTest(change=change):
                self.assertNotIn(
                    "inference_source_model",
                    live.bind_inference_source(
                        {
                            **valid,
                            **change,
                            "inference_source_model": live.SOURCE_MODEL,
                        },
                        session,
                    ),
                )
        session.inference_enabled = False
        self.assertNotIn(
            "inference_source_model", live.bind_inference_source(valid, session)
        )

    def test_analytics_dtype_and_pair_refusal_before_source_read(self):
        metadata = {
            "source_directory": "/synthetic/source",
            "source_identity": "a" * 64,
            "catalog": [{"dtype": "F32"}],
        }
        jobs = AnalyticsJobs(
            "unused",
            Path("/synthetic/source"),
            inference_busy=lambda: False,
            fetch_model=lambda: metadata,
            available=lambda: 6 * live.GIB,
            reap=lambda p: True,
        )
        with self.assertRaisesRegex(ValueError, "all-BF16"):
            jobs._model()
        metadata["catalog"][0]["dtype"] = "F16"
        with self.assertRaisesRegex(ValueError, "all-BF16"):
            jobs._model()
        metadata["comparison_identity"] = "pair"
        with self.assertRaisesRegex(ValueError, "checkpoint comparison"):
            jobs._model()


if __name__ == "__main__":
    unittest.main()
