"""Review boundary regressions with synthetic data and existing process doubles."""

from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import inference_observations as observation
import live_inference as live
from inference_model_descriptor import describe_config
from atlas_host import acquisition
import model_descriptor_contracts
import observation_contracts
import profile_runtime_doubles


class ReviewGaps(unittest.TestCase):
    def test_unknown_model_family_retains_explicit_refusal_contract(self):
        config = model_descriptor_contracts.config()
        self.assertEqual(describe_config(config)["model_type"], "qwen3")
        try:
            describe_config({**config, "model_type": "qwen3_moe"})
        except Exception as error:
            self.assertIs(type(error), ValueError)
            self.assertEqual(str(error), "Unsupported explicit built-in architecture")
        else:
            self.fail("Unknown model family was accepted")

    def test_generation_boundaries_refuse_before_model_access(self):
        valid = {
            "prompt": "x" * 4096,
            "max_new_tokens": 32,
            "layer": live.LAYERS - 1,
        }
        session = live.Session("unused", Path("/unused"))
        with (
            patch.object(live, "available", return_value=6 * live.GIB),
            patch.object(
                live, "verify_model", side_effect=ValueError("Pinned check")
            ) as model_check,
            patch("subprocess.Popen", side_effect=AssertionError("No worker")),
        ):
            with self.assertRaisesRegex(ValueError, "^Pinned check$"):
                session.start(valid)
            model_check.assert_called_once_with(Path("/unused"))
            for field, value in (
                ("prompt", "x" * 4097),
                ("max_new_tokens", 33),
                ("layer", live.LAYERS),
            ):
                model_check.reset_mock()
                with self.subTest(field=field), self.assertRaises(ValueError) as error:
                    session.start({**valid, field: value})
                self.assertNotEqual(str(error.exception), "Pinned check")
                model_check.assert_not_called()
                self.assertIsNone(session.process)

    def test_admission_refusal_precedes_model_access(self):
        session = live.Session("unused", Path("/unused"))
        with (
            patch.object(live, "available", return_value=4 * live.GIB),
            patch.object(
                live, "verify_model", side_effect=ValueError("Pinned check")
            ) as model_check,
            patch("subprocess.Popen", side_effect=AssertionError("No worker")),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "^Need 4.75 GiB available RAM before loading the inference model$",
            ):
                session.start({"prompt": "fixture"})
            model_check.assert_not_called()
            self.assertIsNone(session.process)

    def test_observation_position_accepts_last_and_refuses_one_past(self):
        record = observation_contracts.lens()
        record["position"] = record["logit_lens"]["position"] = 158
        selected = {"kind": "logit_lens"}
        observation.validate_record(record, selected, 29)
        one_past = deepcopy(record)
        one_past["position"] = one_past["logit_lens"]["position"] = 159
        with self.assertRaisesRegex(ValueError, "Invalid observation site/position"):
            observation.validate_record(one_past, selected, 29)

    def test_acquisition_plan_refuses_one_byte_over_budget_before_io(self):
        data = {"model.safetensors": b"synthetic data", "LICENSE": b"fixture license"}
        manifest = {
            "version": 1,
            "repository": "fixtures/tiny",
            "revision": "a" * 40,
            "provenance": "owner_expected",
            "license": {"id": "Apache-2.0", "accepted": True, "file": "LICENSE"},
            "files": [
                {
                    "name": name,
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
                for name, raw in data.items()
            ],
        }
        total = sum(map(len, data.values()))
        with patch.object(
            acquisition, "open_public_data", side_effect=AssertionError("No network")
        ) as fetch:
            accepted = acquisition.plan(manifest, "Tiny", max_bytes=total)
            self.assertEqual(accepted["payload_bytes"], total)
            with self.assertRaisesRegex(
                ValueError, "Pinned files exceed explicit acquisition byte budget"
            ):
                acquisition.plan(manifest, "Tiny", max_bytes=total - 1)
            fetch.assert_not_called()

    def test_expired_profile_heartbeat_cannot_renew_owner_lease(self):
        fixture = profile_runtime_doubles.ServiceTests()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        grant, _ = fixture.start()
        deadline = grant.deadline
        fixture.clock.now = 1
        fixture.api.handle("heartbeat", fixture.owner)
        self.assertEqual(fixture.service.record["lease"], 16)
        self.assertEqual(grant.deadline, deadline)
        fixture.clock.now = 16
        with self.assertRaisesRegex(ValueError, "Owner lease expired"):
            fixture.api.handle("heartbeat", fixture.owner)
        self.assertEqual(fixture.service.record["lease"], 16)
        self.assertEqual(grant.deadline, deadline)

    def test_worker_rss_limit_accepts_boundary_and_reaps_one_kib_over(self):
        cap_kib = 1536 * 1024
        for rss_kib in (cap_kib, cap_kib + 1):
            with self.subTest(rss_kib=rss_kib):
                session = live.Session("unused", Path("/unused"))
                stream = types.SimpleNamespace(fileno=lambda: 42, close=lambda: None)
                worker = types.SimpleNamespace(
                    pid=424242, stdin=None, stdout=stream, poll=lambda: None
                )
                session.process, session.status = worker, "running"
                session.last_seen = session.started = 100

                def status(path, *args, **kwargs):
                    self.assertEqual(str(path), "/proc/424242/status")
                    return f"VmRSS:\t{rss_kib} kB\n"

                with (
                    patch.object(live, "available", return_value=6 * live.GIB),
                    patch.object(live.time, "monotonic", return_value=100),
                    patch.object(Path, "read_text", status),
                    patch.object(live.os, "read", side_effect=BlockingIOError),
                    patch.object(live, "signal_and_reap", return_value=True) as reap,
                    patch("subprocess.Popen", side_effect=AssertionError("No worker")),
                ):
                    try:
                        session.tick()
                        if rss_kib == cap_kib:
                            self.assertIs(session.process, worker)
                            self.assertEqual(session.status, "running")
                            reap.assert_not_called()
                        else:
                            self.assertIsNone(session.process)
                            self.assertEqual(session.status, "resource_limit")
                            self.assertEqual(session.details["error"], "resource_limit")
                            reap.assert_called_once_with(worker)
                    finally:
                        # The simulated process must never reach real OS cleanup.
                        session.process = None


if __name__ == "__main__":
    unittest.main()
