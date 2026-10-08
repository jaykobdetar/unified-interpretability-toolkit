"""Pure native-to-host progress contracts; no native execution or model I/O."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
from atlas_host.profile_adapter import native_progress

BINDING = json.loads((ROOT / "docs/SOURCE-BINDING-V2.json").read_text())
MODEL = "m_" + "c" * 64


def native(**changes):
    result = {
        "schema": "weight-atlas.strength.v1",
        "identity": "d" * 64,
        "binding": deepcopy(BINDING),
        "state": "ready",
        "complete": False,
        "visited_values": 5,
        "total_values": 12,
        "authorized_remaining_values": 7,
        "active_seconds": 0.0121,
        "remaining_active_ms": 4987.9,
        "allocated_state_bytes": 17000,
        "error": None,
        "control": {"seed": 42, "algorithm": "swap-or-not-8-v1"},
    }
    return {**result, **changes}


class NativeProjection(unittest.TestCase):
    def project(self, value=None, **remaining):
        return native_progress(
            MODEL,
            BINDING,
            native() if value is None else value,
            {"values": 7, "wall_ms": 3000, "cpu_ms": 2500, **remaining},
        )

    def test_native_slice_binding_and_host_total_budgets_are_preserved(self):
        result = self.project(values=3)
        self.assertEqual(result["binding"], BINDING)
        self.assertEqual(result["total_values"], 12)  # one slice, not 24 native values
        self.assertEqual(result["elapsed_active_ms"], 13)
        self.assertEqual(
            result["remaining_authorized_work"],
            {"values": 3, "wall_ms": 3000, "cpu_ms": 2500},
        )

    def test_zero_host_budget_and_paused_core_cannot_authorize_continuation(self):
        for value, remaining in [
            (native(state="paused"), {}),
            (native(), {"cpu_ms": 0}),
            (native(), {"wall_ms": 0}),
            (native(), {"values": 0}),
        ]:
            result = self.project(value, **remaining)
            self.assertEqual(result["state"], "partial")
            self.assertFalse(result["complete"])
            self.assertEqual(result["error"], "budget_exhausted")
            self.assertFalse(any(result["remaining_authorized_work"].values()))

    def test_remaining_native_active_time_is_floored_not_reissued(self):
        result = self.project(native(remaining_active_ms=8.9))
        self.assertEqual(result["remaining_authorized_work"]["wall_ms"], 8)

    def test_complete_is_only_full_slice_and_freezes_remaining_work(self):
        result = self.project(
            native(
                state="complete",
                complete=True,
                visited_values=12,
                authorized_remaining_values=0,
            )
        )
        self.assertTrue(result["complete"])
        self.assertFalse(any(result["remaining_authorized_work"].values()))
        with self.assertRaises(ValueError):
            self.project(native(state="complete", complete=True))
        expired = self.project(
            native(
                state="complete",
                complete=True,
                visited_values=12,
                authorized_remaining_values=0,
            ),
            cpu_ms=0,
        )
        self.assertFalse(expired["complete"])
        self.assertEqual(expired["error"], "resource_limit")

    def test_accepted_binding_and_head_descriptor_match_both_lanes(self):
        for host, data in [
            (
                "tests/fixtures/host-source-binding-v2.json",
                "docs/SOURCE-BINDING-V2.json",
            ),
            (
                "tests/fixtures/host-head-layout-v1.json",
                "tests/fixtures/smollm2-head-layout-v1.json",
            ),
        ]:
            self.assertEqual(
                json.loads((ROOT / host).read_text()),
                json.loads((ROOT / data).read_text()),
            )

    def test_raw_native_error_never_enters_visitor_progress(self):
        result = self.project(
            native(state="invalid", error="/private/owner/source error")
        )
        self.assertEqual(result["state"], "error")
        self.assertEqual(result["error"], "worker_error")
        self.assertNotIn("/private", json.dumps(result))

    def test_foreign_slice_unknown_control_large_seed_and_changed_caps_refuse(self):
        for change in [
            {
                "binding": {
                    **BINDING,
                    "slice": {"leading_indices": [0], "display_axes": [1, 2]},
                }
            },
            {"control": {"seed": 2**32, "algorithm": "swap-or-not-8-v1"}},
            {"control": {"seed": 1, "algorithm": "unknown"}},
            {"active_seconds": float("nan")},
            {"total_values": 24},
            {"allocated_state_bytes": 32 * 1024**2 + 1},
            {"state": "resuming"},
        ]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.project(native(**change))
        with self.assertRaises(ValueError):
            self.project(wall_ms=5001)


if __name__ == "__main__":
    unittest.main(verbosity=2)
