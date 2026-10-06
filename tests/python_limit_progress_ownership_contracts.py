"""Actual progress packing and pure ownership-inventory boundary contracts."""

from copy import deepcopy
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import threading
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import host_contracts as fixture
import profile_runtime_doubles as runtime
from atlas_host import profile_os, profile_service, progress
from atlas_host.common import canonical
from atlas_host.supervisor import Supervisor


class ProgressOwnershipLimits(unittest.TestCase):
    def packed(self, target):
        for count in range(1, 17):
            records = []
            for index in range(count):
                value = fixture.record()
                value["binding"]["name"] = "x"
                records.append({"id": f"{index:032x}", "revision": index + 1, **value})
            trial = {
                "version": 1,
                "model_id": fixture.MODEL_ID,
                "reset": True,
                "records": records,
                "has_more": False,
                "cursor": {"epoch": "a" * 32, "revision": count, "scope": "public"},
            }
            missing = target - len(canonical(trial))
            if not 0 <= missing <= 511 * count:
                continue
            for record in records:
                added = min(missing, 511)
                record["binding"]["name"] += "x" * added
                missing -= added
            self.assertEqual(len(canonical(trial)), target)
            store = progress.ProgressStore(fixture.MODEL_ID)
            store.epoch = "a" * 32
            for item in records:
                value = {
                    key: value
                    for key, value in item.items()
                    if key not in ("id", "revision")
                }
                store.publish(item["id"], value)
            return store, count
        self.fail("No legal record packing reaches the independent target")

    def test_actual_trial_byte_cap_exact_and_one_past(self):
        for size, all_records in ((16256, True), (16257, False)):
            store, count = self.packed(size)
            result = store.snapshot()
            self.assertEqual(result["has_more"], not all_records)
            self.assertEqual(
                len(result["records"]), count if all_records else count - 1
            )
            self.assertLessEqual(len(canonical(result)), 16384)
            if all_records:
                self.assertEqual(len(canonical(result)), 16256)
            else:
                cursor = result["cursor"]
                following = store.snapshot(cursor)
                self.assertEqual(len(following["records"]), 1)
                self.assertFalse(following["has_more"])

    def test_final_envelope_byte_guard_isolated(self):
        # The final guard is separate from the 128-byte trial reserve. A
        # selective serializer double isolates it without changing the latter.
        for size, accepted in ((16384, True), (16385, False)):
            store = progress.ProgressStore(fixture.MODEL_ID)
            with patch.object(progress, "canonical", return_value=b"x" * size):
                if accepted:
                    store.snapshot()
                else:
                    with self.assertRaisesRegex(ValueError, "response exceeds"):
                        store.snapshot()

    def test_reconciliation_zero_one_64_and_65_predecessors(self):
        data = {
            "model_id": "m_" + "a" * 64,
            "context_id": "ctx",
            "tab_capability": "d" * 64,
        }
        for count, accepted in ((0, True), (1, True), (64, True), (65, False)):
            service = object.__new__(profile_service.ProfileService)
            service.lock = threading.RLock()
            service.supervisor = Supervisor()
            service.context = runtime.Context()
            service.record = None
            service._cancel = Mock()
            for _ in range(count):
                service.record = {
                    "previous": service.record,
                    "token": object(),
                    "runtime": None,
                    "model_id": data["model_id"],
                    "context_id": "ctx",
                    "tab": "d" * 64,
                    "pending": False,
                    "watch": NS(closed=True),
                    "status": {"state": "cancelled"},
                }
            if accepted:
                result = service.reconcile(data)
                self.assertTrue(result["no_owned_work"])
            else:
                with self.assertRaisesRegex(ValueError, "inventory unavailable"):
                    service.reconcile(data)
                service._cancel.assert_not_called()

    def test_process_inventory_admission_before_inert_popen(self):
        for count, accepted in ((0, True), (3, True), (4, False), (5, False)):
            book = profile_os.ProcessBook(
                stats=lambda pid: {"start": 1, "cpu": 0, "rss": 0},
                descendants=lambda pid: set(),
                available=lambda: 6 * 1024**3,
            )
            book.owned = [
                NS(reaped=False, streams_closed=True, unexpected=set())
                for _ in range(count)
            ]
            with patch.object(
                profile_os.subprocess,
                "Popen",
                side_effect=RuntimeError("Reached inert spawn boundary"),
            ) as spawn:
                if accepted:
                    with self.assertRaisesRegex(RuntimeError, "inert spawn"):
                        book.spawn(["inert"])
                    spawn.assert_called_once()
                else:
                    with self.assertRaisesRegex(ValueError, "inventory exceeds"):
                        book.spawn(["inert"])
                    spawn.assert_not_called()
            # Admitted control intentionally stops at Popen; no process, signal
            # or registration is created, and no success handle is claimed.


if __name__ == "__main__":
    unittest.main()
