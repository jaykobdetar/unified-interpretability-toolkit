"""Profile protocol and progress boundary tables; no executor, process or listener."""

from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
import host_contracts
import host_profile_adapter
from atlas_host.common import canonical
from atlas_host.profile_adapter import native_progress
from atlas_host.profile_api import PrivateProfileAPI
from atlas_host import progress


class ProfileLimits(unittest.TestCase):
    def edges(self, operation, low, high):
        for number, accepted in (
            (low - 1, False),
            (low, True),
            (high, True),
            (high + 1, False),
        ):
            with self.subTest(number=number):
                if accepted:
                    operation(number)
                else:
                    with self.assertRaises(ValueError):
                        operation(number)

    def project(self, changes=None, remaining=None):
        return native_progress(
            host_profile_adapter.MODEL,
            host_profile_adapter.BINDING,
            host_profile_adapter.native(**(changes or {})),
            {"values": 7, "wall_ms": 3000, "cpu_ms": 2500, **(remaining or {})},
        )

    def test_native_counts_allocated_state_and_control_seed(self):
        self.edges(
            lambda n: self.project(
                {"visited_values": n, "authorized_remaining_values": min(7, 12 - n)}
            ),
            0,
            12,
        )
        self.edges(lambda n: self.project({"authorized_remaining_values": n}), 0, 7)
        self.edges(lambda n: self.project({"allocated_state_bytes": n}), 0, 33554432)
        self.edges(
            lambda n: self.project(
                {"control": {"seed": n, "algorithm": "swap-or-not-8-v1"}}
            ),
            0,
            2**32 - 1,
        )
        self.edges(lambda n: self.project({"total_values": n}), 12, 12)

    def test_native_finite_time_bounds_and_rounding(self):
        high = 2**53 / 1000
        for field in ("active_seconds", "remaining_active_ms"):
            for number, accepted in (
                (math.nextafter(0.0, -math.inf), False),
                (0.0, True),
                (high, True),
                (math.nextafter(high, math.inf), False),
                (math.inf, False),
                (math.nan, False),
            ):
                if accepted:
                    self.project({field: number})
                else:
                    with self.assertRaises(ValueError):
                        self.project({field: number})
        self.assertEqual(
            self.project({"active_seconds": 0.0011})["elapsed_active_ms"], 2
        )
        self.assertEqual(
            self.project({"remaining_active_ms": 1.9})["remaining_authorized_work"][
                "wall_ms"
            ],
            1,
        )

    def test_native_host_remaining_work_limits(self):
        for field, high in (("values", 2**63 - 1), ("wall_ms", 5000), ("cpu_ms", 4000)):
            self.edges(lambda n: self.project(remaining={field: n}), 0, high)

    def test_native_real_encoded_status_byte_cap(self):
        value = host_profile_adapter.native(padding="")
        base = len(canonical(value))
        for size, accepted in ((16384, True), (16385, False)):
            changes = {"padding": "x" * (size - base)}
            self.assertEqual(
                len(canonical(host_profile_adapter.native(**changes))), size
            )
            if accepted:
                self.project(changes)
            else:
                with self.assertRaisesRegex(ValueError, "status bound"):
                    self.project(changes)

    def api(self, result=None):
        service = SimpleNamespace(
            **{
                action: Mock(return_value={} if result is None else result)
                for action in (
                    "start",
                    "status",
                    "page",
                    "cancel",
                    "heartbeat",
                    "reconcile",
                )
            }
        )
        return PrivateProfileAPI(service, Mock()), service

    def request(self, action):
        data = {
            "version": 1,
            "model_id": host_contracts.MODEL_ID,
            "context_id": "fixture",
            "tab_capability": "a" * 64,
        }
        if action == "start":
            data.update(
                binding=host_contracts.binding_value(), seed=1, values=1, restart=False
            )
        elif action != "reconcile":
            data.update(job_id="b" * 32, job_capability="c" * 64)
            if action == "page":
                data.update(revision="d" * 64, axis="rows", start=0, count=1)
        return data

    def test_api_start_version_seed_and_value_edges(self):
        api, _ = self.api()
        for field, low, high in (
            ("version", 1, 1),
            ("seed", 0, 2**32 - 1),
            ("values", 1, 6),
        ):
            self.edges(
                lambda n: api.handle(
                    "start",
                    {**self.request("start"), field: n},
                    admission=SimpleNamespace(remaining=lambda: {}),
                ),
                low,
                high,
            )

    def test_api_owner_page_and_identity_lengths(self):
        api, _ = self.api()
        for field, low, high in (("start", 0, 2**63 - 1), ("count", 1, 1024)):
            self.edges(
                lambda n: api.handle("page", {**self.request("page"), field: n}),
                low,
                high,
            )
        for field, length in (
            ("job_id", 32),
            ("job_capability", 64),
            ("tab_capability", 64),
            ("revision", 64),
        ):
            self.edges(
                lambda n: api.handle("page", {**self.request("page"), field: "e" * n}),
                length,
                length,
            )
        self.edges(
            lambda n: api.handle(
                "page", {**self.request("page"), "model_id": "m_" + "e" * n}
            ),
            64,
            64,
        )

    def test_api_real_request_body_utf8_byte_edges(self):
        api, _ = self.api()
        for size, accepted in ((8192, True), (8193, False)):
            data = self.request("reconcile")
            data["context_id"] = ""
            available = size - len(canonical(data))
            data["context_id"] = "é" * (available // 2) + "x" * (available % 2)
            self.assertEqual(len(canonical(data)), size)
            if accepted:
                api.handle("reconcile", data)
            else:
                with self.assertRaisesRegex(ValueError, "body limit"):
                    api.handle("reconcile", data)

    def test_api_response_bytes_for_all_routes(self):
        base = len(canonical({"padding": ""}))
        for action in ("start", "status", "page", "cancel", "heartbeat", "reconcile"):
            high = 2097152 if action == "page" else 16384
            for size, accepted in ((high, True), (high + 1, False)):
                result = {"padding": "x" * (size - base)}
                self.assertEqual(len(canonical(result)), size)
                api, service = self.api(result)
                kwargs = (
                    {"admission": SimpleNamespace(remaining=lambda: {})}
                    if action == "start"
                    else {}
                )
                if accepted:
                    code, returned, headers = api.handle(
                        action, self.request(action), **kwargs
                    )
                    self.assertEqual(code, 202 if action == "start" else 200)
                    self.assertEqual(returned, result)
                    self.assertEqual(headers, {"Cache-Control": "no-store"})
                else:
                    with self.assertRaisesRegex(ValueError, "response exceeds bound"):
                        api.handle(action, self.request(action), **kwargs)
                getattr(service, action).assert_called_once()

    def test_progress_declared_integer_ranges_and_total_coupling(self):
        for field, low, high in (
            ("visited_values", 0, 6),
            ("total_values", 6, 6),
            ("elapsed_active_ms", 0, 2**63 - 1),
        ):

            def validate(n):
                record = host_contracts.record(**{field: n})
                if field == "visited_values":
                    record["remaining_authorized_work"]["values"] = min(4, 6 - n)
                return progress.validate_progress(record)

            self.edges(validate, low, high)
        for field, high in (
            ("values", 4),
            ("wall_ms", 2**63 - 1),
            ("cpu_ms", 2**63 - 1),
        ):

            def validate_work(n):
                record = host_contracts.record()
                record["remaining_authorized_work"][field] = n
                return progress.validate_progress(record)

            self.edges(validate_work, 0, high)

    def test_progress_key_owner_lengths_and_capacity_behaviour(self):
        self.assertEqual(
            (
                progress.ProgressStore.MAX_RECORDS,
                progress.ProgressStore.MAX_PAGE,
                progress.ProgressStore.MAX_BYTES,
            ),
            (64, 16, 16384),
        )
        for field, length in (("key", 32), ("owner", 64)):

            def publish(n):
                store = progress.ProgressStore(host_contracts.MODEL_ID)
                key, owner = ("e" * n, None) if field == "key" else ("e" * 32, "f" * n)
                return store.publish(key, host_contracts.record(), owner=owner)

            self.edges(publish, length, length)
        public = progress.ProgressStore(host_contracts.MODEL_ID)
        for n in range(65):
            public.publish(f"{n:032x}", host_contracts.record())
        self.assertEqual(len(public._records), 64)
        self.assertNotIn("0" * 32, public._records)
        self.assertEqual(public.floor, 1)
        private = progress.ProgressStore(host_contracts.MODEL_ID)
        for n in range(64):
            private.publish(
                f"{n:032x}", host_contracts.record(kind="profile"), owner="f" * 64
            )
        with self.assertRaisesRegex(ValueError, "capacity reserved"):
            private.publish(
                f"{64:032x}", host_contracts.record(kind="profile"), owner="f" * 64
            )
        self.assertEqual(len(private._records), 64)

    def test_progress_page_count_and_cursor_floor_edges(self):
        for count in (15, 16, 17):
            store = progress.ProgressStore(host_contracts.MODEL_ID)
            for n in range(count):
                store.publish(f"{n:032x}", host_contracts.record())
            result = store.snapshot()
            self.assertEqual(len(result["records"]), min(count, 16))
            self.assertEqual(result["has_more"], count > 16)
        store = progress.ProgressStore(host_contracts.MODEL_ID)
        for n in range(65):
            store.publish(f"{n:032x}", host_contracts.record())
        for revision, reset in (
            (store.floor - 1, True),
            (store.floor, False),
            (store.revision, False),
            (store.revision + 1, True),
        ):
            result = store.snapshot(
                {"epoch": store.epoch, "revision": revision, "scope": "public"}
            )
            self.assertEqual(result["reset"], reset)


if __name__ == "__main__":
    unittest.main()
