"""Snapshot and owner-lifecycle boundaries using existing in-memory OS doubles."""

import hashlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
import profile_worker_primitives as primitive
import profile_runtime_doubles as runtime
from atlas_host import profile_os, profile_service, profile_snapshot as snapshot
from atlas_host.profile_worker import ProfileJob
from atlas_host.supervisor import Supervisor


class SnapshotLimits(unittest.TestCase):
    def fixture(self):
        case = primitive.Base(methodName="runTest")
        case.setUp()
        self.addCleanup(case.doCleanups)
        return case

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

    def test_layout_axis_edges_and_reachable_overlap_boundary(self):
        self.assertEqual(
            (snapshot.MAX_STATE, snapshot.FIXED_RESERVE), (33554432, 2228224)
        )
        for axis in ("rows", "cols"):
            self.edges(
                lambda n: snapshot.layout(
                    primitive.selected(
                        **{axis: n, "cols" if axis == "rows" else "rows": 1}
                    )
                ),
                1,
                200000,
            )
        # Independent fixed vectors pin the closest reachable shape on this
        # 152-byte axis increment; exact odd-byte one-past is not reachable.
        frame, live, raw = snapshot.layout(primitive.selected(200000, 6086))
        self.assertEqual((frame, live, len(raw)), (9892623, 33554286, 327))
        with self.assertRaisesRegex(ValueError, "overlap and restore"):
            snapshot.layout(primitive.selected(200000, 6087))

    def test_layout_binding_serialization_cap_isolated(self):
        # Binding's own field caps mask this 16384-byte serializer guard.
        for size, accepted in ((0, False), (1, True), (16384, True), (16385, False)):
            with patch.object(snapshot, "canonical", return_value=b"x" * size):
                if accepted:
                    snapshot.layout(primitive.selected())
                else:
                    with self.assertRaisesRegex(ValueError, "binding too large"):
                        snapshot.layout(primitive.selected())
        self.edges(
            lambda seed: snapshot.profile_identity(primitive.selected(), seed),
            0,
            2**32 - 1,
        )

    def test_validation_visited_limits_and_exclusive_deadline(self):
        case = self.fixture()
        for visited, accepted in ((0, True), (12, True), (13, False)):
            raw, *_ = primitive.frame(
                primitive.selected(), visited=visited if accepted else 12
            )
            if not accepted:
                # Only the trusted fixture header's visited count is changed;
                # checksums/revision are computed from the resulting bytes.
                raw = bytearray(raw)
                import struct

                struct.pack_into("<Q", raw, 40, visited)
                raw = bytes(raw)
            if accepted:
                self.assertEqual(case.validate(raw).visited, visited)
            else:
                with self.assertRaises(ValueError):
                    case.validate(raw)
        raw, *_ = primitive.frame(primitive.selected())
        fd = case.os.put(raw)
        for now, accepted in ((math.nextafter(5.0, 0.0), True), (5.0, False)):
            if accepted:
                snapshot.validate(
                    fd,
                    primitive.selected(),
                    17,
                    hashlib.sha256(raw).hexdigest(),
                    deadline=5,
                    clock=lambda: now,
                )
            else:
                with self.assertRaisesRegex(ValueError, "deadline exhausted"):
                    snapshot.validate(
                        fd,
                        primitive.selected(),
                        17,
                        hashlib.sha256(raw).hexdigest(),
                        deadline=5,
                        clock=lambda: now,
                    )

    def store(self, rows=3, cols=4):
        case = self.fixture()
        store = snapshot.SnapshotStore(primitive.selected(rows, cols), 17)
        raw, *_ = primitive.frame(store.selected, visited=rows * cols)
        fd = store.begin()
        case.os.files[fd] = [raw, snapshot.SEALS]
        revision = hashlib.sha256(raw).hexdigest()
        store.publish(
            revision,
            minimum_visited=0,
            maximum_visited=rows * cols,
            deadline=5,
            source_check=lambda: None,
            final_check=lambda: None,
            clock=lambda: 0,
        )
        self.addCleanup(store.close)
        return store, revision

    def test_page_count_position_and_output_clamping(self):
        store, revision = self.store(1024, 1)
        self.edges(
            lambda n: store.page(revision, "rows", 0, n, source_check=lambda: None),
            1,
            1024,
        )
        self.assertEqual(
            len(
                store.page(revision, "rows", 0, 1024, source_check=lambda: None)[
                    "original"
                ]
            ),
            1024,
        )
        self.edges(
            lambda n: store.page(revision, "rows", n, 1, source_check=lambda: None),
            0,
            1023,
        )
        self.assertEqual(
            store.page(revision, "rows", 1023, 1024, source_check=lambda: None)["end"],
            1024,
        )
        self.edges(
            lambda n: store.page(revision, "columns", n, 1, source_check=lambda: None),
            0,
            0,
        )

    def test_page_output_guard_isolated_from_current_record_caps(self):
        store, revision = self.store()
        canonical = snapshot.canonical
        for size, accepted in ((2097152, True), (2097153, False)):

            def encoded(value):
                return (
                    b"x" * size
                    if isinstance(value, dict) and "original" in value
                    else canonical(value)
                )

            with patch.object(snapshot, "canonical", side_effect=encoded):
                if accepted:
                    store.page(revision, "rows", 0, 3, source_check=lambda: None)
                else:
                    with self.assertRaisesRegex(ValueError, "output cap"):
                        store.page(revision, "rows", 0, 3, source_check=lambda: None)

    def test_supervisor_frame_reservation_edges(self):
        def retain(size):
            supervisor = Supervisor()
            owner = object()
            supervisor.retain_profile(owner, size)
            self.assertEqual(supervisor.snapshot_reservation, 2 * size)
            supervisor.close_profile(owner, all_handles_closed=True)

        self.edges(retain, 1, 16777216)

    def test_job_start_effective_wall_cpu_values_and_tab_minimum(self):
        case = self.fixture()
        for field, low, high in (
            ("values", 1, 12),
            # Remaining integer milliseconds are floored after the 0.8-second
            # cleanup reserve, then must exceed 200. With binary64 arithmetic,
            # 1001 ms yields 200, so the effective zero-elapsed edge is 1002.
            ("wall_ms", 1002, 5000),
            ("cpu_ms", 1, 4000),
        ):

            def start(number):
                platform = primitive.Platform(case.os)
                job = ProfileJob(
                    primitive.selected(), 17, "t" * 32, "context", platform
                )
                options = {"values": 1, "wall_ms": 5000, "cpu_ms": 4000, field: number}
                job.start("t" * 32, job.job_capability, "context", **options)
                self.assertEqual(platform.calls, 1)

            self.edges(start, low, high)
        for length, accepted in ((31, False), (32, True), (33, True)):
            if accepted:
                ProfileJob(
                    primitive.selected(),
                    17,
                    "t" * length,
                    "context",
                    primitive.Platform(case.os),
                )
            else:
                with self.assertRaises(ValueError):
                    ProfileJob(
                        primitive.selected(),
                        17,
                        "t" * length,
                        "context",
                        primitive.Platform(case.os),
                    )

    def test_watch_slots_limit_and_nonrenewing_deadline(self):
        loop = profile_service.WatchdogLoop(SimpleNamespace())
        first = loop.bind(lambda: None, 5)
        second = loop.bind(lambda: None, 5)
        with self.assertRaisesRegex(ValueError, "Watchdog unavailable"):
            loop.bind(lambda: None, 5)
        first.register(lambda: None, 5)
        with self.assertRaisesRegex(ValueError, "cannot renew"):
            first.register(lambda: None, math.nextafter(5.0, math.inf))
        first.close()
        third = loop.bind(lambda: None, 5)
        second.close()
        third.close()
        self.assertEqual(loop.slots, [])

    def test_owned_thread_child_text_and_partial_receipt_caps(self):
        for count, accepted in ((0, True), (64, True), (65, False)):
            with (
                patch.object(
                    Path,
                    "iterdir",
                    return_value=iter([Path(f"/synthetic/{n}") for n in range(count)]),
                ),
                patch.object(Path, "read_text", return_value=""),
            ):
                if accepted:
                    self.assertEqual(profile_os.children(42), set())
                else:
                    with self.assertRaisesRegex(ValueError, "thread inventory"):
                        profile_os.children(42)
        for size, accepted in ((4096, True), (4097, False)):
            with (
                patch.object(
                    Path, "iterdir", return_value=iter([Path("/synthetic/task")])
                ),
                patch.object(Path, "read_text", return_value=" " * size),
            ):
                if accepted:
                    self.assertEqual(profile_os.children(42), set())
                else:
                    with self.assertRaisesRegex(ValueError, "child inventory"):
                        profile_os.children(42)
        for size, accepted in ((16384, True), (16385, False)):
            case = runtime.ProcessTests(methodName="runTest")
            child = case.make()
            child.process.stdout = SimpleNamespace(fileno=lambda: 123, close=Mock())
            child.eof, child.drainable = False, True
            child.read = Mock(return_value=b"x" * size)
            with patch.object(child, "stop") as stop:
                child._drain()
            self.assertEqual(child.invalid, not accepted)
            self.assertEqual(stop.call_count, int(not accepted))
            child.read.assert_called_once_with(123, 16385)


if __name__ == "__main__":
    unittest.main()
