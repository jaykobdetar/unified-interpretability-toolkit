"""Deterministic Popen/register interleavings. No processes or real clocks."""

import importlib.util
import os
from pathlib import Path
import sys, threading, types, unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from atlas_host.profile_os import ProcessBook
from atlas_host.profile_platform import ProfilePlatform
from atlas_host.profile_observation import SpawnObservationPending
from atlas_host.supervisor import Supervisor
from profile_host_supervisor import Clock
import profile_worker_primitives as primitives

if os.environ.get("ATLAS_TEST_OLD_PROCESS_BOOK"):
    spec = importlib.util.spec_from_file_location(
        "atlas_host.old_process_book", os.environ["ATLAS_TEST_OLD_PROCESS_BOOK"]
    )
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    ProcessBook = old.ProcessBook


class SpawnRegistration(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.clock = Clock()
        self.rss = 4096
        self.extra = set()
        self.child_descendants = set()
        self.book = ProcessBook(
            stats=lambda pid: {
                "start": 1,
                "cpu": 0.0,
                "rss": 1024 if pid == os.getpid() else self.rss,
            },
            descendants=lambda pid: (
                ({42} | self.extra) if pid == os.getpid() else self.child_descendants
            ),
            available=lambda: 6 * 1024**3,
        )
        self.book.clock = lambda: self.now
        self.book.spawn_started = 0.0
        self.book.spawning = True
        self.supervisor = Supervisor()
        self.token = self.supervisor.acquire("profile", "ctx")
        self.hooks = types.SimpleNamespace(
            resources=self.book.resources,
            watch=lambda callback, deadline: setattr(self, "guard", callback)
            or types.SimpleNamespace(close=lambda: None),
        )
        self.platform = ProfilePlatform(
            self.supervisor, self.token, self.clock.grant(), self.hooks
        )
        self.job = types.SimpleNamespace(
            store=types.SimpleNamespace(owned_storage_bytes=860),
            expired=False,
            cancel_event=threading.Event(),
            guard=lambda: setattr(self, "called_job_guard", True),
        )
        self.supervisor.snapshot_reservation = 1720
        self.called_job_guard = False
        self.platform.arm_watchdog(self.job)

    def test_candidate_rss_is_counted_without_claiming_registration_or_adopting(self):
        with self.assertRaises(SpawnObservationPending) as captured:
            self.book.resources()
        self.assertEqual(
            captured.exception.observed,
            {
                "rss_bytes": 5120,
                "available_bytes": 6 * 1024**3,
                "all_owned_accounted": False,
                "descendants_clear": True,
            },
        )
        self.assertEqual(self.book.owned, [])
        self.assertTrue(self.supervisor.busy())

    def test_watchdog_keeps_grant_unpoisoned_until_registration_then_checks_normally(
        self,
    ):
        self.guard()
        self.assertFalse(self.job.expired)
        self.assertFalse(self.job.cancel_event.is_set())
        self.assertTrue(self.token.current())
        self.assertFalse(self.called_job_guard)
        self.book.owned = [
            types.SimpleNamespace(
                pid=42, start=1, reaped=False, unexpected=set(), lock=threading.RLock()
            )
        ]
        self.book.spawning = False
        self.guard()
        self.assertTrue(self.called_job_guard)
        self.assertTrue(self.token.current())
        self.assertTrue(self.platform.resources_ok(860))
        self.assertTrue(self.supervisor.busy())

    def test_registration_stall_refuses_after_one_watch_interval(self):
        self.now = 0.05
        self.guard()
        self.assertTrue(self.job.expired)
        self.assertTrue(self.token.poisoned)
        self.assertTrue(self.supervisor.busy())
        self.assertFalse(self.book.settled())

    def test_original_grant_deadline_is_not_renewed_by_pending_spawn(self):
        self.clock.now = 6
        self.guard()
        self.assertTrue(self.job.expired)
        self.assertTrue(self.token.poisoned)

    def test_real_rss_plus_reserved_snapshot_limit_is_never_deferred(self):
        self.rss = 768 * 1024**2 - 1024
        self.guard()
        self.assertTrue(self.job.expired)
        self.assertTrue(self.token.poisoned)

    def test_low_headroom_is_never_deferred(self):
        self.book.available = lambda: 3 * 1024**3
        self.guard()
        self.assertTrue(self.job.expired)

    def test_second_unregistered_child_or_descendant_still_refuses(self):
        self.extra = {43}
        self.guard()
        self.assertTrue(self.job.expired)
        self.setUp()
        self.child_descendants = {44}
        self.guard()
        self.assertTrue(self.job.expired)

    def test_unregistered_child_outside_spawn_phase_still_refuses(self):
        self.book.spawning = False
        self.guard()
        self.assertTrue(self.job.expired)

    def test_pending_state_cannot_be_used_as_complete_owner_observation(self):
        with self.assertRaises(SpawnObservationPending):
            self.platform.resources_ok(860)
        self.assertFalse(self.book.settled())
        self.assertTrue(self.supervisor.busy())


class PrimitivePending(unittest.TestCase):
    def make(self):
        case = primitives.LifecycleTests(
            "test_terminal_only_after_reap_validation_no_autochain"
        )
        case.setUp()
        self.addCleanup(case.doCleanups)
        return case.job()

    def test_primitive_watchdog_pending_does_not_poison_healthy_job_or_publish(self):
        platform, job, _ = self.make()

        def pending(_):
            raise SpawnObservationPending({})

        platform.resources_ok = pending
        job.guard()
        self.assertFalse(job.expired)
        self.assertFalse(job.cancel_event.is_set())
        self.assertIsNone(job.accepted)
        self.assertEqual(job.state, "running")

    def test_primitive_pending_still_checks_original_time_budget(self):
        platform, job, _ = self.make()

        def pending(_):
            raise SpawnObservationPending({})

        platform.resources_ok = pending
        platform.now = 6
        job.guard()
        self.assertTrue(job.expired)
        self.assertTrue(job.cancel_event.is_set())
        self.assertIsNone(job.accepted)


if __name__ == "__main__":
    unittest.main(verbosity=2)
