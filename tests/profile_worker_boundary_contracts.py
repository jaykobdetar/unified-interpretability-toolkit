"""Profile worker boundary contracts using existing in-memory ownership doubles."""

from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from atlas_host import profile_worker as worker
import profile_worker_primitives as primitive


class ProfileWorkerBoundaries(unittest.TestCase):
    def success(self, operation):
        try:
            return operation()
        except Exception as error:
            self.fail(f"Expected successful in-memory worker operation: {error!r}")

    def refuses(self, operation, message):
        try:
            operation()
        except ValueError as error:
            self.assertEqual(str(error), message)
        except Exception as error:
            self.fail(f"Expected exact ValueError, received {error!r}")
        else:
            self.fail("Expected exact ValueError")

    def idle(self):
        case = primitive.Base("runTest")
        case.setUp()
        self.addCleanup(case.doCleanups)
        platform = primitive.Platform(case.os)
        with patch.object(
            worker.secrets, "token_hex", return_value="f" * 64
        ) as entropy:
            job = self.success(
                lambda: worker.ProfileJob(
                    primitive.selected(), 17, "t" * 32, "context", platform
                )
            )
        entropy.assert_called_once_with(32)
        return case, platform, job, ("t" * 32, "f" * 64, "context")

    def started(self, values=5):
        case, platform, job, auth = self.idle()
        platform.spawn = Mock(wraps=platform.spawn)
        reply = self.success(lambda: job.start(*auth, values))
        self.assertIsInstance(reply, dict)
        return case, platform, job, auth, reply

    def finished(self, values=5):
        case, platform, job, auth, _reply = self.started(values)
        platform.child.reaped = True
        signals = []
        for _ in range(20):
            signals.append(self.success(job.tick))
            if job.state in ("complete", "partial", "error", "cancelled"):
                break
        self.assertIn(job.state, ("complete", "partial"))
        return case, platform, job, auth, signals

    def test_constructor_retains_owner_and_initial_state(self):
        _case, platform, job, _auth = self.idle()
        self.assertEqual(
            (job.tab, job.context, job.job_capability, job.state, job.error),
            ("t" * 32, "context", "f" * 64, "idle", None),
        )
        self.assertIs(job.platform, platform)
        self.assertEqual(job.store.selected, primitive.selected())
        self.assertEqual(
            (
                job.child,
                job.token,
                job.watchdog,
                job.validation,
                job.owner_thread,
                job.cpu_child,
                job.receipt,
                job.accepted,
                job._accepted_snapshot,
                job.expired,
                job.lease_end,
            ),
            (None, None, None, None, None, 0.0, None, None, None, False, 15.0),
        )
        self.assertFalse(job.cancel_event.is_set())

    def test_owner_checks_preserve_valid_returns_and_refusal_text(self):
        _case, _platform, job, auth = self.idle()
        self.assertIsNone(self.success(lambda: job._owns(*auth)))
        self.refuses(
            lambda: job._owns(auth[0], auth[1], "different-context"),
            "Private profile owner mismatch",
        )
        job.owner_thread = threading.get_ident()
        self.assertIsNone(self.success(job._owner_context))
        job.owner_thread += 1
        self.refuses(job._owner_context, "Wrong job-owning execution context")

    def test_cpu_clock_keeps_owner_delta_and_child_sum(self):
        _case, platform, job, _auth = self.idle()
        job.cpu_start, platform.cpu, job.cpu_child = 1.0, 5.0, 2.0
        self.assertEqual(self.success(job._cpu), 6.0)
        platform.cpu = 0.5
        self.refuses(job._cpu, "Invalid job-local owner CPU clock")

    def test_time_and_slot_budgets_keep_reserve_and_lease_boundaries(self):
        _case, platform, job, _auth, _reply = self.started()
        job.deadline, job.lease_end, platform.now = 10.0, 20.0, 8.5
        self.assertIsNone(self.success(lambda: job._time_budget(1.0)))
        job.deadline, platform.now = 30.0, 19.5
        self.assertIsNone(self.success(job._time_budget))
        job.deadline, platform.now = 5.0, 5.0
        self.refuses(job._time_budget, "Total profile grant exhausted")
        platform.now = 3.5
        self.assertIsNone(self.success(lambda: job._budget(0.8)))
        platform.token.held = False
        self.refuses(job._budget, "Shared heavy-slot ownership lost")

    def test_start_request_and_initial_public_status_are_exact(self):
        _case, platform, job, _auth, reply = self.started()
        platform.spawn.assert_called_once()
        request, output, input_fd = platform.spawn.call_args.args
        self.assertEqual(
            request,
            dict(
                binding=primitive.selected(),
                seed=17,
                values=5,
                wall_ms=4200,
                cpu_ms=4000,
            ),
        )
        self.assertEqual(output, job.store.pending)
        self.assertIsNone(input_fd)
        self.assertEqual(
            reply,
            dict(
                state="running",
                error=None,
                accepted=None,
                cleanup_pending=False,
                resume_available=False,
            ),
        )
        self.assertEqual(
            (
                job.started,
                job.deadline,
                job.cpu_start,
                job.cpu_budget,
                job.values,
                platform.calls,
            ),
            (0.0, 5.0, 0.0, 4.0, 5, 1),
        )
        self.assertIs(job.child, platform.child)
        self.assertIs(job.token, platform.token)

    def test_guard_samples_cpu_and_signals_local_resource_failure(self):
        _case, platform, job, _auth, _reply = self.started()
        platform.child.cpu = 0.25
        self.assertIsNone(self.success(job.guard))
        self.assertEqual(job.cpu_child, 0.25)
        self.assertEqual(job.sampled["cpu_seconds"], 0.25)
        platform.valid_resources = False
        self.assertIsNone(self.success(job.guard))
        self.assertTrue(job.expired)
        self.assertTrue(job.cancel_event.is_set())
        self.assertTrue(platform.child.stopped)

    def test_completed_guard_does_not_sample_or_expire_terminal_work(self):
        _case, platform, job, _auth, _signals = self.finished(12)
        platform.valid_resources = False
        self.assertIsNone(self.success(job.guard))
        self.assertFalse(job.expired)
        self.assertFalse(job.cancel_event.is_set())
        self.assertEqual(job.state, "complete")

    def test_sample_retains_record_identity_and_exact_cpu(self):
        _case, _platform, job, _auth = self.idle()
        sample = dict(cpu_seconds=0.75, reaped=False, exit_code=None, receipt=None)
        self.assertIsNone(self.success(lambda: job._sample(sample)))
        self.assertEqual(job.cpu_child, 0.75)
        self.assertIs(job.sampled, sample)
        self.refuses(
            lambda: job._sample({**sample, "cpu_seconds": 0.25}),
            "Owned child CPU clock regressed",
        )

    def test_cancel_only_signals_owned_work(self):
        _case, platform, job, auth, _reply = self.started()
        self.assertIsNone(self.success(lambda: job.cancel(*auth)))
        self.assertTrue(job.cancel_event.is_set())
        self.assertTrue(platform.child.stopped)
        self.assertTrue(platform.token.held)
        self.assertIs(job.child, platform.child)

    def test_heartbeat_changes_only_lease_and_keeps_exact_expiry(self):
        _case, platform, job, auth, _reply = self.started()
        original = job.deadline, job.cpu_budget, job.values
        platform.now = 2.0
        self.assertIsNone(self.success(lambda: job.heartbeat(*auth)))
        self.assertEqual(job.lease_end, 17.0)
        self.assertEqual((job.deadline, job.cpu_budget, job.values), original)
        platform.now = job.lease_end
        self.refuses(lambda: job.heartbeat(*auth), "Owner lease expired")

    def test_status_copies_accepted_values_and_pins_cleanup_fields(self):
        _case, _platform, job, auth = self.idle()
        job.state, job.error = "stopping", "example"
        job.accepted = dict(
            revision="a" * 64, visited_values=1, total_values=12, complete=False
        )
        status = self.success(lambda: job.status(*auth))
        self.assertEqual(
            status,
            dict(
                state="stopping",
                error="example",
                accepted=job.accepted,
                cleanup_pending=True,
                resume_available=False,
            ),
        )
        self.assertIsNot(status["accepted"], job.accepted)
        status["accepted"]["visited_values"] = 2
        self.assertEqual(job.accepted["visited_values"], 1)

    def test_page_forwards_count_and_accepted_publication(self):
        _case, platform, job, auth, _signals = self.finished()
        revision = job.store.info.revision
        page = self.success(lambda: job.page(*auth, revision, "rows", 0, 1))
        self.assertEqual(
            (page["revision"], page["start"], page["end"]), (revision, 0, 1)
        )
        self.assertEqual((len(page["original"]), len(page["control"])), (1, 1))
        platform.now = job.lease_end
        self.refuses(
            lambda: job.page(*auth, revision, "rows", 0, 1),
            "Profile owner lease expired",
        )

    def test_failure_preserves_code_and_closes_validation_before_cleanup(self):
        _case, platform, job, _auth, _reply = self.started()
        close = Mock()
        job.validation = SimpleNamespace(close=close)
        self.assertIsNone(self.success(lambda: job._fail("local_example")))
        self.assertEqual(
            (job.error, job.state, job.validation), ("local_example", "stopping", None)
        )
        close.assert_called_once_with()
        self.assertTrue(platform.child.stopped)
        self.assertTrue(platform.token.held)

    def test_cleanup_releases_token_and_watchdog_after_fake_reap(self):
        case, platform, job, _auth, _reply = self.started()
        platform.child.reaped = True
        job.state, job.error = "stopping", "example"
        self.assertIs(self.success(job._finish_cleanup), True)
        self.assertEqual(
            (platform.token.held, platform.token.releases, platform.watch_closed),
            (False, 1, True),
        )
        self.assertEqual(
            (
                job.child,
                job.token,
                job.watchdog,
                job.state,
                job.store.pending,
                job.store._owned,
                case.os.files,
            ),
            (None, None, None, "error", None, (), {}),
        )

    def test_tick_publishes_exact_partial_and_complete_results(self):
        for values in (5, 12):
            with self.subTest(values=values):
                _case, platform, job, _auth, signals = self.finished(values)
                self.assertIs(signals[-1], True)
                self.assertEqual(job.state, "complete" if values == 12 else "partial")
                self.assertEqual(
                    job.accepted,
                    dict(
                        revision=job.store.info.revision,
                        visited_values=values,
                        total_values=12,
                        complete=values == 12,
                    ),
                )
                self.assertEqual(
                    job._accepted_snapshot,
                    (job.store.publication_serial, job.store.info.revision),
                )
                self.assertIsNone(job.child)
                self.assertIsNone(job.token)
                self.assertFalse(platform.token.held)
                self.assertTrue(platform.watch_closed)
                self.assertEqual(platform.calls, 1)


if __name__ == "__main__":
    unittest.main()
