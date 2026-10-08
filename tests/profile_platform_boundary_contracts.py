"""Platform callbacks and exact values with inert grants, children and storage."""

from copy import deepcopy
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from atlas_host import profile_platform as module


class PlatformBoundaryTests(unittest.TestCase):
    def ok(self, function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except BaseException as error:
            self.fail(
                f"Valid inert platform call raised {type(error).__name__}: {error}"
            )

    def error(self, message, function, *args, **kwargs):
        try:
            function(*args, **kwargs)
        except BaseException as error:
            self.assertIs(type(error), ValueError)
            self.assertEqual(str(error), message)
        else:
            self.fail("Expected the existing bounded refusal")

    def parts(self):
        token = NS(
            kind="profile",
            context="context",
            current=Mock(return_value=True),
            poison=Mock(),
            release=Mock(),
        )
        grant = NS(
            clock=Mock(return_value=4.25),
            cpu=NS(total=Mock(return_value=2.5)),
            remaining=Mock(return_value={"wall_ms": 1000, "cpu_ms": 900}),
            deadline=9.0,
            sample_child=Mock(),
        )
        supervisor = NS(
            snapshot_reservation=32,
            acquire=Mock(return_value=token),
            admission_aborted=Mock(),
            child_finished=Mock(),
            retain_profile=Mock(),
            close_profile=Mock(),
        )
        child = NS(
            sample=Mock(return_value={"cpu_seconds": 0.25, "reaped": True}), stop=Mock()
        )
        registration = NS(close=Mock())
        observed = {
            "rss_bytes": 17,
            "available_bytes": 6 * module.GIB,
            "all_owned_accounted": True,
            "descendants_clear": True,
        }
        hooks = NS(
            diagnostics=NS(event=Mock()),
            source_check=Mock(),
            start_gate=Mock(return_value=(6 * module.GIB, 30 * module.GIB)),
            launch_policy=None,
            spawn=Mock(return_value=child),
            resources=Mock(return_value=observed),
            watch=Mock(return_value=registration),
            no_child_created=Mock(return_value=True),
            snapshot_closed=Mock(return_value=True),
            close_admission_watch=Mock(),
        )
        return supervisor, token, grant, hooks, child, registration

    def make(self):
        supervisor, token, grant, hooks, child, registration = self.parts()
        platform = self.ok(module.ProfilePlatform, supervisor, token, grant, hooks)
        self.assertIs(platform.supervisor, supervisor)
        self.assertIs(platform.reserved, token)
        self.assertIs(platform.grant, grant)
        self.assertIs(platform.hooks, hooks)
        self.assertIs(platform.diagnostics, hooks.diagnostics)
        self.assertIsInstance(platform.token, module.DeferredToken)
        self.assertIs(platform.token.platform, platform)
        self.assertIs(platform.token.requested, False)
        self.assertIsNone(platform.child)
        self.assertIsNone(platform.runtime)
        self.assertIs(platform.watch_closed, False)
        self.assertIs(platform.spawn_attempted, False)
        self.assertIs(platform.closed, False)
        token.current.reset_mock()
        return platform, child, registration

    def job(self):
        return NS(
            state="complete",
            error=None,
            child=None,
            validation=None,
            watchdog=None,
            store=NS(
                pending=None,
                require_settled=Mock(),
                frame_bytes=16,
                latest=object(),
                owned_storage_bytes=32,
                close=Mock(),
            ),
            accepted={"revision": 7},
            expired=False,
            cancel_event=NS(set=Mock()),
            guard=Mock(),
            job_capability="b" * 64,
            start=Mock(),
            tick=Mock(),
            status=Mock(return_value={"state": "idle", "error": None}),
            page=Mock(return_value={"end": 4}),
            _owns=Mock(),
        )

    def runtime(self):
        supervisor, token, grant, hooks, _, _ = self.parts()
        job = self.job()
        selected = {"inert": True}
        with patch.object(module, "ProfileJob", return_value=job) as constructor:
            runtime = self.ok(
                module.SupervisedProfile,
                supervisor,
                grant,
                hooks,
                selected,
                17,
                "a" * 64,
                "context",
            )
        self.assertIs(runtime.diagnostics, hooks.diagnostics)
        self.assertEqual(runtime.auth, ("a" * 64, "b" * 64, "context"))
        self.assertIs(runtime.job, job)
        constructor.assert_called_once_with(
            selected, 17, "a" * 64, "context", runtime.platform
        )
        supervisor.retain_profile.assert_called_once_with(runtime, 16)
        grant.remaining.assert_called_once_with()
        grant.remaining.reset_mock()
        return runtime

    def test_deferred_constructor_keeps_owner_and_false_intent(self):
        platform, _, _ = self.make()
        token = self.ok(module.DeferredToken, platform)
        self.assertIs(token.platform, platform)
        self.assertIs(token.requested, False)

    def test_deferred_current_is_boolean_and_queries_once(self):
        platform, _, _ = self.make()
        self.assertIs(self.ok(platform.token.current), True)
        platform.reserved.current.assert_called_once_with()
        platform.token.requested = True
        self.assertIs(self.ok(platform.token.current), False)

    def test_deferred_poison_forwards_once_without_release_intent(self):
        platform, _, _ = self.make()
        self.assertIsNone(self.ok(platform.token.poison))
        platform.reserved.poison.assert_called_once_with()
        self.assertIs(platform.token.requested, False)
        platform.reserved.release.assert_not_called()

    def test_deferred_release_records_only_boolean_intent(self):
        platform, _, _ = self.make()
        self.assertIsNone(self.ok(platform.token.release))
        self.assertIs(platform.token.requested, True)
        platform.reserved.poison.assert_not_called()
        platform.reserved.release.assert_not_called()

    def test_platform_constructor_retains_owners_and_initial_state(self):
        self.make()

    def test_clock_forwards_exact_fraction_and_one_read(self):
        platform, _, _ = self.make()
        self.assertEqual(self.ok(platform.clock), 4.25)
        platform.grant.clock.assert_called_once_with()

    def test_owner_cpu_forwards_exact_fraction_and_one_read(self):
        platform, _, _ = self.make()
        self.assertEqual(self.ok(platform.owner_cpu), 2.5)
        platform.grant.cpu.total.assert_called_once_with()

    def test_acquire_retains_token_and_exact_existing_refusal(self):
        platform, _, _ = self.make()
        self.assertIs(self.ok(platform.acquire), platform.token)
        platform.grant.remaining.assert_called_once_with()
        platform.reserved.current.return_value = False
        self.error("Supervised compute ownership lost", platform.acquire)

    def test_start_gate_passes_exact_bound_observations_once(self):
        platform, _, _ = self.make()
        with patch.object(module, "check_start") as gate:
            self.assertIsNone(self.ok(platform.start_gate))
        gate.assert_called_once_with(None, 6 * module.GIB, 30 * module.GIB)
        platform.grant.remaining.assert_called_once_with()
        platform.hooks.start_gate.assert_called_once_with()

    def test_source_check_has_one_active_grant_read_and_none_when_closed(self):
        platform, _, _ = self.make()
        self.assertIsNone(self.ok(platform.source_check))
        platform.hooks.source_check.assert_called_once_with()
        platform.grant.remaining.assert_called_once_with()
        platform.closed = True
        platform.grant.remaining.reset_mock()
        platform.hooks.source_check.reset_mock()
        self.assertIsNone(self.ok(platform.source_check))
        platform.hooks.source_check.assert_called_once_with()
        platform.grant.remaining.assert_not_called()

    def test_spawn_retains_child_and_exact_bounded_copy(self):
        platform, child, _ = self.make()
        request = {"wall_ms": 1400, "cpu_ms": 800, "nested": {"value": 7}}
        before = deepcopy(request)
        self.assertIs(self.ok(platform.spawn, request, 19, None), child)
        self.assertIs(platform.child, child)
        self.assertIs(platform.spawn_attempted, True)
        platform.grant.remaining.assert_called_once_with(reserve_ms=800)
        actual = platform.hooks.spawn.call_args.args
        self.assertEqual(
            actual, ({"wall_ms": 1000, "cpu_ms": 800, "nested": {"value": 7}}, 19, None)
        )
        self.assertIsNot(actual[0], request)
        self.assertIsNot(actual[0]["nested"], request["nested"])
        self.assertEqual(request, before)
        self.error(
            "One disposable Restart child only", platform.spawn, request, 19, None
        )

    def test_resource_exact_ceilings_and_bool_result(self):
        platform, _, _ = self.make()
        frame = 32 * module.MIB
        platform.supervisor.snapshot_reservation = frame
        platform.hooks.resources.return_value["rss_bytes"] = 768 * module.MIB - frame
        self.assertIs(self.ok(platform.resources_ok, frame), True)
        platform.hooks.resources.return_value["rss_bytes"] += 1
        self.assertIs(self.ok(platform.resources_ok, frame), False)

    def test_watchdog_keeps_deadline_registration_and_close(self):
        platform, _, registration = self.make()
        job = self.job()
        watch = self.ok(platform.arm_watchdog, job)
        self.assertEqual(platform.hooks.watch.call_count, 1)
        callback, deadline = platform.hooks.watch.call_args.args
        self.assertTrue(callable(callback))
        self.assertEqual(deadline, 9.0)
        self.assertIsNone(self.ok(watch.close))
        registration.close.assert_called_once_with()
        self.assertIs(platform.watch_closed, True)
        platform.hooks.watch.return_value = None
        self.error(
            "Independent watchdog registration required", platform.arm_watchdog, job
        )

    def test_watchdog_refusal_keeps_exact_diagnostic_and_owned_stop(self):
        platform, child, _ = self.make()
        job = self.job()
        self.ok(platform.arm_watchdog, job)
        callback = platform.hooks.watch.call_args.args[0]
        platform.child = child
        refusal = ValueError("inert grant expired")
        platform.grant.remaining.side_effect = refusal
        self.assertIsNone(self.ok(callback))
        self.assertIs(job.expired, True)
        job.cancel_event.set.assert_called_once_with()
        child.stop.assert_called_once_with()
        platform.reserved.poison.assert_called_once_with()
        platform.hooks.diagnostics.event.assert_called_once_with(
            "profile_platform.ProfilePlatform.arm_watchdog.guard.catch109", refusal
        )

    def test_finish_returns_bool_after_exact_finalization(self):
        platform, _, _ = self.make()
        job = self.job()
        platform.watch_closed = True
        self.assertIs(self.ok(platform.finish, job), True)
        self.assertIs(platform.closed, True)
        platform.supervisor.admission_aborted.assert_called_once_with(
            platform.reserved, no_child_created=True
        )
        platform.reserved.release.assert_called_once_with()
        self.assertIs(self.ok(platform.finish, job), True)
        self.assertEqual(platform.reserved.release.call_count, 1)

    def test_finish_settlement_refusal_retains_exact_diagnostic(self):
        platform, _, _ = self.make()
        job = self.job()
        refusal = ValueError("inert storage unsettled")
        job.store.require_settled.side_effect = refusal
        self.assertIs(self.ok(platform.finish, job), False)
        platform.hooks.diagnostics.event.assert_called_once_with(
            "profile_platform.ProfilePlatform.finish.catch136", refusal
        )
        platform.reserved.poison.assert_called_once_with()
        platform.reserved.release.assert_not_called()

    def test_finish_resource_refusal_keeps_public_error_and_closed_storage(self):
        platform, _, _ = self.make()
        job = self.job()
        platform.watch_closed = True
        platform.grant.remaining.side_effect = ValueError("inert grant expired")
        self.assertIs(self.ok(platform.finish, job), True)
        self.assertEqual((job.state, job.error), ("error", "resource_limit"))
        self.assertIsNone(job.accepted)
        job.store.close.assert_called_once_with()

    def test_runtime_constructor_retains_exact_auth_and_job(self):
        self.runtime()

    def test_runtime_start_keeps_arguments_and_one_finalization(self):
        runtime = self.runtime()
        with patch.object(runtime.platform, "finish", return_value=False) as finish:
            self.assertIsNone(self.ok(runtime.start, 13))
        runtime.job.start.assert_called_once_with(
            *runtime.auth, 13, wall_ms=1000, cpu_ms=900
        )
        finish.assert_called_once_with(runtime.job)
        runtime.job.state = "idle"
        refusal = ValueError("inert start refusal")
        runtime.job.start.side_effect = refusal
        with patch.object(runtime.platform, "finish", return_value=False):
            self.error("inert start refusal", runtime.start, 13)
        self.assertEqual(
            (runtime.job.state, runtime.job.error), ("error", "admission_failed")
        )

    def test_restart_retains_new_platform_and_exact_request(self):
        runtime = self.runtime()
        self.error(
            "Previous operation requires cleanup",
            runtime.restart,
            runtime.platform.grant,
            runtime.platform.hooks,
            13,
        )
        runtime.platform.closed = True
        previous = runtime.platform
        with patch.object(runtime, "start") as start:
            self.assertIsNone(
                self.ok(runtime.restart, previous.grant, previous.hooks, 13)
            )
        self.assertIsNot(runtime.platform, previous)
        self.assertIs(runtime.job.platform, runtime.platform)
        start.assert_called_once_with(13)

    def test_tick_forwards_once_and_retains_none_return(self):
        runtime = self.runtime()
        with patch.object(runtime.platform, "finish", return_value=False) as finish:
            self.assertIsNone(self.ok(runtime.tick))
        runtime.job.tick.assert_called_once_with()
        finish.assert_called_once_with(runtime.job)

    def test_status_preserves_existing_record_and_pending_field_types(self):
        runtime = self.runtime()
        original = {"state": "idle", "error": None}
        runtime.job.status.return_value = original
        self.assertIs(self.ok(runtime.status, *runtime.auth), original)
        runtime.job.status.return_value = {"state": "complete", "error": None}
        result = self.ok(runtime.status, *runtime.auth)
        self.assertEqual(
            result,
            {
                "state": "stopping",
                "error": None,
                "accepted": None,
                "cleanup_pending": True,
                "resume_available": False,
            },
        )
        self.assertIs(result["cleanup_pending"], True)
        self.assertIs(result["resume_available"], False)

    def test_page_pins_owned_arguments_and_existing_acceptance_refusal(self):
        runtime = self.runtime()
        args = (*runtime.auth, 7, "rows", 2, 3)
        self.error(
            "Profile revision has not completed final acceptance", runtime.page, *args
        )
        runtime.job._owns.reset_mock()
        runtime.platform.closed = True
        self.assertIs(self.ok(runtime.page, *args), runtime.job.page.return_value)
        runtime.job.page.assert_called_once_with(*args)
        runtime.job._owns.assert_called_once_with(*runtime.auth)

    def test_close_keeps_callback_counts_and_unsettled_refusal(self):
        runtime = self.runtime()
        runtime.platform.closed = True
        self.assertIsNone(self.ok(runtime.close))
        runtime.job.store.close.assert_called_once_with()
        runtime.platform.hooks.snapshot_closed.assert_called_once_with(
            runtime.job.store
        )
        runtime.platform.supervisor.close_profile.assert_called_once_with(
            runtime, all_handles_closed=True
        )
        runtime.platform.hooks.snapshot_closed.return_value = False
        self.error(
            "All snapshot handles, including retiring storage, must be closed",
            runtime.close,
        )


if __name__ == "__main__":
    unittest.main()
