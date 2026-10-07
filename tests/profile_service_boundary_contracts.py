"""Service and watchdog boundaries use inert threads, clocks and owned children."""

from functools import partial
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, Mock, patch

from atlas_host import profile_service as module
import profile_runtime_doubles as existing


class ServiceBoundaryTests(unittest.TestCase):
    def ok(self, function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except BaseException as error:
            self.fail(
                f"Valid inert service call raised {type(error).__name__}: {error}"
            )

    def error(self, message, function, *args, **kwargs):
        try:
            function(*args, **kwargs)
        except BaseException as error:
            self.assertIs(type(error), ValueError)
            self.assertEqual(str(error), message)
        else:
            self.fail("Expected the existing bounded refusal")

    def condition(self):
        condition = MagicMock()
        condition.__enter__.return_value = condition
        condition.__exit__.return_value = None
        condition.wait_for.side_effect = lambda predicate, timeout: predicate()
        return condition

    def watch(self):
        meter = NS(register=Mock(), freeze=Mock(), read=Mock(return_value=0.0))
        clock = Mock(return_value=4.0)
        diagnostics = NS(event=Mock())
        watch = self.ok(
            module.WatchdogLoop, meter, clock=clock, diagnostics=diagnostics
        )
        self.assertIs(watch.meter, meter)
        self.assertIs(watch.clock, clock)
        self.assertIs(watch.diagnostics, diagnostics)
        self.assertIs(watch.stopped, False)
        self.assertIsNone(watch.lifetime)
        self.assertIsNone(watch.thread)
        self.assertEqual(watch.slots, [])
        watch.condition = self.condition()
        return watch

    def slot(self):
        watch = self.watch()
        callback = Mock()
        slot = self.ok(module.WatchSlot, watch, callback, 9.0)
        self.assertIs(slot.loop, watch)
        self.assertIs(slot.callback, callback)
        self.assertEqual(slot.deadline, 9.0)
        self.assertIs(slot.running, False)
        self.assertIs(slot.failed, False)
        self.assertIs(slot.closed, False)
        watch.slots.append(slot)
        return slot

    def case(self):
        case = existing.ServiceTests("runTest")
        self.addCleanup(case.doCleanups)
        diagnostics = NS(event=Mock())
        factory = partial(module.ProfileService, diagnostics=diagnostics)
        with patch.object(existing, "ProfileService", factory):
            self.ok(case.setUp)
        service = case.service
        self.assertIs(service.diagnostics, diagnostics)
        self.assertIs(service.stopped, False)
        self.assertIsNone(service.owner_cleanup)
        self.assertIsNone(service.thread)
        self.assertIsNone(service.record)
        self.assertIs(case.watch.stopped, False)
        self.assertIsNone(case.watch.lifetime)
        return case

    def admitted(self):
        case = self.case()
        grant, response = self.ok(case.start)
        self.assertEqual(response["version"], 1)
        self.assertEqual(response["model_id"], case.data["model_id"])
        self.assertEqual(response["context_id"], case.data["context_id"])
        self.assertEqual(response["state"], "admitting")
        self.assertEqual(len(response["job_id"]), 32)
        self.assertEqual(len(response["job_capability"]), 64)
        self.assertEqual(case.service.record["lease"], case.clock.now + 15)
        self.assertIsInstance(case.service.record["watch"], module.WatchSlot)
        return case, grant, response

    def test_slot_constructor_keeps_exact_initial_flags(self):
        self.slot()

    def test_slot_register_keeps_owner_and_equal_deadline(self):
        slot = self.slot()
        replacement = Mock()
        self.assertIs(self.ok(slot.register, replacement, 9.0), slot)
        self.assertIs(slot.callback, replacement)
        self.assertEqual(slot.deadline, 9.0)
        self.error("Watchdog cannot renew grant", slot.register, replacement, 10.0)

    def test_slot_disarm_retains_boolean_close_and_existing_failure(self):
        slot = self.slot()
        self.assertIsNone(self.ok(slot.disarm))
        self.assertIs(slot.closed, True)
        self.assertNotIn(slot, slot.loop.slots)
        slot.failed = True
        self.error("Independent watchdog callback failed", slot.disarm)

    def test_slot_close_keeps_wait_bound_and_failure_text(self):
        slot = self.slot()
        self.assertIsNone(self.ok(slot.close))
        self.assertIs(slot.closed, True)
        self.assertNotIn(slot, slot.loop.slots)
        slot.loop.condition.wait_for.assert_called_once()
        self.assertEqual(
            slot.loop.condition.wait_for.call_args.kwargs, {"timeout": 0.15}
        )
        slot.failed = True
        self.error("Independent watchdog callback failed", slot.close)

    def test_watch_constructor_retains_dependencies_without_starting(self):
        self.watch()

    def test_lifetime_install_has_no_early_callback(self):
        watch = self.watch()
        callback = Mock()
        self.assertIsNone(self.ok(watch.set_lifetime, callback))
        self.assertIs(watch.lifetime, callback)
        callback.assert_not_called()
        self.error("Lifetime watch already installed", watch.set_lifetime, Mock())

    def test_bind_retains_exact_slot_deadline_and_one_notification(self):
        watch = self.watch()
        callback = Mock()
        slot = self.ok(watch.bind, callback, 9.0)
        self.assertIsInstance(slot, module.WatchSlot)
        self.assertIs(watch.slots[0], slot)
        self.assertIs(slot.callback, callback)
        self.assertEqual(slot.deadline, 9.0)
        watch.condition.notify_all.assert_called_once_with()

    def test_watch_pulse_calls_lifetime_and_slot_once(self):
        watch = self.watch()
        callback, lifetime = Mock(), Mock()
        self.ok(watch.set_lifetime, lifetime)
        self.ok(watch.bind, callback, 9.0)
        self.assertIsNone(self.ok(watch.pulse))
        lifetime.assert_called_once_with()
        callback.assert_called_once_with()

    def test_watch_pulse_retains_exact_callback_diagnostic(self):
        watch = self.watch()
        refusal = ValueError("inert callback refusal")
        slot = self.ok(watch.bind, Mock(side_effect=refusal), 9.0)
        self.assertIsInstance(slot, module.WatchSlot)
        self.assertIsNone(self.ok(watch.pulse))
        self.assertIs(slot.failed, True)
        self.assertIs(slot.running, False)
        self.assertIn(slot, watch.slots)
        watch.diagnostics.event.assert_called_once_with(
            "profile_service.WatchdogLoop.pulse.catch75", refusal
        )

    def test_watch_run_uses_bounded_wait_and_freezes_once(self):
        for lifetime, expected in [(None, None), (Mock(), 0.05)]:
            with self.subTest(wait=expected):
                watch = self.watch()
                watch.lifetime = lifetime
                watch.ready = NS(set=Mock())
                watch.condition.wait.side_effect = lambda value: setattr(
                    watch, "stopped", True
                )
                self.assertIsNone(self.ok(watch.run))
                watch.meter.register.assert_called_once_with()
                watch.ready.set.assert_called_once_with()
                watch.condition.wait.assert_called_once_with(expected)
                watch.meter.freeze.assert_called_once_with()

    def test_watch_start_records_exact_thread_and_wait_without_launching(self):
        watch = self.watch()
        thread = NS(start=Mock())
        watch.ready = NS(wait=Mock(return_value=True))
        with patch.object(
            module.threading, "Thread", return_value=thread
        ) as constructor:
            self.assertIsNone(self.ok(watch.start))
        constructor.assert_called_once_with(
            target=watch.run, name="atlas-profile-watchdog", daemon=True
        )
        thread.start.assert_called_once_with()
        watch.ready.wait.assert_called_once_with(0.5)
        self.assertIs(watch.thread, thread)

    def test_watch_close_keeps_one_notification_and_join_bound(self):
        watch = self.watch()
        watch.thread = NS(join=Mock(), is_alive=Mock(return_value=False))
        self.assertIsNone(self.ok(watch.close))
        self.assertIs(watch.stopped, True)
        watch.condition.notify_all.assert_called_once_with()
        watch.thread.join.assert_called_once_with(0.2)

    def test_service_constructor_keeps_diagnostics_and_initial_state(self):
        self.case()

    def test_shutdown_request_sets_each_event_once(self):
        case = self.case()
        service = case.service
        service.shutdown_requested = NS(set=Mock())
        service.wake = NS(set=Mock())
        self.assertIsNone(self.ok(service.request_shutdown))
        service.shutdown_requested.set.assert_called_once_with()
        service.wake.set.assert_called_once_with()

    def test_admission_retains_exact_grant_meter_and_register_count(self):
        case = self.case()
        service = case.service
        self.error("Supervising execution contexts unavailable", service.admission)
        service.thread = NS(is_alive=Mock(return_value=True))
        service.watchdog.thread = NS(is_alive=Mock(return_value=True))
        meter = NS(register=Mock(), read=Mock(return_value=0.0))
        with patch.object(module, "ThreadMeter", return_value=meter) as constructor:
            result = self.ok(service.admission)
        self.assertIsInstance(result[0], module.AdmissionGrant)
        self.assertIs(result[1], meter)
        self.assertIs(result[0].clock, service.clock)
        constructor.assert_called_once_with(diagnostics=service.diagnostics)
        meter.register.assert_called_once_with()

    def test_owner_returns_exact_record_after_one_authorization(self):
        case, _, _ = self.admitted()
        service = case.service
        with patch.object(
            service.context, "authorize", wraps=service.context.authorize
        ) as authorize:
            self.assertIs(self.ok(service._owner, case.owner), service.record)
        authorize.assert_called_once_with(
            case.data["model_id"], case.data["context_id"], case.data["tab_capability"]
        )
        self.error(
            "Private job mismatch", service._owner, {**case.owner, "job_id": "wrong"}
        )

    def test_snapshot_retains_exact_ids_status_and_detached_values(self):
        case, _, response = self.admitted()
        record = case.service.record
        record["status"] = {
            "state": "complete",
            "accepted": {
                "revision": "fixed",
                "visited_values": 5,
                "total_values": 12,
                "complete": False,
            },
            "error": None,
            "cleanup_pending": False,
            "resume_available": False,
        }
        result = self.ok(case.service._snapshot, record)
        self.assertEqual(
            result,
            {
                "version": 1,
                "model_id": case.data["model_id"],
                "context_id": case.data["context_id"],
                "job_id": response["job_id"],
                **record["status"],
            },
        )
        result["accepted"]["visited_values"] = 99
        self.assertEqual(record["status"]["accepted"]["visited_values"], 5)
        record["cancel"] = True
        self.assertEqual(
            self.ok(case.service._snapshot, record),
            {
                "version": 1,
                "model_id": case.data["model_id"],
                "context_id": case.data["context_id"],
                "job_id": response["job_id"],
                "state": "stopping",
                "accepted": None,
                "error": None,
                "cleanup_pending": True,
                "resume_available": False,
            },
        )

    def test_start_pins_lease_ids_and_initial_status(self):
        case, _, response = self.admitted()
        self.assertEqual(case.service.record["lease"], 15)
        self.assertEqual(response["state"], "admitting")

    def test_admission_guard_keeps_diagnostic_and_single_cancel_wake(self):
        case, grant, _ = self.admitted()
        case.clock.now = 6
        service = case.service
        service.wake = NS(set=Mock())
        with patch.object(grant, "cancel", wraps=grant.cancel) as cancel:
            self.assertIsNone(self.ok(case.watch.pulse))
        cancel.assert_called_once_with()
        service.wake.set.assert_called_once_with()
        self.assertIs(service.record["cancel"], True)
        service.diagnostics.event.assert_called_once()
        self.assertEqual(
            service.diagnostics.event.call_args.args[0],
            "profile_service.ProfileService.start.admission_guard.catch187",
        )

    def test_finish_admission_sets_exact_ready_flag_and_one_wake(self):
        case, grant, _ = self.admitted()
        service = case.service
        service.wake = NS(set=Mock())
        self.assertIsNone(self.ok(service.finish_admission, grant))
        self.assertIs(service.record["ready"], True)
        service.wake.set.assert_called_once_with()

    def test_status_composes_each_owner_snapshot_once(self):
        case, _, _ = self.admitted()
        service = case.service
        with (
            patch.object(service, "_owner", wraps=service._owner) as owner,
            patch.object(service, "_snapshot", wraps=service._snapshot) as snapshot,
        ):
            result = self.ok(service.status, case.owner)
        owner.assert_called_once_with(case.owner)
        snapshot.assert_called_once_with(service.record)
        self.assertEqual(result, service._snapshot(service.record))

    def test_page_forwards_exact_range_and_existing_missing_runtime_refusal(self):
        case, _, _ = self.admitted()
        service = case.service
        record = service.record
        record["status"]["cleanup_pending"] = False
        data = {
            **case.owner,
            "revision": "fixed",
            "axis": "rows",
            "start": 2,
            "count": 3,
        }
        self.error("No accepted profile", service.page, data)
        runtime = NS(page=Mock(return_value={"end": 5}))
        record["runtime"] = runtime
        self.assertIs(self.ok(service.page, data), runtime.page.return_value)
        runtime.page.assert_called_once_with(
            record["tab"], record["cap"], record["context_id"], "fixed", "rows", 2, 3
        )

    def test_cancel_record_keeps_stopping_fields_and_one_wake(self):
        case, _, _ = self.admitted()
        service = case.service
        service.wake = NS(set=Mock())
        self.assertIsNone(self.ok(service._cancel, service.record))
        self.assertEqual(
            service.record["status"],
            {
                "state": "stopping",
                "accepted": None,
                "error": None,
                "cleanup_pending": True,
                "resume_available": False,
            },
        )
        service.wake.set.assert_called_once_with()

    def test_cancel_settled_record_keeps_cancelled_fields(self):
        case, _, _ = self.admitted()
        service = case.service
        record = service.record
        self.ok(record["watch"].close)
        self.ok(
            service.supervisor.admission_aborted, record["token"], no_child_created=True
        )
        self.ok(record["token"].release)
        record["pending"] = False
        self.assertIsNone(self.ok(service._cancel, record))
        self.assertEqual(
            record["status"],
            {
                "state": "cancelled",
                "accepted": None,
                "error": None,
                "cleanup_pending": False,
                "resume_available": False,
            },
        )

    def test_cancel_composes_each_mutation_and_snapshot_once(self):
        case, _, _ = self.admitted()
        service = case.service
        with (
            patch.object(service, "_cancel", wraps=service._cancel) as cancel,
            patch.object(service, "_snapshot", wraps=service._snapshot) as snapshot,
        ):
            result = self.ok(service.cancel, case.owner)
        cancel.assert_called_once_with(service.record)
        snapshot.assert_called_once_with(service.record)
        self.assertEqual(result, service._snapshot(service.record))

    def test_reconcile_keeps_terminal_boolean_and_pending_fields(self):
        case = self.case()
        service = case.service
        data = {
            k: case.data[k]
            for k in ("version", "model_id", "context_id", "tab_capability")
        }
        result = self.ok(service.reconcile, data)
        self.assertEqual(
            result,
            {
                "state": "cancelled",
                "cleanup_pending": False,
                "resume_available": False,
                "no_owned_work": True,
            },
        )
        self.assertIs(result["no_owned_work"], True)
        self.ok(case.start)
        self.assertEqual(
            self.ok(service.reconcile, data),
            {"state": "stopping", "cleanup_pending": True, "resume_available": False},
        )

    def test_heartbeat_keeps_exact_lease_and_one_runtime_callback(self):
        case, grant, _ = self.admitted()
        service = case.service
        record = service.record
        record["runtime"] = NS(job=NS(heartbeat=Mock()))
        case.clock.now = 1
        deadline = grant.deadline
        result = self.ok(service.heartbeat, case.owner)
        self.assertEqual(record["lease"], 16)
        self.assertEqual(grant.deadline, deadline)
        record["runtime"].job.heartbeat.assert_called_once_with(
            record["tab"], record["cap"], record["context_id"]
        )
        self.assertEqual(result, service._snapshot(record))

    def fake_runtime(self):
        return NS(
            job=NS(
                job_capability="initial", cancel_event=NS(set=Mock()), state="running"
            ),
            platform=NS(closed=False, child=None),
            start=Mock(),
            tick=Mock(),
            status=Mock(
                return_value={
                    "state": "running",
                    "accepted": None,
                    "error": None,
                    "cleanup_pending": False,
                    "resume_available": False,
                }
            ),
        )

    def test_step_keeps_exact_start_value_and_single_tick(self):
        case, grant, _ = self.admitted()
        service = case.service
        runtime = self.fake_runtime()
        service.runtime_factory = Mock(return_value=runtime)
        self.ok(service.finish_admission, grant)
        self.assertIsNone(self.ok(service.step))
        runtime.start.assert_called_once_with(case.data["values"])
        runtime.tick.assert_called_once_with()
        self.assertIs(service.record["runtime"], runtime)

    def test_step_failure_keeps_runtime_error_and_owned_cancel(self):
        case, grant, _ = self.admitted()
        service = case.service
        runtime = self.fake_runtime()
        runtime.start.side_effect = ValueError("inert runtime refusal")
        service.runtime_factory = Mock(return_value=runtime)
        self.ok(service.finish_admission, grant)
        self.assertIsNone(self.ok(service.step))
        self.assertEqual(
            service.record["status"],
            {
                "state": "stopping",
                "accepted": None,
                "error": "runtime_error",
                "cleanup_pending": True,
                "resume_available": False,
            },
        )
        self.assertIs(service.record["cancel"], True)
        runtime.job.cancel_event.set.assert_called_once_with()

    def test_owner_run_keeps_idle_wait_and_single_meter_lifecycle(self):
        case = self.case()
        service = case.service
        service.owner_meter = NS(register=Mock(), freeze=Mock())
        service.ready = NS(set=Mock())
        service.wake = NS(
            wait=Mock(side_effect=lambda timeout: setattr(service, "stopped", True)),
            clear=Mock(),
        )
        with patch.object(service, "step") as step:
            self.assertIsNone(self.ok(service.run))
        step.assert_called_once_with()
        service.owner_meter.register.assert_called_once_with()
        service.ready.set.assert_called_once_with()
        service.wake.wait.assert_called_once_with(0.25)
        service.wake.clear.assert_called_once_with()
        service.owner_meter.freeze.assert_called_once_with()

    def test_owner_run_waits_on_exact_retained_unreaped_child(self):
        case = self.case()
        service = case.service
        service.owner_meter = NS(register=Mock(), freeze=Mock())
        service.ready = NS(set=Mock())
        child = NS(
            reaped=False,
            wait=Mock(side_effect=lambda timeout: setattr(service, "stopped", True)),
        )
        runtime = self.fake_runtime()
        runtime.platform.child = child
        service.record = {"runtime": runtime}
        with patch.object(service, "step"):
            self.assertIsNone(self.ok(service.run))
        child.wait.assert_called_once_with(0.05)
        service.owner_meter.freeze.assert_called_once_with()

    def test_start_threads_keeps_exact_owner_and_wait_without_launching(self):
        case = self.case()
        service = case.service
        service.ready = NS(wait=Mock(return_value=True))
        thread = NS(start=Mock())
        with (
            patch.object(service.watchdog, "start") as start,
            patch.object(
                module.threading, "Thread", return_value=thread
            ) as constructor,
        ):
            self.assertIsNone(self.ok(service.start_threads))
        start.assert_called_once_with()
        constructor.assert_called_once_with(
            target=service.run, name="atlas-profile-owner", daemon=True
        )
        thread.start.assert_called_once_with()
        service.ready.wait.assert_called_once_with(0.5)
        self.assertIs(service.thread, thread)

    def test_close_keeps_exact_join_single_watch_close_and_bool(self):
        case = self.case()
        service = case.service
        service.thread = NS(join=Mock(), is_alive=Mock(return_value=False))
        with patch.object(service.watchdog, "close") as close:
            self.assertIs(self.ok(service.close), True)
        service.thread.join.assert_called_once_with(0.3)
        close.assert_called_once_with()
        self.assertIs(service.stopped, True)


if __name__ == "__main__":
    unittest.main()
