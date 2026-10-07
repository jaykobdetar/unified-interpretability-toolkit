"""Static operation bookkeeping uses existing inert owners and fake children."""

from types import SimpleNamespace as NS
import threading
import unittest
from unittest.mock import Mock, patch

from atlas_host import hosted_runtime as module
import dense_static_contracts as existing
import host_picker_contracts as picker


class StaticOperationBoundaryTests(unittest.TestCase):
    def ok(self, function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except BaseException as error:
            self.fail(f"Valid inert operation raised {type(error).__name__}: {error}")

    def error(self, message, function, *args):
        try:
            function(*args)
        except BaseException as error:
            self.assertIs(type(error), ValueError)
            self.assertEqual(str(error), message)
        else:
            self.fail("Expected the existing refusal")

    def cleanup(self, case, operation):
        for child, _, _ in operation.children:
            child.reaped = True
        if not operation.finished:
            operation.abort()
        case.watch.close()

    def owner(self, kind="metadata", *, reader=False):
        case = existing.OperationTests(methodName="runTest")
        self.addCleanup(case.doCleanups)
        self.ok(case.setUp)
        if reader:
            operation = self.ok(case.operation, kind)
        else:
            operation = self.ok(module.StaticOperation, case.app, case.grant(), kind)
        self.addCleanup(self.cleanup, case, operation)
        return case, operation

    def test_constructor_exact_owner_grant_context_and_watch(self):
        case = existing.OperationTests(methodName="runTest")
        self.addCleanup(case.doCleanups)
        self.ok(case.setUp)
        case.app.static_operations = []
        grant = case.grant()
        with (
            patch.object(
                case.supervisor, "acquire", wraps=case.supervisor.acquire
            ) as acquire,
            patch.object(case.watch, "bind", wraps=case.watch.bind) as bind,
        ):
            operation = self.ok(module.StaticOperation, case.app, grant, "metadata")
        self.addCleanup(self.cleanup, case, operation)
        self.assertIs(operation.app, case.app)
        self.assertIs(operation.grant, grant)
        self.assertIsNone(operation.prepared)
        self.assertIsNone(operation.context)
        for name in ["mutated", "failed", "aborting", "finished", "sent"]:
            self.assertIs(getattr(operation, name), False)
        self.assertEqual(case.app.static_operations, [operation])
        acquire.assert_called_once_with("metadata", "dense-static")
        bind.assert_called_once_with(operation.pulse, grant.deadline)

    def test_child_series_keep_mutable_shape_baseline_and_seen(self):
        case, operation = self.owner()
        self.assertIsNone(self.ok(operation.add_child, case.child, 0.25))
        self.assertEqual(operation.children, [[case.child, 0.25, 0.25]])
        self.assertIs(type(operation.children[0]), list)

    def test_reader_deduplication_baseline_and_none_result(self):
        case, operation = self.owner()
        channel = NS(child=case.child, settled_cpu=0.25)
        reader = NS(channel=channel)
        for _ in range(2):
            self.assertIsNone(self.ok(operation.add_reader, reader))
        self.assertEqual(operation.channels, [channel])
        self.assertEqual(operation.children, [[case.child, 0.25, 0.25]])

    def test_observe_exact_cpu_delta_seen_value_and_single_sample(self):
        case, operation = self.owner()
        self.ok(operation.add_child, case.child, 0.25)
        case.child.cpu = 1.25
        with patch.object(
            operation.grant, "sample_child", wraps=operation.grant.sample_child
        ) as sample:
            self.assertIsNone(self.ok(operation._observe, case.child))
        sample.assert_called_once_with(1.0)
        self.assertEqual(operation.children, [[case.child, 0.25, 1.25]])

    def test_check_exact_reservation_for_each_kind_and_single_grant_read(self):
        for kind, growth in [("metadata", 65536), ("tile", 2 * 1024**2)]:
            case, operation = self.owner(kind)
            namespace = operation.check.__func__.__globals__
            reserve = Mock(wraps=namespace["reservation"])
            with (
                patch.dict(namespace, {"reservation": reserve}),
                patch.object(
                    operation.grant, "remaining", wraps=operation.grant.remaining
                ) as remaining,
            ):
                self.assertIsNone(self.ok(operation.check))
            reserve.assert_called_once_with(100 * 1024**3, cache_growth=growth)
            remaining.assert_called_once_with()

    def test_publication_single_check_and_none_result(self):
        _, operation = self.owner()
        with patch.object(operation, "check", wraps=operation.check) as check:
            self.assertIsNone(self.ok(operation.check_publication))
        check.assert_called_once_with()

    def test_publication_binding_refusal_keeps_wording(self):
        operation = object.__new__(module.StaticOperation)
        operation.lock = threading.RLock()
        operation.check = Mock()
        operation.context = "inert-context"
        operation.app = NS(host=NS(static_bound=None))
        self.error(
            "Complete current selected static binding required",
            operation.check_publication,
        )
        operation.check.assert_called_once_with()

    def test_failure_boolean_single_cancel_and_none_result(self):
        case, operation = self.owner(reader=True)
        with patch.object(
            operation.grant, "cancel", wraps=operation.grant.cancel
        ) as cancel:
            self.assertIsNone(self.ok(operation._fail))
        self.assertIs(operation.failed, True)
        self.assertTrue(case.child.stopped)
        cancel.assert_called_once_with()

    def test_failed_pulse_mutation_boolean_and_single_failure_call(self):
        _, operation = self.owner(reader=True)
        with (
            patch.object(
                operation,
                "check",
                side_effect=ValueError("inert observation unavailable"),
            ),
            patch.object(operation, "_fail", wraps=operation._fail) as failed,
        ):
            self.assertIsNone(self.ok(operation.pulse))
        self.assertIs(operation.mutated, True)
        failed.assert_called_once_with()

    def test_aborting_pulse_calls_reap_once(self):
        _, operation = self.owner()
        operation.aborting = True
        with patch.object(operation, "_reap", return_value=False) as reap:
            self.assertIsNone(self.ok(operation.pulse))
        reap.assert_called_once_with()

    def test_reap_exact_true_finished_boolean_and_single_disarm(self):
        _, operation = self.owner()
        with patch.object(
            operation.slot, "disarm", wraps=operation.slot.disarm
        ) as disarm:
            self.assertIs(self.ok(operation._reap), True)
        self.assertIs(operation.finished, True)
        disarm.assert_called_once_with()

    def test_abort_exact_true_boolean_and_single_join(self):
        case, operation = self.owner(reader=True)
        case.child.reaped = True
        with patch.object(operation.slot, "close", wraps=operation.slot.close) as close:
            self.assertIs(self.ok(operation.abort), True)
        self.assertIs(operation.aborting, True)
        close.assert_called_once_with()

    def test_publish_single_write_boolean_and_exact_settled_watermark(self):
        case, operation = self.owner(reader=True)
        case.child.cpu = 1.25
        case.channel.settled_cpu = 0.25
        self.assertIs(type(operation.children[0]), list)
        operation.children[0][1:] = [0.25, 0.25]
        write = Mock()
        self.assertIsNone(self.ok(operation.publish, write))
        write.assert_called_once_with()
        self.assertIs(operation.finished, True)
        self.assertEqual(case.channel.settled_cpu, 1.25)

    def test_fixture_acquire_exact_public_receipt_fields(self):
        case = picker.HostFixtureTests(methodName="runTest")
        self.addCleanup(case.doCleanups)
        self.ok(case.setUp)
        self.addCleanup(case.host.close)
        receipt = self.ok(case.acquire)
        self.assertEqual(
            receipt,
            {
                "api_version": 1,
                "model_id": case.ids[0],
                "context_id": case.host.context,
                "capability": next(iter(case.host.leases)),
                "lease_seconds": 15,
                "view_kind": "fixture",
                "state": "starting",
            },
        )


if __name__ == "__main__":
    unittest.main()
