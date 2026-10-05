"""DSI-1/2 counterexamples inverted into regressions; inert clocks/readers only."""

from types import SimpleNamespace as NS
from pathlib import Path
import json
import sys
import unittest
from unittest.mock import Mock, patch

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "tools"),
    str(Path(__file__).parent),
]

import dense_static_contracts as owner
import static_model_contracts as tiny
from atlas_host.common import canonical, require
from atlas_host import dense_static_admission as policy
from atlas_host.hosted_runtime import NativeChannel, StaticOperation
from atlas_host.profile_http import HostedHandler
from atlas_host.runtime_adapter import FixtureHost, HostError
from profile_runtime_doubles import Socket


class PublicationCounterexamples(unittest.TestCase):
    def operation(self):
        case = owner.OperationTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        op = case.operation()
        case.host.reader = case.reader
        op.context = "ctx"
        op.mutated = False
        op.prepared = NS(check=Mock())
        case.host.static_prepared = op.prepared
        case.host.static_bound = NS(prepared=op.prepared)
        case.app.dense_policy = object.__new__(policy.BoundDenseStaticAdmission)
        case.host.dense_policy = case.app.dense_policy
        case.app.binary = object()
        seal = patch.object(
            policy.BoundDenseStaticAdmission, "check_binary", return_value=None
        )
        checked = seal.start()
        self.addCleanup(seal.stop)
        case.channel.read("/api/model", "ctx", operation=op)
        return case, op, checked

    def test_late_selected_source_refusal_after_ack_writes_no_success(self):
        case, op, _ = self.operation()
        op.prepared.check.side_effect = ValueError("Inert current receipt refusal")
        wrote = Mock()
        with self.assertRaises(ValueError):
            op.publish(wrote)
        wrote.assert_not_called()
        op.prepared.check.assert_called()
        self.assertTrue(case.host.stopping)
        self.assertTrue(case.child.stopped)
        self.assertTrue(case.supervisor.busy())
        self.assertEqual(case.channel.settled_cpu, 0.0)
        case.child.reaped = True
        case.watch.pulse()
        self.assertFalse(case.supervisor.busy())

    def test_selected_reader_reaped_after_ack_cannot_publish_active_state(self):
        case, op, _ = self.operation()
        case.child.reaped = True
        wrote = Mock()
        with self.assertRaises(ValueError):
            op.publish(wrote)
        wrote.assert_not_called()
        self.assertTrue(case.host.stopping)
        self.assertFalse(case.supervisor.busy())
        self.assertEqual(case.channel.settled_cpu, 0.0)

    def test_live_bound_publication_rechecks_source_policy_binary_seals(self):
        case, op, seals = self.operation()
        op.prepared.check.reset_mock()
        seals.reset_mock()
        wrote = Mock()
        op.publish(wrote)
        wrote.assert_called_once()
        self.assertGreaterEqual(op.prepared.check.call_count, 3)
        self.assertGreaterEqual(seals.call_count, 3)
        self.assertTrue(
            all(call.args == (case.app.binary,) for call in seals.call_args_list)
        )
        self.assertFalse(case.supervisor.busy())
        self.assertTrue(op.finished)

    def test_stale_private_seal_before_publication_refuses_without_writing(self):
        case, op, seals = self.operation()
        seals.side_effect = ValueError("Inert stale evidence/binary")
        wrote = Mock()
        with self.assertRaises(ValueError):
            op.publish(wrote)
        wrote.assert_not_called()
        self.assertTrue(case.host.stopping)
        self.assertTrue(case.supervisor.busy())

    def test_disabled_or_changed_owner_policy_after_ack_cannot_publish(self):
        for disabled in ("app", "host", "different_owner"):
            with self.subTest(disabled=disabled):
                case, op, _ = self.operation()
                wrote = Mock()
                if disabled == "app":
                    case.app.dense_policy = None
                elif disabled == "host":
                    case.host.dense_policy = None
                else:
                    case.app.dense_policy = object.__new__(
                        policy.BoundDenseStaticAdmission
                    )
                with self.assertRaises(ValueError):
                    op.publish(wrote)
                wrote.assert_not_called()
                self.assertTrue(case.host.stopping)
                self.assertTrue(case.supervisor.busy())
                self.assertEqual(case.channel.settled_cpu, 0.0)
                case.child.reaped = True
                case.watch.pulse()
                self.assertFalse(case.supervisor.busy())

    def test_source_or_reader_loss_during_write_revokes_without_settling_watermark(
        self,
    ):
        for source_loss in (True, False):
            case, op, _ = self.operation()

            def write():
                if source_loss:
                    op.prepared.check.side_effect = ValueError(
                        "Inert post-write source refusal"
                    )
                else:
                    case.child.reaped = True

            with self.assertRaises(ValueError):
                op.publish(write)
            self.assertTrue(case.host.stopping)
            self.assertEqual(case.channel.settled_cpu, 0.0)
            case.child.reaped = True
            case.watch.pulse()
            self.assertFalse(case.supervisor.busy())

    def test_retired_old_child_reaped_and_selected_new_child_live_publish(self):
        case, op, seals = self.operation()
        old = case.child
        old.cpu = 0.1
        old.reaped = True
        case.channel.failed = True
        new = owner.Child()
        new.cpu = 0.2
        channel = NativeChannel(
            Socket(),
            new,
            case.supervisor,
            case.service,
            case.book,
            clock=lambda: case.now,
            wait=lambda *args: None,
        )
        reader = NS(child=new, channel=channel)
        op.add_reader(reader)
        op.mutated = True
        case.host.reader = reader
        case.host.context = op.context = "new-context"
        case.host.leases = {"new-lease": 15}
        channel.read("/api/model", "new-context", operation=op)
        op.publish(lambda: None)
        self.assertTrue(op.finished)
        self.assertFalse(case.supervisor.busy())
        self.assertAlmostEqual(op.grant.child_cpu, 0.3)
        self.assertEqual(channel.settled_cpu, 0.2)


class ReacquisitionCounterexamples(unittest.TestCase):
    def setUp(self):
        self.fixture = tiny.StaticContracts()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        case = owner.OperationTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        self.case = case
        source = self.fixture
        self.requests = []
        reader = NS(
            child=case.child,
            channel=NativeChannel(
                Socket(),
                case.child,
                case.supervisor,
                case.service,
                case.book,
                clock=lambda: case.now,
                wait=lambda *args: None,
            ),
        )
        reader.initialize = lambda: None
        reader.alive = lambda: not case.child.reaped and not reader.channel.failed
        reader.ready = reader.alive
        reader.stop = lambda: True

        def read(path, *, operation):
            self.requests.append(path)
            return (
                200,
                canonical(source.native_model(operation.prepared)),
                "application/json",
            )

        reader.read = read
        self.host = FixtureHost(
            source.registry,
            source.root / "cache",
            lambda *args, **kw: reader,
            static_policy=source.policy,
            static_admission=lambda _: True,
            clock=lambda: case.now,
        )

        class InitialOp:
            context = None
            prepared = None
            mutated = False

            def check(self):
                pass

        self.first = self.host.acquire(
            {"model_id": source.identifier}, operation=InitialOp()
        )
        self.assertEqual(self.first["state"], "active")
        self.original_leases = dict(self.host.leases)
        self.sealed = object.__new__(policy.BoundDenseStaticAdmission)
        self.host.dense_policy = self.sealed
        self.host.static_admission = Mock(return_value=True)
        case.app.host = self.host
        case.app.dense_policy = self.sealed
        case.app.binary = object()
        case.app.static_operations = []

        def begin(grant, kind):
            require(case.app.dense_policy is not None, "Inert disabled owner policy")
            return StaticOperation(case.app, grant, kind)

        case.app.begin_static = Mock(side_effect=begin)
        self.handler = HostedHandler.__new__(HostedHandler)
        self.handler.server = NS(application=case.app)
        self.grant = case.grant()
        self.handler.profile_admission = self.grant
        self.handler.connection = NS(settimeout=Mock())
        self.hint = patch.object(
            policy.BoundDenseStaticAdmission, "allows", return_value=False
        ).start()
        self.sealcheck = patch.object(
            policy.BoundDenseStaticAdmission, "check_binary", return_value=None
        ).start()
        self.addCleanup(patch.stopall)

    def request(self, identifier=None):
        return self.handler.dispatch_host(
            "POST",
            "/api/view-contexts",
            {"model_id": identifier or self.fixture.identifier},
        )

    def test_same_reader_denied_admission_with_false_hint_issues_no_new_lease(self):
        self.host.static_admission.side_effect = ValueError(
            "Inert refused sealed admission"
        )
        with self.assertRaises(HostError):
            self.request()
        self.assertEqual(self.host.leases, self.original_leases)
        self.host.static_admission.assert_called_once_with(self.host.static_prepared)
        self.case.app.begin_static.assert_called_once_with(self.grant, "metadata")
        self.hint.assert_not_called()
        self.assertTrue(self.handler.static_finalizer.abort())
        self.assertFalse(self.case.child.stopped)
        self.assertFalse(self.host.stopping)

    def test_valid_same_reader_reacquire_has_original_grant_and_publication_owner(self):
        status, raw, mime = self.request()
        self.assertEqual(status, 202)
        self.assertEqual(json.loads(raw)["state"], "active")
        op = self.handler.static_finalizer
        self.assertIs(op.grant, self.grant)
        self.assertEqual(self.grant.deadline, 5)
        self.assertTrue(self.case.supervisor.busy())
        self.host.static_admission.assert_called_once_with(self.host.static_prepared)
        self.case.app.begin_static.assert_called_once_with(self.grant, "metadata")
        self.hint.assert_not_called()
        self.assertEqual(
            self.requests, ["/api/model"]
        )  # No repeated full catalog needed.
        from host_atlas import HostHandler

        with patch.object(HostHandler, "send") as write:
            self.handler.send(status, raw, mime)
        write.assert_called_once()
        self.assertFalse(self.case.supervisor.busy())
        self.assertTrue(op.finished)
        self.sealcheck.assert_called()
        self.case.service.admission.assert_not_called()

    def test_late_reuse_seal_failure_revokes_staged_lease_without_native_command(self):
        status, raw, mime = self.request()
        op = self.handler.static_finalizer
        self.assertFalse(op.sent)
        self.sealcheck.side_effect = ValueError("Inert late owner seal failure")
        from host_atlas import HostHandler

        with patch.object(HostHandler, "send") as write:
            with self.assertRaises(ValueError):
                self.handler.send(status, raw, mime)
        write.assert_not_called()
        self.assertEqual(self.host.leases, {})
        self.assertTrue(self.host.stopping)
        self.assertTrue(self.case.child.stopped)
        self.assertTrue(self.case.supervisor.busy())
        self.assertEqual(self.host.reader.channel.settled_cpu, 0.0)
        self.case.child.reaped = True
        self.case.watch.pulse()
        self.assertFalse(self.case.supervisor.busy())

    def test_direct_dense_reuse_without_private_operation_is_refused(self):
        with self.assertRaises(HostError):
            self.host.acquire({"model_id": self.fixture.identifier})
        self.assertEqual(self.host.leases, self.original_leases)
        self.host.static_admission.assert_not_called()

    def test_disabled_policy_and_foreign_id_cannot_take_ungated_reuse_path(self):
        self.case.app.dense_policy = None
        with self.assertRaises(ValueError):
            self.request()
        self.assertEqual(self.host.leases, self.original_leases)
        self.case.app.dense_policy = self.sealed
        self.case.app.begin_static.reset_mock()
        with self.assertRaises(ValueError):
            self.request("m_" + "f" * 64)
        self.case.app.begin_static.assert_not_called()
        self.assertEqual(self.host.leases, self.original_leases)

    def test_true_hint_with_private_admission_false_still_refuses(self):
        self.hint.return_value = True
        self.host.static_admission.return_value = False
        with self.assertRaises(HostError):
            self.request()
        self.assertEqual(self.host.leases, self.original_leases)
        self.hint.assert_not_called()
        self.handler.static_finalizer.abort()


if __name__ == "__main__":
    unittest.main()
