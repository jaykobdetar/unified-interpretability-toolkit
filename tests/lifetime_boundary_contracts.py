"""Lifetime ownership and callbacks use inert, bounded resource doubles."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from atlas_host import lifetime_guard as module
from atlas_host.profile_observation import SpawnObservationPending


class LifetimeBoundaryTests(unittest.TestCase):
    def make(self):
        observed = {
            "rss_bytes": 17,
            "available_bytes": 29,
            "all_owned_accounted": True,
            "descendants_clear": True,
        }
        book = SimpleNamespace(
            resources=Mock(return_value=observed), request_stop_all=Mock()
        )
        supervisor = SimpleNamespace(
            prevent_admission=Mock(), charged_snapshot_bytes=Mock(return_value=31)
        )
        cleanup, diagnostics = Mock(), SimpleNamespace(event=Mock())
        guard = module.LifetimeGuard(book, supervisor, cleanup, diagnostics=diagnostics)
        self.assertIs(guard.book, book)
        self.assertIs(guard.supervisor, supervisor)
        self.assertIs(guard.request_cleanup, cleanup)
        self.assertIs(guard.diagnostics, diagnostics)
        self.assertIs(guard.failed, False)
        self.assertIs(guard.stopping, False)
        return guard, book, supervisor, cleanup, diagnostics, observed

    def ok(self, function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except BaseException as error:
            self.fail(
                f"Valid bounded lifetime call raised {type(error).__name__}: {error}"
            )

    def test_constructor_retains_exact_owners_and_optional_diagnostics(self):
        guard, book, supervisor, cleanup, _, _ = self.make()
        plain = module.LifetimeGuard(book, supervisor, cleanup)
        self.assertIsNone(plain.diagnostics)
        self.assertIs(plain.book, guard.book)
        self.assertIs(plain.supervisor, guard.supervisor)
        self.assertIs(plain.request_cleanup, guard.request_cleanup)
        self.assertIs(plain.failed, False)
        self.assertIs(plain.stopping, False)

    def test_shutdown_marks_stopping_before_ordered_owner_callbacks(self):
        guard, book, supervisor, cleanup, _, _ = self.make()
        events = []

        def record(name):
            self.assertIs(guard.stopping, True)
            events.append(name)

        supervisor.prevent_admission.side_effect = lambda: record("admission")
        cleanup.side_effect = lambda: record("cleanup")
        book.request_stop_all.side_effect = lambda: record("stop")
        self.assertIsNone(self.ok(guard.request_shutdown))
        self.assertEqual(events, ["admission", "cleanup", "stop"])
        self.assertIs(guard.failed, False)

    def test_pulse_passes_exact_observation_charge_and_false_pending(self):
        guard, book, supervisor, cleanup, diagnostics, observed = self.make()
        with patch.object(module, "check_memory") as memory:
            self.assertIs(self.ok(guard.pulse), True)
        book.resources.assert_called_once_with()
        supervisor.charged_snapshot_bytes.assert_called_once_with()
        memory.assert_called_once_with(observed, 31, allow_pending=False)
        self.assertIs(memory.call_args.args[0], observed)
        cleanup.assert_not_called()
        diagnostics.event.assert_not_called()
        book.request_stop_all.assert_not_called()

    def test_pending_observation_retains_exact_values_and_charge(self):
        guard, book, _, cleanup, _, observed = self.make()
        pending = SpawnObservationPending(observed)
        book.resources.side_effect = pending
        with patch.object(module, "check_memory") as memory:
            self.assertIs(self.ok(guard.pulse), True)
        memory.assert_called_once_with(pending.observed, 31, allow_pending=True)
        self.assertIs(memory.call_args.args[0], pending.observed)
        cleanup.assert_not_called()

    def test_resource_refusal_diagnostic_sticky_failure_and_retry(self):
        guard, book, supervisor, cleanup, diagnostics, _ = self.make()
        refusal = ValueError("bounded observation unavailable")
        with patch.object(module, "check_memory", side_effect=refusal):
            self.assertIs(self.ok(guard.pulse), False)
        diagnostics.event.assert_called_once_with(
            "lifetime_guard.resource_refusal", refusal
        )
        self.assertIs(guard.failed, True)
        self.assertIs(guard.stopping, True)
        with patch.object(module, "check_memory"):
            self.assertIs(self.ok(guard.pulse), False)
        self.assertEqual(supervisor.prevent_admission.call_count, 2)
        self.assertEqual(cleanup.call_count, 2)
        self.assertEqual(book.request_stop_all.call_count, 2)

    def test_stop_refusal_keeps_failure_and_retries_owned_cleanup(self):
        guard, book, _, cleanup, diagnostics, _ = self.make()
        refusal = ValueError("bounded stop uncertain")
        guard.stopping = True
        book.request_stop_all.side_effect = refusal
        with patch.object(module, "check_memory"):
            self.assertIs(self.ok(guard.pulse), False)
            book.request_stop_all.side_effect = None
            self.assertIs(self.ok(guard.pulse), False)
        diagnostics.event.assert_called_once_with(
            "lifetime_guard.stop_pending", refusal
        )
        self.assertEqual(cleanup.call_count, 2)
        self.assertEqual(book.request_stop_all.call_count, 2)
        self.assertIs(guard.failed, True)
        self.assertIs(guard.stopping, True)


if __name__ == "__main__":
    unittest.main()
