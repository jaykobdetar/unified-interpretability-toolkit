"""Pure work-ledger and progress projection contracts; no clocks or workers run."""

from copy import deepcopy
import unittest
from unittest.mock import Mock

from atlas_host.budget import WorkGrant
from atlas_host.profile_adapter import native_progress
import host_profile_adapter as fixtures


class ProfileLedgerBoundaryTests(unittest.TestCase):
    def ok(self, function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except BaseException as error:
            self.fail(f"Valid inert call raised {type(error).__name__}: {error}")

    def grant(self):
        clock, cpu = Mock(return_value=3.25), Mock(return_value=1.5)
        grant = self.ok(
            WorkGrant, 9, 2750, 2100, clock=clock, cpu_clock=cpu, lease_ms=4000
        )
        return grant, clock, cpu

    def test_constructor_preserves_state_clock_identity_and_units(self):
        grant, clock, cpu = self.grant()
        self.assertIs(grant.clock, clock)
        self.assertIs(grant.cpu_clock, cpu)
        self.assertEqual(grant.maximum, 9)
        self.assertEqual(grant.wall_ms, 2750)
        self.assertEqual(grant.cpu_ms, 2100)
        self.assertEqual(grant.started, 3.25)
        self.assertEqual(grant.cpu_started, 1.5)
        self.assertEqual(grant.deadline, 6.0)
        self.assertEqual(grant.lease_seconds, 4.0)
        self.assertEqual(grant.lease_end, 7.25)
        self.assertEqual(grant.visited, 0)
        self.assertIs(grant.failed, False)
        self.assertIs(grant.in_flight, False)
        clock.assert_called_once_with()
        cpu.assert_called_once_with()

    def test_remaining_keeps_exact_subtraction_and_clock_reads(self):
        grant, clock, cpu = self.grant()
        clock.reset_mock()
        cpu.reset_mock()
        clock.return_value, cpu.return_value = 3.5, 1.625
        self.assertEqual(
            self.ok(grant.remaining), {"values": 9, "wall_ms": 2500, "cpu_ms": 1975}
        )
        self.assertEqual(clock.call_count, 2)
        cpu.assert_called_once_with()
        self.assertEqual(grant.visited, 0)

    def test_heartbeat_checks_once_and_updates_only_lease(self):
        grant, clock, _ = self.grant()
        grant.remaining = Mock(
            return_value={"values": 9, "wall_ms": 2250, "cpu_ms": 2100}
        )
        clock.return_value = 3.75
        clock.reset_mock()
        self.assertIsNone(self.ok(grant.heartbeat))
        grant.remaining.assert_called_once_with()
        clock.assert_called_once_with()
        self.assertEqual(grant.lease_end, 7.75)
        self.assertEqual(grant.deadline, 6.0)
        self.assertEqual(grant.visited, 0)

    def test_advance_preserves_callback_arguments_settlement_and_return(self):
        grant, _, _ = self.grant()
        grant.remaining = Mock(
            return_value={"values": 3, "wall_ms": 2250, "cpu_ms": 2100}
        )

        def advance(count, deadline):
            self.assertIs(grant.in_flight, True)
            self.assertEqual(count, 2)
            self.assertEqual(deadline, 6.0)
            return 1

        callback = Mock(side_effect=advance)
        self.assertEqual(self.ok(grant.advance, 2, callback), 1)
        callback.assert_called_once_with(2, 6.0)
        self.assertEqual(grant.remaining.call_count, 2)
        self.assertEqual(grant.visited, 1)
        self.assertIs(grant.failed, False)
        self.assertIs(grant.in_flight, False)

    def test_native_projection_preserves_exact_receipt_and_input_copies(self):
        selected = deepcopy(fixtures.BINDING)
        native = fixtures.native()
        remaining = {"values": 3, "wall_ms": 3000, "cpu_ms": 2500}
        before = deepcopy((selected, native, remaining))
        result = self.ok(native_progress, fixtures.MODEL, selected, native, remaining)
        self.assertEqual(
            result,
            {
                "model_id": fixtures.MODEL,
                "kind": "profile",
                "binding": fixtures.BINDING,
                "state": "running",
                "visited_values": 5,
                "total_values": 12,
                "elapsed_active_ms": 13,
                "remaining_authorized_work": remaining,
                "complete": False,
                "error": None,
            },
        )
        self.assertEqual((selected, native, remaining), before)
        result["binding"]["shape"][0] = 99
        result["remaining_authorized_work"]["values"] = 0
        self.assertEqual((selected, native, remaining), before)


if __name__ == "__main__":
    unittest.main()
