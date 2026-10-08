"""Supervisor accounting and ownership contracts using inert local objects."""

from collections.abc import Callable
from contextlib import ExitStack
from types import SimpleNamespace, TracebackType
from typing import TypeVar
import unittest
from unittest.mock import patch

from atlas_host.supervisor import (
    AdmissionGrant,
    ComputeToken,
    CpuLedger,
    HostedLegacyLane,
    Supervisor,
)

T = TypeVar("T")


class LockProbe:
    def __init__(self) -> None:
        self.enters = 0

    def __enter__(self) -> "LockProbe":
        self.enters += 1
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass


class SupervisorContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for target in (
            "subprocess.Popen",
            "socket.socketpair",
            "threading.Thread.start",
            "os.pidfd_open",
            "os.kill",
            "signal.pidfd_send_signal",
            "os.memfd_create",
        ):
            self.stack.enter_context(
                patch(target, side_effect=AssertionError("Live work forbidden"))
            )

    def success(self, operation: Callable[[], T]) -> T:
        try:
            return operation()
        except BaseException as error:
            self.fail(f"Unexpected {type(error).__name__}: {error}")

    def refuses(self, message: str, operation: Callable[[], object]) -> None:
        caught = None
        try:
            operation()
        except BaseException as error:
            caught = error
        self.assertIs(type(caught), ValueError)
        self.assertEqual(str(caught), message)

    def test_ledger_copies_meter_map_and_keeps_cumulative_monotonic_clocks(
        self,
    ) -> None:
        parts = {"admission": 1.0, "owner": 2.0, "watchdog": 3.0}
        meters = {key: lambda key=key: parts[key] for key in parts}
        ledger = self.success(lambda: CpuLedger(meters))
        self.assertEqual(self.success(ledger.total), 6.0)
        self.assertEqual(self.success(ledger.total), 6.0)
        meters["admission"] = lambda: 99.0
        self.assertEqual(self.success(ledger.total), 6.0)
        parts["owner"] = 2.5
        self.assertEqual(self.success(ledger.total), 6.5)
        parts["owner"] = 2.0
        self.refuses("Job CPU clock regressed/unavailable", ledger.total)
        zero = self.success(lambda: CpuLedger({key: lambda: 0.0 for key in parts}))
        self.assertEqual(self.success(zero.total), 0.0)

    def test_grant_preserves_initial_clocks_reserve_and_child_replacement(self) -> None:
        clock = SimpleNamespace(wall=2.0, cpu=1.0)
        ledger = SimpleNamespace(total=lambda: clock.cpu)
        grant = self.success(lambda: AdmissionGrant(lambda: clock.wall, ledger))
        self.assertEqual(
            self.success(grant.remaining), {"wall_ms": 5000, "cpu_ms": 4000}
        )
        self.success(lambda: grant.sample_child(0.0))
        self.success(lambda: grant.sample_child(0.5))
        self.success(lambda: grant.sample_child(0.5))
        self.refuses("Child CPU clock regressed", lambda: grant.sample_child(0.25))
        clock.wall, clock.cpu = 2.5, 1.25
        self.assertEqual(
            self.success(lambda: grant.remaining(750)),
            {"wall_ms": 3750, "cpu_ms": 3250},
        )
        probe = LockProbe()
        grant.lock = probe
        self.success(grant.cancel)
        self.assertEqual(probe.enters, 1)
        self.refuses("Original admission cancelled", grant.remaining)

    def test_token_identity_poison_and_release_require_owned_completion(self) -> None:
        supervisor = self.success(Supervisor)
        token = self.success(lambda: supervisor.acquire("profile", "context"))
        self.assertEqual((token.kind, token.context), ("profile", "context"))
        self.assertRegex(token.operation, r"^[0-9a-f]{32}$")
        self.assertIs(supervisor.active, token)
        self.assertFalse(token.poisoned)
        self.assertFalse(token._completion)
        self.assertTrue(self.success(token.current))
        self.refuses("Confirmed completion/reap required before release", token.release)
        other = self.success(
            lambda: ComputeToken(supervisor, "other", "profile", "other")
        )
        self.assertFalse(self.success(other.current))
        self.refuses("Stale compute token", other.poison)
        self.success(token.poison)
        self.assertTrue(token.poisoned)
        self.assertFalse(self.success(token.current))
        self.assertTrue(self.success(supervisor.busy))
        self.success(lambda: supervisor.admission_aborted(token, no_child_created=True))
        self.success(token.release)
        self.assertIsNone(supervisor.active)
        self.assertFalse(self.success(supervisor.busy))

    def test_initial_state_and_stop_retain_active_ownership(self) -> None:
        supervisor = self.success(Supervisor)
        self.assertIsNone(supervisor.active)
        self.assertIsNone(supervisor.profile_session)
        self.assertEqual(supervisor.snapshot_reservation, 0)
        self.assertFalse(supervisor.stopping)
        self.assertFalse(self.success(supervisor.busy))
        self.assertEqual(self.success(supervisor.charged_snapshot_bytes), 0)
        token = self.success(lambda: supervisor.acquire("profile", "context"))
        self.success(supervisor.prevent_admission)
        self.assertTrue(supervisor.stopping)
        self.assertIs(supervisor.active, token)
        self.assertTrue(token.poisoned)
        self.assertFalse(token._completion)
        self.assertTrue(self.success(supervisor.busy))
        self.refuses(
            "Hosted provider is stopping", lambda: supervisor.acquire("tile", "context")
        )

    def test_snapshot_charge_covers_retiring_storage_and_exact_owner_close(
        self,
    ) -> None:
        supervisor = self.success(Supervisor)
        store = SimpleNamespace(owned_storage_bytes=0)
        owner = SimpleNamespace(job=SimpleNamespace(store=store))
        self.success(lambda: supervisor.retain_profile(owner, 16))
        self.assertIs(supervisor.profile_session, owner)
        self.assertEqual(supervisor.snapshot_reservation, 32)
        for actual, expected in ((0, 32), (31, 32), (32, 32), (33, 33)):
            store.owned_storage_bytes = actual
            self.assertEqual(self.success(supervisor.charged_snapshot_bytes), expected)
        self.refuses(
            "Profile storage cleanup unconfirmed",
            lambda: supervisor.close_profile(object(), all_handles_closed=True),
        )
        self.refuses(
            "Profile storage cleanup unconfirmed",
            lambda: supervisor.close_profile(owner, all_handles_closed=False),
        )
        self.success(lambda: supervisor.close_profile(owner, all_handles_closed=True))
        self.assertIsNone(supervisor.profile_session)
        self.assertEqual(supervisor.snapshot_reservation, 0)
        self.success(lambda: supervisor.retain_profile(owner, 16 * 1024**2))
        self.assertEqual(supervisor.snapshot_reservation, 32 * 1024**2)

    def test_ownership_check_and_inert_admission_completion(self) -> None:
        supervisor = self.success(Supervisor)
        token = self.success(lambda: supervisor.acquire("profile", "context"))
        other = self.success(
            lambda: ComputeToken(supervisor, "other", "profile", "other")
        )
        self.success(lambda: supervisor._owns(token))
        self.refuses("Stale compute operation", lambda: supervisor._owns(other))
        self.refuses(
            "Admission cleanup is uncertain",
            lambda: supervisor.admission_aborted(token, no_child_created=False),
        )
        self.assertFalse(token._completion)
        self.success(lambda: supervisor.admission_aborted(token, no_child_created=True))
        self.assertTrue(token._completion)
        self.success(token.release)

    def test_child_completion_needs_all_three_receipts(self) -> None:
        supervisor = self.success(Supervisor)
        token = self.success(lambda: supervisor.acquire("profile", "context"))
        for reaped, clear, finalized in (
            (False, True, True),
            (True, False, True),
            (True, True, False),
        ):
            self.refuses(
                "Owned work is not finalized",
                lambda: supervisor.child_finished(
                    token, reaped=reaped, descendants_clear=clear, finalized=finalized
                ),
            )
            self.assertFalse(token._completion)
        self.success(
            lambda: supervisor.child_finished(
                token, reaped=True, descendants_clear=True, finalized=True
            )
        )
        self.assertTrue(token._completion)
        self.success(token.release)

    def test_native_command_and_exact_ack_retain_request_identity(self) -> None:
        supervisor = self.success(Supervisor)
        token = self.success(lambda: supervisor.acquire("tile", "context"))
        payload = {"tensor": 0}
        command = self.success(lambda: supervisor.native_command(token, payload, 7000))
        self.assertEqual(
            command,
            {
                "version": 1,
                "operation_id": token.operation,
                "kind": "tile",
                "remaining_ms": 7000,
                "request": payload,
            },
        )
        self.assertIs(command["request"], payload)
        ack = {
            "version": 1,
            "operation_id": token.operation,
            "complete": True,
            "numeric_idle": True,
        }
        self.success(lambda: supervisor.native_ack(token, ack))
        self.assertTrue(token._completion)
        self.success(token.release)
        self.assertFalse(self.success(supervisor.busy))

    def test_transport_loss_keeps_busy_until_confirmed_renderer_reap(self) -> None:
        supervisor = self.success(Supervisor)
        token = self.success(lambda: supervisor.acquire("calibration", "context"))
        self.success(lambda: supervisor.transport_lost(token))
        self.assertTrue(token.poisoned)
        self.assertFalse(token._completion)
        self.assertFalse(supervisor.stopping)
        self.assertTrue(self.success(supervisor.busy))
        self.refuses(
            "Owned renderer reap not confirmed",
            lambda: supervisor.renderer_reaped(
                token, reaped=True, descendants_clear=False
            ),
        )
        self.success(
            lambda: supervisor.renderer_reaped(
                token, reaped=True, descendants_clear=True
            )
        )
        self.assertTrue(token._completion)
        self.success(token.release)
        self.assertFalse(self.success(supervisor.busy))
        next_token = self.success(lambda: supervisor.acquire("tile", "next"))
        self.assertTrue(self.success(next_token.current))

    def test_legacy_lane_retains_objects_and_each_existing_busy_reason(self) -> None:
        supervisor = self.success(Supervisor)
        inference = SimpleNamespace(process=None, status="idle")
        analytics = SimpleNamespace(busy=False)
        lane = self.success(lambda: HostedLegacyLane(supervisor, inference, analytics))
        self.assertIs(lane.supervisor, supervisor)
        self.assertIs(lane.inference, inference)
        self.assertIs(lane.analytics, analytics)
        self.assertFalse(self.success(lane.busy))
        inference.process = object()
        self.assertTrue(self.success(lane.busy))
        inference.process = None
        for status in ("loading", "running", "stopping"):
            inference.status = status
            self.assertTrue(self.success(lane.busy))
        inference.status = "idle"
        analytics.busy = True
        self.assertTrue(self.success(lane.busy))

    def test_legacy_reservation_and_completion_keep_kind_context_and_busy_guard(
        self,
    ) -> None:
        supervisor = self.success(Supervisor)
        inference = SimpleNamespace(process=None, status="idle")
        analytics = SimpleNamespace(busy=False)
        lane = self.success(lambda: HostedLegacyLane(supervisor, inference, analytics))
        for kind in ("inference", "analytics"):
            token = self.success(lambda: lane.reserve(kind, "context"))
            self.assertEqual((token.kind, token.context), (kind, "context"))
            self.success(
                lambda: lane.finish(
                    token, reaped=True, descendants_clear=True, finalized=True
                )
            )
            self.assertFalse(self.success(supervisor.busy))
        inference.process = object()
        self.refuses(
            "Hosted compute busy; no queue",
            lambda: lane.reserve("analytics", "context"),
        )


if __name__ == "__main__":
    unittest.main()
