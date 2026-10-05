"""Nonblocking, dependency-injected disposable-profile lifecycle; NO HTTP activation.

The host supplies one dedicated serial job-owning context and an independently
scheduled watchdog. Neither status nor HTTP polling drives work. No platform
provider/default scheduler is installed here. See PROFILE-WORKER-INTERFACE.md.
"""

import math
import secrets
import threading
import time
from copy import deepcopy

from .startup_diagnostics import diagnose, diagnosed
from .common import canonical, digest, fields, integer, require
from .profile_snapshot import SnapshotStore
from .profile_observation import SpawnObservationPending

WALL_MAX = 5.0
CPU_MAX = 4.0
FINALIZE_RESERVE = 0.8
POLL_SLICE = 0.002


class ProfileJob:
    """Trusted internal object; capabilities are never included in status/receipts.

    platform: acquire(), source_check(), start_gate(), resources_ok(memfd_bytes),
    spawn(request, output_fd, input_fd), arm_watchdog(job), owner_cpu(), clock().
    token: current(), release(), poison().
    child: initialize(), sample()->{cpu_seconds,reaped,exit_code,receipt}, stop().
    sample must be bounded/nonblocking; CPU is cumulative for this owned child,
    including final wait4 rusage after reap. Ownership is held before initialize.
    """

    def __init__(self, selected, seed, tab_capability, context_id, platform):
        require(
            type(tab_capability) is str and len(tab_capability) >= 32,
            "Private tab capability required",
        )
        require(type(context_id) is str and context_id, "Owned context required")
        self.diagnostics = getattr(platform, "diagnostics", None)
        self.store = SnapshotStore(selected, seed, diagnostics=self.diagnostics)
        self.tab = tab_capability
        self.context = context_id
        self.platform = platform
        self.job_capability = secrets.token_hex(32)
        self.state = "idle"
        self.error = None
        self.child = self.token = self.watchdog = self.validation = None
        self.cancel_event = threading.Event()
        self.owner_thread = None
        self.cpu_child = 0.0
        self.receipt = None
        self.accepted = None
        self._accepted_snapshot = (
            None  # (publication serial, revision), not store.latest.
        )
        self._publication_before = self.store.publication_serial
        self._retire_failed_replacement = False
        self.expired = False
        self.lease_end = platform.clock() + 15

    def _owns(self, tab, job, context):
        require(
            type(tab) is str
            and type(job) is str
            and context == self.context
            and secrets.compare_digest(tab, self.tab)
            and secrets.compare_digest(job, self.job_capability),
            "Private profile owner mismatch",
        )

    def _owner_context(self):
        require(
            self.owner_thread == threading.get_ident(),
            "Wrong job-owning execution context",
        )

    def _cpu(self):
        elapsed = self.platform.owner_cpu() - self.cpu_start
        require(
            math.isfinite(elapsed) and elapsed >= 0, "Invalid job-local owner CPU clock"
        )
        return elapsed + self.cpu_child

    def _time_budget(self, reserve=0.0):
        now = self.platform.clock()
        require(
            not self.cancel_event.is_set() and not self.expired,
            "Profile cancelled/expired",
        )
        require(
            now < self.lease_end
            and now < self.deadline - reserve
            and self._cpu() < self.cpu_budget,
            "Total profile grant exhausted",
        )

    def _budget(self, reserve=0.0):
        self._time_budget(reserve)
        require(
            self.token is not None and self.token.current(),
            "Shared heavy-slot ownership lost",
        )

    def start(
        self, tab, job, context, values, *, wall_ms=5000, cpu_ms=4000, resume=False
    ):
        """Explicit action in the dedicated owner context; no queue or renewal.

        Public resume is deliberately closed pending runtime qualification.
        Stored restore primitives are available separately for scoped tests.
        """
        self._owns(tab, job, context)
        require(
            self.state in ("idle", "complete", "partial", "error"),
            "Existing job requires cleanup",
        )
        require(self.child is None and self.token is None, "Owned work still exists")
        require(not resume, "Resume unqualified; explicit Restart only")
        integer(values, 1, self.store.selected["rows"] * self.store.selected["cols"])
        integer(wall_ms, 1, 5000)
        integer(cpu_ms, 1, 4000)
        require(wall_ms > 1000, "Insufficient validation/cleanup margin")
        self.owner_thread = threading.get_ident()
        self.started = self.platform.clock()
        self.cpu_start = self.platform.owner_cpu()
        self.deadline = self.started + wall_ms / 1000
        self.cpu_budget = cpu_ms / 1000
        self.lease_end = self.started + 15
        self.cpu_child = 0.0
        self.values = values
        self.error = None
        self.accepted = None
        self.receipt = None
        self._publication_before = self.store.publication_serial
        self._retire_failed_replacement = False
        self.cancel_event.clear()
        self.expired = False
        self.state = "admitting"
        try:
            # Watchdog registration MUST NOT depend on HTTP requests/ticks.
            self.watchdog = self.platform.arm_watchdog(self)
            require(self.watchdog is not None, "Independent watchdog required")
            self.token = self.platform.acquire()  # Common slot; busy means refusal.
            require(self.token is not None, "Shared heavy slot busy")
            self._budget(FINALIZE_RESERVE)
            self.platform.start_gate()
            self.platform.source_check()
            output = self.store.begin()
            self._budget(FINALIZE_RESERVE)
            remaining = self.deadline - self.platform.clock() - FINALIZE_RESERVE
            cpu_remaining = self.cpu_budget - self._cpu()
            request = {
                "binding": deepcopy(self.store.selected),
                "seed": self.store.seed,
                "values": values,
                "wall_ms": math.floor(remaining * 1000),
                "cpu_ms": math.floor(cpu_remaining * 1000),
            }
            require(
                request["wall_ms"] > 200 and request["cpu_ms"] > 0,
                "No child budget remains",
            )
            # No HTTP/visitor paths. Platform resolves trusted registry source.
            self.child = self.platform.spawn(request, output, None)
            require(self.child is not None, "Worker spawn did not return ownership")
            self.child.initialize()  # Failure retains self.child for cleanup.
            self.state = "running"
            self.guard()
        except BaseException as diagnostic_error:
            diagnose(self, "profile_worker.ProfileJob.start.catch124", diagnostic_error)
            self._fail("admission_failed")
            raise
        return self.status(tab, job, context)

    def guard(self):
        """Independently scheduled nonblocking watchdog, not an HTTP poll hook.

        platform owner_cpu must query the job owner's clock even when called
        here from a watchdog thread; using this thread's CPU would be incorrect.
        """
        if self.state in (
            "idle",
            "complete",
            "partial",
            "error",
            "cancelled",
            "finishing",
        ):
            return
        try:
            try:
                require(
                    self.platform.resources_ok(self.store.owned_storage_bytes),
                    "Owned resources exceeded",
                )
            except SpawnObservationPending:
                if self.token is None and self.state == "admitting":
                    self._time_budget()
                else:
                    self._budget()
                return  # Watchdog-only transitional observation; never success.
            if self.cancel_event.is_set() and not self.expired:
                if self.child is not None:
                    self.child.stop()
                return
            if self.child is not None:
                # sample serializes its wait4/receipt ownership internally.
                sample = self.child.sample()
                self._sample(sample)
            if self.token is None and self.state == "admitting":
                self._time_budget()
            else:
                self._budget()
        except BaseException as diagnostic_error:
            diagnose(self, "profile_worker.ProfileJob.guard.catch151", diagnostic_error)
            self.expired = True
            self.cancel_event.set()
            if self.child is not None:
                self.child.stop()  # Signal only; never wait in watchdog/cancel.

    def _sample(self, sample):
        fields(sample, ("cpu_seconds", "reaped", "exit_code", "receipt"))
        cpu = sample["cpu_seconds"]
        require(
            type(cpu) in (int, float) and math.isfinite(cpu) and cpu >= self.cpu_child,
            "Owned child CPU clock regressed",
        )
        require(type(sample["reaped"]) is bool, "Invalid reap evidence")
        self.cpu_child = cpu
        self.sampled = sample

    def cancel(self, tab, job, context):
        self._owns(tab, job, context)
        self.cancel_event.set()
        # Owner context performs handle cleanup. No synchronous wait or scan.
        if self.child is not None:
            self.child.stop()

    def heartbeat(self, tab, job, context):
        self._owns(tab, job, context)
        now = self.platform.clock()
        require(
            not self.cancel_event.is_set() and now < self.lease_end,
            "Owner lease expired",
        )
        self.lease_end = now + 15  # Never changes deadline, CPU budget or values.

    def status(self, tab, job, context):
        self._owns(tab, job, context)
        return {
            "state": self.state,
            "error": self.error,
            "accepted": None if self.accepted is None else dict(self.accepted),
            "cleanup_pending": self.state == "stopping",
            "resume_available": False,
        }

    def page(self, tab, job, context, revision, axis, start, count):
        self._owns(tab, job, context)
        require(
            not self.cancel_event.is_set() and self.platform.clock() < self.lease_end,
            "Profile owner lease expired",
        )
        accepted = self._accepted_snapshot
        require(
            accepted is not None and accepted[1] == revision,
            "Snapshot revision has not been accepted",
        )
        return self.store.page(
            revision,
            axis,
            start,
            count,
            source_check=self.platform.source_check,
            expected_publication=accepted[0],
        )

    def _fail(self, code):
        if self.store.publication_serial != self._publication_before:
            # Detect the swap even if old-close raised before the generator
            # returned, or the first post-swap budget check failed.
            self._accepted_snapshot = None
            self._retire_failed_replacement = True
        self.error = code
        self.state = "stopping"
        if self.validation is not None:
            self.validation.close()
            self.validation = None
        if self.child is not None:
            self.child.stop()

    def _finish_cleanup(self):
        # Owner-context only. A lost/unreapable child keeps token poisoned/held.
        if self.child is not None:
            sample = self.child.sample()
            self._sample(sample)
            if not sample["reaped"]:
                if self.token is not None:
                    self.token.poison()
                return False
            self.child = None
        self.store.abort_candidate()
        if self.cancel_event.is_set() or self._retire_failed_replacement:
            self._accepted_snapshot = None
            self.store.close()  # Cancellation or failed swap retires all results.
        self.store.require_settled()  # Uncertain closes retain/poison the slot.
        if self.token is not None:
            self.token.release()
            self.token = None
        if self.watchdog is not None:
            self.watchdog.close()
            self.watchdog = None
        self.state = (
            "cancelled" if self.cancel_event.is_set() and not self.expired else "error"
        )
        return True

    def tick(self):
        """Owner scheduler calls independently of HTTP, bounded validation step.

        Platform callbacks must be nonblocking/bounded. Hard watchdog remains
        independent even if a callback or filesystem syscall becomes slow.
        """
        self._owner_context()
        if self.cancel_event.is_set() and self.state in (
            "idle",
            "partial",
            "complete",
            "error",
        ):
            self._fail("cancelled")
        if self.state == "stopping":
            try:
                return self._finish_cleanup()
            except BaseException as diagnostic_error:
                diagnose(
                    self, "profile_worker.ProfileJob.tick.catch239", diagnostic_error
                )
                if self.token is not None:
                    self.token.poison()
                return False
        if self.state not in ("running", "validating"):
            return False
        try:
            self.guard()
            self._budget()
            if self.state == "running":
                sample = self.child.sample()
                self._sample(sample)
                if not sample["reaped"]:
                    return False
                self.child = None
                require(sample["exit_code"] == 0, "Worker failed")
                receipt = sample["receipt"]
                fields(
                    receipt,
                    (
                        "schema",
                        "revision",
                        "visited_values",
                        "new_values",
                        "frame_bytes",
                        "live_bytes",
                    ),
                )
                require(
                    len(canonical(receipt)) <= 16384
                    and receipt["schema"] == "weight-atlas.profile-candidate.v1",
                    "Invalid candidate receipt",
                )
                digest(receipt["revision"])
                integer(receipt["visited_values"], 0, self.values)
                require(
                    receipt["new_values"] == receipt["visited_values"]
                    and receipt["frame_bytes"] == self.store.frame_bytes
                    and receipt["live_bytes"] == self.store.live_bytes,
                    "Candidate allowance/layout mismatch",
                )
                self.receipt = receipt
                self._budget()
                self.platform.source_check()
                self.validation = self.store.publication_steps(
                    receipt["revision"],
                    minimum_visited=receipt["visited_values"],
                    maximum_visited=receipt["visited_values"],
                    deadline=self.deadline,
                    source_check=self.platform.source_check,
                    final_check=self._budget,
                    clock=self.platform.clock,
                )
                self.state = "validating"
            end = min(self.deadline, self.platform.clock() + POLL_SLICE)
            # Both wall slice and fixed step count: mock/static clocks cannot
            # accidentally turn one host interaction into an unbounded scan.
            for _ in range(4):
                if self.platform.clock() >= end:
                    break
                self._budget()
                try:
                    next(self.validation)
                except StopIteration as done:
                    self.validation = None
                    info = done.value
                    self._budget()  # Includes publication and old-handle close.
                    self.state = "finishing"
                    self.watchdog.close()
                    self.watchdog = None
                    self._time_budget()
                    self.token.release()
                    self.token = None
                    self._time_budget()
                    # Terminal progress only after reap, validation and cleanup.
                    self._accepted_snapshot = (
                        self.store.publication_serial,
                        info.revision,
                    )
                    self.accepted = {
                        "revision": info.revision,
                        "visited_values": info.visited,
                        "total_values": info.total,
                        "complete": info.visited == info.total,
                    }
                    self.state = "complete" if info.visited == info.total else "partial"
                    return True
        except BaseException as diagnostic_error:
            diagnose(self, "profile_worker.ProfileJob.tick.catch287", diagnostic_error)
            # A late publication/cleanup is never a terminal success. Retire
            # published handles too if the final cleanup check missed budget.
            if self.state == "finishing":
                self.expired = True
                self.cancel_event.set()
            self._fail("resource_limit" if self.expired else "worker_error")
        return False
