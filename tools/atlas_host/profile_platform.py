"""Inactive supervised provider for ProfileJob. No default OS provider or routes.

This adapts the shared supervisor, HTTP-anchored grant, dedicated owner executor,
independent watchdog and explicit whole-tree observations. The injected OS hooks
must be reviewed/qualified before a new hosted launcher can install this module.
"""

from copy import deepcopy

from .startup_diagnostics import diagnose, diagnosed
from .common import fields, integer, require
from .profile_worker import ProfileJob
from .profile_observation import SpawnObservationPending
from .validation_policy import check_start

MIB = 1024**2
GIB = 1024**3
TERMINAL = {"complete", "partial", "error", "cancelled"}


class DeferredToken:
    """Primitive release is intent; supervisor release follows terminal checks."""

    def __init__(self, platform):
        self.platform = platform
        self.requested = False

    def current(self):
        return self.platform.reserved.current() and not self.requested

    def poison(self):
        self.platform.reserved.poison()

    def release(self):
        self.requested = True


class ProfilePlatform:
    """hooks: clock/cpu are in grant; other hooks never spawn implicitly.

    Required hooks: start_gate()->(available_bytes,disk_bytes), source_check(),
    spawn(request,output_fd,input_fd)->owned child, resources()->strict snapshot,
    watch(callback, deadline)->independent registration.close().
    watch must start independently before returning and close must be bounded.
    """

    def __init__(self, supervisor, reserved, grant, hooks):
        require(
            reserved.kind == "profile" and reserved.current(),
            "Reserved profile operation required",
        )
        self.supervisor, self.reserved, self.grant, self.hooks = (
            supervisor,
            reserved,
            grant,
            hooks,
        )
        self.diagnostics = getattr(hooks, "diagnostics", None)
        self.token = DeferredToken(self)
        self.child = None
        self.watch_closed = False
        self.spawn_attempted = False
        self.runtime = None
        self.closed = False

    def clock(self):
        return self.grant.clock()

    def owner_cpu(self):
        return self.grant.cpu.total()

    def acquire(self):
        self.grant.remaining()
        require(self.reserved.current(), "Supervised compute ownership lost")
        return self.token

    def start_gate(self):
        self.grant.remaining()
        available, disk = self.hooks.start_gate()
        check_start(getattr(self.hooks, "launch_policy", None), available, disk)

    def source_check(self):
        # Page reads after a completed grant still validate source/owner context,
        # but cannot create another grant or revive numeric work.
        self.hooks.source_check()
        if not self.closed:
            self.grant.remaining()

    def spawn(self, request, output_fd, input_fd):
        require(
            not self.spawn_attempted and input_fd is None,
            "One disposable Restart child only",
        )
        self.spawn_attempted = True
        remaining = self.grant.remaining(reserve_ms=800)
        request = deepcopy(request)
        request["wall_ms"] = min(request["wall_ms"], remaining["wall_ms"])
        request["cpu_ms"] = min(request["cpu_ms"], remaining["cpu_ms"])
        require(
            request["wall_ms"] > 200 and request["cpu_ms"] > 0,
            "Admission consumed child allowance",
        )
        # Hook resolves frozen binary/owner receipt; no paths from HTTP request.
        # Must return ownership before fallible post-spawn initialization.
        self.child = self.hooks.spawn(request, output_fd, None)
        require(self.child is not None, "Missing owned child")
        return self.child

    def resources_ok(self, frame_bytes):
        integer(frame_bytes, 0, 32 * MIB)
        pending = None
        try:
            observed = self.hooks.resources()
        except SpawnObservationPending as error:
            pending, observed = error, error.observed
        fields(
            observed,
            (
                "rss_bytes",
                "available_bytes",
                "all_owned_accounted",
                "descendants_clear",
            ),
        )
        integer(observed["rss_bytes"])
        integer(observed["available_bytes"])
        limits_ok = (
            observed["descendants_clear"] is True
            and observed["rss_bytes"]
            + max(frame_bytes, self.supervisor.snapshot_reservation)
            <= 768 * MIB
            and observed["available_bytes"] >= 13 * GIB // 4
        )
        if pending is not None and limits_ok:
            raise pending  # A complete ownership observation is still required.
        okay = limits_ok and observed["all_owned_accounted"] is True
        if not okay:
            diagnose(
                self,
                "profile_platform.resource_refusal",
                frame_bytes=frame_bytes,
                reservation_bytes=self.supervisor.snapshot_reservation,
                **observed,
            )
        return okay

    def arm_watchdog(self, job):
        platform = self

        def guard():
            try:
                if platform.child is not None:
                    sample = platform.child.sample()
                    platform.grant.sample_child(sample["cpu_seconds"])
                platform.grant.remaining()
                require(platform.reserved.current(), "Shared ownership uncertain")
                try:
                    require(
                        platform.resources_ok(job.store.owned_storage_bytes),
                        "Profile resource envelope exceeded",
                    )
                except SpawnObservationPending:
                    # Still enforce the original deadline/CPU and exact token.
                    # No accepted result or ownership release occurs here.
                    platform.grant.remaining()
                    require(platform.reserved.current(), "Shared ownership uncertain")
                    return
                job.guard()
            except BaseException as diagnostic_error:
                diagnose(
                    self,
                    "profile_platform.ProfilePlatform.arm_watchdog.guard.catch109",
                    diagnostic_error,
                )
                job.expired = True
                job.cancel_event.set()
                if platform.child is not None:
                    platform.child.stop()
                platform.reserved.poison()

        registration = self.hooks.watch(guard, self.grant.deadline)
        require(registration is not None, "Independent watchdog registration required")

        class Registration:
            def close(self):
                registration.close()
                platform.watch_closed = True

        return Registration()

    def finish(self, job):
        """Owner context only; called before an API-visible terminal snapshot."""
        if self.closed:
            return True
        if (
            job.state not in TERMINAL
            or job.child is not None
            or job.validation is not None
        ):
            return False
        if job.watchdog is not None or job.store.pending is not None:
            return False
        if not self.watch_closed and hasattr(self.hooks, "close_admission_watch"):
            self.hooks.close_admission_watch()
            self.watch_closed = True
        try:
            job.store.require_settled()
        except BaseException as diagnostic_error:
            diagnose(
                self,
                "profile_platform.ProfilePlatform.finish.catch136",
                diagnostic_error,
            )
            self.reserved.poison()
            return False
        if self.child is not None:
            sample = self.child.sample()
            self.grant.sample_child(sample["cpu_seconds"])
            if sample["reaped"] is not True:
                self.reserved.poison()
                return False
        # A failed spawn that did not return a handle has an explicit hook proof.
        if self.spawn_attempted and self.child is None:
            if not self.hooks.no_child_created():
                self.reserved.poison()
                return False
        observed = self.hooks.resources()
        if (
            observed.get("all_owned_accounted") is not True
            or observed.get("descendants_clear") is not True
        ):
            self.reserved.poison()
            return False
        if job.state in ("complete", "partial"):
            try:
                self.grant.remaining()
                require(self.watch_closed, "Watchdog cleanup not confirmed")
                require(
                    self.resources_ok(
                        job.store.frame_bytes * int(job.store.latest is not None)
                    ),
                    "Final resource envelope exceeded",
                )
            except BaseException as diagnostic_error:
                diagnose(
                    self,
                    "profile_platform.ProfilePlatform.finish.catch160",
                    diagnostic_error,
                )
                job.accepted = None
                job.state, job.error = "error", "resource_limit"
                job.store.close()
        if self.child is None:
            self.supervisor.admission_aborted(self.reserved, no_child_created=True)
        else:
            self.supervisor.child_finished(
                self.reserved, reaped=True, descendants_clear=True, finalized=True
            )
        self.reserved.release()
        self.closed = True
        return True


class SupervisedProfile:
    """One session-owned result, independent of HTTP. Caller owns its executor.

    Construction and start occur in the same dedicated owner context. The outer
    grant was already captured at HTTP admission; no work-queue waiting allowed.
    """

    def __init__(
        self, supervisor, grant, hooks, selected, seed, tab, context, *, reserved=None
    ):
        self.diagnostics = getattr(hooks, "diagnostics", None)
        self.platform = None
        token = reserved or supervisor.acquire("profile", context)
        require(
            token.kind == "profile" and token.context == context,
            "Profile reservation mismatch",
        )
        try:
            grant.remaining()
            self.platform = ProfilePlatform(supervisor, token, grant, hooks)
            self.job = ProfileJob(selected, seed, tab, context, self.platform)
            supervisor.retain_profile(self, self.job.store.frame_bytes)
            self.auth = (tab, self.job.job_capability, context)
        except BaseException as diagnostic_error:
            diagnose(
                self,
                "profile_platform.SupervisedProfile.__init__.catch190",
                diagnostic_error,
            )
            # No child/memfd is created by the primitive constructor.
            supervisor.admission_aborted(token, no_child_created=True)
            token.release()
            raise

    def start(self, values):
        try:
            remaining = self.platform.grant.remaining()
            self.job.start(*self.auth, values, **remaining)
        except BaseException as diagnostic_error:
            diagnose(
                self,
                "profile_platform.SupervisedProfile.start.catch200",
                diagnostic_error,
            )
            if (
                self.job.state in ("idle", "complete", "partial", "error")
                and self.job.child is None
            ):
                self.job.state, self.job.error = "error", "admission_failed"
            raise
        finally:
            self.platform.finish(self.job)

    def restart(self, grant, hooks, values):
        require(
            self.platform.closed and self.job.state in ("complete", "partial", "error"),
            "Previous operation requires cleanup",
        )
        token = self.platform.supervisor.acquire("profile", self.auth[2])
        self.platform = ProfilePlatform(self.platform.supervisor, token, grant, hooks)
        self.job.platform = self.platform
        self.start(values)

    def tick(self):
        self.job.tick()
        self.platform.finish(self.job)

    def status(self, tab, job, context):
        result = self.job.status(tab, job, context)
        # Primitive completion is not public until outer anchored ledger and
        # native/shared ownership finalization have been verified too.
        if result["state"] in TERMINAL and not self.platform.closed:
            return {
                "state": "stopping",
                "error": result["error"],
                "accepted": None,
                "cleanup_pending": True,
                "resume_available": False,
            }
        return result

    def page(self, tab, capability, context, revision, axis, start, count):
        # Never trust store.latest as proof of accepted public revision.
        self.job._owns(tab, capability, context)
        require(
            self.platform.closed
            and self.job.state in ("complete", "partial")
            and self.job.accepted is not None
            and self.job.accepted["revision"] == revision,
            "Profile revision has not completed final acceptance",
        )
        return self.job.page(tab, capability, context, revision, axis, start, count)

    def close(self):
        require(
            self.platform.closed and self.job.child is None,
            "Worker cleanup must finish before session close",
        )
        self.job.store.close()
        require(
            self.platform.hooks.snapshot_closed(self.job.store) is True,
            "All snapshot handles, including retiring storage, must be closed",
        )
        self.platform.supervisor.close_profile(self, all_handles_closed=True)
