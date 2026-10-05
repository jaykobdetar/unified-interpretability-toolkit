"""Inactive independent owner/watchdog execution and authenticated profile service."""

from copy import deepcopy
import secrets
import threading
import time

from .startup_diagnostics import diagnose, diagnosed
from .common import require
from .supervisor import AdmissionGrant, CpuLedger
from .profile_platform import SupervisedProfile
from .profile_os import ThreadMeter


class WatchSlot:
    def __init__(self, loop, callback, deadline):
        self.loop, self.callback, self.deadline = loop, callback, deadline
        self.running = False
        self.failed = False
        self.closed = False

    def register(self, callback, deadline):
        with self.loop.condition:
            require(
                not self.closed and not self.failed and deadline <= self.deadline,
                "Watchdog cannot renew grant",
            )
            self.callback = callback
            return self

    def disarm(self):
        """Stop new callbacks without waiting for this callback to join itself."""
        with self.loop.condition:
            require(not self.failed, "Independent watchdog callback failed")
            self.closed = True
            if not self.running and self in self.loop.slots:
                self.loop.slots.remove(self)

    def close(self):
        with self.loop.condition:
            self.closed = True
            require(
                self.loop.condition.wait_for(lambda: not self.running, timeout=0.15),
                "Independent watchdog cleanup uncertain",
            )
            require(not self.failed, "Independent watchdog callback failed")
            if self in self.loop.slots:
                self.loop.slots.remove(self)


class WatchdogLoop:
    def __init__(self, meter, *, clock=time.monotonic, diagnostics=None):
        self.diagnostics = diagnostics
        self.meter, self.clock = meter, clock
        self.condition = threading.Condition()
        self.slots = []
        self.stopped = False
        self.ready = threading.Event()
        self.thread = None
        self.lifetime = None

    def set_lifetime(self, callback):
        require(
            self.thread is None and self.lifetime is None,
            "Lifetime watch already installed",
        )
        self.lifetime = callback

    def bind(self, callback, deadline):
        with self.condition:
            self.slots = [
                s for s in self.slots if not s.closed or s.running or s.failed
            ]
            require(
                not self.stopped
                and not any(s.failed for s in self.slots)
                and len(self.slots) < 2,
                "Watchdog unavailable",
            )
            slot = WatchSlot(self, callback, deadline)
            self.slots.append(slot)
            self.condition.notify_all()
            return slot

    def pulse(self):
        # Public only to deterministic doubles; production loop owns calls.
        with self.condition:
            slots = list(self.slots)
            lifetime = self.lifetime
        if lifetime is not None:
            lifetime()  # Guard handles/refuses errors; independent of operation slots.
        for slot in slots:
            with self.condition:
                if slot.closed:
                    continue
                slot.running = True
                callback = slot.callback
            try:
                callback()
            except BaseException as diagnostic_error:
                diagnose(
                    self, "profile_service.WatchdogLoop.pulse.catch75", diagnostic_error
                )
                slot.failed = True  # Retain failure; never kill the watchdog loop.
            finally:
                with self.condition:
                    slot.running = False
                    if slot.closed and not slot.failed and slot in self.slots:
                        self.slots.remove(slot)
                    self.condition.notify_all()

    @diagnosed("profile_service.execution_thread")
    def run(self):
        self.meter.register()
        self.ready.set()
        try:
            while True:
                with self.condition:
                    if self.stopped:
                        return
                    self.condition.wait(
                        0.05 if self.slots or self.lifetime is not None else None
                    )
                self.pulse()
        finally:
            self.meter.freeze()

    def start(self):
        require(self.thread is None, "Watchdog already started")
        self.thread = threading.Thread(
            target=self.run, name="atlas-profile-watchdog", daemon=True
        )
        self.thread.start()
        require(self.ready.wait(0.5), "Watchdog startup unavailable")

    def close(self):
        with self.condition:
            require(not self.slots, "Active watchdog ownership remains")
            self.stopped = True
            self.condition.notify_all()
        if self.thread:
            self.thread.join(0.2)
            require(not self.thread.is_alive(), "Watchdog thread cleanup pending")


class ProfileService:
    """One record and no job queue. Existing tab capabilities authorize ownership.

    context.authorize(model,context,tab) and context.resolve(request) use the
    selected registry receipt/native binding. hooks_factory(source, watch_slot)
    provides LinuxProfileHooks. Imports and construction do not start threads.
    """

    def __init__(
        self,
        supervisor,
        context,
        hooks_factory,
        *,
        clock=time.monotonic,
        runtime_factory=SupervisedProfile,
        owner_meter=None,
        watchdog=None,
        diagnostics=None,
    ):
        self.diagnostics = diagnostics
        self.supervisor, self.context, self.hooks_factory = (
            supervisor,
            context,
            hooks_factory,
        )
        self.clock, self.runtime_factory = clock, runtime_factory
        self.owner_meter = owner_meter or ThreadMeter(diagnostics=diagnostics)
        self.watch_meter = (
            ThreadMeter(diagnostics=diagnostics) if watchdog is None else watchdog.meter
        )
        self.watchdog = watchdog or WatchdogLoop(
            self.watch_meter, clock=clock, diagnostics=diagnostics
        )
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.ready = threading.Event()
        self.record = None
        self.thread = None
        self.stopped = False
        self.shutdown_requested = threading.Event()
        self.owner_cleanup = None

    def request_shutdown(self):
        self.shutdown_requested.set()
        self.wake.set()

    def admission(self):
        # Called before HTTP framing/body parse. Freeze on that same HTTP thread.
        require(
            self.thread is not None
            and self.thread.is_alive()
            and self.watchdog.thread is not None
            and self.watchdog.thread.is_alive(),
            "Supervising execution contexts unavailable",
        )
        require(
            not self.shutdown_requested.is_set(), "Hosted provider shutdown pending"
        )
        meter = ThreadMeter(diagnostics=self.diagnostics)
        meter.register()
        ledger = CpuLedger(
            {
                "admission": meter.read,
                "owner": self.owner_meter.read,
                "watchdog": self.watch_meter.read,
            }
        )
        return AdmissionGrant(self.clock, ledger), meter

    def _owner(self, data, *, job=True):
        self.context.authorize(
            data["model_id"], data["context_id"], data["tab_capability"]
        )
        r = self.record
        require(
            r is not None
            and r["model_id"] == data["model_id"]
            and r["context_id"] == data["context_id"]
            and secrets.compare_digest(r["tab"], data["tab_capability"]),
            "Profile is not owned",
        )
        if job:
            require(
                r["id"] == data["job_id"]
                and secrets.compare_digest(r["cap"], data["job_capability"]),
                "Private job mismatch",
            )
        return r

    def _snapshot(self, r):
        if (r["cancel"] or self.shutdown_requested.is_set()) and r["status"][
            "state"
        ] != "cancelled":
            status = {
                "state": "stopping",
                "accepted": None,
                "error": r["status"]["error"],
                "cleanup_pending": True,
                "resume_available": False,
            }
        else:
            status = r["status"]
        return {
            "version": 1,
            "model_id": r["model_id"],
            "context_id": r["context_id"],
            "job_id": r["id"],
            **deepcopy(status),
        }

    def start(self, data, grant):
        with self.lock:
            require(
                not self.shutdown_requested.is_set(), "Hosted provider shutdown pending"
            )
            self.context.authorize(
                data["model_id"], data["context_id"], data["tab_capability"]
            )
            source = self.context.resolve(data)
            grant.remaining()
            previous = self.record
            if previous is not None and previous["status"]["state"] != "cancelled":
                self._owner(data, job=False)
                require(
                    data["restart"]
                    and not previous["status"]["cleanup_pending"]
                    and previous["status"]["state"]
                    in ("complete", "partial", "error", "cancelled"),
                    "Explicit Restart after cleanup required",
                )
            token = self.supervisor.acquire("profile", data["context_id"])
            r = {
                "id": secrets.token_hex(16),
                "cap": secrets.token_hex(32),
                "tab": data["tab_capability"],
                "model_id": data["model_id"],
                "context_id": data["context_id"],
                "source": source,
                "data": deepcopy(data),
                "grant": grant,
                "token": token,
                "runtime": None,
                "previous": previous,
                "cancel": False,
                "lease": self.clock() + 15,
                "pending": True,
                "ready": False,
                "status": {
                    "state": "admitting",
                    "accepted": None,
                    "error": None,
                    "cleanup_pending": True,
                    "resume_available": False,
                },
            }

            def admission_guard():
                try:
                    grant.remaining()
                    require(
                        self.clock() < r["lease"] and not r["cancel"],
                        "Owner lease/cancel",
                    )
                except BaseException as diagnostic_error:
                    diagnose(
                        self,
                        "profile_service.ProfileService.start.admission_guard.catch187",
                        diagnostic_error,
                    )
                    r["cancel"] = True
                    grant.cancel()
                    token.poison()
                    runtime = r["runtime"]
                    if runtime is not None:
                        runtime.job.cancel_event.set()
                        if runtime.platform.child is not None:
                            runtime.platform.child.stop()
                    self.wake.set()

            try:
                r["watch"] = self.watchdog.bind(admission_guard, grant.deadline)
            except BaseException as diagnostic_error:
                diagnose(
                    self,
                    "profile_service.ProfileService.start.catch199",
                    diagnostic_error,
                )
                self.supervisor.admission_aborted(token, no_child_created=True)
                token.release()
                raise
            self.record = r
            self.wake.set()
            return {**self._snapshot(r), "job_capability": r["cap"]}

    def finish_admission(self, grant):
        # HTTP bridge freezes its CPU meter after response handling, before wake.
        with self.lock:
            r = self.record
            if r is not None and r["grant"] is grant:
                r["ready"] = True
                self.wake.set()

    def status(self, data):
        with self.lock:
            return self._snapshot(self._owner(data))

    def page(self, data):
        require(
            not self.shutdown_requested.is_set(), "Hosted provider shutdown pending"
        )
        with self.lock:
            r = self._owner(data)
            require(
                not r["cancel"]
                and self.clock() < r["lease"]
                and not r["status"]["cleanup_pending"],
                "Profile unavailable",
            )
            runtime = r["runtime"]
            require(runtime is not None, "No accepted profile")
            return runtime.page(
                r["tab"],
                r["cap"],
                r["context_id"],
                data["revision"],
                data["axis"],
                data["start"],
                data["count"],
            )

    def _cancel(self, r):
        if r["status"]["state"] == "cancelled":
            return
        if (
            r["runtime"] is None
            and not r["pending"]
            and r["watch"].closed
            and self.supervisor.active is None
            and self.supervisor.profile_session is None
        ):
            r["cancel"] = True
            r["status"] = {
                "state": "cancelled",
                "accepted": None,
                "error": None,
                "cleanup_pending": False,
                "resume_available": False,
            }
            return
        r["cancel"] = True
        r["status"] = {
            "state": "stopping",
            "accepted": None,
            "error": None,
            "cleanup_pending": True,
            "resume_available": False,
        }
        runtime = r["runtime"]
        if runtime is not None:
            runtime.job.cancel_event.set()
            if runtime.platform.child is not None:
                runtime.platform.child.stop()
        self.wake.set()

    def cancel(self, data):
        with self.lock:
            r = self._owner(data)
            self._cancel(r)
            return self._snapshot(r)

    def reconcile(self, data):
        # Prove only this authorized tab's ownership, never global availability.
        # A retained predecessor may still own storage during explicit Restart.
        with self.lock, self.supervisor.lock:
            self.context.authorize(
                data["model_id"], data["context_id"], data["tab_capability"]
            )
            records, r = [], self.record
            while r is not None:
                require(
                    len(records) < 64 and all(r is not item for item in records),
                    "Profile ownership inventory unavailable",
                )
                records.append(r)
                r = r["previous"]
            active, session = self.supervisor.active, self.supervisor.profile_session
            require(
                active is None
                or active.kind != "profile"
                or any(item["token"] is active for item in records),
                "Profile token ownership unavailable",
            )
            require(
                session is None or any(item["runtime"] is session for item in records),
                "Profile storage ownership unavailable",
            )
            require(
                session is not None or self.supervisor.snapshot_reservation == 0,
                "Profile storage ownership unavailable",
            )
            owned = [
                item
                for item in records
                if item["model_id"] == data["model_id"]
                and item["context_id"] == data["context_id"]
                and secrets.compare_digest(item["tab"], data["tab_capability"])
            ]
            if self.record is not None and any(self.record is item for item in owned):
                self._cancel(self.record)
            # Even a terminal label alone cannot discharge reachable handles.
            if any(
                item["status"]["state"] != "cancelled"
                or item["pending"]
                or item["runtime"] is not None
                or not item["watch"].closed
                or item["token"] is active
                for item in owned
            ):
                return {
                    "state": "stopping",
                    "cleanup_pending": True,
                    "resume_available": False,
                }
            return {
                "state": "cancelled",
                "cleanup_pending": False,
                "resume_available": False,
                "no_owned_work": True,
            }

    def heartbeat(self, data):
        with self.lock:
            require(
                not self.shutdown_requested.is_set(), "Hosted provider shutdown pending"
            )
            r = self._owner(data)
            require(
                not r["cancel"] and self.clock() < r["lease"], "Owner lease expired"
            )
            r["lease"] = self.clock() + 15
            if r["runtime"] is not None:
                r["runtime"].job.heartbeat(r["tab"], r["cap"], r["context_id"])
            return self._snapshot(r)

    def step(self):
        # Dedicated owner only; HTTP status never calls this method.
        r = self.record
        if r is None or r["status"]["state"] == "cancelled":
            return
        if self.shutdown_requested.is_set() and not r["cancel"]:
            with self.lock:
                self._cancel(r)
        if not r["ready"] and not r["cancel"]:
            return
        if self.clock() >= r["lease"]:
            with self.lock:
                self._cancel(r)
        try:
            if r["pending"]:
                r["pending"] = False
                if r["previous"] and r["previous"]["runtime"] is not None:
                    r["previous"]["runtime"].close()
                r["previous"] = None
                if r["cancel"]:
                    r["watch"].close()
                    self.supervisor.admission_aborted(r["token"], no_child_created=True)
                    r["token"].release()
                    with self.lock:
                        r["status"] = {
                            "state": "cancelled",
                            "accepted": None,
                            "error": None,
                            "cleanup_pending": False,
                            "resume_available": False,
                        }
                    return
                hooks = self.hooks_factory(r["source"], r["watch"])
                runtime = self.runtime_factory(
                    self.supervisor,
                    r["grant"],
                    hooks,
                    r["data"]["binding"],
                    r["data"]["seed"],
                    r["tab"],
                    r["context_id"],
                    reserved=r["token"],
                )
                r["runtime"] = runtime
                runtime.job.job_capability = r["cap"]
                runtime.auth = (r["tab"], r["cap"], r["context_id"])
                runtime.start(r["data"]["values"])
            runtime = r["runtime"]
            if runtime is None:
                return
            if r["cancel"] and runtime.platform.closed:
                runtime.close()
                r["runtime"] = None
                with self.lock:
                    r["status"] = {
                        "state": "cancelled",
                        "accepted": None,
                        "error": None,
                        "cleanup_pending": False,
                        "resume_available": False,
                    }
                return
            runtime.tick()
            if r["cancel"] and runtime.platform.closed:
                runtime.close()
                r["runtime"] = None
                with self.lock:
                    r["status"] = {
                        "state": "cancelled",
                        "accepted": None,
                        "error": None,
                        "cleanup_pending": False,
                        "resume_available": False,
                    }
            else:
                with self.lock:
                    r["status"] = runtime.status(*runtime.auth)
        except BaseException as diagnostic_error:
            diagnose(
                self, "profile_service.ProfileService.step.catch327", diagnostic_error
            )
            r["cancel"] = True
            # All uncertain handles/reservations remain reachable and charged.
            with self.lock:
                r["status"] = {
                    "state": "stopping",
                    "accepted": None,
                    "error": "runtime_error",
                    "cleanup_pending": True,
                    "resume_available": False,
                }
            if self.supervisor.active is r["token"]:
                r["token"].poison()
            runtime = r["runtime"]
            if runtime is not None:
                runtime.job.cancel_event.set()
                if runtime.platform.child is not None:
                    runtime.platform.child.stop()
            else:
                r["cancel"] = True
                # Constructor failure releases only when it proves no child/FD.
                if (
                    self.supervisor.active is None
                    and self.supervisor.profile_session is None
                ):
                    r["watch"].close()
                    r["status"] = {
                        "state": "error",
                        "accepted": None,
                        "error": "admission_failed",
                        "cleanup_pending": False,
                        "resume_available": False,
                    }

    @diagnosed("profile_service.execution_thread")
    def run(self):
        self.owner_meter.register()
        self.ready.set()
        try:
            while not self.stopped:
                self.step()
                if self.shutdown_requested.is_set() and self.owner_cleanup is not None:
                    try:
                        self.owner_cleanup()
                    except BaseException as diagnostic_error:
                        diagnose(
                            self,
                            "profile_service.lifetime_cleanup_pending",
                            diagnostic_error,
                        )
                r = self.record
                runtime = None if r is None else r["runtime"]
                child = None if runtime is None else runtime.platform.child
                if child is not None and not child.reaped:
                    child.wait(0.05)
                elif runtime is not None and runtime.job.state == "validating":
                    continue  # Bounded primitive validation steps, under same grant.
                else:
                    self.wake.wait(0.25)
                    self.wake.clear()
        finally:
            self.owner_meter.freeze()

    def start_threads(self):
        require(self.thread is None, "Owner already started")
        self.watchdog.start()
        self.thread = threading.Thread(
            target=self.run, name="atlas-profile-owner", daemon=True
        )
        self.thread.start()
        require(self.ready.wait(0.5), "Owner startup unavailable")

    def close(self):
        with self.lock:
            if (
                self.record is not None
                and self.record["status"]["state"] != "cancelled"
            ):
                self._cancel(self.record)
                return False
            require(not self.supervisor.busy(), "Owned numeric cleanup pending")
            if self.owner_cleanup is not None and not self.owner_cleanup():
                return False
            require(
                self.supervisor.charged_snapshot_bytes() == 0,
                "Retained snapshot cleanup pending",
            )
            self.stopped = True
            self.wake.set()
        if self.thread:
            self.thread.join(0.3)
            require(not self.thread.is_alive(), "Owner thread cleanup pending")
        self.watchdog.close()
        return True
