"""Independent whole-provider memory enforcement through idle and cleanup."""
from .profile_observation import SpawnObservationPending
from .startup_diagnostics import diagnose
from .validation_policy import check_memory


class LifetimeGuard:
    def __init__(self, book, supervisor, request_cleanup, *, diagnostics=None):
        self.book,self.supervisor,self.request_cleanup=book,supervisor,request_cleanup
        self.diagnostics=diagnostics
        self.failed=False
        self.stopping=False

    def request_shutdown(self):
        self.stopping=True
        self.supervisor.prevent_admission()
        # Wake the owner even if a signal/observation fails. No grant is created.
        self.request_cleanup()
        self.book.request_stop_all()

    def pulse(self):
        try:
            pending=False
            try:
                observed=self.book.resources()
            except SpawnObservationPending as error:
                pending=True;observed=error.observed
            charge=self.supervisor.charged_snapshot_bytes()
            check_memory(observed,charge,allow_pending=pending)
        except BaseException as error:
            diagnose(self,'lifetime_guard.resource_refusal',error)
            self.failed=True
            self.stopping=True
        if self.stopping:
            try:
                self.request_shutdown()
            except BaseException as error:
                # Retain failure and try owned cleanup on subsequent pulses.
                # Never release ownership or report a clean shutdown here.
                diagnose(self,'lifetime_guard.stop_pending',error)
                self.failed=True
        return not self.failed
