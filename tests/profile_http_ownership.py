"""Two authorized tabs and reachable ownership; pure service/OS doubles only."""

from copy import deepcopy
import unittest
from unittest.mock import patch
from atlas_host.profile_api import PrivateProfileAPI
import profile_runtime_doubles as doubles


class Ownership(unittest.TestCase):
    def setUp(self):
        case = doubles.ServiceTests("test_second_start_never_queues")
        case.setUp()
        self.addCleanup(case.doCleanups)
        self.case = case
        self.a = case.data["tab_capability"]
        self.b = "e" * 64

        def authorize(model, context, tab):
            if (model, context) != (case.data["model_id"], "ctx") or tab not in (
                self.a,
                self.b,
            ):
                raise ValueError("foreign tab")

        case.service.context.authorize = authorize
        case.api = PrivateProfileAPI(case.service, authorize)
        self.second = {**case.data, "tab_capability": self.b}
        self.reconcile = {
            k: self.second[k]
            for k in ("version", "model_id", "context_id", "tab_capability")
        }

    def retained(self):
        g, _ = self.case.start()
        self.case.service.finish_admission(g)
        self.case.service.step()
        self.case.finish()
        return g

    def test_declined_second_tab_reconciles_without_affecting_owner_then_starts_after_reset(
        self,
    ):
        c = self.case
        self.retained()
        r = c.service.record
        accepted = deepcopy(r["status"]["accepted"])
        charge = c.service.supervisor.charged_snapshot_bytes()
        lease = r["lease"]
        files = set(c.os.files)
        with self.assertRaisesRegex(ValueError, "not owned"):
            c.api.handle("start", self.second, admission=c.clock.grant())
        result = c.api.handle("reconcile", self.reconcile)[1]
        self.assertEqual(
            result,
            {
                "state": "cancelled",
                "cleanup_pending": False,
                "resume_available": False,
                "no_owned_work": True,
            },
        )
        self.assertIs(c.service.record, r)
        self.assertFalse(r["cancel"])
        self.assertEqual(r["lease"], lease)
        self.assertEqual(r["status"]["accepted"], accepted)
        self.assertEqual(c.service.supervisor.charged_snapshot_bytes(), charge)
        self.assertEqual(set(c.os.files), files)
        for secret in (self.a, self.b, r["id"], r["cap"]):
            self.assertNotIn(secret, str(result))
        c.api.handle("cancel", c.owner)
        c.service.step()
        self.assertEqual(c.service.supervisor.charged_snapshot_bytes(), 0)
        self.assertFalse(c.os.files)
        self.assertTrue(c.api.handle("reconcile", self.reconcile)[1]["no_owned_work"])
        grant = c.clock.grant()
        _, started, _ = c.api.handle("start", self.second, admission=grant)
        self.assertNotEqual(started["job_id"], r["id"])
        self.assertEqual(c.service.record["tab"], self.b)
        c.service.finish_admission(grant)
        c.service.step()
        c.finish()
        owner = {
            **self.reconcile,
            "job_id": started["job_id"],
            "job_capability": started["job_capability"],
        }
        c.api.handle("cancel", owner)
        c.service.step()
        self.assertFalse(c.os.files)

    def test_other_tab_pending_admission_is_never_cancelled_or_declared_globally_free(
        self,
    ):
        c = self.case
        c.start()
        r = c.service.record
        token = c.service.supervisor.active
        self.assertTrue(c.api.handle("reconcile", self.reconcile)[1]["no_owned_work"])
        self.assertFalse(r["cancel"])
        self.assertIs(c.service.supervisor.active, token)
        with self.assertRaises(ValueError):
            c.api.handle("start", self.second, admission=c.clock.grant())
        c.api.handle("cancel", c.owner)
        c.service.step()

    def test_lost_restart_reconciles_replacement_and_retained_predecessor(self):
        c = self.case
        self.retained()
        previous = c.service.record
        c.data["restart"] = True
        c.start()
        r = c.service.record
        self.assertIs(r["previous"], previous)
        self.assertTrue(c.os.files)
        own = {k: c.data[k] for k in self.reconcile}
        self.assertTrue(c.api.handle("reconcile", own)[1]["cleanup_pending"])
        self.assertTrue(c.os.files)
        self.assertIs(r["previous"], previous)
        c.service.step()
        self.assertTrue(c.api.handle("reconcile", own)[1]["no_owned_work"])
        self.assertFalse(c.os.files)

    def test_predecessor_and_uncertain_cleanup_remain_owned(self):
        c = self.case
        self.retained()
        previous = c.service.record
        c.data["restart"] = True
        c.start()
        r = c.service.record
        # Reachable ownership still matters even when the current record is not
        # the requester's: do not cancel that current record as a shortcut.
        r["tab"] = self.b
        own = {k: c.data[k] for k in self.reconcile}
        self.assertTrue(c.api.handle("reconcile", own)[1]["cleanup_pending"])
        self.assertFalse(r["cancel"])
        self.assertFalse(previous["cancel"])
        self.assertTrue(c.os.files)
        r["tab"] = self.a
        c.api.handle("cancel", c.owner)
        c.service.step()

    def test_unmapped_token_session_or_reservation_never_proves_nonownership(self):
        c = self.case
        s = c.service.supervisor
        for field, value in [
            ("active", type("Token", (), {"kind": "profile"})()),
            ("profile_session", object()),
            ("snapshot_reservation", 1),
        ]:
            with self.subTest(field=field), patch.object(s, field, value):
                with self.assertRaisesRegex(ValueError, "ownership unavailable"):
                    c.api.handle("reconcile", self.reconcile)
        self.assertTrue(c.api.handle("reconcile", self.reconcile)[1]["no_owned_work"])

    def test_foreign_capability_cannot_reconcile_or_touch_retained_work(self):
        c = self.case
        self.retained()
        r = c.service.record
        with self.assertRaisesRegex(ValueError, "foreign tab"):
            c.api.handle("reconcile", {**self.reconcile, "tab_capability": "f" * 64})
        self.assertFalse(r["cancel"])
        self.assertTrue(c.os.files)
        c.api.handle("cancel", c.owner)
        c.service.step()


if __name__ == "__main__":
    unittest.main(verbosity=2)
