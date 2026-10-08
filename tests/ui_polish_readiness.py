"""Readiness metadata only: all resources/processes mocked, no launch or I/O."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import live_inference as live


class Readiness(unittest.TestCase):
    def test_single_worker_ownership_without_private_session_data(self):
        with patch.object(live, "available", return_value=6 * live.GIB):
            session = live.Session("unused", Path("/unused"))
        session.id = "private-capability-never-in-metadata"
        self.assertFalse(session.metadata()["busy"])
        self.assertEqual(session.metadata()["queue_capacity"], 0)
        for stage in ("loading", "running", "stopping"):
            session.status = stage
            self.assertTrue(session.metadata()["busy"])
            self.assertEqual(session.metadata()["busy_owner"], "inference session")
            self.assertNotIn(session.id, str(session.metadata()))
        session.status = "idle"
        session.analytics = SimpleNamespace(busy=True)
        self.assertTrue(session.metadata()["busy"])
        self.assertEqual(session.metadata()["busy_owner"], "analytics job")
        session.analytics.busy = False
        self.assertFalse(session.metadata()["busy"])
        self.assertIsNone(session.metadata()["busy_owner"])


if __name__ == "__main__":
    unittest.main()
