"""Pin the private router's existing references, results and inert dispatch."""

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from atlas_host.profile_api import PrivateProfileAPI, production_capabilities
from host_profile_adapter import BINDING


class ProfileAPIBoundaries(unittest.TestCase):
    def success(self, operation):
        try:
            return operation()
        except Exception as error:
            self.fail(f"Expected successful inert router operation: {error!r}")

    def test_disabled_capabilities_are_exact_fresh_records(self):
        first = self.success(production_capabilities)
        second = self.success(production_capabilities)
        self.assertEqual(first, {"profiles_enabled": False, "resume_available": False})
        self.assertEqual(second, first)
        self.assertIsNot(first, second)

    def test_constructor_retains_both_exact_references(self):
        service = object()
        authorize = Mock()
        router = self.success(lambda: PrivateProfileAPI(service, authorize))
        self.assertIs(router.service, service)
        self.assertIs(router.authorize_tab, authorize)
        authorize.assert_not_called()

    def test_all_routes_preserve_status_result_headers_and_request_copy(self):
        base = {
            "version": 1,
            "model_id": "m_" + "a" * 64,
            "context_id": "inert-context",
            "tab_capability": "b" * 64,
        }
        owner = {**base, "job_id": "c" * 32, "job_capability": "d" * 64}
        for action in ("start", "reconcile", "status", "page", "cancel", "heartbeat"):
            with self.subTest(action=action):
                data = deepcopy(base if action in ("start", "reconcile") else owner)
                if action == "start":
                    data.update(
                        binding=deepcopy(BINDING), seed=17, values=1, restart=False
                    )
                elif action == "page":
                    data.update(revision="e" * 64, axis="rows", start=0, count=1)
                original = deepcopy(data)
                result = {"operation": action}
                operation = Mock(return_value=result)
                service = SimpleNamespace(**{action: operation})
                authorize = Mock()
                grant = SimpleNamespace(remaining=Mock(return_value={"wall_ms": 1000}))
                admission = grant if action == "start" else None
                router = self.success(lambda: PrivateProfileAPI(service, authorize))
                reply = self.success(
                    lambda: router.handle(action, data, admission=admission)
                )
                self.assertEqual(
                    reply,
                    (
                        202 if action == "start" else 200,
                        result,
                        {"Cache-Control": "no-store"},
                    ),
                )
                self.assertIs(reply[1], result)
                authorize.assert_called_once_with(
                    base["model_id"], base["context_id"], base["tab_capability"]
                )
                operation.assert_called_once()
                self.assertEqual(operation.call_args.kwargs, {})
                received = operation.call_args.args[0]
                self.assertEqual(received, original)
                self.assertIsNot(received, data)
                self.assertEqual(data, original)
                if action == "start":
                    self.assertIsNot(received["binding"], data["binding"])
                    self.assertIs(operation.call_args.args[1], grant)
                    grant.remaining.assert_called_once_with()
                else:
                    self.assertEqual(len(operation.call_args.args), 1)
                    grant.remaining.assert_not_called()


if __name__ == "__main__":
    unittest.main()
