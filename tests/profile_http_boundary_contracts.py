"""Profile HTTP contracts use existing inert handler fixtures, never a listener."""

from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from atlas_host import profile_http as module
from atlas_host.common import canonical
import profile_http_integration as fixtures
from page_startup_fixture import expected_bundle


class ProfileHttpBoundaryTests(unittest.TestCase):
    def fixture(self, path="/api/models", enabled=True):
        case = fixtures.Integration(methodName="runTest")
        case.setUp()
        self.addCleanup(case.doCleanups)
        handler = case.handler(path, enabled=enabled)
        return case, handler

    def test_admission_cleanup_order_and_identity(self):
        case, handler = self.fixture()
        events = Mock()
        grant = object()
        meter = NS(freeze=events.freeze)
        events.admission.return_value = grant, meter
        case.app.profiles = NS(
            admission=events.admission, finish_admission=events.finish
        )
        operation = NS(finished=False, abort=events.abort)
        handler.static_finalizer = operation
        with patch.object(module.HostHandler, "handle", events.receive):
            self.assertIsNone(handler.handle())
        self.assertIs(handler.profile_admission, grant)
        self.assertEqual(
            [call[0] for call in events.mock_calls],
            ["admission", "receive", "abort", "freeze", "finish"],
        )
        events.finish.assert_called_once_with(grant)

    def test_finished_operation_is_not_aborted(self):
        case, handler = self.fixture()
        grant, meter = object(), NS(freeze=Mock())
        case.app.profiles = NS(
            admission=Mock(return_value=(grant, meter)), finish_admission=Mock()
        )
        operation = NS(finished=True, abort=Mock())
        handler.static_finalizer = operation
        with patch.object(module.HostHandler, "handle", return_value=None):
            self.assertIsNone(handler.handle())
        operation.abort.assert_not_called()
        meter.freeze.assert_called_once_with()
        case.app.profiles.finish_admission.assert_called_once_with(grant)

    def test_dispatch_preserves_arguments_and_response_identity(self):
        case, handler = self.fixture()
        body = {"version": 1}
        response = (200, b'{"models":[]}', "application/json")
        with patch.object(module, "dispatch", return_value=response) as dispatch:
            self.assertIs(handler.dispatch_host("GET", "/api/models", body), response)
        dispatch.assert_called_once_with(
            case.host, "GET", "/api/models", body, operation=None
        )

    def test_plain_send_preserves_status_body_and_default_mime(self):
        _, handler = self.fixture()
        del handler.send
        body = {"version": 1}
        with patch.object(module.HostHandler, "send", return_value=None) as send:
            self.assertIsNone(handler.send(202, body))
        send.assert_called_once_with(202, body, "application/json")

    def test_static_send_preserves_serialization_budget_and_publication(self):
        _, handler = self.fixture()
        del handler.send
        events = Mock()
        events.remaining.return_value = {"wall_ms": 1250}
        events.publish.side_effect = lambda callback: callback()
        events.send.return_value = None
        operation = NS(
            finished=False,
            abort=events.abort,
            check=events.check,
            grant=NS(remaining=events.remaining),
            publish=events.publish,
        )
        handler.static_finalizer = operation
        handler.connection = NS(settimeout=events.timeout)
        body = {"version": 1, "ready": True}
        with patch.object(module.HostHandler, "send", events.send):
            self.assertIsNone(handler.send(200, body))
        self.assertEqual(
            [call[0] for call in events.mock_calls],
            ["check", "remaining", "timeout", "publish", "send"],
        )
        events.timeout.assert_called_once_with(1.25)
        events.send.assert_called_once_with(200, canonical(body), "application/json")
        events.abort.assert_not_called()

    def test_catalog_keeps_exact_enabled_and_disabled_capabilities(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                case, handler = self.fixture(enabled=enabled)
                self.assertIsNone(handler.handle_action())
                self.assertEqual(len(case.sent), 1)
                status, body = case.sent[0]
                self.assertEqual(status, 200)
                self.assertEqual(
                    body,
                    {
                        "api_version": 1,
                        "models": [],
                        "inference_enabled": False,
                        "profiles_enabled": enabled,
                        "resume_available": False,
                    },
                )
                self.assertIs(type(body["profiles_enabled"]), bool)
                case.host.catalog.assert_called_once_with()

    def test_bundle_preserves_order_separator_bytes_and_mime(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                case, handler = self.fixture("/viewer.js", enabled=enabled)
                self.assertIsNone(handler.handle_action())
                bundle = list(module.BUNDLE)
                if enabled:
                    bundle.insert(bundle.index("host-client.js"), "profile-client.js")
                expected = expected_bundle(module.ROOT, bundle)
                self.assertEqual(case.sent, [(200, expected, "text/javascript")])

    def test_existing_ordinary_read_failure_keeps_refusal_record(self):
        case, handler = self.fixture()
        case.host.catalog.side_effect = OSError("inert catalog unavailable")
        with patch.object(module, "diagnose") as diagnose:
            self.assertIsNone(handler.handle_action())
        self.assertEqual(
            case.sent,
            [
                (
                    409,
                    {
                        "version": 1,
                        "code": "unavailable",
                        "error": "Hosted request unavailable or cleanup pending",
                    },
                )
            ],
        )
        self.assertEqual(diagnose.call_count, 1)
        self.assertIs(diagnose.call_args.args[0], case.app)
        self.assertEqual(diagnose.call_args.args[1], "profile_http.request_refused")


if __name__ == "__main__":
    unittest.main()
