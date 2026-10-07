"""Fixture adapter receipts and bookkeeping with existing inert renderers."""

import json
from unittest.mock import Mock, patch
import unittest

from atlas_host import runtime_adapter as module
import host_picker_contracts as existing


class RuntimeAdapterBoundaryTests(unittest.TestCase):
    def ok(self, function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except BaseException as error:
            self.fail(
                f"Valid inert adapter call raised {type(error).__name__}: {error}"
            )

    def case(self, *, acquired=False, bound=False):
        case = existing.HostFixtureTests(methodName="runTest")
        self.addCleanup(case.doCleanups)
        self.ok(case.setUp)
        self.addCleanup(case.host.close)
        lease = self.ok(case.acquire) if acquired or bound else None
        if bound:
            self.ok(case.read, lease)
        return case, lease

    def refusal(self, function, *args):
        try:
            function(*args)
        except BaseException as error:
            self.assertIs(type(error), module.HostError)
            return error
        self.fail("Expected the existing refusal")

    def test_host_error_exact_fields_and_text(self):
        error = module.HostError(409, "ordinary", "Existing message")
        self.assertIs(type(error.status), int)
        self.assertEqual(
            (error.status, error.code, str(error)),
            (409, "ordinary", "Existing message"),
        )
        self.assertEqual(error.args, ("Existing message",))

    def test_refuse_exact_fields_and_text(self):
        error = self.refusal(module.refuse, 409, "ordinary", "Existing message")
        self.assertEqual(
            (error.status, error.code, str(error)),
            (409, "ordinary", "Existing message"),
        )

    def test_constructor_original_state_and_owner_fields(self):
        case, _ = self.case()
        host = case.host
        self.assertIs(host.registry, case.registry)
        self.assertEqual(host.cache_root, case.root / "cache")
        self.assertIs(host.stopping, False)
        self.assertIsNone(host.view_kind)
        self.assertEqual(host.started, 0.0)
        self.assertEqual(host.leases, {})
        self.assertIsNone(host.reader)

    def test_static_views_requires_both_private_configuration_fields(self):
        case, _ = self.case()
        host = case.host
        self.assertIs(host.static_views_enabled, False)
        host.static_policy = object()
        self.assertIs(host.static_views_enabled, False)
        host.static_admission = Mock()
        self.assertIs(host.static_views_enabled, True)

    def test_revoke_readiness_exact_cleared_shapes(self):
        case, _ = self.case(bound=True)
        self.assertIsNone(self.ok(case.host._revoke_readiness))
        for field in [
            "source_identity",
            "model_identity",
            "static_prepared",
            "static_bound",
            "observed_context",
        ]:
            self.assertIsNone(getattr(case.host, field))
        self.assertIs(case.host.observed_alive, False)
        self.assertIs(case.host.observed_ready, False)

    def test_published_ready_returns_boolean_without_reader_calls(self):
        case, _ = self.case(bound=True)
        reader = case.readers[0]
        with (
            patch.object(
                reader,
                "ready",
                side_effect=AssertionError("cached observation required"),
            ),
            patch.object(
                reader,
                "alive",
                side_effect=AssertionError("cached observation required"),
            ),
        ):
            self.assertIs(self.ok(case.host._published_ready), True)

    def test_observe_reader_single_calls_and_exact_context(self):
        case, lease = self.case(acquired=True)
        reader = case.readers[0]
        with (
            patch.object(reader, "ready", wraps=reader.ready) as ready,
            patch.object(reader, "alive", wraps=reader.alive) as alive,
        ):
            self.assertIsNone(self.ok(case.host._observe_reader))
        ready.assert_called_once_with()
        alive.assert_called_once_with()
        self.assertIs(case.host.observed_ready, True)
        self.assertIs(case.host.observed_alive, True)
        self.assertEqual(case.host.observed_context, lease["context_id"])

    def test_check_source_single_receipt_and_fixture_checks(self):
        case, _ = self.case(acquired=True)
        with (
            patch.object(
                case.registry, "owner_receipt", wraps=case.registry.owner_receipt
            ) as receipt,
            patch.object(
                module, "check_fixture", wraps=module.check_fixture
            ) as fixture,
        ):
            self.assertIsNone(self.ok(case.host._check_source))
        receipt.assert_called_once_with(case.ids[0])
        fixture.assert_called_once_with(case.host.entry)

    def test_clear_exact_empty_state_and_none_result(self):
        case, _ = self.case()
        case.host.stopping = True
        case.host.view_kind = "fixture"
        self.assertIsNone(self.ok(case.host._clear))
        self.assertIs(case.host.stopping, False)
        self.assertIsNone(case.host.view_kind)
        self.assertEqual(case.host.leases, {})

    def test_stop_single_owned_stop_and_exact_true_result(self):
        case, _ = self.case(acquired=True)
        with patch.object(
            case.host, "_revoke_readiness", wraps=case.host._revoke_readiness
        ) as revoke:
            self.assertIs(self.ok(case.host._stop), True)
        self.assertEqual(revoke.call_count, 2)
        self.assertEqual(case.readers[0].stops, 1)
        self.assertIsNone(case.host.reader)

    def test_uncertain_stop_retains_boolean_stopping_state(self):
        case, _ = self.case(acquired=True)
        reader = case.readers[0]
        reader.stop_result = False
        self.assertIs(self.ok(case.host._stop), False)
        self.assertIs(case.host.stopping, True)
        self.assertIs(case.host.reader, reader)
        reader.stop_result = True

    def test_tick_single_checks_and_original_starting_window(self):
        case, _ = self.case(acquired=True)
        case.now = 4.5
        with (
            patch.object(
                case.host, "_check_source", wraps=case.host._check_source
            ) as source,
            patch.object(
                case.host, "_observe_reader", wraps=case.host._observe_reader
            ) as observe,
        ):
            self.assertIsNone(self.ok(case.host.tick))
        source.assert_called_once_with()
        observe.assert_called_once_with()
        self.assertEqual(case.readers[0].stops, 0)
        self.assertIs(case.host.reader, case.readers[0])

    def test_close_preserves_stop_result_and_calls_once(self):
        case, _ = self.case()
        with patch.object(case.host, "_stop", return_value=True) as stop:
            self.assertIs(self.ok(case.host.close), True)
        stop.assert_called_once_with()

    def test_catalog_exact_version_and_capacities(self):
        case, _ = self.case()
        result = self.ok(case.host.catalog)
        self.assertEqual(
            (
                result["api_version"],
                result["reader_capacity"],
                result["lease_capacity"],
            ),
            (1, 1, 4),
        )
        self.assertEqual(len(result["models"]), 2)
        self.assertIs(result["inference_enabled"], False)

    def test_owns_exact_boolean_and_single_comparison(self):
        case, lease = self.case(acquired=True)
        with patch.object(
            module.secrets, "compare_digest", wraps=module.secrets.compare_digest
        ) as compare:
            self.assertIs(
                self.ok(case.host._owns, lease["context_id"], lease["capability"]), True
            )
        compare.assert_called_once_with(
            lease["capability"].encode(), lease["capability"].encode()
        )

    def test_acquisition_exact_receipt(self):
        case, lease = self.case(acquired=True)
        self.assertEqual(
            lease,
            {
                "api_version": 1,
                "model_id": case.ids[0],
                "context_id": case.host.context,
                "capability": next(iter(case.host.leases)),
                "lease_seconds": 15,
                "view_kind": "fixture",
                "state": "starting",
            },
        )

    def test_heartbeat_exact_receipt_and_deadline(self):
        case, lease = self.case(acquired=True)
        case.now = 2.0
        result = self.ok(
            case.host.heartbeat,
            lease["context_id"],
            {"capability": lease["capability"]},
        )
        self.assertEqual(
            result,
            {"api_version": 1, "context_id": lease["context_id"], "lease_seconds": 15},
        )
        self.assertEqual(case.host.leases[lease["capability"]], 17.0)

    def test_release_exact_receipt_types(self):
        case, lease = self.case(acquired=True)
        result = self.ok(
            case.host.release, lease["context_id"], {"capability": lease["capability"]}
        )
        self.assertEqual(
            result, {"api_version": 1, "released": True, "cleanup_pending": False}
        )
        self.assertIs(result["released"], True)
        self.assertIs(result["cleanup_pending"], False)

    def test_validate_context_single_tick_and_expected_source_checks(self):
        case, lease = self.case(bound=True)
        with (
            patch.object(case.host, "tick", wraps=case.host.tick) as tick,
            patch.object(
                case.host, "_check_source", wraps=case.host._check_source
            ) as check,
        ):
            self.assertIsNone(
                self.ok(case.host._validate_context, case.ids[0], lease["context_id"])
            )
        tick.assert_called_once_with()
        self.assertEqual(check.call_count, 3)

    def test_starting_reader_keeps_existing_refusal_text(self):
        case, lease = self.case(acquired=True)
        case.readers[0].ready_flag = False
        error = self.refusal(
            case.host._validate_context, case.ids[0], lease["context_id"]
        )
        self.assertEqual(
            (error.status, error.code, str(error)),
            (503, "backend_unavailable", "Fixture renderer is starting"),
        )

    def test_bind_fixture_preserves_exact_native_identities(self):
        case, _ = self.case(acquired=True)
        model = case.readers[0].info
        self.assertIsNone(self.ok(case.host._bind_model, model))
        self.assertEqual(case.host.source_identity, model["source_identity"])
        self.assertEqual(case.host.model_identity, model["model_identity"])

    def test_read_exact_status_mime_and_context_receipt(self):
        case, lease = self.case(acquired=True)
        status, body, mime = self.ok(
            case.host.read, case.ids[0], "model", {"context": [lease["context_id"]]}
        )
        self.assertEqual((status, mime), (200, "application/json"))
        model = json.loads(body)
        self.assertEqual(
            model["host_context"],
            {
                "model_id": case.ids[0],
                "context_id": lease["context_id"],
                "source_identity": case.host.source_identity,
                "model_identity": case.host.model_identity,
            },
        )

    def test_dispatch_exact_catalog_and_acquisition_statuses(self):
        case, _ = self.case()
        status, _, mime = self.ok(module.dispatch, case.host, "GET", "/api/models")
        self.assertEqual((status, mime), (200, "application/json"))
        status, _, mime = self.ok(
            module.dispatch,
            case.host,
            "POST",
            "/api/view-contexts",
            {"model_id": case.ids[0]},
        )
        self.assertEqual((status, mime), (202, "application/json"))

    def test_dispatch_existing_unknown_route_refusal(self):
        case, _ = self.case()
        error = self.refusal(module.dispatch, case.host, "GET", "/not-a-route")
        self.assertEqual(
            (error.status, error.code, str(error)),
            (404, "not_found", "Route is unavailable in fixture host mode"),
        )


if __name__ == "__main__":
    unittest.main()
