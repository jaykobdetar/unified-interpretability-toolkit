"""Tiny owner receipts and inert reader doubles; no process/socket/ML work."""

from copy import deepcopy
from email.message import Message
import json
from pathlib import Path
import shutil
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from atlas_host.common import canonical
from atlas_host.hosted_runtime import HostedApplication
from atlas_host.profile_http import HostedHandler
from atlas_host.runtime_adapter import FixtureHost, HostError, dispatch
from atlas_host.validation_policy import BoundValidationPolicy
import static_model_contracts as fixtures


class Reader:
    def __init__(self, case, entry):
        self.case, self.entry = case, entry
        prepared = case.fixture.policy.prepare(entry["model_id"])
        self.model = case.fixture.native_model(prepared)
        self.model["source_directory"] = entry["root"]
        self.running = self.ready_flag = True
        self.stop_result = True
        self.requests = []
        self.on_read = self.on_initialize = None

    def initialize(self):
        assert self.case.host.reader is self
        assert self.case.host.entry == self.entry
        assert self.case.host.static_prepared is not None
        if self.on_initialize:
            self.on_initialize()

    def alive(self):
        return self.running

    def ready(self):
        return self.ready_flag

    def stop(self):
        if self.stop_result:
            self.running = False
        return self.stop_result

    def read(self, path):
        self.requests.append(path)
        ids = {key: self.model[key] for key in ("source_identity", "model_identity")}
        if path == "/api/model":
            body = deepcopy(self.model)
        elif path.startswith(("/api/view", "/api/inspect")):
            body = {"api_version": 1, "source_binding": ids}
        else:
            body = {"api_version": 1, **ids}
        raw = canonical(body)
        if self.on_read:
            self.on_read()
        return 200, raw, "application/json"


class StaticHostContracts(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.StaticContracts()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.registry, self.identifier = self.fixture.registry, self.fixture.identifier
        self.now = 0.0
        self.readers = []
        self.admission = Mock(return_value=True)
        self.factory_hook = None

        def factory(entry, cache):
            self.assertFalse(cache.is_relative_to(Path(entry["root"])))
            reader = Reader(self, entry)
            self.readers.append(reader)
            if self.factory_hook:
                self.factory_hook(reader)
            return reader

        self.factory = Mock(side_effect=factory)
        self.host = FixtureHost(
            self.registry,
            self.fixture.root / "cache",
            self.factory,
            clock=lambda: self.now,
            static_policy=self.fixture.policy,
            static_admission=self.admission,
        )

    def acquire(self, identifier=None, **current):
        return self.host.acquire({"model_id": identifier or self.identifier, **current})

    def read(self, lease, route="model"):
        code, raw, _ = self.host.read(
            lease["model_id"], route, {"context": [lease["context_id"]]}
        )
        self.assertEqual(code, 200)
        return json.loads(raw)

    def activate(self):
        lease = self.acquire()
        self.read(lease)
        return lease

    def other_source(self):
        source = self.fixture.root / "second"
        shutil.copytree(self.fixture.source, source)
        manifest = deepcopy(self.fixture.manifest)
        manifest["repository"] = "fixtures/second"
        identifier = self.registry.register(
            source, manifest, "Second", max_bytes=65536
        )["model_id"]
        self.registry.set_enabled(identifier, True)
        return identifier

    def test_default_admission_is_closed_before_preparation_or_factory(self):
        self.host.static_admission = None
        with patch.object(
            self.fixture.policy, "prepare", side_effect=AssertionError("No admission")
        ):
            with self.assertRaises(HostError) as error:
                self.acquire()
        self.assertEqual(error.exception.code, "static_disabled")
        self.factory.assert_not_called()
        catalog = self.host.catalog()
        self.assertFalse(catalog["static_views_enabled"])
        self.assertTrue(catalog["models"][0]["static_view_candidate"])
        self.assertFalse(catalog["models"][0]["static_view_ready"])

    def test_public_objects_extra_requests_and_invalid_capabilities_cannot_admit(self):
        with self.assertRaises(ValueError):
            FixtureHost(
                self.registry,
                self.fixture.root / "cache",
                self.factory,
                static_policy={},
            )
        with self.assertRaises(ValueError):
            FixtureHost(
                self.registry,
                self.fixture.root / "cache",
                self.factory,
                static_admission=self.admission,
            )
        with self.assertRaises(ValueError):
            self.host.acquire(
                {"model_id": self.identifier, "source_directory": "/public/input"}
            )
        lease = self.activate()
        other = self.other_source()
        with patch.object(
            self.fixture.policy,
            "prepare",
            side_effect=AssertionError("No foreign prepare"),
        ):
            with self.assertRaises(HostError) as error:
                self.acquire(other, context_id=lease["context_id"], capability="0" * 64)
        self.assertEqual(error.exception.code, "stale_context")
        self.assertEqual(len(self.readers), 1)

    def test_preparation_and_explicit_admission_failures_never_construct_reader(self):
        with patch.object(
            self.fixture.policy, "prepare", side_effect=ValueError("private-path")
        ):
            with self.assertRaises(HostError) as error:
                self.acquire()
        self.assertNotIn("private-path", str(error.exception))
        self.factory.assert_not_called()
        self.admission.return_value = False
        with self.assertRaises(HostError):
            self.acquire()
        self.factory.assert_not_called()
        self.assertIsNone(self.host.reader)

    def test_owned_initialization_failure_revokes_before_uncertain_stop(self):
        def fail(reader):
            reader.stop_result = False
            reader.on_initialize = lambda: (_ for _ in ()).throw(
                OSError("private init")
            )

        self.factory_hook = fail
        with self.assertRaises(HostError):
            self.acquire()
        self.assertIs(self.host.reader, self.readers[0])
        self.assertTrue(self.host.stopping)
        self.assertIsNone(self.host.static_prepared)
        self.assertIsNone(self.host.source_identity)
        self.assertFalse(self.host.catalog()["models"][0]["static_view_ready"])
        with self.assertRaises(HostError) as error:
            self.acquire()
        self.assertEqual(error.exception.code, "cleanup_pending")
        self.assertEqual(len(self.readers), 1)
        self.readers[0].stop_result = True
        self.host.tick()
        self.assertIsNone(self.host.reader)

    def test_full_correspondence_is_required_before_values_and_public_readiness(self):
        lease = self.acquire()
        self.assertEqual(lease["view_kind"], "static")
        self.assertFalse(lease["profiles_enabled"])
        with self.assertRaises(HostError):
            self.read(lease, "inspect")
        self.assertEqual(self.readers[0].requests, [])
        model = self.read(lease)
        self.assertTrue(model["static_view_ready"])
        self.assertFalse(model["profiles_enabled"])
        self.assertFalse(model["inference_ready"])
        self.assertFalse(model["fit_verified"])
        self.assertNotIn(lease["capability"], canonical(model).decode())
        self.assertNotIn(str(self.fixture.root), canonical(model).decode())
        self.assertEqual(
            self.read(lease, "inspect")["host_context"]["model_identity"],
            model["model_identity"],
        )

    def test_catalog_uses_published_health_and_no_source_contents_or_lifecycle(self):
        self.activate()
        reader = self.readers[0]
        with (
            patch.object(reader, "alive", side_effect=AssertionError("No sample")),
            patch.object(
                reader, "ready", side_effect=AssertionError("No readiness call")
            ),
            patch.object(reader, "stop", side_effect=AssertionError("No stop")),
            patch(
                "atlas_host.static_models._read",
                side_effect=AssertionError("No contents"),
            ),
        ):
            self.assertTrue(self.host.catalog()["models"][0]["static_view_ready"])
            self.now = 16
            self.assertFalse(self.host.catalog()["models"][0]["static_view_ready"])
        self.assertFalse(self.host.stopping)
        self.assertEqual(len(self.host.leases), 1)

    def test_bind_failure_retains_uncertain_owned_slot_and_clears_proof(self):
        lease = self.acquire()
        reader = self.readers[0]
        reader.stop_result = False
        reader.model["parameter_count"] += 1
        with self.assertRaises(HostError):
            self.read(lease)
        self.assertIs(self.host.reader, reader)
        self.assertTrue(self.host.stopping)
        self.assertIsNone(self.host.static_bound)
        self.assertIsNone(self.host.source_identity)
        self.assertFalse(self.host.catalog()["models"][0]["static_view_ready"])

    def test_disable_and_source_changes_reject_before_channel_read(self):
        lease = self.activate()
        reader = self.readers[0]
        reader.stop_result = False
        reader.requests.clear()
        self.registry.set_enabled(self.identifier, False)
        with self.assertRaises(HostError):
            self.read(lease, "inspect")
        self.assertEqual(reader.requests, [])
        self.assertTrue(self.host.stopping)
        self.assertIsNone(self.host.static_bound)
        self.assertEqual(self.host.catalog()["models"], [])

    def test_source_or_lease_change_after_channel_read_prevents_publication(self):
        lease = self.activate()
        reader = self.readers[0]
        reader.stop_result = False
        reader.on_read = lambda: self.registry.set_enabled(self.identifier, False)
        with self.assertRaises(HostError):
            self.read(lease, "inspect")
        self.assertTrue(self.host.stopping)
        self.assertIsNone(self.host.source_identity)

    def test_changed_synthetic_file_fingerprint_is_checked_after_channel_read(self):
        lease = self.activate()
        reader = self.readers[0]
        reader.stop_result = False
        path = self.fixture.source / "model.safetensors"
        reader.on_read = lambda: path.write_bytes(path.read_bytes()[:-1] + b"x")
        with self.assertRaises(HostError):
            self.read(lease, "inspect")
        self.assertTrue(self.host.stopping)
        self.assertIsNone(self.host.static_bound)
        self.assertFalse(self.host.catalog()["models"][0]["static_view_ready"])

    def test_expiry_death_and_health_loss_revoke_on_coordinator_observation(self):
        self.activate()
        reader = self.readers[0]
        reader.stop_result = False
        reader.running = False
        self.host.tick()
        self.assertTrue(self.host.stopping)
        self.assertIsNone(self.host.static_bound)
        self.assertFalse(self.host.catalog()["models"][0]["static_view_ready"])
        with self.assertRaises(HostError):
            self.acquire(self.other_source())
        self.assertEqual(len(self.readers), 1)

    def test_ready_loss_and_observation_error_do_not_leave_old_ready_proof(self):
        self.activate()
        reader = self.readers[0]
        reader.stop_result = False
        reader.ready_flag = False
        self.host.tick()
        self.assertIsNone(self.host.static_bound)
        self.assertTrue(self.host.stopping)
        reader.stop_result = True
        self.host.tick()
        self.activate()
        reader = self.readers[-1]
        reader.stop_result = False
        with patch.object(reader, "alive", side_effect=RuntimeError("private health")):
            self.host.tick()
        self.assertTrue(self.host.stopping)
        self.assertIsNone(self.host.source_identity)

    def test_cross_tab_busy_and_refused_admission_preserve_current_reader(self):
        lease = self.activate()
        other = self.other_source()
        with patch.object(
            self.fixture.policy,
            "prepare",
            side_effect=AssertionError("No busy prepare"),
        ):
            with self.assertRaises(HostError) as error:
                self.acquire(other)
        self.assertEqual(error.exception.code, "reader_busy")
        self.admission.return_value = False
        with self.assertRaises(HostError):
            self.acquire(
                other, context_id=lease["context_id"], capability=lease["capability"]
            )
        self.assertFalse(self.host.stopping)
        self.assertEqual(len(self.readers), 1)
        self.assertTrue(self.host.catalog()["models"][0]["static_view_ready"])

    def test_late_old_response_cannot_stop_replacement_context(self):
        lease = self.activate()
        old = self.readers[0]
        other = self.other_source()

        def replace():
            self.host.close()
            self.acquire(other)

        old.on_read = replace
        with self.assertRaises(HostError):
            self.read(lease, "inspect")
        self.assertIs(self.host.reader, self.readers[1])
        self.assertFalse(self.host.stopping)
        self.assertEqual(self.host.entry["model_id"], other)
        self.assertTrue(self.readers[1].running)

    def test_expired_response_and_channel_exception_revoke_and_retain(self):
        lease = self.activate()
        reader = self.readers[0]
        reader.stop_result = False
        reader.on_read = lambda: setattr(self, "now", 16.0)
        with self.assertRaises(HostError):
            self.read(lease, "inspect")
        self.assertTrue(self.host.stopping)
        self.assertIsNone(self.host.static_bound)
        reader.stop_result = True
        self.host.tick()
        lease = self.activate()
        reader = self.readers[-1]
        reader.stop_result = False
        with patch.object(reader, "read", side_effect=OSError("private channel")):
            with self.assertRaises(HostError) as error:
                self.read(lease, "inspect")
        self.assertNotIn("private channel", str(error.exception))
        self.assertTrue(self.host.stopping)

    def test_private_profile_binding_is_refused_without_channel_or_remember(self):
        lease = self.activate()
        self.readers[0].requests.clear()
        app = SimpleNamespace(
            host=self.host,
            lock=threading.RLock(),
            contexts=SimpleNamespace(remember=Mock()),
        )
        handler = HostedHandler.__new__(HostedHandler)
        handler.server = SimpleNamespace(application=app)
        handler.command = "GET"
        handler.headers = Message()
        handler.path = f'/api/models/{self.identifier}/binding?context={lease["context_id"]}&tensor=0'
        handler.validated = lambda: (handler.path.split("?")[0], 0)
        handler.send = Mock()
        handler.handle_action()
        self.assertEqual(handler.send.call_args.args[0], 409)
        self.assertEqual(self.readers[0].requests, [])
        app.contexts.remember.assert_not_called()

    def test_static_view_does_not_populate_profile_selection_cache(self):
        lease = self.activate()
        app = SimpleNamespace(
            host=self.host,
            lock=threading.RLock(),
            contexts=SimpleNamespace(remember=Mock()),
        )
        handler = HostedHandler.__new__(HostedHandler)
        handler.server = SimpleNamespace(application=app)
        handler.command = "GET"
        handler.headers = Message()
        handler.path = (
            f'/api/models/{self.identifier}/view?context={lease["context_id"]}&tensor=0'
        )
        handler.validated = lambda: (handler.path.split("?")[0], 0)
        handler.send = Mock()
        handler.handle_action()
        self.assertEqual(handler.send.call_args.args[0], 200)
        app.contexts.remember.assert_not_called()

    def test_production_factory_and_existing_bound_policy_refuse_dense_entry(self):
        entry = self.registry.owner_receipt(self.identifier)
        app = HostedApplication.__new__(HostedApplication)
        app.lifetime = SimpleNamespace(stopping=False)
        app.launch_policy = None
        with patch("atlas_host.hosted_runtime.HostedRenderer") as create:
            with self.assertRaises(ValueError):
                app._renderer(entry, self.fixture.root / "cache")
            create.assert_not_called()
        policy = object.__new__(BoundValidationPolicy)
        with patch.object(BoundValidationPolicy, "check", return_value=None):
            with self.assertRaises(ValueError):
                policy.check_entry(entry)


if __name__ == "__main__":
    unittest.main()
