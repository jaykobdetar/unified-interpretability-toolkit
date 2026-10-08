"""Hosted wiring contracts use existing tiny fixtures and inert process doubles."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
import threading
import unittest
from unittest.mock import Mock, patch

from atlas_host import hosted_runtime as module
from atlas_host.common import canonical
import dense_static_contracts as static
import host_picker_contracts as picker
import profile_runtime_doubles as native


class HostedRuntimeBoundaryTests(unittest.TestCase):
    def ok(self, function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except BaseException as error:
            self.fail(f"Valid inert call raised {type(error).__name__}: {error}")

    def error(self, message, function, *args):
        try:
            function(*args)
        except BaseException as error:
            self.assertIs(type(error), ValueError)
            self.assertEqual(str(error), message)
        else:
            self.fail("Expected existing refusal")

    def channel(self):
        case = native.NativeTests(methodName="runTest")
        channel = self.ok(case.make)

        def cleanup():
            case.child.reaped = True
            channel.stop()
            case.watch.close()

        self.addCleanup(cleanup)
        return case, channel

    def fixture(self):
        case = picker.HostFixtureTests(methodName="runTest")
        self.addCleanup(case.doCleanups)
        self.ok(case.setUp)
        lease = self.ok(case.acquire)
        self.ok(case.read, lease)
        return case, lease

    def operation(self):
        case = static.OperationTests(methodName="runTest")
        self.addCleanup(case.doCleanups)
        self.ok(case.setUp)
        operation = self.ok(case.operation)

        def cleanup():
            case.child.reaped = True
            operation.abort()
            case.watch.close()

        self.addCleanup(cleanup)
        return case, operation

    def context(self):
        case, lease = self.fixture()
        lock = threading.RLock()
        context = self.ok(module.HostedContexts, case.host, lock)
        value = {
            "version": 2,
            "source_identity": case.host.source_identity,
            "model_identity": case.host.model_identity,
            "tensor": 0,
            "name": "matrix",
            "dtype": "BF16",
            "shape": [3, 5],
            "rows": 3,
            "cols": 5,
            "slice": {"leading_indices": [], "display_axes": [0, 1]},
        }
        data = {
            "model_id": lease["model_id"],
            "context_id": lease["context_id"],
            "tab_capability": lease["capability"],
            "binding": value,
        }
        return case, context, lock, value, data

    def test_native_constructor_state_and_dependencies(self):
        case, channel = self.channel()
        self.assertIs(channel.child, case.child)
        self.assertIs(channel.supervisor, case.s)
        self.assertIs(channel.failed, False)
        self.assertEqual(channel.settled_cpu, 0.0)
        self.assertEqual(channel.cpu_seen, 0.0)
        self.assertIsNone(channel.token)
        self.assertIsNone(channel.slot)

    def test_native_io_exact_bytes_wait_and_checks(self):
        _, channel = self.channel()
        channel.stream = NS(recv=Mock(return_value=b"abc"), close=Mock())
        channel.wait = Mock()
        check = Mock()
        result = self.ok(channel._io, None, 3, 5.0, check=check)
        self.assertIs(type(result), bytes)
        self.assertEqual(result, b"abc")
        channel.wait.assert_called_once_with([channel.stream], [], [], 5.0)
        self.assertEqual(check.call_count, 2)

    def test_native_reap_receipt_and_single_observations(self):
        case, channel = self.channel()
        token = case.s.acquire("metadata", "ctx")
        case.s.native_command(token, {"route": "model", "query": {}}, 5000)
        channel.token = token
        case.child.reaped = True
        channel.slot = NS(disarm=Mock(), close=Mock())
        with patch.object(
            channel.book, "resources", wraps=channel.book.resources
        ) as resources:
            self.assertIs(self.ok(channel._reap_owned, token), True)
        resources.assert_called_once_with()
        channel.slot.disarm.assert_called_once_with()
        self.assertIsNone(channel.token)
        self.assertFalse(case.s.busy())

    def test_native_failure_retains_single_stop_and_none_result(self):
        case, channel = self.channel()
        token = case.s.acquire("metadata", "ctx")
        channel.token = token
        with patch.object(case.child, "stop", wraps=case.child.stop) as stop:
            self.assertIsNone(self.ok(channel._fail_owned, token))
        self.assertIs(channel.failed, True)
        stop.assert_called_once_with()
        self.assertTrue(case.s.busy())

    def test_native_idle_exact_observation_and_original_allowance(self):
        case, channel = self.channel()
        case.child.sample = lambda: {"cpu_seconds": 3.95, "reaped": case.child.reaped}
        self.assertIsNone(self.ok(channel.idle_pulse))
        self.assertEqual(channel.cpu_seen, 3.95)
        case.child.sample = lambda: {"cpu_seconds": 3.95, "reaped": case.child.reaped}

    def test_static_read_exact_normal_receipt(self):
        case, operation = self.operation()
        self.assertEqual(
            self.ok(case.channel._static_read, "/api/model", "ctx", operation),
            (200, b"{}", "application/json"),
        )

    def test_native_read_exact_receipt_and_grant_observations(self):
        case, channel = self.channel()
        grant = case.clock.grant()
        channel.service.admission = lambda: (grant, case.meter)
        with (
            patch.object(grant, "sample_child", wraps=grant.sample_child) as sample,
            patch.object(grant, "remaining", wraps=grant.remaining) as remaining,
        ):
            result = self.ok(channel.read, "/api/model", "ctx")
        self.assertEqual(result, (200, b"{}", "application/json"))
        sample.assert_called_once_with(0.1)
        self.assertEqual(remaining.call_count, 3)
        self.assertTrue(case.meter.frozen)
        self.assertFalse(case.s.busy())

    def test_native_guard_observation_refusal_preserves_cleanup_calls(self):
        case, channel = self.channel()
        resources = channel.book.resources
        channel.book.resources = lambda: {**resources(), "available_bytes": 0}
        with patch.object(channel, "_fail_owned", wraps=channel._fail_owned) as fail:
            self.error("Native operation expired", channel.read, "/api/model", "ctx")
        self.assertEqual(fail.call_count, 2)
        channel.book.resources = resources
        self.assertTrue(case.meter.frozen)

    def test_native_stop_exact_boolean_and_close_count(self):
        case, channel = self.channel()
        case.child.reaped = True
        with patch.object(channel.stream, "close") as close:
            self.assertIs(self.ok(channel.stop), True)
        close.assert_called_once_with()
        self.assertIs(channel.failed, True)

    def renderer(self):
        renderer = object.__new__(module.HostedRenderer)
        renderer.child = NS(
            initialize=Mock(), stop=Mock(), sample=Mock(return_value={"reaped": False})
        )
        renderer.channel = NS(
            failed=False,
            read=Mock(return_value=(200, b"{}", "application/json")),
            stop=Mock(return_value=True),
        )
        return renderer

    def test_renderer_constructor_inert_spawn_arguments(self):
        stream, peer = Mock(), Mock()
        peer.fileno.return_value = 73
        binary = NS(path=Path("/inert/native"), check=Mock())
        child = NS()
        book = NS(settled=Mock(return_value=True), spawn=Mock(return_value=child))
        supervisor, service = object(), object()
        entry = {
            "root": "/inert/source",
            "name": "fixture",
            "manifest": {"revision": "r1"},
        }
        with patch.object(module.socket, "socketpair", return_value=(stream, peer)):
            renderer = self.ok(
                module.HostedRenderer,
                entry,
                Path("/inert/cache"),
                binary,
                book,
                supervisor,
                service,
            )
        stream.setblocking.assert_called_once_with(False)
        self.assertIs(stream.setblocking.call_args.args[0], False)
        self.assertEqual(
            book.spawn.call_args.args[0],
            [
                "/inert/native",
                "hosted-renderer",
                "--model",
                "/inert/source",
                "--cache",
                "/inert/cache",
                "--name",
                "fixture",
                "--revision",
                "r1",
                "--channel-fd",
                "73",
            ],
        )
        self.assertEqual(book.spawn.call_count, 1)
        self.assertEqual(
            book.spawn.call_args.kwargs, {"pass_fds": (73,), "receipt": False}
        )
        self.assertIs(book.spawn.call_args.kwargs["receipt"], False)
        self.assertIs(renderer.channel.child, child)
        peer.close.assert_called_once_with()

    def test_renderer_initialize_single_callback_and_none(self):
        renderer = self.renderer()
        self.assertIsNone(self.ok(renderer.initialize))
        renderer.child.initialize.assert_called_once_with()
        renderer.child.stop.assert_not_called()

    def test_renderer_alive_exact_boolean_single_sample(self):
        renderer = self.renderer()
        self.assertIs(self.ok(renderer.alive), True)
        renderer.child.sample.assert_called_once_with()

    def test_renderer_ready_single_observation(self):
        renderer = self.renderer()
        with patch.object(renderer, "alive", wraps=renderer.alive) as alive:
            self.assertIs(self.ok(renderer.ready), True)
        alive.assert_called_once_with()

    def test_renderer_read_exact_forwarding_and_identity(self):
        renderer = self.renderer()
        result = self.ok(renderer.read, "/api/model")
        self.assertIs(result, renderer.channel.read.return_value)
        renderer.channel.read.assert_called_once_with(
            "/api/model", "hosted-renderer", operation=None
        )

    def test_renderer_stop_exact_boolean_single_callback(self):
        renderer = self.renderer()
        self.assertIs(self.ok(renderer.stop), True)
        renderer.channel.stop.assert_called_once_with()

    def test_context_constructor_exact_dependencies_and_empty_map(self):
        case, context, lock, _, _ = self.context()
        self.assertIs(context.host, case.host)
        self.assertIs(context.lock, lock)
        self.assertIs(type(context.bindings), dict)
        self.assertEqual(context.bindings, {})

    def test_context_authorize_current_and_expired_live_tab(self):
        case, context, _, _, data = self.context()
        args = [data[k] for k in ("model_id", "context_id", "tab_capability")]
        with patch.object(case.host, "_owns", wraps=case.host._owns) as owns:
            self.assertIsNone(self.ok(context.authorize, *args))
        owns.assert_called_once_with(args[1], args[2])
        case.now = 100
        self.error("Current live tab ownership required", context.authorize, *args)

    def test_context_remember_exact_canonical_copied_binding(self):
        _, context, _, value, data = self.context()
        expected = deepcopy(value)
        self.assertIsNone(
            self.ok(context.remember, data["model_id"], data["context_id"], value)
        )
        self.assertEqual(context.bindings, {canonical(expected): expected})
        value["shape"][0] = 99
        self.assertEqual(context.bindings, {canonical(expected): expected})

    def test_context_resolve_exact_source_and_local_recheck(self):
        case, context, _, value, data = self.context()
        self.ok(context.remember, data["model_id"], data["context_id"], value)
        result = self.ok(context.resolve, data)
        self.assertIs(type(result), dict)
        self.assertEqual(set(result), {"root", "revision", "cache", "binding", "check"})
        self.assertEqual(result["root"], case.host.entry["root"])
        self.assertEqual(result["revision"], case.host.entry["manifest"]["revision"])
        self.assertEqual(result["cache"], str(case.host.cache_root))
        self.assertEqual(result["binding"], value)
        with patch.object(module, "check_fixture", wraps=module.check_fixture) as check:
            self.assertIsNone(self.ok(result["check"]))
        check.assert_called_once_with(case.host.entry)
        case.host.context = "revoked"
        self.error("Source context changed", result["check"])

    def app(self):
        app = object.__new__(module.HostedApplication)
        app.lock = threading.RLock()
        app.lifetime = NS(stopping=False, pulse=Mock(), request_shutdown=Mock())
        app.host = NS(
            cache_root=Path("/inert/cache"), close=Mock(return_value=True), tick=Mock()
        )
        app.book = NS(cleanup=Mock(return_value=True))
        app.supervisor = NS(
            active=None,
            busy=Mock(return_value=False),
            charged_snapshot_bytes=Mock(return_value=0),
        )
        app.profiles = NS(start_threads=Mock(), close=Mock(return_value=True))
        app.static_operations = []
        app.dense_policy = None
        app.launch_policy = None
        return app

    def test_application_constructor_wiring_without_threads(self):
        case, _ = self.fixture()
        with (
            patch.object(module, "require_platform", return_value={"inert": True}),
            patch.object(module, "FrozenBinary", return_value=NS()),
            patch.object(module, "ProcessBook", return_value=NS()),
        ):
            app = self.ok(
                module.HostedApplication,
                case.registry,
                case.root / "other-cache",
                "/inert/native",
                "0" * 64,
            )
        self.assertIs(type(app.static_operations), list)
        self.assertEqual(app.static_operations, [])
        self.assertEqual(app.profiles.owner_cleanup, app._owned_shutdown_step)
        self.assertEqual(app.api.authorize_tab, app.contexts.authorize)
        self.assertIsNone(app.profiles.thread)
        self.assertIsNone(app.profiles.watchdog.thread)

    def test_static_admission_inert_wiring_and_conservative_reservation(self):
        # Reuse the existing synthetic policy/operation wiring fixture. This is
        # not qualification or activation of an external model or policy.
        case, operation = self.operation()
        app = self.app()
        app.supervisor = case.supervisor
        app.host = case.host
        app.book = case.book
        app.binary = NS()
        app.dense_policy = object.__new__(module.BoundDenseStaticAdmission)
        prepared = NS(entry={"model_id": "m_" + "a" * 64}, check=Mock())
        operation.prepared = prepared
        app.static_operations = [operation]
        with (
            patch.object(module.BoundDenseStaticAdmission, "admit", return_value=True),
            patch.object(module.BoundDenseStaticAdmission, "check_binary"),
            patch.object(module, "check_cache_available"),
            patch.object(module, "available_bytes", return_value=6 * 1024**3),
            patch.object(
                module, "reservation", wraps=module.reservation
            ) as reservation,
            patch.object(operation, "check", wraps=operation.check) as check,
        ):
            self.assertIs(self.ok(app._admit_static, prepared), True)
        check.assert_called_once_with()
        reservation.assert_called_once_with(100 * 1024**3, metadata=65536)

    def test_begin_static_forwards_owner_grant_kind_and_identity(self):
        app = self.app()
        app.dense_policy = object()
        grant, marker = object(), object()
        with patch.object(module, "StaticOperation", return_value=marker) as operation:
            self.assertIs(self.ok(app.begin_static, grant, "metadata"), marker)
        operation.assert_called_once_with(app, grant, "metadata")

    def test_lifetime_pulse_observes_once_and_retains_unfinished_operation(self):
        app = self.app()
        app.dense_policy = object()
        reader = NS(channel=NS(idle_pulse=Mock()))
        app.host.reader, app.host.view_kind = reader, "static"
        operation = NS(aborting=True, finished=False, pulse=Mock())
        app.static_operations = [operation]
        self.assertIsNone(self.ok(app._lifetime_pulse))
        app.lifetime.pulse.assert_called_once_with()
        reader.channel.idle_pulse.assert_called_once_with()
        operation.pulse.assert_called_once_with()
        self.assertEqual(app.static_operations, [operation])

    def test_fixture_renderer_constructor_arguments(self):
        case, _ = self.fixture()
        app = self.app()
        app.binary, app.diagnostics = object(), object()
        path = case.root / "renderer-cache"
        marker = object()
        with patch.object(module, "HostedRenderer", return_value=marker) as renderer:
            self.assertIs(self.ok(app._renderer, case.host.entry, path), marker)
        renderer.assert_called_once_with(
            case.host.entry,
            path,
            app.binary,
            app.book,
            app.supervisor,
            app.profiles,
            app.diagnostics,
            operation=None,
        )

    def test_start_threads_inert_affinity_and_single_launch(self):
        app = self.app()
        with (
            patch.object(module, "require_platform"),
            patch.object(module, "available_bytes", return_value=6 * 1024**3),
            patch.object(
                module.shutil, "disk_usage", return_value=NS(free=100 * 1024**3)
            ),
            patch.object(module.os, "sched_getaffinity", return_value={2, 3}),
            patch.object(module.os, "sched_setaffinity") as affinity,
        ):
            self.assertIsNone(self.ok(app.start_threads))
        affinity.assert_called_once_with(0, {2})
        app.profiles.start_threads.assert_called_once_with()

    def test_tick_single_owner_observations(self):
        app = self.app()
        self.assertIsNone(self.ok(app.tick))
        app.lifetime.pulse.assert_called_once_with()
        app.book.cleanup.assert_called_once_with(stopping_only=True)
        app.host.tick.assert_called_once_with()

    def test_owned_shutdown_exact_boolean_and_single_callbacks(self):
        app = self.app()
        operation = NS(finished=False, abort=Mock())
        app.static_operations = [operation]
        self.assertIs(self.ok(app._owned_shutdown_step), True)
        operation.abort.assert_called_once_with()
        app.host.close.assert_called_once_with()
        app.book.cleanup.assert_called_once_with()

    def test_close_orders_shutdown_cleanup_and_profile_close(self):
        app = self.app()
        calls = Mock()
        app.lifetime.request_shutdown = calls.shutdown
        app._owned_shutdown_step = calls.cleanup
        app.profiles.close = calls.close
        calls.close.return_value = True
        self.assertIs(self.ok(app.close), True)
        self.assertEqual(
            [call[0] for call in calls.mock_calls], ["shutdown", "cleanup", "close"]
        )


if __name__ == "__main__":
    unittest.main()
