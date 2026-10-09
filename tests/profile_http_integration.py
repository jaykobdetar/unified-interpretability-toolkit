"""Focused owner-mode HTTP/launcher doubles. No server, thread, process or socket."""

from contextlib import ExitStack
from email.message import Message
import io, json
import re
from pathlib import Path
import sys, threading, types, unittest
from unittest.mock import Mock, patch

from atlas_host.profile_http import HostedHandler
from atlas_host.profile_api import production_capabilities
import profile_atlas
from page_startup_fixture import native_viewer


class Integration(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for target in [
            "subprocess.Popen",
            "socket.socket",
            "socket.socketpair",
            "threading.Thread.start",
            "os.memfd_create",
            "os.pidfd_open",
            "os.kill",
            "signal.pidfd_send_signal",
        ]:
            self.stack.enter_context(
                patch(target, side_effect=AssertionError("Live work forbidden"))
            )

    def handler(self, path, method="GET", body=None, enabled=True):
        h = object.__new__(HostedHandler)
        h.path = path
        h.command = method
        raw = b"" if body is None else json.dumps(body).encode()
        h.rfile = io.BytesIO(raw)
        h.headers = Message()
        for k, v in {
            "Host": "127.0.0.1:8798",
            "Origin": "http://127.0.0.1:8798",
            "X-Atlas-Local": "1",
            "Content-Type": "application/json",
            "Content-Length": str(len(raw)),
        }.items():
            h.headers[k] = v
        self.host = types.SimpleNamespace(
            catalog=Mock(
                return_value={
                    "api_version": 1,
                    "models": [],
                    "profiles_enabled": False,
                    "inference_enabled": False,
                }
            ),
            _validate_context=Mock(),
            reader=types.SimpleNamespace(
                read=Mock(
                    return_value=(
                        200,
                        b'{"api_version":1,"source_binding":{"tensor":0}}',
                        "application/json",
                    )
                )
            ),
        )
        self.supervisor = types.SimpleNamespace(
            busy=lambda: False, profile_session=None
        )
        self.app = types.SimpleNamespace(
            lock=threading.RLock(),
            host=self.host,
            supervisor=self.supervisor,
            contexts=types.SimpleNamespace(remember=Mock()),
            api=types.SimpleNamespace(
                handle=Mock(
                    return_value=(202, {"version": 1, "resume_available": False}, {})
                )
            ),
        )
        h.server = types.SimpleNamespace(
            application=self.app,
            host=self.host,
            server_port=8798,
            profile_controls=enabled,
        )
        h.profile_admission = object()
        self.sent = []
        h.send = lambda *args: self.sent.append(args)
        return h

    def test_capabilities_and_bundle_are_specific_to_explicit_owner_launcher(self):
        for enabled in (False, True):
            h = self.handler("/api/models", enabled=enabled)
            h.handle_action()
            self.assertEqual(self.sent[-1][1]["profiles_enabled"], enabled)
            self.assertFalse(self.sent[-1][1]["resume_available"])
            h = self.handler("/viewer.js", enabled=enabled)
            h.handle_action()
            script = self.sent[-1][1]
            pattern = (
                rb"\bshared\s*\.\s*AtlasProfiles\s*=\s*"
                rb'\(\s*await\s+import\(\s*"\./profile-client\.js"\s*\)\s*\)\s*\.\s*default\b'
                if native_viewer(Path(__file__).resolve().parents[1])
                else rb"\broot\s*\.\s*AtlasProfiles\s*=\s*api\b"
            )
            self.assertEqual(re.search(pattern, script) is not None, enabled)
        self.assertEqual(
            production_capabilities(),
            {"profiles_enabled": False, "resume_available": False},
        )

    def test_retained_snapshot_blocks_source_release_even_without_numeric_token(self):
        h = self.handler(
            "/api/view-contexts/ctx/release", "POST", {"capability": "a" * 64}
        )
        self.supervisor.profile_session = object()
        with patch(
            "host_atlas.HostHandler.handle_action",
            side_effect=AssertionError("No source replacement"),
        ):
            h.handle_action()
        self.assertEqual(self.sent[-1][0], 409)

    def test_source_release_allowed_only_after_confirmed_profile_cleanup(self):
        h = self.handler(
            "/api/view-contexts/ctx/release", "POST", {"capability": "a" * 64}
        )
        with patch("host_atlas.HostHandler.handle_action") as dispatch:
            h.handle_action()
        dispatch.assert_called_once()

    def test_profile_start_receives_original_admission_but_status_does_not(self):
        for action in ["start", "status", "page", "cancel", "heartbeat", "reconcile"]:
            h = self.handler("/api/profiles/" + action, "POST", {"version": 1})
            h.handle_action()
            self.assertIs(
                self.app.api.handle.call_args.kwargs["admission"],
                h.profile_admission if action == "start" else None,
            )

    def test_disabled_control_refuses_even_with_valid_same_origin(self):
        h = self.handler("/api/profiles/start", "POST", {}, enabled=False)
        h.handle_action()
        self.app.api.handle.assert_not_called()
        self.assertEqual(self.sent[-1][0], 409)

    def test_binding_is_native_read_with_current_context_and_no_grant_replay(self):
        h = self.handler("/api/models/m_owner/binding?context=ctx&tensor=7&slice=1")
        h.handle_action()
        self.host._validate_context.assert_called_once_with("m_owner", "ctx")
        self.host.reader.read.assert_called_once_with("/api/binding?tensor=7&slice=1")
        self.app.contexts.remember.assert_called_once_with(
            "m_owner", "ctx", {"tensor": 0}
        )

    def test_admission_precedes_receive_and_is_finished_even_when_http_raises(self):
        h = self.handler("/api/profiles/start", "POST", {})
        events = []
        grant = object()
        meter = types.SimpleNamespace(freeze=lambda: events.append("freeze"))
        self.app.profiles = types.SimpleNamespace(
            admission=lambda: events.append("admission") or (grant, meter),
            finish_admission=lambda g: events.append(("finish", g)),
        )
        with patch(
            "host_atlas.HostHandler.handle",
            side_effect=lambda: events.append("receive")
            or (_ for _ in ()).throw(ValueError("mock framing")),
        ):
            with self.assertRaises(ValueError):
                h.handle()
        self.assertEqual(events, ["admission", "receive", "freeze", ("finish", grant)])

    def test_actual_private_router_service_full_partial_pages_and_owned_reset(self):
        import profile_runtime_doubles as doubles

        case = doubles.ServiceTests("test_second_start_never_queues")
        case.setUp()
        self.addCleanup(case.doCleanups)
        service = case.service

        def request(action, data, grant=None):
            h = self.handler("/api/profiles/" + action, "POST", data)
            self.app.api = case.api
            h.profile_admission = grant
            h.handle_action()
            return self.sent[-1]

        grant = case.clock.grant()
        code, result = request("start", {**case.data, "values": 12}, grant)[:2]
        self.assertEqual(code, 202)
        self.assertFalse(hasattr(case, "hooks"))
        owner = {
            k: case.data[k]
            for k in ("version", "model_id", "context_id", "tab_capability")
        }
        owner.update(job_id=result["job_id"], job_capability=result["job_capability"])
        service.finish_admission(grant)
        service.step()
        case.finish()
        code, full = request("status", owner)[:2]
        self.assertEqual(full["state"], "complete")
        for axis, n in [("rows", 3), ("columns", 4)]:
            code, page = request(
                "page",
                {
                    **owner,
                    "revision": full["accepted"]["revision"],
                    "axis": axis,
                    "start": 0,
                    "count": n,
                },
            )[:2]
            self.assertEqual(code, 200)
            self.assertEqual(len(page["original"]), n)
            self.assertEqual(len(page["control"]), n)
            self.assertTrue(
                all(v["complete"] for v in page["original"] + page["control"])
            )
        old_owner = owner
        grant = case.clock.grant()
        code, result = request(
            "start", {**case.data, "values": 5, "restart": True}, grant
        )[:2]
        owner = {
            **owner,
            "job_id": result["job_id"],
            "job_capability": result["job_capability"],
        }
        self.assertEqual(request("status", old_owner)[0], 409)
        service.finish_admission(grant)
        service.step()
        case.finish()
        code, partial = request("status", owner)[:2]
        self.assertEqual(partial["accepted"]["visited_values"], 5)
        self.assertEqual(partial["state"], "partial")
        self.assertFalse(partial["resume_available"])
        request("cancel", owner)
        service.step()
        self.assertEqual(request("status", owner)[1]["state"], "cancelled")
        self.assertEqual(service.supervisor.charged_snapshot_bytes(), 0)
        self.assertFalse(case.os.files)

    def test_launcher_refuses_non_loopback_before_constructing_provider(self):
        with patch.object(profile_atlas, "HostedApplication") as app:
            with self.assertRaises(ValueError):
                profile_atlas.run({"bind": "0.0.0.0"}, Path("/qualified/bin"), "a" * 64)
        app.assert_not_called()

    def test_launcher_cannot_apply_tiny_recipe_and_owns_cleanup_after_listener_failure(
        self,
    ):
        calls = []
        app = types.SimpleNamespace(
            start_threads=lambda: calls.append("threads"),
            close=Mock(side_effect=[False, True]),
        )
        config = {
            "paths": {"registry": "registry.json", "cache": "cache"},
            "bind": "127.0.0.1",
            "ports": {"coordinator": 8798},
        }
        with (
            patch.object(profile_atlas, "Registry"),
            patch.object(
                profile_atlas, "HostedApplication", return_value=app
            ) as create,
            patch.object(
                profile_atlas, "HTTPServer", side_effect=OSError("bind failed")
            ) as server,
            patch.object(profile_atlas.time, "sleep") as wait,
        ):
            with self.assertRaises(OSError):
                profile_atlas.run(config, Path("/qualified/bin"), "a" * 64)
        self.assertEqual(create.call_args.kwargs, {})
        self.assertEqual(server.call_args.args[0], ("127.0.0.1", 8798))
        self.assertEqual(app.close.call_count, 2)
        wait.assert_called_once_with(0.05)
        self.assertEqual(calls, ["threads"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
