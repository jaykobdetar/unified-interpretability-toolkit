"""Runtime checks with fake children, clocks, sockets and memfds; no services."""

import json
from pathlib import Path
import signal
import struct
import sys
import types
import unittest
from unittest.mock import patch

from atlas_host.profile_service import ProfileService, WatchdogLoop
from atlas_host.profile_os import OwnedProcess, strict_json, LinuxProfileHooks
from atlas_host.hosted_runtime import NativeChannel
from atlas_host.profile_api import PrivateProfileAPI
from atlas_host.supervisor import Supervisor
from profile_host_supervisor import Clock, Hooks
from profile_worker_primitives import Base, selected


class Meter:
    def __init__(self):
        self.frozen = False

    def read(self):
        return 0.0

    def freeze(self):
        self.frozen = True


class Context:
    def authorize(self, model, context, tab):
        if (model, context, tab) != ("m_" + "a" * 64, "ctx", "d" * 64):
            raise ValueError("foreign owner")

    def resolve(self, data):
        return {"binding": data["binding"]}


class ServiceTests(Base):
    def setUp(self):
        super().setUp()
        self.clock = Clock()
        self.watch = WatchdogLoop(Meter(), clock=lambda: self.clock.now)

        def hooks(source, slot):
            self.hooks = Hooks(self.os, self.clock)
            self.hooks.watch = lambda callback, deadline: slot.register(
                callback, deadline
            )
            return self.hooks

        self.service = ProfileService(
            Supervisor(),
            Context(),
            hooks,
            clock=lambda: self.clock.now,
            owner_meter=Meter(),
            watchdog=self.watch,
        )
        self.api = PrivateProfileAPI(self.service, Context().authorize)
        self.data = {
            "version": 1,
            "model_id": "m_" + "a" * 64,
            "context_id": "ctx",
            "tab_capability": "d" * 64,
            "binding": selected(),
            "seed": 17,
            "values": 5,
            "restart": False,
        }

    def start(self):
        g = self.clock.grant()
        _, r, _ = self.api.handle("start", self.data, admission=g)
        self.owner = {
            k: self.data[k]
            for k in ("version", "model_id", "context_id", "tab_capability")
        }
        self.owner.update(job_id=r["job_id"], job_capability=r["job_capability"])
        return g, r

    def finish(self):
        self.hooks.child.reaped = True
        for _ in range(30):
            self.service.step()
            if self.service.record["status"]["state"] in ("partial", "complete"):
                return
        self.fail(self.service.record["status"])

    def test_full_start_status_page_cancel_flow_has_no_poll_driven_work(self):
        g, r = self.start()
        self.service.step()
        self.assertFalse(hasattr(self, "hooks"))
        self.assertEqual(self.api.handle("status", self.owner)[1]["state"], "admitting")
        self.service.finish_admission(g)
        self.service.step()
        self.finish()
        status = self.api.handle("status", self.owner)[1]
        self.assertEqual(status["state"], "partial")
        self.assertNotIn("job_capability", status)
        page = self.api.handle(
            "page",
            {
                **self.owner,
                "revision": status["accepted"]["revision"],
                "axis": "rows",
                "start": 0,
                "count": 2,
            },
        )[1]
        self.assertEqual(page["end"], 2)
        self.api.handle("cancel", self.owner)
        self.service.step()
        self.service.step()
        self.assertEqual(self.api.handle("status", self.owner)[1]["state"], "cancelled")
        self.assertEqual(self.service.supervisor.snapshot_reservation, 0)

    def test_full_slice_completes_and_explicit_restart_replaces_owned_result(self):
        self.data["values"] = 12
        g, _ = self.start()
        self.service.finish_admission(g)
        self.service.step()
        self.finish()
        status = self.service.status(self.owner)
        self.assertEqual(status["state"], "complete")
        self.assertTrue(status["accepted"]["complete"])
        self.assertEqual(status["accepted"]["visited_values"], 12)
        self.data["restart"] = True
        self.data["seed"] = 18
        g, _ = self.start()
        self.service.finish_admission(g)
        self.service.step()
        self.finish()
        self.assertEqual(len(self.os.files), 1)
        self.api.handle("cancel", self.owner)
        self.service.step()
        self.assertFalse(self.os.files)

    def test_foreign_tab_cannot_read_cancel_or_heartbeat(self):
        self.start()
        for action in ("status", "cancel", "heartbeat"):
            with self.assertRaises(ValueError):
                self.api.handle(action, {**self.owner, "tab_capability": "c" * 64})

    def test_admission_deadline_before_owner_handoff_stops_without_spawn(self):
        g, _ = self.start()
        self.clock.now = 6
        self.watch.pulse()
        self.service.step()
        self.assertFalse(hasattr(self, "hooks"))
        self.assertFalse(self.service.supervisor.busy())
        self.assertEqual(self.service.record["status"]["state"], "cancelled")

    def test_lost_admission_response_reconciles_only_same_tab_without_new_grant(self):
        self.start()
        data = {
            k: self.data[k]
            for k in ("version", "model_id", "context_id", "tab_capability")
        }
        self.assertTrue(self.api.handle("reconcile", data)[1]["cleanup_pending"])
        self.service.step()
        self.assertFalse(self.api.handle("reconcile", data)[1]["cleanup_pending"])
        self.assertFalse(self.service.supervisor.busy())

    def test_completed_result_lease_expires_without_http(self):
        g, _ = self.start()
        self.service.finish_admission(g)
        self.service.step()
        self.finish()
        self.clock.now = 16
        self.service.step()
        self.service.step()
        self.assertEqual(self.service.record["status"]["state"], "cancelled")
        self.assertEqual(self.service.supervisor.snapshot_reservation, 0)

    def test_cancelled_child_retains_busy_until_reaped(self):
        g, _ = self.start()
        self.service.finish_admission(g)
        self.service.step()
        self.api.handle("cancel", self.owner)
        self.service.step()
        self.assertTrue(self.service.supervisor.busy())
        self.hooks.child.reaped = True
        self.service.step()
        self.service.step()
        self.assertFalse(self.service.supervisor.busy())

    def test_second_start_never_queues(self):
        self.start()
        with self.assertRaises(ValueError):
            self.api.handle("start", self.data, admission=self.clock.grant())

    def test_heartbeat_renews_only_owner_lease_not_original_grant(self):
        g, _ = self.start()
        end = g.deadline
        self.clock.now = 1
        self.api.handle("heartbeat", self.owner)
        self.assertEqual(g.deadline, end)
        self.assertEqual(self.service.record["lease"], 16)


class ProcessTests(unittest.TestCase):
    def make(self):
        self.clock = Clock()
        self.done = False
        process = types.SimpleNamespace(pid=42, stdout=None, returncode=None)

        def wait(pid, flags):
            return (
                (42, 0, types.SimpleNamespace(ru_utime=0.4, ru_stime=0.1))
                if self.done
                else (0, 0, None)
            )

        self.child = OwnedProcess(
            process,
            clock=lambda: self.clock.now,
            stats=lambda pid: {"start": 17, "cpu": 0.2, "rss": 100},
            descendants=lambda pid: set(),
            wait4=wait,
        )
        return self.child

    def test_wait4_final_cpu_replaces_sample_and_exit_does_not_need_popen_poll(self):
        c = self.make()
        self.assertFalse(c.sample()["reaped"])
        self.done = True
        s = c.sample()
        self.assertTrue(s["reaped"])
        self.assertEqual(s["cpu_seconds"], 0.5)
        self.assertEqual(c.sample()["cpu_seconds"], 0.5)
        self.assertEqual(c.process.returncode, 0)

    def test_cancel_escalates_while_already_stopping(self):
        c = self.make()
        with patch("atlas_host.profile_os.os.kill") as kill:
            c.stop()
            self.clock.now = 0.3
            c.sample()
            self.assertEqual(
                [x.args[1] for x in kill.call_args_list],
                [signal.SIGTERM, signal.SIGKILL],
            )
        self.assertFalse(c.reaped)

    def test_uninitialized_pipe_never_blocks_sample_and_closes_after_cancelled_reap(
        self,
    ):
        c = self.make()
        closed = []
        c.process.stdout = types.SimpleNamespace(
            fileno=lambda: 123, close=lambda: closed.append(True)
        )
        c.eof = False
        c.drainable = False
        c.read = lambda *a: (_ for _ in ()).throw(AssertionError("Blocking pipe read"))
        self.assertFalse(c.sample()["reaped"])
        with patch("atlas_host.profile_os.os.kill"):
            c.stop()
            self.done = True
            self.assertTrue(c.sample()["reaped"])
        self.assertTrue(closed)

    def test_strict_receipt_rejects_duplicate_trailing_nonfinite(self):
        for raw in [b'{"a":1,"a":2}', b"{}{}", b'{"a":NaN}']:
            with self.assertRaises(ValueError):
                strict_json(raw)

    def test_worker_argv_uses_trusted_source_and_only_output_fd(self):
        calls = []
        b = selected()
        source = {
            "root": "/owner/model",
            "revision": "pinned",
            "cache": "/owner/cache",
            "binding": b,
            "check": lambda: None,
        }
        book = types.SimpleNamespace(
            spawn=lambda argv, **kw: calls.append((argv, kw)) or object()
        )
        binary = types.SimpleNamespace(path="/owner/qualified-bin", check=lambda: None)
        h = LinuxProfileHooks(book, binary, source, None)
        h.spawn(
            {"binding": b, "seed": 0, "values": 1, "wall_ms": 1000, "cpu_ms": 500},
            19,
            None,
        )
        self.assertEqual(calls[0][1], {"pass_fds": (19,)})
        self.assertNotIn("tab_capability", " ".join(calls[0][0]))
        with self.assertRaises(ValueError):
            h.spawn({}, 19, 20)


class Socket:
    def __init__(self, wrong=False):
        self.sent = bytearray()
        self.out = None
        self.wrong = wrong

    def send(self, data):
        n = min(7, len(data))
        self.sent.extend(data[:n])
        return n

    def recv(self, n):
        if self.out is None:
            cmd = json.loads(self.sent[4:])
            ack = {
                "version": 1,
                "operation_id": "wrong" if self.wrong else cmd["operation_id"],
                "complete": True,
                "numeric_idle": True,
            }
            body = b"{}"
            header = json.dumps(
                {
                    "ack": ack,
                    "status": 200,
                    "mime": "application/json",
                    "body_bytes": len(body),
                }
            ).encode()
            self.out = bytearray(struct.pack("!I", len(header)) + header + body)
        part = bytes(self.out[: min(n, 11)])
        del self.out[: len(part)]
        return part

    def close(self):
        pass


class WatchTests(unittest.TestCase):
    def test_failed_callback_is_retained_and_cannot_be_reported_clean(self):
        loop = WatchdogLoop(Meter(), clock=lambda: 0)
        slot = loop.bind(lambda: (_ for _ in ()).throw(ValueError("fault")), 5)
        loop.pulse()
        self.assertTrue(slot.failed)
        with self.assertRaises(ValueError):
            slot.close()


class NativeTests(unittest.TestCase):
    def make(self, wrong=False):
        self.clock = Clock()
        self.s = Supervisor()
        self.meter = Meter()
        self.watch = WatchdogLoop(Meter(), clock=lambda: self.clock.now)
        self.child = types.SimpleNamespace(
            cpu=0.0, unexpected=set(), reaped=False, stopped=False
        )
        self.child.sample = lambda: {
            "cpu_seconds": 0.1,
            "reaped": self.child.reaped,
            "exit_code": 0,
            "receipt": None,
        }
        self.child.stop = lambda: setattr(self.child, "stopped", True)
        service = types.SimpleNamespace(
            admission=lambda: (self.clock.grant(), self.meter), watchdog=self.watch
        )
        book = types.SimpleNamespace(
            resources=lambda: {
                "rss_bytes": 100,
                "available_bytes": 6 * 1024**3,
                "all_owned_accounted": True,
                "descendants_clear": True,
            }
        )
        self.channel = NativeChannel(
            Socket(wrong),
            self.child,
            self.s,
            service,
            book,
            clock=lambda: self.clock.now,
            wait=lambda *a: None,
        )
        return self.channel

    def test_matching_native_ack_releases_and_partial_io_is_bounded(self):
        c = self.make()
        self.assertEqual(c.read("/api/model", "ctx"), (200, b"{}", "application/json"))
        self.assertFalse(self.s.busy())
        self.assertTrue(self.meter.frozen)

    def test_wrong_ack_keeps_slot_and_independent_reap_releases(self):
        c = self.make(True)
        with self.assertRaises(ValueError):
            c.read("/api/model", "ctx")
        self.assertTrue(self.s.busy())
        self.assertTrue(self.child.stopped)
        self.child.reaped = True
        self.watch.pulse()
        self.assertFalse(self.s.busy())

    def test_native_busy_never_sends_socket_bytes(self):
        c = self.make()
        self.s.acquire("profile", "ctx")
        with self.assertRaises(ValueError):
            c.read("/api/model", "ctx")
        self.assertFalse(c.stream.sent)
        self.assertTrue(self.meter.frozen)


class HttpTests(unittest.TestCase):
    def request(
        self,
        changes=None,
        duplicate=False,
        enabled=True,
        body=b"{}",
        path="/api/profiles/status",
    ):
        import io
        from email.message import Message
        from atlas_host.profile_http import HostedHandler

        h = object.__new__(HostedHandler)
        h.command = "POST"
        h.path = path
        h.rfile = io.BytesIO(body)
        h.headers = Message()
        for k, v in {
            "Host": "127.0.0.1:8798",
            "Origin": "http://127.0.0.1:8798",
            "X-Atlas-Local": "1",
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
            **(changes or {}),
        }.items():
            h.headers[k] = v
        if duplicate:
            h.headers["Host"] = "127.0.0.1:8798"
        calls = []
        responses = []
        api = types.SimpleNamespace(
            handle=lambda *a, **kw: calls.append((a, kw))
            or (200, {"state": "running"}, {})
        )
        h.server = types.SimpleNamespace(
            server_port=8798,
            application=types.SimpleNamespace(api=api),
            profile_controls=enabled,
        )
        h.profile_admission = object()
        h.send = lambda *a: responses.append(a)
        h.handle_action()
        return calls, responses

    def test_guarded_private_post_routes_without_live_server(self):
        calls, responses = self.request()
        self.assertEqual(len(calls), 1)
        self.assertEqual(responses[0][0], 200)
        self.assertIsNone(calls[0][1]["admission"])

    def test_disabled_production_gate_refuses_before_service(self):
        calls, r = self.request(enabled=False)
        self.assertFalse(calls)
        self.assertEqual(r[0][0], 409)

    def test_foreign_origin_host_missing_local_header_duplicate_and_query_refuse(self):
        for changes in [
            {"Origin": "https://elsewhere.invalid"},
            {"Host": "elsewhere.invalid"},
            {"X-Atlas-Local": "0"},
        ]:
            self.assertFalse(self.request(changes)[0])
        self.assertFalse(self.request(duplicate=True)[0])
        self.assertFalse(self.request(path="/api/profiles/status?x=1")[0])

    def test_duplicate_json_and_oversized_body_refuse(self):
        self.assertFalse(self.request(body=b'{"x":1,"x":2}')[0])
        self.assertFalse(self.request(body=b" " * 8193)[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
