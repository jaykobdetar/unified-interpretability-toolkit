"""Caller-specific runtime boundaries using existing fake OS and channel fixtures."""

import ast
from copy import deepcopy
from email.message import Message
import hashlib
import io
import json
import math
from pathlib import Path
import resource
import sys
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import host_atlas
import host_picker_contracts as picker
import live_inference as live
import profile_runtime_doubles as runtime
import profile_worker_primitives as primitive
import python_limit_runtime_lifetime_contracts as lifetime
from atlas_host import (
    hosted_runtime as hosted,
    profile_http,
    profile_snapshot as snapshot,
)
from atlas_host import profile_worker
from atlas_host.profile_platform import ProfilePlatform
from atlas_host.supervisor import Supervisor

MIB, GIB = 1024**2, 1024**3


class RuntimeCompletionLimits(unittest.TestCase):
    def check(self, operation, accepted):
        if accepted:
            return operation()
        with self.assertRaises(ValueError):
            operation()
        return None

    def fixture(self, cls):
        case = cls(methodName="runTest")
        case.setUp()
        self.addCleanup(case.doCleanups)
        return case

    def test_singleton_queries_in_all_native_and_renderer_callers(self):
        fixture = lifetime.RuntimeLifetimeLimits(methodName="runTest")
        for static in (False, True):
            for values, accepted in (([], False), (["1"], True), (["1", "2"], False)):
                _, channel, read = fixture.native(static=static)
                with patch.object(hosted, "parse_qs", return_value={"q": values}):
                    self.check(read, accepted)
                    if not accepted:
                        self.assertFalse(channel.stream.sent)
            _, channel, read = fixture.native(static=static, path="/api/model?q=1&q=2")
            self.check(read, False)
            self.assertFalse(channel.stream.sent)
        case = self.fixture(picker.HostFixtureTests)
        lease = case.acquire()
        for values, accepted in (
            ([], False),
            ([lease["context_id"]], True),
            ([lease["context_id"]] * 2, False),
        ):
            self.check(
                lambda: case.host.read(lease["model_id"], "model", {"context": values}),
                accepted,
            )
        # parse_qs cannot produce a zero-length value list. The empty-list
        # native cases exercise the defensive guard with a parser double.

    def test_native_guard_resources_and_hosted_authorization_clock(self):
        for rss, available, charge, accepted in (
            (768 * MIB, 13 * GIB // 4, 0, True),
            (768 * MIB + 1, 13 * GIB // 4, 0, False),
            (768 * MIB - 1, 13 * GIB // 4, 1, True),
            (768 * MIB, 13 * GIB // 4, 1, False),
            (0, 13 * GIB // 4 - 1, 0, False),
        ):
            case = runtime.NativeTests(methodName="runTest")
            channel = case.make()
            channel.book.resources = lambda: {
                "rss_bytes": rss,
                "available_bytes": available,
                "all_owned_accounted": True,
                "descendants_clear": True,
            }
            with patch.object(case.s, "charged_snapshot_bytes", return_value=charge):
                self.check(lambda: channel.read("/api/model", "ctx"), accepted)
        host = NS(
            stopping=False,
            entry={"model_id": "m"},
            context="c",
            _owns=lambda *_: True,
            leases={"t": 15},
            clock=Mock(return_value=0),
        )
        contexts = hosted.HostedContexts(host, threading.RLock())
        for now, accepted in (
            (0, True),
            (math.nextafter(15.0, 0), True),
            (15, False),
            (math.nextafter(15.0, math.inf), False),
        ):
            host.clock.return_value = now
            self.check(lambda: contexts.authorize("m", "c", "t"), accepted)

    def test_hosted_http_body_headers_query_and_write_budget_edges(self):
        fixture = runtime.HttpTests(methodName="runTest")
        for size, accepted in ((0, False), (8192, True), (8193, False)):
            raw = b"{}" + b" " * (size - 2) if size else b""
            calls, response = fixture.request(body=raw)
            self.assertEqual(bool(calls), accepted)
            self.assertEqual(response[0][0], 200 if accepted else 409)
        with patch.object(profile_http, "strict_json", return_value={}):
            self.assertTrue(fixture.request(body=b"0")[0])
        # One byte cannot encode the required JSON object; this selectively
        # isolates private length admission, rather than claiming a valid body.
        for key in (
            "Host",
            "Origin",
            "X-Atlas-Local",
            "Content-Type",
            "Sec-Fetch-Site",
        ):
            handler = object.__new__(profile_http.HostedHandler)
            handler.headers = Message()
            handler.headers[key] = "inert"
            handler.headers[key] = "inert"
            handler.server = NS(application=NS())
            handler.validated = Mock(return_value=("/api/profiles/status", 2))
            handler.send = Mock()
            handler.handle_action()
            self.assertEqual(handler.send.call_args.args[0], 409)
        for context, accepted in (("c", True), ("c&context=c", False)):
            handler = object.__new__(profile_http.HostedHandler)
            handler.command = "GET"
            path = "/api/models/m/binding"
            handler.path = path + "?context=" + context
            handler.headers = Message()
            handler.validated = Mock(return_value=(path, 0))
            host = NS(
                _validate_context=Mock(),
                view_kind="fixture",
                reader=NS(
                    read=Mock(
                        return_value=(200, b'{"source_binding":{}}', "application/json")
                    )
                ),
            )
            app = NS(host=host, lock=threading.RLock(), contexts=NS(remember=Mock()))
            handler.server = NS(application=app)
            handler.send = Mock()
            handler.handle_action()
            self.assertEqual(handler.send.call_args.args[0], 200 if accepted else 409)
            self.assertEqual(host.reader.read.called, accepted)
        for size, accepted in ((0, True), (2 * MIB, True), (2 * MIB + 1, False)):
            handler = object.__new__(profile_http.HostedHandler)
            operation = NS(
                finished=False,
                check=Mock(),
                abort=Mock(),
                grant=NS(remaining=lambda: {"wall_ms": 5000}),
                publish=lambda write: write(),
            )
            handler.static_finalizer = operation
            handler.connection = Mock()
            with patch.object(profile_http.HostHandler, "send") as send:
                self.check(lambda: handler.send(200, b"x" * size), accepted)
                self.assertEqual(send.called, accepted)
                self.assertEqual(operation.abort.called, not accepted)
        for wall, expected in (
            (1, 0.001),
            (2999, 2.999),
            (3000, 3),
            (3001, 3),
            (5000, 3),
        ):
            handler = object.__new__(profile_http.HostedHandler)
            handler.static_finalizer = NS(
                finished=False,
                check=Mock(),
                abort=Mock(),
                grant=NS(remaining=lambda: {"wall_ms": wall}),
                publish=lambda write: write(),
            )
            handler.connection = Mock()
            with patch.object(profile_http.HostHandler, "send"):
                handler.send(200, b"")
            handler.connection.settimeout.assert_called_once_with(expected)
        # Remaining AdmissionGrant duration may floor to zero, as separately
        # pinned; this layer clamps only the upper write timeout, not its lower.

    def test_profile_resource_integer_domain_and_charged_snapshot_edges(self):
        high = 2**63 - 1
        for field in ("snapshot_reservation", "actual"):
            for number, accepted in (
                (-1, False),
                (0, True),
                (high, True),
                (high + 1, False),
                (True, False),
            ):
                supervisor = Supervisor()
                supervisor.profile_session = NS(
                    job=NS(
                        store=NS(owned_storage_bytes=number if field == "actual" else 0)
                    )
                )
                supervisor.snapshot_reservation = (
                    number if field == "snapshot_reservation" else 0
                )
                result = self.check(supervisor.charged_snapshot_bytes, accepted)
                if accepted:
                    self.assertEqual(result, number)
        supervisor = Supervisor()
        self.assertEqual(supervisor.charged_snapshot_bytes(), 0)
        supervisor.snapshot_reservation = 1
        self.check(supervisor.charged_snapshot_bytes, False)
        supervisor.snapshot_reservation = 0
        platform = ProfilePlatform(
            supervisor,
            supervisor.acquire("profile", "ctx"),
            Mock(),
            NS(resources=Mock()),
        )
        for rss, available, frame, accepted in (
            (0, 13 * GIB // 4, 0, True),
            (-1, 13 * GIB // 4, 0, False),
            (0, -1, 0, False),
            (0, high, 0, True),
            (0, high + 1, 0, False),
            (0, 13 * GIB // 4, -1, False),
            (0, 13 * GIB // 4, 32 * MIB, True),
            (0, 13 * GIB // 4, 32 * MIB + 1, False),
        ):
            platform.hooks.resources.return_value = {
                "rss_bytes": rss,
                "available_bytes": available,
                "all_owned_accounted": True,
                "descendants_clear": True,
            }
            self.check(lambda: platform.resources_ok(frame), accepted)
        # RSS signed63's upper is masked by the smaller 768 MiB envelope.
        # Its full legal aggregate upper is covered by existing resource tests.

    def test_snapshot_seed_all_callers_and_validation_chunk_bound(self):
        fixture = self.fixture(primitive.Base)
        selected = primitive.selected(1, 1)
        for seed, accepted in (
            (-1, False),
            (0, True),
            (2**32 - 1, True),
            (2**32, False),
        ):
            self.check(lambda: snapshot.profile_identity(selected, seed), accepted)
            store = self.check(lambda: snapshot.SnapshotStore(selected, seed), accepted)
            if store is not None:
                store.close()
            raw, *_ = primitive.frame(
                selected, seed=seed if accepted else 17, visited=0
            )
            self.check(lambda: fixture.validate(raw, selected, seed=seed), accepted)
        selected = primitive.selected(257, 1)
        raw, *_ = primitive.frame(selected, visited=0)
        with patch.object(fixture.os, "pread", wraps=fixture.os.pread) as pread:
            fixture.validate(raw, selected)
        sizes = [call.args[1] for call in pread.call_args_list]
        self.assertIn(256 * 24, sizes)
        self.assertIn(24, sizes)
        self.assertLessEqual(max(sizes), 256 * 24)

    def test_profile_tick_receipt_visit_and_validation_slice_edges(self):
        fixture = self.fixture(primitive.LifecycleTests)
        for visits, accepted in ((-1, False), (0, True), (1, True), (2, False)):
            platform, job, _ = fixture.job(values=1)
            platform.child.reaped = True
            platform.child.receipt.update(visited_values=visits, new_values=visits)
            steps = []

            def validate(*_a, **_k):
                while True:
                    steps.append(None)
                    yield None

            with patch.object(job.store, "publication_steps", side_effect=validate):
                job.tick()
            self.assertEqual(job.state == "validating", accepted)
            self.assertEqual(len(steps), 4 if accepted else 0)
            job.store.close()
        self.assertEqual(profile_worker.POLL_SLICE, 0.002)
        for now, expected_steps in (
            (0.0, 4),
            (math.nextafter(0.002, 0), 4),
            (0.002, 0),
        ):
            platform, job, _ = fixture.job(values=1)
            platform.child.reaped = True
            steps = []

            def validate(*_a, **_k):
                while True:
                    steps.append(None)
                    yield None

            # Freeze the slice-start sample at zero, then advance the injected
            # clock before the first yielded validation step.
            clock = Mock(
                side_effect=[
                    0,
                    now,
                    now,
                    now,
                    now,
                    now,
                    now,
                    now,
                    now,
                    now,
                    now,
                    now,
                    now,
                    now,
                    now,
                    now,
                ]
            )
            with (
                patch.object(job.store, "publication_steps", side_effect=validate),
                patch.object(platform, "clock", clock),
                patch.object(job, "_budget"),
            ):
                job.tick()
            self.assertEqual(
                len(steps),
                expected_steps,
                (now, job.state, job.error, clock.call_count),
            )
            job.store.close()
        for size, accepted in ((16384, True), (16385, False)):
            platform, job, _ = fixture.job(values=1)
            platform.child.reaped = True
            with (
                patch.object(profile_worker, "canonical", return_value=b"x" * size),
                patch.object(
                    job.store, "publication_steps", return_value=iter([None] * 10)
                ),
            ):
                job.tick()
            self.assertEqual(job.state == "validating", accepted)
            job.store.close()
        # Closed receipt fields cannot occupy 16 KiB; only the serialization
        # boundary is isolated. Normal visit/slice cases use real serialization.

    def test_legacy_latency_fixed_resource_and_startup_wait_guards(self):
        tree = ast.parse((ROOT / "tools/final_open_latency.py").read_text())
        loops = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.While) and "Startup timeout" in ast.unparse(node)
        ]
        self.assertEqual(len(loops), 1)
        for elapsed, accepted in (
            (math.nextafter(30.0, 0), True),
            (30.0, False),
            (math.nextafter(30.0, math.inf), False),
        ):
            namespace = {
                "proc": NS(poll=lambda: None),
                "get": Mock(side_effect=[urllib.error.URLError("inert"), (b"{}", {})]),
                "base": "inert",
                "start": 0,
                "time": NS(perf_counter=lambda: elapsed, sleep=Mock()),
                "urllib": urllib,
            }
            operation = lambda: exec(
                compile(
                    ast.Module(body=loops, type_ignores=[]),
                    "original-startup-wait",
                    "exec",
                ),
                namespace,
            )
            if accepted:
                operation()
                namespace["time"].sleep.assert_called_once_with(0.003)
            else:
                with self.assertRaisesRegex(AssertionError, "Startup timeout"):
                    operation()
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        limit = next(
            node for node in calls if ast.unparse(node.func) == "resource.setrlimit"
        )
        wait = next(node for node in calls if ast.unparse(node.func) == "proc.wait")
        namespace = {
            "resource": NS(RLIMIT_AS=resource.RLIMIT_AS, setrlimit=Mock()),
            "proc": NS(wait=Mock()),
        }
        for call in (limit, wait):
            exec(
                compile(
                    ast.fix_missing_locations(
                        ast.Module(body=[ast.Expr(value=call)], type_ignores=[])
                    ),
                    "original-fixed-wait-limit",
                    "exec",
                ),
                namespace,
            )
        namespace["resource"].setrlimit.assert_called_once_with(
            resource.RLIMIT_AS, (768 * MIB, 768 * MIB)
        )
        namespace["proc"].wait.assert_called_once_with(timeout=10)

    def test_validation_slice_respects_earlier_absolute_deadline(self):
        fixture = self.fixture(primitive.LifecycleTests)
        for now, expected in ((math.nextafter(0.001, 0), 4), (0.001, 0)):
            platform, job, _ = fixture.job(values=1)
            platform.child.reaped = True
            job.deadline = 0.001
            steps = []

            def validate(*_a, **_k):
                while True:
                    steps.append(None)
                    yield None

            clock = Mock(side_effect=[0] + [now] * 8)
            with (
                patch.object(job.store, "publication_steps", side_effect=validate),
                patch.object(platform, "clock", clock),
                patch.object(job, "_budget"),
            ):
                job.tick()
            self.assertEqual(len(steps), expected)
            job.store.close()
        # The original time/resource budget is isolated only for this min()
        # branch. Separate real-budget tests retain the cleanup reserve.

    def test_legacy_renderer_admission_and_actual_notice_bytes(self):
        entry = {"root": "/inert", "name": "inert", "manifest": {"revision": "a" * 40}}
        for available, accepted in ((19 * GIB // 4 - 1, False), (19 * GIB // 4, True)):
            with (
                patch.object(host_atlas, "available", return_value=available),
                patch.object(Path, "is_file", return_value=True),
                patch.object(
                    host_atlas.subprocess, "Popen", return_value=Mock()
                ) as spawn,
            ):
                self.check(
                    lambda: host_atlas.Renderer(entry, "/inert", 8796, None), accepted
                )
                self.assertEqual(spawn.called, accepted)
            with (
                patch.object(host_atlas, "available", return_value=available),
                patch.object(
                    host_atlas,
                    "load_config",
                    return_value={
                        "paths": {"registry": "/inert", "cache": "/inert"},
                        "ports": {"renderer": 8796},
                    },
                ),
                patch.object(host_atlas.os, "sched_setaffinity"),
                patch.object(host_atlas, "Registry", return_value=Mock()),
                patch.object(
                    host_atlas,
                    "FixtureHost",
                    side_effect=RuntimeError("Reached inert bootstrap boundary"),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError if accepted else SystemExit,
                    "inert bootstrap" if accepted else "4.75 GiB",
                ):
                    host_atlas.main(["--config", "inert"])
        base = json.dumps({"listening": "http://127.0.0.1:8796"}).encode()
        for size, accepted in ((len(base) + 1, True), (8192, True), (8193, False)):
            renderer = object.__new__(host_atlas.Renderer)
            renderer.raw, renderer.announced, renderer.port = bytearray(), False, 8796
            renderer.process = NS(stdout=NS(fileno=lambda: 19), poll=lambda: None)
            raw = base + b" " * (size - len(base) - 1) + b"\n"
            with patch.object(host_atlas.os, "read", return_value=raw) as read:
                self.check(renderer.ready, accepted)
                read.assert_called_once_with(19, 8193)

    def test_optional_layout_receipt_config_total_bytes_and_deadline_edges(self):
        for config_size, total, accepted in (
            (65536, 384 * MIB, True),
            (65537, 384 * MIB, False),
            (65536, 384 * MIB + 1, False),
        ):
            raw = b"x" * config_size
            receipt = {
                "files": {
                    "config.json": (1, 1, config_size, 1, 1),
                    "model.safetensors": (1, 1, total - config_size, 1, 1),
                }
            }
            with (
                patch.object(live, "layout_fingerprints", return_value=receipt),
                patch.object(
                    live,
                    "MANIFEST",
                    {"files": {"config.json": hashlib.sha256(raw).hexdigest()}},
                ),
                patch.object(
                    Path, "open", side_effect=lambda *_a, **_k: io.BytesIO(raw)
                ),
                patch.object(live.time, "monotonic", return_value=0),
                patch.object(live, "available", return_value=6 * GIB),
                patch.object(
                    live, "verify_model", side_effect=lambda _path, check: check()
                ) as verify,
            ):
                self.assertEqual(
                    live.verified_layout_receipt(Path("/inert"), required=False),
                    receipt if accepted else None,
                )
                self.assertEqual(verify.called, accepted)
        # Pinned-config hash and source verification are trusted doubles. This
        # proves bounded optional admission, not a qualified model installation.
        for elapsed, accepted in ((math.nextafter(5.0, 0), True), (5.0, False)):
            raw = b"{}"
            receipt = {"files": {"config.json": (1, 1, 2, 1, 1)}}
            with (
                patch.object(live, "layout_fingerprints", return_value=receipt),
                patch.object(
                    live,
                    "MANIFEST",
                    {"files": {"config.json": hashlib.sha256(raw).hexdigest()}},
                ),
                patch.object(
                    Path, "open", side_effect=lambda *_a, **_k: io.BytesIO(raw)
                ),
                patch.object(live.time, "monotonic", side_effect=[0, elapsed]),
                patch.object(live, "available", return_value=6 * GIB),
                patch.object(
                    live, "verify_model", side_effect=lambda _path, check: check()
                ),
            ):
                self.assertEqual(
                    live.verified_layout_receipt(Path("/inert"), required=False),
                    receipt if accepted else None,
                )

    def test_proxy_complete_body_and_deadline_tick_clamps(self):
        for size, accepted in ((0, True), (2 * MIB, True), (2 * MIB + 1, False)):
            response = NS(
                status=200,
                read=Mock(return_value=b"x" * size),
                getheader=Mock(return_value="application/json"),
                close=Mock(),
            )
            connection = NS(
                sock=Mock(),
                connect=Mock(),
                request=Mock(),
                getresponse=Mock(return_value=response),
                close=Mock(),
            )
            with (
                patch.object(
                    live.http.client, "HTTPConnection", return_value=connection
                ),
                patch.object(live.time, "monotonic", return_value=0),
                patch.object(live, "available", return_value=6 * GIB),
            ):
                operation = lambda: live.proxy_request(
                    8796, "GET", "/api/model", NS(tick=Mock())
                )
                if accepted:
                    self.assertEqual(operation()[1], b"x" * size)
                else:
                    with self.assertRaises(live.BackendError) as error:
                        operation()
                    self.assertEqual(
                        (error.exception.status, error.exception.code),
                        (502, "backend_response_limit"),
                    )
            response.read.assert_called_once_with(2 * MIB + 1)
            response.close.assert_called_once()
            connection.close.assert_called_once()
        for remaining, expected in (
            (0.001, 0.001),
            (0.099999, 0.099999),
            (0.1, 0.1),
            (0.100001, 0.1),
        ):
            for writer in (False, True):
                deadline = NS(remaining=Mock(return_value=remaining))
                connection = Mock()
                connection.recv_into.return_value = 1
                connection.send.return_value = 1
                if writer:
                    live.DeadlineSocket(connection, deadline).sendall(b"x")
                else:
                    reader = live.DeadlineReader(connection, deadline)
                    self.assertEqual(reader.readinto(bytearray(1)), 1)
                    reader.close()
                connection.settimeout.assert_called_once_with(expected)
                self.assertEqual(deadline.remaining.call_count, 2)


if __name__ == "__main__":
    unittest.main()
