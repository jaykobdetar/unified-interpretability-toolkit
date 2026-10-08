"""Native and owner-clock boundaries with existing inert lifecycle fixtures."""

from copy import deepcopy
import math
from pathlib import Path
import signal
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import host_picker_contracts as host_fixture
import profile_runtime_doubles as runtime
import profile_spawn_registration as registration
from python_limit_transport_contracts import ReplySocket
from atlas_host import hosted_runtime, profile_os, profile_platform
from atlas_host.common import canonical
from atlas_host.profile_observation import SpawnObservationPending


class RuntimeLifetimeLimits(unittest.TestCase):
    def native(self, *, static=False, path="/api/model", header=None, body=2):
        case = runtime.NativeTests(methodName="runTest")
        channel = case.make()
        channel.stream = ReplySocket(body, header)
        if not static:
            return case, channel, lambda: channel.read(path, "ctx")
        operation = object.__new__(hosted_runtime.StaticOperation)
        operation.app = NS(supervisor=case.s)
        operation.token = case.s.acquire("metadata", "ctx")
        operation.grant = case.clock.grant()
        operation.sent = False
        operation.check = Mock()
        return case, channel, lambda: channel.read(path, "ctx", operation=operation)

    def test_both_native_paths_query_and_command_byte_edges(self):
        for static in (False, True):
            for fields, accepted in ((0, True), (12, True), (13, False)):
                path = "/api/model?" + "&".join(f"q{n}=x" for n in range(fields))
                case, channel, read = self.native(static=static, path=path)
                # Inert transport admission only: this does not assert that a
                # real model endpoint accepts these arbitrary parameter names.
                if accepted:
                    self.assertEqual(read()[0], 200)
                else:
                    with self.assertRaisesRegex(ValueError, "Max number of fields"):
                        read()
                    self.assertFalse(channel.stream.sent)
            for size, accepted in ((16384, True), (16385, False)):
                case, channel, read = self.native(static=static)
                original = case.s.native_command

                def command(*args):
                    value = original(*args)
                    value["padding"] = ""
                    value["padding"] = "x" * (size - len(canonical(value)))
                    self.assertEqual(len(canonical(value)), size)
                    return value

                with patch.object(case.s, "native_command", side_effect=command):
                    if accepted:
                        self.assertEqual(read()[0], 200)
                    else:
                        with self.assertRaisesRegex(ValueError, "command exceeds"):
                            read()
                        self.assertFalse(channel.stream.sent)
                # This trusted command double isolates the framing guard rather
                # than claiming an accepted native command schema with padding.

    def test_static_path_actual_header_and_body_limits(self):
        for header, body, accepted in (
            (None, 0, True),
            (None, 2097152, True),
            (None, -1, False),
            (None, 2097153, False),
            (16384, 2, True),
            (16385, 2, False),
            (0, 2, False),
        ):
            _, _, read = self.native(static=True, header=header, body=body)
            if accepted:
                self.assertEqual((read()[0]), 200)
            else:
                with self.assertRaises(ValueError):
                    read()
        # An empty/one-byte header cannot encode the closed JSON response.

    def test_native_io_absolute_deadline_and_idle_cpu_observation(self):
        case, channel, _ = self.native()
        for now, accepted in ((math.nextafter(5.0, 0.0), True), (5.0, False)):
            case.clock.now = now
            if accepted:
                channel._io(b"x", None, 5)
            else:
                with self.assertRaisesRegex(ValueError, "deadline expired"):
                    channel._io(b"x", None, 5)
        for cpu, accepted in (
            (0.0, True),
            (math.nextafter(4.0, 0.0), True),
            (4.0, False),
            (-1.0, False),
            (math.inf, False),
            (math.nan, False),
        ):
            _, channel, _ = self.native()
            channel.child.sample = lambda: {"cpu_seconds": cpu}
            if accepted:
                channel.idle_pulse()
            else:
                with self.assertRaises(ValueError):
                    channel.idle_pulse()

    def host(self):
        case = host_fixture.HostFixtureTests(methodName="runTest")
        case.setUp()
        self.addCleanup(case.doCleanups)
        return case

    def test_reader_four_tab_slots_and_15_second_lease_edge(self):
        case = self.host()
        leases = [case.acquire() for _ in range(4)]
        self.assertEqual(len(case.host.leases), 4)
        with self.assertRaisesRegex(
            host_fixture.HostError, "capacity reached"
        ) as error:
            case.host.acquire({"model_id": case.ids[0]})
        self.assertEqual(
            (error.exception.status, error.exception.code), (409, "reader_busy")
        )
        case.read(leases[0])
        case.now = math.nextafter(15.0, 0.0)
        case.host.tick()
        self.assertEqual(len(case.host.leases), 4)
        case.now = 15.0
        case.host.tick()
        self.assertFalse(case.host.leases)
        self.assertIsNone(case.host.reader)

    def test_unbound_reader_exact_five_second_start_deadline(self):
        for now, alive in ((math.nextafter(5.0, 0.0), True), (5.0, False)):
            case = self.host()
            case.acquire()
            case.now = now
            case.host.tick()
            self.assertEqual(case.host.reader is not None, alive)

    def test_profile_child_effective_wall_and_cpu_clamps(self):
        for wall, cpu, accepted in (
            (200, 1, False),
            (201, 1, True),
            (5000, 4000, True),
            (201, 0, False),
        ):
            platform = object.__new__(profile_platform.ProfilePlatform)
            platform.spawn_attempted = False
            platform.grant = NS(
                remaining=Mock(return_value={"wall_ms": wall, "cpu_ms": cpu})
            )
            platform.hooks = NS(spawn=Mock(return_value=object()))
            request = {"wall_ms": 5000, "cpu_ms": 4000}
            if accepted:
                platform.spawn(request, 19, None)
                platform.grant.remaining.assert_called_once_with(reserve_ms=800)
                self.assertEqual(
                    platform.hooks.spawn.call_args.args[0],
                    {"wall_ms": wall, "cpu_ms": cpu},
                )
                self.assertEqual(request, {"wall_ms": 5000, "cpu_ms": 4000})
            else:
                with self.assertRaisesRegex(ValueError, "child allowance"):
                    platform.spawn(request, 19, None)
                platform.hooks.spawn.assert_not_called()

    def test_postspawn_observation_window_zero_and_50_ms(self):
        for now, pending in (
            (-math.ulp(0.0), False),
            (0.0, True),
            (math.nextafter(0.05, 0.0), True),
            (0.05, False),
        ):
            case = registration.SpawnRegistration(methodName="runTest")
            case.setUp()
            self.addCleanup(case.doCleanups)
            case.now = now
            if pending:
                with self.assertRaises(SpawnObservationPending):
                    case.book.resources()
            else:
                result = case.book.resources()
                self.assertFalse(result["all_owned_accounted"])
                self.assertEqual(result["available_bytes"], 0)

    def test_owned_stop_escalation_exact_200_ms(self):
        for now, killed in ((math.nextafter(0.2, 0.0), False), (0.2, True)):
            case = runtime.ProcessTests(methodName="runTest")
            child = case.make()
            with patch.object(child, "_signal") as send:
                child.stop()
                case.clock.now = now
                child.stop()
                self.assertEqual(send.call_args_list[0].args, (signal.SIGTERM,))
                self.assertEqual(child.kill_sent, killed)
                self.assertEqual(send.call_count, 1 + int(killed))
                if killed:
                    self.assertEqual(send.call_args_list[1].args, (signal.SIGKILL,))


if __name__ == "__main__":
    unittest.main()
