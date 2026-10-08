"""Deterministic policy/lifetime regressions; no thread, process, memfd or socket."""

from contextlib import ExitStack
import hashlib, json
from pathlib import Path
import sys, tempfile, types, unittest
from unittest.mock import Mock, patch

from atlas_host import validation_policy as policy
from atlas_host.hosted_runtime import HostedApplication
from atlas_host.lifetime_guard import LifetimeGuard
from atlas_host.profile_observation import SpawnObservationPending
from atlas_host.profile_os import ProcessBook, LinuxProfileHooks
from atlas_host.profile_platform import ProfilePlatform
from atlas_host.profile_service import WatchdogLoop, ProfileService
from atlas_host.supervisor import Supervisor
import profile_host_supervisor as provider
import profile_runtime_doubles as doubles


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def observed(rss=0, available=policy.STOP_RESERVE):
    return dict(
        rss_bytes=rss,
        available_bytes=available,
        all_owned_accounted=True,
        descendants_clear=True,
    )


class Pure(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for target in (
            "subprocess.Popen",
            "socket.socketpair",
            "threading.Thread.start",
            "os.pidfd_open",
            "os.kill",
            "signal.pidfd_send_signal",
            "os.memfd_create",
        ):
            self.stack.enter_context(
                patch(target, side_effect=AssertionError("Live work forbidden"))
            )


class PolicyTests(Pure):
    def setUp(self):
        super().setUp()
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(
            patch.object(policy, "package_root", return_value=self.root)
        )
        fixture = b"tiny synthetic fixture"
        binary = b"fake binary, never executed"
        self.stack.enter_context(patch.object(policy, "BINARY_SHA", digest(binary)))
        self.stack.enter_context(patch.object(policy, "FIXTURE_SHA", digest(fixture)))
        for name, raw in {
            "Cargo.toml": b"test",
            "Cargo.lock": b"test",
            "tools/make_fixture.py": b"# fixture",
            "tools/atlas_host/example.py": b"# source",
            "src/lib.rs": b"// source",
            "fixtures/tiny-bf16/tiny.safetensors": fixture,
            "fixtures/tiny-bf16/host-manifest.json": b"{}",
            "target/release/weight-atlas-rust": binary,
            "run.py": b"# monitor",
            "runtime_harness.py": b"# harness",
        }.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        self.receipt = self.root / "reviewed-recipe.json"
        self.data = {
            "version": 1,
            "recipe": policy.RECIPE,
            "binary_sha256": policy.BINARY_SHA,
            "fixture_sha256": policy.FIXTURE_SHA,
            "topology": policy.TOPOLOGY,
            "source_sha256": {
                name: digest((self.root / name).read_bytes())
                for name in policy.source_names(self.root)
            },
            "monitor_sha256": digest((self.root / "run.py").read_bytes()),
            "harness_sha256": digest((self.root / "runtime_harness.py").read_bytes()),
        }

    def bind(self, data=None, approved=None):
        raw = json.dumps(self.data if data is None else data).encode()
        self.receipt.write_bytes(raw)
        return policy.bind_reviewed_recipe(
            self.receipt,
            approved or digest(raw),
            monitor_path=self.root / "run.py",
            harness_path=self.root / "runtime_harness.py",
            binary_path=self.root / "target/release/weight-atlas-rust",
        )

    def test_receipt_binds_exact_source_binary_scripts_fixture_and_fixed_command(self):
        bound = self.bind()
        bound.check()
        self.assertEqual(bound.start_bytes, 17 * policy.GIB // 4)
        self.assertEqual(
            bound.command("/usr/bin/python3", self.root),
            [
                "/usr/bin/python3",
                "-B",
                str(self.root / "runtime_harness.py"),
                str(self.root),
                bound.receipt_sha256,
            ],
        )
        for interpreter, root in [
            ("python", self.root),
            ("/usr/bin/python3", self.root / "wrong"),
        ]:
            with self.assertRaises(ValueError):
                bound.command(interpreter, root)
        for name in [
            "tools/atlas_host/example.py",
            "target/release/weight-atlas-rust",
            "run.py",
            "runtime_harness.py",
            "fixtures/tiny-bf16/tiny.safetensors",
            "reviewed-recipe.json",
        ]:
            with self.subTest(name=name):
                bound = self.bind()
                p = self.root / name
                raw = p.read_bytes()
                p.write_bytes(raw + b" changed")
                with self.assertRaises(ValueError):
                    bound.check()
                p.write_bytes(raw)

    def test_new_source_file_after_binding_refuses_instead_of_importing_unbound_code(
        self,
    ):
        bound = self.bind()
        (self.root / "tools/atlas_host/additional.py").write_text("# added source")
        with self.assertRaises(ValueError):
            bound.check()

    def test_recipe_name_peak_topology_and_identity_cannot_select_cheaper_policy(self):
        cases = [
            {"recipe": "unknown"},
            {"binary_sha256": "0" * 64},
            {"fixture_sha256": "0" * 64},
            {"topology": ["arbitrary command"]},
            {"measured_peak": 1},
            {"version": True},
            {"source_sha256": {}},
            {"monitor_sha256": "0" * 64},
            {"harness_sha256": "0" * 64},
        ]
        for change in cases:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.bind({**self.data, **change})
        with self.assertRaises(ValueError):
            self.bind(approved="0" * 64)
        for untrusted in [
            "tiny_bf16_provider_v1",
            {"recipe": policy.RECIPE},
            types.SimpleNamespace(start_bytes=0),
        ]:
            with self.assertRaises(ValueError):
                policy.check_start(untrusted, policy.DEFAULT_START, policy.DISK_RESERVE)

    def gates(self, bound, available, disk):
        boundary = policy.QualificationBoundary(bound) if bound is not None else None
        profiles = types.SimpleNamespace(start_threads=Mock())
        app = types.SimpleNamespace(
            launch_policy=bound,
            host=types.SimpleNamespace(cache_root=self.root),
            profiles=profiles,
        )
        hooks = LinuxProfileHooks(
            None, None, {"cache": self.root}, None, launch_policy=bound
        )
        supervisor = Supervisor()
        token = supervisor.acquire("profile", "context")
        grant = provider.Clock().grant()
        platform = ProfilePlatform(supervisor, token, grant, hooks)
        with (
            patch("atlas_host.hosted_runtime.require_platform"),
            patch("os.sched_setaffinity"),
            patch("os.sched_getaffinity", return_value={7}),
            patch("atlas_host.hosted_runtime.available_bytes", return_value=available),
            patch("atlas_host.profile_os.available_bytes", return_value=available),
            patch("shutil.disk_usage", return_value=types.SimpleNamespace(free=disk)),
        ):
            methods = [
                lambda: HostedApplication.start_threads(app),
                platform.start_gate,
            ]
            if boundary:
                methods.insert(0, lambda: boundary.preflight(available, disk))
            outcomes = []
            for method in methods:
                try:
                    method()
                    outcomes.append(True)
                except ValueError:
                    outcomes.append(False)
            self.assertEqual(profiles.start_threads.call_count, int(outcomes[-2]))
            return outcomes

    def test_all_three_actual_admission_gates_agree_at_threshold_and_one_byte_below(
        self,
    ):
        bound = self.bind()
        threshold = bound.start_bytes
        self.assertEqual(self.gates(bound, threshold, policy.DISK_RESERVE), [True] * 3)
        self.assertEqual(
            self.gates(bound, threshold - 1, policy.DISK_RESERVE), [False] * 3
        )
        self.assertEqual(
            self.gates(bound, threshold, policy.DISK_RESERVE - 1), [False] * 3
        )

    def test_default_internal_gates_remain_five_gib(self):
        self.assertEqual(
            self.gates(None, policy.DEFAULT_START, policy.DISK_RESERVE), [True] * 2
        )
        self.assertEqual(
            self.gates(None, policy.DEFAULT_START - 1, policy.DISK_RESERVE), [False] * 2
        )
        self.assertEqual(
            self.gates(None, 17 * policy.GIB // 4, policy.DISK_RESERVE), [False] * 2
        )

    def test_each_profile_gate_rereads_ram_disk_and_bound_identity(self):
        bound = self.bind()
        hooks = LinuxProfileHooks(
            None, None, {"cache": self.root}, None, launch_policy=bound
        )
        s = Supervisor()
        p = ProfilePlatform(
            s, s.acquire("profile", "context"), provider.Clock().grant(), hooks
        )
        with (
            patch(
                "atlas_host.profile_os.available_bytes",
                side_effect=[
                    bound.start_bytes,
                    bound.start_bytes - 1,
                    bound.start_bytes,
                    bound.start_bytes,
                ],
            ) as ram,
            patch(
                "shutil.disk_usage",
                side_effect=[
                    types.SimpleNamespace(free=x)
                    for x in [
                        policy.DISK_RESERVE,
                        policy.DISK_RESERVE,
                        policy.DISK_RESERVE - 1,
                        policy.DISK_RESERVE,
                    ]
                ],
            ) as disk,
        ):
            p.start_gate()
            with self.assertRaises(ValueError):
                p.start_gate()
            with self.assertRaises(ValueError):
                p.start_gate()
            (self.root / "src/lib.rs").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                p.start_gate()
            self.assertEqual((ram.call_count, disk.call_count), (4, 4))

    def test_app_wires_same_bound_policy_into_hooks_before_any_execution(self):
        bound = self.bind()
        binary = types.SimpleNamespace(path=Path(bound.binary_path), check=Mock())
        with (
            patch("atlas_host.hosted_runtime.require_platform"),
            patch("atlas_host.hosted_runtime.FrozenBinary", return_value=binary),
            patch(
                "atlas_host.hosted_runtime.FixtureHost",
                return_value=types.SimpleNamespace(),
            ),
        ):
            app = HostedApplication(
                None, None, binary.path, policy.BINARY_SHA, launch_policy=bound
            )
            hooks = app.profiles.hooks_factory({"cache": self.root}, None)
            self.assertIs(hooks.launch_policy, bound)
            self.assertIs(app.launch_policy, bound)
            self.assertEqual(app.profiles.watchdog.lifetime, app.lifetime.pulse)
            self.assertIsNone(app.profiles.thread)
            self.assertIsNone(app.profiles.watchdog.thread)
            (self.root / "src/lib.rs").write_bytes(b"changed source")
            with self.assertRaises(ValueError):
                HostedApplication(
                    None, None, binary.path, policy.BINARY_SHA, launch_policy=bound
                )

    def test_outer_always_charges_maximum_snapshot_reservation(self):
        boundary = policy.QualificationBoundary(self.bind())
        boundary.check_sample(observed(policy.TREE_CEILING - policy.SNAPSHOT_MAX))
        with self.assertRaises(ValueError):
            boundary.check_sample(
                observed(policy.TREE_CEILING - policy.SNAPSHOT_MAX + 1)
            )
        with self.assertRaises(ValueError):
            boundary.check_sample(observed(0, policy.STOP_RESERVE - 1))
        with self.assertRaises(ValueError):
            boundary.check_sample({**observed(), "all_owned_accounted": False})


class LifetimeTests(Pure):
    def make(self, charge=1024):
        s = Supervisor()
        store = types.SimpleNamespace(owned_storage_bytes=charge)
        owner = types.SimpleNamespace(job=types.SimpleNamespace(store=store))
        s.retain_profile(owner, charge)
        book = types.SimpleNamespace(
            resources=Mock(return_value=observed(policy.TREE_CEILING - 2 * charge)),
            request_stop_all=Mock(),
        )
        cleanup = Mock()
        guard = LifetimeGuard(book, s, cleanup)
        return guard, s, store, book, cleanup, owner

    def test_idle_snapshot_remains_charged_at_exact_and_one_byte_over_ceiling(self):
        guard, s, store, book, cleanup, _ = self.make()
        self.assertFalse(s.busy())
        self.assertEqual(s.charged_snapshot_bytes(), 2048)
        self.assertTrue(guard.pulse())
        cleanup.assert_not_called()
        book.resources.return_value = observed(policy.TREE_CEILING - 2048 + 1)
        self.assertFalse(guard.pulse())
        self.assertTrue(s.stopping)
        self.assertEqual(s.charged_snapshot_bytes(), 2048)
        cleanup.assert_called_once()
        book.request_stop_all.assert_called_once()

    def test_uncertain_storage_over_reservation_is_never_uncharged(self):
        guard, s, store, book, cleanup, owner = self.make()
        store.owned_storage_bytes = 4096
        self.assertEqual(s.charged_snapshot_bytes(), 4096)
        book.resources.return_value = observed(policy.TREE_CEILING - 4096)
        self.assertTrue(guard.pulse())
        with self.assertRaises(ValueError):
            s.close_profile(owner, all_handles_closed=False)
        self.assertEqual(s.charged_snapshot_bytes(), 4096)
        book.resources.return_value["rss_bytes"] += 1
        self.assertFalse(guard.pulse())
        self.assertIs(s.profile_session, owner)
        s.close_profile(owner, all_handles_closed=True)
        self.assertEqual(s.charged_snapshot_bytes(), 0)

    def test_failure_is_sticky_and_does_not_release_or_renew_original_token(self):
        guard, s, _, book, _, _ = self.make()
        token = s.acquire("profile", "c")
        grant = provider.Clock().grant()
        deadline = grant.deadline
        book.resources.return_value = observed(0, policy.STOP_RESERVE - 1)
        self.assertFalse(guard.pulse())
        self.assertIs(s.active, token)
        self.assertFalse(token.current())
        book.resources.return_value = observed()
        self.assertFalse(guard.pulse())
        self.assertEqual(grant.deadline, deadline)
        with self.assertRaises(ValueError):
            s.acquire("profile", "c")
        with self.assertRaises(ValueError):
            token.release()

    def test_observation_failures_refuse_and_stop_errors_keep_cleanup_retriable(self):
        for result in [
            ValueError("probe failed"),
            {**observed(), "all_owned_accounted": False},
            {**observed(), "descendants_clear": False},
            {**observed(), "rss_bytes": True},
        ]:
            with self.subTest(result=result):
                guard, s, _, book, cleanup, _ = self.make()
                if isinstance(result, Exception):
                    book.resources.side_effect = result
                else:
                    book.resources.return_value = result
                book.request_stop_all.side_effect = ValueError("signal uncertain")
                self.assertFalse(guard.pulse())
                self.assertFalse(guard.pulse())
                self.assertEqual(cleanup.call_count, 2)
                self.assertEqual(book.request_stop_all.call_count, 2)
                self.assertEqual(s.charged_snapshot_bytes(), 2048)

    def test_typed_bounded_spawn_pending_still_charges_memory_and_stop_reserve(self):
        guard, s, _, book, _, _ = self.make()
        data = {**observed(policy.TREE_CEILING - 2048), "all_owned_accounted": False}
        book.resources.side_effect = SpawnObservationPending(data)
        self.assertTrue(guard.pulse())
        data["rss_bytes"] += 1
        self.assertFalse(guard.pulse())

    def test_whole_root_renderer_worker_overlap_uses_one_aggregate(self):
        with patch("os.getpid", return_value=1):
            book = ProcessBook(
                stats=lambda pid: {
                    "start": pid,
                    "cpu": 0.0,
                    "rss": {1: 100, 2: 200, 3: 300}[pid],
                },
                descendants=lambda pid: {2, 3} if pid == 1 else set(),
                available=lambda: policy.STOP_RESERVE,
            )
        book.owned = [
            types.SimpleNamespace(
                pid=p,
                reaped=False,
                start=p,
                unexpected=False,
                lock=__import__("threading").RLock(),
                stop=Mock(),
            )
            for p in (2, 3)
        ]
        self.assertEqual(book.resources()["rss_bytes"], 600)
        s = Supervisor()
        guard = LifetimeGuard(book, s, Mock())
        self.assertTrue(guard.pulse())
        book.stats = lambda pid: {
            "start": pid,
            "cpu": 0.0,
            "rss": policy.TREE_CEILING // 3 + 1,
        }
        self.assertFalse(guard.pulse())

    def test_stop_attempts_all_owned_children_and_never_spawns_after_shutdown(self):
        book = ProcessBook()
        first = types.SimpleNamespace(
            stop=Mock(side_effect=ValueError("uncertain")), reaped=False
        )
        second = types.SimpleNamespace(stop=Mock(), reaped=False)
        book.owned = [first, second]
        with self.assertRaises(ValueError):
            book.request_stop_all()
        first.stop.assert_called_once()
        second.stop.assert_called_once()
        with self.assertRaisesRegex(ValueError, "shutdown"):
            book.spawn(["never execute"])
        self.assertEqual(book.owned, [first, second])

    def test_watchdog_samples_without_any_operation_slots_and_uses_timed_os_wait(self):
        meter = types.SimpleNamespace(register=Mock(), freeze=Mock())
        watch = WatchdogLoop(meter)
        callback = Mock()
        watch.set_lifetime(callback)
        watch.pulse()
        callback.assert_called_once()
        with self.assertRaises(ValueError):
            watch.set_lifetime(Mock())

        def wait(timeout):
            self.assertEqual(timeout, 0.05)
            watch.stopped = True

        with patch.object(watch.condition, "wait", side_effect=wait):
            watch.run()
        self.assertEqual(callback.call_count, 2)
        meter.freeze.assert_called_once()

    def test_idle_completed_result_is_retired_by_owner_after_lifetime_refusal(self):
        case = doubles.ServiceTests("test_second_start_never_queues")
        case.setUp()
        self.addCleanup(case.doCleanups)
        grant, _ = case.start()
        case.service.finish_admission(grant)
        case.service.step()
        case.finish()
        s = case.service.supervisor
        self.assertGreater(s.charged_snapshot_bytes(), 0)
        self.assertEqual(case.watch.slots, [])
        deadline = grant.deadline
        book = types.SimpleNamespace(
            resources=lambda: observed(policy.TREE_CEILING), request_stop_all=Mock()
        )
        guard = LifetimeGuard(book, s, case.service.request_shutdown)
        case.watch.set_lifetime(guard.pulse)
        case.watch.pulse()
        self.assertEqual(case.service.status(case.owner)["state"], "stopping")
        self.assertIsNone(case.service.status(case.owner)["accepted"])
        with self.assertRaises(ValueError):
            case.service.page({})
        with self.assertRaises(ValueError):
            case.service.heartbeat(case.owner)
        case.service.step()
        self.assertEqual(case.service.status(case.owner)["state"], "cancelled")
        self.assertEqual(s.charged_snapshot_bytes(), 0)
        self.assertFalse(case.os.files)
        self.assertEqual(grant.deadline, deadline)
        with self.assertRaises(ValueError):
            case.start()

    def test_service_cannot_stop_lifetime_watch_until_source_and_storage_close(self):
        s = Supervisor()
        watch = WatchdogLoop(types.SimpleNamespace())
        service = ProfileService(s, None, None, watchdog=watch)
        service.owner_cleanup = Mock(return_value=False)
        watch.close = Mock()
        self.assertFalse(service.close())
        self.assertFalse(service.stopped)
        watch.close.assert_not_called()
        service.owner_cleanup.return_value = True
        store = types.SimpleNamespace(owned_storage_bytes=10)
        owner = types.SimpleNamespace(job=types.SimpleNamespace(store=store))
        s.retain_profile(owner, 10)
        with self.assertRaises(ValueError):
            service.close()
        self.assertFalse(service.stopped)
        watch.close.assert_not_called()
        s.close_profile(owner, all_handles_closed=True)
        self.assertTrue(service.close())
        watch.close.assert_called_once()

    def test_lifetime_poison_blocks_native_publication_even_if_memory_recovers(self):
        case = doubles.NativeTests(
            "test_matching_native_ack_releases_and_partial_io_is_bounded"
        )
        channel = case.make()
        original = channel.stream.recv

        def recv(size):
            case.s.prevent_admission()
            return original(size)

        channel.stream.recv = recv
        with self.assertRaises(ValueError):
            channel.read("/api/model", "ctx")
        self.assertTrue(case.child.stopped)
        self.assertTrue(case.s.busy())
        case.child.reaped = True
        case.watch.pulse()
        self.assertFalse(case.s.busy())

    def test_host_tick_samples_before_idle_work_and_retains_shutdown_cleanup(self):
        app = types.SimpleNamespace(
            lifetime=types.SimpleNamespace(pulse=Mock(), stopping=True),
            book=types.SimpleNamespace(cleanup=Mock()),
            lock=__import__("threading").RLock(),
            host=types.SimpleNamespace(close=Mock(), tick=Mock()),
        )
        HostedApplication.tick(app)
        app.lifetime.pulse.assert_called_once()
        app.host.close.assert_called_once()
        app.host.tick.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
