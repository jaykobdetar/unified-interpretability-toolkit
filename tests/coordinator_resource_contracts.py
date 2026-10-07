"""Inert coordinator resource ownership and tiny-file identity contracts."""

from contextlib import ExitStack
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import live_inference as live


class ResourceContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.content = {
            "config.json": b"fixture-config",
            "model.safetensors": b"fixture-weights",
        }
        for name, raw in self.content.items():
            (self.root / name).write_bytes(raw)
        self.manifest = {
            "revision": "fixture-revision",
            "files": {
                name: hashlib.sha256(raw).hexdigest()
                for name, raw in self.content.items()
            },
        }
        patcher = mock.patch.object(live, "MANIFEST", self.manifest)
        patcher.start()
        self.addCleanup(patcher.stop)

    def outcome(self, function, *args, **kwargs):
        try:
            return function(*args, **kwargs), None
        except (ValueError, TypeError, OSError) as error:
            return None, (type(error), str(error))

    def doubles(self, budget, now=0.0):
        stack = ExitStack()
        self.addCleanup(stack.close)
        check = stack.enter_context(mock.patch.object(budget, "check"))
        stack.enter_context(mock.patch.object(live.time, "monotonic", return_value=now))
        timer_state = stack.enter_context(
            mock.patch.object(live.signal, "getitimer", return_value=(0.0, 0.0))
        )
        previous = object()
        stack.enter_context(
            mock.patch.object(live.signal, "getsignal", return_value=previous)
        )
        install = stack.enter_context(mock.patch.object(live.signal, "signal"))
        timer = stack.enter_context(mock.patch.object(live.signal, "setitimer"))
        return SimpleNamespace(
            stack=stack,
            check=check,
            timer_state=timer_state,
            previous=previous,
            install=install,
            timer=timer,
        )

    def test_process_alive(self):
        for result, expected in ((None, True), (0, False), (3, False)):
            child = SimpleNamespace(poll=mock.Mock(return_value=result))
            self.assertIs(live.process_alive(child), expected)
            child.poll.assert_called_once_with()
        child = SimpleNamespace(poll=mock.Mock(side_effect=OSError("inert poll")))
        self.assertIs(live.process_alive(child), True)

    def test_signal_and_reap(self):
        for terminate in (False, True):
            child = mock.Mock()
            with mock.patch.object(live, "process_alive", return_value=True) as alive:
                self.assertIs(
                    live.signal_and_reap(child, terminate=terminate, timeout=0.2), True
                )
            alive.assert_called_once_with(child)
            expected = [
                mock.call.terminate() if terminate else mock.call.kill(),
                mock.call.wait(timeout=0.2),
            ]
            self.assertEqual(child.mock_calls, expected)
        child = mock.Mock()
        child.wait.side_effect = subprocess.TimeoutExpired("inert-child", 0.2)
        with mock.patch.object(live, "process_alive", return_value=False):
            self.assertIs(live.signal_and_reap(child), False)
        child.kill.assert_not_called()
        child.terminate.assert_not_called()

    def test_budget_init(self):
        budget = live.SweepAdmissionBudget(12.5, 9.25)
        self.assertEqual((budget.deadline, budget.cpu_deadline), (12.5, 9.25))

    def test_budget_check(self):
        budget = live.SweepAdmissionBudget(2.0, 3.0)
        for wall, cpu, memory, expected in (
            (0.0, 0.0, 6 * live.GIB, None),
            (
                2.0,
                0.0,
                6 * live.GIB,
                "Total sweep wall budget exhausted during verification; no further work",
            ),
            (
                0.0,
                3.0,
                6 * live.GIB,
                "Total sweep CPU budget exhausted during verification; no further work",
            ),
            (
                0.0,
                0.0,
                3 * live.GIB,
                "Sweep verification stopped by available-memory reserve",
            ),
        ):
            with (
                mock.patch.object(live.time, "monotonic", return_value=wall),
                mock.patch.object(live.time, "process_time", return_value=cpu),
                mock.patch.object(live, "available", return_value=memory),
            ):
                self.assertEqual(
                    self.outcome(budget.check),
                    (None, None if expected is None else (ValueError, expected)),
                )

    def test_remaining_cpu(self):
        for deadline, expected in ((100.0, 90), (4.9, 4), (0.5, None)):
            budget = live.SweepAdmissionBudget(120.0, deadline)
            with (
                mock.patch.object(budget, "check") as check,
                mock.patch.object(live.time, "process_time", return_value=0.0),
            ):
                result, error = self.outcome(budget.remaining_cpu)
            check.assert_called_once_with()
            self.assertEqual(result, expected)
            self.assertEqual(
                error,
                (
                    None
                    if expected is not None
                    else (
                        ValueError,
                        "Less than one CPU second remains; no worker started",
                    )
                ),
            )

    def test_verification(self):
        budget = live.SweepAdmissionBudget(10.0, 90.0)
        doubles = self.doubles(budget)
        with doubles.stack:
            with budget.verification():
                self.assertEqual(doubles.check.call_count, 1)
            self.assertEqual(doubles.check.call_count, 2)
            self.assertEqual(
                doubles.timer.call_args_list,
                [
                    mock.call(live.signal.ITIMER_REAL, 0.1),
                    mock.call(live.signal.ITIMER_REAL, 0),
                ],
            )
            self.assertTrue(callable(doubles.install.call_args_list[0].args[1]))
            self.assertEqual(
                doubles.install.call_args_list[-1],
                mock.call(live.signal.SIGALRM, doubles.previous),
            )
            doubles.timer_state.return_value = (0.1, 0.0)
            with self.assertRaises(ValueError) as caught:
                with budget.verification():
                    self.fail("An owned timer must refuse entry")
            self.assertEqual(
                str(caught.exception),
                "An existing deadline timer prevents sweep verification",
            )

    def test_arm(self):
        for remaining, expected in ((0.0000001, 0.000001), (10.0, 0.1)):
            budget = live.SweepAdmissionBudget(remaining, 90.0)
            doubles = self.doubles(budget)
            with doubles.stack:
                with budget.verification():
                    self.assertEqual(
                        doubles.timer.call_args_list,
                        [mock.call(live.signal.ITIMER_REAL, expected)],
                    )
                self.assertEqual(
                    doubles.timer.call_args_list[-1],
                    mock.call(live.signal.ITIMER_REAL, 0),
                )

    def test_alarm(self):
        budget = live.SweepAdmissionBudget(10.0, 90.0)
        doubles = self.doubles(budget)
        events = []
        doubles.check.side_effect = lambda: events.append("check")
        doubles.timer.side_effect = lambda *args: events.append(("timer", *args))
        with doubles.stack:
            with budget.verification():
                alarm = doubles.install.call_args_list[0].args[1]
                events.clear()
                alarm(live.signal.SIGALRM, None)
                self.assertEqual(
                    events, ["check", ("timer", live.signal.ITIMER_REAL, 0.1)]
                )

    def test_verify_model(self):
        reads = []

        class Tracked:
            def __init__(self, path, mode):
                self.file = open(path, mode)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.file.close()

            def read(self, bound):
                reads.append(bound)
                return self.file.read(bound)

        with mock.patch.object(Path, "open", autospec=True, side_effect=Tracked):
            self.assertEqual(self.outcome(live.verify_model, self.root), (None, None))
        self.assertEqual(reads, [1024 * 1024] * 4)
        self.manifest["files"]["config.json"] = "0" * 64
        self.assertEqual(
            self.outcome(live.verify_model, self.root),
            (None, (ValueError, "Pinned file hash mismatch: config.json")),
        )

    def test_checkpoint(self):
        check = mock.Mock()
        self.assertIsNone(live.verify_model(self.root, check=check))
        # Two files, each with one data read and one EOF read: seven callbacks each.
        self.assertEqual(check.call_args_list, [mock.call()] * 14)

    def test_layout_fingerprints(self):
        expected = {}
        for name in self.content:
            value = (self.root / name).stat()
            expected[name] = (
                value.st_dev,
                value.st_ino,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )
        self.assertEqual(
            live.layout_fingerprints(self.root),
            {"directory": str(self.root), "files": expected},
        )
        (self.root / "model.safetensors").unlink()
        self.assertIsNone(live.layout_fingerprints(self.root))

    def test_verified_receipt(self):
        before = live.layout_fingerprints(self.root)
        incoming = io.BytesIO(self.content["config.json"])
        with (
            mock.patch.object(
                live, "layout_fingerprints", return_value=before
            ) as fingerprints,
            mock.patch.object(live, "verify_model") as verify,
            mock.patch.object(Path, "open", return_value=incoming),
            mock.patch.object(incoming, "read", wraps=incoming.read) as read,
        ):
            result = live.verified_layout_receipt(self.root, required=False)
        self.assertIs(result, before)
        self.assertEqual(
            fingerprints.call_args_list, [mock.call(self.root), mock.call(self.root)]
        )
        verify.assert_called_once()
        self.assertEqual(verify.call_args.args, (self.root,))
        self.assertTrue(callable(verify.call_args.kwargs["check"]))
        read.assert_called_once_with(65537)
        with (
            mock.patch.object(
                live, "layout_fingerprints", side_effect=[before, {"changed": True}]
            ),
            mock.patch.object(live, "verify_model") as verify,
        ):
            self.assertIsNone(live.verified_layout_receipt(self.root, required=True))
        verify.assert_called_once_with(self.root, check=None)

    def test_receipt_check(self):
        before = live.layout_fingerprints(self.root)
        with (
            mock.patch.object(live, "layout_fingerprints", return_value=before),
            mock.patch.object(live, "verify_model") as verify,
            mock.patch.object(
                Path, "open", return_value=io.BytesIO(self.content["config.json"])
            ),
            mock.patch.object(live.time, "monotonic", return_value=10.0) as clock,
            mock.patch.object(
                live, "available", return_value=6 * live.GIB
            ) as available,
        ):
            self.assertIs(
                live.verified_layout_receipt(self.root, required=False), before
            )
            check = verify.call_args.kwargs["check"]
            clock.return_value = 14.5
            self.assertEqual(self.outcome(check), (None, None))
            clock.return_value = 10.0
            available.return_value = 13 * live.GIB // 4
            self.assertEqual(self.outcome(check), (None, None))
            clock.return_value = 15.0
            self.assertEqual(
                self.outcome(check),
                (
                    None,
                    (ValueError, "Configuration receipt verification budget reached"),
                ),
            )

    def test_current_binding(self):
        source = "a" * 64
        identity = hashlib.sha256(
            (f'["weight-atlas-model-v1","{source}","fixture-revision"]').encode()
        ).hexdigest()
        receipt = live.layout_fingerprints(self.root)
        marker = {"fixture": True}
        session = SimpleNamespace(
            model=self.root, head_layout_receipt=receipt, head_layout_binding=marker
        )
        model = {
            "source_directory": str(self.root),
            "source_identity": source,
            "revision": "fixture-revision",
            "model_identity": identity,
        }
        self.assertEqual(
            live.current_layout_binding(model, session),
            {
                "source_identity": source,
                "model_identity": identity,
                "weights_sha256": self.manifest["files"]["model.safetensors"],
                "config_sha256": self.manifest["files"]["config.json"],
            },
        )
        self.assertIsNone(
            live.current_layout_binding({**model, "revision": "changed"}, session)
        )
        session.head_layout_receipt = None
        self.assertIs(live.current_layout_binding(model, session), marker)


if __name__ == "__main__":
    unittest.main()
