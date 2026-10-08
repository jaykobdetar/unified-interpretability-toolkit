"""Worker and coordinator edge checks with stopped model access and fake clocks."""

from contextlib import nullcontext
import io
import json
import math
from pathlib import Path
import resource
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
import inference_worker as worker
import inference_sweep as sweep
import inference_edits as edits
import live_inference as live
import prompt_pair_contracts
import review_limit_contracts
import sweep_contracts
from analytics import svd


class ReachedValidation(Exception):
    pass


class WorkerLimits(unittest.TestCase):
    def generate(self, count=1, limit=1, layer=0):
        tokenizer = SimpleNamespace(encode=lambda *_a, **_k: [1] * max(0, count))
        worker.generate(
            None,
            tokenizer,
            None,
            "fixture",
            limit,
            layer,
            record=lambda _event: (_ for _ in ()).throw(ReachedValidation()),
        )

    def test_worker_prompt_generation_and_layer_edges_before_model_access(self):
        self.assertEqual((worker.MAX_PROMPT, worker.MAX_NEW), (128, 32))
        for field, low, high in (("count", 1, 128), ("limit", 1, 32), ("layer", 0, 29)):
            for value, accepted in (
                (low - 1, False),
                (low, True),
                (high, True),
                (high + 1, False),
            ):
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ReachedValidation if accepted else ValueError),
                ):
                    self.generate(**{field: value})

    def test_generation_prefill_record_precedes_engine_access(self) -> None:
        ids = [11, 22, 33]
        tokenizer = SimpleNamespace(encode=lambda *_a, **_k: ids)
        for site in ("block", "attention", "mlp"):
            events: list[dict[str, object]] = []

            def stop(event: dict[str, object]) -> None:
                events.append(event)
                raise ReachedValidation()

            with self.assertRaises(ReachedValidation):
                worker.generate(
                    None,
                    tokenizer,
                    None,
                    "fixture",
                    2,
                    7,
                    record=stop,
                    activation_site=site,
                )
            self.assertEqual(
                events,
                [
                    {
                        "type": "prefill",
                        "prompt_tokens": 3,
                        "prompt_ids": [11, 22, 33],
                        "layer": 7,
                        "activation_site": site,
                        "seed": 0,
                        "sampling": "greedy",
                        "dtype": "float32",
                    }
                ],
            )

    def test_worker_cpu_argument_and_inherited_os_limit_caps(self):
        for cpu, accepted in ((0, False), (1, True), (90, True), (91, False)):
            for inherited in ((resource.RLIM_INFINITY, resource.RLIM_INFINITY), (1, 2)):
                with (
                    self.subTest(cpu=cpu, inherited=inherited),
                    patch.object(worker.resource, "getrlimit", return_value=inherited),
                    patch.object(worker.resource, "setrlimit") as apply,
                ):
                    if accepted:
                        worker.configure_worker_limits(cpu)
                        expected = (
                            [
                                (resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3)),
                                (resource.RLIMIT_CPU, (cpu, cpu)),
                            ]
                            if inherited[0] == resource.RLIM_INFINITY
                            else [
                                (resource.RLIMIT_AS, (1, 1)),
                                (resource.RLIMIT_CPU, (1, 1)),
                            ]
                        )
                        self.assertEqual(
                            [call.args for call in apply.call_args_list], expected
                        )
                    else:
                        with self.assertRaises(ValueError):
                            worker.configure_worker_limits(cpu)
                        apply.assert_not_called()

    def test_sweep_remaining_cpu_rounding_and_wall_memory_cutoffs(self):
        for remaining, expected in (
            (math.nextafter(1.0, 0.0), None),
            (1.0, 1),
            (90.0, 90),
            (91.0, 90),
        ):
            # Keep the injected CPU clock nonnegative while testing the clamp.
            budget = live.SweepAdmissionBudget(120, remaining)
            with (
                patch.object(live.time, "monotonic", return_value=0),
                patch.object(live.time, "process_time", return_value=0),
                patch.object(live, "available", return_value=13 * live.GIB // 4),
            ):
                if expected is None:
                    with self.assertRaisesRegex(ValueError, "one CPU second"):
                        budget.remaining_cpu()
                else:
                    self.assertEqual(budget.remaining_cpu(), expected)
        for wall, cpu, memory, accepted in (
            (math.nextafter(120.0, 0.0), 0, 13 * live.GIB // 4, True),
            (120, 0, 13 * live.GIB // 4, False),
            (0, math.nextafter(90.0, 0.0), 13 * live.GIB // 4, True),
            (0, 90, 13 * live.GIB // 4, False),
            (0, 0, 13 * live.GIB // 4 - 1, False),
        ):
            with (
                patch.object(live.time, "monotonic", return_value=wall),
                patch.object(live.time, "process_time", return_value=cpu),
                patch.object(live, "available", return_value=memory),
            ):
                budget = live.SweepAdmissionBudget(120, 90)
                if accepted:
                    budget.check()
                else:
                    with self.assertRaises(ValueError):
                        budget.check()

    def test_sweep_timer_ownership_and_arm_clamps_without_signals(self):
        for remaining, expected in ((1e-7, 1e-6), (0.1, 0.1), (0.100001, 0.1)):
            budget = live.SweepAdmissionBudget(remaining, 90)
            with (
                patch.object(live.time, "monotonic", return_value=0),
                patch.object(live.time, "process_time", return_value=0),
                patch.object(live, "available", return_value=6 * live.GIB),
                patch.object(live.signal, "getitimer", return_value=(0.0, 0.0)),
                patch.object(live.signal, "getsignal", return_value=object()),
                patch.object(live.signal, "signal"),
                patch.object(live.signal, "setitimer") as timer,
            ):
                with budget.verification():
                    pass
                self.assertEqual(
                    timer.call_args_list[0].args, (live.signal.ITIMER_REAL, expected)
                )
                self.assertEqual(
                    timer.call_args_list[-1].args, (live.signal.ITIMER_REAL, 0)
                )
        budget = live.SweepAdmissionBudget(120, 90)
        with (
            patch.object(budget, "check"),
            patch.object(live.signal, "getitimer", return_value=(0.1, 0)),
            patch.object(live.signal, "setitimer") as timer,
        ):
            with self.assertRaisesRegex(ValueError, "existing deadline timer"):
                with budget.verification():
                    pass
            timer.assert_not_called()

    def test_backend_deadline_tick_and_available_memory_edges(self):
        self.assertEqual((live.UPSTREAM_DEADLINE, live.IO_TICK), (5, 0.1))
        for now, accepted in ((math.nextafter(5.0, 0.0), True), (5.0, False)):
            clock = Mock(return_value=0.0)
            session = SimpleNamespace(tick=Mock())
            deadline = live.OperationDeadline(session, clock)
            clock.return_value = now
            with patch.object(live, "available", return_value=13 * live.GIB // 4):
                if accepted:
                    self.assertGreater(deadline.remaining(), 0)
                else:
                    with self.assertRaises(live.BackendError) as error:
                        deadline.remaining()
                    self.assertEqual(
                        (error.exception.status, error.exception.code),
                        (504, "backend_timeout"),
                    )
        for now, due in ((math.nextafter(0.1, 0.0), False), (0.1, True)):
            clock = Mock(return_value=0.0)
            session = SimpleNamespace(tick=Mock())
            deadline = live.OperationDeadline(session, clock)
            deadline.last_tick = 0.0
            clock.return_value = now
            with patch.object(live, "available", return_value=13 * live.GIB // 4):
                deadline.remaining()
            self.assertEqual(session.tick.call_count, int(due))
        deadline = live.OperationDeadline(
            SimpleNamespace(tick=lambda: None), lambda: 0.0
        )
        with patch.object(live, "available", return_value=13 * live.GIB // 4 - 1):
            with self.assertRaises(live.BackendError) as error:
                deadline.remaining()
        self.assertEqual(
            (error.exception.status, error.exception.code), (503, "resource_limit")
        )

    def test_worker_main_argument_count_and_total_sweep_deadline(self):
        data, _ = sweep_contracts.admitted()
        raw = json.dumps(data).encode()
        for deadline, accepted in (
            (0.0, False),
            (math.nextafter(0.0, math.inf), True),
            (120.0, True),
            (math.nextafter(120.0, math.inf), False),
            (math.inf, False),
        ):
            with (
                patch.object(
                    worker.sys, "argv", ["worker", "/unused", str(deadline), "90"]
                ),
                patch.object(
                    worker.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(raw))
                ),
                patch.object(worker.os, "sched_getaffinity", return_value={0}),
                patch.object(worker.os, "sched_setaffinity"),
                patch.object(worker.os, "nice"),
                patch.object(worker, "configure_worker_limits"),
                patch.object(worker.time, "monotonic", return_value=0),
                patch.object(
                    live.SweepAdmissionBudget,
                    "verification",
                    return_value=nullcontext(),
                ),
                patch.object(
                    live, "verify_model", side_effect=ReachedValidation
                ) as verify,
                patch.object(
                    worker,
                    "load_engine",
                    side_effect=AssertionError("No model loading"),
                ),
            ):
                with self.assertRaises(ReachedValidation if accepted else ValueError):
                    worker.main()
                self.assertEqual(verify.call_count, int(accepted))
        for count in (0, 1, 3, 5):
            with (
                patch.object(worker.sys, "argv", ["worker"] * count),
                patch.object(worker.os, "sched_getaffinity", return_value={0}),
                patch.object(worker.os, "sched_setaffinity"),
                patch.object(worker.os, "nice"),
                patch.object(live, "verify_model") as verify,
            ):
                with self.assertRaisesRegex(ValueError, "budget arguments"):
                    worker.main()
                verify.assert_not_called()

    def test_sweep_tokenized_prompt_edges_before_engine_access(self):
        data, _ = sweep_contracts.admitted()
        for count, accepted in ((0, False), (1, True), (128, True), (129, False)):
            tokenizer = SimpleNamespace(
                encode=lambda *_a, **_kw: SimpleNamespace(ids=[1] * count)
            )
            with (
                patch.object(worker.sys, "argv", ["worker", "/unused", "120", "90"]),
                patch.object(
                    worker.sys,
                    "stdin",
                    SimpleNamespace(buffer=io.BytesIO(json.dumps(data).encode())),
                ),
                patch.object(worker.os, "sched_getaffinity", return_value={0}),
                patch.object(worker.os, "sched_setaffinity"),
                patch.object(worker.os, "nice"),
                patch.object(worker, "configure_worker_limits"),
                patch.object(worker.time, "monotonic", return_value=0),
                patch.object(
                    live.SweepAdmissionBudget,
                    "verification",
                    return_value=nullcontext(),
                ),
                patch.object(live, "verify_model"),
                patch.dict(
                    sys.modules,
                    {
                        "tokenizers": SimpleNamespace(
                            Tokenizer=SimpleNamespace(from_file=lambda _path: tokenizer)
                        )
                    },
                ),
                patch.object(
                    worker, "load_engine", side_effect=ReachedValidation
                ) as load,
            ):
                with self.assertRaises(ReachedValidation if accepted else ValueError):
                    worker.main()
                self.assertEqual(load.call_count, int(accepted))

    def test_worker_preview_bounded_read_preserves_actual_8193_byte_acceptance(self):
        # The coordinator enforces 8192 bytes. The worker independently reads
        # at most 8193; it currently has no separate len(raw)>8192 refusal.
        # Preserve that actual behaviour rather than invent a missing check.
        data = {
            "mode": "prompt_pair_preview",
            "source_model": prompt_pair_contracts.SOURCE_MODEL,
            "prompts": ["A", "B"],
        }
        raw = json.dumps(data).encode()
        for size in (8192, 8193):
            stream = Mock(wraps=io.BytesIO(raw + b" " * (size - len(raw))))
            events = []
            with (
                patch.object(worker.sys, "argv", ["worker", "/unused"]),
                patch.object(worker.sys, "stdin", SimpleNamespace(buffer=stream)),
                patch.object(worker.os, "sched_getaffinity", return_value={0}),
                patch.object(worker.os, "sched_setaffinity"),
                patch.object(worker.os, "nice"),
                patch.object(worker, "configure_worker_limits"),
                patch.object(live, "verify_model"),
                patch.object(
                    worker,
                    "load_engine",
                    side_effect=AssertionError("Preview cannot load model"),
                ),
                patch.object(worker, "emit", side_effect=events.append),
                patch.dict(
                    sys.modules,
                    {
                        "tokenizers": SimpleNamespace(
                            Tokenizer=SimpleNamespace(
                                from_file=lambda _path: prompt_pair_contracts.Tokenizer()
                            )
                        )
                    },
                ),
            ):
                worker.main()
            stream.readline.assert_called_once_with(8193)
            self.assertEqual(events[0]["type"], "preview_done")
            self.assertFalse(events[0]["preview"]["model_loaded"])

    def test_worker_loaded_dispatch_preserves_execution_arguments(self) -> None:
        plain = {
            "prompt": "fixture",
            "max_new_tokens": 2,
            "layer": 7,
            "activation_site": "attention",
            "observation": {"kind": "attention", "head": 2},
        }
        sweep_request, sweep_plan = sweep_contracts.admitted()
        pair_request = prompt_pair_contracts.request()
        cases = [
            (plain, "generation"),
            ({**plain, "mode": "future-mode"}, "generation"),
            ({**plain, "mode": []}, "generation"),
            ({**plain, "mode": {}}, "generation"),
            ({**plain, "mode": "future-mode", "edits": []}, "comparison"),
            (pair_request, "prompt_pair"),
            (sweep_request, "sweep"),
        ]
        torch = SimpleNamespace(__version__="fixture-torch")
        tokenizer = object()
        model = SimpleNamespace(
            _atlas_verified_layout={"sentinel": "layout"},
            config=SimpleNamespace(_attn_implementation="eager"),
        )
        for data, kind in cases:
            calls = {
                name: Mock()
                for name in ("generation", "comparison", "prompt_pair", "sweep")
            }
            events: list[dict[str, object]] = []
            record = Mock(side_effect=events.append)
            preview_tokenizer = (
                prompt_pair_contracts.Tokenizer()
                if kind == "prompt_pair"
                else SimpleNamespace(encode=lambda *_a, **_k: SimpleNamespace(ids=[1]))
            )
            arguments = ["worker", "/unused"] + (
                ["120", "90"] if kind == "sweep" else []
            )
            with (
                self.subTest(kind=kind, mode=data.get("mode")),
                patch.object(worker.sys, "argv", arguments),
                patch.object(
                    worker.sys,
                    "stdin",
                    SimpleNamespace(buffer=io.BytesIO(json.dumps(data).encode())),
                ),
                patch.object(worker.os, "sched_getaffinity", return_value={0}),
                patch.object(worker.os, "sched_setaffinity"),
                patch.object(worker.os, "nice"),
                patch.object(worker, "configure_worker_limits"),
                patch.object(worker.time, "monotonic", return_value=0),
                patch.object(
                    live.SweepAdmissionBudget,
                    "verification",
                    return_value=nullcontext(),
                ),
                patch.object(live, "verify_model"),
                patch.object(
                    worker, "load_engine", return_value=(torch, tokenizer, model)
                ) as load,
                patch.object(worker, "emit", record),
                patch.object(worker, "generate", calls["generation"]),
                patch.object(worker, "compare", calls["comparison"]),
                patch.object(prompt_pair_contracts.pair, "run", calls["prompt_pair"]),
                patch.object(sweep, "run", calls["sweep"]),
                patch.object(edits, "verified_parameters") as verify_parameters,
                patch.dict(
                    sys.modules,
                    {
                        "transformers": SimpleNamespace(
                            __version__="fixture-transformers"
                        ),
                        "tokenizers": SimpleNamespace(
                            __version__="fixture-tokenizers",
                            Tokenizer=SimpleNamespace(
                                from_file=lambda _path: preview_tokenizer
                            ),
                        ),
                        "safetensors": SimpleNamespace(
                            __version__="fixture-safetensors"
                        ),
                    },
                ),
            ):
                worker.main()
            load.assert_called_once_with(Path("/unused"))
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["type"], "loaded")
            self.assertEqual(
                events[0]["runtime"]["sampling"],
                "none" if kind in ("prompt_pair", "sweep") else "greedy",
            )
            self.assertEqual(sum(call.call_count for call in calls.values()), 1)
            if kind == "generation":
                calls[kind].assert_called_once_with(
                    torch,
                    tokenizer,
                    model,
                    "fixture",
                    2,
                    7,
                    activation_site="attention",
                    observation={"kind": "attention", "head": 2},
                )
            elif kind == "comparison":
                calls[kind].assert_called_once_with(torch, tokenizer, model, data)
            elif kind == "prompt_pair":
                verify_parameters.assert_called_once_with(model)
                preview = prompt_pair_contracts.pair.token_preview(
                    prompt_pair_contracts.Tokenizer(), data["prompts"]
                )
                calls[kind].assert_called_once_with(
                    torch,
                    tokenizer,
                    model,
                    data,
                    preview,
                    worker.CAPTURE_SITES,
                    record,
                )
            else:
                calls[kind].assert_called_once_with(
                    torch,
                    tokenizer,
                    model,
                    data,
                    sweep_plan,
                    120.0,
                    calls["generation"],
                    record,
                    cpu_deadline=90,
                )
            if kind != "prompt_pair":
                verify_parameters.assert_not_called()

    def test_coordinator_trace_cap_and_bounded_drain(self):
        fixture = review_limit_contracts.ReviewLimits(methodName="runTest")
        for count, accepted in ((32, True), (33, False)):
            events = [
                {"type": "step", "index": i, "activation": [0] * 576}
                for i in range(count)
            ]
            chunks = [(json.dumps(event) + "\n").encode() for event in events]
            with fixture.simulated_worker(chunks=chunks) as (session, process, reap):
                for _ in range(5):
                    session.tick()
                self.assertEqual(len(session.steps), 32)
                self.assertEqual(session.process is process, accepted)
                self.assertEqual(session.status, "running" if accepted else "error")
                self.assertEqual(reap.call_count, int(not accepted))
        # A newline in each valid event keeps the partial-output cap separate.
        chunks = [(json.dumps({"type": "prefill"}) + "\n").encode()] * 9
        with (
            fixture.simulated_worker(chunks=chunks) as (session, _process, _reap),
            patch.object(live.os, "read", side_effect=chunks) as read,
        ):
            session.tick()
            self.assertEqual(read.call_count, 8)
            self.assertTrue(
                all(call.args == (42, 65536) for call in read.call_args_list)
            )

    def test_sweep_selected_cell_cap_independent_of_planner_shape(self):
        # selected_indices is a trusted edit helper; a legal current head selects
        # 64*576 cells. A synthetic trusted rectangle isolates its 65536 cap.
        for count, accepted in ((65536, True), (65537, False)):
            edit = {
                "shape": [1, count],
                "tensor": "synthetic",
                "kind": "rows",
                "start": 0,
                "end": 1,
            }
            if accepted:
                self.assertEqual(
                    len(sweep.selected_indices([edit])["synthetic"]), count
                )
            else:
                with self.assertRaisesRegex(ValueError, "selected-cell cap"):
                    sweep.selected_indices([edit])

    def test_sweep_control_draw_limit_and_coverage_edges(self):
        # Force rejection words at the deterministic rejection-sampler boundary.
        target = {"kind": "offset", "layer": 0, "heads": [0], "offset": 0}
        rejected = SimpleNamespace(digest=lambda: b"\xff" * 32)
        accepted = SimpleNamespace(digest=lambda: b"\0" * 32)
        with patch.object(
            sweep.hashlib, "sha256", side_effect=[rejected] * 127 + [accepted]
        ) as draw:
            self.assertEqual(sweep._choice([0, 1, 2], 0, target, 0), 0)
            self.assertEqual(draw.call_count, 128)
        with patch.object(sweep.hashlib, "sha256", return_value=rejected) as draw:
            with self.assertRaisesRegex(ValueError, "draw limit"):
                sweep._choice([0, 1, 2], 0, target, 0)
            self.assertEqual(draw.call_count, 128)
        _, plan = sweep_contracts.admitted()
        for completed, accepted in ((-1, False), (0, True), (3, True), (4, False)):
            if accepted:
                coverage = sweep.coverage(plan, completed)
                self.assertEqual(len(coverage["completed_ids"]), completed)
            else:
                with self.assertRaises(ValueError):
                    sweep.coverage(plan, completed)


class OptionalSVDWorkerLimits(unittest.TestCase):
    def test_admission_axes_derived_values_memory_disk_and_timeout(self):
        self.assertEqual(
            (svd.MAX_SVD_AXIS, svd.MAX_SVD_VALUES, svd.TIMEOUT_SECONDS), (64, 4096, 5)
        )
        for axis in ("rows", "cols"):
            for number, accepted in ((1, True), (64, True), (65, False)):
                region = {"row": 0, "col": 0, "rows": 1, "cols": 1, axis: number}
                with (
                    patch.object(
                        Path,
                        "read_text",
                        return_value=f"MemAvailable: {15 * 1024**3 // 4 // 1024} kB\n",
                    ),
                    patch.object(
                        svd.shutil,
                        "disk_usage",
                        return_value=SimpleNamespace(free=25 * 1024**3),
                    ),
                    patch.object(
                        svd.subprocess,
                        "run",
                        return_value=SimpleNamespace(returncode=0, stdout="{}"),
                    ) as spawn,
                ):
                    response = svd.run(
                        [0] * number, [region["rows"], region["cols"]], region
                    )
                    self.assertEqual(response["available"], accepted)
                    self.assertEqual(spawn.call_count, int(accepted))
                    if accepted:
                        self.assertEqual(spawn.call_args.kwargs["timeout"], 5)
        for memory, disk, accepted in (
            (15 * 1024**3 // 4, 25 * 1024**3, True),
            (15 * 1024**3 // 4 - 1024, 25 * 1024**3, False),
            (15 * 1024**3 // 4, 25 * 1024**3 - 1, False),
        ):
            with (
                patch.object(
                    Path,
                    "read_text",
                    return_value=f"MemAvailable: {memory // 1024} kB\n",
                ),
                patch.object(
                    svd.shutil, "disk_usage", return_value=SimpleNamespace(free=disk)
                ),
                patch.object(
                    svd.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=0, stdout="{}"),
                ) as spawn,
            ):
                response = svd.run([0], [1], {"row": 0, "col": 0, "rows": 1, "cols": 1})
                self.assertEqual(response["available"], accepted)
                self.assertEqual(spawn.call_count, int(accepted))

    def invoke_worker(self, raw):
        with (
            patch.object(svd.os, "sched_getaffinity", return_value={0}),
            patch.object(svd.os, "sched_setaffinity"),
            patch.object(svd.os, "nice"),
            patch.object(
                svd.resource,
                "getrlimit",
                return_value=(resource.RLIM_INFINITY, resource.RLIM_INFINITY),
            ),
            patch.object(svd.resource, "setrlimit") as limits,
            patch.object(svd.sys, "stdin", io.StringIO(raw)),
            patch.object(svd.sys, "stdout", io.StringIO()),
            patch.dict(sys.modules, {"numpy": object()}),
            patch.object(svd, "compute_with_numpy", return_value={}) as compute,
        ):
            svd.worker()
            self.assertEqual(
                [call.args for call in limits.call_args_list],
                [
                    (resource.RLIMIT_AS, (768 * 1024**2, 768 * 1024**2)),
                    (resource.RLIMIT_CPU, (4, 4)),
                ],
            )
            self.assertEqual(compute.call_count, 1)

    def test_worker_real_input_character_count_and_dimensions(self):
        base = {"rows": 1, "cols": 1, "values": [0], "seed": 1}
        raw = json.dumps(base)
        self.invoke_worker(raw + " " * (200000 - len(raw)))
        with self.assertRaisesRegex(ValueError, "input cap"):
            self.invoke_worker(raw + " " * (200001 - len(raw)))
        for axis in ("rows", "cols"):
            for number, accepted in ((0, False), (1, True), (64, True), (65, False)):
                data = {**base, axis: number, "values": [0] * number}
                if accepted:
                    self.invoke_worker(json.dumps(data))
                else:
                    with self.assertRaises(ValueError):
                        self.invoke_worker(json.dumps(data))
        # 64 squared reaches the derived value cap; larger counts violate an axis.
        self.invoke_worker(
            json.dumps({**base, "rows": 64, "cols": 64, "values": [0] * 4096})
        )


if __name__ == "__main__":
    unittest.main()
