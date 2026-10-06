"""Inert runtime packets, bounded helpers and exact selected-coordinate controls."""

from contextlib import nullcontext
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import inference_prompt_pair as pair
import inference_sweep as sweep
import prompt_pair_contracts as pair_tests
import sweep_contracts as sweep_tests


class RuntimeContracts(unittest.TestCase):
    def test_integer(self) -> None:
        for value, expected in (
            (2, True),
            (7, True),
            (1, False),
            (8, False),
            (True, False),
            (2.0, False),
            (None, False),
        ):
            self.assertIs(sweep._int(value, 2, 7), expected)
        self.assertIs(sweep._int(True, 0, 2), False)

    def test_canonical(self) -> None:
        self.assertEqual(
            sweep._canonical({"z": [1, "é"], "a": 2}), '{"a":2,"z":[1,"é"]}'
        )
        with self.assertRaises(ValueError):
            sweep._canonical({"a": math.inf})

    def test_space(self) -> None:
        self.assertEqual(sweep._space({"kind": "head", "layer": 7}), (7, "output"))
        self.assertEqual(sweep._space({"kind": "offset", "layer": 3}), (3, "query"))

    def test_choice(self) -> None:
        pool = [
            {"kind": "offset", "layer": 0, "heads": [1], "offset": i} for i in range(11)
        ]
        target = {"kind": "head", "layer": 0, "head": 0}
        with patch.object(
            sweep.hashlib,
            "sha256",
            return_value=SimpleNamespace(digest=lambda: b"\0\0\0\7" + bytes(28)),
        ) as hashed:
            self.assertIs(sweep._choice(pool, 7, target, 2), pool[7])
        hashed.assert_called_once_with(
            b'["weight-atlas-sweep-control-v2",7,{"head":0,"kind":"head","layer":0},2,0]'
        )

    def test_indices(self) -> None:
        edits = [
            {"tensor": "held", "shape": [2, 3], "kind": "element", "row": 1, "col": 2},
            {"tensor": "held", "shape": [2, 3], "kind": "rows", "start": 0, "end": 1},
            {
                "tensor": "held",
                "shape": [2, 3],
                "kind": "columns",
                "start": 1,
                "end": 2,
            },
        ]
        self.assertEqual(sweep.selected_indices(edits), {"held": [0, 1, 2, 4, 5]})

    def test_undo(self) -> None:
        parameter = sweep_tests.Parameter()
        model = {sweep_tests.NAME: parameter}
        before = [sweep_tests.bits(parameter.get(i)) for i in range(576)]
        with (
            patch.object(sweep, "verified_parameters", return_value=model),
            patch.object(sweep, "apply_edits", side_effect=sweep_tests.apply),
        ):
            with sweep.temporary_edits(
                sweep_tests.Torch, model, [sweep_tests.row()], sweep_tests.SOURCE_MODEL
            ) as changed:
                self.assertEqual(
                    changed,
                    {
                        "selected_cells": 576,
                        "changed_cells": 575,
                        "parameter_delta_l2": math.sqrt(1.25**2 + 574 * 0.25**2),
                    },
                )
        self.assertEqual(
            [sweep_tests.bits(parameter.get(i)) for i in range(576)], before
        )

    def plan(self) -> dict[str, Any]:
        return {
            "cases": [{"id": "target-1"}, {"id": "matched_control-1"}],
            "prompt_count": 2,
            "targets": [{"kind": "head", "head": 0}],
        }

    def test_record_ids(self) -> None:
        self.assertEqual(
            sweep.record_ids(self.plan()),
            [
                "target-1/prompt-1",
                "target-1/prompt-2",
                "matched_control-1/prompt-1",
                "matched_control-1/prompt-2",
            ],
        )

    def test_coverage(self) -> None:
        plan = self.plan()
        for count in range(5):
            value = sweep.coverage(plan, count)
            ids = [
                "target-1/prompt-1",
                "target-1/prompt-2",
                "matched_control-1/prompt-1",
                "matched_control-1/prompt-2",
            ]
            self.assertEqual(value["completed_ids"], ids[:count])
            self.assertEqual(value["unrun_ids"], ids[count:])
            self.assertIs(value["complete"], count == 4)

    def test_metrics(self) -> None:
        sweep_tests.Contracts().test_independent_small_vocabulary_metrics()
        value = math.log(3.0)
        result = sweep.metrics(
            sweep_tests.Torch,
            sweep_tests.Vec([0.0, value]),
            sweep_tests.Vec([value, 0.0]),
        )
        self.assertEqual(result["logit_delta_max_abs"], value)

    def sweep_scenario(
        self,
        *,
        deadline: float = 120,
        clock: float = 0,
        cpu_deadline: float | None = None,
        cpu_clock: float = 0,
    ) -> tuple[list[dict[str, Any]], Mock]:
        data, plan = sweep_tests.admitted()
        model = {sweep_tests.NAME: sweep_tests.Parameter()}
        events: list[dict[str, Any]] = []
        tokenizer = SimpleNamespace(decode=lambda ids, **kw: str(ids[0]))

        def generate(
            torch: Any,
            tok: Any,
            held: Any,
            prompt: str,
            limit: int,
            layer: int,
            record: Any,
            scores: Any,
            *,
            activation_site: str,
        ) -> None:
            v = held[sweep_tests.NAME].get(1)
            scores(sweep_tests.Vec([v, 0.0]))
            record(
                {
                    "type": "step",
                    "activation": [v] * 576,
                    "layer": layer,
                    "activation_site": activation_site,
                    "activation_kind": "fixture",
                    "position": 0,
                    "input_token_id": 1,
                    "compute_ms": 1.0,
                    "top_logits": [{"id": 0, "value": v}, {"id": 1, "value": 0.0}],
                }
            )

        callback = Mock(side_effect=generate)
        with (
            patch.object(sweep, "verified_parameters", return_value=model),
            patch.object(sweep, "apply_edits", side_effect=sweep_tests.apply),
        ):
            try:
                sweep.run(
                    sweep_tests.Torch,
                    tokenizer,
                    model,
                    data,
                    plan,
                    deadline,
                    callback,
                    events.append,
                    clock=lambda: clock,
                    cpu_deadline=cpu_deadline,
                    cpu_clock=lambda: cpu_clock,
                )
            except Exception as error:
                self.fail(
                    f"Valid inert sweep scenario must complete: {type(error).__name__}: {error}"
                )
        for call in callback.call_args_list:
            self.assertIs(call.args[0], sweep_tests.Torch)
            self.assertIs(call.args[1], tokenizer)
            self.assertIs(call.args[2], model)
        return events, callback

    def test_sweep_run(self) -> None:
        events, callback = self.sweep_scenario()
        steps = [x for x in events if x["type"] == "step"]
        self.assertEqual(events[-1]["status"], "complete")
        self.assertTrue(events[-1]["coverage"]["complete"])
        self.assertEqual(
            events[-1]["compute_total_ms"], sum(x["compute_ms"] for x in steps)
        )
        self.assertEqual(events[-1]["reason"], "accepted_plan_complete")
        self.assertEqual(callback.call_count, 2 * len(steps))

    def test_sweep_exhausted(self) -> None:
        for deadline, cpu in ((9.5, None), (120, 4.5)):
            events, callback = self.sweep_scenario(deadline=deadline, cpu_deadline=cpu)
            callback.assert_not_called()
            self.assertEqual(events[-1]["status"], "time_limit")
            self.assertEqual(events[-1]["coverage"]["completed_ids"], [])

    def test_sweep_partial(self) -> None:
        events, callback = self.sweep_scenario(deadline=9.5)
        callback.assert_not_called()
        self.assertEqual(events[-1]["status"], "time_limit")
        self.assertEqual(events[-1]["coverage"]["completed_ids"], [])
        self.assertEqual(events[-1]["compute_total_ms"], 0.0)

    def test_sweep_probe(self) -> None:
        events, callback = self.sweep_scenario()
        self.assertGreater(len(events), 1)
        for call in callback.call_args_list:
            self.assertEqual(call.args[4:6], (1, 7))
            self.assertEqual(call.kwargs, {"activation_site": "block"})

    def pair_scenario(
        self, site: str
    ) -> tuple[list[dict[str, Any]], list[tuple[Any, bool]], list[int]]:
        events: list[dict[str, Any]] = []
        calls: list[tuple[Any, bool]] = []
        removals: list[int] = []

        class Tensor:
            def __init__(self, values: list[float]) -> None:
                self.values = values

            def detach(self) -> Any:
                return self

            def float(self) -> Any:
                return self

            def tolist(self) -> list[float]:
                return self.values

        class Hidden:
            def __init__(self, side: int) -> None:
                self.side = side

            def __getitem__(self, key: tuple[int, int]) -> Tensor:
                batch, position = key
                self_test.assertEqual(batch, 0)
                return Tensor([float(10 * (self.side + 1) + position)] + [0.0] * 575)

        class Site:
            def __init__(self) -> None:
                self.callback: Any = None

            def register_forward_hook(self, callback: Any) -> Any:
                self.callback = callback
                return SimpleNamespace(remove=lambda: removals.append(1))

        sites = {name: Site() for name in ("block", "attention", "mlp")}
        sites["block"].self_attn = sites["attention"]
        sites["block"].mlp = sites["mlp"]
        self_test = self

        class Inner:
            layers = [sites["block"]] * 30

            def __call__(self, tokens: Any, *, use_cache: bool) -> Any:
                side = len(calls)
                calls.append((tokens, use_cache))
                output = Hidden(side)
                sites[site].callback(
                    sites[site], (), (output,) if site == "block" else output
                )
                return object()

        request = {**pair_tests.request(), "activation_site": site}
        preview = pair.token_preview(pair_tests.Tokenizer(), request["prompts"])
        tokenizer = SimpleNamespace(
            encode=lambda text, **kw: pair_tests.Tokenizer().encode(text, **kw).ids
        )
        torch = SimpleNamespace(tensor=lambda value: value, inference_mode=nullcontext)
        with patch("time.perf_counter", side_effect=[1.0, 1.25]):
            pair.run(
                torch,
                tokenizer,
                SimpleNamespace(model=Inner()),
                request,
                preview,
                {name: name + " fixture" for name in sites},
                events.append,
            )
        return events, calls, removals

    def test_pair_run(self) -> None:
        for site in ("block", "attention", "mlp"):
            events, calls, removals = self.pair_scenario(site)
            self.assertEqual(calls, [([[1, 2, 3]], False), ([[1, 4]], False)])
            self.assertEqual(removals, [1, 1])
            self.assertEqual(
                [x["comparison_phase"] for x in events if x["type"] == "prefill"],
                ["prompt A", "prompt B"],
            )
            self.assertEqual(
                events[-1]["record_count"],
                len([x for x in events if x["type"] == "step"]),
            )
            self.assertEqual(events[-1]["reason"], "prompt_pair_complete")
            self.assertEqual(events[-1]["compute_total_ms"], 250.0)

    def test_pair_capture(self) -> None:
        for site in ("block", "attention", "mlp"):
            events, _, _ = self.pair_scenario(site)
            step = next(x for x in events if x["type"] == "step")
            self.assertEqual(
                step["prompt_pair"]["a"]["activation"], [12.0] + [0.0] * 575
            )
            self.assertEqual(
                step["prompt_pair"]["b"]["activation"], [21.0] + [0.0] * 575
            )
            self.assertEqual(step["activation"], [9.0] + [0.0] * 575)


if __name__ == "__main__":
    unittest.main()
