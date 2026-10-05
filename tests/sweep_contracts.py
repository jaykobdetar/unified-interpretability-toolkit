"""Pure planner, sparse FP32 reference undo and mocked lifecycle tests; no ML/child."""

from contextlib import nullcontext
from copy import deepcopy
import math
import io
from pathlib import Path
import struct
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import inference_sweep as sweep
import live_inference as live
from inference_edits import SOURCE_MODEL

NAME = "model.layers.0.self_attn.q_proj.weight"


def request():
    return {
        "mode": "sweep",
        "source_model": SOURCE_MODEL,
        "prompts": ["public synthetic"],
        "targets": [{"kind": "offset", "layer": 0, "heads": [0], "offset": 0}],
        "operation": "zero",
        "seed": 7,
        "capture_layer": 7,
        "activation_site": "block",
    }


def admitted(data=None):
    value = request() if data is None else data
    plan = sweep.build_plan(value, False)
    return {**value, "plan_digest": plan["digest"]}, plan


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def number(value):
    return struct.unpack("<f", struct.pack("<I", value))[0]


class Scalar(float):
    def sqrt(self):
        return Scalar(math.sqrt(self))


class Vec:
    def __init__(self, values):
        self.values = list(values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def clone(self):
        return Vec(self.values)

    def double(self):
        return self

    def view(self, dtype):
        return Vec(bits(v) for v in self.values) if dtype == "int32" else self

    def __ne__(self, other):
        return Vec(a != b for a, b in zip(self.values, other.values))

    def __sub__(self, other):
        return Vec(a - b for a, b in zip(self.values, other.values))

    def square(self):
        return Vec(v * v for v in self.values)

    def sum(self):
        return Scalar(math.fsum(self.values))

    def mean(self):
        return Scalar(self.sum() / len(self.values))

    def abs(self):
        return Vec(abs(v) for v in self.values)

    def max(self):
        return Scalar(max(self.values))


class Parameter:
    def __init__(self):
        self.values = {0: bits(-0.0), 1: bits(1.25)}
        self.corrupt = False

    def get(self, index):
        return number(self.values.get(index, bits(0.25)))

    def put(self, index, value):
        self.values[index] = bits(value)

    def reshape(self, *_args):
        return self

    def index_select(self, _axis, index):
        return Vec(self.get(i) for i in index)

    def index_copy_(self, _axis, index, values):
        for i, v in zip(index, values.values):
            self.put(i, v)
        if self.corrupt and index:
            self.values[index[0]] ^= 1


class Torch:
    long = "long"
    int32 = "int32"
    no_grad = staticmethod(nullcontext)
    tensor = staticmethod(lambda values, **_kwargs: list(values))
    equal = staticmethod(lambda a, b: a.values == b.values)
    argmax = staticmethod(
        lambda v: max(range(len(v.values)), key=lambda i: v.values[i])
    )

    @staticmethod
    def softmax(v, dim):
        exp = [math.exp(n - max(v.values)) for n in v.values]
        total = math.fsum(exp)
        return Vec(x / total for x in exp)


def apply(_torch, model, edits, _source):
    for edit in edits:
        for index in sweep.selected_indices([edit])[edit["tensor"]]:
            param = model[edit["tensor"]]
            param.put(
                index,
                (
                    0.0
                    if edit["operation"] == "zero"
                    else param.get(index) * edit["scale"]
                ),
            )


def row(start=0, end=1, operation="zero", **kwargs):
    return {
        "tensor": NAME,
        "shape": [576, 576],
        "kind": "rows",
        "start": start,
        "end": end,
        "operation": operation,
        **kwargs,
    }


class Contracts(unittest.TestCase):
    def test_mocked_admission_deadline_includes_verification_and_cancel_coverage(self):
        data, plan = admitted()
        clock = SimpleNamespace(value=100.0)

        class Stream(io.BytesIO):
            def close(self):
                self.was_closed = True

            def fileno(self):
                return 123456

        child = SimpleNamespace(stdin=Stream(), stdout=Stream(), poll=lambda: None)

        def verify(_path, **_kwargs):
            clock.value += 10

        with (
            patch.object(live, "available", return_value=6 * live.GIB),
            patch.object(live.time, "monotonic", side_effect=lambda: clock.value),
            patch.object(live, "verify_model", side_effect=verify) as pins,
            patch.object(live.subprocess, "Popen", return_value=child) as spawn,
            patch.object(live.os, "set_blocking"),
            patch.object(live, "signal_and_reap", return_value=True),
        ):
            session = live.Session("unused", Path("/unused"))
            session.start(data)
            self.assertEqual(session.started, 100.0)
            self.assertEqual(session.last_seen, 110.0)
            self.assertEqual(float(spawn.call_args.args[0][-2]), 220.0)
            self.assertEqual(spawn.call_count, 1)
            self.assertEqual(pins.call_count, 1)
            self.assertNotIn(session.id, str(session.metadata()))
            with patch.object(session, "tick"), self.assertRaises(ValueError):
                session.start(data)
            self.assertEqual(spawn.call_count, 1)
            session.details["sweep_current"] = sweep.record_ids(plan)[0]
            session.stop()
            self.assertEqual(session.status, "cancelled")
            self.assertIsNone(session.process)
            self.assertEqual(session.details["sweep_coverage"]["completed_ids"], [])
            self.assertEqual(
                session.details["sweep_coverage"]["interrupted_id"], "empty/prompt-1"
            )

    def test_versioned_control_and_plan_digest_vectors(self):
        vectors = [
            (0, 16, "cc886f2cb5b86aec2b5984a4830957b85216ea3038e7937a8cf3e939f927e177"),
            (7, 19, "811d53d19e9bd205e2f6b1aa33388533ee9038d46f3cf97a908f0ad475d51663"),
            (
                4294967295,
                60,
                "dc6788f83655fc113122456393250638892e21767e7a9af1505c09f663c13773",
            ),
        ]
        for seed, offset, digest in vectors:
            data = request()
            data["seed"] = seed
            _, plan = admitted(data)
            self.assertEqual(plan["cases"][2]["target"]["offset"], offset)
            self.assertEqual(plan["digest"], digest)

    def test_reproducible_expanded_plan_controls_and_maximum_product(self):
        data = request()
        data.update(
            prompts=["one", "two"],
            targets=[
                {"kind": "head", "layer": 0, "head": 0},
                {"kind": "offset", "layer": 1, "heads": list(range(8)), "offset": 3},
            ],
        )
        accepted, plan = admitted(data)
        self.assertEqual(plan, sweep.build_plan(accepted))
        self.assertEqual(plan["records"], 10)
        self.assertEqual(plan["prefills"], 20)
        self.assertEqual(
            [c["role"] for c in plan["cases"]],
            ["empty_control", "target", "matched_control", "target", "matched_control"],
        )
        for i in (1, 3):
            a, b = plan["cases"][i : i + 2]
            self.assertEqual(a["selected_cells"], b["selected_cells"])
            self.assertEqual(
                [e["operation"] for e in a["edits"]],
                [e["operation"] for e in b["edits"]],
            )
            self.assertFalse(sweep._rows(a["target"]) & sweep._rows(b["target"]))
        self.assertEqual(len(plan["cases"][3]["edits"]), 8)
        self.assertEqual(plan["cases"][1]["selected_cells"], 64 * 576)
        self.assertIn("subset", plan["coverage"])
        self.assertIn("contiguous", plan["cases"][2]["control_geometry"])
        other = deepcopy(accepted)
        other["prompts"][0] = "changed"
        with self.assertRaises(ValueError):
            sweep.build_plan(other)
        self.assertEqual(sweep.schema()["wall_seconds"], 120)
        self.assertEqual(sweep.schema()["worker_cpu_seconds"], 90)

    def test_strict_input_caps_seed_and_no_hidden_expansion(self):
        changes = {
            "seed": [True, -1, 2**32, 1.5],
            "prompts": [[], ["a", "b", "c"], [True], ["é" * 1025]],
            "targets": [
                [],
                [{"kind": "head", "layer": 0, "head": 0}] * 3,
                [{"kind": "offset", "layer": 0, "heads": list(range(9)), "offset": 0}],
                [{"kind": "head", "layer": False, "head": 0}],
            ],
            "capture_layer": [True, 30],
            "activation_site": ["q_proj", None],
        }
        for field, values in changes.items():
            for value in values:
                data = request()
                data[field] = value
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ValueError),
                ):
                    sweep.build_plan(data, False)
        for field, value in [
            ("max_new_tokens", 2),
            ("edits", []),
            ("retry", True),
            ("wall_seconds", 3600),
        ]:
            data = request()
            data[field] = value
            with self.assertRaises(ValueError):
                sweep.build_plan(data, False)
        data = request()
        data.update(operation="scale", scale=0.5)
        _, plan = admitted(data)
        self.assertTrue(
            all(e["scale"] == 0.5 for c in plan["cases"][1:] for e in c["edits"])
        )
        for value in (True, float("inf"), 101):
            data["scale"] = value
            with self.assertRaises(ValueError):
                sweep.build_plan(data, False)

    def test_union_deduplicates_overlap_and_caps_before_snapshot(self):
        edits = [
            row(),
            {
                "tensor": NAME,
                "shape": [576, 576],
                "kind": "columns",
                "start": 0,
                "end": 1,
                "operation": "zero",
            },
        ]
        self.assertEqual(len(sweep.selected_indices(edits)[NAME]), 1151)
        with self.assertRaises(ValueError):
            sweep.selected_indices([row(0, 115)])

    def test_bitwise_restore_overlap_aliases_and_failure(self):
        parameter = Parameter()
        model = {NAME: parameter}
        original = {
            i: bits(parameter.get(i))
            for i in range(576 * 576)
            if i < 576 or i % 576 == 0
        }
        edits = [
            row(operation="scale", scale=2),
            {
                "tensor": NAME,
                "shape": [576, 576],
                "kind": "columns",
                "start": 0,
                "end": 1,
                "operation": "scale",
                "scale": -1,
            },
        ]
        with (
            patch.object(sweep, "verified_parameters", return_value=model),
            patch.object(sweep, "apply_edits", side_effect=apply),
        ):
            with sweep.temporary_edits(Torch, model, edits, SOURCE_MODEL) as change:
                self.assertEqual(change["selected_cells"], 1151)
                self.assertEqual(parameter.get(1), 2.5)
            self.assertEqual({i: bits(parameter.get(i)) for i in original}, original)
            with self.assertRaisesRegex(RuntimeError, "controlled"):
                with sweep.temporary_edits(Torch, model, [row()], SOURCE_MODEL):
                    raise RuntimeError("controlled computation failure")
            self.assertEqual({i: bits(parameter.get(i)) for i in original}, original)
        embed = Parameter()
        tied = {"model.embed_tokens.weight": embed, "lm_head.weight": embed}
        edit = {
            "tensor": "model.embed_tokens.weight",
            "shape": [49152, 576],
            "kind": "element",
            "row": 0,
            "col": 0,
            "operation": "scale",
            "scale": -1,
        }
        with (
            patch.object(sweep, "verified_parameters", return_value=tied),
            patch.object(sweep, "apply_edits", side_effect=apply),
        ):
            with sweep.temporary_edits(Torch, tied, [edit], SOURCE_MODEL):
                self.assertEqual(bits(tied["lm_head.weight"].get(0)), bits(0.0))
            self.assertEqual(bits(tied["lm_head.weight"].get(0)), bits(-0.0))
            embed.corrupt = True
            with self.assertRaises(sweep.RestorationFailure):
                with sweep.temporary_edits(Torch, tied, [edit], SOURCE_MODEL):
                    pass

    def test_independent_small_vocabulary_metrics(self):
        log3 = math.log(3)
        values = sweep.metrics(Torch, Vec([0.0, log3]), Vec([log3, 0.0]))
        self.assertAlmostEqual(values["logit_delta_rms"], log3)
        self.assertAlmostEqual(values["softmax_total_variation"], 0.5)
        self.assertEqual(values["baseline_argmax_id"], 1)
        self.assertEqual(values["edited_argmax_id"], 0)
        self.assertAlmostEqual(values["baseline_argmax_logit_delta"], -log3)

    def test_mocked_sequence_restores_before_each_baseline_and_stops_partial(self):
        data, plan = admitted()
        model = {NAME: Parameter()}
        calls = []
        events = []

        def generate(
            _torch,
            _tokenizer,
            _model,
            prompt,
            limit,
            layer,
            record,
            scores,
            activation_site,
        ):
            calls.append((prompt, limit, bits(model[NAME].get(0))))
            v = model[NAME].get(1)
            logits = Vec([v, 0.0])
            scores(logits)
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

        tokenizer = type(
            "Tokenizer", (), {"decode": staticmethod(lambda ids, **_kw: str(ids[0]))}
        )()
        with (
            patch.object(sweep, "verified_parameters", return_value=model),
            patch.object(sweep, "apply_edits", side_effect=apply),
        ):
            sweep.run(
                Torch,
                tokenizer,
                model,
                data,
                plan,
                120,
                generate,
                events.append,
                clock=lambda: 0,
            )
        self.assertEqual(len(calls), 6)
        self.assertTrue(events[-1]["coverage"]["complete"])
        self.assertEqual(events[-1]["status"], "complete")
        self.assertEqual(
            [bits(model[NAME].get(i)) for i in (0, 1)], [bits(-0.0), bits(1.25)]
        )
        events.clear()
        calls.clear()
        sweep.run(
            Torch,
            tokenizer,
            model,
            data,
            plan,
            120,
            generate,
            events.append,
            clock=lambda: 115,
        )
        self.assertFalse(calls)
        self.assertEqual(events[-1]["status"], "time_limit")
        self.assertEqual(len(events[-1]["coverage"]["unrun_ids"]), 3)

    def test_apply_failure_restores_before_propagating(self):
        param = Parameter()
        model = {NAME: param}
        original = [bits(param.get(i)) for i in range(576)]

        def fail(*args):
            apply(*args)
            raise RuntimeError("controlled apply failure")

        with (
            patch.object(sweep, "verified_parameters", return_value=model),
            patch.object(sweep, "apply_edits", side_effect=fail),
            self.assertRaisesRegex(RuntimeError, "controlled"),
        ):
            with sweep.temporary_edits(Torch, model, [row()], SOURCE_MODEL):
                self.fail("No yield after failed apply")
        self.assertEqual([bits(param.get(i)) for i in range(576)], original)

    def test_baseline_repeat_failure_aborts_without_next_intervention(self):
        data, plan = admitted()
        model = {NAME: Parameter()}
        calls = []
        events = []

        def generate(
            _torch,
            _tokenizer,
            _model,
            prompt,
            limit,
            layer,
            record,
            scores,
            activation_site,
        ):
            calls.append(1)
            v = 2.0 if len(calls) == 3 else 1.0
            scores(Vec([v, 0.0]))
            record(
                {
                    "type": "step",
                    "activation": [0.0] * 576,
                    "layer": layer,
                    "activation_site": activation_site,
                    "activation_kind": "fixture",
                    "position": 0,
                    "input_token_id": 1,
                    "compute_ms": 1.0,
                    "top_logits": [{"id": 0, "value": v}, {"id": 1, "value": 0.0}],
                }
            )

        tokenizer = type(
            "Tokenizer", (), {"decode": staticmethod(lambda ids, **_kw: str(ids[0]))}
        )()
        with (
            patch.object(sweep, "verified_parameters", return_value=model),
            patch.object(sweep, "apply_edits", side_effect=apply),
            self.assertRaises(sweep.RestorationFailure),
        ):
            sweep.run(
                Torch,
                tokenizer,
                model,
                data,
                plan,
                120,
                generate,
                events.append,
                clock=lambda: 0,
            )
        self.assertEqual(len(calls), 3)
        self.assertEqual(sum(e["type"] == "step" for e in events), 1)
        self.assertFalse(any(e["type"] == "sweep_done" for e in events))


if __name__ == "__main__":
    unittest.main()
