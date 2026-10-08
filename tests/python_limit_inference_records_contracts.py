"""Held CPU protocol record edges, without model execution or ML imports."""

import contextlib
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
import inference_edits as edits
import inference_sweep as sweep
import launch

FIXTURES = ROOT / "tests/fixtures/cpu-protocols"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())


def rows(name):
    return [
        json.loads(line)
        for line in (FIXTURES / (name + ".jsonl")).read_text().splitlines()
    ]


class InferenceRecordLimits(unittest.TestCase):
    def check(self, operation, accepted):
        if accepted:
            operation()
        else:
            with self.assertRaises(ValueError):
                operation()

    def test_edit_branch_generated_count_and_text_character_edges(self):
        baseline = next(item for item in rows("edit_empty") if "baseline" in item)
        for count, accepted in ((0, False), (1, True), (32, True), (33, False)):
            value = deepcopy(baseline)
            value["index"] = max(0, count - 1)
            for side in ("baseline", "edited"):
                value[side]["generated_ids"] = [3] * count
                value[side]["token_id"] = 3
            self.check(lambda: edits.validate_pair(value), accepted)
        for length, accepted in ((0, True), (8192, True), (8193, False)):
            value = deepcopy(baseline)
            value["baseline"]["generated_text"] = "é" * length
            self.check(lambda: edits.validate_pair(value), accepted)
        # This guard counts decoded characters: 8192 two-byte characters are
        # accepted. Coordinator envelope byte limits are separately enforced.

    def sweep(self):
        request = MANIFEST["cases"]["sweep_head_zero"]["request"]
        plan = sweep.build_plan(request)
        step = next(
            item
            for item in rows("sweep_head_zero")
            if "sweep" in item and item["sweep"]["role"] != "empty_control"
        )
        sweep.validate_step(step, plan)
        return step, plan

    def test_sweep_changed_count_and_finite_nonnegative_metric_domains(self):
        baseline, plan = self.sweep()
        maximum = baseline["sweep"]["selected_cells"]
        for number, accepted in (
            (-1, False),
            (0, True),
            (maximum, True),
            (maximum + 1, False),
        ):
            value = deepcopy(baseline)
            value["sweep"]["changed_cells"] = number
            self.check(lambda: sweep.validate_step(value, plan), accepted)
        for field in ("parameter_delta_l2", "logit_delta_rms", "logit_delta_max_abs"):
            for number, accepted in (
                (-math.ulp(0.0), False),
                (0.0, True),
                (sys.float_info.max, True),
                (math.inf, False),
                (math.nan, False),
            ):
                value = deepcopy(baseline)
                destination = (
                    value["sweep"]
                    if field == "parameter_delta_l2"
                    else value["sweep"]["metrics"]
                )
                destination[field] = number
                self.check(lambda: sweep.validate_step(value, plan), accepted)

    def test_sweep_total_variation_tolerance_and_argmax_ids(self):
        baseline, plan = self.sweep()
        for number, accepted in (
            (-math.ulp(0.0), False),
            (0.0, True),
            (1 + 1e-12, True),
            (math.nextafter(1 + 1e-12, math.inf), False),
        ):
            value = deepcopy(baseline)
            value["sweep"]["metrics"]["softmax_total_variation"] = number
            self.check(lambda: sweep.validate_step(value, plan), accepted)
        for field in ("baseline_argmax_id", "edited_argmax_id"):
            for number, accepted in (
                (-1, False),
                (0, True),
                (49151, True),
                (49152, False),
            ):
                value = deepcopy(baseline)
                value["sweep"]["metrics"][field] = number
                self.check(lambda: sweep.validate_step(value, plan), accepted)

    def test_sweep_candidate_count_and_piece_character_limits(self):
        baseline, plan = self.sweep()
        template = baseline["sweep"]["candidates"][0]
        for count, accepted in ((0, False), (1, True), (10, True), (11, False)):
            value = deepcopy(baseline)
            value["sweep"]["candidates"] = [{**template, "id": n} for n in range(count)]
            self.check(lambda: sweep.validate_step(value, plan), accepted)
        for length, accepted in ((0, True), (1024, True), (1025, False)):
            value = deepcopy(baseline)
            value["sweep"]["candidates"][0]["piece"] = "é" * length
            self.check(lambda: sweep.validate_step(value, plan), accepted)

    def test_sweep_complete_actual_encoded_request_edge(self):
        for size, accepted in ((8192, True), (8193, False)):
            data = deepcopy(MANIFEST["cases"]["sweep_head_zero"]["request"])
            data.pop("plan_digest")
            data["prompts"] = ["a"]
            base = len(
                json.dumps(
                    {**data, "plan_digest": "a" * 64}, ensure_ascii=False
                ).encode()
            )
            controls, plain = divmod(size - base, 6)
            data["prompts"][0] += "\0" * controls + "x" * plain
            self.assertLessEqual(len(data["prompts"][0].encode()), 2048)
            self.assertEqual(
                len(
                    json.dumps(
                        {**data, "plan_digest": "a" * 64}, ensure_ascii=False
                    ).encode()
                ),
                size,
            )
            self.check(lambda: sweep.build_plan(data, False), accepted)

    def test_cleanup_margin_exact_wall_ten_and_cpu_five_seconds(self):
        data = MANIFEST["cases"]["sweep_head_zero"]["request"]
        plan = sweep.build_plan(data)
        for wall, cpu, accepted in (
            (10.0, 5.0, True),
            (math.nextafter(10.0, 0.0), 5.0, False),
            (10.0, math.nextafter(5.0, 0.0), False),
        ):
            generated = Mock(side_effect=RuntimeError("Reached inert prefill boundary"))
            records = []
            operation = lambda: sweep.run(
                None,
                None,
                None,
                data,
                plan,
                wall,
                generated,
                records.append,
                clock=lambda: 0,
                cpu_deadline=cpu,
                cpu_clock=lambda: 0,
            )
            if accepted:
                with self.assertRaisesRegex(RuntimeError, "inert prefill"):
                    operation()
                generated.assert_called_once()
            else:
                operation()
                generated.assert_not_called()
                self.assertEqual(records[-1]["status"], "time_limit")
                self.assertEqual(records[-1]["coverage"]["completed_ids"], [])

    def test_launcher_port_zero_one_65535_and_65536_preflight(self):
        for port, accepted in ((0, False), (1, True), (65535, True), (65536, False)):
            with (
                patch.object(launch.os, "access", return_value=True),
                patch.object(
                    launch.subprocess,
                    "call",
                    side_effect=AssertionError("No build/server allowed"),
                ),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                if accepted:
                    self.assertEqual(
                        launch.main(["--demo", "--check", "--port", str(port)]), 0
                    )
                else:
                    with self.assertRaises(SystemExit) as error:
                        launch.main(["--demo", "--check", "--port", str(port)])
                    self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
