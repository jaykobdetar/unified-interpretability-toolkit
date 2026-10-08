"""Portable public packet and intake contracts; inert callbacks, no ML engine."""

from __future__ import annotations

from contextlib import nullcontext
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, call, patch

import inference_edits as edits
import inference_experiments as experiments
import inference_prompt_pair as pair
import inference_worker as worker
import live_inference as live
import prompt_pair_contracts
import python_limit_worker_contracts as worker_contracts
import sweep_contracts


class SavedInput(io.BytesIO):
    saved: bytes = b""

    def close(self) -> None:
        self.saved = self.getvalue()
        super().close()


def trace(
    index: int, token: int, text: str, eos: bool, tops: list[int]
) -> dict[str, Any]:
    return {
        "type": "step",
        "index": index,
        "token_id": token,
        "token_piece": str(token),
        "generated_text": text,
        "eos": eos,
        "compute_ms": 2,
        "compute_total_ms": (index + 1) * 2,
        "top_logits": [{"id": value, "value": 0} for value in tops],
        "activation": [0.0] * 576,
        "layer": 7,
        "activation_site": "attention",
    }


def side(ids: list[int], text: str, eos: bool) -> dict[str, Any]:
    return {
        "token_id": ids[-1],
        "token_piece": str(ids[-1]),
        "generated_text": text,
        "eos": eos,
        "compute_ms": 2,
        "compute_total_ms": len(ids) * 2,
        "generated_ids": ids,
    }


class InterfaceContracts(unittest.TestCase):
    def test_comparison_packets_and_callback_order(self) -> None:
        baseline = [
            trace(0, 1, "b1", False, [0, 1]),
            trace(1, 2, "b12", False, [0, 1]),
            trace(2, 3, "b123", False, [0, 1]),
            trace(3, 4, "b1234", True, [0, 1]),
        ]
        edited = [
            trace(0, 1, "e1", False, [0, 2]),
            trace(1, 4, "e14", False, [0, 2]),
            trace(2, 2, "e142", False, [0, 2]),
        ]
        events: list[dict[str, Any]] = []
        order: list[str] = []
        args: list[tuple[Any, ...]] = []
        options: list[dict[str, Any]] = []
        engine, tokenizer, model = (
            object(),
            SimpleNamespace(decode=lambda ids, **_: str(ids[0])),
            object(),
        )
        observation = {"held": "opaque"}
        request = {
            "prompt": "synthetic",
            "max_new_tokens": 4,
            "layer": 7,
            "activation_site": "attention",
            "observation": observation,
            "edits": [],
            "source_model": edits.SOURCE_MODEL,
        }

        def generate(*values: Any, **kwargs: Any) -> None:
            branch = len(args)
            args.append(values[:6])
            options.append(kwargs)
            order.append("baseline" if branch == 0 else "edited")
            record, capture = values[6:8]
            record({"type": "prefill", "producer": branch})
            for event in baseline if branch == 0 else edited:
                capture(
                    [1.0, 3.0, 2.0, 4.0, 8.0, 9.0]
                    if branch == 0
                    else [2.0, 1.0, 3.0, 6.0, 7.0, 8.0]
                )
                record(event)
            record({"type": "done", "suppressed": branch})

        with (
            patch.object(worker, "generate", side_effect=generate),
            patch.object(edits, "validate_edits", return_value=[]) as validate,
            patch.object(
                edits,
                "verified_parameters",
                side_effect=lambda _: order.append("verify"),
            ) as verify,
            patch.object(
                edits, "apply_edits", side_effect=lambda *_: order.append("apply")
            ) as apply,
        ):
            worker.compare(engine, tokenizer, model, request, events.append)
        self.assertEqual(order, ["verify", "baseline", "apply", "edited"])
        self.assertEqual(args, [(engine, tokenizer, model, "synthetic", 4, 7)] * 2)
        self.assertEqual(
            options, [{"activation_site": "attention", "observation": observation}] * 2
        )
        validate.assert_called_once_with([], edits.SOURCE_MODEL)
        verify.assert_called_once_with(model)
        apply.assert_called_once_with(engine, model, [], edits.SOURCE_MODEL)

        candidate_rows = [
            {
                "id": 0,
                "piece": "0",
                "baseline_logit": 1.0,
                "edited_logit": 2.0,
                "delta": 1.0,
            },
            {
                "id": 1,
                "piece": "1",
                "baseline_logit": 3.0,
                "edited_logit": 1.0,
                "delta": -2.0,
            },
            {
                "id": 2,
                "piece": "2",
                "baseline_logit": 2.0,
                "edited_logit": 3.0,
                "delta": 1.0,
            },
        ]
        # Fixed consumed prefixes include divergence and one ended branch.
        rows = [
            (side([1], "b1", False), side([1], "e1", False), "matched_prefix"),
            (side([1, 2], "b12", False), side([1, 4], "e14", False), "matched_prefix"),
            (
                side([1, 2, 3], "b123", False),
                side([1, 4, 2], "e142", False),
                "different_prefix",
            ),
            (side([1, 2, 3, 4], "b1234", True), None, "branch_ended"),
        ]
        expected: list[dict[str, Any]] = [
            {"type": "prefill", "comparison_phase": "baseline", "edits": []},
            {"type": "prefill", "producer": 0},
            {"type": "prefill", "comparison_phase": "edited"},
            {"type": "prefill", "producer": 1},
        ]
        for index, (left, right, alignment) in enumerate(rows):
            expected.append(
                {
                    **(edited[index] if index < 3 else baseline[3]),
                    "baseline": left,
                    "edited": right,
                    "alignment": alignment,
                    "score_kind": "raw FP32 logits",
                    "activation_branch": "edited" if index < 3 else "baseline",
                    "candidates": (
                        candidate_rows
                        if index < 3
                        else [
                            {
                                "id": 0,
                                "piece": "0",
                                "baseline_logit": 1.0,
                                "edited_logit": None,
                                "delta": None,
                            },
                            {
                                "id": 1,
                                "piece": "1",
                                "baseline_logit": 3.0,
                                "edited_logit": None,
                                "delta": None,
                            },
                        ]
                    ),
                }
            )
        expected.append(
            {
                "type": "done",
                "generated_tokens": 4,
                "comparison_phase": "complete",
                "baseline": {
                    "generated_ids": [1, 2, 3, 4],
                    "generated_text": "b1234",
                    "reason": "eos",
                },
                "edited": {
                    "generated_ids": [1, 4, 2],
                    "generated_text": "e142",
                    "reason": "token_limit",
                },
                "reason": "comparison_complete",
                "compute_total_ms": 14,
            }
        )
        self.assertEqual(events, expected)
        for event in events:
            if event["type"] == "step":
                edits.validate_pair(event)

    def test_coordinator_requests_and_mode_policy(self) -> None:
        generation = {
            "prompt": "synthetic",
            "max_new_tokens": 3,
            "layer": 7,
            "activation_site": "attention",
        }
        comparison = {**generation, "edits": [], "source_model": edits.SOURCE_MODEL}
        preview = {
            "mode": "prompt_pair_preview",
            "source_model": edits.SOURCE_MODEL,
            "prompts": ["A", "B"],
        }
        paired = prompt_pair_contracts.request()
        sweep, _ = sweep_contracts.admitted()
        preview_wire = {
            "mode": "prompt_pair_preview",
            "prompts": ["A", "B"],
            "source_model": edits.SOURCE_MODEL,
        }
        pair_wire = {
            "mode": "prompt_pair",
            "prompts": ["A", "B"],
            "source_model": edits.SOURCE_MODEL,
            "layer": 7,
            "activation_site": "block",
            "positions": [{"a": 2, "b": 1}],
            "preview_digest": paired["preview_digest"],
        }
        for data, expected, mode in [
            (generation, generation, "generation"),
            (comparison, comparison, "generation"),
            (preview, preview_wire, "prompt_pair_preview"),
            (paired, pair_wire, "prompt_pair"),
            (sweep, sweep, "sweep"),
        ]:
            with self.subTest(mode=mode, comparison="edits" in data):
                stdin = SavedInput()
                process = SimpleNamespace(
                    stdin=stdin, stdout=SimpleNamespace(fileno=lambda: 42)
                )
                session = live.Session("configured-python", Path("/unused"))
                with (
                    patch.object(session, "tick"),
                    patch.object(session, "snapshot", return_value={}),
                    patch.object(live, "available", return_value=6 * live.GIB),
                    patch.object(live, "verify_model") as verify,
                    patch.object(live.time, "monotonic", return_value=0),
                    patch.object(live.time, "process_time", return_value=0),
                    patch.object(
                        live.SweepAdmissionBudget,
                        "verification",
                        return_value=nullcontext(),
                    ),
                    patch.object(
                        live.subprocess, "Popen", return_value=process
                    ) as launch,
                    patch.object(live.os, "set_blocking") as blocking,
                ):
                    session.start(data)
                self.assertEqual(
                    stdin.saved,
                    json.dumps(expected, allow_nan=False, ensure_ascii=False).encode()
                    + b"\n",
                )
                self.assertEqual(session.mode, mode)
                self.assertEqual(
                    session.pair_request, expected if mode == "prompt_pair" else None
                )
                self.assertEqual(
                    launch.call_args.args[0],
                    [
                        "configured-python",
                        "-B",
                        str(live.ROOT / "tools/inference_worker.py"),
                        "/unused",
                        *(["120", "90"] if mode == "sweep" else []),
                    ],
                )
                self.assertEqual(verify.call_count, 1)
                blocking.assert_called_once_with(42, False)
                session.process = None
        for mode in ("unknown", "comparison", [], {}):
            session = live.Session("configured-python", Path("/unused"))
            with (
                patch.object(live, "verify_model") as verify,
                patch.object(live.subprocess, "Popen") as launch,
                self.assertRaisesRegex(ValueError, "^Unknown inference mode$"),
            ):
                session.start({**comparison, "mode": mode})
            verify.assert_not_called()
            launch.assert_not_called()

    def test_preview_exact_packet_and_no_engine(self) -> None:
        data = {
            "mode": "prompt_pair_preview",
            "source_model": edits.SOURCE_MODEL,
            "prompts": ["A", "B"],
        }
        tokenizer = prompt_pair_contracts.Tokenizer()
        preview = pair.token_preview(tokenizer, ["A", "B"])
        raw = json.dumps(data).encode()
        stream = Mock(wraps=io.BytesIO(raw + b" " * (8193 - len(raw))))
        events: list[dict[str, Any]] = []
        with (
            patch.object(worker.sys, "argv", ["worker", "/unused"]),
            patch.object(worker.sys, "stdin", SimpleNamespace(buffer=stream)),
            patch.object(worker.os, "sched_getaffinity", return_value={0}),
            patch.object(worker.os, "sched_setaffinity"),
            patch.object(worker.os, "nice"),
            patch.object(worker, "configure_worker_limits"),
            patch.object(live, "verify_model") as verify,
            patch.dict(
                sys.modules,
                {
                    "tokenizers": SimpleNamespace(
                        Tokenizer=SimpleNamespace(from_file=lambda _: tokenizer)
                    )
                },
            ),
            patch.object(worker, "load_engine") as load,
            patch.object(worker, "emit", side_effect=events.append),
        ):
            worker.main()
        self.assertEqual(
            events,
            [{"type": "preview_done", "preview": preview, "reason": "token_preview"}],
        )
        self.assertEqual(stream.readline.call_args_list, [call(8193)])
        verify.assert_called_once_with(Path("/unused"))
        load.assert_not_called()

    def test_loaded_dispatch_existing_contract(self) -> None:
        worker_contracts.WorkerLimits(
            "test_worker_loaded_dispatch_preserves_execution_arguments"
        ).test_worker_loaded_dispatch_preserves_execution_arguments()

    def test_loaded_selector_priority_and_fallback(self) -> None:
        for request, kind in [
            ({"mode": "sweep", "edits": []}, experiments.Kind.SWEEP),
            ({"mode": "prompt_pair", "edits": []}, experiments.Kind.PROMPT_PAIR),
            ({"edits": []}, experiments.Kind.COMPARISON),
            ({}, experiments.Kind.GENERATION),
            *[
                ({"mode": value}, experiments.Kind.GENERATION)
                for value in ("unknown", [], {})
            ],
            ({"mode": "unknown", "edits": []}, experiments.Kind.COMPARISON),
        ]:
            with self.subTest(request=request):
                self.assertEqual(experiments.select(request).kind, kind)


if __name__ == "__main__":
    unittest.main()
