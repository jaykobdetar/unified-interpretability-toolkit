"""Exact registry callbacks and public protocol boundaries, with inert values."""

from pathlib import Path
import sys
from typing import get_type_hints
import unittest
from unittest.mock import Mock, call, patch

import inference_experiments as experiments
import inference_generation as generation


class ExecutionContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.context = experiments.Context(
            torch=object(),
            tokenizer=object(),
            model=object(),
            generate=Mock(),
            compare=Mock(),
            prompt_pair=Mock(),
            sweep=Mock(),
            record=Mock(),
            capture_sites={"block": "held"},
            verified_parameters=Mock(),
            pair_preview={"preview": "held"},
            sweep_plan={"plan": "held"},
            sweep_deadline=17.5,
            cpu_allowance=29,
        )
        self.request = {"prompt": "held", "max_new_tokens": 17, "layer": 3}

    def test_preview_packet(self) -> None:
        self.assertIsNone(experiments.run_preview(self.context, self.request))
        self.context.record.assert_called_once_with(
            {
                "type": "preview_done",
                "preview": self.context.pair_preview,
                "reason": "token_preview",
            }
        )
        self.assertIs(
            self.context.record.call_args.args[0]["preview"], self.context.pair_preview
        )

    def test_sweep_arguments(self) -> None:
        c = self.context
        self.assertIsNone(experiments.run_sweep(c, self.request))
        c.sweep.assert_called_once_with(
            c.torch,
            c.tokenizer,
            c.model,
            self.request,
            c.sweep_plan,
            c.sweep_deadline,
            c.generate,
            c.record,
            cpu_deadline=c.cpu_allowance,
        )
        self.assertIs(c.sweep.call_args.args[3], self.request)
        self.assertIs(c.sweep.call_args.args[4], c.sweep_plan)

    def test_prompt_pair_arguments_and_verify_order(self) -> None:
        c = self.context
        ordered = Mock()
        ordered.attach_mock(c.verified_parameters, "verify")
        ordered.attach_mock(c.prompt_pair, "pair")
        self.assertIsNone(experiments.run_prompt_pair(c, self.request))
        self.assertEqual(
            ordered.mock_calls,
            [
                call.verify(c.model),
                call.pair(
                    c.torch,
                    c.tokenizer,
                    c.model,
                    self.request,
                    c.pair_preview,
                    c.capture_sites,
                    c.record,
                ),
            ],
        )
        self.assertIs(c.prompt_pair.call_args.args[3], self.request)

    def test_comparison_arguments(self) -> None:
        c = self.context
        self.assertIsNone(experiments.run_comparison(c, self.request))
        c.compare.assert_called_once_with(c.torch, c.tokenizer, c.model, self.request)
        self.assertIs(c.compare.call_args.args[3], self.request)

    def test_generation_arguments_and_defaults(self) -> None:
        c = self.context
        self.assertIsNone(experiments.run_generation(c, self.request))
        c.generate.assert_called_once_with(
            c.torch,
            c.tokenizer,
            c.model,
            "held",
            17,
            3,
            activation_site="block",
            observation=None,
        )
        c.generate.reset_mock()
        observation = {"held": "observation"}
        request = {
            **self.request,
            "activation_site": "attention",
            "observation": observation,
        }
        experiments.run_generation(c, request)
        c.generate.assert_called_once_with(
            c.torch,
            c.tokenizer,
            c.model,
            "held",
            17,
            3,
            activation_site="attention",
            observation=observation,
        )
        self.assertIs(c.generate.call_args.kwargs["observation"], observation)

    def test_execute_retains_selected_callback_and_original_values(self) -> None:
        selected = Mock()
        with patch.object(experiments, "select", return_value=selected) as select:
            self.assertIsNone(experiments.execute(self.request, self.context))
        select.assert_called_once_with(self.request)
        self.assertIs(select.call_args.args[0], self.request)
        selected.execute.assert_called_once_with(self.context, self.request)
        self.assertIs(selected.execute.call_args.args[0], self.context)
        self.assertIs(selected.execute.call_args.args[1], self.request)

    def test_coordinator_closed_modes_and_edit_priority(self) -> None:
        for request, mode, kind in [
            ({}, "sweep", experiments.Kind.SWEEP),
            ({"edits": []}, "sweep", experiments.Kind.SWEEP),
            ({}, "prompt_pair_preview", experiments.Kind.PREVIEW),
            ({}, "prompt_pair", experiments.Kind.PROMPT_PAIR),
            ({}, "generation", experiments.Kind.GENERATION),
            ({"edits": []}, "generation", experiments.Kind.COMPARISON),
        ]:
            with self.subTest(mode=mode, request=request):
                self.assertIs(
                    experiments.for_coordinator(request, mode),
                    experiments.REGISTRY[kind],
                )
        for mode in (None, "", "unknown", "comparison", [], {}):
            with self.subTest(mode=mode):
                self.assertIsNone(experiments.for_coordinator({"edits": []}, mode))


class ProtocolContracts(unittest.TestCase):
    def test_tokenizer_encode(self) -> None:
        self.assertEqual(
            get_type_hints(generation.Tokenizer.encode),
            {"text": str, "add_special_tokens": bool, "return": list[int]},
        )

    def test_tokenizer_decode(self) -> None:
        self.assertEqual(
            get_type_hints(generation.Tokenizer.decode),
            {"ids": list[int], "skip_special_tokens": bool, "return": str},
        )

    def test_generate(self) -> None:
        hints = get_type_hints(experiments.Generate.__call__)
        self.assertEqual(hints["prompt"], object)
        self.assertEqual(hints["record"], experiments.Record)
        self.assertEqual(hints["return"], type(None))

    def test_compare(self) -> None:
        hints = get_type_hints(experiments.Compare.__call__)
        self.assertEqual(hints["request"], experiments.Request)
        self.assertEqual(hints["record"], experiments.Record)
        self.assertEqual(hints["return"], type(None))

    def test_prompt_pair(self) -> None:
        hints = get_type_hints(experiments.PromptPair.__call__)
        self.assertEqual(hints["request"], experiments.Request)
        self.assertEqual(hints["record"], experiments.Record)
        self.assertEqual(hints["return"], type(None))

    def test_sweep(self) -> None:
        hints = get_type_hints(experiments.Sweep.__call__)
        self.assertEqual(hints["request"], experiments.Request)
        self.assertEqual(hints["record"], experiments.Record)
        self.assertEqual(hints["return"], type(None))


if __name__ == "__main__":
    unittest.main()
