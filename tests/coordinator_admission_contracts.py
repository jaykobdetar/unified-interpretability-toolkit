"""Exact coordinator admission forwarding and refusals with inert ownership."""

from contextlib import ExitStack, nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, call, patch

import live_inference as live
from experiment_interface_contracts import SavedInput


class AdmissionContracts(unittest.TestCase):
    def contracts(self) -> SimpleNamespace:
        return SimpleNamespace(
            architecture=SimpleNamespace(
                layers=5, capture_sites={"block": "b", "attention": "a", "mlp": "m"}
            ),
            edits=SimpleNamespace(source_model={"canonical": "held"}),
            validate_observation=Mock(return_value={"clean": "observation"}),
            validate_edits=Mock(return_value=[{"clean": "edit"}]),
            validate_request=Mock(return_value={"mode": "prompt_pair", "layer": 3}),
            build_plan=Mock(return_value={"records": 2}),
        )

    def start(
        self, data: dict[str, Any], contracts: SimpleNamespace
    ) -> SimpleNamespace:
        stream = SavedInput()
        child = SimpleNamespace(stdin=stream, stdout=SimpleNamespace(fileno=lambda: 42))
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(live, "available", return_value=6 * live.GIB)
            )
            stack.enter_context(patch.object(live.time, "monotonic", return_value=10))
            stack.enter_context(patch.object(live.time, "process_time", return_value=3))
            session = live.Session("unused", Path("/unused"))
            stack.enter_context(patch.object(session, "tick"))
            stack.enter_context(patch.object(session, "snapshot", return_value={}))
            stack.enter_context(
                patch.object(live, "_contracts", return_value=contracts)
            )
            stack.enter_context(patch.object(live, "verify_model"))
            stack.enter_context(
                patch.object(live.subprocess, "Popen", return_value=child)
            )
            stack.enter_context(patch.object(live.os, "set_blocking"))
            stack.enter_context(
                patch.object(live.sweep, "coverage", return_value={"completed": 0})
            )
            stack.enter_context(
                patch.object(
                    live.SweepAdmissionBudget,
                    "verification",
                    return_value=nullcontext(),
                )
            )
            try:
                session.start(data)
            finally:
                # This is only an inert object; no OS child or descriptor was created.
                session.process = None
        return SimpleNamespace(session=session, wire=stream.saved)

    def test_generation_defaults_and_exact_request_order(self) -> None:
        contracts = self.contracts()
        result = self.start({"prompt": "λ", "ignored": "held"}, contracts)
        self.assertEqual(
            result.wire,
            '{"prompt": "λ", "max_new_tokens": 16, "layer": 0, "activation_site": "block"}\n'.encode(),
            "coordinator admission default wire",
        )
        self.assertEqual(result.session.capture_layer, 0)
        self.assertIsNone(result.session.observation)
        contracts.validate_observation.assert_not_called()
        contracts.validate_edits.assert_not_called()
        contracts.validate_request.assert_not_called()
        contracts.build_plan.assert_not_called()

    def test_observation_then_edit_forwarding_and_identity(self) -> None:
        contracts = self.contracts()
        ordered = Mock()
        ordered.attach_mock(contracts.validate_observation, "observation")
        ordered.attach_mock(contracts.validate_edits, "edits")
        observation, edits, source = {"raw": "observation"}, [], {"raw": "source"}
        result = self.start(
            {
                "prompt": "held",
                "activation_site": "attention",
                "observation": observation,
                "edits": edits,
                "source_model": source,
            },
            contracts,
        )
        self.assertEqual(
            ordered.mock_calls,
            [call.observation(observation, "attention"), call.edits(edits, source)],
            "coordinator admission validator order",
        )
        self.assertIs(contracts.validate_observation.call_args.args[0], observation)
        self.assertIs(contracts.validate_edits.call_args.args[0], edits)
        self.assertIs(contracts.validate_edits.call_args.args[1], source)
        self.assertIs(
            result.session.observation, contracts.validate_observation.return_value
        )
        expected = {
            "prompt": "held",
            "max_new_tokens": 16,
            "layer": 0,
            "activation_site": "attention",
            "edits": [{"clean": "edit"}],
            "source_model": {"canonical": "held"},
            "observation": {"clean": "observation"},
        }
        self.assertEqual(
            result.wire,
            json.dumps(expected, allow_nan=False, ensure_ascii=False).encode() + b"\n",
            "coordinator admission normalized wire",
        )

    def test_pair_and_sweep_forwarding_and_capture_state(self) -> None:
        for mode in ("prompt_pair_preview", "prompt_pair", "sweep"):
            contracts = self.contracts()
            data = {"mode": mode, "capture_layer": 2, "raw": "held"}
            result = self.start(data, contracts)
            session = result.session
            self.assertIsNone(session.observation)
            if mode == "sweep":
                contracts.build_plan.assert_called_once_with(data)
                self.assertIs(contracts.build_plan.call_args.args[0], data)
                self.assertIs(session.sweep_plan, contracts.build_plan.return_value)
                self.assertEqual(
                    session.capture_layer,
                    2,
                    "coordinator admission sweep capture layer",
                )
                self.assertEqual(json.loads(result.wire), data)
                contracts.validate_request.assert_not_called()
            else:
                contracts.validate_request.assert_called_once_with(data)
                self.assertIs(contracts.validate_request.call_args.args[0], data)
                self.assertEqual(
                    session.capture_layer, 3, "coordinator admission pair capture layer"
                )
                self.assertEqual(
                    json.loads(result.wire), contracts.validate_request.return_value
                )
                self.assertIs(
                    session.pair_request,
                    (
                        contracts.validate_request.return_value
                        if mode == "prompt_pair"
                        else None
                    ),
                )
                contracts.build_plan.assert_not_called()

    def test_exact_refusals_and_precedence_before_model_admission(self) -> None:
        cases = [
            (
                {"prompt": "", "max_new_tokens": 0, "layer": -1},
                "Enter a nonempty prompt of at most 4096 UTF-8 bytes",
            ),
            (
                {"prompt": "held", "max_new_tokens": 0, "layer": -1},
                "Generate 1–32 tokens",
            ),
            ({"prompt": "held", "max_new_tokens": True}, "Generate 1–32 tokens"),
            (
                {"prompt": "held", "layer": 5, "activation_site": "bad"},
                "Choose activation layer 0–4",
            ),
            ({"prompt": "held", "layer": True}, "Choose activation layer 0–4"),
            (
                {"prompt": "held", "activation_site": None, "source_model": {}},
                "Choose activation site block, attention, or mlp",
            ),
            (
                {"prompt": "held", "source_model": {}},
                "source_model requires an explicit edits list",
            ),
            ({"mode": "unknown", "prompt": ""}, "Unknown inference mode"),
        ]
        for data, message in cases:
            contracts = self.contracts()
            session = live.Session("unused", Path("/unused"))
            with (
                self.subTest(data=data),
                patch.object(session, "tick"),
                patch.object(live, "_contracts", return_value=contracts),
                patch.object(
                    live,
                    "available",
                    side_effect=AssertionError("Validation must precede admission"),
                ),
                patch.object(live, "verify_model") as verify,
                patch.object(live.subprocess, "Popen") as spawn,
                self.assertRaises(ValueError) as caught,
            ):
                session.start(data)
            self.assertEqual(
                str(caught.exception), message, "coordinator admission exact refusal"
            )
            verify.assert_not_called()
            spawn.assert_not_called()
            self.assertIsNone(session.process)


if __name__ == "__main__":
    unittest.main()
