"""Held caller packets, default binding values and exact callback forwarding."""

from __future__ import annotations
from collections.abc import Callable
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, patch

import inference_edits as edits
import inference_observations as observations
import inference_prompt_pair as pair
import inference_sweep as sweep
import inference_worker as worker
import live_inference as live
import experiment_interface_contracts as interface_tests
import review_limit_contracts as review_tests
import prompt_pair_contracts as pair_tests

MODULES = {
    "edits": edits,
    "observations": observations,
    "prompt_pair": pair,
    "sweep": sweep,
}


class Contracts(unittest.TestCase):
    def test_worker_generate(self) -> None:
        values = [object() for _ in range(7)]
        with patch.object(worker, "run_generation", return_value=None) as run:
            worker.generate(*values[:6], record=values[6])
        self.assertEqual(run.call_args.args[:7], tuple(values))
        binding = run.call_args.args[7]
        self.assertEqual((binding.width, binding.vocab), (576, 49152))
        self.assertEqual(binding.capture_sites, worker.CAPTURE_SITES)

    def test_worker_main(self) -> None:
        for preview_only in (True, False):
            data = (
                {
                    "mode": "prompt_pair_preview",
                    "source_model": edits.SOURCE_MODEL,
                    "prompts": ["A", "B"],
                }
                if preview_only
                else {"prompt": "x", "max_new_tokens": 1, "layer": 0}
            )
            stream = Mock(wraps=io.BytesIO(json.dumps(data).encode()))
            tokenizer = pair_tests.Tokenizer()
            torch = SimpleNamespace(__version__="held")
            model = SimpleNamespace(
                _atlas_verified_layout={"held": "layout"},
                config=SimpleNamespace(_attn_implementation="eager"),
            )
            with (
                patch.object(worker.sys, "argv", ["worker", "/unused"]),
                patch.object(worker.sys, "stdin", SimpleNamespace(buffer=stream)),
                patch.object(worker.os, "sched_getaffinity", return_value={0}),
                patch.object(worker.os, "sched_setaffinity"),
                patch.object(worker.os, "nice"),
                patch.object(worker, "configure_worker_limits"),
                patch.object(live, "verify_model"),
                patch.object(
                    worker, "load_engine", return_value=(torch, tokenizer, model)
                ) as load,
                patch.object(worker, "emit") as emit,
                patch.object(worker, "execute") as execute,
                patch.dict(
                    sys.modules,
                    {
                        "transformers": SimpleNamespace(__version__="held"),
                        "tokenizers": SimpleNamespace(
                            __version__="held",
                            Tokenizer=SimpleNamespace(from_file=lambda _: tokenizer),
                        ),
                        "safetensors": SimpleNamespace(__version__="held"),
                    },
                ),
            ):
                worker.main()
            self.assertEqual(stream.readline.call_args.args, (8193,))
            context = execute.call_args.args[1]
            self.assertEqual(context.capture_sites, worker.CAPTURE_SITES)
            if preview_only:
                load.assert_not_called()
                emit.assert_not_called()
            else:
                self.assertEqual(emit.call_args.args[0]["runtime"]["seed"], 0)

    def test_live_metadata(self) -> None:
        held = json.loads(
            (
                Path(__file__).resolve().parent
                / "fixtures/inference-public-responses.json"
            ).read_text()
        )["responses"]["idle_metadata"]["body_utf8"]
        self.assertEqual(
            json.dumps(live.Session("unused", Path("/unused")).metadata()),
            held,
        )

    def test_live_start(self) -> None:
        review_tests.ReviewLimits().test_generation_limits_both_ends_and_source_edit_requirement()
        interface_tests.InterfaceContracts().test_coordinator_requests_and_mode_policy()

    def test_live_tick(self) -> None:
        fixture = review_tests.ReviewLimits(methodName="runTest")
        for event in (
            {"type": "step", "index": 0, "activation": [0.0] * 575},
            {"type": "step", "index": -1, "activation": [0.0] * 576},
        ):
            with fixture.simulated_worker(
                chunks=[(json.dumps(event) + "\n").encode()]
            ) as (session, _, reap):
                session.tick()
                self.assertEqual(session.status, "error")
                self.assertEqual(len(session.steps), 0)
                reap.assert_called_once()
        chunks = [b'{"type":"prefill"}\n'] * 9
        with (
            fixture.simulated_worker(chunks=chunks) as (session, _, _),
            patch.object(live.os, "read", side_effect=chunks) as read,
        ):
            session.tick()
            self.assertEqual(read.call_count, 8)

    def factory(self, module: Any) -> None:
        binding = module._bindings()
        value = binding.architecture
        self.assertEqual(
            (
                value.width,
                value.layers,
                value.query_heads,
                value.kv_heads,
                value.head_dim,
                value.vocab_size,
            ),
            (576, 30, 9, 3, 64, 49152),
        )
        self.assertIs(value.description, module.ARCH)
        self.assertEqual(
            value.capture_sites,
            (
                module.CAPTURE_SITES
                if hasattr(module, "CAPTURE_SITES")
                else worker.CAPTURE_SITES
            ),
        )
        if module is edits:
            self.assertIs(value.config, edits.CONFIG)
        if module is sweep:
            self.assertEqual(binding.max_cells, 65536)

    def test_factory_edits(self) -> None:
        self.factory(edits)

    def test_factory_observations(self) -> None:
        self.factory(observations)

    def test_factory_prompt_pair(self) -> None:
        self.factory(pair)

    def test_factory_sweep(self) -> None:
        self.factory(sweep)

    def callback(
        self,
        module: Any,
        field: str,
        callee: str,
        args: tuple[Any, ...],
        result: Any,
        void: bool = False,
    ) -> None:
        binding = module._bindings()
        mock = Mock(return_value=result)
        with patch.object(module, callee, mock):
            actual = getattr(binding, field)(*args)
        mock.assert_called_once_with(*args)
        if void:
            self.assertIsNone(actual)
        else:
            self.assertIs(actual, result)

    def test_callback_edits_validate(self) -> None:
        self.callback(
            MODULES["edits"],
            "validate_edits",
            "validate_edits",
            (["value"], {"source": 1}),
            [{"held": 1}],
            void=False,
        )

    def test_callback_edits_parameters(self) -> None:
        self.callback(
            MODULES["edits"],
            "verified_parameters",
            "verified_parameters",
            (object(),),
            {"held": 1},
            void=False,
        )

    def test_callback_observations_integer(self) -> None:
        self.callback(
            MODULES["observations"], "integer", "_integer", (3, 2, 7), True, void=False
        )

    def test_callback_observations_finite(self) -> None:
        self.callback(
            MODULES["observations"], "finite", "_finite", (3.5,), True, void=False
        )

    def test_callback_prompt_pair_validate(self) -> None:
        self.callback(
            MODULES["prompt_pair"],
            "validate_edits",
            "validate_edits",
            (["value"], {"source": 1}),
            [{"held": 1}],
            void=False,
        )

    def test_callback_prompt_pair_digest(self) -> None:
        self.callback(
            MODULES["prompt_pair"],
            "digest_for",
            "digest_for",
            (["A", "B"], [[1], [2]]),
            "held",
            void=False,
        )

    def test_callback_prompt_pair_preview(self) -> None:
        self.callback(
            MODULES["prompt_pair"],
            "validate_preview",
            "validate_preview",
            ({"held": 1},),
            None,
            void=True,
        )

    def test_callback_prompt_pair_delta(self) -> None:
        self.callback(
            MODULES["prompt_pair"],
            "difference",
            "difference",
            ([1.0], [2.0]),
            ([1.0], {"held": 1}),
            void=False,
        )

    def test_callback_sweep_integer(self) -> None:
        self.callback(MODULES["sweep"], "integer", "_int", (3, 2, 7), True, void=False)

    def test_callback_sweep_canonical(self) -> None:
        self.callback(
            MODULES["sweep"],
            "canonical",
            "_canonical",
            ({"held": 1},),
            "held",
            void=False,
        )

    def test_callback_sweep_target(self) -> None:
        self.callback(
            MODULES["sweep"],
            "target",
            "_target",
            ({"held": 1},),
            {"held": 1},
            void=False,
        )

    def test_callback_sweep_rows(self) -> None:
        self.callback(
            MODULES["sweep"], "rows", "_rows", ({"held": 1},), {3}, void=False
        )

    def test_callback_sweep_space(self) -> None:
        self.callback(
            MODULES["sweep"], "space", "_space", ({"held": 1},), (7, "held"), void=False
        )

    def test_callback_sweep_edits(self) -> None:
        self.callback(
            MODULES["sweep"],
            "edits",
            "_edits",
            ({"held": 1}, "zero", None),
            [{"held": 1}],
            void=False,
        )

    def test_callback_sweep_choice(self) -> None:
        self.callback(
            MODULES["sweep"],
            "choice",
            "_choice",
            ([{"held": 1}], 7, {"held": 2}, 3),
            {"held": 3},
            void=False,
        )

    def test_callback_sweep_limits(self) -> None:
        self.callback(MODULES["sweep"], "schema", "schema", (), {"held": 1}, void=False)

    def test_callback_sweep_validate(self) -> None:
        self.callback(
            MODULES["sweep"],
            "validate_edits",
            "validate_edits",
            (["value"], {"source": 1}),
            [{"held": 1}],
            void=False,
        )

    def test_callback_sweep_ids(self) -> None:
        self.callback(
            MODULES["sweep"],
            "record_ids",
            "record_ids",
            ({"held": 1},),
            ["held"],
            void=False,
        )


if __name__ == "__main__":
    unittest.main()
