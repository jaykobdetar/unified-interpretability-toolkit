"""Legacy facade argument/default identity and inert entrypoint delegation."""

from __future__ import annotations
from contextlib import ExitStack
import io
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, patch

import inference_edits as edits
import inference_observations as observations
import inference_worker as worker
import live_inference as live
import prompt_pair_contracts as pair_tests


class FacadeContracts(unittest.TestCase):
    def routed(self, module: ModuleType, name: str, args: tuple[Any, ...]) -> None:
        binding, result = object(), {"held": name}
        with (
            patch.object(module, "_bindings", return_value=binding) as factory,
            patch.object(module.contract, name, return_value=result) as operation,
        ):
            actual = getattr(module, name)(*args)
        factory.assert_called_once_with()
        operation.assert_called_once_with(*args, binding)
        self.assertIs(operation.call_args.args[-1], binding)
        for index, value in enumerate(args):
            self.assertIs(operation.call_args.args[index], value)
        self.assertIs(actual, result)

    def test_edits_schema(self) -> None:
        self.routed(edits, "schema", ())

    def test_edits_validate_edits(self) -> None:
        self.routed(edits, "validate_edits", ([], object()))

    def test_edits_verified_parameters(self) -> None:
        self.routed(edits, "verified_parameters", (object(),))

    def test_edits_apply_edits(self) -> None:
        self.routed(edits, "apply_edits", (object(), object(), [], object()))

    def test_edits_validate_pair(self) -> None:
        self.routed(edits, "validate_pair", ({"held": 1},))

    def test_observations_schema(self) -> None:
        self.routed(observations, "schema", ())

    def test_observations_validate_observation(self) -> None:
        self.routed(observations, "validate_observation", ({"held": 1}, "attention"))

    def test_observations_validate_record(self) -> None:
        step = {"held": 1}
        self.routed(observations, "validate_record", (step, object(), 7))
        binding = object()
        with (
            patch.object(observations, "_bindings", return_value=binding),
            patch.object(
                observations.contract, "validate_record", return_value=None
            ) as operation,
        ):
            self.assertIsNone(observations.validate_record(step))
        operation.assert_called_once_with(step, None, None, binding)

    def test_observations_lens_record(self) -> None:
        self.routed(observations, "lens_record", (*[object() for _ in range(5)], 7, 9))

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
            with self.subTest(value=value):
                self.assertIs(observations._integer(value, 2, 7), expected)
        self.assertIs(observations._integer(True, 0, 2), False)

    def test_finite(self) -> None:
        for value, expected in (
            (0, True),
            (-2.5, True),
            (math.inf, False),
            (-math.inf, False),
            (math.nan, False),
            (True, False),
            (None, False),
            ("1", False),
        ):
            with self.subTest(value=value):
                self.assertIs(observations._finite(value), expected)

    def test_worker_loader(self) -> None:
        torch, model_factory, tokenizer_factory = object(), object(), object()
        transformers = SimpleNamespace(
            LlamaForCausalLM=model_factory, PreTrainedTokenizerFast=tokenizer_factory
        )
        directory, model, expected, parameters = (
            Path("/unused"),
            object(),
            (object(), object(), object()),
            {"held": "parameters"},
        )
        with (
            patch.dict(sys.modules, {"torch": torch, "transformers": transformers}),
            patch.object(worker, "bound_load_engine", return_value=expected) as load,
            patch.object(
                edits, "verified_parameters", return_value=parameters
            ) as verify,
        ):
            self.assertIs(worker.load_engine(directory), expected)
            load.assert_called_once()
            self.assertIs(load.call_args.args[0], directory)
            runtime = load.call_args.args[1]
            self.assertIs(runtime.torch, torch)
            self.assertIs(runtime.model_factory, model_factory)
            self.assertIs(runtime.tokenizer_factory, tokenizer_factory)
            self.assertIs(runtime.verified_parameters(model), parameters)
            verify.assert_called_once_with(model)

    def contract_graph(self, module: ModuleType) -> None:
        graph = module._contracts()
        value = graph.architecture
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
        self.assertIs(value.capture_sites, module.CAPTURE_SITES)
        for field in ("edits", "observations", "pairs", "sweeps"):
            self.assertIs(getattr(graph, field).architecture, value)

    def test_worker_contracts(self) -> None:
        self.contract_graph(worker)

    def test_live_contracts(self) -> None:
        self.contract_graph(live)

    def test_worker_compare(self) -> None:
        args = tuple(object() for _ in range(5))
        with patch.object(worker, "run_comparison", return_value=None) as run:
            self.assertIsNone(worker.compare(*args))
        run.assert_called_once()
        self.assertEqual(run.call_args.args[:5], args)
        binding = run.call_args.args[5]
        for field in ("apply_edits", "validate_edits", "verified_parameters"):
            self.assertIs(getattr(binding, field), getattr(edits, field))
        self.assertIs(binding.generate, worker.generate)
        self.assertIs(binding.paired_step, worker.paired_step)

    def test_worker_main(self) -> None:
        for request in (
            {"prompt": "x", "max_new_tokens": 1, "layer": 0},
            pair_tests.request(),
        ):
            stream = SimpleNamespace(
                readline=Mock(wraps=io.BytesIO(json.dumps(request).encode()).readline)
            )
            tokenizer = pair_tests.Tokenizer()
            torch = SimpleNamespace(__version__="held")
            model = SimpleNamespace(
                _atlas_verified_layout={"held": "layout"},
                config=SimpleNamespace(_attn_implementation="eager"),
            )
            with ExitStack() as stack:
                for obj, name, value in (
                    (worker.sys, "argv", ["worker", "/unused"]),
                    (worker.sys, "stdin", SimpleNamespace(buffer=stream)),
                ):
                    stack.enter_context(patch.object(obj, name, value))
                stack.enter_context(
                    patch.object(worker.os, "sched_getaffinity", return_value={0})
                )
                for obj, name in (
                    (worker.os, "sched_setaffinity"),
                    (worker.os, "nice"),
                    (worker, "configure_worker_limits"),
                    (live, "verify_model"),
                ):
                    stack.enter_context(patch.object(obj, name))
                stack.enter_context(
                    patch.object(
                        worker, "load_engine", return_value=(torch, tokenizer, model)
                    )
                )
                emit = stack.enter_context(patch.object(worker, "emit"))
                execute = stack.enter_context(patch.object(worker, "execute"))
                stack.enter_context(
                    patch.dict(
                        sys.modules,
                        {
                            "transformers": SimpleNamespace(__version__="held"),
                            "tokenizers": SimpleNamespace(
                                __version__="held",
                                Tokenizer=SimpleNamespace(
                                    from_file=lambda _: tokenizer
                                ),
                            ),
                            "safetensors": SimpleNamespace(__version__="held"),
                        },
                    )
                )
                worker.main()
            stream.readline.assert_called_once_with(8193)
            self.assertEqual(emit.call_args.args[0]["runtime"]["seed"], 0)
            execute.assert_called_once()
            context = execute.call_args.args[1]
            self.assertIs(
                context.verified_parameters,
                (
                    edits.verified_parameters
                    if request.get("mode") == "prompt_pair"
                    else None
                ),
            )


if __name__ == "__main__":
    unittest.main()
