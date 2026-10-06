"""Inert worker forwarding, output encoding and loader callback contracts."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import inference_worker as worker


class WorkerPackageTests(unittest.TestCase):
    def test_emit(self):
        with patch("builtins.print") as output:
            worker.emit({"first": 1, "second": [2, 3]})
            output.assert_called_once_with('{"first":1,"second":[2,3]}', flush=True)
        with patch("builtins.print") as output:
            with self.assertRaises(ValueError):
                worker.emit({"nonfinite": float("nan")})
            output.assert_not_called()

    def test_loader_layout_callback(self):
        torch, model_factory, tokenizer_factory = object(), object(), object()
        transformer_module = SimpleNamespace(
            LlamaForCausalLM=model_factory, PreTrainedTokenizerFast=tokenizer_factory
        )
        directory, model = Path("/unused"), object()
        layout = {"held": "layout"}
        engine = (torch, object(), model)
        with (
            patch.dict(
                sys.modules, {"torch": torch, "transformers": transformer_module}
            ),
            patch.object(worker, "bound_load_engine", return_value=engine) as load,
            patch.object(
                worker, "verify_attention_layout", return_value=layout
            ) as verify,
        ):
            self.assertIs(worker.load_engine(directory), engine)
            load.assert_called_once()
            self.assertIs(load.call_args.args[0], directory)
            runtime = load.call_args.args[1]
            self.assertIs(runtime.verify_attention_layout(model), layout)
            verify.assert_called_once_with(model)

    def test_generate(self):
        graph = worker._contracts()
        args = tuple(object() for _ in range(6))
        record, scores, observation, expected = (
            object(),
            [object()],
            {"held": "observation"},
            object(),
        )
        with (
            patch.object(worker, "_contracts", return_value=graph) as contracts,
            patch.object(worker, "run_generation", return_value=expected) as run,
        ):
            self.assertIs(
                worker.generate(
                    *args,
                    record=record,
                    capture_scores=scores,
                    activation_site="mlp",
                    observation=observation,
                ),
                expected,
            )
            contracts.assert_called_once_with()
            run.assert_called_once()
            actual = run.call_args.args
            for value, held in zip(actual[:6], args, strict=True):
                self.assertIs(value, held)
            self.assertIs(actual[6], record)
            binding = actual[7]
            self.assertEqual(binding.max_prompt, 128)
            self.assertEqual(binding.max_new, 32)
            self.assertEqual(
                (
                    binding.width,
                    binding.layers,
                    binding.heads,
                    binding.kv_heads,
                    binding.head_dim,
                    binding.vocab,
                ),
                (576, 30, 9, 3, 64, 49152),
            )
            self.assertIs(binding.capture_sites, graph.architecture.capture_sites)
            self.assertEqual(binding.attention_semantics, worker.ATTENTION_SEMANTICS)
            for name in ("validate_observation", "validate_record", "lens_record"):
                self.assertEqual(getattr(binding, name), getattr(graph, name))
            self.assertIs(run.call_args.kwargs["capture_scores"], scores)
            self.assertIs(run.call_args.kwargs["observation"], observation)
            self.assertEqual(run.call_args.kwargs["activation_site"], "mlp")
        with patch.object(worker, "run_generation", return_value=expected) as run:
            self.assertIs(worker.generate(*args), expected)
            self.assertIs(run.call_args.args[6], worker.emit)
            self.assertEqual(
                run.call_args.kwargs,
                {
                    "capture_scores": None,
                    "activation_site": "block",
                    "observation": None,
                },
            )

    def test_valid_worker_limits(self):
        for cpu, inherited in ((1, (-1, -1)), (90, (-1, -1)), (7, (2, 4))):
            with (
                self.subTest(cpu=cpu, inherited=inherited),
                patch.object(worker.resource, "getrlimit", return_value=inherited),
                patch.object(worker.resource, "setrlimit") as apply,
            ):
                try:
                    worker.configure_worker_limits(cpu)
                except Exception as exc:
                    self.fail("admitted worker limit must complete: " + repr(exc))
                address = 3 * 1024**3 if inherited[0] == -1 else 2
                seconds = cpu if inherited[0] == -1 else 2
                self.assertEqual(
                    [call.args for call in apply.call_args_list],
                    [
                        (worker.resource.RLIMIT_AS, (address, address)),
                        (worker.resource.RLIMIT_CPU, (seconds, seconds)),
                    ],
                )

    def test_paired_step(self):
        args = (3, [], [], [object()], [object()], object())
        expected = {"held": "comparison packet"}
        with patch.object(worker, "comparison_step", return_value=expected) as step:
            self.assertIs(worker.paired_step(*args), expected)
            step.assert_called_once_with(*args)
            for actual, held in zip(step.call_args.args, args, strict=True):
                self.assertIs(actual, held)


if __name__ == "__main__":
    unittest.main()
