"""Current pair/sweep facades forward exact references, defaults and results."""

from pathlib import Path
import sys
from types import ModuleType
from typing import Any
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import inference_prompt_pair as pair
import inference_sweep as sweep


class RoutingContracts(unittest.TestCase):
    def routed(self, module: ModuleType, name: str, arguments: tuple[Any, ...]) -> None:
        binding, result = object(), object()
        with (
            patch.object(module, "_bindings", return_value=binding) as factory,
            patch.object(module.contract, name, return_value=result) as operation,
        ):
            actual = getattr(module, name)(*arguments)
        self.assertIs(actual, result)
        factory.assert_called_once_with()
        operation.assert_called_once_with(*arguments, bindings=binding)
        self.assertIs(operation.call_args.kwargs["bindings"], binding)
        for index, value in enumerate(arguments):
            self.assertIs(operation.call_args.args[index], value)

    def test_pair_validate_request(self) -> None:
        self.routed(pair, "validate_request", ({"held": 1},))

    def test_pair_digest_for(self) -> None:
        self.routed(pair, "digest_for", (["A", "B"], [[1], [2]]))

    def test_pair_token_preview(self) -> None:
        self.routed(pair, "token_preview", (object(), ["A", "B"]))

    def test_pair_validate_preview(self) -> None:
        self.routed(pair, "validate_preview", ({"held": 1},))

    def test_pair_validate_positions(self) -> None:
        self.routed(pair, "validate_positions", ({"held": 1}, {"held": 2}))

    def test_pair_difference(self) -> None:
        self.routed(pair, "difference", ([1.0], [2.0]))

    def test_pair_validate_step(self) -> None:
        self.routed(pair, "validate_step", ({"held": 1}, {"held": 2}))

    def test_sweep_schema(self) -> None:
        self.routed(sweep, "schema", ())

    def test_sweep_target(self) -> None:
        self.routed(sweep, "_target", ({"held": 1},))

    def test_sweep_rows(self) -> None:
        self.routed(sweep, "_rows", ({"held": 1},))

    def test_sweep_edits(self) -> None:
        self.routed(sweep, "_edits", ({"held": 1}, "zero", 0.5))

    def test_sweep_build_plan(self) -> None:
        data = {"held": 1}
        self.routed(sweep, "build_plan", (data, False))
        binding, result = object(), object()
        with (
            patch.object(sweep, "_bindings", return_value=binding),
            patch.object(
                sweep.contract, "build_plan", return_value=result
            ) as operation,
        ):
            self.assertIs(sweep.build_plan(data), result)
        operation.assert_called_once_with(data, True, bindings=binding)

    def test_sweep_validate_step(self) -> None:
        self.routed(sweep, "validate_step", ({"held": 1}, {"held": 2}))


if __name__ == "__main__":
    unittest.main()
