"""Held validation/record bindings and one-call routing through service methods."""

from pathlib import Path
import sys
from types import ModuleType
from typing import Any
import unittest
from unittest.mock import patch

import inference_services as services
import inference_worker as worker


class ServiceContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.contracts = worker._contracts()
        self.step = {"held": "step"}
        self.request = {"held": "request"}
        self.preview = {"held": "preview"}
        self.plan = {"held": "plan"}

    def routed(
        self,
        method: str,
        module: ModuleType,
        operation: str,
        arguments: tuple[Any, ...],
        binding: str,
        *,
        keyword: bool = False,
        result: bool = True,
    ) -> None:
        expected = {"held": method}
        field = getattr(self.contracts, binding)
        with patch.object(module, operation, return_value=expected) as callback:
            actual = getattr(self.contracts, method)(*arguments)
        self.assertIs(actual, expected if result else None)
        if keyword:
            callback.assert_called_once_with(*arguments, bindings=field)
            self.assertIs(callback.call_args.kwargs["bindings"], field)
        else:
            callback.assert_called_once_with(*arguments, field)
            self.assertIs(callback.call_args.args[-1], field)
        for index, value in enumerate(arguments):
            self.assertIs(callback.call_args.args[index], value)

    def test_comparison_schema(self) -> None:
        self.routed("comparison_schema", services.edit, "schema", (), "edits")

    def test_observations_schema(self) -> None:
        self.routed(
            "observations_schema", services.observation, "schema", (), "observations"
        )

    def test_sweep_schema(self) -> None:
        self.routed(
            "sweep_schema", services.sweep, "schema", (), "sweeps", keyword=True
        )

    def test_validate_edits(self) -> None:
        self.routed(
            "validate_edits",
            services.edit,
            "validate_edits",
            ([self.step], object()),
            "edits",
        )

    def test_validate_pair(self) -> None:
        self.routed(
            "validate_pair",
            services.edit,
            "validate_pair",
            (self.step,),
            "edits",
            result=False,
        )

    def test_validate_observation(self) -> None:
        self.routed(
            "validate_observation",
            services.observation,
            "validate_observation",
            (self.step, "attention"),
            "observations",
        )

    def test_validate_record(self) -> None:
        self.routed(
            "validate_record",
            services.observation,
            "validate_record",
            (self.step, self.request, 7),
            "observations",
            result=False,
        )
        with patch.object(services.observation, "validate_record") as callback:
            self.assertIsNone(self.contracts.validate_record(self.step))
        callback.assert_called_once_with(
            self.step, None, None, self.contracts.observations
        )

    def test_lens_record(self) -> None:
        self.routed(
            "lens_record",
            services.observation,
            "lens_record",
            (object(), object(), object(), object(), object(), 7, 9),
            "observations",
        )

    def test_validate_request(self) -> None:
        self.routed(
            "validate_request",
            services.pair,
            "validate_request",
            (self.request,),
            "pairs",
            keyword=True,
        )

    def test_token_preview(self) -> None:
        self.routed(
            "token_preview",
            services.pair,
            "token_preview",
            (object(), ["A", "B"]),
            "pairs",
            keyword=True,
        )

    def test_validate_positions(self) -> None:
        self.routed(
            "validate_positions",
            services.pair,
            "validate_positions",
            (self.request, self.preview),
            "pairs",
            keyword=True,
            result=False,
        )

    def test_validate_preview(self) -> None:
        self.routed(
            "validate_preview",
            services.pair,
            "validate_preview",
            (self.preview,),
            "pairs",
            keyword=True,
            result=False,
        )

    def test_validate_pair_step(self) -> None:
        self.routed(
            "validate_pair_step",
            services.pair,
            "validate_step",
            (self.step, self.request),
            "pairs",
            keyword=True,
            result=False,
        )

    def test_build_plan(self) -> None:
        self.routed(
            "build_plan",
            services.sweep,
            "build_plan",
            (self.request, False),
            "sweeps",
            keyword=True,
        )
        expected = {"held": "default plan"}
        with patch.object(
            services.sweep, "build_plan", return_value=expected
        ) as callback:
            self.assertIs(self.contracts.build_plan(self.request), expected)
        callback.assert_called_once_with(
            self.request, True, bindings=self.contracts.sweeps
        )

    def test_validate_sweep_step(self) -> None:
        self.routed(
            "validate_sweep_step",
            services.sweep,
            "validate_step",
            (self.step, self.plan),
            "sweeps",
            keyword=True,
            result=False,
        )


if __name__ == "__main__":
    unittest.main()
