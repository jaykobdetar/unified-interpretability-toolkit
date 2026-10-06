"""Pure contract operations sharing one held architecture and data boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atlas_host.inference_geometry import Architecture
from atlas_host import inference_edit_contract as edit
from atlas_host import inference_observation_contract as observation
import inference_pair_contract as pair
import inference_sweep_contract as sweep


@dataclass(frozen=True)
class InferenceContracts:
    architecture: Architecture
    edits: edit.EditBindings
    observations: observation.ObservationBindings
    pairs: pair.PairBindings
    sweeps: sweep.SweepBindings

    def comparison_schema(self) -> dict[str, Any]:
        return edit.schema(self.edits)

    def observations_schema(self) -> dict[str, Any]:
        return observation.schema(self.observations)

    def sweep_schema(self) -> dict[str, Any]:
        return sweep.schema(bindings=self.sweeps)

    def validate_edits(self, values: Any, source: Any) -> list[dict[str, Any]]:
        return edit.validate_edits(values, source, self.edits)

    def validate_pair(self, step: dict[str, Any]) -> None:
        edit.validate_pair(step, self.edits)

    def validate_observation(self, value: Any, site: str) -> dict[str, Any]:
        return observation.validate_observation(value, site, self.observations)

    def validate_record(
        self, step: dict[str, Any], selected: Any = None, layer: Any = None
    ) -> None:
        observation.validate_record(step, selected, layer, self.observations)

    def lens_record(
        self,
        torch: Any,
        tokenizer: Any,
        residual: Any,
        model: Any,
        final_logits: Any,
        layer: int,
        position: int,
    ) -> dict[str, Any]:
        return observation.lens_record(
            torch,
            tokenizer,
            residual,
            model,
            final_logits,
            layer,
            position,
            self.observations,
        )

    def validate_request(self, data: dict[str, Any]) -> dict[str, Any]:
        return pair.validate_request(data, bindings=self.pairs)

    def token_preview(self, tokenizer: Any, prompts: Any) -> dict[str, Any]:
        return pair.token_preview(tokenizer, prompts, bindings=self.pairs)

    def validate_positions(
        self, request: dict[str, Any], preview: dict[str, Any]
    ) -> None:
        pair.validate_positions(request, preview, bindings=self.pairs)

    def validate_preview(self, value: Any) -> None:
        pair.validate_preview(value, bindings=self.pairs)

    def validate_pair_step(self, step: dict[str, Any], request: dict[str, Any]) -> None:
        pair.validate_step(step, request, bindings=self.pairs)

    def build_plan(self, data: Any, require_digest: bool = True) -> dict[str, Any]:
        return sweep.build_plan(data, require_digest, bindings=self.sweeps)

    def validate_sweep_step(self, step: dict[str, Any], plan: dict[str, Any]) -> None:
        sweep.validate_step(step, plan, bindings=self.sweeps)
