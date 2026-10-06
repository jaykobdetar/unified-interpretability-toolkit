"""Versioned, bounded window SVD summaries. NumPy is supplied only by the owned worker."""

import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from typing import Any, NotRequired, TypedDict, TypeVar

from .core import checked_values, geometry, permutation

SCHEMA = "weight-atlas.svd-window-summary.v2"
ALGORITHM = "uncentered-independent-svd-xorshift32-v1"
MAX_AXIS = 128
MAX_VALUES = 16384
MAX_BODY = 63488
PREVIEW_AXIS = 16
MAX_BF16 = 3.3895313892515355e38


class DisplaySlice(TypedDict):
    leading_indices: list[int]
    display_axes: list[int]


class SourceBinding(TypedDict):
    version: int
    model_identity: str
    source_identity: str
    tensor: int
    name: str
    dtype: str
    shape: Sequence[int]
    rows: int
    cols: int
    slice: DisplaySlice


class Fit(TypedDict):
    singular_values: list[float]
    energy_fractions: list[float | None]
    frobenius_energy: float
    rank_one_residual_energy: float
    rank_one_residual_energy_fraction: float | None
    zero_energy: bool
    rank_one_residual_preview: list[float]


class SummaryBinding(TypedDict):
    results: NotRequired[dict[str, Fit]]
    schema: str
    algorithm: str
    source_binding: SourceBinding
    source_identity: str
    model_identity: str
    tensor_id: int
    tensor: str
    dtype: str
    shape: Sequence[int]
    region: Mapping[str, int]
    seed: int


class Coverage(TypedDict):
    visited_values: int
    total_tensor_values: int
    tensor_fraction: float
    full_tensor: bool
    full_model: bool


class Preview(TypedDict):
    origin: list[int]
    shape: list[int]
    displayed_values: int
    omitted_values: int
    positions: list[int]


class Control(TypedDict):
    kind: str
    seed: int
    effective_seed: int
    permutation_sha256: str
    digest_encoding: str
    preview_position_to_source: list[int]
    omitted_mapping_entries: int


class Summary(SummaryBinding):
    cache_key: str
    centered: bool
    source_validation: str
    coverage: Coverage
    preview: Preview
    control: Control


_Report = TypeVar("_Report", bound=Mapping[str, Any])


def encoded(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, allow_nan=False, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def check_window(shape: Sequence[int], region: Mapping[str, int]) -> None:
    geometry(shape, region)
    if (
        region["rows"] > MAX_AXIS
        or region["cols"] > MAX_AXIS
        or region["rows"] * region["cols"] > MAX_VALUES
    ):
        raise ValueError(
            "SVD summary accepts at most 128 × 128 / 16384 native BF16 values"
        )


def binding(
    model: Mapping[str, Any],
    tensor: Mapping[str, Any],
    region: Mapping[str, int],
    seed: int,
) -> SummaryBinding:
    check_window(tensor["shape"], region)
    for key in ("source_identity", "model_identity"):
        value = model.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError(
                "SVD summary requires validated source and revision identities"
            )
    revision = model.get("revision")
    if not isinstance(revision, str) or not 1 <= len(revision.encode()) <= 256:
        raise ValueError("SVD summary requires a bounded renderer revision assertion")
    if model["model_identity"] != digest(
        ["weight-atlas-model-v1", model["source_identity"], revision]
    ):
        raise ValueError("Renderer revision binding differs from source identity")
    if (
        not isinstance(tensor["name"], str)
        or not 1 <= len(tensor["name"].encode()) <= 1024
        or tensor["dtype"] != "BF16"
    ):
        raise ValueError("SVD summary requires a bounded native BF16 tensor name")
    shape = tensor["shape"]
    source_binding: SourceBinding = {
        "version": 2,
        "model_identity": model["model_identity"],
        "source_identity": model["source_identity"],
        "tensor": tensor["id"],
        "name": tensor["name"],
        "dtype": "BF16",
        "shape": shape,
        "rows": 1 if len(shape) == 1 else shape[0],
        "cols": shape[-1],
        "slice": {
            "leading_indices": [],
            "display_axes": [0] if len(shape) == 1 else [0, 1],
        },
    }
    return {
        "schema": SCHEMA,
        "algorithm": ALGORITHM,
        "source_binding": source_binding,
        "source_identity": model["source_identity"],
        "model_identity": model["model_identity"],
        "tensor_id": tensor["id"],
        "tensor": tensor["name"],
        "dtype": "BF16",
        "shape": tensor["shape"],
        "region": region,
        "seed": seed,
    }


def skeleton(
    model: Mapping[str, Any],
    tensor: Mapping[str, Any],
    region: Mapping[str, int],
    seed: int,
) -> Summary:
    bound = binding(model, tensor, region, seed)
    rows, cols = region["rows"], region["cols"]
    count, total = rows * cols, math.prod(tensor["shape"])
    pr, pc = min(rows, PREVIEW_AXIS), min(cols, PREVIEW_AXIS)
    positions = [r * cols + c for r in range(pr) for c in range(pc)]
    order = permutation(count, seed)
    return {
        **bound,
        "cache_key": digest(bound),
        "centered": False,
        "source_validation": "Complete headers/index and file fingerprints; no fresh full-content hashes",
        "coverage": {
            "visited_values": count,
            "total_tensor_values": total,
            "tensor_fraction": count / total,
            "full_tensor": count == total,
            "full_model": False,
        },
        "preview": {
            "origin": [0, 0],
            "shape": [pr, pc],
            "displayed_values": pr * pc,
            "omitted_values": count - pr * pc,
            "positions": positions,
        },
        "control": {
            "kind": "same-window exact multiset; independently fitted",
            "seed": seed,
            "effective_seed": seed or 0x6D2B79F5,
            "permutation_sha256": hashlib.sha256(
                struct.pack(f"<{count}I", *order)
            ).hexdigest(),
            "digest_encoding": "unsigned little-endian uint32 position-to-source indices",
            "preview_position_to_source": [order[i] for i in positions],
            "omitted_mapping_entries": count - pr * pc,
        },
    }


def compute_with_numpy(
    np: Any,
    values: Sequence[float],
    *,
    model: Mapping[str, Any],
    tensor: Mapping[str, Any],
    region: Mapping[str, int],
    seed: int,
) -> Summary:
    checked_values(values, tensor["shape"], region)
    report = skeleton(model, tensor, region, seed)
    order = permutation(len(values), seed)
    results: dict[str, Fit] = {}
    for label, data in (("original", values), ("shuffled", [values[i] for i in order])):
        matrix = np.asarray(data, dtype=np.float64).reshape(
            region["rows"], region["cols"]
        )
        u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
        residual = matrix - singular[0] * np.outer(u[:, 0], vt[0, :])
        spectrum = [float(x) for x in singular]
        total = math.fsum(float(x) * float(x) for x in data)
        residual_energy = math.fsum(float(x) * float(x) for x in residual.ravel())
        results[label] = {
            "singular_values": spectrum,
            "energy_fractions": [x * x / total if total else None for x in spectrum],
            "frobenius_energy": total,
            "rank_one_residual_energy": residual_energy,
            "rank_one_residual_energy_fraction": (
                residual_energy / total if total else None
            ),
            "zero_energy": total == 0,
            "rank_one_residual_preview": [
                float(residual.ravel()[i]) for i in report["preview"]["positions"]
            ],
        }
    report["results"] = results
    validate(report, model=model, tensor=tensor, region=region, seed=seed)
    return report


def validate(
    report: _Report,
    *,
    model: Mapping[str, Any],
    tensor: Mapping[str, Any],
    region: Mapping[str, int],
    seed: int,
) -> _Report:
    """Coordinator verifies exact binding, deterministic map, closed shape and body budget."""
    if len(json.dumps(report, allow_nan=False).encode()) > MAX_BODY:
        raise ValueError("SVD summary exceeds 63488-byte body cap")
    expected = skeleton(model, tensor, region, seed)
    if (
        not isinstance(report, dict)
        or set(report) != set(expected) | {"results"}
        or encoded({k: report.get(k) for k in expected}) != encoded(expected)
    ):
        raise ValueError("SVD summary contract or source binding differs")
    results = report["results"]
    if not isinstance(results, dict) or set(results) != {"original", "shuffled"}:
        raise ValueError("SVD summary requires independently fitted paired results")
    fields = {
        "singular_values",
        "energy_fractions",
        "frobenius_energy",
        "rank_one_residual_energy",
        "rank_one_residual_energy_fraction",
        "zero_energy",
        "rank_one_residual_preview",
    }
    n, preview = (
        min(region["rows"], region["cols"]),
        expected["preview"]["displayed_values"],
    )
    max_energy = expected["coverage"]["visited_values"] * MAX_BF16**2
    max_singular = math.sqrt(max_energy)

    def finite(x: Any) -> bool:
        try:
            return type(x) in (int, float) and math.isfinite(x)
        except OverflowError:
            return False

    for result in results.values():
        if (
            not isinstance(result, dict)
            or set(result) != fields
            or type(result["zero_energy"]) is not bool
        ):
            raise ValueError("Invalid closed SVD summary result")
        s, fractions, residual = (
            result[k]
            for k in (
                "singular_values",
                "energy_fractions",
                "rank_one_residual_preview",
            )
        )
        if (
            not all(isinstance(a, list) for a in (s, fractions, residual))
            or len(s) != n
            or len(fractions) != n
            or len(residual) != preview
        ):
            raise ValueError("SVD summary spectrum or preview length differs")
        if (
            any(not finite(x) or not 0 <= x <= max_singular * (1 + 1e-10) for x in s)
            or any(s[i] < s[i + 1] for i in range(n - 1))
            or any(not finite(x) or abs(x) > 2 * max_singular for x in residual)
        ):
            raise ValueError("SVD summary contains invalid numeric arrays")
        energy, re, ratio = (
            result[k]
            for k in (
                "frobenius_energy",
                "rank_one_residual_energy",
                "rank_one_residual_energy_fraction",
            )
        )
        if (
            not finite(energy)
            or not finite(re)
            or not 0 <= energy <= max_energy
            or not 0 <= re <= energy * (1 + 1e-10)
            or result["zero_energy"] != (energy == 0)
        ):
            raise ValueError("SVD summary contains invalid energy")
        if energy == 0:
            if (
                ratio is not None
                or any(x is not None for x in fractions)
                or re != 0
                or any(s)
                or any(residual)
            ):
                raise ValueError("Zero-energy SVD summary differs")
        else:
            if (
                not finite(ratio)
                or not 0 <= ratio <= 1 + 1e-10
                or any(not finite(x) or not 0 <= x <= 1 + 1e-10 for x in fractions)
            ):
                raise ValueError("Invalid SVD energy fractions")
            if not math.isclose(
                ratio, re / energy, rel_tol=1e-10, abs_tol=1e-12
            ) or not math.isclose(
                math.fsum(fractions), 1, rel_tol=1e-10, abs_tol=1e-12
            ):
                raise ValueError("SVD energy accounting differs")
            if any(
                not math.isclose(f, x * x / energy, rel_tol=1e-10, abs_tol=1e-12)
                for x, f in zip(s, fractions)
            ):
                raise ValueError("SVD spectrum energy differs")
    if (
        results["original"]["frobenius_energy"]
        != results["shuffled"]["frobenius_energy"]
    ):
        raise ValueError("Shuffled control energy differs from exact multiset")
    return report
