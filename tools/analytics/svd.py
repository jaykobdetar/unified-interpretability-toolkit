"""Optional, isolated local NumPy SVD. Never invoked implicitly by core/source.

Integration must own the heavy slot. A separate interpreter with an existing
NumPy installation is required. No installs, downloads, models, or source access.
"""

import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TYPE_CHECKING, TypedDict

from .core import Unavailable, checked_values, permutation
from atlas_host.memory import available_bytes as _available_bytes
from .runtime import configure_svd_worker as _configure_worker

MAX_SVD_AXIS = 64
MAX_SVD_VALUES = 4096
TIMEOUT_SECONDS = 5


class Fit(TypedDict):
    singular_values: list[float]
    energy_fractions: list[float | None]
    rank_one_residual_energy_fraction: float | None
    rank_one_residual: list[float]
    zero_energy: bool


class Available(TypedDict):
    available: Literal[True]
    scope: str
    region: Mapping[str, int]
    seed: int
    control: str
    centered: bool
    results: Any


def compute_with_numpy(
    np: Any, values: Sequence[float], rows: int, cols: int, seed: int
) -> dict[str, Fit]:
    """Called only inside a resource-limited child; separately fits both matrices."""
    results: dict[str, Fit] = {}
    order = permutation(len(values), seed)
    for label, data in [("original", values), ("shuffled", [values[i] for i in order])]:
        matrix = np.asarray(data, dtype=np.float64).reshape(rows, cols)
        u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
        energy = singular * singular
        total = float(np.sum(energy))
        residual = matrix - singular[0] * np.outer(u[:, 0], vt[0, :])
        ratio = float(np.sum(residual * residual)) / total if total else None
        results[label] = {
            "singular_values": singular.tolist(),
            "energy_fractions": (
                (energy / total).tolist() if total else [None] * len(singular)
            ),
            "rank_one_residual_energy_fraction": ratio,
            "rank_one_residual": residual.ravel().tolist(),
            "zero_energy": total == 0,
        }
    return results


def run(
    values: Sequence[float],
    shape: Sequence[int],
    region: Mapping[str, int],
    seed: int = 1,
    python: str | os.PathLike[str] = sys.executable,
) -> Available | Unavailable:
    checked_values(values, shape, region)
    h, w = region["rows"], region["cols"]
    if h > MAX_SVD_AXIS or w > MAX_SVD_AXIS or len(values) > MAX_SVD_VALUES:
        return {
            "available": False,
            "reason": "Excluded: SVD cap is 64 by 64 / 4096 values; select a bounded native window",
        }
    permutation(0, seed)  # Validate before spawning.
    mem = _available_bytes()
    if mem < 3 * 1024**3 + 768 * 1024**2 or shutil.disk_usage(".").free < 25 * 1024**3:
        return {
            "available": False,
            "reason": "Paused by existing memory/disk reserve gates",
        }
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    request = json.dumps(
        {"values": values, "rows": h, "cols": w, "seed": seed}, allow_nan=False
    )
    try:
        proc = subprocess.run(
            [python, "-B", "-m", "analytics.svd", "--worker"],
            input=request,
            cwd=Path(__file__).resolve().parent.parent,
            env=env,
            text=True,
            capture_output=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return {
            "available": False,
            "reason": "Excluded: isolated SVD exceeded 5 second wall-time cap",
        }
    except OSError as exc:
        return {
            "available": False,
            "reason": f"Optional local Python unavailable: {type(exc).__name__}",
        }
    if proc.returncode:
        return {
            "available": False,
            "reason": "Optional local NumPy worker failed or is unavailable",
        }
    result = json.loads(proc.stdout)
    return {
        "available": True,
        "scope": (
            "full tensor"
            if len(values) == math.prod(shape)
            else "selected native window only"
        ),
        "region": region,
        "seed": seed,
        "control": "exact multiset; each SVD fitted separately",
        "centered": False,
        "results": result,
    }


def worker() -> None:
    _configure_worker()
    raw = sys.stdin.read(200001)
    if len(raw) > 200000:
        raise ValueError("Worker input cap exceeded")
    data = json.loads(raw)
    rows, cols = data["rows"], data["cols"]
    from .core import integer

    integer(rows, 1, MAX_SVD_AXIS, "rows")
    integer(cols, 1, MAX_SVD_AXIS, "cols")
    checked_values(
        data["values"], [rows, cols], {"row": 0, "col": 0, "rows": rows, "cols": cols}
    )
    if TYPE_CHECKING:
        # The actual optional numerical module remains an opaque external boundary.
        np: Any
    else:
        import numpy as np

    print(
        json.dumps(
            compute_with_numpy(np, data["values"], rows, cols, data["seed"]),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("Use the guarded run() API under the serialized heavy slot")
    worker()
