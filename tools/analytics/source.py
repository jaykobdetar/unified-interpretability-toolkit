"""Read-only bounded adapter for a host-validated BF16 catalog.

Construct Catalog only from Source's validated metadata, never an HTTP payload.
No directory discovery, source writes, whole tensor reads, or persistent cache.
"""

from dataclasses import dataclass
import math
import os
from pathlib import Path
import stat
import struct
import time

from .core import (
    MAX_AXIS,
    MAX_TENSORS,
    MAX_TOP,
    MAX_VALUES,
    analyze,
    digest,
    geometry,
    integer,
    statistics,
)


def fingerprint(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def optional_index_fingerprint(root):
    try:
        st = (Path(root) / "model.safetensors.index.json").lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(st.st_mode):
        raise ValueError("Source index must be a regular non-symlink file")
    return fingerprint(st)


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple
    shard: str
    byte_offset: int


class Catalog:
    def __init__(self, root, tensors, source_identity):
        self.root = Path(root).resolve(strict=True)
        if not source_identity or not isinstance(source_identity, str):
            raise ValueError("Validated host source identity required")
        if not 0 < len(tensors) <= MAX_TENSORS:
            raise ValueError(
                "Catalog must contain 1..512 tensors; never silently truncate catalog"
            )
        self.tensors = tuple(tensors)
        self.files = {}
        names = set()
        for tensor in self.tensors:
            if (
                not isinstance(tensor.name, str)
                or not tensor.name
                or len(tensor.name) > 512
                or tensor.name in names
            ):
                raise ValueError("Invalid or duplicate tensor name")
            names.add(tensor.name)
            if Path(tensor.shard).name != tensor.shard or not tensor.shard.endswith(
                ".safetensors"
            ):
                raise ValueError("Catalog shard must be a local safetensors basename")
            if len(tensor.shape) not in (1, 2):
                raise ValueError("Catalog adapter supports vectors and matrices only")
            for n in tensor.shape:
                integer(n, 1, 200000, "shape")
            integer(tensor.byte_offset, 0, 2**63 - 1, "byte offset")
            path = self.root / tensor.shard
            st = path.lstat()
            if not stat.S_ISREG(st.st_mode):
                raise ValueError("Source shard must be a regular non-symlink file")
            if tensor.byte_offset + 2 * math.prod(tensor.shape) > st.st_size:
                raise ValueError("Catalog tensor exceeds shard extent")
            self.files[tensor.shard] = fingerprint(st)
        self.index_fingerprint = optional_index_fingerprint(self.root)
        self.host_identity = source_identity
        self.identity = digest(
            {
                "host_identity": source_identity,
                "files": self.files,
                "index": self.index_fingerprint,
                "tensors": [vars(t) for t in self.tensors],
            }
        )

    def verify(self):
        if optional_index_fingerprint(self.root) != self.index_fingerprint:
            raise ValueError(
                "Source index identity changed; discard analytics and caches"
            )
        for name, expected in self.files.items():
            if fingerprint((self.root / name).lstat()) != expected:
                raise ValueError(
                    "Source identity changed; discard analytics and caches"
                )

    def read(self, tensor, region):
        if tensor not in self.tensors:
            raise ValueError("Tensor not in validated catalog")
        r, c, h, w = geometry(tensor.shape, region)
        full_cols = tensor.shape[-1]
        self.verify()
        fd = os.open(
            self.root / tensor.shard, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        )
        try:
            if fingerprint(os.fstat(fd)) != self.files[tensor.shard]:
                raise ValueError("Source changed before read")
            values = []
            for row in range(r, r + h):
                raw = os.pread(
                    fd, 2 * w, tensor.byte_offset + 2 * (row * full_cols + c)
                )
                if len(raw) != 2 * w:
                    raise ValueError("Short source read")
                values.extend(
                    struct.unpack("<f", struct.pack("<I", word << 16))[0]
                    for (word,) in struct.iter_unpack("<H", raw)
                )
            if fingerprint(os.fstat(fd)) != self.files[tensor.shard]:
                raise ValueError("Source changed during read")
            self.verify()
            return values
        finally:
            os.close(fd)

    def region(self, name, region, **options):
        tensor = next((t for t in self.tensors if t.name == name), None)
        if tensor is None:
            raise ValueError("Unknown tensor")
        return analyze(
            self.read(tensor, region),
            source_identity=self.identity,
            tensor=tensor.name,
            shape=list(tensor.shape),
            region=region,
            **options,
        )

    def validate_cached(self, result):
        self.verify()
        if result.get("source_identity") != self.identity:
            raise ValueError("Cached analytics belong to another source")
        # Caller must also match the exact request cache key. This validates source only.
        return result

    def model_outliers(self, top=16, value_budget=MAX_VALUES, seed=1):
        """Deterministic catalog-order prefix windows, not a representative sample.

        Full model is possible only for catalogs whose entire values fit the cap.
        Budget/time exclusions are retained for every tensor. Row/column means
        use visited cells only; no cross-tensor normalization is implied.
        """
        integer(top, 1, MAX_TOP, "top")
        integer(value_budget, 1, MAX_VALUES, "value budget")
        integer(seed, 0, 2**32 - 1, "seed")
        self.verify()
        deadline = time.monotonic() + 2.0
        remaining = value_budget
        coverage = []
        results = {
            side: {"values": [], "rows": [], "columns": []}
            for side in ("original", "shuffled")
        }
        visited = 0
        for tensor in self.tensors:
            total = math.prod(tensor.shape)
            row_count, col_count = (
                (1, tensor.shape[0]) if len(tensor.shape) == 1 else tensor.shape
            )
            entry = {
                "tensor": tensor.name,
                "total_values": total,
                "visited_values": 0,
                "full_tensor": False,
                "region": None,
            }
            if remaining == 0 or time.monotonic() >= deadline:
                entry["excluded_reason"] = (
                    "value budget exhausted"
                    if remaining == 0
                    else "time budget exhausted"
                )
                coverage.append(entry)
                continue
            w = min(col_count, MAX_AXIS, remaining)
            h = min(row_count, MAX_AXIS, remaining // w)
            region = {"row": 0, "col": 0, "rows": h, "cols": w}
            values = self.read(tensor, region)
            pair = analyze(
                values,
                source_identity=self.identity,
                tensor=tensor.name,
                shape=list(tensor.shape),
                region=region,
                top=top,
                seed=seed,
            )
            used = len(values)
            remaining -= used
            visited += used
            entry.update(
                visited_values=used,
                full_tensor=used == total,
                region=region,
                excluded_reason=(
                    None
                    if used == total
                    else "bounded prefix window; remaining values excluded"
                ),
            )
            coverage.append(entry)
            for side in ("original", "shuffled"):
                s = pair[side]
                for kind, items in (
                    ("values", s["top_values"]),
                    ("rows", s["top_rows"]),
                    ("columns", s["top_columns"]),
                ):
                    for item in items:
                        results[side][kind].append(
                            {
                                **item,
                                "tensor": tensor.name,
                                "region": region,
                                "full_axis": (
                                    (
                                        w == col_count
                                        if kind == "rows"
                                        else h == row_count
                                    )
                                    if kind != "values"
                                    else None
                                ),
                                "score_scope": (
                                    "absolute raw value"
                                    if kind == "values"
                                    else "mean absolute over visited cells"
                                ),
                            }
                        )
                    score = "abs" if kind == "values" else "mean_abs"
                    results[side][kind] = sorted(
                        results[side][kind],
                        key=lambda x: (-x[score], x["tensor"], x["index"]),
                    )[:top]
        self.verify()
        total = sum(math.prod(t.shape) for t in self.tensors)
        return {
            "schema": "weight-atlas.model-outliers.v1",
            "source_identity": self.identity,
            "coverage": {
                "total_tensors": len(self.tensors),
                "visited_tensors": sum(x["visited_values"] > 0 for x in coverage),
                "total_values": total,
                "visited_values": visited,
                "full_model": visited == total,
                "tensors": coverage,
                "selection": "deterministic catalog-order prefix windows; not random sampling",
            },
            "rankings": results,
            "control": {
                "seed": seed,
                "kind": "independent same-region exact-multiset permutation within each visited tensor window",
                "statistics": "refit separately; raw magnitudes; no fitted calibration",
            },
            "warning": "Raw magnitudes across tensors are not standardized significance scores",
        }


def read_layout_evidence(config_path, implementation_path):
    """Hash bounded local evidence files; a hash alone does not review semantics.

    The returned digest pair must match an application-owned reviewed profile.
    Re-read on each request (including cache hits) if head labels are enabled.
    """
    import hashlib
    import json

    def read(path):
        path = Path(path)
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_size > 1024 * 1024:
            raise ValueError("Layout evidence must be a regular file of at most 1 MiB")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            if fingerprint(os.fstat(fd)) != fingerprint(before):
                raise ValueError("Evidence identity changed before read")
            raw = os.read(fd, 1024 * 1024 + 1)
            if (
                len(raw) != before.st_size
                or fingerprint(os.fstat(fd)) != fingerprint(before)
                or fingerprint(path.lstat()) != fingerprint(before)
            ):
                raise ValueError("Evidence identity changed during read")
            return raw
        finally:
            os.close(fd)

    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("Duplicate configuration key")
            out[key] = value
        return out

    config_raw = read(config_path)
    config = json.loads(config_raw, object_pairs_hook=unique)
    evidence = {
        "config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "config_canonical_sha256": digest(config),
        "implementation_sha256": hashlib.sha256(read(implementation_path)).hexdigest(),
        "review_requirement": "Exact digest pair must match application-owned reviewed layout profile",
    }
    return config, evidence
