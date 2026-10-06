"""Source-only dense static-view policy; no HTTP, ML, launch or registry writes.

Catalog is metadata-only. Explicit activation prepares bounded config/headers;
readiness additionally requires a complete trusted Rust catalog and live reader
attestation from the parent-owned host. Nothing grants inference permission.
"""

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import struct

from .common import canonical, digest, fields, identity, integer, require
from .registry import fingerprint
from .inference_model_descriptor import MAX_CONFIG_BYTES, pinned_descriptor

MAX_HEADER_BYTES = 2 * 1024**2
MAX_ACTIVATION_METADATA = 8 * 1024**2
DENSE_BYTES = {"BF16": 2, "F16": 2, "F32": 4}
DISPLAY_AXIS_LIMIT = 200000
PUBLIC_MODEL_KEYS = {
    "api_version",
    "extensions",
    "backend",
    "revision",
    "representation",
    "parameter_count",
    "global_max",
    "calibration_complete",
    "source_bytes",
    "header_bytes_read",
    "source_identity",
    "model_identity",
    "catalog",
    "rules",
    "coverage",
    "render_semantics",
}
PUBLIC_TENSOR_KEYS = {
    "id",
    "name",
    "shape",
    "rows",
    "cols",
    "count",
    "dtype",
    "element_bytes",
    "available",
    "unavailable_reason",
    "shard",
    "shard_id",
    "byte_offset",
    "max_level",
    "min_level",
    "calibration_complete",
    "slice_required",
    "display_axes",
    "max_abs",
    "median_nonzero_abs",
    "q99",
    "quantile_order_statistics",
    "quantile_interpolation",
    "robust_clipped_count",
    "rule_status",
    "exact_zero_count",
    "unique_bit_patterns",
    "calibration_method",
}


def _json(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "Duplicate static metadata field")
            value[key] = item
        return value

    try:
        return json.loads(
            raw,
            object_pairs_hook=unique,
            parse_constant=lambda _: (_ for _ in ()).throw(
                ValueError("Nonfinite static metadata")
            ),
        )
    except (UnicodeError, RecursionError) as error:
        raise ValueError("Invalid bounded static metadata") from error


def _inventory(entry):
    """Bounded names only; no source file contents or recursive traversal."""
    expected = {item["name"] for item in entry["manifest"]["files"]}
    try:
        root = Path(entry["root"])
        if str(root.resolve(strict=True)) != entry["root"] or not root.is_dir():
            return False
        observed = set()
        with os.scandir(root) as source:
            for item in source:
                if item.name not in expected or item.name in observed:
                    return False
                observed.add(item.name)
                if len(observed) > 80:
                    return False
        return observed == expected
    except OSError:
        return False


def _current(registry, entry):
    require(
        entry["enabled"]
        and registry.owner_receipt(entry["model_id"]) == entry
        and registry._unchanged(entry)
        and _inventory(entry),
        "Static owner receipt/source changed",
    )


def _read(entry, name, maximum, budget, *, header=False):
    """Header/auxiliary bytes only, under saved receipt and current fingerprints."""
    fd = os.open(
        Path(entry["root"]) / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    )
    with os.fdopen(fd, "rb", buffering=0) as source:
        saved = entry["fingerprints"][name]
        require(
            fingerprint(os.fstat(source.fileno())) == saved,
            "Static file fingerprint changed",
        )
        if header:
            prefix = source.read(8)
            require(len(prefix) == 8, "Missing static safetensors prefix")
            size = struct.unpack("<Q", prefix)[0]
            integer(size, 2, min(maximum, saved["bytes"] - 8))
        else:
            size = integer(saved["bytes"], 1, maximum)
        require(
            budget[0] + size + (8 if header else 0) <= MAX_ACTIVATION_METADATA,
            "Static activation metadata budget exceeded",
        )
        budget[0] += size + (8 if header else 0)
        raw = source.read(size)
        require(
            len(raw) == size and fingerprint(os.fstat(source.fileno())) == saved,
            "Static metadata changed while reading",
        )
    require(
        fingerprint((Path(entry["root"]) / name).lstat()) == saved,
        "Static source changed while reading",
    )
    return raw


def _native_fingerprint(saved):
    return {
        "size": saved["bytes"],
        "dev": saved["device"],
        "inode": saved["inode"],
        "mtime": saved["mtime_ns"] // 10**9,
        "mtime_ns": saved["mtime_ns"] % 10**9,
        "ctime": saved["ctime_ns"] // 10**9,
        "ctime_ns": saved["ctime_ns"] % 10**9,
    }


def native_source_identity(root, shards, index, index_stat):
    """Exact serde_json sorted-map identity in accepted src/source.rs.

    Index metadata is restricted to flat JSON scalar values without floats to
    avoid Python/Rust float-serialization differences. Never uses payload hashes
    or substitutes content identity for the native header/fingerprint identity.
    """
    return hashlib.sha256(
        canonical(
            {
                "schema": 1,
                "root": root,
                "shards": shards,
                "index": index,
                "index_stat": index_stat,
            }
        )
    ).hexdigest()


class PreparedStatic:
    """Private host object, never accepted from a browser or deserialized."""

    def __init__(
        self, registry, entry, descriptor, tensors, source_identity, header_bytes
    ):
        self.registry, self.entry = registry, deepcopy(entry)
        self.descriptor, self.tensors = deepcopy(descriptor), deepcopy(tensors)
        self.source_identity = source_identity
        self.header_bytes = header_bytes
        self.model_identity = hashlib.sha256(
            canonical(
                [
                    "weight-atlas-model-v1",
                    source_identity,
                    entry["manifest"]["revision"],
                ]
            )
        ).hexdigest()
        self.descriptor_digest = identity(
            "weight-atlas-static-descriptor-v1", descriptor
        )

    def check(self):
        _current(self.registry, self.entry)


class BoundStatic:
    """Internal correspondence proof; host must still own a live ready reader."""

    def __init__(self, prepared):
        self.prepared = prepared

    def public_binding(self):
        value = self.prepared
        return {
            "schema": "weight-atlas-static-binding-v1",
            "model_id": value.entry["model_id"],
            "content_digest": value.entry["content_digest"],
            "descriptor_digest": value.descriptor_digest,
            "source_identity": value.source_identity,
            "model_identity": value.model_identity,
            "evidence": "saved_owner_hash_receipt_current_fingerprints_config_headers_and_bound_catalog",
            "fresh_payload_hashes_recomputed": False,
            "inference_ready": False,
            "fit_verified": False,
        }


class StaticPolicy:
    def __init__(self, registry):
        self.registry = registry

    def catalog(self, *, active_binding=None, reader_ready=False):
        require(type(reader_ready) is bool, "Trusted reader readiness must be boolean")
        require(
            active_binding is None or type(active_binding) is BoundStatic,
            "Only internal static binding accepted",
        )
        data = self.registry._load()  # one bounded registry metadata snapshot
        models = []
        for entry in data["models"]:
            if not entry["enabled"]:
                continue
            manifest = entry["manifest"]
            names = {item["name"] for item in manifest["files"]}
            current = self.registry._unchanged(entry) and _inventory(entry)
            candidate = bool(
                current
                and manifest["provenance"] == "owner_expected"
                and "config.json" in names
            )
            prepared = active_binding.prepared if active_binding is not None else None
            ready = bool(
                candidate
                and reader_ready
                and prepared is not None
                and prepared.registry is self.registry
                and prepared.entry == entry
            )
            models.append(
                {
                    "model_id": entry["model_id"],
                    "name": entry["name"],
                    "repository": manifest["repository"],
                    "revision": manifest["revision"],
                    "license": manifest["license"]["id"],
                    "source_bytes": sum(f["bytes"] for f in manifest["files"]),
                    "state": (
                        "registered_static_candidate"
                        if candidate
                        else "static_unavailable"
                    ),
                    "verification": "saved_owner_hash_receipt_and_current_fingerprints",
                    "hash_provenance": manifest["provenance"],
                    "static_view_candidate": candidate,
                    "static_view_ready": ready,
                    "view_ready": ready,
                    "inference_ready": False,
                    "fit_verified": False,
                }
            )
        return {
            "version": 1,
            "registry_revision": data["revision"],
            "models": models,
            "inference_enabled": False,
            "downloads_enabled": False,
        }

    def prepare(self, identifier):
        """Explicit activation only: <=8MiB metadata, no weight payload read."""
        entry = self.registry.owner_receipt(identifier)
        _current(self.registry, entry)
        manifest = entry["manifest"]
        files = {item["name"]: item for item in manifest["files"]}
        require(
            manifest["provenance"] == "owner_expected" and "config.json" in files,
            "Pinned owner dense configuration required",
        )
        budget = [0]
        raw = _read(entry, "config.json", MAX_CONFIG_BYTES, budget)
        descriptor = pinned_descriptor(raw, manifest)
        shards, tensors, shard_for = [], {}, {}
        names = sorted(name for name in files if name.endswith(".safetensors"))
        for shard_id, name in enumerate(names):
            raw = _read(entry, name, MAX_HEADER_BYTES, budget, header=True)
            header = _json(raw)
            require(
                type(header) is dict and 1 <= len(header) <= 4096,
                "Invalid bounded static tensor header",
            )
            metadata = header.pop("__metadata__", {})
            require(
                type(metadata) is dict
                and len(metadata) <= 64
                and all(
                    type(k) is str
                    and type(v) is str
                    and len(k) <= 128
                    and len(v) <= 4096
                    for k, v in metadata.items()
                ),
                "Invalid static safetensors metadata",
            )
            payload = files[name]["bytes"] - 8 - len(raw)
            spans = []
            for tensor_name, item in header.items():
                require(
                    tensor_name in descriptor["parameter_shapes"]
                    and tensor_name not in tensors,
                    "Unexpected or repeated static tensor",
                )
                fields(item, ("dtype", "shape", "data_offsets"))
                dtype, shape, offsets = (
                    item["dtype"],
                    item["shape"],
                    item["data_offsets"],
                )
                require(
                    type(dtype) is str and dtype in DENSE_BYTES,
                    "Packed/unknown static encoding refused",
                )
                require(
                    type(shape) is list
                    and shape == descriptor["parameter_shapes"][tensor_name]
                    and all(type(v) is int for v in shape),
                    "Static shape differs from pinned descriptor",
                )
                require(
                    type(offsets) is list and len(offsets) == 2,
                    "Invalid static tensor offsets",
                )
                start, end = [integer(v, 0, payload) for v in offsets]
                count = math.prod(shape)
                require(
                    end - start == count * DENSE_BYTES[dtype],
                    "Static dense byte extent differs",
                )
                rows, cols = (1, shape[0]) if len(shape) == 1 else shape
                require(
                    max(rows, cols) <= DISPLAY_AXIS_LIMIT,
                    "Static display axis unavailable",
                )
                tensors[tensor_name] = {
                    "name": tensor_name,
                    "shape": shape,
                    "rows": rows,
                    "cols": cols,
                    "count": count,
                    "dtype": dtype,
                    "element_bytes": DENSE_BYTES[dtype],
                    "shard": name,
                    "shard_id": shard_id,
                    "byte_offset": 8 + len(raw) + start,
                    "available": True,
                    "unavailable_reason": None,
                    "min_level": 0,
                    "max_level": (max(rows, cols) - 1).bit_length(),
                    "slice_required": False,
                    "display_axes": [0] if len(shape) == 1 else [0, 1],
                }
                shard_for[tensor_name] = name
                spans.append((start, end))
            cursor = 0
            for start, end in sorted(spans):
                require(
                    start == cursor and end > start, "Static payload has gaps/overlaps"
                )
                cursor = end
            require(cursor == payload, "Static header does not cover payload")
            shards.append(
                {
                    "name": name,
                    "fingerprint": _native_fingerprint(entry["fingerprints"][name]),
                    "header_sha256": hashlib.sha256(raw).hexdigest(),
                    "data_start": 8 + len(raw),
                }
            )
        require(
            tensors.keys() == descriptor["parameter_shapes"].keys(),
            "Incomplete canonical static inventory",
        )
        for tensor_id, name in enumerate(sorted(tensors)):
            tensors[name]["id"] = tensor_id
        index = None
        index_stat = None
        if "model.safetensors.index.json" in files:
            name = "model.safetensors.index.json"
            raw = _read(entry, name, MAX_HEADER_BYTES, budget)
            require(
                hashlib.sha256(raw).hexdigest() == files[name]["sha256"],
                "Static index differs from owner hash",
            )
            index = _json(raw)
            fields(index, ("weight_map",), ("metadata",))
            require(
                index["weight_map"] == shard_for,
                "Static shard index differs from actual headers",
            )
            metadata = index.get("metadata", {})
            require(
                type(metadata) is dict
                and len(metadata) <= 64
                and all(
                    type(k) is str
                    and len(k) <= 128
                    and (
                        type(v) in (bool, str)
                        or v is None
                        or type(v) is int
                        and -(2**63) <= v <= 2**64 - 1
                    )
                    for k, v in metadata.items()
                ),
                "Unsupported static index metadata",
            )
            index_stat = _native_fingerprint(entry["fingerprints"][name])
        source = native_source_identity(entry["root"], shards, index, index_stat)
        _current(self.registry, entry)
        return PreparedStatic(
            self.registry,
            entry,
            descriptor,
            tensors,
            source,
            sum(shard["data_start"] for shard in shards),
        )

    def bind(self, prepared, model, *, reader_ready=False):
        """Parent calls only after its exact owned renderer is alive and ready."""
        require(
            type(prepared) is PreparedStatic and prepared.registry is self.registry,
            "Private prepared static object required",
        )
        require(reader_ready is True, "Owned live ready renderer attestation required")
        prepared.check()
        require(
            type(model) is dict
            and len(canonical(model)) <= 2 * 1024**2
            and type(model.get("api_version")) is int
            and model["api_version"] == 1,
            "Invalid bounded Rust model metadata",
        )
        require(
            "comparison_identity" not in model and "coordinate_space" not in model,
            "Derived comparison is not a static source catalog",
        )
        digest(model.get("source_identity"))
        digest(model.get("model_identity"))
        require(
            model.get("source_directory") == prepared.entry["root"]
            and model.get("revision") == prepared.entry["manifest"]["revision"]
            and model["source_identity"] == prepared.source_identity
            and model["model_identity"] == prepared.model_identity,
            "Static renderer source/revision identity differs",
        )
        expected_bytes = sum(
            f["bytes"]
            for f in prepared.entry["manifest"]["files"]
            if f["name"].endswith(".safetensors")
        )
        require(
            model.get("source_bytes") == expected_bytes
            and type(model.get("source_bytes")) is int
            and model.get("parameter_count") == prepared.descriptor["parameter_count"]
            and type(model.get("parameter_count")) is int
            and model.get("header_bytes_read") == prepared.header_bytes
            and type(model.get("header_bytes_read")) is int,
            "Static renderer totals differ",
        )
        catalog = model.get("catalog")
        require(
            type(catalog) is list and len(catalog) == len(prepared.tensors),
            "Static renderer catalog incomplete",
        )
        observed = {}
        for position, tensor in enumerate(catalog):
            require(
                type(tensor) is dict
                and type(tensor.get("name")) is str
                and tensor["name"] not in observed,
                "Duplicate/invalid renderer tensor",
            )
            expected = prepared.tensors.get(tensor["name"])
            require(expected is not None, "Unexpected renderer tensor")
            for key, value in expected.items():
                require(
                    tensor.get(key) == value and type(tensor.get(key)) is type(value),
                    "Static renderer tensor correspondence differs",
                )
            require(
                tensor["id"] == position
                and all(
                    type(v) is int for v in tensor["shape"] + tensor["display_axes"]
                ),
                "Invalid renderer native catalog order/dimensions",
            )
            observed[tensor["name"]] = tensor
        prepared.check()
        return BoundStatic(prepared)

    def project_model(self, binding, model, *, reader_ready=False):
        """Public model metadata after host-owned correspondence/lease checks."""
        require(type(binding) is BoundStatic, "Internal static binding required")
        self.bind(binding.prepared, model, reader_ready=reader_ready)
        result = {
            name: deepcopy(model[name]) for name in PUBLIC_MODEL_KEYS if name in model
        }
        result["catalog"] = [
            {
                key: deepcopy(value)
                for key, value in tensor.items()
                if key in PUBLIC_TENSOR_KEYS
            }
            for tensor in model["catalog"]
        ]
        if (
            type(result.get("coverage")) is dict
            and result["coverage"].get("calibration_error") is not None
        ):
            result["coverage"]["calibration_error"] = "calibration_failed"
        result.update(
            name=binding.prepared.entry["name"],
            content_digest=binding.prepared.entry["content_digest"],
            static_model_descriptor=deepcopy(binding.prepared.descriptor),
            static_binding=binding.public_binding(),
            static_view_candidate=True,
            static_view_ready=True,
            inference_editable=False,
            inference_ready=False,
            inference_enabled=False,
            fit_verified=False,
            identity_validation="Saved owner hashes and current fingerprints; complete bounded config/header/index and Rust catalog correspondence. No fresh payload hash or inference qualification.",
        )
        return result
