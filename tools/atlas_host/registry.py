"""Owner-managed installed-file receipts. No network or model loading code."""

from atlas_host import limits as _limits

from collections.abc import Callable, Iterator
import fcntl
import hashlib
import os
import re
import shutil
import stat
import tempfile
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Any, TypedDict, cast

from .common import (
    canonical,
    digest,
    fields,
    identity,
    integer,
    label,
    read_json,
    require,
)
from .config import LOCAL_LIMITS
from .progress import model_id


class RegistryEntry(TypedDict):
    model_id: str
    content_digest: str
    name: str
    root: str
    manifest: dict[str, Any]
    fingerprints: dict[str, dict[str, int]]
    verified_at: str
    enabled: bool


class RegistryData(TypedDict):
    version: int
    revision: int
    models: list[RegistryEntry]


MAX_REGISTRY_BYTES = _limits.REGISTRY_MAX_BYTES
MAX_ENTRIES = _limits.REGISTRY_MAX_ENTRIES
MAX_MANIFEST_BYTES = _limits.REGISTRY_MAX_MANIFEST_BYTES
DATA_NAMES = {
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "README.md",
    "LICENSE",
    "LICENSE.txt",
    "NOTICE",
    "NOTICE.txt",
}


def validate_manifest(value: dict[str, Any]) -> dict[str, Any]:
    fields(
        value, ("version", "repository", "revision", "license", "provenance", "files")
    )
    integer(value["version"], 1, 1)
    require(
        type(value["repository"]) is str
        and re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}",
            value["repository"],
        )
        is not None,
        "Use a repository name, not a URL or path",
    )
    require(
        value["provenance"] in ("owner_expected", "synthetic_fixture"),
        "Invalid hash provenance",
    )
    revision = value["revision"]
    require(
        type(revision) is str
        and (
            re.fullmatch("[0-9a-f]{40}", revision) is not None
            or value["provenance"] == "synthetic_fixture"
            and revision == "fixture-v1"
        ),
        "Full pinned commit required",
    )
    fields(value["license"], ("id", "accepted", "file"))
    label(value["license"]["id"], 128)
    require(
        type(value["license"]["accepted"]) is bool and value["license"]["accepted"],
        "Explicit owner license acceptance required",
    )
    require(
        type(value["files"]) is list and 1 <= len(value["files"]) <= 80,
        "Invalid file count",
    )
    names: set[str] = set()
    shards = 0
    for item in value["files"]:
        fields(item, ("name", "bytes", "sha256"))
        name = item["name"]
        label(name, 180)
        tensor = (
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.safetensors", name) is not None
        )
        require(
            tensor or name in DATA_NAMES, "File type/name not in safe data allowlist"
        )
        require(name not in names, "Duplicate manifest file")
        names.add(name)
        size = integer(item["bytes"], 1)
        if not tensor:
            require(size <= 8 * 1024**2, "Auxiliary data file exceeds byte bound")
        shards += int(tensor)
        digest(item["sha256"])
    require(1 <= shards <= 64, "Require 1–64 safetensors shards")
    license_file = value["license"]["file"]
    if value["provenance"] == "synthetic_fixture":
        require(
            license_file is None or license_file in ("LICENSE", "LICENSE.txt"),
            "Invalid license file",
        )
    else:
        require(
            license_file in ("LICENSE", "LICENSE.txt"), "Pinned license file required"
        )
    require(
        license_file is None or license_file in names,
        "License file missing from manifest",
    )
    require(len(canonical(value)) <= MAX_MANIFEST_BYTES, "Manifest exceeds byte limit")
    result = deepcopy(value)
    result["files"].sort(key=lambda file: file["name"])
    return result


def content_digest(manifest: dict[str, Any]) -> str:
    manifest = validate_manifest(manifest)
    # Acceptance and local bookkeeping do not alter the underlying file content.
    return identity(
        "weight-atlas-content-v1",
        {
            "version": 1,
            "repository": manifest["repository"],
            "revision": manifest["revision"],
            "license": {
                "id": manifest["license"]["id"],
                "file": manifest["license"]["file"],
            },
            "files": manifest["files"],
        },
    )


def fingerprint(info: os.stat_result) -> dict[str, int]:
    require(stat.S_ISREG(info.st_mode), "Source must be a regular file")
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def reservation(
    free_bytes: int,
    *,
    remaining_download: int = 0,
    staging: int = 0,
    cache_growth: int = 0,
    metadata: int = 65536,
    reserve: int = LOCAL_LIMITS["disk_reserve_bytes"],
) -> dict[str, int]:
    """Pure planner. Does not acquire space, evict files, or grant a download."""
    values = [remaining_download, staging, cache_growth, metadata]
    for number in [free_bytes, reserve, *values]:
        integer(number)
    require(reserve >= LOCAL_LIMITS["disk_reserve_bytes"], "Cannot lower disk reserve")
    needed = sum(values)
    require(free_bytes >= needed + reserve, "Insufficient disk after reservation")
    return {
        "reserved_bytes": needed,
        "reserve_bytes": reserve,
        "remaining_free_bytes": free_bytes - needed,
    }


class Registry:
    def __init__(
        self,
        path: str | PathLike[str],
        *,
        free_bytes: Callable[[Path], int] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.path = Path(path).resolve()
        self.clock = clock
        self._free_bytes = free_bytes or (lambda path: shutil.disk_usage(path).free)

    def _disk_guard(self, extra: int) -> dict[str, int]:
        parent = self.path.parent
        while not parent.exists():
            parent = parent.parent
        return reservation(self._free_bytes(parent), metadata=extra)

    def _load(self) -> RegistryData:
        if not self.path.exists():
            return {"version": 1, "revision": 0, "models": []}
        require(
            self.path.is_file() and not self.path.is_symlink(),
            "Registry must be a regular file",
        )
        data: RegistryData = read_json(self.path, MAX_REGISTRY_BYTES)
        fields(data, ("version", "revision", "models"))
        integer(data["version"], 1, 1)
        integer(data["revision"])
        require(
            type(data["models"]) is list and len(data["models"]) <= MAX_ENTRIES,
            "Too many models",
        )
        identifiers: set[str] = set()
        for entry in data["models"]:
            fields(
                entry,
                (
                    "model_id",
                    "content_digest",
                    "name",
                    "root",
                    "manifest",
                    "fingerprints",
                    "verified_at",
                    "enabled",
                ),
            )
            model_id(entry["model_id"])
            digest(entry["content_digest"])
            require(
                entry["model_id"] == "m_" + entry["content_digest"],
                "Registry identity mismatch",
            )
            require(entry["model_id"] not in identifiers, "Duplicate registry identity")
            identifiers.add(entry["model_id"])
            label(entry["name"], 128)
            label(entry["root"], 4096)
            require(Path(entry["root"]).is_absolute(), "Registry root must be absolute")
            label(entry["verified_at"], 64)
            require(type(entry["enabled"]) is bool, "Invalid enabled flag")
            manifest = validate_manifest(entry["manifest"])
            require(
                content_digest(manifest) == entry["content_digest"],
                "Receipt identity mismatch",
            )
            fields(entry["fingerprints"], [file["name"] for file in manifest["files"]])
            for record in entry["fingerprints"].values():
                fields(record, ("device", "inode", "bytes", "mtime_ns", "ctime_ns"))
                for value in record.values():
                    integer(value)
        return data

    @contextmanager
    def _writer(self) -> Iterator[None]:
        self._disk_guard(MAX_REGISTRY_BYTES + 65536)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock_path = self.path.with_name(self.path.name + ".lock")
        fd = os.open(
            lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600
        )
        try:
            require(
                stat.S_ISREG(os.fstat(fd).st_mode),
                "Registry lock must be a regular file",
            )
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("Registry writer busy") from error
            yield
        finally:
            os.close(fd)

    def _save(self, data: RegistryData) -> None:
        raw = canonical(data)
        require(len(raw) <= MAX_REGISTRY_BYTES, "Registry exceeds byte bound")
        self._disk_guard(len(raw) + 65536)
        fd, temporary = tempfile.mkstemp(
            prefix="." + self.path.name + ".", dir=self.path.parent
        )
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(raw)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def register(
        self,
        root: str | PathLike[str],
        manifest: dict[str, Any],
        name: str,
        *,
        max_bytes: int = 1024**2,
        timeout_ms: int = 1000,
        publish_disabled: bool = False,
    ) -> dict[str, Any]:
        """Verify already installed files. No renderer/inference readiness claim."""
        manifest = validate_manifest(manifest)
        label(name, 128)
        integer(max_bytes, 1)
        integer(timeout_ms, 1, 5000)
        require(type(publish_disabled) is bool, "Disabled publication must be boolean")
        total = sum(file["bytes"] for file in manifest["files"])
        require(total <= max_bytes, "Verification exceeds explicit byte allowance")
        root = Path(root).resolve(strict=True)
        require(root.is_dir(), "Installed source must be a directory")
        require(
            not self.path.is_relative_to(root), "Registry must be outside model source"
        )
        end = self.clock() + timeout_ms / 1000

        def checkpoint() -> None:
            require(self.clock() < end, "Verification time allowance exhausted")

        with self._writer():
            data = self._load()
            content = content_digest(manifest)
            identifier = "m_" + content
            old = next(
                (entry for entry in data["models"] if entry["model_id"] == identifier),
                None,
            )
            require(
                old is not None or len(data["models"]) < MAX_ENTRIES,
                "Registry model limit reached",
            )
            fingerprints: dict[str, dict[str, int]] = {}
            for item in manifest["files"]:
                checkpoint()
                path = root / item["name"]
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(fd, "rb") as source:
                    before = fingerprint(os.fstat(source.fileno()))
                    require(
                        before["bytes"] == item["bytes"], "Pinned file size mismatch"
                    )
                    measured = hashlib.sha256()
                    remaining = item["bytes"]
                    while remaining:
                        checkpoint()
                        block = source.read(min(65536, remaining))
                        checkpoint()
                        require(bool(block), "Pinned file ended early")
                        measured.update(block)
                        remaining -= len(block)
                    require(not source.read(1), "Pinned file grew during verification")
                    require(
                        before
                        == fingerprint(os.fstat(source.fileno()))
                        == fingerprint(path.lstat()),
                        "Source changed during verification",
                    )
                    require(
                        measured.hexdigest() == item["sha256"],
                        "Pinned file hash mismatch",
                    )
                    fingerprints[item["name"]] = before
            checkpoint()
            entry: RegistryEntry = {
                "model_id": identifier,
                "content_digest": content,
                "name": name,
                "root": str(root),
                "manifest": manifest,
                "fingerprints": fingerprints,
                "verified_at": datetime.now(timezone.utc).isoformat(),
                # Acquisition must atomically publish a disabled receipt,
                # including replacement of a previously enabled identity.
                # Standalone registration preserves its existing behavior.
                "enabled": (
                    old["enabled"]
                    if old is not None and not publish_disabled
                    else False
                ),
            }
            require(self._unchanged(entry), "Source changed before receipt publication")
            checkpoint()
            data["models"] = [
                entry if item["model_id"] == identifier else item
                for item in data["models"]
            ]
            if old is None:
                data["models"].append(entry)
            data["revision"] += 1
            self._save(data)
            return {
                "model_id": identifier,
                "content_digest": content,
                "verified_bytes": total,
                "verification": "matched_owner_expectations",
                "renderer_ready": False,
                "inference_ready": False,
                "enabled": entry["enabled"],
            }

    @staticmethod
    def _unchanged(entry: RegistryEntry) -> bool:
        try:
            return all(
                fingerprint((Path(entry["root"]) / name).lstat()) == expected
                for name, expected in entry["fingerprints"].items()
            )
        except (OSError, ValueError):
            return False

    def set_enabled(self, identifier: str, enabled: bool) -> dict[str, Any]:
        model_id(identifier)
        require(type(enabled) is bool, "Enabled must be boolean")
        with self._writer():
            data = self._load()
            entry = next(
                (entry for entry in data["models"] if entry["model_id"] == identifier),
                None,
            )
            require(entry is not None, "Unknown installed model")
            entry = cast(RegistryEntry, entry)
            require(
                not enabled or self._unchanged(entry),
                "Source changed; reverify before enabling",
            )
            entry["enabled"] = enabled
            data["revision"] += 1
            self._save(data)
        return {
            "model_id": identifier,
            "enabled": enabled,
            "registry_revision": data["revision"],
        }

    def catalog(self) -> dict[str, Any]:
        data = self._load()
        models: list[dict[str, Any]] = []
        for entry in data["models"]:
            if not entry["enabled"]:
                continue
            manifest = entry["manifest"]
            unchanged = self._unchanged(entry)
            models.append(
                {
                    "model_id": entry["model_id"],
                    "name": entry["name"],
                    "repository": manifest["repository"],
                    "revision": manifest["revision"],
                    "license": manifest["license"]["id"],
                    "source_bytes": sum(item["bytes"] for item in manifest["files"]),
                    "state": (
                        "verified_pending_renderer" if unchanged else "source_changed"
                    ),
                    "verification": "saved_hash_receipt_and_current_fingerprint",
                    "hash_provenance": manifest["provenance"],
                    "view_ready": False,
                    "inference_ready": False,
                }
            )
        return {"version": 1, "registry_revision": data["revision"], "models": models}

    def owner_receipt(self, identifier: str) -> RegistryEntry:
        """Local owner CLI only; deliberately not a visitor HTTP projection."""
        model_id(identifier)
        entry = next(
            (
                entry
                for entry in self._load()["models"]
                if entry["model_id"] == identifier
            ),
            None,
        )
        require(entry is not None, "Unknown installed model")
        return deepcopy(cast(RegistryEntry, entry))
