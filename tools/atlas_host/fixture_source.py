"""Exact installed tiny-fixture receipts; no launcher or provider dependency."""

import hashlib
from pathlib import Path
from typing import cast

from .common import require
from .registry import RegistryEntry, fingerprint

FIXTURE_SHA = "c0075bfc55f9e51ccac3c5511ea55a5ca19744b002921e8d2e4ae3f60d321be3"
FIXTURE_FILES = [{"name": "tiny.safetensors", "bytes": 244, "sha256": FIXTURE_SHA}]


def fixture_entry(entry: RegistryEntry) -> bool:
    return cast(
        bool,
        (
            entry["manifest"]["provenance"] == "synthetic_fixture"
            and entry["manifest"]["files"] == FIXTURE_FILES
        ),
    )


def check_fixture(entry: RegistryEntry, *, hash_bytes: bool = False) -> None:
    require(
        entry["enabled"] and fixture_entry(entry),
        "Only enabled exact synthetic fixtures may activate",
    )
    root = Path(entry["root"])
    # Match all files the source loader could interpret, not only the listed shard.
    interpreted = {
        path.name
        for path in root.iterdir()
        if path.name.endswith(".safetensors")
        or path.name == "model.safetensors.index.json"
    }
    require(interpreted == {"tiny.safetensors"}, "Fixture source inventory changed")
    path = root / "tiny.safetensors"
    before = fingerprint(path.lstat())
    require(
        before == entry["fingerprints"]["tiny.safetensors"],
        "Fixture fingerprint changed",
    )
    if hash_bytes:
        # Exactly 244 known fixture bytes, not an arbitrary model hash request.
        import os

        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as source:
            require(
                fingerprint(os.fstat(source.fileno())) == before,
                "Fixture source changed",
            )
            raw = source.read(245)
            require(
                len(raw) == 244 and hashlib.sha256(raw).hexdigest() == FIXTURE_SHA,
                "Fixture hash differs from the allowed synthetic source",
            )
        require(
            fingerprint(path.lstat()) == before, "Fixture changed during activation"
        )


# Retain the existing callable import addresses for saved Python references.
fixture_entry.__module__ = __package__ + ".runtime_adapter"
check_fixture.__module__ = __package__ + ".runtime_adapter"
