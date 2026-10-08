"""Strict clients of the real receipt helpers; heterogeneous JSON stays explicit."""

from os import PathLike
from typing import Any

from atlas_host import common, registry


def accepted_values() -> tuple[int, str, str]:
    number: int = common.integer(3, 0, 4)
    label: str = common.label("fixture", 128)
    digest: str = common.digest("a" * 64)
    common.require(True, "fixture condition")
    return number, label, digest


def receipt(
    value: dict[str, Any], path: str | PathLike[str]
) -> tuple[dict[str, Any], str, bytes]:
    common.fields(value, ("version",), ("files",))
    pairs: dict[str, Any] = common._unique([("version", 1)])
    parsed: Any = common.read_json(path, 32768)
    digest: str = common.identity("fixture-domain", pairs)
    encoded: bytes = common.canonical(parsed)
    validated: dict[str, Any] = registry.validate_manifest(value)
    content: str = registry.content_digest(validated)
    return validated, content + digest, encoded
