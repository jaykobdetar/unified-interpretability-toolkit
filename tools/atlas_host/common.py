"""Small strict JSON and type helpers shared by local host contracts."""

from collections.abc import Iterable, Mapping
import hashlib
from os import PathLike
import json
import re
from pathlib import Path
from typing import Any


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def fields(
    value: Mapping[str, Any], required: Iterable[str], optional: Iterable[str] = ()
) -> None:
    require(type(value) is dict, "Expected an object")
    require(
        set(required) <= value.keys() <= set(required) | set(optional),
        "Missing or unknown fields",
    )


def integer(value: int, low: int | float = 0, high: int | float = 2**63 - 1) -> int:
    require(
        type(value) is int and low <= value <= high, "Integer outside allowed range"
    )
    return value


def label(value: str, maximum: int | float = 128) -> str:
    require(
        type(value) is str
        and 0 < len(value) <= maximum
        and not any(ord(c) < 32 or ord(c) == 127 for c in value),
        "Invalid label",
    )
    return value


def digest(value: str) -> str:
    require(
        type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None,
        "Expected lowercase SHA-256 digest",
    )
    return value


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def identity(domain: object, value: object) -> str:
    return hashlib.sha256(canonical([domain, value])).hexdigest()


def _unique(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def read_json(path: str | PathLike[str], limit: int) -> Any:
    with Path(path).open("rb") as source:
        raw = source.read(limit + 1)
    require(len(raw) <= limit, "JSON file exceeds byte limit")
    try:
        return json.loads(
            raw,
            object_pairs_hook=_unique,
            parse_constant=lambda _: (_ for _ in ()).throw(
                ValueError("Nonfinite JSON")
            ),
        )
    except (UnicodeError, RecursionError) as error:
        raise ValueError("Invalid bounded JSON") from error
